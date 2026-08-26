#!/usr/bin/env bash
# =============================================================================
# Build libtrtjl.so — the C ABI shim that lets Julia ccall into TensorRT.
#
# What it does:
#   1. locates the TensorRT runtime library (the pip wheel inside the sibling
#      NeuralBloch repo's venv — the same install that built model.engine);
#   2. reads the exact TensorRT version out of that library;
#   3. fetches the matching TensorRT headers from the NVIDIA/TensorRT GitHub
#      repository (Apache-2.0; no headers ship with the pip wheel) into
#      third_party/include/ — cached, so this happens once;
#   4. compiles trt_shim.cpp with g++ (no cmake needed), baking rpaths so the
#      resulting .so finds libnvinfer and libcudart without LD_LIBRARY_PATH;
#   5. sanity-checks the exported symbols.
#
# Overrides (all optional):
#   NEURALBLOCH_DIR   sibling NeuralBloch checkout (default: ../NeuralBloch
#                     next to this repository — same convention as the rest
#                     of the project)
#   TENSORRT_LIB_DIR  directory containing libnvinfer.so.* (skips the venv
#                     search)
#   TRT_INCLUDE_DIR   directory containing NvInfer.h (skips the GitHub fetch;
#                     use on machines without network access)
#   CUDA_HOME         CUDA toolkit root (default: /usr/local/cuda)
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$HERE/libtrtjl.so"

say()  { printf '%s\n' "$*"; }
fail() { printf 'build_shim.sh: %s\n' "$*" >&2; exit 1; }

# --- 1. TensorRT runtime library --------------------------------------------
if [[ -n "${TENSORRT_LIB_DIR:-}" ]]; then
    TRTLIB="$TENSORRT_LIB_DIR"
else
    NB="${NEURALBLOCH_DIR:-$ROOT/../NeuralBloch}"
    TRTLIB=""
    for d in "$NB"/.venv/lib/python*/site-packages/tensorrt_libs; do
        [[ -d "$d" ]] && TRTLIB="$d" && break
    done
    [[ -n "$TRTLIB" ]] || fail "no tensorrt_libs found under $NB/.venv — is the \
NeuralBloch venv set up (uv sync in $NB)? Or set TENSORRT_LIB_DIR."
fi

NVINFER=""
for f in "$TRTLIB"/libnvinfer.so.*; do
    [[ -f "$f" && "$f" != *builder* && "$f" != *plugin* ]] && NVINFER="$f" && break
done
[[ -n "$NVINFER" ]] || fail "no libnvinfer.so.* in $TRTLIB"
SONAME="$(basename "$NVINFER")"          # e.g. libnvinfer.so.11
say "TensorRT library : $NVINFER"

# --- 2. exact version, from the library's own version symbol -----------------
VERSYM="$(nm -D "$NVINFER" | grep -o 'tensorrt_version_[0-9_]*' | head -1 || true)"
[[ -n "$VERSYM" ]] || fail "cannot find a tensorrt_version_* symbol in $NVINFER"
IFS='_' read -r _ _ MAJ MIN PAT BLD <<< "$VERSYM"
say "TensorRT version : $MAJ.$MIN.$PAT.$BLD"

# --- 3. headers --------------------------------------------------------------
if [[ -n "${TRT_INCLUDE_DIR:-}" ]]; then
    INC="$TRT_INCLUDE_DIR"
    [[ -f "$INC/NvInfer.h" ]] || fail "TRT_INCLUDE_DIR=$INC has no NvInfer.h"
else
    INC="$HERE/third_party/include"
    if [[ ! -f "$INC/NvInfer.h" ]]; then
        # The pip wheel ships no headers; the NVIDIA/TensorRT OSS repository
        # publishes them (Apache-2.0), tagged per major.minor release.
        TAG="v$MAJ.$MIN"
        TMP="$(mktemp -d)"
        trap 'rm -rf "$TMP"' EXIT
        say "fetching TensorRT $TAG headers from github.com/NVIDIA/TensorRT ..."
        if ! git clone --quiet --depth 1 --branch "$TAG" --filter=blob:none \
                 --sparse https://github.com/NVIDIA/TensorRT "$TMP/TensorRT" 2>/dev/null; then
            TAG="release/$MAJ.$MIN"
            git clone --quiet --depth 1 --branch "$TAG" --filter=blob:none \
                --sparse https://github.com/NVIDIA/TensorRT "$TMP/TensorRT" \
                || fail "cannot clone NVIDIA/TensorRT at v$MAJ.$MIN or $TAG — \
no network? Point TRT_INCLUDE_DIR at a directory with the TensorRT headers."
        fi
        git -C "$TMP/TensorRT" sparse-checkout set include --quiet
        mkdir -p "$INC"
        cp -r "$TMP/TensorRT/include/." "$INC/"
        say "headers cached in $INC"
    fi
fi

# Recent headers indirect NV_TENSORRT_* through TRT_*_ENTERPRISE macros;
# older ones define the numbers directly. Accept either spelling.
hdr_ver() {
    grep -oP "#define (TRT_${1}_ENTERPRISE|NV_TENSORRT_${1}) \K[0-9]+" \
        "$INC/NvInferVersion.h" | head -1
}
HDR_MAJ="$(hdr_ver MAJOR)"
HDR_MIN="$(hdr_ver MINOR)"
HDR_PAT="$(hdr_ver PATCH)"
[[ -n "$HDR_MAJ" && -n "$HDR_MIN" && -n "$HDR_PAT" ]] || fail "cannot parse \
the version macros in $INC/NvInferVersion.h"
[[ "$HDR_MAJ.$HDR_MIN" == "$MAJ.$MIN" ]] || fail "header version $HDR_MAJ.$HDR_MIN \
does not match library $MAJ.$MIN — delete $HERE/third_party and rerun"
[[ "$HDR_PAT" == "$PAT" ]] || say "note: header patch $HDR_PAT vs library patch \
$PAT (same $MAJ.$MIN API — fine for the runtime interface)"
say "TensorRT headers : $INC ($HDR_MAJ.$HDR_MIN.$HDR_PAT)"

# --- 4. compile --------------------------------------------------------------
CUDA="${CUDA_HOME:-/usr/local/cuda}"
[[ -f "$CUDA/include/cuda_runtime_api.h" ]] || fail "no CUDA toolkit headers \
at $CUDA/include — set CUDA_HOME"

say "compiling $OUT ..."
g++ -O3 -std=c++17 -fPIC -shared "$HERE/trt_shim.cpp" -o "$OUT" \
    -isystem "$INC" -isystem "$CUDA/include" \
    -L"$TRTLIB" -l:"$SONAME" \
    -L"$CUDA/lib64" -lcudart \
    -Wl,-rpath,"$TRTLIB" -Wl,-rpath,"$CUDA/lib64" \
    -Wl,--no-undefined
# Deliberately NOT -static-libstdc++: Julia dlopens with RTLD_DEEPBIND, and a
# second (static) C++ runtime inside the shim then shadows the already-loaded
# libstdc++ with its own uninitialized locale state — libnvinfer's static
# initializers segfault on it. Dynamic linking reuses the one libstdc++ the
# process already has (Julia 1.12 bundles GLIBCXX 3.4.33; g++ 13 and
# libnvinfer need no more than that).

# --- 5. sanity checks --------------------------------------------------------
NSYM="$(nm -D --defined-only "$OUT" | grep -c ' T trtjl_')"
[[ "$NSYM" -eq 20 ]] || fail "expected 20 exported trtjl_* symbols, found $NSYM \
(C++ name mangling leaking through?)"
ldd "$OUT" | grep -q 'not found' && fail "unresolved dependencies: $(ldd "$OUT" | grep 'not found')"

say "ok: $OUT ($NSYM trtjl_* symbols, linked against $SONAME)"
