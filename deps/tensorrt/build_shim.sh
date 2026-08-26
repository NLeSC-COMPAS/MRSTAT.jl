#!/usr/bin/env bash
# =============================================================================
# Build libtrtjl.so — the C ABI shim that lets Julia ccall into TensorRT.
#
# PREREQUISITE — a system TensorRT installation (NVIDIA's Debian/RPM
# packages), with a version matching models/model.engine: engines are locked
# to the exact TensorRT version that built them. This script checks and
# hints; it never installs or downloads the runtime:
#   https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html
#
# What it does:
#   1. determines the required TensorRT version from the engine's own header
#      (models/model.engine; TENSORRT_VERSION overrides);
#   2. locates the system libnvinfer via ldconfig (TENSORRT_LIB_DIR overrides,
#      e.g. for a rootless NVIDIA tar install) and verifies its version
#      against the engine;
#   3. finds the TensorRT headers — the system ones (libnvinfer-headers-dev)
#      when present and matching, else a version-pinned fetch from the
#      NVIDIA/TensorRT GitHub repository cached in third_party/
#      (TRT_INCLUDE_DIR overrides);
#   4. compiles trt_shim.cpp with g++ alone (no cmake) and sanity-checks the
#      exported symbols.
#
# Overrides (all optional):
#   TENSORRT_VERSION   required version, e.g. 11.2.1.2 (default: engine header)
#   TENSORRT_LIB_DIR   directory containing libnvinfer.so.<N> (default: ldconfig)
#   TRT_INCLUDE_DIR    directory containing NvInfer.h (default: system, else fetch)
#   MRSTAT_MODELS_DIR  artifacts directory holding model.engine (default: ../../models)
#   CUDA_HOME          CUDA toolkit root (default: /usr/local/cuda)
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$HERE/libtrtjl.so"

say()  { printf '%s\n' "$*"; }
fail() { printf 'build_shim.sh: %s\n' "$*" >&2; exit 1; }

# --- 1. required TensorRT version (the engine is the contract) ---------------
MODELS="${MRSTAT_MODELS_DIR:-$ROOT/models}"
ENGINE="$MODELS/model.engine"
REQ_VER="${TENSORRT_VERSION:-}"
if [ -z "$REQ_VER" ] && [ -f "$ENGINE" ]; then
    if [ "$(head -c 4 "$ENGINE")" = "ftrt" ]; then
        REQ_VER="$(od -A n -t u1 -j 24 -N 4 "$ENGINE" | awk '{print $1"."$2"."$3"."$4}')"
        say "required TensorRT : $REQ_VER (read from $ENGINE)"
    else
        say "note: $ENGINE has an unexpected header — cannot derive the required version"
    fi
fi
[ -n "$REQ_VER" ] || say "note: no engine and no TENSORRT_VERSION — building against \
whatever TensorRT is installed"

hint() {
    local maj="${REQ_VER%%.*}"
    [ -n "$maj" ] || maj="<major>"
    say "TensorRT is a system prerequisite of the TensorRT forward model, installed"
    say "through NVIDIA's packages — never by this repository:"
    say "  Debian/Ubuntu: sudo apt-get install libnvinfer$maj libnvinfer-headers-dev libnvinfer-dev"
    say "  RPM:           sudo dnf install libnvinfer$maj libnvinfer-headers-devel libnvinfer-devel"
    if [ -n "$REQ_VER" ]; then
        say "models/model.engine requires exactly TensorRT $REQ_VER — list what the NVIDIA"
        say "repo offers with 'apt-cache madison libnvinfer$maj' and pin that version."
    fi
    say "Docs: https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html"
}

# --- 2. the system TensorRT runtime ------------------------------------------
NVINFER=""
if [ -n "${TENSORRT_LIB_DIR:-}" ]; then
    for f in "$TENSORRT_LIB_DIR"/libnvinfer.so.*; do
        [[ -f "$f" && "$f" =~ libnvinfer\.so\.[0-9]+$ ]] && NVINFER="$f" && break
    done
    [ -n "$NVINFER" ] || { hint; fail "no libnvinfer.so.<N> in TENSORRT_LIB_DIR=$TENSORRT_LIB_DIR"; }
else
    NVINFER="$(ldconfig -p 2>/dev/null | sed -n 's/.*=> \(.*libnvinfer\.so\.[0-9]*\)$/\1/p' | head -1)"
    [ -n "$NVINFER" ] || { hint; fail "no system libnvinfer found (ldconfig) — install \
TensorRT first, or point TENSORRT_LIB_DIR at an existing install"; }
fi
SONAME="$(basename "$NVINFER")"
TRTLIB="$(cd "$(dirname "$NVINFER")" && pwd)"
say "TensorRT library  : $NVINFER"

VERSYM="$(nm -D "$NVINFER" | grep -o 'tensorrt_version_[0-9_]*' | head -1 || true)"
[ -n "$VERSYM" ] || fail "cannot find a tensorrt_version_* symbol in $NVINFER"
LIB_VER="$(printf '%s' "$VERSYM" | sed 's/tensorrt_version_//; s/_/./g')"
say "TensorRT version  : $LIB_VER"
if [ -n "$REQ_VER" ] && [ "$REQ_VER" != "$LIB_VER" ]; then
    hint
    fail "the engine requires TensorRT $REQ_VER but the installed library is $LIB_VER — \
engines are version-locked: install TensorRT $REQ_VER, or rebuild the engine in \
NeuralBloch against $LIB_VER, then re-run this script"
fi
IFS='.' read -r MAJ MIN _PAT _BLD <<< "$LIB_VER"

# --- 3. headers ---------------------------------------------------------------
# Recent headers indirect NV_TENSORRT_* through TRT_*_ENTERPRISE macros;
# older ones define the numbers directly. Accept either spelling.
hdr_ver() { # <include dir> <MAJOR|MINOR|PATCH>
    grep -oP "#define (TRT_${2}_ENTERPRISE|NV_TENSORRT_${2}) \K[0-9]+" \
        "$1/NvInferVersion.h" 2>/dev/null | head -1
}
INC=""
if [ -n "${TRT_INCLUDE_DIR:-}" ]; then
    [ -f "$TRT_INCLUDE_DIR/NvInfer.h" ] || fail "TRT_INCLUDE_DIR=$TRT_INCLUDE_DIR has no NvInfer.h"
    INC="$TRT_INCLUDE_DIR"
else
    # System headers first (libnvinfer-headers-dev), a tar install's include/
    # next, then the cached fetch — first one matching the library's
    # major.minor wins.
    for c in /usr/include/x86_64-linux-gnu /usr/include "$TRTLIB/../include" \
             "$HERE/third_party/include"; do
        [ -f "$c/NvInfer.h" ] || continue
        if [ "$(hdr_ver "$c" MAJOR).$(hdr_ver "$c" MINOR)" = "$MAJ.$MIN" ]; then
            INC="$c"
            break
        fi
    done
fi
if [ -z "$INC" ]; then
    # No matching headers installed: fetch the runtime header chain from the
    # NVIDIA/TensorRT OSS repository (Apache-2.0), tagged per major.minor,
    # into a local cache. This covers lean installs that carry only the
    # runtime package.
    INC="$HERE/third_party/include"
    TAG="v$MAJ.$MIN"
    TMP="$(mktemp -d)"
    trap 'rm -rf "$TMP"' EXIT
    say "no matching system headers — fetching TensorRT $TAG headers from github.com/NVIDIA/TensorRT ..."
    if ! git clone --quiet --depth 1 --branch "$TAG" --filter=blob:none \
             --sparse https://github.com/NVIDIA/TensorRT "$TMP/TensorRT" 2>/dev/null; then
        TAG="release/$MAJ.$MIN"
        git clone --quiet --depth 1 --branch "$TAG" --filter=blob:none \
            --sparse https://github.com/NVIDIA/TensorRT "$TMP/TensorRT" \
            || fail "cannot clone NVIDIA/TensorRT at v$MAJ.$MIN or $TAG — no network? \
Install libnvinfer-headers-dev, or point TRT_INCLUDE_DIR at the TensorRT headers."
    fi
    git -C "$TMP/TensorRT" sparse-checkout set include --quiet
    rm -rf "$INC"
    mkdir -p "$INC"
    cp -r "$TMP/TensorRT/include/." "$INC/"
    [ "$(hdr_ver "$INC" MAJOR).$(hdr_ver "$INC" MINOR)" = "$MAJ.$MIN" ] \
        || fail "fetched headers report $(hdr_ver "$INC" MAJOR).$(hdr_ver "$INC" MINOR), \
expected $MAJ.$MIN — delete $HERE/third_party and rerun"
    say "headers cached in $INC"
fi
HDR_PAT="$(hdr_ver "$INC" PATCH)"
[ "$HDR_PAT" = "$_PAT" ] || say "note: header patch $HDR_PAT vs library patch $_PAT \
(same $MAJ.$MIN API — fine for the runtime interface)"
say "TensorRT headers  : $INC ($(hdr_ver "$INC" MAJOR).$(hdr_ver "$INC" MINOR).$HDR_PAT)"

# --- 4. compile --------------------------------------------------------------
CUDA="${CUDA_HOME:-/usr/local/cuda}"
[ -f "$CUDA/include/cuda_runtime_api.h" ] || fail "no CUDA toolkit headers \
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
# process already has.

# --- 5. sanity checks --------------------------------------------------------
NSYM="$(nm -D --defined-only "$OUT" | grep -c ' T trtjl_')"
[ "$NSYM" -eq 20 ] || fail "expected 20 exported trtjl_* symbols, found $NSYM \
(C++ name mangling leaking through?)"
ldd "$OUT" | grep -q 'not found' && fail "unresolved dependencies: $(ldd "$OUT" | grep 'not found')"

say "ok: $OUT ($NSYM trtjl_* symbols, linked against $NVINFER)"
