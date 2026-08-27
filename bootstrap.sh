#!/usr/bin/env bash
# One-command setup for MRSTAT.jl. Idempotent — safe to re-run.
#
#   ./bootstrap.sh          set up, then report what is still missing
#   source env.sh           before every Julia session afterwards
#
# What it does NOT do: write anything outside this repository. The two
# sibling checkouts and the trained checkpoint are other people's
# directories, so they are detected and reported, never created or modified.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

ok=0; bad=0
say()  { printf '  %s\n' "$*"; }
pass() { printf '  \033[32m✓\033[0m %s\n' "$*"; ok=$((ok+1)); }
fail() { printf '  \033[31m✗\033[0m %s\n' "$*"; bad=$((bad+1)); }
hdr()  { printf '\n\033[1m%s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
hdr "1. GPU"
# ---------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null; then
    pass "$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
else
    fail "nvidia-smi not found. This package is CUDA-only: the tests hard-fail"
    say  "  without a GPU and every forward model runs on the device."
fi

# ---------------------------------------------------------------------------
hdr "2. Toolchain (user-space, no sudo)"
# ---------------------------------------------------------------------------
command -v juliaup >/dev/null || curl -fsSL https://install.julialang.org | sh -s -- --yes
command -v uv      >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.juliaup/bin:$HOME/.local/bin:$PATH"
pass "julia $(julia --version | awk '{print $3}')"
pass "uv $(uv --version | awk '{print $2}')"

# ---------------------------------------------------------------------------
hdr "3. Python (torch for the surrogate, matplotlib for the scripts)"
# ---------------------------------------------------------------------------
# Reuse a complete interpreter if one is already around — a torch wheel is
# several GB and the sibling bridge usually ships one. Build our own only
# when nothing reusable exists, so this repo never depends on another's venv.
complete() { [ -x "$1" ] && "$1" -c 'import torch, matplotlib' >/dev/null 2>&1; }
PY=""
for c in "${MRSTAT_PYTHON:-}" "$ROOT/.venv/bin/python" "$ROOT/../NeuralBloch.jl/.venv/bin/python"; do
    [ -n "$c" ] || continue
    if complete "$c"; then PY="$(cd "$(dirname "$c")" && pwd)/$(basename "$c")"; break; fi
done

if [ -n "$PY" ]; then
    pass "reusing $PY"
else
    say "no interpreter with both torch and matplotlib — building ./.venv"
    [ -d .venv ] || uv venv --python 3.12 .venv
    # The default PyPI torch wheel is a CUDA 13 build needing driver >= 580.
    # On drivers 525-579 install from the cu126 index instead:
    #   uv pip install --python .venv/bin/python torch \
    #       --index-url https://download.pytorch.org/whl/cu126
    uv pip install --quiet --python .venv/bin/python torch numpy matplotlib
    PY="$ROOT/.venv/bin/python"
    pass "created $PY"
fi
"$PY" - <<'PYEOF' || true
import torch
print(f"    torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
PYEOF

# ---------------------------------------------------------------------------
hdr "4. NeuralBloch.jl (surrogate only)"
# ---------------------------------------------------------------------------
# The bridge package is a *weak* dependency: the project instantiates and the
# Bloch and TensorRT models run without it. When the checkout is present,
# env.sh puts it on the Julia load path and `using NeuralBloch` activates the
# surrogate's package extension; the test suite resolves it from here too
# (test/Project.toml [sources]).
if [ -d ../NeuralBloch.jl ]; then
    pass "../NeuralBloch.jl"
else
    say "· ../NeuralBloch.jl missing (surrogate only — Bloch and TensorRT run without it):"
    say "    git clone https://github.com/NLeSC-COMPAS/NeuralBloch.jl.git beside this repo"
fi

# ---------------------------------------------------------------------------
hdr "5. NeuralBloch python repository (the surrogate's model code; artifact source)"
# ---------------------------------------------------------------------------
NB="${NEURALBLOCH_DIR:-$ROOT/../NeuralBloch}"
[ -d "$NB" ] && NB="$(cd "$NB" && pwd)"
if [ -d "$NB" ]; then
    pass "$NB"
    [ -f "$NB/src/model.py" ] && pass "  src/model.py" || say "· src/model.py missing (the surrogate cannot run without it)"
else
    say "· not found at $NB (surrogate only — its python model code lives there,"
    say "  and it is where the models/ artifacts below are copied from):"
    say "    git clone git@github.com:NLeSC-COMPAS/NeuralBloch.git beside this repo,"
    say "    or set NEURALBLOCH_DIR to its path."
fi

# ---------------------------------------------------------------------------
hdr "6. Model artifacts (models/)"
# ---------------------------------------------------------------------------
# The runtime artifacts live in-repo under models/ (gitignored): the loaders
# look only there, never in the sibling checkout. Missing files are copied in
# from the checkout when it is present — never overwritten — and a copy that
# differs from the checkout gets a drift warning naming the cp to refresh it.
MODELS="${MRSTAT_MODELS_DIR:-$ROOT/models}"
mkdir -p "$MODELS/rf_data"
artifact() { # <models-relative path> <source path> <required: yes|no> <hint>
    local dst="$MODELS/$1" src="$2"
    if [ -f "$dst" ]; then
        if [ -f "$src" ] && ! cmp -s "$dst" "$src"; then
            pass "models/$1 ($(du -h "$dst" | cut -f1)) — DIFFERS from the checkout copy"
            say  "    (rebuilt upstream? refresh with: cp \"$src\" \"$dst\")"
        else
            pass "models/$1 ($(du -h "$dst" | cut -f1))"
        fi
    elif [ -f "$src" ]; then
        cp "$src" "$dst"
        pass "models/$1 — copied in from $src"
    elif [ "$3" = yes ]; then
        fail "models/$1 missing — $4"
    else
        say  "· models/$1 missing (optional) — $4"
    fi
}
artifact model.pt     "$NB/outputs/model.pt"     no  "only the surrogate needs it; the checkpoint is gitignored upstream — train it there or copy one in (without it load_surrogate() and the test suite cannot run)"
artifact model.engine "$NB/outputs/model.engine" no  "only the TensorRT model needs it; build it in NeuralBloch (uv run run.py export_onnx && uv run run.py build_engine), then re-run ./bootstrap.sh or copy it in"
artifact rf_data/rf_angle_list.txt "$NB/data/rf_data/rf_angle_list.txt" no "the validation scripts' ground-truth arm rebuilds the training sequence from it"
artifact rf_data/rf_phase_list.txt "$NB/data/rf_data/rf_phase_list.txt" no "the validation scripts' ground-truth arm rebuilds the training sequence from it"

# ---------------------------------------------------------------------------
hdr "7. Julia environment"
# ---------------------------------------------------------------------------
export JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE="$PY"
julia --project=. -e '
    using Pkg
    # Every dependency comes from the General registry — NeuralBloch.jl is a
    # weak dependency reached through the load path (env.sh), so there is
    # nothing to develop and no sibling is needed to resolve.
    Pkg.resolve(); Pkg.instantiate(); Pkg.precompile()' >/tmp/mrstat-bootstrap.$$ 2>&1 \
    && pass "project resolved, instantiated and precompiled" \
    || { fail "Julia setup failed:"; sed 's/^/    /' /tmp/mrstat-bootstrap.$$; }
rm -f /tmp/mrstat-bootstrap.$$

# ---------------------------------------------------------------------------
hdr "8. TensorRT forward model (optional)"
# ---------------------------------------------------------------------------
# The TensorRT *runtime* is a system prerequisite — NVIDIA's Debian/RPM
# packages, matching the engine's version; this script detects and hints,
# never installs. The Bloch and surrogate routes are complete without it.
#   https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html
ENGINE="$MODELS/model.engine"
ENG_VER=""
[ -f "$ENGINE" ] && \
    ENG_VER="$(od -A n -t u1 -j 24 -N 4 "$ENGINE" | awk '{print $1"."$2"."$3"."$4}')"
TRT_LIB="$(ldconfig -p 2>/dev/null | sed -n 's/.*=> \(.*libnvinfer\.so\.[0-9]*\)$/\1/p' | head -1)"
if [ -n "$TRT_LIB" ]; then
    LIB_VER="$(nm -D "$TRT_LIB" 2>/dev/null | grep -o 'tensorrt_version_[0-9_]*' | head -1 \
               | sed 's/tensorrt_version_//; s/_/./g')"
    pass "system TensorRT ${LIB_VER:-unknown} ($TRT_LIB)"
    if [ -n "$ENG_VER" ] && [ -n "$LIB_VER" ] && [ "$ENG_VER" != "$LIB_VER" ]; then
        fail "models/model.engine was built with TensorRT $ENG_VER, the system has $LIB_VER"
        say  "    engines are version-locked: install TensorRT $ENG_VER (list versions with"
        say  "    apt-cache madison libnvinfer${ENG_VER%%.*}) or rebuild the engine in"
        say  "    NeuralBloch, then re-run deps/tensorrt/build_shim.sh."
    fi
else
    MAJ="${ENG_VER%%.*}"
    say "· system TensorRT not installed (optional; only the TensorRT model needs it):"
    say "    Debian/Ubuntu: sudo apt-get install libnvinfer${MAJ:-<major>} libnvinfer-headers-dev"
    say "    RPM:           sudo dnf install libnvinfer${MAJ:-<major>} libnvinfer-headers-devel"
    if [ -n "$ENG_VER" ]; then
        say "    models/model.engine needs exactly version $ENG_VER — list what the NVIDIA"
        say "    repo offers with 'apt-cache madison libnvinfer$MAJ' and pin that version."
    fi
fi
if [ -f "$ROOT/deps/tensorrt/libtrtjl.so" ]; then
    pass "shim:   deps/tensorrt/libtrtjl.so"
else
    say "· shim not built — run deps/tensorrt/build_shim.sh once to enable the TensorRT model"
fi

# ---------------------------------------------------------------------------
printf '\n\033[1m%d ok, %d missing\033[0m\n' "$ok" "$bad"
if [ "$bad" -eq 0 ]; then
    cat <<'EOF'

Ready. Every session from now on:

    source env.sh
    julia --project=. -e 'using Pkg; Pkg.test()'      # 169 tests
    julia --project=. scripts/compare_models.jl       # the comparison
EOF
else
    echo
    echo "Fix the items marked ✗ above, then re-run ./bootstrap.sh."
    exit 1
fi
