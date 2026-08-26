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
hdr "4. Sibling checkout (the one Julia path dependency)"
# ---------------------------------------------------------------------------
# NeuralBloch.jl is not in the General registry, so it is a `Pkg.develop`
# path dep. Manifest.toml is not tracked, so a fresh clone has none and
# cannot resolve NeuralBloch at all until step 6 points the dep at this
# checkout. Every other dependency, BlochSimulators included, comes from
# the registry and needs nothing here.
if [ -d ../NeuralBloch.jl ]; then
    pass "../NeuralBloch.jl"
else
    fail "../NeuralBloch.jl missing — git clone"
    say  "    https://github.com/NLeSC-COMPAS/NeuralBloch.jl.git beside this repo."
fi

# ---------------------------------------------------------------------------
hdr "5. NeuralBloch python repository (model code, RF tables, checkpoint)"
# ---------------------------------------------------------------------------
NB="${NEURALBLOCH_DIR:-$ROOT/../NeuralBloch}"
[ -d "$NB" ] && NB="$(cd "$NB" && pwd)"
if [ -d "$NB" ]; then
    pass "$NB"
    for f in src/model.py data/rf_data/rf_angle_list.txt data/rf_data/rf_phase_list.txt; do
        [ -f "$NB/$f" ] && pass "  $f" || fail "  $f missing"
    done
    if [ -f "$NB/outputs/model.pt" ]; then
        pass "  outputs/model.pt ($(du -h "$NB/outputs/model.pt" | cut -f1))"
    else
        fail "  outputs/model.pt missing — the checkpoint is gitignored upstream;"
        say  "    train it there or copy one in. Without it load_surrogate() fails"
        say  "    and only the BlochModel route works."
    fi
else
    fail "not found at $NB — clone git@github.com:NLeSC-COMPAS/NeuralBloch.git"
    say  "  beside this repo, or set NEURALBLOCH_DIR to its path."
fi

# ---------------------------------------------------------------------------
hdr "6. Julia environment"
# ---------------------------------------------------------------------------
if [ -d ../NeuralBloch.jl ]; then
    export JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE="$PY"
    julia --project=. -e '
        using Pkg
        # Point the path dep at THIS machine, then resolve from scratch.
        Pkg.develop(PackageSpec(path = abspath("../NeuralBloch.jl")))
        Pkg.instantiate(); Pkg.precompile()' >/tmp/mrstat-bootstrap.$$ 2>&1 \
        && pass "path dep set, project instantiated and precompiled" \
        || { fail "Julia setup failed:"; sed 's/^/    /' /tmp/mrstat-bootstrap.$$; }
    rm -f /tmp/mrstat-bootstrap.$$
else
    fail "skipped — ../NeuralBloch.jl missing (see 4)"
fi

# ---------------------------------------------------------------------------
hdr "7. TensorRT forward model (optional)"
# ---------------------------------------------------------------------------
# Only load_tensorrt_model() needs these two; the Bloch and surrogate routes
# are complete without them, so absences here are informational, never
# counted as failures.
if [ -f "$NB/outputs/model.engine" ]; then
    pass "engine: $NB/outputs/model.engine ($(du -h "$NB/outputs/model.engine" | cut -f1))"
else
    say "· engine absent at $NB/outputs/model.engine — build it in NeuralBloch:"
    say "    uv run run.py export_onnx && uv run run.py build_engine"
fi
if [ -f "$ROOT/deps/tensorrt/libtrtjl.so" ]; then
    pass "shim:   deps/tensorrt/libtrtjl.so"
else
    say "· shim not built — run deps/tensorrt/build_shim.sh once to enable it"
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
