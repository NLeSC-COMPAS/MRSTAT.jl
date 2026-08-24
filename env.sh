# Bind PythonCall to a Python interpreter that has torch, so `load_surrogate()`
# — and everything downstream of it — works.
#
# Usage:   source env.sh      # then start Julia as usual
#
# Sets exactly two variables:
#   JULIA_CONDAPKG_BACKEND=Null   stops PythonCall/CondaPkg from building an
#                                 environment of its own; we bind an existing one
#   JULIA_PYTHONCALL_EXE=<python> the interpreter to bind
#
# The interpreter is the first candidate below that can `import torch`:
#   1. $JULIA_PYTHONCALL_EXE   — already set by you
#   2. $MRSTAT_PYTHON          — explicit override
#   3. ./.venv/bin/python      — created by ./bootstrap.sh
#   4. ../NeuralBloch.jl/.venv/bin/python — the sibling bridge's venv, if present
#   5. python3 on PATH
#
# Nothing here requires a sibling repository: any Python with a CUDA-enabled
# torch will do. Run ./bootstrap.sh once if you have none.

_mrstat_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

_mrstat_pick_python() {
    local c
    for c in "${JULIA_PYTHONCALL_EXE:-}" \
             "${MRSTAT_PYTHON:-}" \
             "$_mrstat_root/.venv/bin/python" \
             "$_mrstat_root/../NeuralBloch.jl/.venv/bin/python" \
             "$(command -v python3 2>/dev/null || true)"; do
        [ -n "$c" ] && [ -x "$c" ] || continue
        if "$c" -c 'import torch' >/dev/null 2>&1; then
            # Normalise, so the exported path reads cleanly (no embedded ../).
            printf '%s' "$(cd "$(dirname "$c")" && pwd)/$(basename "$c")"
            return 0
        fi
    done
    return 1
}

if _mrstat_python="$(_mrstat_pick_python)"; then
    export JULIA_CONDAPKG_BACKEND=Null
    export JULIA_PYTHONCALL_EXE="$_mrstat_python"
    "$_mrstat_python" - <<'PY'
import torch
cuda = "CUDA ok" if torch.cuda.is_available() else "NO CUDA -- the package is GPU-only"
try:
    import matplotlib          # PythonPlot needs it in this interpreter,
    mpl = ""                   # and cannot install it under CONDAPKG_BACKEND=Null
except ImportError:
    mpl = "\n  WARNING: matplotlib missing -- scripts/ that plot will fail at `using PythonPlot`"
print(f"MRSTAT: torch {torch.__version__}, {cuda}{mpl}")
PY
    echo "MRSTAT: PythonCall bound to $JULIA_PYTHONCALL_EXE"
else
    echo "MRSTAT: no Python with torch found -- PythonCall left unbound." >&2
    echo "        Run ./bootstrap.sh, or point MRSTAT_PYTHON at an interpreter" >&2
    echo "        that has a CUDA-enabled torch, then source this file again." >&2
fi

unset -f _mrstat_pick_python
unset _mrstat_root _mrstat_python
