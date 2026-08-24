# MRSTAT.jl unit tests.
#
# Run with:
#   source env.sh                      # binds PythonCall to the torch venv
#   julia --project=. -e 'using Pkg; Pkg.test()'
#
# The suite covers deterministic in-package invariants only. Comparisons
# against external references (pure-python model execution, cross-model
# accuracy, benchmarks) are validation *scripts* under scripts/, not unit
# tests.
#
# Layout: this file is a thin entry point; each included file carries its own
# top-level @testset. The loaded surrogate is an expensive shared fixture, so
# it is created once here (top-level, not inside a @testset — testset scopes
# are local and invisible to included files).

using Test
using CUDA
using MRSTAT

# Package-quality checks need no GPU — run them first so they report even on
# a machine without one.
include("test_quality.jl")

# Everything below exercises the CUDA (+ torch, for the surrogate) stack.
# The package is CUDA-only by design, so a missing GPU is a hard failure,
# not a skip.
CUDA.functional() || error(
    "MRSTAT tests require a functional CUDA GPU (the package is CUDA-only)")

const SURROGATE = load_surrogate()

include("test_forwardmodels.jl")
include("test_objective.jl")
include("test_synthetic_data.jl")
