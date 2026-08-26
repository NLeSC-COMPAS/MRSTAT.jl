# =============================================================================
# SurrogateModel — the learned forward model
# =============================================================================
#
# Wraps the trained PyTorch surrogate from the sibling `NeuralBloch`
# repository (checkpoint `outputs/model.pt`) as a ForwardModel: one network
# forward predicts the echo trains and, when asked, their T₁/T₂ derivatives
# for all voxels at once.
#
# No data is copied between the two languages: Julia hands torch a
# description of the input buffer (via NeuralBloch.jl's DLPack export) and
# wraps torch's output buffers directly as Julia arrays (tensor-pointer
# wrap). The only device work besides the network itself is three small
# broadcasts that write `[log T₁, log T₂, B₁]` into the input buffer. All of
# it — staging broadcasts, torch kernels, and the reconstruction's own CUDA
# kernels afterwards — is queued on the current Julia task's stream
# (`on_task_stream`), which guarantees execution order by itself, so no
# synchronization fences are needed anywhere.
#
# The pulse sequence is baked into the checkpoint (FISP3D, 600 readouts,
# TR 5.58 ms, TE 2.745 ms, TI 10.84 ms, inversion prepulse, 5 repetitions
# sampling the last): the surrogate *is* the sequence, and experiments must
# use a trajectory with `nreadouts == nechos(model)` — `check_experiment`
# enforces that, plus the trained B₁ range.
#
# The NeuralBloch.jl-touching half — the load_surrogate implementation and
# the two prediction methods — lives in the MRSTATNeuralBlochExt package
# extension, so loading MRSTAT alone never requires the NeuralBloch.jl
# package; `using NeuralBloch` activates it (env.sh puts a ../NeuralBloch.jl
# checkout on the Julia load path). This file keeps everything that does not
# touch NeuralBloch: the type, its contract methods, and the input staging.

"""
    SurrogateModel <: ForwardModel

Handle to the loaded surrogate: the FCModel (on the GPU, in eval mode, with
its decoder matrices re-ordered for zero-copy complex output — see
[`load_surrogate`](@ref)), the echo-train length `NTR` it was trained for,
and a persistent `(3, N)` input staging buffer that is (re)allocated only
when the voxel count changes.
"""
mutable struct SurrogateModel <: ForwardModel
    model::Py                     # FCModel: .cuda().eval(), U rows permuted
    NTR::Int                      # echo-train length baked into the checkpoint
    Xbuf::CUDA.CuMatrix{Float32}  # (3, N) staging: rows = log T₁, log T₂, B₁
end

Base.show(io::IO, s::SurrogateModel) =
    print(io, "SurrogateModel(NTR = $(s.NTR), staged voxels = $(size(s.Xbuf, 2)))")

nechos(s::SurrogateModel) = s.NTR

# The surrogate is only trustworthy on the parameter box it was trained on
# (T₁ ∈ [0.1, 5] s, T₂ ∈ [0.01, 2] s); outside it, its predictions are
# extrapolations with no accuracy guarantee. The default bounds therefore
# pin the box-constrained solver to that region. (A physical simulator has
# no such validity edge — see the generic default_bounds.)
default_bounds(::SurrogateModel) = (
    T₁T₂ρˣρʸ(log(1.0), log(0.100), 1.0, 0.0),    # x0
    T₁T₂ρˣρʸ(log(0.1), log(0.010), -Inf, -Inf),  # LB
    T₁T₂ρˣρʸ(log(5.0), log(2.000), Inf, Inf),    # UB
)

function check_experiment(s::SurrogateModel, trajectory, transmit_field)
    nreadouts(trajectory) == nechos(s) || throw(ArgumentError(
        "trajectory has $(nreadouts(trajectory)) readouts but the surrogate " *
        "predicts $(nechos(s)) echos per voxel (fixed by its checkpoint)"))
    all(b -> 0.8 ≤ b ≤ 1.2, transmit_field) || throw(ArgumentError(
        "transmit field leaves the surrogate's trained B₁ range [0.8, 1.2]"))
    return nothing
end

# --- loading -----------------------------------------------------------------

# Read the i-th dimension of one weight matrix stored in the checkpoint.
# These reads are how the network architecture is reconstructed: the
# checkpoint's weight shapes are the source of truth. Building the model
# from the yaml config files in the NeuralBloch repo instead could silently
# construct a network that doesn't match the saved weights; deriving the
# sizes from the weights makes that impossible, and the strict
# `load_state_dict` afterwards double-checks every key and shape anyway.
_dim(sd::Py, key::String, i::Int) = pyconvert(Int, sd[key].shape[i-1])

# Count how many Linear layers one sub-network has, by counting its
# `.weight` entries in the checkpoint. This count is exactly the
# `num_layers` argument FCModel's SubNetwork constructor expects (its layer
# list is: first Linear + (num_layers - 2) hidden Linears + final Linear).
function _num_layers(keys::Vector{String}, subnet::String)
    prefix = "$(subnet).net."
    return count(k -> startswith(k, prefix) && endswith(k, ".weight"), keys)
end

"""
    neuralbloch_repo() -> String

Path to the sibling NeuralBloch python repository — the model code, the
trained checkpoint and the RF tables all live there.

Defaults to `../NeuralBloch` next to this package, which is how the
repositories are meant to be checked out. Set the `NEURALBLOCH_DIR`
environment variable to point elsewhere: `pkgdir` resolves to the *worktree*
root, so a git worktree created outside the shared parent directory is the
usual reason the default misses.
"""
function neuralbloch_repo()
    dir = get(ENV, "NEURALBLOCH_DIR",
              joinpath(dirname(pkgdir(@__MODULE__)), "NeuralBloch"))
    isdir(dir) || error(
        "NeuralBloch repository not found at $dir; it is expected as a " *
        "sibling checkout of this package. Set the NEURALBLOCH_DIR " *
        "environment variable to its path.")
    return dir
end

"""
    load_surrogate(; neuralbloch_dir, model_path) -> SurrogateModel

Load the trained surrogate for use as a reconstruction forward model.

- Imports the model *code* (`FCModel`) from the sibling `NeuralBloch`
  repository — the network runs through its own implementation, not a
  re-implementation — and rebuilds the architecture with dimensions derived
  from the checkpoint's weight shapes.
- Loads the checkpoint strictly (every key and shape validated).
- **Re-orders the rows of the three decoder matrices** `U_signal`/`U_dT1`/
  `U_dT2` from the training layout (complex values as concatenated halves
  `[re₁..re_NTR | im₁..im_NTR]`) to interleaved pairs
  `[re₁, im₁, re₂, im₂, …]`. The matrices are bias-free, so this is an exact
  permutation of output features with zero runtime cost — and it makes
  `torch.view_as_complex` a free view, which is what lets the outputs come
  back to Julia as `ComplexF32` matrices without any interleaving copy.
  The permutation is verified numerically against the unpermuted model
  before the function returns.
- Moves the model to the GPU in eval mode.

This is the only place torch is imported — loading MRSTAT alone never
touches Python beyond what PythonPlot already does. The model *code* comes
from the sibling repository resolved by [`neuralbloch_repo`](@ref)
(`../NeuralBloch` next to this package, or `NEURALBLOCH_DIR`); the
*checkpoint* comes from the in-repo artifacts directory resolved by
[`models_dir`](@ref) (`models/model.pt`, populated by `bootstrap.sh`).
Pass `model_path`/`neuralbloch_dir` to override either directly.

The implementation lives in the `MRSTATNeuralBlochExt` package extension:
run `using NeuralBloch` first (after `source env.sh`, which puts a
`../NeuralBloch.jl` checkout on the Julia load path) — without it this
function raises an error saying exactly that.
"""
function load_surrogate(; kwargs...)
    impl = _load_surrogate_impl[]
    impl === nothing && error(
        "the surrogate lives behind the NeuralBloch.jl package extension — " *
        "`source env.sh` (it puts a ../NeuralBloch.jl checkout on the Julia " *
        "load path), then run `using NeuralBloch` before load_surrogate(); " *
        "see the README's SurrogateModel section.")
    return impl(; kwargs...)
end

# Installed by MRSTATNeuralBlochExt.__init__ — the extension holds the
# implementation because it is the code that talks to NeuralBloch.jl. A Ref
# rather than an extension method, so the no-extension case gets the
# instructive error above instead of a MethodError (an extension cannot add
# and fall back on the same zero-positional-argument method).
const _load_surrogate_impl = Ref{Any}(nothing)

# Convert a Julia path to a Python str, so it can be compared against and
# inserted into Python's sys.path entries (which are str objects).
pystr_path(p::AbstractString) = PythonCall.pystr(String(p))

# --- input staging -----------------------------------------------------------

# Write the network's input for the current tissue parameters into the
# persistent (3, N) staging buffer: row 1 = log T₁, row 2 = log T₂,
# row 3 = B₁. These raw values are exactly what the model was trained on —
# NeuralBloch's data pipeline (data_loading.py) applies the logs and nothing
# else, no normalization. The buffer lives in the model handle and is only
# reallocated when the voxel count changes, so during a reconstruction
# (fixed voxel count, many objective evaluations) this function performs
# three small GPU broadcasts and allocates nothing.
function _stage_input!(s::SurrogateModel, parameters)
    T₁ = parameters.T₁
    T₁ isa CUDA.CuArray || throw(ArgumentError(
        "surrogate expects GPU parameters — send them over with `gpu(f32(parameters))`"))
    N = length(parameters)
    N > 0 || throw(ArgumentError("empty parameter set"))
    if size(s.Xbuf, 2) != N
        s.Xbuf = CUDA.CuMatrix{Float32}(undef, 3, N)
    end
    X = s.Xbuf
    @views begin
        X[1, :] .= log.(T₁)
        X[2, :] .= log.(parameters.T₂)
        X[3, :] .= parameters.B₁
    end
    return N
end

# The interface implementation (compute_echos, compute_echos_and_derivatives)
# and the load_surrogate body live in ext/MRSTATNeuralBlochExt.jl.
