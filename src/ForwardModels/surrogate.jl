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
"""
function load_surrogate(;
    neuralbloch_dir::AbstractString = neuralbloch_repo(),
    model_path::AbstractString = joinpath(models_dir(), "model.pt"),
)
    isfile(model_path) || error(
        "Surrogate checkpoint not found at $model_path — run ./bootstrap.sh " *
        "next to a NeuralBloch checkout (it copies outputs/model.pt in), copy " *
        "the file in yourself, or pass `model_path`.")
    isfile(joinpath(neuralbloch_dir, "src", "model.py")) || error(
        "NeuralBloch python repo not found at $neuralbloch_dir " *
        "(no src/model.py) — pass `neuralbloch_dir`.")

    tm = torch()   # lazy import with venv-binding guidance on failure
    pyconvert(Bool, tm.cuda.is_available()) || error(
        "torch reports no CUDA device; the MR-STAT reconstruction requires " *
        "a GPU-enabled torch (check the Python environment PythonCall is bound to).")

    # The network class comes from the NeuralBloch repository itself — the
    # model runs through its own implementation, not a re-implementation.
    # Putting the repo root on Python's import path makes
    # `from src.model import FCModel` work exactly as NeuralBloch's own
    # scripts do (they run with that root as the working directory).
    sys = pyimport("sys")
    dir = pystr_path(neuralbloch_dir)
    pyconvert(Bool, sys.path.__contains__(dir)) || sys.path.insert(0, dir)
    FCModel = pyimport("src.model").FCModel

    # Load the checkpoint the same way NeuralBloch's load_model (utils.py)
    # does: torch.load with map_location, then a strict state-dict load into
    # a freshly built FCModel. The file stores {model_state, epoch, NTR}.
    ckpt = tm.load(model_path; map_location = "cpu")
    NTR = pyconvert(Int, ckpt["NTR"])
    sd = ckpt["model_state"]
    keys = pyconvert(Vector{String}, pylist(sd.keys()))

    config = pydict(Dict{String,Int}(
        "input_dim"          => _dim(sd, "subnet_signal.net.0.weight", 2),
        "NB_signal"          => _dim(sd, "U_signal.weight", 2),
        "NB_dT1"             => _dim(sd, "U_dT1.weight", 2),
        "NB_dT2"             => _dim(sd, "U_dT2.weight", 2),
        "hidden_size_signal" => _dim(sd, "subnet_signal.net.0.weight", 1),
        "hidden_size_dT1"    => _dim(sd, "subnet_dT1.net.0.weight", 1),
        "hidden_size_dT2"    => _dim(sd, "subnet_dT2.net.0.weight", 1),
        "num_layers_signal"  => _num_layers(keys, "subnet_signal"),
        "num_layers_dT1"     => _num_layers(keys, "subnet_dT1"),
        "num_layers_dT2"     => _num_layers(keys, "subnet_dT2"),
    ))
    _dim(sd, "U_signal.weight", 1) == 2NTR || error(
        "checkpoint inconsistent: U_signal has $(_dim(sd, "U_signal.weight", 1)) " *
        "rows but NTR = $NTR expects $(2NTR)")

    model = FCModel(NTR, config)
    model.load_state_dict(sd)   # strict=True: validates every key and shape
    model.eval()

    # What follows re-orders the decoder outputs, so first record what the
    # untouched model produces on a fixed probe input — the self-check below
    # compares against this reference.
    probe = tm.tensor(((0.18f0, -2.53f0, 1.05f0),))   # some [logT₁, logT₂, B₁]
    ref = pywith(tm.no_grad()) do _
        model(probe)
    end

    # Re-order the rows of the three decoder matrices so the model emits
    # complex values as adjacent (Re, Im) pairs instead of two separate
    # halves. This is what makes the zero-copy complex output possible, and
    # it costs nothing at inference time: permuting the rows of a bias-free
    # Linear layer permutes its output features exactly. The permutation
    # [0, NTR, 1, NTR+1, …] puts old row k (Re of echo k+1) at position 2k
    # and old row NTR+k (Im of echo k+1) at position 2k+1.
    perm = tm.stack((tm.arange(0, NTR), tm.arange(NTR, 2NTR)); dim = 1).reshape(-1)
    for U in (model.U_signal, model.U_dT1, model.U_dT2)
        U.weight.data = U.weight.data.index_select(0, perm)
    end

    # Self-check: undoing the interleave on the permuted model's output must
    # reproduce the reference bit-for-bit. Exact comparison (torch.equal) is
    # correct here because a row permutation only reorders whole dot
    # products — it cannot change any individual value — so any difference
    # at all means the assumed checkpoint layout is wrong.
    out = pywith(tm.no_grad()) do _
        model(probe)
    end
    for i in 0:2
        deinterleaved = out[i].view(1, NTR, 2).transpose(1, 2).reshape(1, 2NTR)
        pyconvert(Bool, tm.equal(deinterleaved, ref[i])) || error(
            "surrogate load self-check failed: permuted decoder output " *
            "$(i) does not match the checkpoint's layout — refusing to " *
            "continue with a mis-ordered model")
    end

    model = model.cuda()
    model.eval()

    return SurrogateModel(model, NTR, CUDA.CuMatrix{Float32}(undef, 3, 0))
end

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

# View a raw (N, 2·NTR) network output as an (N, NTR) complex tensor,
# without copying anything. This only works because load_surrogate permuted
# the decoder rows so that adjacent value pairs are already (Re, Im) of one
# echo; both `view` and `view_as_complex` then just reinterpret the same
# memory.
_as_complex(tm::Py, t::Py, N::Int, NTR::Int) = tm.view_as_complex(t.view(N, NTR, 2))

# --- the interface implementation ---------------------------------------------

"""
    compute_echos(s::SurrogateModel, parameters) -> echos

Predict the echo train for every voxel with the signal sub-network:
`echos::CuMatrix{ComplexF32}` of size `(nechos(s), length(parameters))`,
indexed `echos[readout, voxel]`, at unit proton density.

This is the cheap path for objective mode 0 (cost/residual evaluations
during the solver's step-acceptance loop): it runs only the signal
sub-network, because those evaluations never use the derivatives.

The result is zero-copy in both directions: the returned Julia matrix wraps
torch's output buffer directly (and keeps the tensor alive for as long as
the Julia array is referenced), and because the torch forward runs on the
current Julia task's CUDA stream, any Julia kernel that later reads `echos`
on this task is automatically ordered after it — no fences.
"""
function compute_echos(s::SurrogateModel, parameters)
    N = _stage_input!(s, parameters)
    tm = torch()
    mc = on_task_stream() do
        X = to_torch(s.Xbuf; sync = false, layout = :reversed)  # (N, 3), zero-copy
        pywith(tm.no_grad()) do _
            m = s.model.U_signal(s.model.subnet_signal(X))
            _as_complex(tm, m, N, s.NTR)
        end
    end
    # torch's row-major (N, NTR) complex tensor is, byte for byte, a Julia
    # column-major (NTR, N) matrix — which is exactly the
    # echos[readout, voxel] orientation the reconstruction kernels index.
    return from_torch(mc; sync = false)
end

"""
    compute_echos_and_derivatives(s::SurrogateModel, parameters) -> (echos, ∂echos)

Predict the echo train and its partial derivatives with respect to T₁ and
T₂ (physical units, per second) for every voxel with one full network
forward. `echos` is as in [`compute_echos`](@ref); `∂echos` is a
`(nechos(s), N)` matrix with element type `∂mˣʸ∂T₁T₂{ComplexF32}` — the
exact structure the `Jv`/`Jᴴv` kernels index — realized as a `StructArray`
over the two prediction matrices, so no interleaving copy is ever made.
The log-space chain rule (`∂/∂log T = T·∂/∂T`) is *not* applied here; the
kernels' `lin_scale` factors do that, unchanged.
"""
function compute_echos_and_derivatives(s::SurrogateModel, parameters)
    N = _stage_input!(s, parameters)
    tm = torch()
    mc, d1c, d2c = on_task_stream() do
        X = to_torch(s.Xbuf; sync = false, layout = :reversed)
        pywith(tm.no_grad()) do _
            out = s.model(X)   # (signal, dm/dT₁, dm/dT₂), each (N, 2·NTR)
            (_as_complex(tm, out[0], N, s.NTR),
             _as_complex(tm, out[1], N, s.NTR),
             _as_complex(tm, out[2], N, s.NTR))
        end
    end
    echos = from_torch(mc; sync = false)
    ∂T₁ = from_torch(d1c; sync = false)
    ∂T₂ = from_torch(d2c; sync = false)
    # The kernels expect ∂echos to be one matrix whose elements hold both
    # derivatives (∂echos[r, v].∂T₁ / .∂T₂). A StructArray over the two
    # prediction matrices provides exactly that view without interleaving
    # the data into a new buffer — zero copies. Passing a StructArray into
    # @cuda kernels is already established practice in this codebase (the
    # `parameters` StructArray takes the same route).
    ∂echos = StructArray{∂mˣʸ∂T₁T₂{ComplexF32}}((∂T₁, ∂T₂))
    return echos, ∂echos
end
