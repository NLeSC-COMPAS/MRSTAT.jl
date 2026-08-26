# =============================================================================
# MRSTATNeuralBlochExt — the surrogate's NeuralBloch.jl-touching half
# =============================================================================
#
# Everything here is the code that talks to NeuralBloch.jl (torch import,
# zero-copy tensor crossings, task-stream scoping): the load_surrogate
# implementation and the two ForwardModel prediction methods, moved verbatim
# out of src/ForwardModels/surrogate.jl. The type, its contract methods and
# the input staging stay in the package — this extension only supplies the
# behavior that needs the bridge, and activates automatically on
# `using NeuralBloch` (env.sh puts a ../NeuralBloch.jl checkout on the Julia
# load path). Loading MRSTAT alone never requires any of this.
module MRSTATNeuralBlochExt

using MRSTAT.ForwardModels: ForwardModels, SurrogateModel, neuralbloch_repo, models_dir,
                            _stage_input!, _dim, _num_layers, pystr_path
import MRSTAT.ForwardModels: compute_echos, compute_echos_and_derivatives
using MRSTAT.DerivativeOperations: ∂mˣʸ∂T₁T₂
using NeuralBloch: torch, to_torch, from_torch, on_task_stream
using PythonCall: Py, pyimport, pyconvert, pywith, pydict, pylist
using CUDA
using StructArrays: StructArray

# The implementation behind ForwardModels.load_surrogate — same signature,
# same defaults, same behavior; see that function's docstring.
function _load_surrogate(;
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

function __init__()
    ForwardModels._load_surrogate_impl[] = _load_surrogate
    return nothing
end

end # module MRSTATNeuralBlochExt
