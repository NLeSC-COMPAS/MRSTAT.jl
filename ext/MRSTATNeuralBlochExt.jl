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
using PythonCall: Py, pyimport, pyconvert, pywith, pydict, pylist, pyhasattr
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

    # The decoders are Sequential(Linear, Unflatten), so their Linear weights
    # sit at the `U_*.0.weight` keys.
    config = pydict(Dict{String,Int}(
        "input_dim"          => _dim(sd, "subnet_signal.net.0.weight", 2),
        "NB_signal"          => _dim(sd, "U_signal.0.weight", 2),
        "NB_dT1"             => _dim(sd, "U_dT1.0.weight", 2),
        "NB_dT2"             => _dim(sd, "U_dT2.0.weight", 2),
        "hidden_size_signal" => _dim(sd, "subnet_signal.net.0.weight", 1),
        "hidden_size_dT1"    => _dim(sd, "subnet_dT1.net.0.weight", 1),
        "hidden_size_dT2"    => _dim(sd, "subnet_dT2.net.0.weight", 1),
        "num_layers_signal"  => _num_layers(keys, "subnet_signal"),
        "num_layers_dT1"     => _num_layers(keys, "subnet_dT1"),
        "num_layers_dT2"     => _num_layers(keys, "subnet_dT2"),
    ))
    _dim(sd, "U_signal.0.weight", 1) == 2NTR || error(
        "checkpoint inconsistent: U_signal has $(_dim(sd, "U_signal.0.weight", 1)) " *
        "rows but NTR = $NTR expects $(2NTR)")

    model = FCModel(NTR, config)
    model.load_state_dict(sd)   # strict=True: validates every key and shape
    model.eval()

    # Contract guard: the model takes physical [T₁, T₂, B₁] (it applies the
    # log to T₁/T₂ internally) and emits each output as (batch, NTR, 2) with
    # (Re, Im) paired on the last axis; forward_signal is the signal-only
    # sub-path compute_echos runs, sharing the input transform and signal
    # head with the full forward. Probe both once on the CPU model and
    # refuse to continue on any disagreement — every zero-copy view
    # downstream builds on exactly these shapes.
    pyhasattr(model, "forward_signal") || error(
        "the FCModel at $neuralbloch_dir has no forward_signal method — " *
        "update the NeuralBloch checkout (the signal-only fast path needs it)")
    probe = tm.tensor(((1.2f0, 0.08f0, 1.05f0),))   # physical [T₁, T₂, B₁], in-box
    out, sig = pywith(tm.no_grad()) do _
        model(probe), model.forward_signal(probe)
    end
    shape(t::Py) = pyconvert(Vector{Int}, pylist(t.shape))
    for i in 0:2
        shape(out[i]) == [1, NTR, 2] || error(
            "surrogate load self-check failed: output $i has shape " *
            "$(shape(out[i])), expected (1, $NTR, 2) — the model code's " *
            "output contract has drifted from what this wrapper expects")
    end
    pyconvert(Bool, tm.equal(sig, out[0])) || error(
        "surrogate load self-check failed: forward_signal disagrees with " *
        "the full forward — refusing to continue with a divergent fast path")

    model = model.cuda()
    model.eval()

    return SurrogateModel(model, NTR, CUDA.CuMatrix{Float32}(undef, 3, 0))
end

# View a (N, NTR, 2) network output — (Re, Im) paired on the last axis, the
# layout the model's decoders emit natively — as an (N, NTR) complex tensor
# without copying anything: view_as_complex just reinterprets the same
# memory.
_as_complex(tm::Py, t::Py) = tm.view_as_complex(t)

# --- the interface implementation ---------------------------------------------

"""
    compute_echos(s::SurrogateModel, parameters) -> echos

Predict the echo train for every voxel with the signal sub-network:
`echos::CuMatrix{ComplexF32}` of size `(nechos(s), length(parameters))`,
indexed `echos[readout, voxel]`, at unit proton density.

This is the cheap path for objective mode 0 (cost/residual evaluations
during the solver's step-acceptance loop): it runs `FCModel.forward_signal`
— the input transform, signal sub-network and signal decoder only — because
those evaluations never use the derivatives. `forward_signal` shares its
modules with the full forward, so the two paths agree bitwise (checked once
at load time).

The result is zero-copy in both directions: the returned Julia matrix wraps
torch's output buffer directly (and keeps the tensor alive for as long as
the Julia array is referenced), and because the torch forward runs on the
current Julia task's CUDA stream, any Julia kernel that later reads `echos`
on this task is automatically ordered after it — no fences.
"""
function compute_echos(s::SurrogateModel, parameters)
    _stage_input!(s, parameters)
    tm = torch()
    mc = on_task_stream() do
        X = to_torch(s.Xbuf; sync = false, layout = :reversed)  # (N, 3), zero-copy
        pywith(tm.no_grad()) do _
            _as_complex(tm, s.model.forward_signal(X))   # (N, NTR, 2)
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
    _stage_input!(s, parameters)
    tm = torch()
    mc, d1c, d2c = on_task_stream() do
        X = to_torch(s.Xbuf; sync = false, layout = :reversed)
        pywith(tm.no_grad()) do _
            out = s.model(X)   # (signal, dm/dT₁, dm/dT₂), each (N, NTR, 2)
            (_as_complex(tm, out[0]),
             _as_complex(tm, out[1]),
             _as_complex(tm, out[2]))
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
