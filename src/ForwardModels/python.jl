# =============================================================================
# PythonModel — the naive Python baseline forward model
# =============================================================================
#
# The same trained NeuralBloch checkpoint the other two learned models run
# (`models/model.pt`), driven through PyTorch the way one wires Julia to a
# Python model on the first attempt: every value crosses the language
# boundary as an ordinary copy, and every copy goes through host memory.
#
# This model exists to be the *baseline* of the forward-model comparison. It
# is deliberately not optimized: it is what `SurrogateModel`'s zero-copy
# DLPack bridge and `TensorRTModel`'s compiled engine are measured against,
# so their numbers can be read as what their machinery actually buys.
# Nothing here should ever be made faster — a faster baseline measures less.
#
# What "naive" means, precisely:
#   - the GPU tissue parameters are copied down to the host with `Array` and
#     assembled into a plain `(N, 3)` Float32 matrix — one host allocation
#     and three device→host transfers per call, no persistent staging buffer;
#   - that matrix is handed to Python as a numpy array
#     (`numpy.ascontiguousarray`, a real copy into numpy-owned memory),
#     wrapped by `torch.from_numpy` and pushed to the device with `.cuda()`;
#   - each network output comes back with `.detach().cpu().numpy()`, is
#     copied into a Julia `Array{Float32,3}` by `pyconvert`, reassembled from
#     its (Re, Im) pairs into a `(NTR, N)` ComplexF32 matrix, and uploaded
#     with `CuArray`.
#
# No DLPack, no tensor-pointer wrapping, no `view_as_complex`, no pinned
# memory, no persistent buffers, no stream games. The host round trip is what
# orders everything — every crossing blocks the host — which is why this path
# needs no stream discipline at all, and is exactly the property the zero-copy
# arm has to work for.
#
# The network itself still runs on the GPU, in eval mode under `no_grad`, and
# `compute_echos` still takes the checkpoint's signal-only sub-path. The
# comparison therefore isolates data plumbing and language overhead, not
# GPU-vs-CPU compute or a different amount of network work. Numerically this
# model *is* the surrogate: the same weights through the same torch kernels on
# the same inputs, so the two agree bitwise (the test suite asserts it).

"""
    PythonModel <: ForwardModel

The naive Python baseline: the trained FCModel (on the GPU, in eval mode)
driven through PythonCall with ordinary conversions and copies at every
crossing — no DLPack, no zero-copy views, no persistent buffers.

Deliberately the slowest of the learned forward models, and deliberately
kept that way: it is the reference point that says what
[`SurrogateModel`](@ref)'s zero-copy bridge and [`TensorRTModel`](@ref)'s
compiled engine buy. See [`load_python_model`](@ref).

Fields: the model, the torch and numpy modules (resolved once at load time,
so the hot path never re-imports), and the echo-train length `NTR` baked
into the checkpoint.
"""
struct PythonModel <: ForwardModel
    model::Py     # FCModel: .cuda().eval()
    torch::Py     # the torch module
    numpy::Py     # the numpy module
    NTR::Int      # echo-train length baked into the checkpoint
end

Base.show(io::IO, m::PythonModel) =
    print(io, "PythonModel(NTR = $(m.NTR), naive PythonCall baseline)")

nechos(m::PythonModel) = m.NTR

# Same weights as the surrogate ⇒ the same validity edge: the parameter box
# the network was trained on (T₁ ∈ [0.1, 5] s, T₂ ∈ [0.01, 2] s). Spelled out
# rather than shared, as for TensorRTModel: the three learned models are
# independent implementations that happen to agree.
default_bounds(::PythonModel) = (
    T₁T₂ρˣρʸ(log(1.0), log(0.100), 1.0, 0.0),    # x0
    T₁T₂ρˣρʸ(log(0.1), log(0.010), -Inf, -Inf),  # LB
    T₁T₂ρˣρʸ(log(5.0), log(2.000), Inf, Inf),    # UB
)

function check_experiment(m::PythonModel, trajectory, transmit_field)
    nreadouts(trajectory) == nechos(m) || throw(ArgumentError(
        "trajectory has $(nreadouts(trajectory)) readouts but the python " *
        "baseline predicts $(nechos(m)) echos per voxel (fixed by its checkpoint)"))
    all(b -> 0.8 ≤ b ≤ 1.2, transmit_field) || throw(ArgumentError(
        "transmit field leaves the surrogate's trained B₁ range [0.8, 1.2]"))
    return nothing
end

# --- loading -----------------------------------------------------------------

# Import torch, with the same binding guidance NeuralBloch.jl's `torch()`
# gives — this model reaches Python through PythonCall directly (PythonCall is
# already a hard dependency of MRSTAT), so it must produce that guidance
# itself instead of borrowing the bridge's.
function _torch_module()
    try
        return pyimport("torch")
    catch err
        exe = try
            pyconvert(String, pyimport("sys").executable)
        catch
            "unknown"
        end
        error("""
            the python baseline could not import `torch` (Python in use: $exe).

            Bind PythonCall to a Python environment that has PyTorch:
              • `source env.sh` before starting Julia (it picks an interpreter
                with torch and sets JULIA_PYTHONCALL_EXE / JULIA_CONDAPKG_BACKEND);
              • or export JULIA_CONDAPKG_BACKEND=Null and
                JULIA_PYTHONCALL_EXE=/path/to/venv/bin/python yourself.

            Original error: $(sprint(showerror, err))""")
    end
end

"""
    load_python_model(; neuralbloch_dir, model_path) -> PythonModel

Load the trained checkpoint as the naive Python baseline forward model.

Same artifact and same code path into Python as [`load_surrogate`](@ref) —
the model *code* (`FCModel`) is imported from the sibling `NeuralBloch`
repository resolved by [`neuralbloch_repo`](@ref), the architecture is
rebuilt from the checkpoint's weight shapes, the state dict is loaded
strictly, and the model's I/O contract (physical `[T₁, T₂, B₁]` in, each
output `(batch, NTR, 2)` with (Re, Im) paired on the last axis,
`forward_signal` agreeing bitwise with the full forward) is probed once
before the model is moved to the GPU in eval mode.

What differs is everything *after* loading: this model exchanges data with
torch through ordinary PythonCall conversions and host round trips, where
[`load_surrogate`](@ref)'s result uses NeuralBloch.jl's zero-copy DLPack
bridge. It therefore needs no NeuralBloch.jl checkout and no package
extension — PythonCall, a CUDA-enabled torch, the sibling model code and
`models/model.pt` are the whole requirement.

The *checkpoint* comes from the in-repo artifacts directory resolved by
[`models_dir`](@ref) (`models/model.pt`, populated by `bootstrap.sh`); pass
`model_path`/`neuralbloch_dir` to override either location.
"""
function load_python_model(;
    neuralbloch_dir::AbstractString = neuralbloch_repo(),
    model_path::AbstractString = joinpath(models_dir(), "model.pt"),
)
    isfile(model_path) || error(
        "Model checkpoint not found at $model_path — run ./bootstrap.sh " *
        "next to a NeuralBloch checkout (it copies outputs/model.pt in), copy " *
        "the file in yourself, or pass `model_path`.")
    isfile(joinpath(neuralbloch_dir, "src", "model.py")) || error(
        "NeuralBloch python repo not found at $neuralbloch_dir " *
        "(no src/model.py) — pass `neuralbloch_dir`.")

    tm = _torch_module()
    pyconvert(Bool, tm.cuda.is_available()) || error(
        "torch reports no CUDA device; the MR-STAT reconstruction requires " *
        "a GPU-enabled torch (check the Python environment PythonCall is bound to).")
    np = pyimport("numpy")

    # The network class comes from the NeuralBloch repository itself — the
    # model runs through its own implementation, not a re-implementation.
    sys = pyimport("sys")
    dir = pystr_path(neuralbloch_dir)
    pyconvert(Bool, sys.path.__contains__(dir)) || sys.path.insert(0, dir)
    FCModel = pyimport("src.model").FCModel

    # The checkpoint stores {model_state, epoch, NTR}; the architecture is
    # rebuilt from the saved weight shapes (see `_dim` / `_num_layers` in
    # surrogate.jl for why the shapes are the source of truth), and the
    # strict `load_state_dict` afterwards validates every key and shape.
    ckpt = tm.load(model_path; map_location = "cpu")
    NTR = pyconvert(Int, ckpt["NTR"])
    sd = ckpt["model_state"]
    keys = pyconvert(Vector{String}, pylist(sd.keys()))
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

    # Contract guard, as in the surrogate loader: physical [T₁, T₂, B₁] in,
    # each output (batch, NTR, 2) with (Re, Im) paired on the last axis, and
    # forward_signal agreeing bitwise with the full forward. This baseline
    # reassembles the pairs itself, so the pairing is exactly as load-bearing
    # here as the complex view is there.
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
            "python baseline load self-check failed: output $i has shape " *
            "$(shape(out[i])), expected (1, $NTR, 2) — the model code's " *
            "output contract has drifted from what this wrapper expects")
    end
    pyconvert(Bool, tm.equal(sig, out[0])) || error(
        "python baseline load self-check failed: forward_signal disagrees " *
        "with the full forward — refusing to continue with a divergent fast path")

    model = model.cuda()
    model.eval()

    return PythonModel(model, tm, np, NTR)
end

# --- the crossings, one ordinary step at a time --------------------------------
#
# Each stage is its own small function: the reconstruction just composes them,
# and `scripts/compare_models_N.jl` times them individually to show where the
# baseline's milliseconds go. They are internal (not exported) — the model's
# public surface is the ForwardModel interface.

# Stage 1 — device→host. Copy the tissue parameters down and assemble the
# network's `(N, 3)` Float32 input: column 1 = T₁, 2 = T₂, 3 = B₁, physical
# units (FCModel applies the log to T₁/T₂ itself). Three device→host
# transfers and a fresh host matrix every call — no staging buffer is kept.
function _py_host_input(::PythonModel, parameters)
    T₁ = parameters.T₁
    T₁ isa CUDA.CuArray || throw(ArgumentError(
        "the python baseline expects GPU parameters — send them over with " *
        "`gpu(f32(parameters))`"))
    N = length(parameters)
    N > 0 || throw(ArgumentError("empty parameter set"))
    X = Matrix{Float32}(undef, N, 3)
    X[:, 1] = Array(T₁)
    X[:, 2] = Array(parameters.T₂)
    X[:, 3] = Array(parameters.B₁)
    return X
end

# Stage 2 — host→python→device. `ascontiguousarray` copies the Julia matrix
# into a C-contiguous numpy array that owns its memory (so torch never holds
# a pointer into Julia's heap), `from_numpy` wraps it, `.cuda()` performs the
# host→device transfer. Float32 all the way: numpy inherits the element type
# and nothing here promotes.
_py_device_input(m::PythonModel, X::Matrix{Float32}) =
    m.torch.from_numpy(m.numpy.ascontiguousarray(X)).cuda()

# Stage 3 — the network. Identical work to the surrogate's: eval mode, no
# grad, all three heads (or the signal head alone for the cheap path).
_py_forward(m::PythonModel, x::Py) = pywith(m.torch.no_grad()) do _
    m.model(x)                  # (signal, ∂m/∂T₁, ∂m/∂T₂), each (N, NTR, 2)
end
_py_forward_signal(m::PythonModel, x::Py) = pywith(m.torch.no_grad()) do _
    m.model.forward_signal(x)   # (N, NTR, 2)
end

# Stage 4 — device→host. A blocking transfer of the `(N, NTR, 2)` output into
# a host numpy array.
_py_to_host(t::Py) = t.detach().cpu().numpy()

# Stage 5 — python→Julia. PythonCall's ordinary array conversion: a full copy
# into a Julia-owned `(N, NTR, 2)` array. Note it also transposes storage
# order — numpy is row-major, Julia is column-major — so this is a strided
# gather, not a memcpy.
_py_to_julia(a::Py) = pyconvert(Array{Float32,3}, a)

# Stage 6 — pairs→complex, `(N, NTR, 2)` → `(NTR, N)`. The model emits (Re, Im)
# paired on the last axis; the reconstruction wants a ComplexF32 matrix
# indexed `[readout, voxel]`. Built the plain way: interleave, then transpose.
_py_pair_to_complex(A::Array{Float32,3}) =
    permutedims(complex.(@view(A[:, :, 1]), @view(A[:, :, 2])))

# Stage 7 — host→device. The result the reconstruction owns and mutates.
_py_upload(H::Matrix{ComplexF32}) = CUDA.CuArray(H)

# Stages 4–7 for one network output.
_py_result(t::Py) = _py_upload(_py_pair_to_complex(_py_to_julia(_py_to_host(t))))

# --- the interface implementation ---------------------------------------------

"""
    compute_echos(m::PythonModel, parameters) -> echos

Predict the echo train for every voxel with the checkpoint's signal
sub-network: `echos::CuMatrix{ComplexF32}` of size
`(nechos(m), length(parameters))`, indexed `echos[readout, voxel]`, at unit
proton density.

The cheap path for objective mode 0, as for the surrogate: it runs
`FCModel.forward_signal`, so only one output crosses back. Every crossing is
an ordinary copy through host memory — see the file header and the
`_py_*` stages.
"""
function compute_echos(m::PythonModel, parameters)
    X = _py_host_input(m, parameters)
    x = _py_device_input(m, X)
    return _py_result(_py_forward_signal(m, x))
end

"""
    compute_echos_and_derivatives(m::PythonModel, parameters) -> (echos, ∂echos)

Predict the echo train and its partial derivatives with respect to T₁ and T₂
(physical units, per second — no chain rule is applied here; the `Jv`/`Jᴴv`
kernels do that) for every voxel with one full network forward. `echos` is as
in [`compute_echos`](@ref); `∂echos` is the usual `(nechos(m), N)`
`StructArray` over the two derivative matrices with element type
`∂mˣʸ∂T₁T₂{ComplexF32}`.

Three outputs cross back instead of one, each by the same ordinary route:
device→host, numpy→Julia copy, (Re, Im) pairs interleaved into a transposed
ComplexF32 matrix, host→device upload.
"""
function compute_echos_and_derivatives(m::PythonModel, parameters)
    X = _py_host_input(m, parameters)
    x = _py_device_input(m, X)
    out = _py_forward(m, x)
    echos = _py_result(out[0])
    ∂T₁ = _py_result(out[1])
    ∂T₂ = _py_result(out[2])
    return echos, StructArray{∂mˣʸ∂T₁T₂{ComplexF32}}((∂T₁, ∂T₂))
end
