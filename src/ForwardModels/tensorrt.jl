# =============================================================================
# TensorRTModel — the compiled forward model
# =============================================================================
#
# Runs the same trained NeuralBloch surrogate as `SurrogateModel`, but through
# its prebuilt TensorRT engine (`outputs/model.engine` in the sibling
# repository) instead of the live PyTorch module. No Python is involved
# anywhere: Julia ccalls into `deps/tensorrt/libtrtjl.so`, a thin C wrapper
# over the TensorRT C++ runtime (built once by `deps/tensorrt/build_shim.sh`).
#
# Data crossing is zero-copy on the input and one broadcast on each output:
# the persistent `(3, N)` staging buffer is byte-identical to the engine's
# row-major `(N, 3)` input tensor, so its raw device pointer is bound directly
# (`setTensorAddress`) and TensorRT reads it in place. The engine writes its
# `(N, 2·NTR)` outputs into persistent buffers the same way; those store each
# voxel's echo train as two concatenated halves `[Re₁..Re_NTR | Im₁..Im_NTR]`
# — a layout frozen into the engine at export time — so one fused broadcast
# per output combines the halves into the `(NTR, N)` ComplexF32 matrix the
# reconstruction kernels index. That broadcast is the only copy in the path.
#
# Everything — staging broadcasts, the engine enqueue, the combine broadcasts,
# and the reconstruction's own kernels afterwards — runs on the current Julia
# task's CUDA stream, so execution order is guaranteed by the stream alone and
# no host synchronization happens anywhere.
#
# The engine is a black box with a hard edge: its optimization profile fixes
# the admissible batch range (queried at load — for the shipped example engine
# MIN 10000 / OPT 14400 / MAX 40000 voxels), its echo count NTR is baked in,
# and it always computes all three outputs, so `compute_echos` costs the same
# as `compute_echos_and_derivatives` (the PyTorch surrogate's signal-only
# fast path has no engine equivalent). Serving other batch sizes or sequences
# means rebuilding the engine in the NeuralBloch repository — never here.

using Libdl: dlopen, dlsym

"""
    TrtApi

The shim's 20 `trtjl_*` entry points, resolved once with `dlsym` at load time
into plain function pointers. Call sites use these directly, so the hot path
pays a raw indirect call — no per-call symbol lookup, and the package itself
never references the shim library (it loads fine when the shim was never
built).
"""
struct TrtApi
    build_version::Ptr{Cvoid}
    lib_version::Ptr{Cvoid}
    last_error::Ptr{Cvoid}
    open::Ptr{Cvoid}
    close::Ptr{Cvoid}
    nb_io::Ptr{Cvoid}
    io_name::Ptr{Cvoid}
    is_input::Ptr{Cvoid}
    elt_size::Ptr{Cvoid}
    engine_shape::Ptr{Cvoid}
    profile_shape::Ptr{Cvoid}
    device_mem_size::Ptr{Cvoid}
    nb_aux_streams::Ptr{Cvoid}
    set_input_shape::Ptr{Cvoid}
    current_shape::Ptr{Cvoid}
    set_address::Ptr{Cvoid}
    enqueue::Ptr{Cvoid}
    capture::Ptr{Cvoid}
    has_graph::Ptr{Cvoid}
    replay::Ptr{Cvoid}
end

TrtApi(lib::Ptr{Cvoid}) =
    TrtApi((dlsym(lib, Symbol(:trtjl_, name)) for name in fieldnames(TrtApi))...)

"""
    TensorRTModel <: ForwardModel

Handle to the loaded engine: the shim context, the echo-train length `NTR`
and admissible batch range read from the engine itself, and the persistent
device buffers TensorRT is bound to — the `(3, N)` input staging buffer and
one raw `(2·NTR, N)` buffer per output. Buffers are (re)allocated and rebound
only when the voxel count changes; holding them as fields keeps them rooted
for as long as TensorRT holds their raw pointers.
"""
mutable struct TensorRTModel <: ForwardModel
    h::Ptr{Cvoid}                       # shim context (runtime+engine+execution context)
    api::TrtApi
    NTR::Int                            # echo-train length baked into the engine
    minB::Int                           # admissible batch range: the engine's
    optB::Int                           #   optimization profile (MIN/OPT/MAX)
    maxB::Int
    N::Int                              # currently bound batch size (0 = none yet)
    Xbuf::CUDA.CuMatrix{Float32}        # (3, N) staging: rows = log T₁, log T₂, B₁
    raw_signal::CUDA.CuMatrix{Float32}  # (2NTR, N) halves layout, written by the engine
    raw_dT1::CUDA.CuMatrix{Float32}
    raw_dT2::CUDA.CuMatrix{Float32}
    use_graph::Bool                     # capture a CUDA graph per batch size?
    graph_ok::Bool                      # a graph for the current binding is live
end

Base.show(io::IO, m::TensorRTModel) = print(io,
    "TensorRTModel(NTR = $(m.NTR), batch ∈ [$(m.minB), $(m.maxB)], " *
    "bound = $(m.N), graph = $(m.graph_ok))")

nechos(m::TensorRTModel) = m.NTR

# The engine runs the surrogate's weights, so it inherits the surrogate's
# validity edge: the parameter box the network was trained on. Same values as
# SurrogateModel's — duplicated deliberately, the two implementations share no
# code.
default_bounds(::TensorRTModel) = (
    T₁T₂ρˣρʸ(log(1.0), log(0.100), 1.0, 0.0),    # x0
    T₁T₂ρˣρʸ(log(0.1), log(0.010), -Inf, -Inf),  # LB
    T₁T₂ρˣρʸ(log(5.0), log(2.000), Inf, Inf),    # UB
)

function check_experiment(m::TensorRTModel, trajectory, transmit_field)
    nreadouts(trajectory) == nechos(m) || throw(ArgumentError(
        "trajectory has $(nreadouts(trajectory)) readouts but the engine " *
        "predicts $(nechos(m)) echos per voxel (fixed at engine build time)"))
    all(b -> 0.8 ≤ b ≤ 1.2, transmit_field) || throw(ArgumentError(
        "transmit field leaves the surrogate's trained B₁ range [0.8, 1.2]"))
    m.minB ≤ length(transmit_field) ≤ m.maxB || throw(ArgumentError(
        "experiment has $(length(transmit_field)) voxels but the engine's " *
        "optimization profile only serves batches in [$(m.minB), $(m.maxB)] — " *
        "rebuild the engine in NeuralBloch (config/export.yaml) for other sizes"))
    return nothing
end

# --- ccall plumbing ----------------------------------------------------------

_lasterr(api::TrtApi, h::Ptr{Cvoid}) =
    unsafe_string(ccall(api.last_error, Cstring, (Ptr{Cvoid},), h))
_lasterr(m::TensorRTModel) = _lasterr(m.api, m.h)

@inline _streamptr() = reinterpret(Ptr{Cvoid}, CUDA.stream().handle)
@inline _devptr(A::CUDA.CuArray) = Ptr{Cvoid}(UInt(pointer(A)))

# Engine-level (static) shape of a tensor, in TensorRT (row-major) order;
# -1 marks a dynamic dimension.
function _engine_shape(api::TrtApi, h::Ptr{Cvoid}, name::String)
    dims = Vector{Int64}(undef, 8)
    nd = ccall(api.engine_shape, Int32, (Ptr{Cvoid}, Cstring, Ptr{Int64}), h, name, dims)
    nd < 0 && error("cannot query the engine shape of tensor $name")
    return Tuple(dims[1:nd])
end

# The optimization-profile shape of an input tensor. sel: 0=MIN, 1=OPT, 2=MAX.
function _profile_shape(api::TrtApi, h::Ptr{Cvoid}, name::String, sel::Int)
    dims = Vector{Int64}(undef, 8)
    nd = ccall(api.profile_shape, Int32, (Ptr{Cvoid}, Cstring, Int32, Ptr{Int64}),
               h, name, sel, dims)
    nd < 0 && error("cannot query the optimization profile of tensor $name")
    return Tuple(dims[1:nd])
end

# --- loading -----------------------------------------------------------------

"""
    load_tensorrt_model(; engine_path, shim_path, device, use_graph) -> TensorRTModel

Load the prebuilt TensorRT engine as a reconstruction forward model.

- `engine_path` defaults to `model.engine` in the in-repo artifacts directory
  (resolved by [`models_dir`](@ref), so `MRSTAT_MODELS_DIR` relocates it);
  `bootstrap.sh` copies the file in from a NeuralBloch checkout when one is
  present. Engines are built there, never here.
- `shim_path` defaults to the in-repo shim `deps/tensorrt/libtrtjl.so`;
  build it once with `deps/tensorrt/build_shim.sh`.
- `device` selects the GPU (default: the current CUDA device).
- `use_graph = true` captures the whole engine invocation as a CUDA graph
  once per batch size, so each inference costs a single `cudaGraphLaunch`
  submission instead of one launch per engine layer.

The loader validates the engine against the contract this model expects —
one FP32 input `x` of shape `(batch, 3)` and three FP32 outputs
`signal`/`dT1`/`dT2` of shape `(batch, 2·NTR)` — and reads `NTR` plus the
admissible batch range from the engine, so an engine rebuilt upstream with a
different profile (or echo count) is picked up automatically or rejected
loudly, never mis-read.

Everything TensorRT does runs in the same CUDA (primary) context CUDA.jl
uses, which is what makes raw `CuArray` pointers directly bindable.
"""
function load_tensorrt_model(;
    engine_path::AbstractString = joinpath(models_dir(), "model.engine"),
    shim_path::AbstractString = joinpath(pkgdir(@__MODULE__), "deps", "tensorrt", "libtrtjl.so"),
    device = nothing,
    use_graph::Bool = true,
)
    isfile(shim_path) || error(
        "TensorRT shim not built at $shim_path — run deps/tensorrt/build_shim.sh " *
        "once (it compiles the C wrapper against the system TensorRT install).")
    isfile(engine_path) || error(
        "TensorRT engine not found at $engine_path — put one there: run " *
        "./bootstrap.sh next to a NeuralBloch checkout (it copies " *
        "outputs/model.engine in, building it upstream with `uv run run.py " *
        "export_onnx && uv run run.py build_engine` if needed), or copy the " *
        "file in yourself; `engine_path`/MRSTAT_MODELS_DIR override the location.")

    CUDA.functional() || error(
        "the TensorRT forward model needs a working CUDA GPU")
    device === nothing || CUDA.device!(device)
    # Make CUDA.jl bind the device's primary context on this thread before
    # TensorRT initializes, so both sides live in the same context and device
    # pointers can cross freely.
    CUDA.context()

    api = TrtApi(dlopen(shim_path))
    bv = ccall(api.build_version, Int32, ())
    lv = ccall(api.lib_version, Int32, ())
    bv == lv || @warn(
        "the shim was built against TensorRT $bv but the runtime reports $lv — " *
        "rerun deps/tensorrt/build_shim.sh if loading fails")

    # The execution context eagerly allocates the engine's scratch memory
    # (~440 MiB for the shipped engine). If that collides with memory CUDA.jl's
    # pool is sitting on, release the pool once and retry.
    h = ccall(api.open, Ptr{Cvoid}, (Cstring,), engine_path)
    if h == C_NULL
        CUDA.reclaim()
        h = ccall(api.open, Ptr{Cvoid}, (Cstring,), engine_path)
    end
    h == C_NULL && error(
        "loading the TensorRT engine failed: " * _lasterr(api, C_NULL) *
        " — if the engine was rebuilt against a different TensorRT version, " *
        "install the matching system TensorRT and re-run deps/tensorrt/build_shim.sh.")

    empty() = CUDA.CuMatrix{Float32}(undef, 0, 0)
    m = TensorRTModel(h, api, 0, 0, 0, 0, 0,
                      CUDA.CuMatrix{Float32}(undef, 3, 0), empty(), empty(), empty(),
                      use_graph, false)
    finalizer(close, m)

    # Validate the engine against the contract this model is written for.
    nio = ccall(api.nb_io, Int32, (Ptr{Cvoid},), h)
    names = [unsafe_string(ccall(api.io_name, Cstring, (Ptr{Cvoid}, Int32), h, i))
             for i in 0:nio-1]
    sort(names) == ["dT1", "dT2", "signal", "x"] || error(
        "engine at $engine_path serves I/O tensors $names — expected exactly " *
        "x, signal, dT1, dT2 (is this a NeuralBloch surrogate engine?)")
    for name in names
        isin = ccall(api.is_input, Int32, (Ptr{Cvoid}, Cstring), h, name)
        (isin == 1) == (name == "x") || error(
            "engine tensor $name has the wrong I/O direction")
        es = ccall(api.elt_size, Int32, (Ptr{Cvoid}, Cstring), h, name)
        es == 4 || error(
            "engine tensor $name has a $es-byte element type, expected Float32 " *
            "(4 bytes) — this model does not handle reduced-precision I/O")
    end
    xdims = _engine_shape(api, h, "x")
    (length(xdims) == 2 && xdims[2] == 3) || error(
        "engine input x has TensorRT shape $xdims, expected (batch, 3) = " *
        "[log T₁, log T₂, B₁] per voxel")
    sdims = _engine_shape(api, h, "signal")
    (length(sdims) == 2 && sdims[2] ≥ 2 && iseven(sdims[2])) || error(
        "engine output signal has TensorRT shape $sdims, expected (batch, 2·NTR)")
    NTR = Int(sdims[2]) ÷ 2
    for name in ("dT1", "dT2")
        d = _engine_shape(api, h, name)
        d == sdims || error(
            "engine output $name has shape $d but signal has $sdims — " *
            "the three outputs must agree")
    end

    m.NTR = NTR
    m.minB = Int(_profile_shape(api, h, "x", 0)[1])
    m.optB = Int(_profile_shape(api, h, "x", 1)[1])
    m.maxB = Int(_profile_shape(api, h, "x", 2)[1])
    1 ≤ m.minB ≤ m.optB ≤ m.maxB || error(
        "engine optimization profile is inconsistent: MIN $(m.minB), " *
        "OPT $(m.optB), MAX $(m.maxB)")

    @info "TensorRT forward model loaded" engine = engine_path tensorrt = lv NTR #=
        =# batch = (min = m.minB, opt = m.optB, max = m.maxB) #=
        =# scratch_MiB = round(ccall(api.device_mem_size, Int64, (Ptr{Cvoid},), h) / 2^20; digits = 1) #=
        =# aux_streams = ccall(api.nb_aux_streams, Int32, (Ptr{Cvoid},), h)
    return m
end

"""
    close(m::TensorRTModel)

Release the engine, its execution context and any captured CUDA graph. Also
runs as the finalizer; safe to call more than once. The device buffers are
ordinary CuArrays and go back to the pool via GC.
"""
function Base.close(m::TensorRTModel)
    if m.h != C_NULL
        ccall(m.api.close, Cvoid, (Ptr{Cvoid},), m.h)
        m.h = C_NULL
    end
    return nothing
end

# --- binding & staging -------------------------------------------------------

# (Re)bind the engine to freshly allocated I/O buffers for batch size N.
# Runs only when the voxel count changes — during a reconstruction (fixed
# count, many objective evaluations) this is a one-time setup. The engine's
# optimization profile is a hard gate on N: outside it, TensorRT has no
# compiled kernels for the shape, so fail with the actual range.
function _bind!(m::TensorRTModel, N::Int)
    N == m.N && return nothing
    m.minB ≤ N ≤ m.maxB || throw(ArgumentError(
        "batch of $N voxels is outside the engine's optimization profile " *
        "[$(m.minB), $(m.maxB)] (opt $(m.optB)) — the range is a property of " *
        "the prebuilt engine; rebuild it in NeuralBloch (config/export.yaml: " *
        "MIN/OPT/MAX_BATCH_SIZE) to serve other sizes"))

    m.N = 0   # binding in progress: a failure below must force a clean rebind
    m.Xbuf = CUDA.CuMatrix{Float32}(undef, 3, N)
    # Deterministic content (any finite value) rather than recycled pool
    # bytes, so the graph-capture warm-up below computes on defined data.
    fill!(m.Xbuf, 0.0f0)
    m.raw_signal = CUDA.CuMatrix{Float32}(undef, 2m.NTR, N)
    m.raw_dT1 = CUDA.CuMatrix{Float32}(undef, 2m.NTR, N)
    m.raw_dT2 = CUDA.CuMatrix{Float32}(undef, 2m.NTR, N)

    # Julia's column-major (3, N) buffer is byte-identical to the row-major
    # (N, 3) tensor the engine expects — dims are simply reversed.
    rc = ccall(m.api.set_input_shape, Int32, (Ptr{Cvoid}, Cstring, Ptr{Int64}, Int32),
               m.h, "x", Int64[N, 3], 2)
    rc == 0 || error("setInputShape(x → ($N, 3)) failed: " * _lasterr(m))

    # Cross-check the shapes TensorRT resolved for the outputs before trusting
    # our buffer sizes.
    dims = Vector{Int64}(undef, 8)
    for name in ("signal", "dT1", "dT2")
        nd = ccall(m.api.current_shape, Int32, (Ptr{Cvoid}, Cstring, Ptr{Int64}),
                   m.h, name, dims)
        (nd == 2 && dims[1] == N && dims[2] == 2m.NTR) || error(
            "engine resolved output $name to $(Tuple(dims[1:max(nd, 0)])), " *
            "expected ($N, $(2m.NTR))")
    end

    for (name, A) in (("x", m.Xbuf), ("signal", m.raw_signal),
                      ("dT1", m.raw_dT1), ("dT2", m.raw_dT2))
        rc = ccall(m.api.set_address, Int32, (Ptr{Cvoid}, Cstring, Ptr{Cvoid}),
                   m.h, name, _devptr(A))
        rc == 0 || error("setTensorAddress($name) failed: " * _lasterr(m))
    end

    m.N = N
    m.graph_ok = false
    if m.use_graph
        # Capture the whole engine invocation as a CUDA graph, once per batch
        # size: the hot path then submits a single cudaGraphLaunch instead of
        # one kernel launch per engine layer. The shim warms up and
        # synchronizes on the task stream first (reading the constants written
        # above), captures on its own private stream, and replay later runs on
        # whatever stream each call provides. An engine that cannot be
        # captured keeps the per-call enqueueV3 path — more host overhead,
        # same results.
        rc = ccall(m.api.capture, Int32, (Ptr{Cvoid}, Ptr{Cvoid}), m.h, _streamptr())
        if rc == 0
            m.graph_ok = true
        else
            @warn("CUDA-graph capture failed — falling back to per-call enqueueV3",
                  code = rc, detail = _lasterr(m), maxlog = 1)
        end
    end
    return nothing
end

# Write the network input for the current tissue parameters into the staging
# buffer: row 1 = log T₁, row 2 = log T₂, row 3 = B₁ — the exact values the
# network was trained on (NeuralBloch's data pipeline applies the logs and
# nothing else). Three small GPU broadcasts on the task stream; TensorRT reads
# the buffer in place afterwards, ordered by the stream.
function _stage!(m::TensorRTModel, parameters)
    T₁ = parameters.T₁
    T₁ isa CUDA.CuArray || throw(ArgumentError(
        "the TensorRT forward model expects GPU parameters — send them over " *
        "with `gpu(f32(parameters))`"))
    N = length(parameters)
    N > 0 || throw(ArgumentError("empty parameter set"))
    _bind!(m, N)
    X = m.Xbuf
    @views begin
        X[1, :] .= log.(T₁)
        X[2, :] .= log.(parameters.T₂)
        X[3, :] .= parameters.B₁
    end
    return N
end

# Submit one engine invocation on the current task's stream. Asynchronous:
# nothing here or downstream synchronizes the host; stream order alone makes
# the results visible to whoever reads the output buffers next on this task.
@inline function _infer!(m::TensorRTModel)
    s = _streamptr()
    if m.graph_ok
        rc = ccall(m.api.replay, Int32, (Ptr{Cvoid}, Ptr{Cvoid}), m.h, s)
        rc == 0 || error("cudaGraphLaunch failed: " * _lasterr(m))
    else
        rc = ccall(m.api.enqueue, Int32, (Ptr{Cvoid}, Ptr{Cvoid}), m.h, s)
        rc == 0 || error("enqueueV3 failed: " * _lasterr(m))
    end
    return nothing
end

# Combine an output's two concatenated halves [Re₁..Re_NTR | Im₁..Im_NTR]
# (the layout frozen into the engine) into a fresh (NTR, N) complex matrix —
# one fused device broadcast, the only copy in the whole path. Fresh because
# the reconstruction mutates results in place (phase encoding) and results
# from successive calls must not alias.
function _combine(m::TensorRTModel, raw::CUDA.CuMatrix{Float32}, N::Int)
    out = CUDA.CuMatrix{ComplexF32}(undef, m.NTR, N)
    @views out .= complex.(raw[1:m.NTR, :], raw[m.NTR+1:2m.NTR, :])
    return out
end

# --- the interface implementation --------------------------------------------

"""
    compute_echos(m::TensorRTModel, parameters) -> echos

Predict the echo train for every voxel with one engine invocation:
`echos::CuMatrix{ComplexF32}` of size `(nechos(m), length(parameters))`,
indexed `echos[readout, voxel]`, at unit proton density.

Unlike the PyTorch surrogate, the engine has no signal-only sub-path — it
always computes all three outputs — so this costs the same as
[`compute_echos_and_derivatives`](@ref); the derivative outputs are simply
left in their raw buffers.
"""
function compute_echos(m::TensorRTModel, parameters)
    N = _stage!(m, parameters)
    _infer!(m)
    return _combine(m, m.raw_signal, N)
end

"""
    compute_echos_and_derivatives(m::TensorRTModel, parameters) -> (echos, ∂echos)

Predict the echo train and its partial derivatives with respect to T₁ and T₂
(physical units, per second — the engine emits them that way, matching the
convention the `Jv`/`Jᴴv` kernels expect) for every voxel with one engine
invocation. `∂echos` is the usual `(nechos(m), N)` `StructArray` over the two
derivative matrices with element type `∂mˣʸ∂T₁T₂{ComplexF32}`.
"""
function compute_echos_and_derivatives(m::TensorRTModel, parameters)
    N = _stage!(m, parameters)
    _infer!(m)
    echos = _combine(m, m.raw_signal, N)
    ∂T₁ = _combine(m, m.raw_dT1, N)
    ∂T₂ = _combine(m, m.raw_dT2, N)
    return echos, StructArray{∂mˣʸ∂T₁T₂{ComplexF32}}((∂T₁, ∂T₂))
end
