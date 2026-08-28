# The four-arm forward-model comparison of scripts/compare_models.jl, run at one
# phantom size N and written to disk as machine-readable results.
#
# Run with:
#   source env.sh                                   # binds PythonCall to the torch venv
#   MRSTAT_N=150 MRSTAT_OUT=/some/dir julia --project=. scripts/compare_models_N.jl
#   julia --project=. scripts/compare_models_N.jl 150 /some/dir   # same, positionally
#
# scripts/compare_models_N.sh drives the four sizes serially, one process each.
#
# Why one N per process
# ---------------------
# The comparison is a GPU benchmark, and every arm has state a second size would
# disturb: the TensorRT engine re-captures its CUDA graph on a batch-size change,
# torch's caching allocator keeps whatever the largest size needed, and Julia's
# JIT has already specialized on the first size's array types. A fresh process
# per N pays its own untimed warm-up and measures a cold, unshared machine — and
# an N that fails (the engine's optimization profile is a hard edge) cannot take
# the other sizes' results with it.
#
# What changes with N, and what does not
# --------------------------------------
# The echo train is fixed at nechos(model) = 600 readouts by the trained
# checkpoint, so the geometry has one degree of freedom left: an N×N image
# acquired over K phase-encode cycles needs K*N == 600 readouts, hence
# K = 600 ÷ N — 6, 5, 4, 3 at N = 100, 120, 150, 200. Three consequences worth
# stating plainly, because they are what the sweep's scaling laws are read
# against:
#   - voxels grow as N² (10,000 / 14,400 / 22,500 / 40,000) — this is the
#     forward models' work and the bytes that cross the language boundary;
#   - readouts stay at 600, while samples per readout grow as N, so the raw data
#     grows as N — the solver's residual, not the forward model's output;
#   - the k-space sampling pattern changes: K cycles of N phase-encode lines
#     each. That is inherent to keeping 600 readouts while N moves, so the four
#     reconstructions are not the same inverse problem at four resolutions. Read
#     the timings as scaling laws; read the accuracy as self-consistency, which
#     is what each arm's inverse crime measures.
#
# The phantom is seeded (MRSTAT_SEED, default 1234) immediately before it is
# built. make_phantom draws its per-region T₁/T₂/ρ with `rand`, and the
# Shepp-Logan BrainWeb template yields the same ten regions in the same
# `unique` order at every N in this sweep, so one seed makes the tissue values
# identical across sizes: what differs between two N is the sampling of the
# phantom, not the phantom. The drawn values are recorded in the results file
# (phantom.T₁_values …) so the claim is checkable rather than asserted.
#
# Everything else — arms, warm-up discipline, the three speed levels, the
# accuracy and parity checks — is scripts/compare_models.jl's, deliberately
# unchanged, so a point of this sweep is directly comparable with that script's
# N=120 baseline. The differences are that every timing is a median of several
# samples with all samples kept, that a few extra quantities are recorded
# (mode-0 forward per arm, host allocation per arm, GPU memory marks, the
# TensorRT-vs-surrogate parity, per-call time series), and that nothing is
# plotted.

ENV["MPLBACKEND"] = "Agg"   # matplotlib without a display; must precede PythonPlot

using MRSTAT
using NeuralBloch   # activates the surrogate's package extension
using MRSTAT.TrustRegionReflective: SolverOptions, solver
# The python baseline's crossing stages, timed individually in level 1b — the
# same functions its compute_echos_and_derivatives composes, so the breakdown
# cannot drift from what the arm actually runs.
using MRSTAT.ForwardModels: _py_host_input, _py_device_input, _py_forward,
                            _py_forward_signal, _py_to_host, _py_to_julia,
                            _py_pair_to_complex, _py_upload, pyconvert
using BlochSimulators
using ComputationalResources: CUDALibs
using CUDA
using StructArrays
using Statistics
using LinearAlgebra: norm
using Printf
using Random
using Dates
using TOML
using Logging: NullLogger, with_logger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

envint(key, default) = parse(Int, get(ENV, key, string(default)))
envflag(key) = get(ENV, key, "0") in ("1", "true", "yes")

const N       = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : envint("MRSTAT_N", 120)
const OUTDIR  = length(ARGS) ≥ 2 ? ARGS[2] :
                get(ENV, "MRSTAT_OUT", joinpath(@__DIR__, "output", "compare_N"))
const SEED    = envint("MRSTAT_SEED", 1234)
const REPS    = envint("MRSTAT_REPS", 7)        # samples per level-1 timing
const STAGEREPS = envint("MRSTAT_STAGE_REPS", 7)  # samples per level-1b stage
const MAXITER = envint("MRSTAT_MAX_ITER", 20)   # SolverOptions' own default
const SKIP_RECON = envflag("MRSTAT_SKIP_RECON") # levels 1 and 1b only (smoke test)

mkpath(OUTDIR)
const RESFILE = joinpath(OUTDIR, "N$(N).toml")

# The results file is rewritten after every stage, so a crash (or an arm that
# the engine's profile refuses) leaves everything measured so far on disk.
const RES = Dict{String,Any}()
sect(name) = get!(RES, name, Dict{String,Any}())
subsect(name, arm) = get!(sect(name), arm, Dict{String,Any}())
function save_results()
    open(RESFILE, "w") do io
        TOML.print(x -> string(x), io, RES; sorted = true)
    end
    return nothing
end

banner(s) = (println("═"^76); println(s); println("═"^76))

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

meta = sect("meta")
meta["N"] = N
meta["seed"] = SEED
meta["level1_reps"] = REPS
meta["stage_reps"] = STAGEREPS
meta["max_iter_trf"] = MAXITER
meta["skip_recon"] = SKIP_RECON
meta["script"] = @__FILE__
meta["started"] = string(now())
meta["hostname"] = gethostname()
meta["julia_version"] = string(VERSION)
meta["gpu"] = CUDA.name(CUDA.device())
meta["cuda_runtime"] = string(CUDA.runtime_version())
meta["cuda_driver"] = string(CUDA.driver_version())
meta["cuda_jl"] = string(pkgversion(CUDA))
meta["gpu_total_MiB"] = CUDA.total_memory() / 2^20
let dir = dirname(@__DIR__)
    for (key, cmd) in ("git_branch" => `git -C $dir rev-parse --abbrev-ref HEAD`,
                       "git_commit" => `git -C $dir rev-parse HEAD`)
        meta[key] = try strip(read(cmd, String)) catch; "unknown" end
    end
end
save_results()

banner("MR-STAT four-arm forward-model comparison at N = $N")
@printf("  results  : %s\n", RESFILE)
@printf("  seed     : %d   solver iterations: %d   level-1 reps: %d\n", SEED, MAXITER, REPS)
@printf("  GPU      : %s (%.0f MiB)\n", meta["gpu"], meta["gpu_total_MiB"])

# ---------------------------------------------------------------------------
# Memory bookkeeping
# ---------------------------------------------------------------------------
#
# `CUDA.memory_info()` reports the device's free/total bytes — the whole GPU,
# not just this process, which is exactly what "how much of the card does this
# configuration need" means (the machine is otherwise idle; the sweep is
# strictly serial). `CUDA.cached_memory()` is what CUDA.jl's pool is holding on
# to, and torch keeps its own allocator statistics; recording all three keeps
# the three consumers separable.

gpu_used_MiB() = let (free, total) = CUDA.memory_info(); (total - free) / 2^20 end
const MEMMARKS = Vector{Dict{String,Any}}()
function mem_mark!(label)
    used = gpu_used_MiB()
    push!(MEMMARKS, Dict{String,Any}(
        "label" => label,
        "gpu_used_MiB" => used,
        "cuda_jl_pool_MiB" => CUDA.cached_memory() / 2^20))
    sect("memory")["marks"] = MEMMARKS
    sect("memory")["gpu_peak_used_MiB"] = maximum(m -> m["gpu_used_MiB"], MEMMARKS)
    return used
end
mem_mark!("process start")

# ---------------------------------------------------------------------------
# The four forward models over the same physics
# ---------------------------------------------------------------------------

banner("Loading the four forward models")

surrogate = load_surrogate()
pybase = load_python_model()
@assert nechos(pybase) == nechos(surrogate)

# The training sequence, reproduced from NeuralBloch's data_generation.jl +
# config/data.yaml (frozen for this checkpoint) — the same lines
# compare_models.jl runs.
neuralbloch_dir = neuralbloch_repo()
readvec(p) = [parse(Float64, l) for l in eachline(p) if !isempty(strip(l))]
flip_deg = readvec(joinpath(neuralbloch_dir, "data", "rf_data", "rf_angle_list.txt"))
phase_deg = readvec(joinpath(neuralbloch_dir, "data", "rf_data", "rf_phase_list.txt"))
rf_train = flip_deg .* exp.(1im .* deg2rad.(phase_deg))
sequence = gpu(f32(FISP3D(rf_train, 0.00558, 0.002745, 64, 0.01084, 0.000, 5, true, true, 1, 0.0)))
bloch = BlochModel(sequence)
@assert nechos(bloch) == nechos(surrogate) == 600

tensorrt = let
    shim = joinpath(dirname(@__DIR__), "deps", "tensorrt", "libtrtjl.so")
    engine = joinpath(models_dir(), "model.engine")
    if isfile(shim) && isfile(engine)
        m = load_tensorrt_model()
        @assert nechos(m) == nechos(surrogate)
        m
    else
        println("note: TensorRT arm skipped — shim or engine missing")
        nothing
    end
end
if tensorrt !== nothing
    meta["trt_batch_min"] = tensorrt.minB
    meta["trt_batch_opt"] = tensorrt.optB
    meta["trt_batch_max"] = tensorrt.maxB
end

# The geometry the checkpoint allows at this N: K*N == 600 readouts.
nechos(surrogate) % N == 0 || error(
    "N = $N does not divide the checkpoint's $(nechos(surrogate)) readouts — choose " *
    "an N with K = $(nechos(surrogate))/N integral (100, 120, 150, 200 …); padding " *
    "or subsampling the geometry to fit would compare four different experiments")
K = nechos(surrogate) ÷ N
meta["K"] = K
meta["nechos"] = nechos(surrogate)
meta["nvoxels"] = N^2
meta["samples_per_readout"] = N
@printf("  geometry : N = %d, K = %d, %d readouts, %d voxels\n", N, K, K * N, N^2)
save_results()
mem_mark!("models loaded")

# ---------------------------------------------------------------------------
# One experiment; one raw-data set per forward model that owns its own physics
# ---------------------------------------------------------------------------

# Seeded immediately before the phantom is drawn — the whole point of the
# control (see the header). generate_synthetic_data calls make_phantom, which
# is the only consumer of randomness in the pipeline.
Random.seed!(SEED)
raw_sur, coordinates, coils, trajectory, transmit_field, phantom =
    generate_synthetic_data(surrogate; N = N, K = K)
@assert length(coordinates) == N^2

params_true = StructArray{T₁T₂B₁ρˣρʸ{Float32}}((
    phantom.T₁, phantom.T₂, gpu(f32(vec(transmit_field))), phantom.ρˣ, phantom.ρʸ))

# The phantom fingerprint: the per-region tissue values actually drawn. Equal
# across N ⇒ the sizes really do sample one phantom.
let ph = sect("phantom")
    ph["nregions"] = length(unique(Array(phantom.T₁)))
    ph["T₁_values"] = sort(unique(Float64.(Array(phantom.T₁))))
    ph["T₂_values"] = sort(unique(Float64.(Array(phantom.T₂))))
    ph["ρˣ_values"] = sort(unique(Float64.(Array(phantom.ρˣ))))
    ph["B₁_range"] = [Float64(minimum(transmit_field)), Float64(maximum(transmit_field))]
end

# The Bloch arm's raw data: same phantom and geometry, echos from the simulator.
echos_bloch = compute_echos(bloch, params_true)
phase_encoding!(echos_bloch, trajectory, coordinates)
raw_bloch = magnetization_to_signal(CUDALibs(), echos_bloch, params_true, trajectory,
                                    coordinates, coils)
echos_bloch = nothing
GC.gc(); CUDA.reclaim()

T₁true = Array(phantom.T₁)
T₂true = Array(phantom.T₂)
ρtrue = abs.(complex.(Array(phantom.ρˣ), Array(phantom.ρʸ)))
save_results()
mem_mark!("data generated")

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

const TM = pybase.torch
syncboth() = (CUDA.synchronize(); TM.cuda.synchronize(); nothing)

# Level 1's definition, from compare_models.jl: a warm call, then CUDA events
# around the call. Repeated here and reduced by the median, with every sample
# kept in the results file — one sample is what the N=120 baseline reported and
# it is enough to be misled by a single GC pause. The wall clock around the
# same region is recorded alongside: for the python arm most of the work is on
# the host, and the two numbers agreeing is the evidence that bracketing host
# work with GPU events measures what it looks like it measures.
function timed_call(f; n = REPS)
    f(); syncboth()
    gpu = Float64[]; wall = Float64[]
    for _ in 1:n
        syncboth()
        w = @elapsed begin
            g = CUDA.@elapsed f()
        end
        push!(gpu, 1000 * Float64(g)); push!(wall, 1000 * w)
    end
    return Dict{String,Any}("ms" => median(gpu), "ms_all" => gpu,
                            "ms_min" => minimum(gpu), "ms_max" => maximum(gpu),
                            "wall_ms" => median(wall))
end

# Wall clock with both runtimes synchronized either side (torch queues on its
# own stream, CUDA.jl on the task stream), median of n, after a warm call —
# level 1b's definition, for stages that are host work with no kernel to put an
# event around.
function stage_ms(f; n = STAGEREPS)
    f(); syncboth()
    ts = map(1:n) do _
        syncboth()
        @elapsed begin; f(); syncboth(); end
    end
    return 1000 * median(ts)
end

# Host memory a call allocates, and what garbage collecting it costs.
function alloc_stats(f; n = 3)
    f(); syncboth()
    bytes = Float64[]; gc = Float64[]; tot = Float64[]
    for _ in 1:n
        st = @timed begin; f(); syncboth(); end
        push!(bytes, st.bytes); push!(gc, st.gctime); push!(tot, st.time)
    end
    return Dict{String,Any}("host_MiB_per_call" => median(bytes) / 2^20,
                            "gc_ms_per_call" => 1000 * median(gc),
                            "gc_fraction" => sum(gc) / sum(tot))
end

# Hand every cached byte back to the driver, between measured sections.
#
# Three allocators share the card and only one of them yields to the others:
# CUDA.jl's pool releases on demand, but torch's caching allocator holds
# whatever its largest call needed until it is told otherwise, and TensorRT's
# scratch is fixed at load. At N=200 one network output is 183 MiB and a
# level-1 arm makes a dozen of them, so an arm that starts after torch has
# cached 8 GiB finds a full card and dies — measured, not hypothesized: the
# first run of this sweep lost the python arm at N=200 to exactly that, with
# CUDA.jl's own pool holding barely a gigabyte of the 10.5 GiB in use.
#
# Called only between arms and between sections, never inside a timed region,
# and every arm re-warms before its samples — so this costs measurements
# nothing, and it makes each arm's memory marks its own footprint instead of
# the running total of everyone before it.
function release_gpu!()
    GC.gc()
    CUDA.reclaim()
    TM.cuda.empty_cache()
    return nothing
end

torch_mem() = Dict{String,Any}(
    "torch_allocated_MiB" => pyconvert(Float64, TM.cuda.memory_allocated()) / 2^20,
    "torch_reserved_MiB"  => pyconvert(Float64, TM.cuda.memory_reserved()) / 2^20,
    "torch_peak_MiB"      => pyconvert(Float64, TM.cuda.max_memory_allocated()) / 2^20)

# The arms, in compare_models.jl's execution order. (The reports order them
# python → Bloch → surrogate → TensorRT; execution order is kept as the
# baseline ran it so a point of this sweep is comparable with that run.)
arms = Any[("surrogate", surrogate), ("bloch", bloch)]
tensorrt === nothing || push!(arms, ("tensorrt", tensorrt))
push!(arms, ("python", pybase))
meta["arms"] = [a[1] for a in arms]

# Each arm runs inside this: an arm that throws (the engine's optimization
# profile is a hard edge, and a size outside it must fail rather than be padded
# or subsampled) records why and leaves the others alone.
function attempt(f::Function, label, arm)
    try
        return f()
    catch err
        io = IOBuffer()
        showerror(io, err, catch_backtrace())
        msg = String(take!(io))
        println("!! arm $arm failed at $label:\n$msg")
        d = subsect("failures", "$(arm)_$(label)")
        d["error"] = first(msg, 4000)
        save_results()
        return nothing
    end
end

# ---------------------------------------------------------------------------
# Speed level 1: the forward model alone
# ---------------------------------------------------------------------------

banner("Speed level 1: forward model alone ($(length(params_true)) voxels)")
for (name, model) in arms
    attempt("level1", name) do
        d = subsect("level1", name)
        merge!(d, timed_call(() -> compute_echos_and_derivatives(model, params_true)))
        # The mode-0 forward as well: the surrogate has a signal-only sub-path,
        # the engine has none (it always computes all three outputs), and the
        # difference is what the solver's cheap calls actually cost.
        d0 = timed_call(() -> compute_echos(model, params_true))
        d["signal_ms"] = d0["ms"]
        d["signal_ms_all"] = d0["ms_all"]
        merge!(d, alloc_stats(() -> compute_echos_and_derivatives(model, params_true)))
        d["ns_per_voxel"] = 1e6 * d["ms"] / N^2
        @printf("  %-10s : %8.2f ms   (mode-0 %7.2f ms, %6.1f ns/voxel, host %6.1f MiB/call)\n",
                name, d["ms"], d["signal_ms"], d["ns_per_voxel"], d["host_MiB_per_call"])
        mem_mark!("level1 $name")
        save_results()
    end
    # Hand every allocator's cache back between arms (see release_gpu!).
    release_gpu!()
end
sect("memory")["torch_after_level1"] = torch_mem()
save_results()

# ---------------------------------------------------------------------------
# Speed level 1b: where the python baseline's milliseconds go
# ---------------------------------------------------------------------------

banner("Speed level 1b: where the python baseline's milliseconds go")
attempt("level1b", "python") do
    d = sect("level1b")
    nvox = length(params_true)
    Xhost = _py_host_input(pybase, params_true)
    xdev = _py_device_input(pybase, Xhost)
    outs = _py_forward(pybase, xdev)
    anp = _py_to_host(outs[0])
    ajl = _py_to_julia(anp)
    hmat = _py_pair_to_complex(ajl)

    d["s1_gpu_to_host_input_ms"]  = stage_ms(() -> _py_host_input(pybase, params_true))
    d["s2_host_to_torch_dev_ms"]  = stage_ms(() -> _py_device_input(pybase, Xhost))
    d["s3_forward_ms"]            = stage_ms(() -> _py_forward(pybase, xdev))
    d["s3_forward_signal_ms"]     = stage_ms(() -> _py_forward_signal(pybase, xdev))
    d["s4_out_to_host_ms"]        = stage_ms(() -> _py_to_host(outs[0]))
    d["s5_numpy_to_julia_ms"]     = stage_ms(() -> _py_to_julia(anp))
    d["s6_pairs_to_complex_ms"]   = stage_ms(() -> _py_pair_to_complex(ajl))
    d["s7_host_to_device_ms"]     = stage_ms(() -> _py_upload(hmat))
    d["whole_call_ms"]            = stage_ms(() -> compute_echos_and_derivatives(pybase, params_true); n = REPS)
    d["whole_call_mode0_ms"]      = stage_ms(() -> compute_echos(pybase, params_true); n = REPS)
    d["MiB_per_output"]           = nvox * nechos(pybase) * 2 * sizeof(Float32) / 2^20
    d["sum_of_stages_ms"] = d["s1_gpu_to_host_input_ms"] + d["s2_host_to_torch_dev_ms"] +
        d["s3_forward_ms"] + 3 * (d["s4_out_to_host_ms"] + d["s5_numpy_to_julia_ms"] +
                                  d["s6_pairs_to_complex_ms"] + d["s7_host_to_device_ms"])
    d["sum_of_stages_mode0_ms"] = d["s1_gpu_to_host_input_ms"] + d["s2_host_to_torch_dev_ms"] +
        d["s3_forward_signal_ms"] + d["s4_out_to_host_ms"] + d["s5_numpy_to_julia_ms"] +
        d["s6_pairs_to_complex_ms"] + d["s7_host_to_device_ms"]

    @printf("  %d voxels × %d echos: %.1f MiB per network output, ×3 outputs\n",
            nvox, nechos(pybase), d["MiB_per_output"])
    for (k, mult) in (("s1_gpu_to_host_input_ms", 1), ("s2_host_to_torch_dev_ms", 1),
                      ("s3_forward_ms", 1), ("s4_out_to_host_ms", 3),
                      ("s5_numpy_to_julia_ms", 3), ("s6_pairs_to_complex_ms", 3),
                      ("s7_host_to_device_ms", 3))
        @printf("  %-30s ×%d %9.1f ms  %5.1f%%\n", k, mult, mult * d[k],
                100 * mult * d[k] / d["whole_call_ms"])
    end
    @printf("  %-34s %9.1f ms  %5.1f%%\n", "sum of stages", d["sum_of_stages_ms"],
            100 * d["sum_of_stages_ms"] / d["whole_call_ms"])
    @printf("  %-34s %9.1f ms\n", "measured whole call", d["whole_call_ms"])
    save_results()
end
mem_mark!("level1b done")
save_results()

# ---------------------------------------------------------------------------
# Numerical parity of the three learned arms, on identical inputs
# ---------------------------------------------------------------------------
#
# The claim the python baseline rests on — same weights, same kernels, so the
# same bits — and the claim the TensorRT arm rests on, that its engine agrees
# with the surrogate far below solver tolerance. Both are measured at every N:
# the engine picks kernels per batch size, so agreement is not a property that
# can be assumed to survive a move to the profile's edges.

banner("Parity on identical inputs")
attempt("parity", "learned") do
    nrmse(a, b) = norm(vec(a) .- vec(b)) / norm(vec(b))
    mx(a, b) = maximum(abs, Array(a) .- Array(b))
    e_su, ∂_su = compute_echos_and_derivatives(surrogate, params_true)
    for (name, model) in (("python", pybase), ("tensorrt", tensorrt))
        model === nothing && continue
        e_x, ∂_x = compute_echos_and_derivatives(model, params_true)
        d = subsect("parity", name)
        d["max_abs_signal"] = mx(e_x, e_su)
        d["max_abs_dT1"] = mx(∂_x.∂T₁, ∂_su.∂T₁)
        d["max_abs_dT2"] = mx(∂_x.∂T₂, ∂_su.∂T₂)
        d["nrmse_signal"] = nrmse(Array(e_x), Array(e_su))
        d["nrmse_dT1"] = nrmse(Array(∂_x.∂T₁), Array(∂_su.∂T₁))
        d["nrmse_dT2"] = nrmse(Array(∂_x.∂T₂), Array(∂_su.∂T₂))
        d["bitwise_identical"] = d["max_abs_signal"] == 0 && d["max_abs_dT1"] == 0 &&
                                 d["max_abs_dT2"] == 0
        @printf("  %-9s vs surrogate: max|Δ| %.3g / %.3g / %.3g   NRMSE %.3g / %.3g / %.3g%s\n",
                name, d["max_abs_signal"], d["max_abs_dT1"], d["max_abs_dT2"],
                d["nrmse_signal"], d["nrmse_dT1"], d["nrmse_dT2"],
                d["bitwise_identical"] ? "  (bitwise identical)" : "")
    end
    save_results()
end
release_gpu!()
mem_mark!("parity done")
save_results()

if SKIP_RECON
    meta["finished"] = string(now())
    save_results()
    println("\nMRSTAT_SKIP_RECON set — levels 2/3 and accuracy skipped.")
    println("compare_models_N (N = $N) finished")
    exit(0)
end

# ---------------------------------------------------------------------------
# Instrumented reconstruction (compare_models.jl's, verbatim)
# ---------------------------------------------------------------------------

function instrumented_recon(model, raw; x0, LB, UB, options = SolverOptions())
    check_experiment(model, trajectory, transmit_field)
    nvox = length(coordinates)
    x0m = repeat(x0', nvox) |> f32
    LBm = repeat(LB', nvox) |> f32
    UBm = repeat(UB', nvox) |> f32
    resource = CUDALibs()

    calls = NamedTuple{(:mode, :seconds),Tuple{Int,Float64}}[]
    objfun = (x, mode) -> begin
        t0 = time()
        result = MRSTAT.objective(x, resource, mode, raw, model, coordinates,
                                  coils, trajectory, transmit_field)
        push!(calls, (mode = mode, seconds = time() - t0))
        result
    end

    wall = @elapsed output = solver(objfun, vec(x0m), vec(LBm), vec(UBm),
                                    options, (x, t) -> nothing)
    return output, calls, wall
end

x0, LB, UB = default_bounds(surrogate)

# Nothing is timed until a throwaway warm-up solve has run with each model: the
# first reconstruction of a session pays one-off JIT compilation for machinery
# every arm shares (Jᴴv in the first mode-2 call, Jv in the first CG product),
# and that cost belongs to the session, not to a forward model.
function warmup!(model, raw)
    redirect_stdout(devnull) do
        with_logger(NullLogger()) do
            instrumented_recon(model, raw; x0, LB, UB,
                               options = SolverOptions(max_iter_trf = 2))
        end
    end
    return nothing
end

banner("Warm-up: compiling the shared Jv/Jᴴv and solver machinery (untimed)")
t_warm = @elapsed for (name, model) in arms
    attempt("warmup", name) do
        warmup!(model, name == "bloch" ? raw_bloch : raw_sur)
    end
end
@printf("  done in %.1f s — compilation only, excluded from every number below.\n", t_warm)
meta["warmup_seconds"] = t_warm
release_gpu!()
mem_mark!("warm-up done")
save_results()

# ---------------------------------------------------------------------------
# Levels 2 and 3: one full reconstruction per arm
# ---------------------------------------------------------------------------
#
# Each arm reconstructs data its own forward model generated, so every arm is
# exactly self-consistent (an inverse crime per arm, deliberately). The
# TensorRT and python arms run the surrogate's weights, so they reconstruct the
# surrogate arm's data — which also makes their walls a same-problem comparison
# with it.

level2(calls, mode) = 1000 * median([c.seconds for c in calls if c.mode == mode])
count_mode(calls, mode) = count(c -> c.mode == mode, calls)
objtotal(calls) = sum(c.seconds for c in calls)
medianfit(calls) = sum(count_mode(calls, m) * level2(calls, m) / 1000 for m in (0, 2))

relerr(a, b) = 100 * median(abs.(a .- b) ./ abs.(b))
p95err(a, b) = 100 * quantile(abs.(a .- b) ./ abs.(b), 0.95)

maps = Dict{String,Any}()

for (name, model) in arms
    banner("Reconstruction: $name")
    attempt("recon", name) do
        raw = name == "bloch" ? raw_bloch : raw_sur
        out, calls, wall = instrumented_recon(model, raw; x0, LB, UB,
                                              options = SolverOptions(max_iter_trf = MAXITER))
        d = subsect("recon", name)
        d["mode0_ms"] = level2(calls, 0)
        d["mode2_ms"] = level2(calls, 2)
        d["n_mode0"] = count_mode(calls, 0)
        d["n_mode2"] = count_mode(calls, 2)
        d["objective_total_s"] = objtotal(calls)
        d["objective_medianfit_s"] = medianfit(calls)
        d["solver_internals_s"] = wall - objtotal(calls)
        d["wall_s"] = wall
        d["trf_iterations"] = size(out.x, 2) - 1
        d["cost_history"] = Float64.(vec(collect(out.f)))
        d["call_modes"] = [c.mode for c in calls]
        d["call_seconds"] = [c.seconds for c in calls]
        d["data_source"] = name == "bloch" ? "bloch" : "surrogate"

        xm = reshape(out.x[:, end], :, 4)
        T₁ = exp.(xm[:, 1]); T₂ = exp.(xm[:, 2])
        ρ = abs.(complex.(xm[:, 3], xm[:, 4]))
        maps[name] = (T₁ = T₁, T₂ = T₂, ρ = ρ)
        a = subsect("accuracy", name)
        a["T₁_median_pct"] = relerr(T₁, T₁true); a["T₁_p95_pct"] = p95err(T₁, T₁true)
        a["T₂_median_pct"] = relerr(T₂, T₂true); a["T₂_p95_pct"] = p95err(T₂, T₂true)
        a["ρ_median_pct"]  = relerr(ρ, ρtrue);   a["ρ_p95_pct"]  = p95err(ρ, ρtrue)

        @printf("  mode-0 %8.1f ms · mode-2 %8.1f ms · calls %d/%d\n",
                d["mode0_ms"], d["mode2_ms"], d["n_mode0"], d["n_mode2"])
        @printf("  objective %7.1f s (expected %.1f s) · solver internals %7.1f s · wall %7.1f s\n",
                d["objective_total_s"], d["objective_medianfit_s"],
                d["solver_internals_s"], d["wall_s"])
        @printf("  accuracy  T₁ %.3f%%/%.3f%%  T₂ %.3f%%/%.3f%%  |ρ| %.3f%%/%.3f%%  (median/p95)\n",
                a["T₁_median_pct"], a["T₁_p95_pct"], a["T₂_median_pct"], a["T₂_p95_pct"],
                a["ρ_median_pct"], a["ρ_p95_pct"])
        mem_mark!("recon $name")
        merge!(d, torch_mem())
        save_results()
        release_gpu!()
    end
end

# ---------------------------------------------------------------------------
# Arm vs arm
# ---------------------------------------------------------------------------

banner("Arm vs arm (median relative difference of the reconstructed maps)")
for (a, b) in (("surrogate", "bloch"), ("tensorrt", "surrogate"), ("python", "surrogate"))
    (haskey(maps, a) && haskey(maps, b)) || continue
    d = subsect("armvsarm", "$(a)_vs_$(b)")
    for (fld, key) in ((:T₁, "T₁_median_pct"), (:T₂, "T₂_median_pct"), (:ρ, "ρ_median_pct"))
        d[key] = relerr(getfield(maps[a], fld), getfield(maps[b], fld))
    end
    d["identical_maps"] = maps[a].T₁ == maps[b].T₁ && maps[a].T₂ == maps[b].T₂ &&
                          maps[a].ρ == maps[b].ρ
    @printf("  %-22s T₁ %.4f%%  T₂ %.4f%%  |ρ| %.4f%%%s\n", "$a vs $b",
            d["T₁_median_pct"], d["T₂_median_pct"], d["ρ_median_pct"],
            d["identical_maps"] ? "   (identical maps)" : "")
end

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

banner("Summary at N = $N ($(N^2) voxels, K = $K)")
@printf("  %-10s %10s %10s %10s %10s %10s %10s\n",
        "arm", "fwd ms", "mode0 ms", "mode2 ms", "obj s", "solver s", "wall s")
for name in ("python", "bloch", "surrogate", "tensorrt")
    haskey(sect("level1"), name) || continue
    l1 = sect("level1")[name]
    r = get(sect("recon"), name, nothing)
    if r === nothing
        @printf("  %-10s %10.1f %10s %10s %10s %10s %10s\n", name, l1["ms"], "-", "-", "-", "-", "-")
    else
        @printf("  %-10s %10.1f %10.1f %10.1f %10.1f %10.1f %10.1f\n", name, l1["ms"],
                r["mode0_ms"], r["mode2_ms"], r["objective_total_s"],
                r["solver_internals_s"], r["wall_s"])
    end
end
let rec = sect("recon")
    if haskey(rec, "bloch")
        for name in ("surrogate", "tensorrt", "python")
            haskey(rec, name) || continue
            @printf("  end-to-end vs Bloch: %-10s %6.2f×\n", name,
                    rec["bloch"]["wall_s"] / rec[name]["wall_s"])
        end
    end
    if haskey(rec, "python")
        for name in ("surrogate", "tensorrt")
            haskey(rec, name) || continue
            @printf("  end-to-end vs python: %-9s %6.2f×\n", name,
                    rec["python"]["wall_s"] / rec[name]["wall_s"])
        end
        haskey(rec, "surrogate") && @printf("  plumbing tax (python − surrogate objective total): %.1f s\n",
                rec["python"]["objective_total_s"] - rec["surrogate"]["objective_total_s"])
    end
end

meta["finished"] = string(now())
mem_mark!("process end")
save_results()
println()
println("results written to $RESFILE")
println("compare_models_N (N = $N) finished")
