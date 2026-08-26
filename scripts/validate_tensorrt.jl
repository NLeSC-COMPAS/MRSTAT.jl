# Contract validation for the TensorRT forward model.
#
# Run with:
#   source env.sh                          # see the README (matplotlib for MRSTAT)
#   julia --project=. scripts/validate_tensorrt.jl
#
# Prerequisites: the shim (deps/tensorrt/build_shim.sh) and the engine
# (../NeuralBloch/outputs/model.engine, built there).
#
# Five parts:
#
#   1. The engine contract as the shim reports it — names, dtypes, shapes,
#      optimization profile, scratch size. A torch-free `polygraphy inspect`.
#
#   2. Accuracy against ground truth: BlochSimulators' FISP3D EPG simulation
#      + finite-difference derivatives on the training sequence, at a
#      profile-valid batch size. Same 15% NMAE gate and same rationale as
#      validate_contract.jl: a convention bug (layout, units, complex
#      pairing) misses by ~100%, true approximation error by a few percent.
#
#   3. Parity against the PyTorch surrogate — the same trained weights, so
#      the two paths must agree to FP32 kernel-reordering levels: the 0.1%
#      NRMSE gate NeuralBloch's own build_engine.py verification uses.
#      Skipped cleanly when torch is unavailable (the TensorRT model itself
#      never touches Python).
#
#   4. Speed at reconstruction size: the engine alone (CUDA-graph replay),
#      the full forward-model call with and without the graph, the PyTorch
#      surrogate for reference, and the host-side submission cost.
#
#   5. A complete self-consistent reconstruction through mrstat_recon — the
#      whole solver running against the TensorRT model, gated on recovering
#      the phantom (accuracy only; timing belongs to compare_models.jl).

using MRSTAT
using BlochSimulators
using ComputationalResources: CUDALibs
using CUDA
using StructArrays
using Statistics
using LinearAlgebra
using Printf
using Random
using Logging: NullLogger, with_logger

nmae(pred, ref) = sum(abs, pred .- ref) / sum(abs, ref)
nrmse(pred, ref) = norm(vec(pred) .- vec(ref)) / norm(vec(ref))

# ---------------------------------------------------------------------------
println("═"^70)
println("Part 1: the engine contract, as the shim reports it")
println("═"^70)

m = load_tensorrt_model()
let api = m.api, h = m.h
    nio = ccall(api.nb_io, Int32, (Ptr{Cvoid},), h)
    dims = Vector{Int64}(undef, 8)
    for i in 0:nio-1
        name = unsafe_string(ccall(api.io_name, Cstring, (Ptr{Cvoid}, Int32), h, i))
        isin = ccall(api.is_input, Int32, (Ptr{Cvoid}, Cstring), h, name)
        es = ccall(api.elt_size, Int32, (Ptr{Cvoid}, Cstring), h, name)
        nd = ccall(api.engine_shape, Int32, (Ptr{Cvoid}, Cstring, Ptr{Int64}), h, name, dims)
        @printf("  %-6s %-8s %d-byte elements, TensorRT dims %s\n",
                isin == 1 ? "input" : "output", name, es, string(Tuple(dims[1:nd])))
    end
    @printf("  profile (batch): MIN %d  OPT %d  MAX %d\n", m.minB, m.optB, m.maxB)
    @printf("  NTR %d readouts, %.1f MiB context scratch, %d aux streams\n",
            m.NTR,
            ccall(api.device_mem_size, Int64, (Ptr{Cvoid},), h) / 2^20,
            ccall(api.nb_aux_streams, Int32, (Ptr{Cvoid},), h))
end

# ---------------------------------------------------------------------------
println("═"^70)
println("Part 2: TensorRT vs BlochSimulators FISP3D ground truth")
println("═"^70)

# The training sequence, reproduced from NeuralBloch's data_generation.jl +
# config/data.yaml exactly as validate_contract.jl does for the surrogate.
neuralbloch_dir = neuralbloch_repo()
readvec(p) = [parse(Float64, l) for l in eachline(p) if !isempty(strip(l))]
flip_deg = readvec(joinpath(neuralbloch_dir, "data", "rf_data", "rf_angle_list.txt"))
phase_deg = readvec(joinpath(neuralbloch_dir, "data", "rf_data", "rf_phase_list.txt"))
rf_train = flip_deg .* exp.(1im .* deg2rad.(phase_deg))
@assert length(rf_train) == nechos(m)
sequence = gpu(f32(FISP3D(rf_train, 0.00558, 0.002745, 64, 0.01084, 0.000, 5, true, true, 1, 0.0)))

# The engine only serves profile-sized batches, so the evaluation set is
# OPT-many random points inside the trained box (T₁ ∈ [0.1,5], T₂ ∈ [0.01,2]
# log-uniform with the physical T₂ < T₁ constraint, B₁ ∈ [0.8,1.2]) — a
# denser probe of the box than a small grid, at the batch size the engine is
# tuned for.
Random.seed!(20260826)
n = m.optB
T₁ = exp.(rand(n) .* (log(5.0) - log(0.1)) .+ log(0.1))
T₂ = min.(exp.(rand(n) .* (log(2.0) - log(0.01)) .+ log(0.01)), 0.8 .* T₁)
B₁ = 0.8 .+ 0.4 .* rand(n)
parameters = map(T₁T₂B₁ρˣρʸ, T₁, T₂, B₁, ones(n), zeros(n)) |> StructArray
parameters = gpu(f32(parameters))
println("  evaluation points: $n random in-box (seeded), T₂ < T₁ enforced")

echos_ref = simulate_magnetization(CUDALibs(), sequence, parameters)
∂ref = simulate_derivatives_finite_difference((:T₁, :T₂), echos_ref, sequence, parameters)

echos_trt, ∂echos_trt = compute_echos_and_derivatives(m, parameters)
@assert size(echos_trt) == size(echos_ref) == (nechos(m), n)

# compute_echos runs the same engine on the same input — bitwise agreement
# with the full forward documents that mode 0 and mode 2 cannot diverge.
@assert Array(compute_echos(m, parameters)) == Array(echos_trt)
println("  ✓ compute_echos bitwise-matches the full forward")

e_sig = nmae(Array(echos_trt), Array(echos_ref))
e_dt1 = nmae(Array(∂echos_trt.∂T₁), Array(∂ref.T₁))
e_dt2 = nmae(Array(∂echos_trt.∂T₂), Array(∂ref.T₂))

println()
println("  NMAE vs FISP3D ground truth (convention bugs ≈ 100%):")
println("    signal : $(round(100 * e_sig, digits = 2)) %")
println("    ∂/∂T₁  : $(round(100 * e_dt1, digits = 2)) %")
println("    ∂/∂T₂  : $(round(100 * e_dt2, digits = 2)) %")

@assert e_sig < 0.15 && e_dt1 < 0.15 && e_dt2 < 0.15 """
    TensorRT model disagrees with ground truth beyond model-accuracy levels —
    this indicates a convention error (layout/units/orientation), not
    approximation error."""

# ---------------------------------------------------------------------------
println("═"^70)
println("Part 3: parity with the PyTorch surrogate (same trained weights)")
println("═"^70)

surrogate = try
    load_surrogate()
catch err
    println("  skipped — surrogate unavailable ($(sprint(showerror, err) |> s -> first(s, 120))…)")
    nothing
end

if surrogate !== nothing
    echos_sur, ∂echos_sur = compute_echos_and_derivatives(surrogate, parameters)
    p_sig = nrmse(Array(echos_trt), Array(echos_sur))
    p_dt1 = nrmse(Array(∂echos_trt.∂T₁), Array(∂echos_sur.∂T₁))
    p_dt2 = nrmse(Array(∂echos_trt.∂T₂), Array(∂echos_sur.∂T₂))
    println("  NRMSE TensorRT vs PyTorch (gate 0.1%, as in NeuralBloch's engine check):")
    println("    signal : $(round(100 * p_sig, digits = 4)) %")
    println("    ∂/∂T₁  : $(round(100 * p_dt1, digits = 4)) %")
    println("    ∂/∂T₂  : $(round(100 * p_dt2, digits = 4)) %")
    @assert p_sig < 0.001 && p_dt1 < 0.001 && p_dt2 < 0.001 """
        TensorRT and PyTorch outputs diverge beyond FP32 kernel-reordering
        levels — the engine does not compute the same function as the
        checkpoint (stale engine file?)."""
end

# ---------------------------------------------------------------------------
println("═"^70)
println("Part 4: speed at reconstruction size ($(m.optB) voxels)")
println("═"^70)

reps = 20
med(f) = median([CUDA.@elapsed f() for _ in 1:reps])

# Warm every path once, then medians.
compute_echos_and_derivatives(m, parameters); CUDA.synchronize()
@assert m.graph_ok "CUDA graph should be live on this engine"

replay(mm) = (rc = ccall(mm.api.replay, Int32, (Ptr{Cvoid}, Ptr{Cvoid}), mm.h,
                         reinterpret(Ptr{Cvoid}, CUDA.stream().handle));
              rc == 0 || error("replay failed"); nothing)
replay(m); CUDA.synchronize()
@assert (@allocated replay(m)) == 0 "graph replay must not allocate"
CUDA.synchronize()

t_engine = med(() -> replay(m))
t_full_graph = med(() -> compute_echos_and_derivatives(m, parameters))
m.graph_ok = false          # same binding, forced per-call enqueueV3
t_full_enq = med(() -> compute_echos_and_derivatives(m, parameters))
m.graph_ok = true

# Host-side submission cost: queue empty, submit one graph launch, don't wait.
submit = Float64[]
for _ in 1:50
    CUDA.synchronize()
    t0 = time_ns()
    replay(m)
    push!(submit, (time_ns() - t0) / 1e3)
end
CUDA.synchronize()

@printf("  engine alone (graph replay)          : %8.2f ms\n", 1000t_engine)
@printf("  forward model, graph                 : %8.2f ms  (stage + engine + combine ×3)\n", 1000t_full_graph)
@printf("  forward model, enqueueV3             : %8.2f ms\n", 1000t_full_enq)
if surrogate !== nothing
    compute_echos_and_derivatives(surrogate, parameters); CUDA.synchronize()
    t_sur = med(() -> compute_echos_and_derivatives(surrogate, parameters))
    @printf("  PyTorch surrogate (reference)        : %8.2f ms  → TensorRT is %.1f× faster\n",
            1000t_sur, t_sur / t_full_graph)
end
@printf("  host submit of one graph launch      : %8.1f µs (median of 50)\n", median(submit))

# ---------------------------------------------------------------------------
println("═"^70)
println("Part 5: end-to-end reconstruction (self-consistent 120×120 phantom)")
println("═"^70)

raw, coordinates, coils, trajectory, transmit_field, phantom = generate_synthetic_data(m)
# The solver narrates every iteration; none of that is a result here, so it
# goes to devnull (the same treatment compare_models.jl gives its warm-up).
output = redirect_stdout(devnull) do
    with_logger(NullLogger()) do
        mrstat_recon(raw, m, coordinates, coils, trajectory, transmit_field)
    end
end

xm = reshape(output.x[:, end], :, 4)
T₁rec = exp.(xm[:, 1]); T₂rec = exp.(xm[:, 2])
T₁true = Array(phantom.T₁); T₂true = Array(phantom.T₂)
relerr(a, b) = 100 * median(abs.(a .- b) ./ abs.(b))
p95err(a, b) = 100 * quantile(abs.(a .- b) ./ abs.(b), 0.95)
@printf("  T₁ error: median %.3f%%, p95 %.3f%%\n", relerr(T₁rec, T₁true), p95err(T₁rec, T₁true))
@printf("  T₂ error: median %.3f%%, p95 %.3f%%\n", relerr(T₂rec, T₂true), p95err(T₂rec, T₂true))
@assert relerr(T₁rec, T₁true) < 1 && relerr(T₂rec, T₂true) < 1 """
    the self-consistent reconstruction should recover the phantom to solver
    tolerance — a large error means the model broke the solver contract."""

println()
println("tensorrt validation PASSED")
