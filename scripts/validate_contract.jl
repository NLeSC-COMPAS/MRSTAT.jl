# Contract validation for the neural surrogate forward model.
#
# Run with (matplotlib-free, terminal only):
#   source ../NeuralBloch.jl/env.sh        # binds PythonCall to the torch venv
#   julia --project=. scripts/validate_contract.jl
#
# Two parts:
#
#   1. Mechanical checks of the zero-copy bridge recipe: pointer identity
#      through both boundary crossings, the view_as_complex layout, and
#      StructArray ∂echos field access inside a CUDA kernel (the one new
#      mechanism the reconstruction kernels rely on).
#
#   2. Accuracy of the surrogate against ground truth: BlochSimulators'
#      FISP3D EPG simulation + finite-difference derivatives on the exact
#      sequence the model was trained on. A *convention* bug (orientation,
#      units, complex layout, log-space mixup) produces errors of order
#      100%; the model's approximation error is a few percent — the 15%
#      gate below cleanly separates the two. Reported numbers are the
#      model's true accuracy.

using MRSTAT
using MRSTAT.ForwardModels
using BlochSimulators
using ComputationalResources: CUDALibs
using CUDA
using StructArrays
using NeuralBloch: torch, to_torch, from_torch
using Statistics
using LinearAlgebra

nmae(pred, ref) = sum(abs, pred .- ref) / sum(abs, ref)

# ---------------------------------------------------------------------------
println("═"^70)
println("Part 1: zero-copy mechanics")
println("═"^70)

# (a) Pointer identity Julia → torch → Julia, real dtype.
A = CUDA.rand(Float32, 4, 6)
tA = to_torch(A; layout = :reversed)
JA = from_torch(tA)
@assert UInt(pointer(JA)) == UInt(pointer(A)) "boundary round trip copied (real path)"
@assert size(JA) == size(A)

# (b) The complex output recipe: (N, 2·NTR) real row-major, viewed as
# complex pairs, must come back as the (NTR, N) ComplexF32 matrix that
# aliases the same bytes — i.e. exactly reinterpret(ComplexF32, ·) of the
# Julia-side original.
tm = torch()
raw = CUDA.rand(Float32, 8, 5)                     # Julia (2·NTR, N) = (8, 5)
traw = to_torch(raw; layout = :reversed)           # torch (5, 8) contiguous
tc = tm.view_as_complex(traw.view(5, 4, 2))        # torch (5, 4) complex64
Jc = from_torch(tc)                                # Julia (4, 5) ComplexF32
@assert UInt(pointer(Jc)) == UInt(pointer(raw)) "complex view path copied"
@assert Array(Jc) == reinterpret(ComplexF32, Array(raw)) "complex pairing wrong"
println("  ✓ pointer identity through both crossings (real and complex paths)")

# (c) StructArray ∂echos inside a CUDA kernel: getindex, field access, and
# FieldVector arithmetic — the access pattern of the Jv/Jᴴv kernels.
∂T₁ = CUDA.rand(ComplexF32, 4, 5)
∂T₂ = CUDA.rand(ComplexF32, 4, 5)
S = StructArray{MRSTAT.DerivativeOperations.∂mˣʸ∂T₁T₂{ComplexF32}}((∂T₁, ∂T₂))
a = CUDA.zeros(ComplexF32, 4, 5)
b = CUDA.zeros(ComplexF32, 4, 5)
function probe_kernel!(a, b, S)
    v = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if v <= size(S, 2)
        @inbounds for r in 1:size(S, 1)
            e = S[r, v]
            a[r, v] = 2f0 * e.∂T₁
            b[r, v] = e.∂T₂ + e.∂T₁
        end
    end
    return nothing
end
CUDA.@sync @cuda threads = 8 probe_kernel!(a, b, S)
@assert Array(a) == 2f0 .* Array(∂T₁) "StructArray field ∂T₁ misread in kernel"
@assert Array(b) == Array(∂T₂) .+ Array(∂T₁) "StructArray arithmetic wrong in kernel"
println("  ✓ StructArray ∂echos indexes correctly inside a @cuda kernel")

# ---------------------------------------------------------------------------
println("═"^70)
println("Part 2: surrogate vs BlochSimulators FISP3D ground truth")
println("═"^70)

surrogate = load_surrogate()
neuralbloch_dir = neuralbloch_repo()

# The training sequence, reproduced from NeuralBloch's data_generation.jl +
# config/data.yaml (frozen for this checkpoint): FISP3D with the repo's RF
# train, TR 5.58 ms, TE 2.745 ms, max_state 64, TI 10.84 ms, TW 0, five
# repetitions (sampling the last), inversion prepulse, wait spoiling, no py
# undersampling, no spoiling-gradient diffusion.
readvec(p) = [parse(Float64, l) for l in eachline(p) if !isempty(strip(l))]
flip_deg = readvec(joinpath(neuralbloch_dir, "data", "rf_data", "rf_angle_list.txt"))
phase_deg = readvec(joinpath(neuralbloch_dir, "data", "rf_data", "rf_phase_list.txt"))
rf_train = flip_deg .* exp.(1im .* deg2rad.(phase_deg))
@assert length(rf_train) == nechos(surrogate)
sequence = FISP3D(rf_train, 0.00558, 0.002745, 64, 0.01084, 0.000, 5, true, true, 1, 0.0)
sequence = gpu(f32(sequence))

# Evaluation grid: interior of the trained box (T₁ ∈ [0.1,5], T₂ ∈ [0.01,2],
# B₁ ∈ [0.8,1.2]), with the physical T₂ < T₁ constraint the training data
# also enforced.
T₁g = collect(logrange(0.15, 4.5, 6))
T₂g = collect(logrange(0.015, 1.5, 6))
B₁g = collect(range(0.82, 1.18, 6))
tuples = [(t1, t2, b1) for t1 in T₁g for t2 in T₂g for b1 in B₁g if t2 < t1]
T₁ = [t[1] for t in tuples]; T₂ = [t[2] for t in tuples]; B₁ = [t[3] for t in tuples]
n = length(T₁)
println("  evaluation points: $n (grid, T₂ < T₁ enforced)")

# One parameter set feeds both forward models (the recon's own idiom).
parameters = map(T₁T₂B₁ρˣρʸ, T₁, T₂, B₁, ones(n), zeros(n)) |> StructArray
parameters = gpu(f32(parameters))

# Ground truth: EPG simulation + finite differences (the exact machinery
# that generated the training targets; physical-unit derivatives).
echos_ref = simulate_magnetization(CUDALibs(), sequence, parameters)
∂ref = simulate_derivatives_finite_difference((:T₁, :T₂), echos_ref, sequence, parameters)

# Surrogate predictions.
echos_sur, ∂echos_sur = compute_echos_and_derivatives(surrogate, parameters)
@assert size(echos_sur) == size(echos_ref) == (nechos(surrogate), n)

# The mode-0 fast path must agree exactly with the full forward (same
# sub-network, same weights, same input — deterministic kernels).
echos_fast = compute_echos(surrogate, parameters)
@assert Array(echos_fast) == Array(echos_sur) "mode-0 fast path diverges from full forward"
println("  ✓ mode-0 fast path bitwise-matches the full forward")

# ∂echos addressing: the StructArray element view and the component
# matrices must be two views of the same numbers.
let r = 17, v = 3
    e = CUDA.@allowscalar ∂echos_sur[r, v]
    c1 = CUDA.@allowscalar ∂echos_sur.∂T₁[r, v]
    c2 = CUDA.@allowscalar ∂echos_sur.∂T₂[r, v]
    @assert e.∂T₁ == c1 && e.∂T₂ == c2 "∂echos element/component views disagree"
end

e_sig = nmae(Array(echos_sur), Array(echos_ref))
e_dt1 = nmae(Array(∂echos_sur.∂T₁), Array(∂ref.T₁))
e_dt2 = nmae(Array(∂echos_sur.∂T₂), Array(∂ref.T₂))

println()
println("  NMAE vs FISP3D ground truth (lower is better; convention bugs ≈ 100%):")
println("    signal : $(round(100 * e_sig, digits = 2)) %")
println("    ∂/∂T₁  : $(round(100 * e_dt1, digits = 2)) %")
println("    ∂/∂T₂  : $(round(100 * e_dt2, digits = 2)) %")

@assert e_sig < 0.15 && e_dt1 < 0.15 && e_dt2 < 0.15 """
    surrogate disagrees with ground truth beyond model-accuracy levels —
    this indicates a convention error (layout/units/orientation), not
    approximation error."""

# ---------------------------------------------------------------------------
println("═"^70)
println("Part 3: forward-model timing at reconstruction size")
println("═"^70)

Nrec = 14_400   # 120×120, the example reconstruction
T₁r = exp.(rand(Nrec) .* (log(2.5) - log(0.3)) .+ log(0.3))
T₂r = exp.(rand(Nrec) .* (log(0.2) - log(0.03)) .+ log(0.03))
B₁r = 0.9 .+ 0.2 .* rand(Nrec)
params_rec = map(T₁T₂B₁ρˣρʸ, T₁r, T₂r, B₁r, ones(Nrec), zeros(Nrec)) |> StructArray
params_rec = gpu(f32(params_rec))

# warm up both paths, then time echos + derivatives (what objective mode 2 needs)
compute_echos_and_derivatives(surrogate, params_rec); CUDA.synchronize()
t_sur = CUDA.@elapsed begin
    compute_echos_and_derivatives(surrogate, params_rec)
end

er = simulate_magnetization(CUDALibs(), sequence, params_rec); CUDA.synchronize()
t_epg = CUDA.@elapsed begin
    er2 = simulate_magnetization(CUDALibs(), sequence, params_rec)
    simulate_derivatives_finite_difference((:T₁, :T₂), er2, sequence, params_rec)
end

println("  echos + ∂echos at $Nrec voxels:")
println("    surrogate        : $(round(1000 * t_sur, digits = 1)) ms")
println("    EPG + finite diff: $(round(1000 * t_epg, digits = 1)) ms")
println("    speedup          : $(round(t_epg / t_sur, digits = 1))×")

println()
println("contract validation PASSED")
