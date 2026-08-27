# Unit tests for the ForwardModel interface and its two implementations.
# Uses the shared SURROGATE fixture loaded in runtests.jl.

using Test
using CUDA
using Random
using StructArrays: StructArray
using BlochSimulators
using ComputationalResources: CUDALibs
using PythonCall: pyconvert
using MRSTAT.ForwardModels: _stage_input!
using MRSTAT.DerivativeOperations: ∂mˣʸ∂T₁T₂

# Random tissue parameters inside the surrogate's trained box (T₂ < T₁
# enforced, as in the training data), on the GPU via the reconstruction's
# own conversion idiom. Deterministic through the explicit RNG.
function random_parameters(rng::AbstractRNG, n::Int)
    T₁ = exp.(rand(rng, n) .* (log(4.5) - log(0.15)) .+ log(0.15))
    T₂ = min.(exp.(rand(rng, n) .* (log(1.5) - log(0.015)) .+ log(0.015)), 0.9 .* T₁)
    B₁ = 0.82 .+ 0.36 .* rand(rng, n)
    ρˣ = 0.5 .+ rand(rng, n)
    ρʸ = rand(rng, n) .- 0.5
    return gpu(f32(StructArray(map(T₁T₂B₁ρˣρʸ, T₁, T₂, B₁, ρˣ, ρʸ))))
end

# A small Bloch model usable for any test geometry: the original example's
# FISP2D flip-angle ramp, one echo per TR.
function small_bloch_model(nTR::Int)
    RF = complex.(collect(range(1, 90; length = nTR)))
    sequence = FISP2D(RF, ones(nTR, 1) .|> complex, 0.010, 0.006, 32, 0.025)
    return BlochModel(gpu(f32(sequence)))
end

@testset "interface generics" begin
    # An implementation that forgets a method gets a message that names it.
    struct _IncompleteModel <: ForwardModel end
    m = _IncompleteModel()
    @test_throws ErrorException compute_echos(m, nothing)
    @test_throws ErrorException compute_echos_and_derivatives(m, nothing)
    @test_throws ErrorException nechos(m)
    # The generic validity region is the wide physical box.
    x0, LB, UB = default_bounds(m)
    @test UB.T₁ == log(7.0) && UB.T₂ == log(3.0)
end

@testset "BlochModel" begin
    rng = Xoshiro(31)
    nTR = 64
    m = small_bloch_model(nTR)
    @test m isa BlochModel
    @test nechos(m) == nTR

    params = random_parameters(rng, 12)

    # compute_echos is exactly simulate_magnetization of the wrapped
    # sequence — same call, same numbers, bit for bit.
    echos = compute_echos(m, params)
    @test echos isa CuMatrix{ComplexF32}
    @test size(echos) == (nTR, 12)
    ref = simulate_magnetization(CUDALibs(), m.sequence, params)
    @test maximum(abs, Array(echos) .- Array(ref)) == 0

    # compute_echos_and_derivatives wraps BlochSimulators' finite
    # differences (physical units) as the kernels' StructArray contract.
    echos2, ∂echos = compute_echos_and_derivatives(m, params)
    @test maximum(abs, Array(echos2) .- Array(ref)) == 0
    @test ∂echos isa StructArray{∂mˣʸ∂T₁T₂{ComplexF32}}
    @test size(∂echos) == (nTR, 12)
    ∂ref = simulate_derivatives_finite_difference((:T₁, :T₂), ref, m.sequence, params)
    @test maximum(abs, Array(∂echos.∂T₁) .- Array(∂ref.T₁)) == 0
    @test maximum(abs, Array(∂echos.∂T₂) .- Array(∂ref.T₂)) == 0

    # Element and component views of ∂echos are the same numbers.
    r, v = 17, 3
    e = CUDA.@allowscalar ∂echos[r, v]
    @test e isa ∂mˣʸ∂T₁T₂{ComplexF32}
    @test e.∂T₁ == CUDA.@allowscalar ∂echos.∂T₁[r, v]
    @test e.∂T₂ == CUDA.@allowscalar ∂echos.∂T₂[r, v]
end

@testset "SurrogateModel: loading" begin
    @test SURROGATE isa SurrogateModel
    @test nechos(SURROGATE) == 600
    # eval mode and CUDA placement, read off the live torch module (the
    # decoder is Sequential(Linear, Unflatten); its Linear sits at [0])
    @test pyconvert(Bool, SURROGATE.model.training) == false
    @test pyconvert(Bool, SURROGATE.model.U_signal[0].weight.is_cuda)
    # the decoder's Linear emits 2·NTR values per voxel, unflattened to
    # (NTR, 2) pairs by the layer behind it
    @test pyconvert(Int, SURROGATE.model.U_signal[0].weight.shape[0]) == 2 * nechos(SURROGATE)
    # the trained box is the default search region
    x0, LB, UB = default_bounds(SURROGATE)
    @test UB.T₁ == log(5.0) && UB.T₂ == log(2.0)
    @test LB.T₁ == log(0.1) && LB.T₂ == log(0.01)
    # missing files fail loudly before any Python work
    @test_throws ErrorException load_surrogate(model_path = "/nonexistent/model.pt")
    @test_throws ErrorException load_surrogate(neuralbloch_dir = "/nonexistent")
end

@testset "SurrogateModel: input staging" begin
    rng = Xoshiro(2024)
    params = random_parameters(rng, 17)

    n = _stage_input!(SURROGATE, params)
    @test n == 17
    @test size(SURROGATE.Xbuf) == (3, 17)
    # The staged rows must be the physical [T₁; T₂; B₁] — the model applies
    # the log to T₁/T₂ internally — so staging is a plain copy and exact
    # equality holds.
    @test Array(SURROGATE.Xbuf[1, :]) == Array(params.T₁)
    @test Array(SURROGATE.Xbuf[2, :]) == Array(params.T₂)
    @test Array(SURROGATE.Xbuf[3, :]) == Array(params.B₁)

    # Same voxel count → the buffer is reused, not reallocated.
    buf = SURROGATE.Xbuf
    _stage_input!(SURROGATE, params)
    @test SURROGATE.Xbuf === buf
    # Different voxel count → resized.
    _stage_input!(SURROGATE, random_parameters(rng, 5))
    @test size(SURROGATE.Xbuf) == (3, 5)
    @test SURROGATE.Xbuf !== buf

    # Guards: CPU parameters and empty parameter sets are refused.
    cpu_params = f32(StructArray(map(T₁T₂B₁ρˣρʸ, [1.0], [0.1], [1.0], [1.0], [0.0])))
    @test_throws ArgumentError _stage_input!(SURROGATE, cpu_params)
    empty_params = StructArray{T₁T₂B₁ρˣρʸ{Float32}}(ntuple(_ -> CUDA.zeros(Float32, 0), 5))
    @test_throws ArgumentError _stage_input!(SURROGATE, empty_params)
end

@testset "SurrogateModel: predictions" begin
    rng = Xoshiro(2025)
    params = random_parameters(rng, 23)
    n = length(params)

    echos = compute_echos(SURROGATE, params)
    @test echos isa CuMatrix{ComplexF32}
    @test size(echos) == (nechos(SURROGATE), n)

    # Determinism: identical inputs give identical outputs.
    echos_again = compute_echos(SURROGATE, params)
    @test maximum(abs, Array(echos) .- Array(echos_again)) == 0

    echos_full, ∂echos = compute_echos_and_derivatives(SURROGATE, params)
    @test echos_full isa CuMatrix{ComplexF32}
    @test size(echos_full) == (nechos(SURROGATE), n)
    # The mode-0 fast path runs the same signal sub-network on the same
    # input — its output must match the full forward bitwise.
    @test maximum(abs, Array(echos) .- Array(echos_full)) == 0

    # ∂echos satisfies the kernels' contract: a matrix of ∂mˣʸ∂T₁T₂ elements
    # whose element view and component matrices are the same numbers.
    @test ∂echos isa StructArray{∂mˣʸ∂T₁T₂{ComplexF32}}
    @test size(∂echos) == (nechos(SURROGATE), n)
    @test ∂echos.∂T₁ isa CuMatrix{ComplexF32}
    @test ∂echos.∂T₂ isa CuMatrix{ComplexF32}
    r, v = 17, 3
    e = CUDA.@allowscalar ∂echos[r, v]
    @test e isa ∂mˣʸ∂T₁T₂{ComplexF32}
    @test e.∂T₁ == CUDA.@allowscalar ∂echos.∂T₁[r, v]
    @test e.∂T₂ == CUDA.@allowscalar ∂echos.∂T₂[r, v]
end

@testset "check_experiment" begin
    # Geometry mismatch is refused for any model; the surrogate also
    # rejects a B₁ map outside its trained range.
    nTR = 64
    m = small_bloch_model(nTR)
    FOV = 22.4
    py = repeat(-4:3, 8)
    k_start = [complex(-4 * 2π / FOV, py[r] * 2π / FOV) for r in 1:nTR]
    traj_ok = CartesianTrajectory2D(nTR, 8, 5e-6, k_start, 2π / FOV, py, 1)
    traj_bad = CartesianTrajectory2D(nTR + 8, 8, 5e-6, vcat(k_start, k_start[1:8]),
                                     2π / FOV, vcat(py, py[1:8]), 1)
    tf = fill(1.0f0, 64)
    @test check_experiment(m, traj_ok, tf) === nothing
    @test_throws ArgumentError check_experiment(m, traj_bad, tf)
    # A wild B₁ is fine for the physical model but not for the surrogate.
    tf_wild = fill(1.5f0, 64)
    @test check_experiment(m, traj_ok, tf_wild) === nothing
    @test_throws ArgumentError check_experiment(SURROGATE, traj_ok, tf_wild)
end
