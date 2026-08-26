# Unit tests for the TensorRT forward model. Uses random_parameters from
# test_forwardmodels.jl and the shared SURROGATE fixture from runtests.jl
# (the engine runs the same trained weights, which gives a tight parity
# reference for free — torch is already a hard dependency of the suite).
#
# The TensorRT artifacts are optional where torch is not: the shim is built
# per machine and the engine per GPU, so when either is absent these tests
# skip with instructions instead of failing. Batch sizes come from the loaded
# engine's own optimization profile, never from constants, so the tests
# survive an engine rebuilt upstream with different MIN/OPT/MAX.

using Test
using CUDA
using Random
using LinearAlgebra
using StructArrays: StructArray
using BlochSimulators
using MRSTAT.DerivativeOperations: ∂mˣʸ∂T₁T₂

_trt_shim = joinpath(pkgdir(MRSTAT), "deps", "tensorrt", "libtrtjl.so")
_trt_engine = try
    joinpath(neuralbloch_repo(), "outputs", "model.engine")
catch err
    ""
end

if !(isfile(_trt_shim) && isfile(_trt_engine))
    @testset "TensorRTModel" begin
        @info """TensorRT tests skipped — artifacts missing:
            shim   $_trt_shim $(isfile(_trt_shim) ? "(found)" : "(build with deps/tensorrt/build_shim.sh)")
            engine $(isempty(_trt_engine) ? "<NeuralBloch repo not found>" : _trt_engine) (build in the NeuralBloch repo)"""
        @test_skip false
    end
else

const TRT = load_tensorrt_model()

_nrmse(pred, ref) = norm(vec(pred) .- vec(ref)) / norm(vec(ref))

@testset "TensorRTModel: loading" begin
    @test TRT isa TensorRTModel
    @test nechos(TRT) == TRT.NTR > 0
    # Same trained model as the surrogate → same echo-train length.
    @test nechos(TRT) == nechos(SURROGATE)
    # The batch range is engine metadata, read at load.
    @test 1 ≤ TRT.minB ≤ TRT.optB ≤ TRT.maxB
    # The engine inherits the surrogate's trained box as its search region.
    x0, LB, UB = default_bounds(TRT)
    @test UB.T₁ == log(5.0) && UB.T₂ == log(2.0)
    @test LB.T₁ == log(0.1) && LB.T₂ == log(0.01)
    # Missing files fail loudly, with a pointer at the remedy.
    @test_throws ErrorException load_tensorrt_model(engine_path = "/nonexistent/model.engine")
    @test_throws ErrorException load_tensorrt_model(shim_path = "/nonexistent/libtrtjl.so")
end

@testset "TensorRTModel: batch gate and staging" begin
    rng = Xoshiro(7)
    params = random_parameters(rng, TRT.minB)

    @test size(compute_echos(TRT, params), 2) == TRT.minB
    # The staged rows are [log T₁; log T₂; B₁], computed with the same GPU
    # broadcast — exact equality, as in the surrogate staging tests.
    @test maximum(abs, Array(TRT.Xbuf[1, :]) .- Array(log.(params.T₁))) == 0
    @test maximum(abs, Array(TRT.Xbuf[2, :]) .- Array(log.(params.T₂))) == 0
    @test maximum(abs, Array(TRT.Xbuf[3, :]) .- Array(params.B₁)) == 0

    # Same voxel count → buffers are reused, not reallocated or rebound.
    buf = TRT.Xbuf
    compute_echos(TRT, params)
    @test TRT.Xbuf === buf
    # Another in-profile voxel count → rebind with fresh buffers.
    if TRT.minB + 16 ≤ TRT.maxB
        compute_echos(TRT, random_parameters(rng, TRT.minB + 16))
        @test size(TRT.Xbuf) == (3, TRT.minB + 16)
        @test TRT.Xbuf !== buf
    end

    # The optimization profile is a hard gate with a named range.
    if TRT.minB > 1
        @test_throws ArgumentError compute_echos(TRT, random_parameters(rng, TRT.minB - 1))
    end
    @test_throws ArgumentError compute_echos(TRT, random_parameters(rng, TRT.maxB + 1))

    # Guards: CPU parameters and empty parameter sets are refused.
    cpu_params = f32(StructArray(map(T₁T₂B₁ρˣρʸ, [1.0], [0.1], [1.0], [1.0], [0.0])))
    @test_throws ArgumentError compute_echos(TRT, cpu_params)
    empty_params = StructArray{T₁T₂B₁ρˣρʸ{Float32}}(ntuple(_ -> CUDA.zeros(Float32, 0), 5))
    @test_throws ArgumentError compute_echos(TRT, empty_params)
end

@testset "TensorRTModel: predictions" begin
    rng = Xoshiro(2026)
    params = random_parameters(rng, TRT.minB)
    n = length(params)

    echos = compute_echos(TRT, params)
    @test echos isa CuMatrix{ComplexF32}
    @test size(echos) == (nechos(TRT), n)
    @test all(isfinite, Array(echos))

    # Determinism: identical inputs give identical outputs.
    echos_again = compute_echos(TRT, params)
    @test maximum(abs, Array(echos) .- Array(echos_again)) == 0
    # Results from successive calls are fresh arrays, never aliased — the
    # reconstruction mutates them in place.
    @test echos !== echos_again

    echos_full, ∂echos = compute_echos_and_derivatives(TRT, params)
    # The engine computes all outputs every time, so the two entry points
    # run the same computation — bitwise agreement.
    @test maximum(abs, Array(echos) .- Array(echos_full)) == 0

    # ∂echos satisfies the kernels' contract.
    @test ∂echos isa StructArray{∂mˣʸ∂T₁T₂{ComplexF32}}
    @test size(∂echos) == (nechos(TRT), n)
    @test ∂echos.∂T₁ isa CuMatrix{ComplexF32}
    @test ∂echos.∂T₂ isa CuMatrix{ComplexF32}
    r, v = 17, 3
    e = CUDA.@allowscalar ∂echos[r, v]
    @test e isa ∂mˣʸ∂T₁T₂{ComplexF32}
    @test e.∂T₁ == CUDA.@allowscalar ∂echos.∂T₁[r, v]
    @test e.∂T₂ == CUDA.@allowscalar ∂echos.∂T₂[r, v]

    # The CUDA-graph fast path and the plain enqueueV3 path are the same
    # computation — force the fallback on the live binding and compare.
    had_graph = TRT.graph_ok
    TRT.graph_ok = false
    echos_enq = compute_echos(TRT, params)
    TRT.graph_ok = had_graph
    @test maximum(abs, Array(echos) .- Array(echos_enq)) == 0
end

@testset "TensorRTModel: parity with the surrogate" begin
    # Same trained weights on two runtimes: differences are FP32 kernel
    # reordering only. Gate at 0.1% NRMSE — the check NeuralBloch's own
    # build_engine.py verification applies to this engine.
    rng = Xoshiro(99)
    params = random_parameters(rng, TRT.minB)
    e_trt, ∂_trt = compute_echos_and_derivatives(TRT, params)
    e_sur, ∂_sur = compute_echos_and_derivatives(SURROGATE, params)
    @test _nrmse(Array(e_trt), Array(e_sur)) < 0.001
    @test _nrmse(Array(∂_trt.∂T₁), Array(∂_sur.∂T₁)) < 0.001
    @test _nrmse(Array(∂_trt.∂T₂), Array(∂_sur.∂T₂)) < 0.001
end

@testset "TensorRTModel: check_experiment" begin
    nTR = nechos(TRT)
    FOV = 22.4
    py = repeat(-4:3, nTR ÷ 8)
    k_start = [complex(-4 * 2π / FOV, py[r] * 2π / FOV) for r in 1:nTR]
    traj_ok = CartesianTrajectory2D(nTR, 8, 5e-6, k_start, 2π / FOV, py, 1)
    traj_bad = CartesianTrajectory2D(nTR + 8, 8, 5e-6, vcat(k_start, k_start[1:8]),
                                     2π / FOV, vcat(py, py[1:8]), 1)

    tf = fill(1.0f0, TRT.minB)
    @test check_experiment(TRT, traj_ok, tf) === nothing
    # Readout-count mismatch is refused.
    @test_throws ArgumentError check_experiment(TRT, traj_bad, tf)
    # B₁ outside the trained range is refused.
    @test_throws ArgumentError check_experiment(TRT, traj_ok, fill(1.5f0, TRT.minB))
    # A voxel count outside the engine's profile is refused up front.
    if TRT.minB > 1
        @test_throws ArgumentError check_experiment(TRT, traj_ok, fill(1.0f0, TRT.minB - 1))
    end
    @test_throws ArgumentError check_experiment(TRT, traj_ok, fill(1.0f0, TRT.maxB + 1))
end

close(TRT)   # release the engine's ~0.4 GiB scratch before the recon tests

end # availability guard
