# Unit tests for the naive Python baseline forward model. Uses
# random_parameters from test_forwardmodels.jl and the shared SURROGATE
# fixture from runtests.jl — the baseline runs the same checkpoint through
# the same torch kernels, only with ordinary conversions instead of the
# zero-copy bridge, so the surrogate is an exact (bitwise) reference.
#
# The baseline needs nothing the surrogate does not already need (torch,
# models/model.pt, the sibling model code), so unlike the TensorRT tests
# these are unconditional.

using Test
using CUDA
using Random
using LinearAlgebra
using StructArrays: StructArray
using BlochSimulators
using PythonCall: pyconvert
using MRSTAT.ForwardModels: _py_host_input, _py_device_input, _py_forward,
                            _py_forward_signal, _py_to_host, _py_to_julia,
                            _py_pair_to_complex, _py_upload
using MRSTAT.DerivativeOperations: ∂mˣʸ∂T₁T₂

const PYMODEL = load_python_model()

@testset "PythonModel: loading" begin
    @test PYMODEL isa PythonModel
    @test nechos(PYMODEL) == 600
    # Same checkpoint as the surrogate → same echo-train length.
    @test nechos(PYMODEL) == nechos(SURROGATE)
    # eval mode and CUDA placement, read off the live torch module.
    @test pyconvert(Bool, PYMODEL.model.training) == false
    @test pyconvert(Bool, PYMODEL.model.U_signal[0].weight.is_cuda)
    @test pyconvert(Int, PYMODEL.model.U_signal[0].weight.shape[0]) == 2 * nechos(PYMODEL)
    # The trained box is the default search region, as for the other two
    # models running these weights.
    x0, LB, UB = default_bounds(PYMODEL)
    @test UB.T₁ == log(5.0) && UB.T₂ == log(2.0)
    @test LB.T₁ == log(0.1) && LB.T₂ == log(0.01)
    # Missing files fail loudly before any Python work.
    @test_throws ErrorException load_python_model(model_path = "/nonexistent/model.pt")
    @test_throws ErrorException load_python_model(neuralbloch_dir = "/nonexistent")
end

@testset "PythonModel: host staging and conversions" begin
    rng = Xoshiro(4242)
    n = 19
    params = random_parameters(rng, n)

    # Stage 1 — the (N, 3) host input: physical [T₁, T₂, B₁] in columns, in
    # Float32, copied down from the GPU. No log, no scaling, no promotion:
    # exact equality holds.
    X = _py_host_input(PYMODEL, params)
    @test X isa Matrix{Float32}
    @test size(X) == (n, 3)
    @test X[:, 1] == Array(params.T₁)
    @test X[:, 2] == Array(params.T₂)
    @test X[:, 3] == Array(params.B₁)
    # A fresh host matrix every call — the baseline keeps no staging buffer.
    @test _py_host_input(PYMODEL, params) !== X

    # Stage 2 — the same numbers as a float32 CUDA tensor. numpy's copy owns
    # its memory, so torch never holds a pointer into Julia's heap.
    x = _py_device_input(PYMODEL, X)
    @test pyconvert(Bool, x.dtype == PYMODEL.torch.float32)
    @test pyconvert(Bool, x.is_cuda)
    @test pyconvert(Int, x.shape[0]) == n && pyconvert(Int, x.shape[1]) == 3
    @test pyconvert(Array{Float32,2}, _py_to_host(x)) == X

    # Stages 5–6 — (N, NTR, 2) paired output → (NTR, N) ComplexF32, on a
    # hand-built array whose every element is distinguishable.
    ntr = 7
    A = reshape(Float32.(1:(n * ntr * 2)), n, ntr, 2)
    H = _py_pair_to_complex(A)
    @test H isa Matrix{ComplexF32}
    @test size(H) == (ntr, n)
    @test all(H[r, v] == complex(A[v, r, 1], A[v, r, 2]) for r in 1:ntr, v in 1:n)

    # Stage 7 — the upload the reconstruction receives.
    @test _py_upload(H) isa CuMatrix{ComplexF32}
    @test Array(_py_upload(H)) == H

    # A full crossing round trip is numerically transparent: what the network
    # emitted is what the reconstruction gets, element for element.
    out = _py_forward_signal(PYMODEL, x)
    B = _py_to_julia(_py_to_host(out))
    @test B isa Array{Float32,3}
    @test size(B) == (n, nechos(PYMODEL), 2)
    echos = compute_echos(PYMODEL, params)
    @test Array(echos) == _py_pair_to_complex(B)

    # Guards: CPU parameters and empty parameter sets are refused.
    cpu_params = f32(StructArray(map(T₁T₂B₁ρˣρʸ, [1.0], [0.1], [1.0], [1.0], [0.0])))
    @test_throws ArgumentError _py_host_input(PYMODEL, cpu_params)
    @test_throws ArgumentError compute_echos(PYMODEL, cpu_params)
    empty_params = StructArray{T₁T₂B₁ρˣρʸ{Float32}}(ntuple(_ -> CUDA.zeros(Float32, 0), 5))
    @test_throws ArgumentError compute_echos(PYMODEL, empty_params)
end

@testset "PythonModel: predictions" begin
    rng = Xoshiro(2027)
    params = random_parameters(rng, 23)
    n = length(params)

    echos = compute_echos(PYMODEL, params)
    @test echos isa CuMatrix{ComplexF32}
    @test size(echos) == (nechos(PYMODEL), n)
    @test all(isfinite, Array(echos))

    # Determinism: identical inputs give identical outputs.
    echos_again = compute_echos(PYMODEL, params)
    @test maximum(abs, Array(echos) .- Array(echos_again)) == 0
    # Results are fresh arrays the caller owns — the reconstruction mutates
    # them in place (phase encoding).
    @test echos !== echos_again

    echos_full, ∂echos = compute_echos_and_derivatives(PYMODEL, params)
    @test echos_full isa CuMatrix{ComplexF32}
    @test size(echos_full) == (nechos(PYMODEL), n)
    # The mode-0 fast path runs the same signal sub-network on the same
    # input — bitwise agreement with the full forward.
    @test maximum(abs, Array(echos) .- Array(echos_full)) == 0

    # ∂echos satisfies the kernels' contract.
    @test ∂echos isa StructArray{∂mˣʸ∂T₁T₂{ComplexF32}}
    @test size(∂echos) == (nechos(PYMODEL), n)
    @test ∂echos.∂T₁ isa CuMatrix{ComplexF32}
    @test ∂echos.∂T₂ isa CuMatrix{ComplexF32}
    @test all(isfinite, Array(∂echos.∂T₁)) && all(isfinite, Array(∂echos.∂T₂))
    r, v = 17, 3
    e = CUDA.@allowscalar ∂echos[r, v]
    @test e isa ∂mˣʸ∂T₁T₂{ComplexF32}
    @test e.∂T₁ == CUDA.@allowscalar ∂echos.∂T₁[r, v]
    @test e.∂T₂ == CUDA.@allowscalar ∂echos.∂T₂[r, v]
end

@testset "PythonModel: parity with the surrogate" begin
    # Same weights, same torch kernels, same Float32 inputs — the only
    # difference is how the numbers travel. Anything but bitwise equality
    # means a crossing is not transparent, so the gate is exact.
    rng = Xoshiro(101)
    params = random_parameters(rng, 61)
    e_py, ∂_py = compute_echos_and_derivatives(PYMODEL, params)
    e_su, ∂_su = compute_echos_and_derivatives(SURROGATE, params)
    @test maximum(abs, Array(e_py) .- Array(e_su)) == 0
    @test maximum(abs, Array(∂_py.∂T₁) .- Array(∂_su.∂T₁)) == 0
    @test maximum(abs, Array(∂_py.∂T₂) .- Array(∂_su.∂T₂)) == 0
    @test norm(vec(Array(e_py)) .- vec(Array(e_su))) == 0   # NRMSE, numerator
    # The cheap paths agree too.
    @test maximum(abs, Array(compute_echos(PYMODEL, params)) .-
                       Array(compute_echos(SURROGATE, params))) == 0
end

@testset "PythonModel: check_experiment" begin
    nTR = nechos(PYMODEL)
    FOV = 22.4
    py = repeat(-4:3, nTR ÷ 8)
    k_start = [complex(-4 * 2π / FOV, py[r] * 2π / FOV) for r in 1:nTR]
    traj_ok = CartesianTrajectory2D(nTR, 8, 5e-6, k_start, 2π / FOV, py, 1)
    traj_bad = CartesianTrajectory2D(nTR + 8, 8, 5e-6, vcat(k_start, k_start[1:8]),
                                     2π / FOV, vcat(py, py[1:8]), 1)

    tf = fill(1.0f0, 64)
    @test check_experiment(PYMODEL, traj_ok, tf) === nothing
    # Readout-count mismatch is refused.
    @test_throws ArgumentError check_experiment(PYMODEL, traj_bad, tf)
    # B₁ outside the trained range is refused.
    @test_throws ArgumentError check_experiment(PYMODEL, traj_ok, fill(1.5f0, 64))
    # Unlike the engine, there is no batch-size gate: any voxel count works.
    @test check_experiment(PYMODEL, traj_ok, fill(1.0f0, 3)) === nothing
end
