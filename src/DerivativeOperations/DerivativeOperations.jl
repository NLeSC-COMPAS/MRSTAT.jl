module DerivativeOperations

    using StaticArrays
    using ComputationalResources
    # using Adapt
    using CUDA
    using LinearAlgebra
    using InteractiveUtils
    using StructArrays

    using BlochSimulators

    const NUM_COILS = 1

    include("types.jl")
    # This module no longer computes derivatives itself. Its former
    # simulate_derivatives.jl (finite-difference Bloch simulations) was
    # removed when the neural surrogate became the forward model: the
    # surrogate predicts ∂echos directly, in the same physical units the
    # finite differences produced, so the Jv/Jᴴv kernels below consume them
    # unchanged.

    include("Jv.jl")
    include("Jhv.jl")

    @inline global_id() = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    const THREADS_PER_BLOCK = 64

    export Jv, Jᴴv, ∂mˣʸ∂T₁T₂, ∂mˣʸ∂T₁T₂ρˣρʸ

end
