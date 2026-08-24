"""
    ForwardModels

The forward-model abstraction of the MR-STAT reconstruction.

The solver needs exactly two things from a forward model: the echo trains
for a set of tissue parameters, and the echo trains together with their
partial derivatives with respect to T₁ and T₂. Everything else in the
reconstruction — phase encoding, signal integration, the `Jv`/`Jᴴv`
Jacobian products, the trust-region solver — is forward-model-agnostic.
This module defines that two-function interface and its implementations:

- [`BlochModel`](@ref) — the physical route: EPG/Bloch simulation of a
  `BlochSimulators` pulse sequence, derivatives by finite differences.
- `SurrogateModel` (see `surrogate.jl`) — the learned route: a trained
  neural network predicts echos and derivatives directly, with zero-copy
  data exchange between Julia and PyTorch.

# The interface

Every `m::ForwardModel` implements:

    compute_echos(m, parameters)                 -> echos
    compute_echos_and_derivatives(m, parameters) -> (echos, ∂echos)
    nechos(m)                                    -> Int

with the contract the reconstruction kernels are written against:

- `echos::CuMatrix{ComplexF32}` of size `(nechos(m), nvoxels)`, indexed
  `echos[readout, voxel]` — transverse magnetization at the echo times at
  unit proton density (ρ and coil sensitivities are applied downstream);
- `∂echos` — same size, element type `∂mˣʸ∂T₁T₂{ComplexF32}` (a matrix
  whose elements carry both `∂T₁` and `∂T₂`), with the derivatives in
  **physical units** (per second). The `Jv`/`Jᴴv` kernels apply the
  log-parametrization chain rule themselves through their `lin_scale`
  factors, so a forward model must never pre-scale.

Optionally specialized:

    default_bounds(m)   -> (x0, LB, UB)   # the model's validity region
    check_experiment(m, trajectory, transmit_field)   # fail-fast validation
"""
module ForwardModels

using BlochSimulators
using CUDA
using StructArrays: StructArray
using ComputationalResources: CUDALibs
using PythonCall: PythonCall, Py, pyimport, pyconvert, pywith, pydict, pylist
using NeuralBloch: torch, to_torch, from_torch, on_task_stream
using ..DerivativeOperations: ∂mˣʸ∂T₁T₂

export ForwardModel, BlochModel, SurrogateModel, load_surrogate,
       neuralbloch_repo,
       compute_echos, compute_echos_and_derivatives, nechos,
       default_bounds, check_experiment

abstract type ForwardModel end

# Required interface — clear errors instead of MethodErrors, so a new
# implementation learns what it is missing.
compute_echos(m::ForwardModel, parameters) = error(
    "compute_echos is not implemented for $(typeof(m)); every ForwardModel " *
    "must return the (nechos, nvoxels) ComplexF32 echo matrix")
compute_echos_and_derivatives(m::ForwardModel, parameters) = error(
    "compute_echos_and_derivatives is not implemented for $(typeof(m)); every " *
    "ForwardModel must return (echos, ∂echos) with physical-unit T₁/T₂ derivatives")
nechos(m::ForwardModel) = error(
    "nechos is not implemented for $(typeof(m)); it must return the number of " *
    "echos (readouts) the model produces per voxel")

"""
    default_bounds(m::ForwardModel) -> (x0, LB, UB)

The default initial guess and box constraints (in the solver's
`[log T₁, log T₂, ρˣ, ρʸ]` parametrization) for reconstructions with this
model. The generic fallback is the wide physical box a simulator can
handle; models with a limited validity region — a surrogate trained on a
finite parameter range — override this so the solver never asks them to
extrapolate. Explicit `x0`/`LB`/`UB` keyword arguments to `mrstat_recon`
take precedence.
"""
default_bounds(::ForwardModel) = (
    T₁T₂ρˣρʸ(log(1.0), log(0.100), 1.0, 0.0),    # x0
    T₁T₂ρˣρʸ(log(0.1), log(0.001), -Inf, -Inf),  # LB
    T₁T₂ρˣρʸ(log(7.0), log(3.000), Inf, Inf),    # UB
)

"""
    check_experiment(m::ForwardModel, trajectory, transmit_field)

Fail-fast validation that an experiment is one this model can serve.
Generically that is one requirement: the trajectory must expect exactly as
many readouts as the model produces echos. Models add their own checks
(the surrogate also validates the B₁ map against its trained range).
"""
function check_experiment(m::ForwardModel, trajectory, transmit_field)
    nreadouts(trajectory) == nechos(m) || throw(ArgumentError(
        "trajectory has $(nreadouts(trajectory)) readouts but the forward " *
        "model produces $(nechos(m)) echos per voxel"))
    return nothing
end

# =============================================================================
# BlochModel — the physical forward model
# =============================================================================

"""
    BlochModel(sequence[, resource]) <: ForwardModel

The Bloch-simulation forward model: echo trains come from
`BlochSimulators.simulate_magnetization` of the wrapped pulse `sequence`
(EPG for the FISP family), and the T₁/T₂ derivatives from BlochSimulators'
finite differences. `sequence` should already be `gpu(f32(...))`-converted;
`resource` defaults to `CUDALibs()`.

The finite-difference step sizes are BlochSimulators' defaults
(ΔT₁ = 0.02 s, ΔT₂ = 0.0025 s). They are chosen large enough to be robust
in Float32 — a tiny step loses the difference to rounding — and they are
the exact convention the neural surrogate was trained against, so both
forward models produce derivatives with the same meaning.
"""
struct BlochModel{S,R} <: ForwardModel
    sequence::S
    resource::R
end
BlochModel(sequence) = BlochModel(sequence, CUDALibs())

Base.show(io::IO, m::BlochModel) =
    print(io, "BlochModel($(nameof(typeof(m.sequence))), $(nechos(m)) echos)")

nechos(m::BlochModel) = Int(BlochSimulators.output_size(m.sequence))

compute_echos(m::BlochModel, parameters) =
    simulate_magnetization(m.resource, m.sequence, parameters)

function compute_echos_and_derivatives(m::BlochModel, parameters)
    echos = compute_echos(m, parameters)
    # BlochSimulators' finite differences return one matrix per requested
    # property; the StructArray view over the T₁ and T₂ matrices is the
    # single ∂echos[readout, voxel] matrix the kernels index — no copy.
    ∂ = simulate_derivatives_finite_difference((:T₁, :T₂), echos, m.sequence, parameters)
    ∂echos = StructArray{∂mˣʸ∂T₁T₂{eltype(echos)}}((∂.T₁, ∂.T₂))
    return echos, ∂echos
end

# The learned forward model (SurrogateModel + load_surrogate).
include("surrogate.jl")

end # module ForwardModels
