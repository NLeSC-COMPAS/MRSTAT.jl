module MRSTAT

using BlochSimulators
using StaticArrays, LinearAlgebra, Statistics, StructArrays
using LinearMaps
using ComputationalResources
using ImagePhantoms
using PythonPlot
using QMRIColors

include("TrustRegionReflective/TrustRegionReflective.jl")
include("DerivativeOperations/DerivativeOperations.jl")

using .TrustRegionReflective
using .DerivativeOperations

# The pluggable forward models: the reconstruction talks to an abstract
# ForwardModel through two functions (compute_echos and
# compute_echos_and_derivatives); the Bloch simulator and the neural
# surrogate are two implementations of that interface. Included after
# DerivativeOperations because the ∂echos element type lives there.
include("ForwardModels/ForwardModels.jl")
using .ForwardModels

# The public surface: the forward-model interface and the reconstruction
# entry points. (load_surrogate is exported by ForwardModels once the
# surrogate implementation is included there.)
export ForwardModel, BlochModel, SurrogateModel, load_surrogate,
       TensorRTModel, load_tensorrt_model,
       PythonModel, load_python_model,
       neuralbloch_repo, models_dir,
       compute_echos, compute_echos_and_derivatives, nechos,
       default_bounds, check_experiment,
       mrstat_recon, example_mrstat_recon, generate_synthetic_data

include("utils/make_phantom.jl")
include("utils/objective.jl")
# include("utils/RelaxationColors.jl")
include("utils/pythonplot.jl")
include("utils/synthetic_data.jl")

const NUM_COILS = 1

function mrstat_recon(
    raw_data::AbstractArray{T},
    model::ForwardModel,
    coordinates,
    coil_sensitivities::AbstractArray{T},
    trajectory::CartesianTrajectory2D,
    transmit_field::AbstractArray;
    x0=nothing,
    LB=nothing,
    UB=nothing,
    trf_options=TrustRegionReflective.SolverOptions(),
    intermediate_plots = false
) where {T<:Complex}

    # Bounds and initial guess default to the forward model's validity
    # region (a physical simulator gets the wide physical box, the surrogate
    # its trained parameter range); explicit keyword arguments win.
    # Note: x0, LB and UB are in log space for T₁ and T₂.
    x0d, LBd, UBd = default_bounds(model)
    x0 = something(x0, x0d)
    LB = something(LB, LBd)
    UB = something(UB, UBd)

    # Fail fast on experiments the model cannot serve (readout-count
    # mismatch; for the surrogate also a B₁ map outside its trained range).
    check_experiment(model, trajectory, transmit_field)

    # Repeat the initial guess and bounds for each voxel
    num_voxels = length(coordinates)
    @assert length(transmit_field) == num_voxels
    x0 = repeat(x0', num_voxels) |> f32
    LB = repeat(LB', num_voxels) |> f32
    UB = repeat(UB', num_voxels) |> f32

    # Check coil sensitivities
    @assert all(Cᵢ -> !iszero(sum(Cᵢ)), coil_sensitivities)

    resource = CUDALibs()
    objfun = (x, mode) -> objective(x, resource, mode, raw_data, model, coordinates, coil_sensitivities, trajectory, transmit_field)

    plotfun(x, figtitle) = intermediate_plots ? plot_T₁T₂ρ(optim_to_physical_pars(x, transmit_field), isqrt(num_voxels), isqrt(num_voxels), figtitle) : nothing
    plotfun(x0, "Initial Guess")

    output = TrustRegionReflective.solver(objfun, vec(x0), vec(LB), vec(UB), trf_options, plotfun)

    return output
end

"""
    example_mrstat_recon(kind::Symbol = :bloch)

Run a complete synthetic MR-STAT experiment and reconstruction with the
chosen forward model:

- `:bloch` — the original self-contained experiment: a FISP2D sequence
  with a linear 1°–90° flip-angle ramp, simulated and reconstructed with
  the Bloch/EPG model. Needs nothing beyond this package.
- `:surrogate` — the neural-surrogate experiment: 600 readouts fixed by
  the trained checkpoint in the sibling NeuralBloch repository (requires
  the torch environment and a prior `using NeuralBloch` — the surrogate
  lives in a package extension; see the README).
- `:tensorrt` — the same surrogate through its prebuilt TensorRT engine,
  no Python in the loop (requires the shim and engine, see the README).
  The 120×120 phantom is exactly the example engine's optimal batch size.
- `:python` — the same checkpoint through the naive PythonCall baseline:
  ordinary conversions and host round trips at every crossing, no
  NeuralBloch.jl bridge needed. Deliberately the slow arm — it is the
  reference the other two learned models are measured against.

All use the same 120×120 phantom geometry, so the forward models can be
compared like for like (`scripts/compare_models_N.jl` does exactly that,
at this and other phantom sizes).
"""
function example_mrstat_recon(kind::Symbol = :bloch)
    N, K = 120, 5
    model = if kind === :bloch
        # The original example sequence: FISP2D, linear flip-angle ramp,
        # one readout per TR — valid for any nTR = K*N.
        nTR = K * N
        RF_train = range(start=1, stop=90, length=nTR) .|> complex
        sliceprofiles = ones(nTR, 1) .|> complex
        sequence = FISP2D(RF_train, sliceprofiles, 0.010, 0.006, 32, 0.025)
        BlochModel(gpu(f32(sequence)))
    elseif kind === :surrogate
        load_surrogate()
    elseif kind === :tensorrt
        load_tensorrt_model()
    elseif kind === :python
        load_python_model()
    else
        throw(ArgumentError("unknown forward model kind $kind — use :bloch, " *
                            ":surrogate, :tensorrt or :python"))
    end

    raw_data, coordinates, coil_sensitivities, trajectory, transmit_field, _ =
        generate_synthetic_data(model; N, K)
    return mrstat_recon(raw_data, model, coordinates, coil_sensitivities,
                        trajectory, transmit_field)
end

end # module MRSTAT
