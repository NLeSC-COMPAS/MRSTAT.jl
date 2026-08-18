# MR-STAT Reconstruction Code

This repository contains an implementation of the partially-matrix free, inexact Gauss-Newton algorithm that can be used for MR-STAT. It uses a 2D numerical brain phantom to generate synthetic data from which it then estimates $T_1$, $T_2$ and proton density. It only supports Cartesian readout trajectories at the moment.

**The forward model is pluggable.** The reconstruction talks to an abstract `ForwardModel` through two functions — `compute_echos` (echo trains) and `compute_echos_and_derivatives` (echo trains plus physical-unit ∂/∂T₁, ∂/∂T₂) — and two implementations are provided:

- **`BlochModel`** — the physical route: EPG/Bloch simulation of any [BlochSimulators.jl](../BlochSimulators.jl) pulse sequence, with derivatives by BlochSimulators' finite differences. Self-contained; works for any sequence and geometry.
- **`SurrogateModel`** — the learned route: the trained PyTorch surrogate from the sibling [NeuralBloch](../NeuralBloch) repository predicts echos and derivatives directly in one network forward. Julia arrays and torch tensors cross the language boundary **zero-copy** in both directions through [NeuralBloch.jl](../NeuralBloch.jl), and all GPU work runs on a single CUDA stream (no synchronization fences). The pulse sequence is baked into the checkpoint (FISP3D, 600 readouts, TR 5.58 ms, TE 2.745 ms, TI 10.84 ms; trained for T₁ ∈ [0.1, 5] s, T₂ ∈ [0.01, 2] s, B₁ ∈ [0.8, 1.2]), so experiments need `nreadouts == 600` and the solver's default bounds pin the search to the trained box — both enforced by `check_experiment`/`default_bounds`.

Everything downstream of the forward model — phase encoding, signal integration, the partially matrix-free `Jv`/`Jᴴv` Jacobian products, the trust-region solver — is shared. At the example's size (120×120 voxels, 600 readouts) the surrogate produces echos + derivatives ~50× faster than the EPG simulation + finite differences (3.2 ms vs 158 ms); end-to-end reconstruction time is dominated by the shared Gauss-Newton products, so the wall-clock gain is Amdahl-limited (see `scripts/compare_models.jl` for the full three-level breakdown).

## Requirements

- An NVIDIA GPU (the codebase assumes CUDA throughout).
- For the surrogate: a Python environment with a CUDA-enabled `torch` for PythonCall to bind to, plus the sibling `NeuralBloch` repository (model code + trained checkpoint `outputs/model.pt`). With the sibling `NeuralBloch.jl` checkout: `source ../NeuralBloch.jl/env.sh` **before starting Julia**. torch is imported lazily — only `load_surrogate()` touches it.
- The local `../BlochSimulators.jl` and `../NeuralBloch.jl` checkouts (tracked as `Pkg.develop` path dependencies in the Manifest).

## Usage

```julia
julia> using MRSTAT
julia> output = example_mrstat_recon(:bloch);       # Bloch/EPG forward model
julia> output = example_mrstat_recon(:surrogate);  # neural surrogate forward model
```

Both examples generate noiseless synthetic data with the chosen model on the same 120×120 phantom geometry and reconstruct it. To reconstruct with your own setup:

```julia
model = BlochModel(gpu(f32(sequence)))    # any BlochSimulators sequence
# or
model = load_surrogate()                  # the trained network

raw, coordinates, coils, trajectory, transmit_field, phantom = generate_synthetic_data(model)
output = mrstat_recon(raw, model, coordinates, coils, trajectory, transmit_field)
```

**Unit tests** (128 tests: the interface, both models, objective gradient consistency for both, Gauss-Newton operator invariants, Aqua package quality):

```julia
julia> using Pkg; Pkg.test()      # after `source ../NeuralBloch.jl/env.sh`; requires a GPU
```

**Validation and comparison scripts** (run manually):

- `scripts/compare_models.jl` — **end-to-end comparison of the two forward models**: same phantom, geometry and physics, reconstructed with each model; reports accuracy vs ground truth and speed at three levels (forward model / per objective call / whole reconstruction).
- `scripts/validate_contract.jl` — surrogate `echos`/`∂echos` vs BlochSimulators FISP3D ground truth on the training sequence (sub-percent NMAE) plus zero-copy mechanics checks.
- `scripts/validate_python_parity.jl` — the Julia surrogate wrapper vs *pure-python* execution of the model: exact equality on identical inputs.
- `scripts/run_recon.jl` — surrogate reconstructions of (a) surrogate-generated and (b) EPG-simulated raw data, with error maps written to `scripts/output/`.

## Structure

- `src/ForwardModels/` — the `ForwardModel` interface, `BlochModel`, and `SurrogateModel` (`surrogate.jl`: checkpoint loading through the NeuralBloch python code, the load-time decoder-row permutation that makes the outputs alias directly as Julia `ComplexF32` matrices, zero-copy staging and prediction).
- The objective function is defined in `utils/objective.jl`; it is also used by the non-linear solver to calculate the gradient and the (Gauss-Newton approximation to) the Hessian. Mode-0 evaluations (cost only) ask the model for echos alone.
- The matrix-vector products with the Jacobian matrix and its adjoint are implemented in a partially matrix-free method, see `DerivativeOperations/Jv.jl` and `DerivativeOperations/Jhv.jl`. They consume any model's `∂echos` unchanged (derivatives arrive in physical units; the kernels apply the log-parametrization chain rule themselves) and use the same k-space phase convention as BlochSimulators v0.9's signal path (`exp(-i·k·x)`) — cost and gradient must describe one forward model.
- The non-linear solver (Trust Region Reflective with Gauss-Newton approximation to the Hessian) is contained in the `TrustRegionReflective` submodule.
