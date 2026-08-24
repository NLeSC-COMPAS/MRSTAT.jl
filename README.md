# MR-STAT Reconstruction Code

This repository contains an implementation of the partially-matrix free, inexact Gauss-Newton algorithm that can be used for MR-STAT. It uses a 2D numerical brain phantom to generate synthetic data from which it then estimates $T_1$, $T_2$ and proton density. It only supports Cartesian readout trajectories at the moment.

**The forward model is pluggable.** The reconstruction talks to an abstract `ForwardModel` through two functions — `compute_echos` (echo trains) and `compute_echos_and_derivatives` (echo trains plus physical-unit ∂/∂T₁, ∂/∂T₂) — and two implementations are provided:

- **`BlochModel`** — the physical route: EPG/Bloch simulation of any [BlochSimulators.jl](../BlochSimulators.jl) pulse sequence, with derivatives by BlochSimulators' finite differences. Self-contained; works for any sequence and geometry.
- **`SurrogateModel`** — the learned route: the trained PyTorch surrogate from the sibling [NeuralBloch](../NeuralBloch) repository predicts echos and derivatives directly in one network forward. Julia arrays and torch tensors cross the language boundary **zero-copy** in both directions through [NeuralBloch.jl](../NeuralBloch.jl), and all GPU work runs on a single CUDA stream (no synchronization fences). The pulse sequence is baked into the checkpoint (FISP3D, 600 readouts, TR 5.58 ms, TE 2.745 ms, TI 10.84 ms; trained for T₁ ∈ [0.1, 5] s, T₂ ∈ [0.01, 2] s, B₁ ∈ [0.8, 1.2]), so experiments need `nreadouts == 600` and the solver's default bounds pin the search to the trained box — both enforced by `check_experiment`/`default_bounds`.

Everything downstream of the forward model — phase encoding, signal integration, the partially matrix-free `Jv`/`Jᴴv` Jacobian products, the trust-region solver — is shared. At the example's size (120×120 voxels, 600 readouts) the surrogate produces echos + derivatives ~45–50× faster than the EPG simulation + finite differences (about 3.2 ms vs 160 ms; the scripts time this with a single `CUDA.@elapsed` sample, so runs land across that band). End-to-end reconstruction is dominated by the shared Gauss-Newton products — one reconstruction spends 374 `Jv`/`Jᴴv` applications against just 40 forward-model calls — so the wall-clock gain is Amdahl-limited to ~1.15×, against a ~1.18× ceiling that no forward-model speedup could beat (see `scripts/compare_models.jl` for the full three-level breakdown).

## Setup

```bash
./bootstrap.sh      # once — sets everything up, then reports whatever it could not
source env.sh       # every session, before starting Julia
```

`bootstrap.sh` is idempotent and writes nothing outside this repository. It installs `juliaup`/`uv` if they are absent, finds or builds a Python with `torch` and `matplotlib`, re-points the two `Pkg.develop` path dependencies at your checkouts, and instantiates the project. Anything it cannot supply it names, with the command to fix it.

`env.sh` sets the two variables PythonCall needs — `JULIA_CONDAPKG_BACKEND=Null` and `JULIA_PYTHONCALL_EXE` — choosing the first interpreter that can `import torch` from `$JULIA_PYTHONCALL_EXE`, `$MRSTAT_PYTHON`, `./.venv`, `../NeuralBloch.jl/.venv`, then `python3`.

**Do not skip it — the failure is silent.** Without `JULIA_CONDAPKG_BACKEND=Null`, PythonCall provisions Python itself through CondaPkg: loading `PythonPlot` makes it download a 1.3 GB conda environment into `.CondaPkg/` and bind to *that* interpreter. Nothing errors. But CondaPkg installs only what the Julia packages declare — PythonCall's `python`/`openssl`/`libstdcxx` plus PythonPlot's `matplotlib` — so the environment has no `torch`, and the first sign of trouble is `load_surrogate()` failing several minutes later. The two requirements pull in opposite directions (CondaPkg supplies matplotlib but never torch; a torch venv has no matplotlib unless you add it), so only one interpreter carrying **both** works. `bootstrap.sh` enforces exactly that.

### What must exist outside this repository

| | Needed for | Where it comes from |
|---|---|---|
| NVIDIA GPU + CUDA driver | everything | the package is CUDA-only; the tests hard-fail without one |
| `../BlochSimulators.jl` (0.9) | everything | `git clone https://github.com/MagneticResonanceImaging/BlochSimulators.jl.git` |
| `../NeuralBloch.jl` (0.3) | the surrogate | **no public remote** — obtain the checkout from its author |
| `../NeuralBloch` | the surrogate | `git clone git@github.com:NLeSC-COMPAS/NeuralBloch.git` — supplies `src/model.py`, the RF tables, and the checkpoint directory |
| `../NeuralBloch/outputs/model.pt` | the surrogate | **gitignored upstream** — train it there, or copy one in |
| Python with CUDA `torch` | the surrogate | `bootstrap.sh` builds `./.venv` if no reusable interpreter exists |
| …and `matplotlib` | the plotting scripts | the same interpreter; PythonPlot cannot install it itself under `CONDAPKG_BACKEND=Null` |

Only the surrogate route needs the NeuralBloch pieces. `BlochModel` and the entire reconstruction work without them, and torch is imported lazily — `load_surrogate()` is the first thing that touches Python.

Two escape hatches for non-standard layouts: **`NEURALBLOCH_DIR`** overrides where the python repository is looked for, and **`MRSTAT_PYTHON`** overrides the interpreter. The first matters more than it sounds: `pkgdir` resolves to the *worktree* root, so a git worktree created outside the shared parent directory will not find its siblings by default.

**Caveat on `Manifest.toml`.** It is tracked, and it records the two path dependencies as *absolute* paths from wherever it was last resolved — i.e. someone else's home directory on a fresh clone. `bootstrap.sh` rewrites them for your machine, which is why it has to run before the first `Pkg.instantiate()`.

## Usage

```julia
julia> using MRSTAT               # after `source env.sh`
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
julia> using Pkg; Pkg.test()      # after `source env.sh`; requires a GPU
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
