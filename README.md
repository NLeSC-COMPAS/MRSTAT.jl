# MR-STAT Reconstruction Code

This repository contains an implementation of the partially-matrix free, inexact Gauss-Newton algorithm that can be used for MR-STAT. It uses a 2D numerical brain phantom to generate synthetic data from which it then estimates $T_1$, $T_2$ and proton density. It only supports Cartesian readout trajectories at the moment.

**The forward model is pluggable.** The reconstruction talks to an abstract `ForwardModel` through two functions — `compute_echos` (echo trains) and `compute_echos_and_derivatives` (echo trains plus physical-unit ∂/∂T₁, ∂/∂T₂). Two implementations ship with the package:

- **`BlochModel`** — the physical route: EPG/Bloch simulation of any [BlochSimulators.jl](https://github.com/MagneticResonanceImaging/BlochSimulators.jl) pulse sequence, with derivatives by finite differences. Self-contained; works for any sequence and geometry.
- **`SurrogateModel`** — the learned route: the trained PyTorch surrogate from [NeuralBloch](https://github.com/NLeSC-COMPAS/NeuralBloch) predicts echos *and* derivatives in a single network forward. Julia arrays and torch tensors cross the language boundary **zero-copy** in both directions through [NeuralBloch.jl](https://github.com/NLeSC-COMPAS/NeuralBloch.jl), on one CUDA stream with no synchronization fences. Both the sequence (FISP3D, 600 readouts, TR 5.58 ms, TE 2.745 ms, TI 10.84 ms) and the trained box (T₁ ∈ [0.1, 5] s, T₂ ∈ [0.01, 2] s, B₁ ∈ [0.8, 1.2]) are baked into the checkpoint; `check_experiment` and `default_bounds` hold experiments to both.

Everything downstream of the forward model — phase encoding, signal integration, the partially matrix-free `Jv`/`Jᴴv` Jacobian products, the trust-region solver — is shared. The surrogate produces echos + derivatives ~40–46× faster than EPG simulation + finite differences (3.1–3.5 ms against ~143 ms at 14,400 voxels), but a reconstruction spends far more time in the shared Gauss-Newton products than in the forward model, so the end-to-end gain is Amdahl-limited to ~1.2×. [`scripts/compare_models.jl`](#scripts) measures all three levels.

## Setup

```bash
./bootstrap.sh      # once — sets everything up, then reports whatever it could not
source env.sh       # every session, before starting Julia
```

`bootstrap.sh` is idempotent and writes nothing outside this repository: it installs `juliaup`/`uv` if they are absent, finds or builds a Python with `torch` and `matplotlib`, re-points the one `Pkg.develop` path dependency (`NeuralBloch.jl`) at your checkout, and instantiates the project. Anything it cannot supply it names, with the command to fix it.

`env.sh` sets the two variables PythonCall needs — `JULIA_CONDAPKG_BACKEND=Null` and `JULIA_PYTHONCALL_EXE` — choosing the first interpreter that can `import torch` from `$JULIA_PYTHONCALL_EXE`, `$MRSTAT_PYTHON`, `./.venv`, `../NeuralBloch.jl/.venv`, then `python3`.

**Do not skip it — the failure is silent and slow.** Without `JULIA_CONDAPKG_BACKEND=Null`, loading `PythonPlot` makes CondaPkg download a 1.3 GB conda environment into `.CondaPkg/` and bind PythonCall to *that* interpreter. Nothing errors: CondaPkg installs only what the Julia packages declare, which covers matplotlib but never torch, so the first sign of trouble is `load_surrogate()` failing several minutes later. Only one interpreter carrying **both** works, and `bootstrap.sh` enforces exactly that.

### What must exist outside this repository

| | Needed for | Where it comes from |
|---|---|---|
| NVIDIA GPU + CUDA driver | everything | the package is CUDA-only; the tests hard-fail without one |
| `../NeuralBloch.jl` (0.3–0.4) | everything | not in the General registry: `git clone https://github.com/NLeSC-COMPAS/NeuralBloch.jl.git` |
| `../NeuralBloch` | the surrogate | `git clone https://github.com/NLeSC-COMPAS/NeuralBloch.git` — supplies `src/model.py`, the RF tables and the checkpoint directory |
| `../NeuralBloch/outputs/model.pt` | the surrogate | **gitignored upstream** — train it there, or copy one in |
| Python with CUDA `torch` **and** `matplotlib` | everything | `bootstrap.sh` builds `./.venv` if no interpreter has both; `using MRSTAT` loads PythonPlot, which imports matplotlib at init |

Every other dependency — BlochSimulators included — comes from the General registry, so `Pkg.instantiate()` fetches it. `NeuralBloch.jl` is the one exception, and since `ForwardModels` imports it at load time the checkout must be there even for a pure `BlochModel` run. The *python* pieces are the genuinely surrogate-only ones: torch is imported lazily, by `load_surrogate()`.

Two escape hatches for non-standard layouts: **`NEURALBLOCH_DIR`** overrides where the python repository is looked for, **`MRSTAT_PYTHON`** overrides the interpreter. The first matters more than it sounds: `pkgdir` resolves to the *worktree* root, so a git worktree created outside the shared parent directory will not find its siblings by default.

## Usage

```julia
julia> using MRSTAT               # after `source env.sh`
julia> output = example_mrstat_recon(:bloch);      # Bloch/EPG forward model
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

## Scripts

Four standalone experiments live in `scripts/`. They are deliberately outside the test suite: each takes between half a minute and two minutes, and two of them write figures. All four need a GPU and the surrogate (checkpoint + `../NeuralBloch`), and all four are run the same way:

```bash
source env.sh
julia --project=. scripts/compare_models.jl      # …or any of the four below
```

The solver narrates every trust-region iteration, so expect a long scroll of `Info: Choose step` and Steihaug-CG dots; the report each script prints is the block at the end. The figures quoted below come from one run on a NVIDIA GeForce RTX 2080 Ti (11 GB VRAM) — timings are hardware-dependent, and the forward-model ratios are single `CUDA.@elapsed` samples, so they move between runs.

**`compare_models.jl` — the two forward models, end to end.** One experiment for both arms: the same 120×120 phantom, geometry, coils and B₁ map, and the same physics (the Bloch arm simulates the FISP3D sequence the surrogate was trained on), so the two models describe one acquisition. Each arm reconstructs data generated by its *own* forward model, so each is exactly self-consistent and both recover the phantom to solver tolerance (0.000% median error in T₁, T₂ and |ρ|); the accuracy number that carries information is therefore the arm-vs-arm one — the two models land on the same maps, 0.000% median |ΔT₁|/T₁. Speed is reported at three levels, because each answers a different question: the forward model alone (40×), one objective call (mode 0: 6.9 ms against 42.1 ms; mode 2: 23.2 ms against 137.1 ms), and end-to-end wall clock — 19.3 s against 23.3 s, a **1.21×** gain against a **1.30×** ceiling that even a free forward model could not beat. The reason sits in the same table: the surrogate arm spends 1.3 s inside the objective and 18.0 s in the shared solver internals, the Steihaug-CG `Jv`/`Jᴴv` products. Nothing is timed until a throwaway warm-up solve has run with *each* model: the ~23 s of one-off JIT compilation for those shared kernels otherwise lands entirely on whichever arm goes first, which is enough to reverse the ranking. Writes `scripts/output/compare_*.png`.

**`validate_contract.jl` — does the surrogate honour the `ForwardModel` contract?** Three parts. **(1) Zero-copy mechanics:** pointer identity through both boundary crossings (real and complex paths), the `view_as_complex` pairing, and `StructArray` `∂echos` field access inside a `@cuda` kernel. **(2) Accuracy against ground truth** — BlochSimulators' FISP3D EPG simulation plus finite-difference derivatives on the training sequence, over a 168-point (T₁, T₂, B₁) grid: 0.62% NMAE on the signal, 0.35% and 0.61% on ∂/∂T₁ and ∂/∂T₂, plus a check that the cheap `compute_echos` path is bitwise identical to the full forward. The 15% gate is loose on purpose: a convention bug (orientation, units, complex layout, log-space mixup) misses by order 100%, while the true approximation error is sub-percent, so the two can never be mistaken for each other. **(3) Forward-model timing at reconstruction size** (14,400 voxels): 3.1 ms against 143.1 ms, 45.8×. Terminal output only.

**`validate_python_parity.jl` — does the Julia wrapper change the numbers?** The reference is NeuralBloch's own way of running the model: a fresh `FCModel` from their `src/model.py`, an *unpermuted* state dict, plain torch tensors, a vanilla `no_grad` forward — none of the wrapper's machinery (decoder permutation, zero-copy crossings, complex views, input staging) touches it. Given the identical staged float32 rows the two paths agree **exactly** — max |Δ| = 0 on the signal and on both derivatives — because the permuted decoder is an exact row reordering of a bias-free matrix and both run the same cuBLAS kernels on the same device. Given the same physical parameters they agree to within 5e-5 (gate: 1e-4), since one side takes the logs in double precision and the other stages Float32. Terminal output only.

**`run_recon.jl` — two surrogate reconstructions of one phantom.** (a) From surrogate-generated data: model and data share one forward model, so the maps come back at 0.00% median *and* p95 error — which is what exercises the whole pipeline end to end (zero-copy bridge, `∂echos`, J products, solver). (b) From EPG-simulated data — the scanner-like case, where the reconstruction inherits the surrogate's approximation error: 0.30% median / 1.16% p95 on T₁, 0.72% / 1.25% on T₂, 0.18% / 0.79% on |ρ|. Writes maps and error maps to `scripts/output/`.

## Structure

- `src/ForwardModels/` — the `ForwardModel` interface and its two implementations. `surrogate.jl` holds checkpoint loading (through NeuralBloch's own python code), the load-time decoder-row permutation that makes the network's output alias directly as a Julia `ComplexF32` matrix, and zero-copy staging and prediction.
- `src/utils/objective.jl` — the objective, used by the solver for the cost, the gradient and the Gauss-Newton approximation to the Hessian. Cost-only evaluations (mode 0) ask the model for echos alone.
- `src/DerivativeOperations/` — the partially matrix-free Jacobian products, `Jv.jl` and `Jhv.jl`. They consume any model's `∂echos` unchanged (derivatives arrive in physical units; the kernels apply the log-parametrization chain rule themselves) and use the same k-space phase convention as BlochSimulators v0.9's signal path, `exp(-i·k·x)` — cost and gradient must describe one forward model.
- `src/TrustRegionReflective/` — the non-linear solver: Trust Region Reflective with a Gauss-Newton approximation to the Hessian.
