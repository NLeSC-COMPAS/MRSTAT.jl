# MR-STAT Reconstruction Code

This repository implements the partially matrix-free, inexact Gauss-Newton algorithm for MR-STAT: it generates synthetic data from a 2D numerical brain phantom and estimates $T_1$, $T_2$ and proton density. Cartesian readout trajectories only.

**The forward model is pluggable.** The reconstruction talks to an abstract `ForwardModel` through two functions — `compute_echos` (echo trains) and `compute_echos_and_derivatives` (echo trains plus physical-unit ∂/∂T₁, ∂/∂T₂). Three implementations ship with the package:

- **`BlochModel`** — the physical route: EPG/Bloch simulation of any [BlochSimulators.jl](https://github.com/MagneticResonanceImaging/BlochSimulators.jl) pulse sequence, derivatives by finite differences. Self-contained; any sequence and geometry.
- **`SurrogateModel`** — the learned route: the trained PyTorch surrogate from [NeuralBloch](https://github.com/NLeSC-COMPAS/NeuralBloch) predicts echos *and* derivatives in one network forward, crossing the language boundary **zero-copy** in both directions through [NeuralBloch.jl](https://github.com/NLeSC-COMPAS/NeuralBloch.jl) — a *weak* dependency whose package extension activates on `using NeuralBloch`. The sequence (FISP3D, 600 readouts, TR 5.58 ms, TE 2.745 ms, TI 10.84 ms) and the trained box (T₁ ∈ [0.1, 5] s, T₂ ∈ [0.01, 2] s, B₁ ∈ [0.8, 1.2]) are baked into the checkpoint; `check_experiment` and `default_bounds` enforce both.
- **`TensorRTModel`** — the compiled route: the *same* trained network through its prebuilt TensorRT engine, **no Python in the path** — Julia `ccall`s into a small C shim over the system TensorRT runtime (`deps/tensorrt/`, compiled once per machine). The `(3, N)` input buffer is bound zero-copy by device pointer; the engine writes its paired (Re, Im) outputs straight into `(NTR, N)` `ComplexF32` buffers, leaving one contiguous device copy per consumed output (irreducible — see the comment at `_combine` in `src/ForwardModels/tensorrt.jl`); the whole invocation replays as a CUDA graph (~5 µs host submit). At 14,400 voxels: engine 1.4 ms, full forward 2.2 ms, PyTorch surrogate 2.5 ms. Echo count and admissible batch range (10,000–40,000, tuned for 14,400) are read from the engine itself; batches outside the profile fail loudly. Engines are built in NeuralBloch, never here.

Everything downstream — phase encoding, signal integration, the matrix-free `Jv`/`Jᴴv` Jacobian products, the trust-region solver — is shared. The surrogate computes echos + derivatives ~38× faster than EPG + finite differences (3.7 vs 141 ms at 14,400 voxels), but reconstructions spend most of their time in the shared Gauss-Newton products, so the end-to-end gain is Amdahl-limited to ~1.1× — [`scripts/compare_models.jl`](#scripts) measures all three levels.

## Setup

```bash
./bootstrap.sh                 # once — sets everything up, then reports whatever it could not
source env.sh                  # every session, before starting Julia
deps/tensorrt/build_shim.sh    # once, optional — compiles the TensorRT shim against the system TensorRT
```

`bootstrap.sh` is idempotent and writes nothing outside this repository: it installs `juliaup`/`uv` when absent, finds or builds a Python with `torch` + `matplotlib`, copies model artifacts into `models/` from a NeuralBloch checkout (see `models/README.md`), instantiates and precompiles the Julia project, and names anything it cannot supply with the command to fix it. `env.sh` binds PythonCall to a torch-capable interpreter (`JULIA_CONDAPKG_BACKEND=Null` + `JULIA_PYTHONCALL_EXE`) and puts a `../NeuralBloch.jl` checkout on the Julia load path so `using NeuralBloch` resolves.

**Do not skip `env.sh` — the failure is silent and slow.** Without it, CondaPkg downloads a 1.3 GB conda environment that carries matplotlib but never torch, and nothing errors until `load_surrogate()` fails minutes later.

### System prerequisites

Checked and hinted by `bootstrap.sh` and `build_shim.sh`, never installed by them:

| prerequisite | needed by | install |
|---|---|---|
| NVIDIA GPU + driver | every forward model | your distro's or [NVIDIA's](https://www.nvidia.com/drivers/) driver packages; verify with `nvidia-smi`. No system CUDA toolkit needed for Julia — CUDA.jl ships its own runtime |
| TensorRT runtime + headers | TensorRT model only | NVIDIA's deb/rpm packages, **version matching `models/model.engine`** (engines are version-locked): `sudo apt-get install libnvinfer11 libnvinfer-headers-dev` — [install guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html). Headers are compile-time only (fetched version-pinned when absent) |
| CUDA toolkit + g++ | TensorRT model only | for the one-time shim compile: `sudo apt-get install cuda-toolkit build-essential` from the same NVIDIA repo; `CUDA_HOME` overrides `/usr/local/cuda` |

### What each forward model needs

The **base setup**, shared by all three, is what `using MRSTAT` itself needs: the instantiated clone (every Julia dependency comes from the General registry), a GPU, and a Python with `matplotlib` — `bootstrap.sh` + `env.sh` handle it.

**`BlochModel`** — the base setup and nothing else.

**`SurrogateModel`** — base plus the PyTorch pieces:
- `../NeuralBloch.jl` checkout — the zero-copy Julia↔torch bridge (weak dependency; `env.sh` stacks it, `using NeuralBloch` activates the extension);
- `../NeuralBloch` checkout — the python model *code* (`src/model.py`) the surrogate runs through;
- a CUDA-enabled `torch` (installed by `bootstrap.sh` when it builds `./.venv`);
- `models/model.pt` — the trained checkpoint, auto-copied from `../NeuralBloch/outputs/`.

**`TensorRTModel`** — base plus the system TensorRT rows above; no Python pieces:
- `models/model.engine` — built upstream (`uv run run.py export_onnx && uv run run.py build_engine`), locked to the GPU architecture and exact TensorRT version that built it;
- the shim, compiled once by `deps/tensorrt/build_shim.sh` (it reads the required TensorRT version from the engine's header and errors with hints on a mismatch).

With `models/` populated, the TensorRT path runs from this repository alone — no torch, no sibling checkout. For non-standard layouts: `NEURALBLOCH_DIR` (python repo), `NEURALBLOCH_JL_DIR` (bridge checkout), `MRSTAT_PYTHON` (interpreter), `MRSTAT_MODELS_DIR` (artifacts directory) — git worktrees outside the shared parent need these, since `pkgdir` resolves to the worktree root.

## Usage

```julia
julia> using MRSTAT               # after `source env.sh`
julia> output = example_mrstat_recon(:bloch);      # Bloch/EPG forward model
julia> output = example_mrstat_recon(:tensorrt);   # the same network as a TensorRT engine
julia> using NeuralBloch          # surrogate only — activates its package extension
julia> output = example_mrstat_recon(:surrogate);  # neural surrogate forward model
```

The examples generate noiseless synthetic data with the chosen model on the same 120×120 phantom geometry and reconstruct it. With your own setup:

```julia
model = BlochModel(gpu(f32(sequence)))    # any BlochSimulators sequence
# or
model = load_surrogate()                  # the trained network, via PyTorch (after `using NeuralBloch`)
# or
model = load_tensorrt_model()             # the trained network, via its TensorRT engine

raw, coordinates, coils, trajectory, transmit_field, phantom = generate_synthetic_data(model)
output = mrstat_recon(raw, model, coordinates, coils, trajectory, transmit_field)
```

**Unit tests** — 169 tests covering the interface, all three models (the TensorRT testsets skip with instructions when the shim or engine is absent), objective gradient consistency, and Aqua package quality. The suite exercises the surrogate, so it needs its pieces (`../NeuralBloch.jl` via the test `[sources]`, `../NeuralBloch`, `models/model.pt`):

```julia
julia> using Pkg; Pkg.test()      # after `source env.sh`; requires a GPU
```

## Scripts

Five standalone experiments in `scripts/`, deliberately outside the test suite (each takes half a minute to a few minutes; two write figures). All need a GPU; the surrogate-driven ones also need `../NeuralBloch`, while `validate_tensorrt.jl` needs only the shim and a populated `models/` (its torch and ground-truth parts skip with a note otherwise). Quoted figures are from one run on an RTX 2080 Ti — timings move between runs.

```bash
source env.sh
julia --project=. scripts/compare_models.jl      # …or any of the four below
```

**`compare_models.jl`** — one experiment, three arms, each reconstructing its own self-consistent data. Speed at three levels: the forward model alone (surrogate 3.7 ms vs Bloch 141 ms, 38×; TensorRT 2.2 ms — 63× vs Bloch, 1.67× vs the surrogate), one objective call (mode 0: 7.5 / 42.1 / 6.4 ms for surrogate / Bloch / TensorRT; mode 2: 22.7 / 136.8 / 21.0 ms), and end-to-end wall clock (20.3 s vs Bloch's 23.2 s — 1.14× against a 1.18× free-forward-model ceiling; TensorRT 20.0 s). All arms recover the phantom to ≤ 0.002% median error and land on the same maps (arm-vs-arm ≤ 0.001%); a throwaway warm-up solve runs first so one-off JIT compilation lands on no arm. Writes `scripts/output/compare_*.png`.

**`validate_contract.jl`** — the surrogate against the `ForwardModel` contract: zero-copy mechanics (pointer identity through both crossings, complex pairing, `StructArray` access in kernels), accuracy against EPG ground truth on a 168-point grid — 0.56% NMAE on the signal, 0.23% / 0.39% on the derivatives, mode-0 bitwise-identical to the full forward (a convention bug misses by ~100%, hence the loose 15% gate) — and timing (3.3 vs 143.3 ms, 43×).

**`validate_tensorrt.jl`** — the engine contract as the shim reports it, EPG accuracy at the engine's optimal batch (0.56% / 0.25% / 0.45% NMAE), parity with the PyTorch surrogate (0.0% NRMSE, gate 0.1%), speed (engine 1.38 ms, full forward 2.22 ms, surrogate 2.52 ms), and a complete reconstruction recovered at 0.000% median error.

**`validate_python_parity.jl`** — the Julia wrapper against NeuralBloch's own way of running the model, on the same `models/model.pt`. Exact agreement (max |Δ| = 0), both on identical staged float32 inputs and end-to-end with each side casting the physical parameters itself.

**`run_recon.jl`** — two surrogate reconstructions of one phantom: self-consistent data comes back at 0.00% median error (exercising the whole pipeline), EPG data inherits the surrogate's approximation error — 0.31% / 0.39% / 0.11% median on T₁ / T₂ / |ρ| (p95 ≤ 1.07%). Writes maps to `scripts/output/`.

## Structure

- `src/ForwardModels/` — the `ForwardModel` interface and its three implementations. `surrogate.jl` holds the surrogate's type, contract methods and input staging; the NeuralBloch.jl-touching half — checkpoint loading through NeuralBloch's own python code, zero-copy predictions whose natively paired (Re, Im) outputs alias directly as Julia `ComplexF32` matrices — lives in the `ext/MRSTATNeuralBlochExt.jl` extension. `tensorrt.jl` holds the engine loader (contract validated; NTR and batch profile read from the engine), the pointer-bound buffers and the CUDA-graph hot path; the C shim and its build script live in `deps/tensorrt/`.
- `models/` — the gitignored runtime artifacts (checkpoint, engine, RF tables), the only place the loaders look; populated by `bootstrap.sh`. See `models/README.md`.
- `src/utils/objective.jl` — the objective (cost, gradient, Gauss-Newton Hessian approximation); cost-only evaluations (mode 0) ask the model for echos alone.
- `src/DerivativeOperations/` — the matrix-free `Jv`/`Jᴴv` products. They consume any model's physical-unit `∂echos` (the kernels apply the log-parametrization chain rule) and use BlochSimulators v0.9's k-space phase convention, `exp(-i·k·x)`.
- `src/TrustRegionReflective/` — the non-linear solver: Trust Region Reflective with a Gauss-Newton Hessian approximation.
