# models/ — runtime model artifacts

Everything in this directory except this README is **gitignored**: trained or compiled
artifacts, *produced* in the sibling [NeuralBloch](https://github.com/NLeSC-COMPAS/NeuralBloch)
repository and *consumed* here. The loaders look **only** in this directory (plus the
explicit overrides below) — never in `../NeuralBloch` — so a machine with this repository,
its system prerequisites, and a populated `models/` runs the TensorRT solution with no
sibling checkout at all.

| file | consumed by | produced by |
|---|---|---|
| `model.pt` | `load_surrogate()` | NeuralBloch training — gitignored there too |
| `model.engine` | `load_tensorrt_model()` | NeuralBloch: `uv run run.py export_onnx && uv run run.py build_engine`. An engine is locked to the GPU architecture and exact TensorRT version that built it |
| `rf_data/rf_angle_list.txt` `rf_data/rf_phase_list.txt` | the validation scripts (they rebuild the training FISP3D sequence for the ground-truth arm) | NeuralBloch `data/rf_data/` |

## Populating

`./bootstrap.sh` copies any missing file in from a `../NeuralBloch` checkout (or
`$NEURALBLOCH_DIR`), and warns when a copy here differs from the checkout — it never
overwrites, so a stale copy after retraining upstream is refreshed by the `cp` command the
warning prints. Without a checkout, copy the files in from wherever they were built:

```bash
scp buildhost:path/to/model.engine models/
scp buildhost:path/to/model.pt     models/
scp buildhost:path/to/rf_*.txt     models/rf_data/
```

## Overrides

| variable | effect |
|---|---|
| `MRSTAT_MODELS_DIR` | use a different artifacts directory (default: this one) |
| `TENSORRT_VERSION` | `build_shim.sh`: override the required TensorRT version (default: read from `model.engine`'s header) |
| `TENSORRT_LIB_DIR` | `build_shim.sh`: directory containing `libnvinfer.so.*` (default: the system install, found via `ldconfig`) |
| `TRT_INCLUDE_DIR` | `build_shim.sh`: TensorRT headers (default: the system headers; else fetched version-pinned from GitHub) |

The TensorRT **runtime is deliberately not an artifact here**: it is a system prerequisite
— NVIDIA's Debian/RPM packages, matching the engine's version — checked and hinted (never
installed) by `bootstrap.sh` and `deps/tensorrt/build_shim.sh`. See
<https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html>.
