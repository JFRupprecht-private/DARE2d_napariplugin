# dare2d-torch — GPU inference for DARE2D (ONNX track, Piste 1 ✅)

Self-contained port of the DARE2D **inference** path off TensorFlow-CPU. It lets
the trained DARE2D checkpoints run on the **GPU** (Quadro RTX 5000) on native
Windows — which TF 2.12 cannot do — **without retraining and without touching
`DARE2d-main/`** (read-only) or the `napari-dare2d` plugin code.

See `../PYTORCH_MIGRATION.md` for the full plan. This folder implements **Piste 1
(ONNX, GPU)** and **Piste 2 (PyTorch port)**. Piste 2 result: **both** models are
faithful torch ports (regression *and* a hand-written qubvel-resnet18-Unet for
segmentation, parity ~1e-7). The segmentation backend is a **switch** — default
**ONNX** (proven/exact) with an opt-in 100%-torch path — so a full-torch pipeline is
available while the safe path stays the default.

## What's here

| file | env | role |
|---|---|---|
| **Piste 1 — ONNX** | | |
| `export_onnx.py` | TF env | rebuild each model via DARE2D's Hydra path, `load_weights`, trace to `.onnx` (dynamic batch, opset 17) |
| `backends.py` | onnx/TF | `OnnxModel`: a Keras-compatible `.model.predict` so `infer_stack`/`inference_strategy` are reused **verbatim**; CUDA DLL setup + provider assertion |
| `verify_parity.py` | TF env | parity gate: TF oracle vs ONNX (seg map, reg head, end-to-end detections) |
| `bench.py` | onnx env | GPU-vs-CPU benchmark (model micro + real end-to-end) |
| `_tf_stub.py` | onnx/torch | minimal fake `tensorflow` so `_api` imports TF-free (a DARE2D util imports TF only for unused metric classes) |
| **Piste 2 — PyTorch** | | |
| `models_torch.py` | torch env | `Regression2dTorch` (NHWC-Flatten fix) + `SegmentationUnetTorch` (faithful qubvel-resnet18-Unet) |
| `_inspect_seg_arch.py` → `seg_arch.json` | TF env | introspect the exact Keras seg graph (the build spec) |
| `keras_dump.py` | TF env | dump Keras weights (native layout) + shapes to `.npz` (`--kind reg|seg`) |
| `convert_to_torch.py` | torch env | `.npz` → torch state_dict (HWIO→OIHW, Dense transpose, BN eps/scale) with shape asserts |
| `verify_parity_torch.py` | torch env | torch ↔ ONNX-oracle parity (`--kind reg|seg`) |
| `torch_backend.py` | torch env | `TorchRegModel`/`TorchSegModel` + `build_hybrid_models(seg_backend=…)` switch |
| `verify_torch_e2e.py` | torch env | end-to-end vs ONNX oracle (`--seg-backend onnx|torch`) |
| `weights_onnx/`, `weights_pt/` | — | exported weights (git-ignored; regenerate) |

**Integration trick (zero edits to existing code):** `OnnxModel` exposes
`obj.model.predict(x_nhwc, verbose=0) -> numpy`, exactly like the DARE2D Keras
wrapper. So `napari_dare2d._api.infer_stack` and the repo's `inference_strategy`
run unchanged — we just pass them ONNX models instead of Keras ones. The shared
numpy/cv2 preprocessing means no preprocessing divergence is possible.

## Results (set 8, mid-stack frames)

**Parity TF ↔ ONNX (CPU, the faithfulness gate):**
- seg probability map: `max|Δ| ≈ 7e-7`, 100.0000 % mask agreement
- regression head: `max|Δ| ≈ 1e-7`
- end-to-end detections: **identical** (same counts, 0 px centre shift, 0° / 0 px)

Confirmed on sets 1, 5, 8 → the export is numerically faithful across the ensemble.

**GPU speedup (RTX 5000, ORT CUDA EP vs ORT CPU):**
- seg U-Net, 49×256×256×3 (one 1024² frame): **~29×**
- regression, 32×64×64×3: **~16×**
- **real `infer_stack` end-to-end: ~2856 → ~154 ms/frame ≈ 18.6×**, GPU detections
  identical to CPU. (End-to-end < model-only because of fixed CPU overhead:
  `equalizeHist`, sliding-window reshape, contour extraction — Amdahl.)

## Piste 2 — PyTorch port (regression ✅ ; segmentation = ONNX fallback)

**Regression → `Regression2dTorch` (faithful).** Weights converted from Keras
(`keras_dump.py` → `convert_to_torch.py`): Conv `HWIO→OIHW`, Dense transpose, and the
critical **NHWC-Flatten** fix (`permute(0,2,3,1)` before flatten — without it the
Dense heads read scrambled features). Parity vs the ONNX oracle: **max|Δ| ≈ 1e-7 on
all 8 sets**. End-to-end hybrid (torch reg on CUDA + ONNX seg) vs oracle: **centres
identical, angle/length Δ ≈ 1e-6** (`verify_torch_e2e.py`).

**Segmentation → both paths, switchable.** `smp.Unet('resnet18')` is **not**
weight-compatible with qubvel-TF (input `bn_data` BN, a stage1 1×1 *projection*
shortcut, preactivation order), so instead `SegmentationUnetTorch` is a **faithful
hand-written port** built straight from the introspected Keras graph
(`seg_arch.json`). Faithful details that make it bit-exact: preactivation resnet18
(BN→ReLU→conv, shortcut from the post-BN-ReLU), explicit **symmetric** `ZeroPadding`
(qubvel uses ZeroPad+`valid`, *not* asymmetric `same`) → exact via `F.pad`, **zero**-pad
before maxpool, the two BN epsilons (encoder `2e-5`, decoder `1e-3`), and `scale=False`
`bn_data`. Parity vs ONNX oracle: **max|Δ| ≈ 1.3e-7, 100% mask agreement on all 8 sets**.

The seg backend is a **switch** (`build_hybrid_models(seg_backend=...)`):
- `"onnx"` (**default**) — the proven, exact ONNX seg (also GPU-ready in the onnx env);
- `"torch"` — the 100%-torch `SegmentationUnetTorch`.

Both give detections identical to TF end-to-end (`verify_torch_e2e.py --seg-backend {onnx,torch}`).
Keeping ONNX the default is deliberate risk mitigation; the torch path is there to use/test.

## Environments

Two envs, bridged by the `.onnx` files (the TF env stays the numerical oracle).

### TF env (export + parity) — existing `napari-env-for-DARE2D-claude`
Added, without disturbing the numpy 1.23.5 / protobuf 4.25.9 pins (constraints
file pins them; `tf2onnx` installed `--no-deps` to dodge its `protobuf<4` pin):
```bash
pip install -c dare2d-torch/_tf_env_constraints.txt "onnx==1.15.0" "onnxruntime==1.16.3"
pip install --no-deps "tf2onnx==1.16.1"
```

### GPU env (inference + benchmark) — `dare2d-onnx`, Python 3.11, numpy 2
```bash
conda create -n dare2d-onnx python=3.11 -y
conda env config vars set PYTHONNOUSERSITE=1 -n dare2d-onnx   # user-site leaks numpy otherwise
conda activate dare2d-onnx
pip install -I "numpy>=1.26"
pip install "onnxruntime-gpu==1.22.0"                          # CUDA 12 build (matches wheels below)
pip install nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 \
            nvidia-cufft-cu12 nvidia-cuda-nvrtc-cu12 nvidia-nvjitlink-cu12
pip install opencv-python scikit-image scipy scikit-learn hydra-core omegaconf \
            matplotlib imageio tifffile click tqdm
```
Notes / gotchas found:
- onnxruntime-gpu **1.27 needs CUDA 13** (`cudart64_13.dll`) but PyPI only ships a
  stub `nvidia-cuda-runtime-cu13==0.0.1`, so we use **1.22 (CUDA 12)** + cu12 wheels.
- cuDNN resolves its sublibraries via **PATH**, not `os.add_dll_directory`, so
  `backends._setup_cuda_dll_path()` prepends every `site-packages/nvidia/*/bin`
  to `PATH` before importing onnxruntime. Without it: silent CPU fallback.
- `build_onnx_models(require_cuda=True)` **asserts** the CUDA EP is active (no
  silent CPU fallback — see PYTORCH_MIGRATION.md §14-A).

### torch env (PyTorch port) — `dare2d-torch`, Python 3.11, numpy 2
```bash
conda create -n dare2d-torch python=3.11 -y
conda env config vars set PYTHONNOUSERSITE=1 -n dare2d-torch
conda activate dare2d-torch
pip install -I "numpy>=1.26"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124  # GPU
pip install onnxruntime segmentation-models-pytorch timm   # onnxruntime = CPU oracle + seg backend
pip install opencv-python scikit-image scipy scikit-learn hydra-core omegaconf \
            tifffile imageio matplotlib                    # for end-to-end infer_stack
```
torch's bundled CUDA coexists with the CPU `onnxruntime` (no CUDA-DLL clash). The
ONNX **seg** runs on CPU in this env; for seg-on-GPU use the `dare2d-onnx` env (Piste 1).

## Reproduce
```bash
# --- Piste 1 (ONNX/GPU) ---
python dare2d-torch/export_onnx.py    --sets 1-8                  # TF env
python dare2d-torch/verify_parity.py  --set 8 --frames 26,27,28,29 # TF env -> "PARITY OK"
python dare2d-torch/bench.py          --set 8 --frames 20-35      # dare2d-onnx env

# --- Piste 2 (PyTorch port: regression + segmentation) ---
python dare2d-torch/keras_dump.py       --kind reg --sets 1-8     # TF env  -> weights_pt/*.npz
python dare2d-torch/keras_dump.py       --kind seg --sets 1-8     # TF env
python dare2d-torch/convert_to_torch.py --kind reg --sets 1-8     # torch env -> *.pt
python dare2d-torch/convert_to_torch.py --kind seg --sets 1-8     # torch env
python dare2d-torch/verify_parity_torch.py --kind reg --sets 1-8  # torch env -> "PARITY OK"
python dare2d-torch/verify_parity_torch.py --kind seg --sets 1-8  # torch env -> "PARITY OK"
python dare2d-torch/verify_torch_e2e.py --set 8 --seg-backend torch  # torch env -> "E2E PARITY OK"
```
