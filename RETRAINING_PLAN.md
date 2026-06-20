# Plan — DARE2D retraining plugin (leave-one-out: train on a subset, test on the complement)

_Plan only (no code yet), grounded in the actual DARE2d-main training stack and the
neuroepithelium dataset. Companion to `PYTORCH_MIGRATION.md`. Dated 2026-06-20._

## 0. Goal
A napari widget to **retrain** DARE2D on a chosen subset of sets and **test on the
complementary set** — e.g. train on sets 1–7, test on set 8. Output checkpoints that
the existing inference plugin can use directly.

## FOLDER LAYOUT (reorganised 2026-06-20)
```
data/neuroepithelium/neuroepithelium/set_1..8   <- training data_dir (movie + division_position*.npy)
models/
  best/                                          <- CURATED checkpoints (READ-ONLY)
    regression_checkpoints/checkpoints_set_{n}_all_but_target/best.h5
    segmentation_checkpoints/checkpoints_set_{n}_all_but_target/best.h5
  <run_name>/                                    <- each retraining writes a NEW sibling folder here
    regression_checkpoints/checkpoints_set_{test}_all_but_target/best.h5
    segmentation_checkpoints/checkpoints_set_{test}_all_but_target/best.h5
set_8/                                           <- demo movie + annotations (annotation viewer)
```
Inference defaults now point at `models/best/...` (`_api.MODELS_DIR`, `DEFAULT_REG_DIR`, `DEFAULT_SEG_DIR`).

## HARD CONSTRAINT — never destroy existing checkpoints
**DO NOT erase or overwrite ANY existing `best.h5`.** DARE2D's training callback writes
`checkpoints_{set}_all_but_target/best.h5` with `save_best_only=True`, which would clobber
curated weights. The plugin MUST instead:
- write every run into a **new folder inside `models/`** — `run name` defaults to the date
  (e.g. `models/2026-06-20/...`), settable; **never** into `models/best/` (the curated dir);
- **refuse to write** if the target path already holds a `best.h5` (auto-suffix the run
  name, e.g. `2026-06-20_2`, or append a timestamp) — overwriting is never allowed;
- treat `models/best/` as **read-only**; switching to a retrained model = point the inference
  plugin's reg/seg dirs at `models/<run_name>/{regression,segmentation}_checkpoints` (same
  layout as `find_checkpoints`), leaving the curated `best/` untouched.

## DECISIONS TAKEN — training backends (2026-06-20, updated)
- **Build BOTH training backends, with a toggle** in the napari training module (mirrors the
  inference Keras/PyTorch switch):
  - **TF backend** = the validated `retrain/train_split.py` driver, run on **CPU (Windows)** or
    **GPU via WSL2**. Zero training-divergence risk (reuses DARE2D's own training).
  - **PyTorch backend** = native-Windows GPU training reusing the faithful `Regression2dTorch` /
    `SegmentationUnetTorch` + the same prepared data; torch `Dataset`/losses/trainer (no WSL).
- **`DARE2d-main/` stays untouched.** If a change there is ever unavoidable, **duplicate it to
  `DARE2d-main_TF/`** and edit the copy (never the original).
- Both backends write to **`models/{run}/`** with the never-overwrite guard; TF emits `best.h5`,
  torch emits `.pt` (directly usable by the torch inference backend; export to `.onnx`/`.h5` optional).

## DECISIONS TAKEN (2026-06-20)
- **Backend for v1 = TF reuse** (Phase A), then PyTorch GPU port later (Phase B).
- **WSL2 + CUDA is available** → Phase A trains on the **GPU inside WSL2** with the
  *existing validated* TF code. This is the sweet spot: GPU speed, zero training-
  divergence risk, low effort (no big port). The torch training port (Phase B) becomes
  **optional** — only worth it to drop the WSL dependency for a 100%-native-Windows path.
- ⇒ The plugin spawns training as a **`wsl` subprocess** (§5). Main new pieces: a thin
  single-split driver (§6), Windows↔WSL path translation, a WSL env with TF-GPU, and
  progress/cancel wiring. **Prerequisite to confirm:** a WSL conda/venv with
  `tensorflow==2.12` (GPU) + the DARE2D deps.

## 1. What already exists (reuse, don't reinvent)
DARE2d-main already does exactly this, as a CLI:
- `scripts/batch_train/batch_train_eval.py` → `BatchTrainingProcedure.run()` loops over
  every target set: trains on **all sets but the target**, validates/tests on the target,
  and saves `checkpoints_{target}_all_but_target/best.h5` — the **same naming the inference
  plugin's `find_checkpoints` reads**. The widget is essentially **one parameterized
  iteration** of this loop.
- `scripts/train/training_pipeline.py` → `TrainingProcedure`: `init_datamodule` /
  `init_model` (Hydra) / `init_callbacks` / `train()` (`trainer.fit`) / `test()`
  (`trainer.test`).
- `dare2d/trainer/basic_trainer.py`: `.fit` = Keras `model.fit(...)`; `.test` =
  `model.evaluate(...)` + prediction metrics → `eval_history` / `test_history`.
- `dare2d/datamodule/datamodule.py`: `set_trains([gens], sample_limit)` accepts a **list**
  of per-set generators → "train on sets X" = list of those sets' generators; val/test =
  the target set's generator. Builds `tf.data.Dataset` (+ augmentation for train).
- Generators `dare2d/datamodule/generator/_2d/{regress,seg}_2dataset.py`: read one set
  (movie `.tiff` + `division_position*.npy`) → crops (regression) / masks (segmentation).

**Reuse strategy (mirror the inference `_api`):** a thin driver imports these classes and
runs a SINGLE split. `DARE2d-main/` stays read-only.

## 2. Data layout (confirmed)
`data/neuroepithelium/neuroepithelium/` is the `data_dir`: `set_1 … set_8`, each =
1 movie `.tiff` + `division_position{i}.npy` (int `[x,y,frame]`, paired rows). `set_8`
there is the same movie+annotations as the demo `set_8/` at the repo root.
`config/batch_training/*.yaml` already maps `set_1..8 -> ${data_dir}/set_N`.

## 2-bis. Data pipeline — PREPROCESSING REQUIRED (discovered while building)
The training generators do **not** read the raw `movie.tiff + division_position*.npy`.
They read a **preprocessed per-frame layout**:
```
<set>/[crop_<n>/]  previmg/{i}.tif  currimg/{i}.tif  nextimg/{i}.tif  div_location/{i}.npy
```
- **Converter = `annotator/preprocessing/format_gastru.py`** (reuse, read-only): for each
  annotated frame `f`, writes prev=`stack[f-2]`, curr=`stack[f-1]`, next=`stack[f]`, and
  `div_location` = the `[x,y]` pairs (drops the frame column). `crop_size>0` tiles each frame
  into `crop_size²` non-overlapping crops (assigning bipoints to the crop containing their
  centre). The neuroepithelium sets are RAW → preprocessing must run first (set_3 has empty
  stub folders; others none).
- **Segmentation masks are NOT on disk** — `seg_2dataset` builds them **on the fly** from the
  bipoints (`cv2.circle` at `cell_radius`, `type_mask=center`). So seg needs the *same* inputs
  as regression (no `currlabel/`, no MATLAB). `convertomask3.m` is an unrelated overlay tool.
- **crop_size**: seg U-Net input = 256², reg crops 64² around centres → preprocess at
  **`crop_size=256`** (currimg = 256² tiles; reg dataset crops 64² from them). To confirm vs the
  originally-trained data; exposed as a parameter.
- **Write derived data to a separate location** (e.g. `data/prepared/<set>/`), leaving the raw
  sets pristine; cache it so preprocessing runs once per set.

Revised pipeline: **preprocess (once) → leave-one-out train → test → `models/{run}/`**.

## 3. Plugin UX (proposed)
Widget **"DARE2D retraining"**:
- **Data folder** (dir; default `data/neuroepithelium/neuroepithelium`).
- **Test set** (1–8). Training sets = the complement (plus an optional multiselect for
  custom subsets).
- **Model**: regression / segmentation / both.
- **Run name** (default = today's date, e.g. `2026-06-20`) — names the new folder under
  `models/`; auto-suffixed if it already exists so **no `best.h5` is ever overwritten**.
- **Hyperparams** with the configs' defaults: reg = 50 epochs × 1000 steps; seg = 50 × 250;
  batch 32; crop 64 (reg) / 256 (seg). Exposed but pre-filled.
- **Output**: a fresh **`models/{run_name}/`** folder (sibling of `models/best/`, never
  inside it) holding `{regression,segmentation}_checkpoints/checkpoints_set_{test}_all_but_target/best.h5`.
  To use it for inference, point the reg/seg dirs at `models/{run_name}/...` — layout matches
  `find_checkpoints`.
- Run → background training with live progress (epoch/step, train & val loss), **Cancel**,
  then report test metrics (reg: length & angle MAE; seg: f-measure).

## 4. THE crux — training backend (decide first)
Training is all TF/Keras (`model.fit`, `tf.data`). TF 2.12 on **native Windows = CPU only**
(same wall as inference, but training is *far* heavier: 2 models × ~50 epochs × hundreds of
steps over 1024² movies).

| Backend | GPU on native Windows | Effort | Notes |
|---|---|---|---|
| **TF reuse, CPU** | no (CPU) | **low** | reuse validated code as-is; correct but **slow** — only ok for small/quick retrains |
| **TF reuse, WSL2 GPU** | via WSL | low–med | fast + validated; needs WSL2 + CUDA, path/env translation, subprocess into WSL |
| **PyTorch training port** | **yes (native)** | **high** | fast native-Windows GPU; **reuses our already-faithful torch models**; must port losses, metrics, data pipeline, trainer loop (details §6) |

## 5. Execution model — subprocess, not in-process
Run training as a **subprocess** (its own Python/env), not in napari's thread:
- doesn't block or destabilise the GUI (TF graph + Qt in one process over hours is fragile);
- lets training run in a **GPU-capable env** (the torch-train env, or WSL2 for TF-GPU)
  independent of napari's env;
- **cancellable** (kill the process);
- monitored via the `scores.json` the trainer already writes + stdout/last-epoch parse, or a
  tiny progress file the driver appends to.
The widget thread just spawns, tails progress, and reports — like the inference worker but
longer-lived and killable.

## 6. Architecture
- **Driver** `retrain_split(data_dir, train_sets, test_set, which_model, hparams, out_dir,
  progress_cb)` — a de-looped `BatchTrainingProcedure`: build per-set generators (train =
  complement, val/test = target), instantiate the model via the same Hydra experiment used
  at inference, `fit`, `test`, save `best.h5` to `out_dir`. Lives in a new module (e.g.
  `napari_dare2d/_train.py` or `dare2d-torch/train_*.py`), imports DARE2d-main (read-only).
- **Phase A backend = TF**: the driver *is* the reuse; near-zero training-divergence risk.
- **Phase B backend = PyTorch** (for native-Windows GPU), reusing the faithful
  `Regression2dTorch` / `SegmentationUnetTorch`:
  - **Losses** (tractable): regression = plain **MSE** on length and on the 2-D angle target
    (`F.mse_loss`); segmentation = the weighted BCE `pixelwise_weighted_binary_crossentropy2`
    → `F.binary_cross_entropy_with_logits` with the per-pixel weight channel. (A decorrelation
    loss exists but the default reg config uses MSE.)
  - **Data pipeline**: port `Regress2Dataset` / `Seg2Dataset` → `torch.utils.data.Dataset`
    (read movie + paired annotations → crops / weighted masks; the seg target carries a
    weight map from `weight_scaling`/`cell_radius`). **Augmentation is albumentations +
    small numpy ops** → framework-agnostic, largely reusable.
  - **Trainer**: a plain torch loop (rmsprop for reg / adam for seg, epochs × steps,
    checkpoint best on val loss, optional early stop).
  - **Exact data specs to reproduce (gathered from the generators):**
    - *Regression sample*: one 64² crop per bipoint, centred on the bipoint **midpoint**
      ((p1+p2)/2) of the prepared (prev,curr,next) image (zero-padded); skip if centre too
      near the border. Input = **min-max normalised** crop `(64,64,3)`. Target:
      `length_output = dist(p1,p2)/64`; `angle_output = [cos(2θ), sin(2θ)]` where
      θ = (p1→p2 angle in deg)/180·π mod π (the "twice-angle" trick). Aug = albumentations
      with `keypoints` (format yx), only kept if it still yields 2 points.
    - *Segmentation sample*: input = min-max normalised `(256,256,3)`; mask built on the fly
      `cv2.circle(center=(p1+p2)/2, radius=cell_radius=8)` (`type_mask=center`), `/255`→{0,1};
      `random_crop` to 256² + albumentations(image, mask). Loss = **weighted BCE**:
      `weight = mask*weight_scaling(=4) + 1`, BCE on the sigmoid prob, mean.
    - Validate each torch `Dataset` against the TF generator (same prepared sample → same
      X/Y within fp tol), like the inference parity gate.
  - **Validation gate**: a torch retrain on a fixed seed should land near the TF run's test
    metrics; minimally, the produced checkpoint must pass the existing inference parity vs
    its own framework.

## 7. Outputs → inference (closing the loop)
Write into a **new `models/{run_name}/`** folder (never `models/best/`; see HARD CONSTRAINT):
`models/{run_name}/{regression,segmentation}_checkpoints/checkpoints_set_{test}_all_but_target/best.h5`.
The trainer's checkpoint callback `filepath` is pointed there, and the driver **asserts no
`best.h5` exists at that path first** (auto-suffix the run name otherwise). Because the run
folder mirrors the `checkpoints_set_{n}_all_but_target/best.h5` layout, the inference plugin
uses a retrained model by pointing its reg/seg dirs at `models/{run_name}/...` — `models/best/`
untouched. For the PyTorch backend, also emit `.pt`/`.onnx` (via the existing
`keras_dump`/`convert`/`export`) into the same run folder so all inference backends can use it.

## 8. Risks / parades
- **Compute time** (CPU TF slow; seg U-Net heaviest) → default to subprocess + WSL/torch-GPU;
  expose a "quick" preset (few epochs) for smoke tests.
- **Memory** (1024² movies, large batches) → batch/crop already small (64/256); guard OOM.
- **Reproducibility** (augmentation RNG) → expose seed; note exact bit-repro across
  frameworks is not guaranteed (same as inference parity being ~1e-7, not bit-exact).
- **DARE2d-main untouched** → driver imports + reuses (like `_api`); no edits upstream.
- **Env** → Phase A needs the TF env (already has TF; torch also present now). Phase B data
  deps (albumentations, etc.) into the torch env.
- **GUI longevity** → subprocess + cancel; persist progress so a closed widget can re-attach.

## 9. Phasing
- **Phase A — TF reuse plugin** (low effort): single-split driver + widget + subprocess +
  progress/cancel + checkpoint wiring. Runnable on CPU now, or WSL2-GPU if available.
  Deliverable: working retraining, zero training-divergence risk.
- **Phase B — PyTorch training backend** (high effort): torch data pipeline + losses +
  trainer, GPU on native Windows, reusing the faithful torch models. Deliverable: fast
  local retraining without WSL.

## 10. Effort (order of magnitude)
| Piece | Effort |
|---|---|
| Phase A: driver + widget + subprocess/monitor + wiring | ~1–2 days |
| Phase B: torch data pipeline (reg + seg + weighted mask + aug) | ~2–3 days |
| Phase B: losses + metrics + trainer loop + validation | ~1–2 days |

## 11. Decision to take before "go"
**Which backend for v1?** (A) TF reuse on CPU/WSL — fastest to ship, slow or needs WSL for
GPU; or (B) go straight to the PyTorch GPU training port — more work, but native-Windows GPU
and consistent with the rest of this project. Recommended: **A first (de-risk UX + wiring),
then B for speed** — same pattern that worked for inference (ONNX/GPU first, torch port next).
