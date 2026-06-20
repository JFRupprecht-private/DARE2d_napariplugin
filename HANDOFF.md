# Handoff — napari plugin for DARE2D

_Condensed brief. Date: 2026-06-17. Scope constraint: work **only** inside `DARE2Dnapariplugin/`._

## Goal
Build a **napari plugin** that lets a user:
1. **Run the DARE2D Python code** (TensorFlow inference) on an image sequence loaded in napari.
2. **Overlay the results** (division centers, orientation angles, axis lengths) on that same sequence.

DARE2D itself does **not** use napari for its core pipeline — it's a TF/Keras codebase analyzing `.tif` stacks `(T, Y, X)` and producing division detections via an 8-model ensemble + consensus post-processing.

## Repo facts (verified)
- Core code lives in `DARE2d-main/dare2d/` (TF 2.12 / Keras 2.12, Hydra configs).
- napari is referenced **only** in `DARE2d-main/annotator/` (a GT annotation tool), not in inference.
- Inference CLI entry: `scripts/inference/multistage_detection2d.py` (takes `--regression`, `--segmentation` `.h5`, `--img`, `--output`).
- Consensus/post-processing: `scripts/postprocessing/main.py` → produces positions, angles, axis lengths, CSV summary.
- Input: TIFF stack `(T, Y, X)`, 8/16-bit grayscale. Output: `division_position*.npy` per frame + `result.tiff` + consensus CSV.
- `tensorflow-graphics` and `tensorflow-probability` (in `requirements.txt`) are **imported nowhere** → safe to drop. These are the two worst to install on Windows.

## Decisions made
- **Conda env created**: `napari-env-for-DARE2D-claude` at `C:\Users\ruppr\.conda\envs\napari-env-for-DARE2D-claude`, **Python 3.10.20** + pip. (README suggests 3.9; TF 2.12 supports 3.8–3.11, napari prefers ≥3.9, 3.10 kept — user confirmed: keep the existing env, do NOT recreate.)
- **GPU decision (2026-06-17)**: **CPU for now**. TF 2.12 has no GPU on native Windows; code stays GPU-ready (switch to WSL2/Linux later → GPU with no code change). Do not chase native-Windows GPU.

## ENV FULLY INSTALLED (2026-06-17) — `pip check` clean
All deps installed into `napari-env-for-DARE2D-claude`. **numpy 1.23.5 is the pivot** — TF 2.12 pins `numpy<1.24`, so the WHOLE stack is frozen to the 2023 generation. Key versions:
- numpy **1.23.5**, tensorflow/keras **2.12.0** (CPU), napari **0.4.18** (last napari working with numpy 1.23; napari ≥0.7 needs numpy≥2 → conflict), magicgui 0.7.3, npe2 0.7.9.
- scikit-image 0.21, scikit-learn 1.3.2, opencv-python 4.11.0.86, albumentations 1.3.1, segmentation-models 1.0.1, zarr 2.16.1, pandas 2.0.3, matplotlib 3.8.4, numba 0.57.1, vispy 0.12.2, pydantic 1.10.26.
- dare2d 1.0 installed `-e`.
- **GOTCHA — do not "upgrade" napari or numpy**: any newer napari/numpy reinstalls numpy≥2 and breaks TF 2.12. If pip pulls numpy 2.x, re-pin `numpy==1.23.5` and downgrade whatever pulled it.
- **segmentation_models** only imports cleanly with `os.environ['SM_FRAMEWORK']='tf.keras'` set BEFORE import (DARE2D's model modules already do this).
- Install used `--trusted-host pypi.org --trusted-host files.pythonhosted.org` (corporate CA / SSL). conda = `C:\ProgramData\miniconda3\Scripts\conda.exe`.
- **Dropped** (unused / out of scope): tensorflow-graphics, tensorflow-probability, augmend (imported nowhere), mlflow, hydra-optuna-sweeper, gdown. hdbscan & moviepy are lazy/optional in postprocessing — add only if needed.
- Verified smoke test passes: napari + full DARE2D inference/postprocessing import chain imports OK.
- **Conda toolchain**: system `conda` not on PATH. Working conda is `C:\ProgramData\miniconda3\Scripts\conda.exe` (v24.4.0). The older `C:\ProgramData\Anaconda3` conda (v4.12) fails with `HTTP 000`.
- **SSL workaround**: corporate/network CA causes `CERTIFICATE_VERIFY_FAILED`. Set `conda config --set ssl_verify false` on the miniconda3 conda to make env creation work. (pip may need an equivalent: `--trusted-host pypi.org --trusted-host files.pythonhosted.org`.)
- **Planned install set** (pip, in two stages), trimmed from `requirements.txt`:
  - Core inference: `tensorflow==2.12.0`, `keras==2.12.0`, `numpy`, `scipy`, `scikit-image`, `scikit-learn`, `tifffile`, `opencv-python`, `pandas`, `matplotlib`, `tqdm`, `rich`, `omegaconf`, `hydra-core`, `segmentation-models`, `albumentations`, `greenlet==2.0.2`.
  - napari/plugin: `napari[all]`, `magicgui`, `npe2`.
  - `pip install -e .` from `DARE2d-main/`.
  - **Dropped**: `tensorflow-graphics`, `tensorflow-probability` (unused); `mlflow`, `hydra-optuna-sweeper` (training/HPO, out of scope); `gdown` (model download only). Add back if needed.

## Planned plugin architecture (not yet built)
- New package: `DARE2Dnapariplugin/napari-dare2d/` (separate from `DARE2d-main/`, which it imports).
- **npe2** plugin: `napari.yaml` manifest + `pyproject.toml`, installed `-e`.
- Dock widget (magicgui):
  - Inputs: the loaded `Image` layer + checkpoints folder (regression + segmentation).
  - "Run DARE2D" button → wraps existing inference + consensus code, runs in a `thread_worker` (don't freeze UI).
  - Outputs as napari layers over the sequence: `Points` (centers per T), `Vectors` (axes from angle+length), optionally `Labels`/`Shapes` for the result map.

## Open questions (blocking before implementation)
1. **Env scope**: full TF inference stack now, or napari + plugin skeleton first and wire TF later?
2. **GPU vs CPU**: TF 2.12 on native Windows is **CPU-only** (GPU needs WSL2). Confirm CPU is acceptable, or go WSL2.
3. **Python**: keep 3.10, or recreate env at 3.9 to match DARE2D's reference setup?

## Next steps
1. ~~Resolve the 3 open questions.~~ DONE (full stack now, CPU, keep existing 3.10 env).
2. ~~Install the chosen dependency set.~~ DONE — env fully installed, `pip check` clean (see above).
3. ~~Verify inference can be called as an **API**.~~ DONE — see "STEP 3 DONE" below.
4. ~~Nail down output formats → map to `Points`/`Vectors`.~~ DONE — see "STEP 4 DONE" below.
5. ~~Scaffold the npe2 plugin + magicgui widget.~~ DONE — see "STEP 5 DONE" below.

## STEP 3 DONE (2026-06-17) — in-process API verified on real data
**Assets found on disk** (root = `DARE2Dnapariplugin/`): `regression_checkpoints/checkpoints_set_{1..8}_all_but_target/best.h5` (×8) + `segmentation_checkpoints/...` (×8) — exact layout the scripts expect. Real input stack `set_8/siractinE2_..._celldivisionlevel.tiff` = **(56,1024,1024) uint8**. Old-format reference detections `set_8/division_position{1..55}.npy`.

**New package `napari-dare2d/`** (imports DARE2d-main, leaves it untouched):
- `_dare2d_api.py` — the in-process API. Functions:
  - `build_models(reg_ckpt, seg_ckpt) -> (reg_wrapper, seg_wrapper)` — Hydra build via `initialize_config_dir` (NOT the script's relative `initialize`) + `GlobalHydra.clear()` around each build (singleton, must be re-entrant in a long napari session). `.h5` = weights-only (`load_weights`).
  - `infer_stack(stack, reg, seg, frames=None, progress_cb=None) -> {frame0: [{x,y,angle,length}]}` — refactor of `multistage_detection2d.main` minus disk I/O / viz. Reuses `inference_strategy`, `crop_img_from_center` (imported from the script via namespace pkg) + `extract_centers`, `convert_values` (from installed `dare2d`).
  - `run_ensemble(stack, reg_ckpts[], seg_ckpts[], frames=None) -> {frame1based: [(model_id,x,y,angle,length)]}` — loops the 8 sets, `K.clear_session()` between them.
  - `consensus(all_dets, n_frames, eps=10, min_models=6, num_models=8, angle_mode="auto") -> {frame1based: [consensus_dict]}` — in-memory; reuses `aggregate_cluster_pick_signed`, `cluster_hdbscan`, `detect_angle_units_and_convert`; drops the wedge/halo drawing + TIFF/CSV + temporal dedup.
- `verify_api.py` — the runnable check (CLAUDE.md). HARD-asserts the API contract; SOFT-reports reference alignment. **Result: OK.** Run: `<env>/python.exe napari-dare2d/verify_api.py`.

**Verified facts / decisions for step 4:**
- `segmentation_models` REQUIRED, confirmed (builds `sm.Unet(resnet18, encoder_weights=None)` → 100% local, no download).
- **Detection coordinate convention** (matches the *current* script): `x = cv2 contour cx = COLUMN`, `y = cy = ROW`. So a napari 2D `Points` layer needs `(row, col) = (det["y"], det["x"])`. The old `set_8` reference `.npy` use the OPPOSITE order: comparing as-is gives 0% match / ~300-400px median, **x/y-swapped drops to ~18px median** → swap confirmed. Checked on mid-stack frames [27,28,29] too (not just frame 1): same result, so it is NOT an early-frame artifact. Residual (~18px median, ~half as many predictions as the reference) is dominated by reference points with no nearby prediction — i.e. we detect a subset of an OLDER, more permissive pipeline's output (reference format `[x,y,2]`, which the current consensus loader would even skip since it only reads dicts). Treat that reference as a loose anchor only; the API faithfully reproduces the current set_8 checkpoint + current script logic.
- Angle: degrees, signed, normalized to (-90, 90]. Length: pixels, in [0, crop_size=64]. `convert_values` returns `(length_px, angle_deg)`.
- Output dict per detection: `{x:int, y:int, angle:float(deg), length:float(px)}`. Consensus dict adds `angle_std_deg, length_std, pos_std, n_models, models, support_fraction, dominant_model`.
- Perf (CPU): 3 frames @1024² ≈ a couple minutes (49 patches/frame, resnet18 U-Net). Full 8-model × 56-frame ensemble will be heavy → widget must run in `thread_worker` with progress + allow a model-subset / frame-range.
- **GOTCHA**: `scripts/` has no `__init__.py` — importable only as an implicit namespace package after putting `DARE2d-main/` on `sys.path` (the shim does this). hdbscan absent → consensus falls back to sklearn DBSCAN (fine).

## STEP 4 DONE (2026-06-18) — output → napari layer mapping
Added to `_dare2d_api.py` (pure numpy, no napari import needed — keeps it testable without a GUI):
- `detections_to_points(per_frame, frame_base=0) -> (points (N,3) [t,y,x], properties)` — properties carries angle, length (+ consensus's angle_std_deg, length_std, pos_std, n_models, support_fraction when present), for colouring/sizing.
- `detections_to_vectors(per_frame, frame_base=0) -> vectors (N,2,3)` — `v[:,0]`=origin at one rod end, `v[:,1]`=full direction `(0, cos(angle)·L, sin(angle)·L)`; with Vectors `length=1` the rod is centred on the detection.
- `to_layer_data(per_frame, frame_base=0, name=...) -> [LayerDataTuple]` — one Points + one Vectors tuple, ready to return from a magicgui widget. **Points meta uses `edge_color` (napari 0.4.18 name; it's `border_color` only in napari ≥0.5).**
- **`frame_base`**: pass `0` for `infer_stack` output (0-based keys), `1` for `consensus` output (1-based keys). napari t = key − frame_base.
- Geometry matches the script's `project_point`: in (row,col), direction = `(cos(angle), sin(angle))`; verified angle 0 → +row, angle 90 → +col, rod centred, |rod| = length.
- Check: `napari-dare2d/verify_layers.py` — fast (no models), asserts geometry + conventions + headless napari 0.4.18 acceptance of the LayerDataTuples. **Result: OK.**

## STEP 5 DONE (2026-06-18) — npe2 plugin + magicgui widget
Package layout under `napari-dare2d/`:
```
napari-dare2d/
  pyproject.toml            # dependencies = [] ON PURPOSE (see below)
  napari_dare2d/
    __init__.py             # import-light (no _widget import -> no Qt when importing _api)
    napari.yaml             # npe2 manifest: widget "DARE2D division detection"
    _api.py                 # the step 3/4 in-process API (moved here; was _dare2d_api.py)
    _widget.py              # magic_factory dock widget
  verify_api.py             # real check (imports napari_dare2d._api)
  verify_layers.py          # fast check
```
- Installed editable: `<env>/python.exe -m pip install --no-build-isolation --no-deps -e napari-dare2d`. **`--no-deps` + `dependencies = []`** are critical: a normal install would re-resolve and pull numpy>=2, breaking TF 2.12. numpy stayed 1.23.5 (verified).
- **Widget** (`napari_dare2d._widget:dare2d_widget`, `@magic_factory`): inputs = Image layer, regression/segmentation checkpoint dirs (default = the project's `*_checkpoints/`), `model_sets` ("1-8" / "1,3,5" / "8"), frame range, consensus `eps` + `min_models`, a `ProgressBar`. Runs in a `napari.qt.threading.thread_worker` **generator** that yields progress (per-frame for a single set, per-model for the ensemble) → updates the bar on the GUI thread. 1 model set → raw detections; ≥2 → consensus. Adds results via `viewer._add_layer_from_data(*to_layer_data(...))`.
- `_api` gained input helpers (pure, no TF): `parse_sets`, `find_checkpoints` (expects `checkpoints_set_{n}_all_but_target/best.h5`), `resolve_frames`, plus `DEFAULT_REG_DIR`/`DEFAULT_SEG_DIR` (derived from `__file__` via `_HERE.parents[1]` = the `DARE2Dnapariplugin/` root).
- `_api.run_ensemble` is kept as the **headless** batch equivalent; the widget inlines the same primitives only to yield per-model progress.
- **Verified**: npe2 manifest validates + widget discoverable; widget builds under a headless napari viewer (all fields present, defaults resolve to the real checkpoint dirs). Qt font/OpenGL warnings under `QT_QPA_PLATFORM=offscreen` are harmless.
- **VALIDATED in a real napari session (2026-06-18)**: drove the widget on the set-8 stack, `model_sets=8`, frames 20–22. Worker ran (~9 s), created `set8 DARE2D centers` (Points) + `set8 DARE2D axes` (Vectors). Geometry confirmed correct (detections sit on cells, axes centred on them) via an independent matplotlib overlay render. Default `point_size` bumped 12→24 (+edge_width 2→3) so the overlay is visible at fit-zoom.
- **Screenshot note**: napari canvas screenshots only work from a VISIBLE window; under `QT_QPA_PLATFORM=offscreen` there's no GL context and `viewer.screenshot()` raises (`QImg2array` NoneType). Layer creation / widget build still work offscreen.
- **Still open (follow-ups)**: full CPU ensemble (8×56 frames @1024²) is slow → keep using `model_sets` + frame-range. `infer_stack` requires 8-bit input (raises on uint16) — a 16-bit rescale option is the likely next feature. Consider colouring Points by angle/length (properties are already attached).
