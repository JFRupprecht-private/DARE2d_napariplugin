# DARE2D napari plugin

A [napari](https://napari.org) plugin that runs **DARE2D** cell-division detection
(TensorFlow inference) on an image sequence loaded in napari, and overlays the
results — division **centres** (Points) and **orientation axes** (Vectors) — on
that same sequence.

DARE2D itself is a TF/Keras codebase (`DARE2d-main/`) that analyses `.tif` stacks
`(T, Y, X)` with an 8-model ensemble + consensus. This plugin wraps its inference
**in process** (no subprocess/CLI, no disk round-trip) behind a small napari-free
API, plus a magicgui dock widget.

## Credits & citation

This plugin wraps **DARE2D**, developed by **Romain Karpinski, Alice Gros,
Marc Karnat, Qazi Saaheelur Rahaman, Jules Vanaret, Mehdi Saadaoui, Sham L. Tlili
and Jean-François Rupprecht**.

If you use this plugin or DARE2D in your research, **please cite the preprint**:

> Karpinski R., Gros A., Karnat M., Saaheelur Rahaman Q., Vanaret J., Saadaoui M.,
> Tlili S. L., Rupprecht J.-F. (2026). *DARE: Division Axis and Region Estimation
> from 2D and 3D Time-Lapse Images.* bioRxiv.
> https://doi.org/10.1101/2024.02.05.578987

## Repository layout

```
DARE2d-main/                 # the wrapped DARE2D code (TF/Keras, Hydra configs)
napari-dare2d/               # the plugin (this project)
  napari_dare2d/
    _api.py                  # in-process inference + consensus + napari mapping
    _widget.py               # magicgui dock widget
    napari.yaml              # npe2 manifest
  verify_api.py              # real check (needs checkpoints + data)
  verify_layers.py           # fast check (geometry, no models)
  pyproject.toml
regression_checkpoints/      # NOT in git — see "Model weights & data" below
segmentation_checkpoints/    # NOT in git
set_8/                       # NOT in git — example stack + reference detections
HANDOFF.md                   # full build log / decisions / gotchas
FOR_DARE3D.md                # notes for porting this to the 3D pipeline
PYTORCH_MIGRATION.md         # plan to move inference to ONNX(GPU) then PyTorch
```

## Model weights & data (not in git)

The trained checkpoints and example data are **excluded** from the repo (the
segmentation `best.h5` files exceed GitHub's 100 MB/file limit). Place them back
in this exact layout before running:

```
regression_checkpoints/checkpoints_set_{1..8}_all_but_target/best.h5
segmentation_checkpoints/checkpoints_set_{1..8}_all_but_target/best.h5
set_8/<your_stack>.tif                      # an 8-bit (T, Y, X) stack, optional
```

Input images must be **8-bit** grayscale `(T, Y, X)` `.tif` stacks (16-bit is not
yet supported).

## Environment

DARE2D pins **TensorFlow 2.12**, which forces an older, tightly-coupled stack —
**numpy 1.23.5** (TF needs `<1.24`), **napari 0.4.18** (last napari that tolerates
numpy 1.23). Inference runs on **CPU** on native Windows (TF ≥2.11 has no Windows
GPU; that's what `PYTORCH_MIGRATION.md` addresses).

```bash
# Python 3.10 conda env (an env named `napari-env-for-DARE2D-claude` is already set up)
conda create -n dare2d-napari python=3.10 -y
conda activate dare2d-napari

# Core inference + napari stack (see HANDOFF.md for the full pinned list / SSL flags)
pip install "tensorflow==2.12.0" "keras==2.12.0" "numpy==1.23.5" \
    scipy "scikit-image<0.22" "scikit-learn<1.4" tifffile "opencv-python==4.11.0.86" \
    "pandas<2.1" "matplotlib<3.9" tqdm rich omegaconf hydra-core hydra-colorlog click \
    imageio "albumentations<1.4" segmentation-models "greenlet==2.0.2" \
    "napari[all]==0.4.18" "magicgui<0.8" npe2

# Install the wrapped DARE2D code, then this plugin (no deps -> don't disturb the pins)
pip install -e ./DARE2d-main
pip install --no-build-isolation --no-deps -e ./napari-dare2d
```

> On a corporate network you may need `--trusted-host pypi.org --trusted-host files.pythonhosted.org`.

## Usage

```bash
conda activate dare2d-napari
napari
```

In napari: **Plugins → DARE2D division detection**. Then:
1. Load a `.tif` stack (drag-and-drop) and select it as the **Image** layer.
2. Point **Regression / Segmentation checkpoints** at the two folders above
   (the defaults already point there).
3. Set **Model sets** (`8` = one set, fast; `1-8` = full ensemble + consensus),
   the frame range, and consensus `eps` / `min_models`.
4. **Run DARE2D** — inference runs in a background thread with a progress bar;
   results appear as a Points layer (centres) and a Vectors layer (axes).

> The full 8-model ensemble over a large stack is slow on CPU — start with one
> model set and a small frame range.

## Checks

```bash
python napari-dare2d/verify_layers.py   # fast: geometry + napari mapping (no models)
python napari-dare2d/verify_api.py      # real: builds set-8 models, runs inference
```

## More

- `HANDOFF.md` — how the env/plugin were built, every decision and gotcha.
- `PYTORCH_MIGRATION.md` — plan to get GPU (ONNX first, then a PyTorch port).
