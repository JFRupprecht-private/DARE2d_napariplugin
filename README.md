<div align="center">

# DARE2D — Division Axis and Region Estimation in 2D time-lapse images

<a href="https://www.tensorflow.org/"><img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white"></a>
<a href="https://keras.io/"><img alt="Keras" src="https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white"></a>
<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://napari.org"><img alt="napari" src="https://img.shields.io/badge/napari-plugin-blueviolet"></a>

</div>

---

**DARE2D** detects cell divisions in 2D time-lapse microscopy and estimates each division's
**center**, **orientation**, and **axis length**, in two stages:

1. **Segmentation (center detection)** — a U-Net that localises division centers.
2. **Regression** — estimates the division-axis orientation and length.

Robust detection uses an **8-model ensemble + consensus**. This repository contains both our initial TensorFlow/Keras framework as well as an updated,
faster **PyTorch** version, together with a **napari plugin** (`napari_dare2d`) that runs it
interactively.

> **Citation.** If you use DARE2D, please cite the preprint:
> Karpinski R., Gros A., Karnat M., Saaheelur Rahaman Q., Vanaret J., Saadaoui M., Tlili S. L.,
> Rupprecht J.-F. (2026). *DARE: Division Axis and Region Estimation from 2D and 3D Time-Lapse
> Images.* bioRxiv. https://doi.org/10.1101/2024.02.05.578987


### Inference backends (CPU & GPU)

Inference runs with either of two interchangeable backends, chosen in the napari widget's
**Inference backend** dropdown:

| Backend | Framework | Notes |
|---|---|---|
| **`keras`** (default) | TensorFlow / Keras | The original models; the only path on native-Windows TF (TF ≥2.11 has no Windows GPU support). |
| **`pytorch`** | faithful PyTorch port (`dare2d-torch/`) | Produces the **same detections** as Keras — verified on set 8 (0 px centre shift, Δangle = Δlength = 0; parity ~1e-7). Loads `.pt` weights generated once via `dare2d-torch/convert_to_torch.py`. |

## Repository contents

```
dare2d/                       # core package: datamodule, models, losses, evaluation, prediction
config/                       # Hydra configuration tree (train, batch_training, experiment, ...)
scripts/                      # CLIs: inference, batch_train (leave-one-out), postprocessing, train
annotator/                    # preprocessing (format_gastru) + annotation utilities
napari-dare2d/                # the napari plugin (in-process inference + retraining widgets)
dare2d-torch/                 # PyTorch GPU port of both models (inference) + ONNX export
retrain/                      # leave-one-out retraining drivers (TF CPU/WSL-GPU, PyTorch-GPU)
main2d.ipynb                  # inference notebook (ensemble + consensus)
Run_dare2d_Retraining.ipynb   # retraining notebook (preprocess -> leave-one-out -> checkpoints)
notebooks/                    # data analysis / training-data display
```

> **`dare2d/` vs `dare2d-torch/`.** `dare2d/` is the importable, `pip install -e .` **package**
> (the TF/Keras backend — used as `import dare2d…` and in the Hydra configs), so it keeps the bare
> name. `dare2d-torch/` is the self-contained **PyTorch** backend (models + inference + ONNX),
> imported by module name off `sys.path` — an equal-status backend, just structured as a plain
> folder rather than an installed package.

## Installation

DARE2D has **two interchangeable backends** — TensorFlow/Keras and PyTorch — installed the same
way: a shared stack (**`requirements-common.txt`**) plus one or both of **`requirements-tf.txt`**
/ **`requirements-torch.txt`** (each pulls in the shared stack via `-r`). The one hard constraint
is **numpy `<1.24`** (pinned to `1.23.5`): TensorFlow 2.12 requires it and Torch 2.6 is compatible
with it, so the pin holds for both. Use a fresh conda env so nothing is re-resolved against newer
numpy.

```bash
git clone https://github.com/qazi05/DARE2d
cd DARE2d

# 1) conda environment
conda create -n dare2d-napari python=3.10 -y
conda activate dare2d-napari

# 2) backend dependencies — each file includes the shared stack (requirements-common.txt).
#    The napari plugin offers both backends in one dropdown, so for the full plugin install both:
pip install -r requirements-tf.txt        # TensorFlow / Keras backend (CPU)
pip install -r requirements-torch.txt     # PyTorch backend (GPU / CUDA)

# 3) napari + its Qt backend — installed explicitly: pinning napari[all] in the requirements
#    does not reliably pull a Qt backend on a fresh resolve.
pip install "napari[all]"

# 4) the DARE2D core, then the plugin (no deps -> don't disturb the pins)
pip install -e .
pip install --no-build-isolation --no-deps -e ./napari-dare2d
```

> On a corporate network you may need `--trusted-host pypi.org --trusted-host files.pythonhosted.org`.
>
> **The two backends are peers.** `requirements-tf.txt` and `requirements-torch.txt` each install
> the shared `requirements-common.txt` plus their framework; the napari widget's **Inference
> backend** dropdown switches between `keras` (TF, CPU) and `pytorch` (GPU) — same detections,
> parity ~1e-7. The PyTorch `.pt` weights ship with the Zenodo data (unzipped next to each
> `best.h5`; see **Models & data**), so the `pytorch` backend uses the same checkpoint fields as
> `keras` — a retrained run dir works the same way, no rename. For a non-CUDA-12.4 machine, swap
> `cu124` for your toolkit in `requirements-torch.txt`. (Native-Windows TF is CPU-only; GPU TF
> training needs WSL2 — see the note under **Retraining**.)

## Models & data

Published on **Zenodo** ([record 17442227](https://zenodo.org/records/17442227)):
`regression_checkpoints.zip`, `segmentation_checkpoints.zip`, `neuroepithelium.zip`, and
`torch_weights.zip` (pre-converted `best.pt` for the GPU/`pytorch` backend, unzipped next to
each `best.h5`). Unzip at the repository root into this layout (kept local, not in git):

```
regression_checkpoints/checkpoints_set_{1..8}_all_but_target/best.h5
segmentation_checkpoints/checkpoints_set_{1..8}_all_but_target/best.h5
data/neuroepithelium/neuroepithelium/set_{1..8}/     # movie .tiff + division_position*.npy
```

Or click **DARE2D download data** in the plugin (Plugins → DARE2D) to fetch and place all of the
above automatically.

Input images must be **8-bit** grayscale `(T, Y, X)` `.tif` stacks.

## Inference

Inference can be run **three ways** — pick whichever fits your workflow:

**1. Terminal (CLI).** Batch every `.tif` in `input/` through the 8-model ensemble:
```bash
python scripts/all_model_inference.py
# or one image with a single model set:
python -m scripts.inference.multistage_detection2d --regression ... --segmentation ... --img ... --output ...
```

**2. Notebook.** Open `main2d.ipynb` — discovers `.tif` inputs, runs the ensemble, generates
consensus detections, and plots them.

**3. napari plugin.** Launch `napari`, then **Plugins → DARE2D division detection**. Load a `.tif`
stack, choose the **Inference backend** (`keras`/CPU or `pytorch`/GPU), point the **Regression /
Segmentation checkpoint** fields at `regression_checkpoints/` and `segmentation_checkpoints/`, set
the model sets / frame range, and **Run** — results appear as a Points layer (centers) and a
Vectors layer (axes). Then **DARE2D save results** exports them to disk — per-frame
`division_position*.npy`, a `*_summary.csv`, and an overlay `*_result.tiff` movie.

## Retraining (leave-one-out)

Train on a subset of sets and test on the complement, producing
`checkpoints_set_{n}_all_but_target/best.h5` — the exact layout inference consumes. Retraining can
be run **three ways**:

**1. Terminal (CLI).** Run the leave-one-out driver for each stage:
```bash
python -m scripts.batch_train.batch_train_eval experiment=regression2d batch_training=regression2d
python -m scripts.batch_train.batch_train_eval experiment=segmentation2d batch_training=center_detection2d
```

**2. Notebook.** Open `Run_dare2d_Retraining.ipynb` — it preprocesses each raw set into the
per-frame layout the generators read, then runs the same leave-one-out training for both stages.

**3. napari plugin.** Launch `napari`, then **Plugins → DARE2D retraining**. Pick the **Test set**
(held out), optional **Train sets** (blank = the rest), **Model** (both / regression / segmentation)
and **Backend** (PyTorch GPU, TensorFlow CPU, or TensorFlow WSL GPU), then **Start retraining** — a
progress bar tracks the epochs and an inline **Stop retraining** button cancels it. Output lands in
`models/<run>/…` (dated; `models/best/` is never overwritten). If the dataset isn't present yet, a
**Download data** button appears in the widget first.

The backends live in `retrain/` (native-Windows TF is CPU-only): a WSL2 TF-GPU path and a
native-Windows PyTorch-GPU backend (`retrain/torch_train.py`), wrapped by both the notebook and the
napari widget above.

> **Windows / WSL note.** Native-Windows TensorFlow 2.12 is **CPU-only** (TF ≥2.11 has no
> Windows GPU support), so on Windows **TF retraining runs on CPU**. Training TensorFlow **on
> the GPU** is only possible through the **TensorFlow (WSL GPU)** backend, which **requires
> WSL2** (with a CUDA-enabled `dare2d-train` env inside it) — without WSL2 that option will not
> work. For native-Windows **GPU** training use the **PyTorch (GPU)** backend instead; the
> **TensorFlow (CPU)** backend also needs no WSL. (Inference never needs WSL: `keras` runs on
> CPU and `pytorch` on GPU.)

## Checks

```bash
python napari-dare2d/verify_layers.py   # fast: geometry + napari mapping (no models)
python napari-dare2d/verify_api.py      # real: builds set-8 models, runs inference (needs checkpoints)
```

## License & attribution

MIT — see [LICENSE](LICENSE). This work was granted access to the HPC resources of IDRIS under the
allocation AD010314339 made by GENCI.
