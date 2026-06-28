# `dare2d/` — the DARE2D core package (TensorFlow / Keras)

This folder is the **importable core package** of DARE2D — the TensorFlow/Keras backend installed
with `pip install -e .` and imported everywhere as `import dare2d…`. It holds the model
architectures, the data pipeline, the losses/metrics, the training loop and the inference helpers.
It does **not** contain CLIs or notebooks (those live in `scripts/`, `training/`, the `*.ipynb`
files) — this package is the library they all call into.

Everything here is wired together through **Hydra**: the configs under the repo-level `config/`
tree instantiate these classes by their dotted path (`_target_: dare2d.model.…`,
`dare2d.datamodule.…`, etc.), so the same code serves training, evaluation and inference with only
config changes. The napari plugin reuses the very same functions in-process via
`napari-dare2d/napari_dare2d/_api.py`.

## What DARE2D does (the two-stage pipeline)

For each frame it stacks three consecutive frames (`prev`, `curr`, `next`) as input channels and:

1. **Segmentation (centre detection)** — a U-Net (`model/segmentation2d_cnn.py`) predicts a binary
   mask of division regions; connected-component centroids give the division **centres**.
2. **Regression** — a small CNN (`model/regression2d_cnn.py`) runs on a `crop_size` patch around
   each centre and predicts the division-axis **length** and **orientation**.

> **Angle trick.** The regression head outputs the cosine and sine of **twice** the angle (so that
> 0° and 180° coincide — a division axis is undirected) plus a length normalised to `[0, 1]`.
> `datamodule/post_processing/regression2d_pp.py:convert_values` decodes those back to
> `(length_px, angle_deg)`.

## Folder map

```
dare2d/
├── __init__.py                     # (empty) package marker
├── io.py                           # low-level image read/reshape helpers + Stdout2file
├── typing.py                       # Point / Bipoint type aliases
│
├── model/                          # network architectures (Hydra _target_ of config/model/)
│   ├── regression2d_cnn.py         #   Regression2dCNN: conv stack -> length (sigmoid) + angle (tanh, cos/sin·2θ)
│   ├── segmentation2d_cnn.py       #   Segmentation2dCNN: qubvel segmentation_models U-Net (1 class, sigmoid)
│   └── tap2d.py                    #   Tap2D temporal-attention multi-task model (experimental, see note)
│
├── datamodule/                     # data loading + augmentation -> tf.data.Dataset
│   ├── datamodule.py               #   Datamodule base: wraps generators, batches, infinite train stream
│   ├── regression2d.py             #   Regression2dDatamodule: albumentations augmentations (keypoints)
│   ├── segmentation2d.py           #   Segmentation2dDatamodule: albumentations augmentations (image+mask)
│   ├── tap2d.py                    #   Tap2Datamodule (experimental)
│   ├── generator/
│   │   ├── abstract_celldataset.py #     AbstractCellDataset (tf.keras.utils.Sequence): loads channels + bipoints
│   │   ├── _2d/cell_2dataset.py    #     Cell2Dataset: 2D base (image loading, min-max norm)
│   │   ├── _2d/regress_2dataset.py #     Regress2Dataset: crops around each division; targets = length + cos/sin·2θ
│   │   ├── _2d/seg_2dataset.py     #     Segmentation2Dataset: builds circle/ellipse/centre masks; random crop
│   │   └── tap2d.py                #     Tap2d generator (experimental)
│   ├── augmentation/
│   │   ├── elastic_histogram.py    #     ElasticHistogram albumentations DualTransform
│   │   └── illumination.py         #     Illumination albumentations DualTransform
│   ├── post_processing/
│   │   └── regression2d_pp.py      #     convert_values: model outputs -> (length_px, angle_deg)
│   └── visualization/
│       └── regression2d_visualisation.py  # project_point / display_length_angle / plot_prediction
│
├── losses/
│   ├── pixelwise_crossentropy.py   # weighted binary cross-entropy, dice + focal-dice (segmentation)
│   └── decor_loss.py               # DecorrelationLoss
│
├── evaluation/                     # Keras metrics + offline evaluators (Hydra _target_ of config/model/metrics/)
│   ├── center_metrics.py           #   Recall/Precision/F-measure metrics, extract_centers, matching
│   ├── evaluate_center2d.py        #   centre matching/visualisation across frames
│   ├── angle_degree.py             #   angle mean-absolute-error metric
│   └── length_regression.py        #   length mean-absolute-error metric
│
├── callbacks/                      # Keras training callbacks (Hydra _target_ of config/callbacks/)
│   ├── display_callback.py         #   DisplayCallback: periodic prediction previews
│   ├── regression_callback.py      #   RegressionCallback (subclass for the regression head)
│   ├── savelast_callback.py        #   SaveLastModel: keep the last weights alongside best
│   └── tap2d_callback.py           #   Tap2DCallback (experimental)
│
└── trainer/                        # training/eval orchestration (Hydra _target_ of config/trainer/)
    ├── basic_trainer.py            #   Trainer: wraps model.fit / model.evaluate + custom evaluations
    └── segmentation2d_trainer.py   #   Segmentation2DTrainer (segmentation specialisation)
```

## The data layout the generators read

The generators do **not** read raw movies. They read a per-frame, per-channel layout produced by
the preprocessor (`annotator/preprocessing/format_gastru.py`, driven by `training/prepare.py`):

```
<data_folder>/
├── previmg/<t>.tif       # frame t-1   (channel index -1)
├── currimg/<t>.tif       # frame t     (channel index  0)
├── nextimg/<t>.tif       # frame t+1   (channel index  1)
└── div_location/<t>.npy  # "bipoints": consecutive rows are the two daughter cells of one division
```

`input_channels` (e.g. `[-1, 0, 1]`) selects which of prev/curr/next to stack as image channels.
A **bipoint** is a pair of points `(p1, p2)` — the two daughter cells; the division **centre** is
their midpoint, the **axis** is the segment between them. Images are resized by the configured
`scale` and min-max normalised. `Segmentation2Dataset` turns bipoints into masks (`circle`/
`ellipse`/`center`, radius `cell_radius`); `Regress2Dataset` crops `crop_size` (default 64) around
each centre and emits `{length_output, angle_output}` targets.

## How it's instantiated (Hydra)

Nothing here is constructed directly; the `config/` tree builds it. For example
`config/model/regression2d_cnn.yaml` starts with `_target_: dare2d.model.regression2d_cnn.Regression2dCNN`
and supplies `im_size`, `channels`, `optimizer`, `losses`, `metrics`. A driver composes a config and
calls `hydra.utils.instantiate(cfg.model)` to get a ready model wrapper. Both model wrappers expose:

- `.model` — the underlying `keras.Model`,
- `.compile()` — compiles with the configured optimizer/losses/metrics,
- `.losses`, `.metrics`, `.evaluations` — the resolved dicts.

The leave-one-out drivers in `scripts/batch_train/` and `training/tf/train_split.py` build a
`datamodule` + `model` + `trainer` from configs and run `trainer.fit(...)`; inference
(`scripts/inference/`, the napari `_api`) builds a model, loads `best.h5` weights, and runs the
sliding-window segmentation (`prediction/center2d_strategy.py`) followed by per-crop regression.

## Conventions & gotchas

- **`SM_FRAMEWORK=tf.keras`** must be set *before* `segmentation_models` is imported;
  `model/segmentation2d_cnn.py` sets it at import time, and callers (scripts, `_api`) also set it
  defensively.
- **8-bit input.** The pipeline assumes 8-bit grayscale frames (the inference path histogram-
  equalises and divides by 255).
- **`scripts/` is an implicit namespace package** (no `__init__.py`); imports resolve once the repo
  root is on `sys.path` — which is why drivers add it explicitly.
- **Vestigial 3D / TAP configs.** A few entries under `config/` (`angle_representation_3d/`,
  `tap_model.yaml`, and `dare2d.losses.3d_losses…` targets) point at 3D modules that are **not**
  shipped in this 2D package, and `model/tap2d.py` is an experimental temporal-attention model not
  used by the standard 2-stage division pipeline. The supported path is `regression2d` +
  `segmentation2d`.

See the repository [`README.md`](../README.md) for installation, the Zenodo data/checkpoints, and
how to run inference and retraining. The PyTorch port of these two models lives in
[`../dare2d-torch/`](../dare2d-torch/); the napari plugin that drives this package in-process lives
in [`../napari-dare2d/`](../napari-dare2d/).
