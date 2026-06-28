# `napari-dare2d/` — the DARE2D napari plugin

This folder is the **napari plugin** (`napari_dare2d`) that runs DARE2D interactively: load a movie,
detect divisions, overlay the centres + axes, inspect ground-truth annotations, and even launch
retraining — all from the napari GUI. It is a thin, in-process layer over the rest of the repo: it
**reuses** the core `dare2d/` (TF/Keras) and `dare2d-torch/` (PyTorch) code rather than
reimplementing inference, so napari results are identical to the CLI/notebook.

It is a separate installable package (`pip install -e ./napari-dare2d`) discovered by napari through
the **npe2 manifest** (`napari.yaml`).

## Folder map

```
napari-dare2d/
├── pyproject.toml                  # packaging; dependencies intentionally EMPTY (see "Install")
├── napari_dare2d/
│   ├── __init__.py                 # import-light; sets KMP_DUPLICATE_LIB_OK so TF+Torch coexist
│   ├── napari.yaml                 # npe2 manifest: registers the 3 widgets below
│   ├── _api.py                     # napari-free in-process API over the DARE2D pipeline (the engine)
│   ├── _widget.py                  # the magicgui dock widgets (the GUI)
│   └── _data.py                    # Zenodo download + "save results" (stdlib-only)
├── verify_layers.py                # fast check: geometry + napari layer mapping (no models)
└── verify_api.py                   # real check: builds set-8 models, runs inference + consensus
```

## The three widgets (registered in `napari.yaml`)

| Widget (Plugins → DARE2D …) | Function | What it does |
|---|---|---|
| **DARE2D division detection** | `_widget.dare2d_widget` | Runs inference on an open Image layer (or a movie you browse to), overlays a **Points** layer (centres) + **Vectors** layer (axes). Picks the **keras**/CPU or **pytorch**/GPU backend, the model set(s), frame range and consensus parameters. After a run it reveals an inline **save results** section and (if the data is missing) a **Download data** button. |
| **DARE2D annotations viewer** | `_widget.annotations_widget` | Loads ground-truth `division_position*.npy` from a set folder and overlays the paired daughter cells (Points) + pair links (Vectors). |
| **DARE2D retraining** | `_widget.retrain_widget` | Launches leave-one-out retraining as a subprocess (PyTorch GPU / TF CPU / TF WSL-GPU), with an epoch progress bar, an inline **Stop** button, and a **Download data** button when the dataset is absent. Writes to `models/<run>/…`; never touches `models/best/`. |

All heavy work runs in a `thread_worker` so the napari UI stays responsive; retraining runs in a
killable subprocess.

## `_api.py` — the engine (napari-free, headless-testable)

Pure functions that wrap the repo's own inference/consensus code, with no Qt/napari import, so they
can be unit-tested headlessly (see `verify_*.py`) and reused outside napari:

- **Model building / inference** — `build_models` (Hydra-instantiate the two TF models from
  `best.h5`), `infer_stack` (two-stage detection over a `(T,Y,X)` stack), `run_ensemble`
  (every model set), `consensus` (HDBSCAN/DBSCAN clustering + aggregation across models).
- **Layer mapping** — `detections_to_points` / `detections_to_vectors` / `to_layer_data`
  turn `{frame: [{x,y,angle,length}]}` into napari `LayerDataTuple`s.
- **Annotations** — `annotation_pairs` / `annotations_to_layer_data` read `division_position*.npy`.
- **Helpers** — `read_stack`, `find_checkpoints`, `parse_sets`, `resolve_frames`, and the default
  paths `MODELS_DIR`, `DEFAULT_REG_DIR` / `DEFAULT_SEG_DIR` (= `models/best/…`), `DATA_DIR`,
  `DEFAULT_ANNOT_DIR` (= set 8's ground truth).

### Coordinate conventions (locked — don't change without updating `verify_layers.py`)

- A detection's `x` = **column**, `y` = **row**; a napari point over a `(T,Y,X)` stack is
  therefore `(t, y, x)`.
- A division axis of angle θ and length L has direction `(0, cos θ·L, sin θ·L)` in `(t, row, col)`,
  drawn centred on the detection (a Vectors layer with `length=1`).
- Ground-truth `division_position{n}.npy` rows are `[row, col, frame]` (= `[y, x, frame]`, the order
  the training pipeline reads); `frame` is **1-based**, so napari `t = frame − 1`. Consecutive rows
  are the two daughter cells of one division.

## Backends

Both backends feed the *same* `infer_stack` / consensus / layer-mapping code; only the model-building
step differs:

- **`keras`** — the TF/Keras models from `dare2d/`, loaded from `best.h5` (CPU on native Windows).
- **`pytorch`** — the faithful port in `dare2d-torch/`, loaded from `best.pt` in the *same*
  checkpoint folders (GPU). Produces identical detections (verified: 0 px centre shift, 0 angle/
  length difference). Selecting `keras` never imports torch and vice-versa (lazy import).

## `_data.py` — download & save

- `download_dataset(root, …)` — fetches the Zenodo record (checkpoints + dataset + torch weights)
  with stdlib `urllib`/`zipfile` (no extra dependency), caches the zips under `_zenodo_cache/`, and
  extracts into the project layout (`models/best/…`, `data/neuroepithelium/…`). Extraction **only
  adds missing files** — it never overwrites, so a populated `models/` (e.g. retrained checkpoints)
  is left intact. A progress bar in the widget is driven off this via a `QTimer`.
- `save_results(image, points, features, out_dir, …)` — writes per-frame `division_position*.npy`
  (`[x, y]` pairs), a `*_summary.csv` (frame, x, y, angle, length, …) and an optional overlay
  `*_result.tiff` movie.

## Install & packaging

`pyproject.toml` declares **no dependencies on purpose**. The plugin runs inside the carefully
version-pinned env (`napari-env-for-DARE2D`, numpy `1.23.5`); letting `pip` re-resolve deps here
would pull a newer numpy and break TensorFlow 2.12. Install it without touching the pins:

```bash
pip install --no-build-isolation --no-deps -e ./napari-dare2d
```

The `_api` adapts to the installed napari version (the Points outline kwarg changed from
`edge_color` in 0.4.x to `border_color` in ≥0.5), so the same code works across napari versions.

## Checks

```bash
python napari-dare2d/verify_layers.py   # fast: geometry + napari LayerDataTuple mapping (no models)
python napari-dare2d/verify_api.py      # real: builds set-8 models, runs inference + consensus
```

`verify_layers.py` asserts the coordinate conventions above and that napari accepts the produced
layers; `verify_api.py` builds the set-8 models from `models/best/` and runs the full pipeline on the
set-8 stack (needs the Zenodo checkpoints + data — fetch them via the widget's **Download data**
button).

See the repository [`README.md`](../README.md) for the end-to-end install, inference and retraining
walkthroughs, and [`../dare2d/`](../dare2d/) for the core package this plugin wraps.
