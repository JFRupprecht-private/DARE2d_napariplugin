"""magicgui dock widget: run DARE2D on a loaded Image layer, overlay results.

Runs inference in a ``thread_worker`` (keeps napari responsive) and adds the
results as a Points layer (division centres) + a Vectors layer (division axes).
All heavy lifting / data mapping lives in ``_api`` (napari-free, tested headless).

Two inference backends, selectable in the widget:
  - **keras**  : the original TF/Keras models (CPU on native Windows).
  - **pytorch**: the faithful torch port in ``dare2d-torch/`` (GPU), loading ``best.pt``
    from the SAME checkpoint dirs as Keras. Produces the same detections.
Both feed the SAME ``infer_stack`` / consensus / layer-mapping code, so only the
model-building step differs.
"""

import datetime
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import napari
import numpy as np
from magicgui import magic_factory
from magicgui.widgets import ProgressBar
from napari.qt.threading import thread_worker
from napari.utils import notifications

from . import _api


def _pytorch_builder(sets, reg_dir, seg_dir):
    """Return ``build(i) -> (reg, seg)`` torch models, or raise a clear error.

    Loads ``best.pt`` from the SAME checkpoint dirs as the Keras backend
    (``<dir>/checkpoints_set_{n}_all_but_target/best.pt``), so a retrained run dir works
    with no rename. The torch port lives in the sibling ``dare2d-torch/`` folder; add it
    to sys.path lazily, inside the worker, so selecting Keras never imports torch.
    """
    dt = _api.PROJECT_ROOT / "dare2d-torch"
    if str(dt) not in sys.path:
        sys.path.insert(0, str(dt))
    try:
        import torch_backend as tb
    except Exception as e:  # torch not installed, etc.
        raise RuntimeError(
            f"PyTorch backend unavailable: {e}. Install torch into this env "
            f"(see requirements-torch.txt)."
        ) from e
    try:
        reg_pts, seg_pts = tb.find_torch_checkpoints(reg_dir, seg_dir, sets)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"{e}. Switch the backend to 'keras', or fetch the weights via 'Download "
            f"DARE2D data' (ships best.pt next to best.h5 in the checkpoint folders); "
            f"devs regenerate with dare2d-torch/convert_to_torch.py."
        ) from e
    return lambda i: (tb.load_torch_regression(reg_pts[i]),
                      tb.load_torch_segmentation(seg_pts[i]))


@magic_factory(
    call_button="Run DARE2D",
    backend={"choices": ["keras", "pytorch"], "label": "Inference backend"},
    reg_dir={"label": "Regression checkpoints", "mode": "d"},
    seg_dir={"label": "Segmentation checkpoints", "mode": "d"},
    model_sets={"label": "Model sets (e.g. 1-7, or 8)"},
    frame_start={"label": "First frame"},
    frame_end={"label": "Last frame (-1 = end)"},
    eps={"label": "Consensus eps (px)"},
    min_models={"label": "Min models (consensus)"},
    pbar={"visible": False, "max": 0, "label": "idle"},
)
def dare2d_widget(
    image: "napari.layers.Image",
    backend: str = "keras",
    reg_dir: Path = _api.DEFAULT_REG_DIR,
    seg_dir: Path = _api.DEFAULT_SEG_DIR,
    model_sets: str = "1-7",
    frame_start: int = 0,
    frame_end: int = -1,
    eps: float = 10.0,
    min_models: int = 6,
    pbar: ProgressBar = None,
):
    """Detect cell divisions in the selected (T, Y, X) stack and overlay them.

    One model set -> raw detections. Several sets -> per-frame consensus.
    ``backend`` picks Keras (TF) or the PyTorch port (GPU); results are the same.
    """
    viewer = napari.current_viewer()
    if image is None:
        raise ValueError("Select an Image layer first.")
    stack = np.asarray(image.data)
    if stack.ndim != 3:
        raise ValueError(f"expected a (T, Y, X) stack, got shape {stack.shape}")

    # Resolve inputs eagerly so bad paths/specs error before the worker starts.
    sets = _api.parse_sets(model_sets)
    frames = _api.resolve_frames(stack.shape[0], frame_start, frame_end)
    n_frames = stack.shape[0]
    base_name = image.name
    # Keras needs the .h5 checkpoints; the torch backend loads .pt by set number.
    reg_ckpts = seg_ckpts = None
    if backend == "keras":
        reg_ckpts, seg_ckpts = _api.find_checkpoints(reg_dir, seg_dir, sets)

    @thread_worker
    def run():
        if backend == "pytorch":
            build = _pytorch_builder(sets, reg_dir, seg_dir)
        else:
            build = lambda i: _api.build_models(reg_ckpts[i], seg_ckpts[i])  # noqa: E731

        def cleanup(reg, seg):
            del reg, seg
            if backend == "keras":
                import tensorflow.keras.backend as K
                K.clear_session()  # free graph between sets (CPU/RAM hygiene)

        if len(sets) == 1:
            reg, seg = build(0)
            per_frame = {}
            for k, i in enumerate(frames):
                per_frame.update(_api.infer_stack(stack, reg, seg, frames=[i]))
                yield (k + 1, len(frames))
            return _api.to_layer_data(per_frame, frame_base=0,
                                      name=f"{base_name} DARE2D ({backend})")

        # Ensemble: per-set detections -> consensus. Same primitives as
        # _api.run_ensemble, inlined so we can yield per-model progress.
        acc = defaultdict(list)
        for m in range(len(sets)):
            reg, seg = build(m)
            res = _api.infer_stack(stack, reg, seg, frames=frames)
            for i, dets in res.items():
                for d in dets:
                    acc[i + 1].append(
                        (m + 1, float(d["x"]), float(d["y"]),
                         float(d["angle"]), float(d["length"]))
                    )
            cleanup(reg, seg)
            yield (m + 1, len(sets))
        cons = _api.consensus(
            dict(acc), n_frames=n_frames, eps=eps,
            min_models=min_models, num_models=len(sets),
        )
        return _api.to_layer_data(cons, frame_base=1,
                                  name=f"{base_name} DARE2D consensus ({backend})")

    def _on_yield(v):
        done, total = v
        pbar.max = total
        pbar.value = done
        pbar.label = f"DARE2D {done}/{total}"

    def _on_return(layer_data):
        for data, meta, ltype in layer_data:
            viewer._add_layer_from_data(data, meta, ltype)

    worker = run()
    worker.yielded.connect(_on_yield)
    worker.returned.connect(_on_return)
    worker.started.connect(lambda: setattr(pbar, "visible", True))
    worker.finished.connect(lambda: setattr(pbar, "visible", False))
    worker.start()
    return worker


@magic_factory(
    call_button="Load annotations",
    folder={"mode": "d", "label": "Annotations folder"},
    show_links={"label": "Draw pair links"},
    point_size={"label": "Point size"},
)
def annotations_widget(
    folder: Path = _api.DEFAULT_ANNOT_DIR,
    show_links: bool = True,
    point_size: int = 12,
):
    """Overlay ground-truth annotations (paired daughter cells) from .npy files.

    Reads ``division_position*.npy`` in ``folder`` and adds a Points layer (every
    annotated cell) and a Vectors layer linking each daughter-cell pair. Fast
    (file reads only), so it runs synchronously.
    """
    viewer = napari.current_viewer()
    if viewer is None:
        raise RuntimeError("no active napari viewer")
    layer_data = _api.annotations_to_layer_data(
        folder, point_size=point_size, show_links=show_links
    )
    for data, meta, ltype in layer_data:
        viewer._add_layer_from_data(data, meta, ltype)


@magic_factory(
    call_button="Load movie",
    folder={"mode": "d", "label": "Set folder"},
)
def load_movie_widget(folder: Path = _api.DEFAULT_ANNOT_DIR):
    """Load just the set's movie .tif as an Image layer (no annotations).

    Opens the lone movie tiff in ``folder`` (defaults to set 8) as a plain Image
    layer -- ready to pick in the DARE2D division-detection widget's image dropdown.
    Fast (one file read), so it runs synchronously.
    """
    viewer = napari.current_viewer()
    if viewer is None:
        raise RuntimeError("no active napari viewer")
    for data, meta, ltype in _api.movie_to_layer_data(folder):
        viewer._add_layer_from_data(data, meta, ltype)


# ---------------------------------------------------------------------------
# Retraining widget (leave-one-out) with a backend toggle
# ---------------------------------------------------------------------------
# Spawns the retrain drivers in retrain/ as a subprocess (training is long/heavy;
# a subprocess keeps napari responsive, can use a different env/WSL, and is killable).
# Three backends -> three commands (see RETRAINING_PLAN.md):
#   PyTorch (GPU)      : python retrain/torch_train.py    (native Windows GPU)
#   TensorFlow (CPU)   : python retrain/train_split.py    (Windows, CPU)
#   TensorFlow (WSL GPU): wsl bash retrain/wsl/run_train.sh (TF 2.12 GPU in WSL2)
# Each writes models/<run>/<reg|seg>_checkpoints/...; models/best/ is never touched.

_RETRAIN_DIR = _api.PROJECT_ROOT / "retrain"
_RUN = {"proc": None, "cancel": False}            # shared with the Stop widget
_EPOCH_RE = re.compile(r"(?:\[epoch |Epoch )(\d+)/(\d+)")
_EXP_MAP = {"both": ["regression2d", "segmentation2d"],
            "regression": ["regression2d"], "segmentation": ["segmentation2d"]}


def _win_to_wsl(p):
    """C:\\a\\b -> /mnt/c/a/b (for invoking WSL on a Windows path)."""
    p = str(p)
    return f"/mnt/{p[0].lower()}{p[2:].replace(chr(92), '/')}"


@magic_factory(
    call_button="Start retraining",
    test_set={"choices": [1, 2, 3, 4, 5, 6, 7, 8], "label": "Test set (held out)"},
    train_sets={"label": "Train sets (blank = complement)"},
    model={"choices": ["both", "regression", "segmentation"]},
    backend={"choices": ["PyTorch (GPU)", "TensorFlow (CPU)", "TensorFlow (WSL GPU)"],
             "label": "Backend"},
    raw_dir={"mode": "d", "label": "Data folder (raw sets)"},
    run_name={"label": "Run name (blank = date)"},
    pbar={"visible": False, "max": 0, "label": "idle"},
)
def retrain_widget(
    test_set: int = 8,
    train_sets: str = "",
    model: str = "both",
    backend: str = "PyTorch (GPU)",
    raw_dir: Path = _api.DATA_DIR,
    run_name: str = "",
    epochs: int = 50,
    steps: int = 1000,
    crop: int = 256,
    pbar: ProgressBar = None,
):
    """Retrain DARE2D on a subset of sets, testing on the held-out ``test_set``.

    Output goes to ``models/<run>/`` (dated; never overwrites an existing best.h5).
    Use **Stop retraining** to cancel. Progress/logs also print to the terminal.
    """
    if _RUN["proc"] is not None and _RUN["proc"].poll() is None:
        raise RuntimeError("a retraining is already running (use Stop retraining first)")
    exps = _EXP_MAP[model]
    rn = run_name.strip() or datetime.date.today().isoformat()
    _RUN["cancel"] = False

    def _cmd(exp):
        common = ["--experiment", exp, "--test-set", str(test_set), "--run-name", rn,
                  "--epochs", str(epochs), "--steps", str(steps), "--crop", str(crop),
                  "--raw-root", str(raw_dir)]
        if train_sets.strip():
            common += ["--train-sets", train_sets.strip()]
        if backend.startswith("PyTorch"):
            return [sys.executable, str(_RETRAIN_DIR / "torch_train.py"), *common]
        if "WSL" in backend:
            sh = _win_to_wsl(_RETRAIN_DIR / "wsl" / "run_train.sh")
            return ["wsl", "-d", "Ubuntu", "bash", sh, *common]
        return [sys.executable, str(_RETRAIN_DIR / "train_split.py"), *common]

    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", SM_FRAMEWORK="tf.keras",
               PYTHONUNBUFFERED="1")

    @thread_worker
    def run():
        for exp in exps:
            if _RUN["cancel"]:
                break
            # decode as UTF-8 w/ replacement: training output has non-cp1252 bytes
            # (progress bars / warnings) that the Windows default codec rejects.
            proc = subprocess.Popen(_cmd(exp), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, encoding="utf-8",
                                    errors="replace", bufsize=1,
                                    env=env, cwd=str(_api.PROJECT_ROOT))
            _RUN["proc"] = proc
            for line in proc.stdout:
                m = _EPOCH_RE.search(line)
                if m:
                    yield (exp, int(m.group(1)), int(m.group(2)))
                if _RUN["cancel"]:
                    proc.terminate()
                    break
            rc = proc.wait()
            if _RUN["cancel"]:
                break
            if rc != 0:
                raise RuntimeError(f"{exp} retraining failed (exit {rc}); see terminal")
        return rn

    def _on_yield(v):
        exp, ep, tot = v
        pbar.max = tot
        pbar.value = ep
        pbar.label = f"{exp} epoch {ep}/{tot}"

    def _done(_=None):
        pbar.label = "cancelled" if _RUN["cancel"] else f"done -> models/{rn}"
        _RUN["proc"] = None

    worker = run()
    worker.yielded.connect(_on_yield)
    worker.returned.connect(_done)
    worker.errored.connect(lambda e: (setattr(pbar, "label", f"error: {e}"),
                                      _RUN.__setitem__("proc", None)))
    worker.started.connect(lambda: setattr(pbar, "visible", True))
    worker.start()
    return worker


@magic_factory(call_button="Stop retraining")
def retrain_stop_widget():
    """Cancel a running retraining (terminates the training subprocess)."""
    _RUN["cancel"] = True
    p = _RUN.get("proc")
    if p is not None and p.poll() is None:
        p.terminate()


# ---------------------------------------------------------------------------
# Download data (Zenodo) + Save results
# ---------------------------------------------------------------------------
@magic_factory(
    call_button="Download DARE2D data (Zenodo)",
    dest={"widget_type": "FileEdit", "mode": "d", "label": "Project root",
          "tooltip": "Project root to populate: checkpoints -> models/best/, dataset -> "
                     "data/neuroepithelium/ (Zenodo record 17442227, ~2 GB)."},
    pbar={"visible": False, "max": 0, "label": "idle"},
)
def dare2d_download_widget(dest: Path = _api.PROJECT_ROOT, pbar: ProgressBar = None):
    """Download the DARE2D checkpoints + neuroepithelium dataset from Zenodo (~2 GB).

    Checkpoints land in ``models/best/`` (the inference defaults) and the dataset in
    ``data/neuroepithelium/``. Runs in a worker thread; percent prints to the terminal.
    """
    from ._data import download_dataset

    state = {"pct": -5}

    def _progress(done, total):
        if total:
            pct = int(100 * done / total)
            if pct >= state["pct"] + 5:
                state["pct"] = pct
                print(f"[DARE2D] download {pct}%  ({done / 1e6:.0f}/{total / 1e6:.0f} MB)")

    @thread_worker
    def run():
        return download_dataset(dest, progress_cb=_progress,
                                log=lambda m: print(f"[DARE2D] {m}"))

    def _done(root):
        pbar.label = f"downloaded → {root}"
        notifications.show_info(f"DARE2D data ready under {root}")

    worker = run()
    worker.returned.connect(_done)
    worker.errored.connect(lambda e: (setattr(pbar, "label", "download failed"),
                                      notifications.show_error(f"DARE2D download failed: {e}")))
    worker.started.connect(lambda: setattr(pbar, "visible", True))
    worker.start()
    notifications.show_info("DARE2D: download started — progress in the terminal.")
    return worker


@magic_factory(
    call_button="Save DARE2D results",
    out_dir={"widget_type": "FileEdit", "mode": "d", "label": "Output folder",
             "tooltip": "Writes division_position*.npy, *_summary.csv and an overlay "
                        "*_result.tiff into this folder."},
    draw_overlay={"label": "Also save overlay movie (.tiff)"},
)
def save_results_widget(out_dir: Path = _api.PROJECT_ROOT / "output",
                        draw_overlay: bool = True):
    """Save the current DARE2D detections (the Points layer) to disk.

    Exports per-frame ``division_position*.npy`` ([x, y] pairs), a summary CSV, and
    (optionally) a ``*_result.tiff`` overlay 'movie'. Run **Run DARE2D** first.
    """
    import napari.layers as nl

    viewer = napari.current_viewer()
    if viewer is None:
        raise RuntimeError("no active napari viewer")
    pts = next((ly for ly in reversed(viewer.layers)
                if isinstance(ly, nl.Points) and "DARE2D" in str(ly.name)), None)
    if pts is None:
        notifications.show_warning("No DARE2D Points layer found — run DARE2D first.")
        return
    img = next((ly for ly in reversed(viewer.layers) if isinstance(ly, nl.Image)), None)
    if img is None:
        notifications.show_warning("No Image layer found to render the overlay.")
        return

    feats = {}
    try:  # napari 0.5: a features DataFrame; fall back to the older properties dict
        df = pts.features
        for c in df.columns:
            feats[c] = np.asarray(df[c])
    except Exception:
        feats = {k: np.asarray(v) for k, v in (pts.properties or {}).items()}

    from ._data import save_results
    out = save_results(np.asarray(img.data), np.asarray(pts.data), feats,
                       out_dir, image_name=str(img.name), draw=draw_overlay)
    notifications.show_info(f"DARE2D: saved results → {out}")
