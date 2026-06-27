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
from magicgui.widgets import CheckBox, FileEdit, ProgressBar, PushButton
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


def _checkpoints_present():
    """True once the inference checkpoints have been downloaded."""
    return (any(_api.DEFAULT_REG_DIR.glob("checkpoints_set_*/best.h5"))
            and any(_api.DEFAULT_SEG_DIR.glob("checkpoints_set_*/best.h5")))


def _dataset_present():
    """True once the raw neuroepithelium dataset has been downloaded."""
    return any(_api.DATA_DIR.glob("set_*"))


def _add_download_section(widget, present_fn):
    """Append a 'Download DARE2D data' button (+ progress bar) that shows only when the
    data ``present_fn`` checks for is missing. Clicking downloads from Zenodo with a live
    progress bar; the button hides itself once the data is present.
    """
    btn = PushButton(text="Download DARE2D data (Zenodo, ~2 GB)")
    btn.tooltip = ("Download the DARE2D checkpoints + neuroepithelium dataset from Zenodo "
                   "(~2 GB) into this project, then this button disappears.")
    bar = ProgressBar(value=0)
    bar.min, bar.max = 0, 100
    bar.visible = False
    btn.visible = not present_fn()
    widget.append(btn)
    widget.append(bar)

    def _on_click():
        from qtpy.QtCore import QTimer

        from ._data import download_dataset

        btn.enabled = False
        bar.visible = True
        bar.label = "starting…"
        state = {"pct": 0, "msg": "downloading…"}

        def _progress(done, total):
            if total:
                state["pct"] = int(100 * done / total)

        @thread_worker
        def run():
            return download_dataset(_api.PROJECT_ROOT, progress_cb=_progress,
                                    log=lambda m: state.__setitem__("msg", m))

        # The worker runs off-thread; a GUI-thread timer marshals progress to the bar.
        timer = QTimer()
        timer.setInterval(200)
        timer.timeout.connect(lambda: (setattr(bar, "value", state["pct"]),
                                       setattr(bar, "label", state["msg"])))

        def _finish(ok, info):
            timer.stop()
            bar.visible = False
            btn.enabled = True
            if ok:
                btn.visible = False          # data now present -> hide the button
                notifications.show_info(f"DARE2D data ready under {info}")
            else:
                notifications.show_error(f"DARE2D download failed: {info}")

        worker = run()
        worker.returned.connect(lambda root: _finish(True, root))
        worker.errored.connect(lambda e: _finish(False, e))
        widget._dl_timer = timer             # keep the timer alive for the run
        worker.start()
        timer.start()

    btn.clicked.connect(_on_click)


_DIV = {}  # holds the division widget's 'save results' controls (revealed after a run)


def _add_save_section(widget):
    """Append a hidden 'save results' section to the division widget; ``_on_return`` reveals
    it once a detection run finishes (replaces the old standalone save-results widget)."""
    out = FileEdit(mode="d", label="Output folder", value=_api.PROJECT_ROOT / "output")
    out.tooltip = ("Folder to write per-frame division_position*.npy, a *_summary.csv and "
                   "(optionally) an overlay *_result.tiff into.")
    overlay = CheckBox(value=True, label="Also save overlay movie (.tiff)")
    overlay.tooltip = "Render a (T, Y, X) RGB overlay tiff with the detections drawn on the movie."
    btn = PushButton(text="Save DARE2D results")
    btn.tooltip = "Save the detections from the last run (the DARE2D Points layer) to disk."
    for w_ in (out, overlay, btn):
        w_.visible = False
        widget.append(w_)

    def _save():
        import napari.layers as nl
        viewer = napari.current_viewer()
        if viewer is None:
            return
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
            for c in pts.features.columns:
                feats[c] = np.asarray(pts.features[c])
        except Exception:
            feats = {k: np.asarray(v) for k, v in (pts.properties or {}).items()}
        from ._data import save_results
        saved = save_results(np.asarray(img.data), np.asarray(pts.data), feats,
                             out.value, image_name=str(img.name), draw=overlay.value)
        notifications.show_info(f"DARE2D: saved results → {saved}")

    btn.clicked.connect(_save)
    _DIV["save_widgets"] = (out, overlay, btn)


def _division_widget_init(widget):
    """Download-data button (when checkpoints missing) + hidden save-results section; tooltips."""
    _add_download_section(widget, _checkpoints_present)
    _add_save_section(widget)
    cb = getattr(widget, "_call_button", None)
    if cb is not None:
        cb.tooltip = ("Detect divisions in the selected stack and overlay the centres "
                      "(Points layer) and division axes (Vectors layer).")


@magic_factory(
    widget_init=_division_widget_init,
    call_button="Run DARE2D",
    image={"label": "Image layer (already open)",
           "tooltip": "Run on an Image layer already open in napari. Leave empty if you "
                      "load a movie file below instead."},
    movie={"mode": "r", "label": "…or load a movie (.tif)",
           "tooltip": "Browse for an 8-bit (T, Y, X) .tif/.tiff stack. It is loaded as an "
                      "Image layer and used as the input when you press Run."},
    backend={"choices": ["keras", "pytorch"], "label": "Inference backend",
             "tooltip": "keras = TensorFlow on CPU (default). pytorch = GPU/CUDA — same "
                        "detections, needs the .pt weights in the checkpoint folders."},
    reg_dir={"label": "Regression checkpoints", "mode": "d",
             "tooltip": "Folder of regression checkpoints "
                        "(…/checkpoints_set_N_all_but_target/best.h5 or best.pt). Point it "
                        "at a retrained run dir to use those weights."},
    seg_dir={"label": "Segmentation checkpoints", "mode": "d",
             "tooltip": "Folder of segmentation (centre-detection) checkpoints, same "
                        "layout as the regression folder."},
    model_sets={"label": "Model sets (e.g. 1-7, or 8)",
                "tooltip": "Which trained model set(s) to run: e.g. '8' for one model, "
                           "'1-7' for the 7-model ensemble. Several sets → per-frame "
                           "consensus across them."},
    frame_start={"label": "First frame",
                 "tooltip": "First frame to process (0-based)."},
    frame_end={"label": "Last frame (-1 = end)",
               "tooltip": "Last frame to process (inclusive); -1 means the final frame."},
    eps={"label": "Consensus eps (px)",
         "tooltip": "Consensus clustering radius in pixels: detections from different "
                    "models within this distance are merged into one division."},
    min_models={"label": "Min models (consensus)",
                "tooltip": "Minimum number of models that must agree (within eps) to keep "
                           "a consensus detection."},
    pbar={"visible": False, "max": 0, "label": "idle"},
)
def dare2d_widget(
    image: "napari.layers.Image",
    movie: Path = Path(""),
    backend: str = "keras",
    reg_dir: Path = _api.DEFAULT_REG_DIR,
    seg_dir: Path = _api.DEFAULT_SEG_DIR,
    model_sets: str = "8",
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
    # A movie file (if given) is loaded as an Image layer and used; otherwise run on the
    # selected Image layer -- so you can load a movie straight from this widget.
    if movie and Path(movie).is_file():
        image = viewer.add_image(_api.read_stack(movie), name=Path(movie).stem)
    if image is None:
        raise ValueError("Pick a movie file (.tif) or select an open Image layer first.")
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
        for w_ in _DIV.get("save_widgets", ()):   # reveal the save-results section
            w_.visible = True

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
_RUN = {"proc": None, "cancel": False}            # proc + cancel flag + the inline Stop button
_EPOCH_RE = re.compile(r"(?:\[epoch |Epoch )(\d+)/(\d+)")
_EXP_MAP = {"both": ["regression2d", "segmentation2d"],
            "regression": ["regression2d"], "segmentation": ["segmentation2d"]}


def _win_to_wsl(p):
    """C:\\a\\b -> /mnt/c/a/b (for invoking WSL on a Windows path)."""
    p = str(p)
    return f"/mnt/{p[0].lower()}{p[2:].replace(chr(92), '/')}"


def _retrain_widget_init(widget):
    """Add a 'Download data' button (when the dataset is missing) and a 'Stop retraining'
    button (shown only while a run is active)."""
    _add_download_section(widget, _dataset_present)
    stop = PushButton(text="Stop retraining")
    stop.visible = False
    stop.tooltip = "Cancel the running retraining (terminates the training subprocess)."

    def _stop():
        _RUN["cancel"] = True
        p = _RUN.get("proc")
        if p is not None and p.poll() is None:
            p.terminate()

    stop.clicked.connect(_stop)
    widget.append(stop)
    _RUN["stop_btn"] = stop
    cb = getattr(widget, "_call_button", None)
    if cb is not None:
        cb.tooltip = ("Launch leave-one-out retraining as a subprocess; progress shows in "
                      "the bar and the terminal.")


@magic_factory(
    widget_init=_retrain_widget_init,
    call_button="Start retraining",
    test_set={"choices": [1, 2, 3, 4, 5, 6, 7, 8], "label": "Test set (held out)",
              "tooltip": "The set held out for validation (leave-one-out); the model "
                         "trains on the others and is tested on this one."},
    train_sets={"label": "Train sets (blank = complement)",
                "tooltip": "Comma-separated sets to train on (e.g. '1,2,3'). Blank = every "
                           "set except the held-out test set."},
    model={"choices": ["both", "regression", "segmentation"],
           "tooltip": "Which stage(s) to retrain: both, just the regression head, or just "
                      "the segmentation (centre-detection) U-Net."},
    backend={"choices": ["PyTorch (GPU)", "TensorFlow (CPU)", "TensorFlow (WSL GPU)"],
             "label": "Backend",
             "tooltip": "PyTorch (GPU) = native-Windows CUDA. TensorFlow (CPU) = native "
                        "Windows (slow). TensorFlow (WSL GPU) = TF 2.12 GPU inside WSL2."},
    raw_dir={"mode": "d", "label": "Data folder (raw sets)",
             "tooltip": "Folder of raw sets (set_1…set_8), each with the movie .tif + "
                        "division_position*.npy ground truth."},
    run_name={"label": "Run name (blank = date)",
              "tooltip": "Output subfolder under models/<run>/; blank = today's date. "
                         "models/best/ is never overwritten."},
    epochs={"tooltip": "Number of training epochs per stage."},
    steps={"tooltip": "Optimizer steps per epoch."},
    crop={"tooltip": "Crop size in pixels that each frame is tiled into for training."},
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
    A **Stop retraining** button appears while a run is active. Progress/logs also
    print to the terminal.
    """
    if _RUN["proc"] is not None and _RUN["proc"].poll() is None:
        raise RuntimeError("a retraining is already running (use the Stop button first)")
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

    _stop_btn = _RUN.get("stop_btn")

    def _set_stop(visible):
        if _stop_btn is not None:
            _stop_btn.visible = visible

    worker = run()
    worker.yielded.connect(_on_yield)
    worker.returned.connect(_done)
    worker.errored.connect(lambda e: (setattr(pbar, "label", f"error: {e}"),
                                      _RUN.__setitem__("proc", None)))
    worker.started.connect(lambda: (setattr(pbar, "visible", True), _set_stop(True)))
    worker.finished.connect(lambda: _set_stop(False))
    worker.start()
    return worker
