"""magicgui dock widget: run DARE2D on a loaded Image layer, overlay results.

Runs inference in a ``thread_worker`` (keeps napari responsive) and adds the
results as a Points layer (division centres) + a Vectors layer (division axes).
All heavy lifting / data mapping lives in ``_api`` (napari-free, tested headless).

Two inference backends, selectable in the widget:
  - **keras**  : the original TF/Keras models (CPU on native Windows).
  - **pytorch**: the faithful torch port in ``dare2d-torch/`` (GPU), loaded by set
    number from ``dare2d-torch/weights_pt/*.pt``. Produces the same detections.
Both feed the SAME ``infer_stack`` / consensus / layer-mapping code, so only the
model-building step differs.
"""

import sys
from collections import defaultdict
from pathlib import Path

import napari
import numpy as np
from magicgui import magic_factory
from magicgui.widgets import ProgressBar
from napari.qt.threading import thread_worker

from . import _api


def _pytorch_builder(sets):
    """Return ``build(i) -> (reg, seg)`` torch models, or raise a clear error.

    The torch port lives in the sibling ``dare2d-torch/`` folder (single home for
    both the porting scripts and the runtime backend); add it to sys.path lazily,
    inside the worker, so selecting Keras never imports torch.
    """
    dt = _api.PROJECT_ROOT / "dare2d-torch"
    if str(dt) not in sys.path:
        sys.path.insert(0, str(dt))
    try:
        import torch_backend as tb
    except Exception as e:  # torch not installed, etc.
        raise RuntimeError(
            f"PyTorch backend unavailable: {e}. Install torch into this env and "
            f"generate weights with dare2d-torch/convert_to_torch.py."
        ) from e
    missing = [n for n in sets if not (tb.WEIGHTS / f"torch_reg_set_{n}.pt").exists()]
    if missing:
        raise RuntimeError(
            f"missing torch weights for set(s) {missing} in {tb.WEIGHTS}. "
            f"Run: python dare2d-torch/convert_to_torch.py --kind reg --sets 1-8 "
            f"(and --kind seg)."
        )
    return lambda i: tb.build_hybrid_models(sets[i], seg_backend="torch")


@magic_factory(
    call_button="Run DARE2D",
    backend={"choices": ["keras", "pytorch"], "label": "Inference backend"},
    reg_dir={"label": "Regression checkpoints", "mode": "d"},
    seg_dir={"label": "Segmentation checkpoints", "mode": "d"},
    model_sets={"label": "Model sets (e.g. 1-8, or 8)"},
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
    model_sets: str = "1-8",
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
            build = _pytorch_builder(sets)
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
