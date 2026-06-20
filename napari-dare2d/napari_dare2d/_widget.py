"""magicgui dock widget: run DARE2D on a loaded Image layer, overlay results.

Runs inference in a ``thread_worker`` (keeps napari responsive) and adds the
results as a Points layer (division centres) + a Vectors layer (division axes).
All heavy lifting / data mapping lives in ``_api`` (napari-free, tested headless).
"""

from collections import defaultdict
from pathlib import Path

import napari
import numpy as np
from magicgui import magic_factory
from magicgui.widgets import ProgressBar
from napari.qt.threading import thread_worker

from . import _api


@magic_factory(
    call_button="Run DARE2D",
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
    """
    viewer = napari.current_viewer()
    if image is None:
        raise ValueError("Select an Image layer first.")
    stack = np.asarray(image.data)
    if stack.ndim != 3:
        raise ValueError(f"expected a (T, Y, X) stack, got shape {stack.shape}")

    # Resolve inputs eagerly so bad paths/specs error before the worker starts.
    sets = _api.parse_sets(model_sets)
    reg_ckpts, seg_ckpts = _api.find_checkpoints(reg_dir, seg_dir, sets)
    frames = _api.resolve_frames(stack.shape[0], frame_start, frame_end)
    n_frames = stack.shape[0]
    base_name = image.name

    @thread_worker
    def run():
        if len(reg_ckpts) == 1:
            reg, seg = _api.build_models(reg_ckpts[0], seg_ckpts[0])
            per_frame = {}
            for k, i in enumerate(frames):
                per_frame.update(_api.infer_stack(stack, reg, seg, frames=[i]))
                yield (k + 1, len(frames))
            return _api.to_layer_data(per_frame, frame_base=0, name=f"{base_name} DARE2D")

        # Ensemble: composed from the same primitives as _api.run_ensemble, but
        # inlined here so we can yield per-model progress to the GUI thread.
        acc = defaultdict(list)
        for m, (r, s) in enumerate(zip(reg_ckpts, seg_ckpts), start=1):
            reg, seg = _api.build_models(r, s)
            res = _api.infer_stack(stack, reg, seg, frames=frames)
            for i, dets in res.items():
                for d in dets:
                    acc[i + 1].append(
                        (m, float(d["x"]), float(d["y"]), float(d["angle"]), float(d["length"]))
                    )
            import tensorflow.keras.backend as K

            del reg, seg
            K.clear_session()  # free graph between sets (CPU/RAM hygiene)
            yield (m, len(reg_ckpts))
        cons = _api.consensus(
            dict(acc), n_frames=n_frames, eps=eps,
            min_models=min_models, num_models=len(reg_ckpts),
        )
        return _api.to_layer_data(cons, frame_base=1, name=f"{base_name} DARE2D consensus")

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
