"""Run a retrained (torch) model on a movie and show the overlay in napari. [TF env]

Loads the reg + seg .pt from a run folder, runs the full inference pipeline on the
movie, and opens napari with the movie + detection layers (centres + division axes).

Usage:
    python retrain/show_result.py --run models/2026-06-21 --test-set 8 \
        --movie set_8/<movie>.tiff
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "napari-dare2d"))
sys.path.insert(0, str(ROOT / "dare2d-torch"))

from skimage import io  # noqa: E402
import napari  # noqa: E402
from napari_dare2d import _api as api  # noqa: E402
import torch_backend as tb  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run folder, e.g. models/2026-06-21")
    ap.add_argument("--test-set", default="8")
    ap.add_argument("--movie", required=True)
    args = ap.parse_args()

    run = Path(args.run)
    tag = f"checkpoints_set_{args.test_set}_all_but_target"
    reg = tb.load_torch_regression(run / "regression_checkpoints" / tag / "best.pt")
    seg = tb.load_torch_segmentation(run / "segmentation_checkpoints" / tag / "best.pt")

    stack = io.imread(args.movie)
    print(f"inferring {stack.shape[0]} frames on {reg.device} (reg) / {seg.device} (seg) ...")
    res = api.infer_stack(stack, reg, seg)
    ndet = sum(len(v) for v in res.values())
    print(f"{ndet} divisions over {len(res)} frames")

    layer_data = api.to_layer_data(res, frame_base=0, name=f"{run.name} set_{args.test_set}")
    v = napari.Viewer()
    v.add_image(stack, name=Path(args.movie).stem)
    for data, meta, ltype in layer_data:
        v._add_layer_from_data(data, meta, ltype)
    print("napari ready")
    napari.run()


if __name__ == "__main__":
    main()
