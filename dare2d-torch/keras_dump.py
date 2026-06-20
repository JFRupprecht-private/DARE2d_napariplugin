"""Dump Keras weights (native layout) to .npz for the torch port. [TF env]

Reads each checkpoint through DARE2D's own Hydra build (architecture guaranteed
identical to inference) and saves raw tensors + their shapes. ``convert_to_torch.py``
(torch env) then remaps layout (HWIO->OIHW, Dense transpose, ...) and asserts the
shapes match before copying -- so any hyperparam drift is caught immediately (§13).

DARE2d-main/ is only read.

Usage (TF env, repo root):
    python dare2d-torch/keras_dump.py --kind reg --sets 1-8
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "napari-dare2d"))

import numpy as np  # noqa: E402

from napari_dare2d import _api as api  # noqa: E402

OUT_DIR = _HERE / "weights_pt"


def dump_regression(model):
    """-> dict of named arrays for Regression2dTorch (conv0..N, fc_len, fc_ang)."""
    out = {}
    ci = 0
    for layer in model.layers:
        cls = layer.__class__.__name__
        ws = layer.get_weights()
        if cls == "Conv2D":
            out[f"conv{ci}.weight"] = ws[0]   # (kh, kw, in, out) HWIO
            out[f"conv{ci}.bias"] = ws[1]
            ci += 1
        elif cls == "Dense":
            units = ws[0].shape[1]            # (in, units)
            tag = "fc_len" if units == 1 else "fc_ang"
            out[f"{tag}.weight"] = ws[0]
            out[f"{tag}.bias"] = ws[1]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=["reg", "seg"], default="reg")
    ap.add_argument("--sets", default="1-8")
    ap.add_argument("--reg-dir", default=str(api.DEFAULT_REG_DIR))
    ap.add_argument("--seg-dir", default=str(api.DEFAULT_SEG_DIR))
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    sets = api.parse_sets(args.sets)
    reg_ck, seg_ck = api.find_checkpoints(args.reg_dir, args.seg_dir, sets)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import tensorflow.keras.backend as K

    for n, rck, sck in zip(sets, reg_ck, seg_ck):
        if args.kind == "reg":
            model = api._build_model("regression2d", rck)
            arrays = dump_regression(model.model)
        else:
            model = api._build_model("segmentation2d", sck)
            # seg: dump every weighted layer verbatim (mapping happens in torch)
            arrays = {}
            for layer in model.model.layers:
                for i, w in enumerate(layer.get_weights()):
                    arrays[f"{layer.name}|{i}"] = w
        path = out_dir / f"keras_{args.kind}_set_{n}.npz"
        np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})
        shapes = {k: tuple(v.shape) for k, v in arrays.items()}
        print(f"set {n}: {path.name}  ({len(arrays)} tensors)  e.g. "
              f"{list(shapes.items())[:2]}")
        del model
        K.clear_session()
    print("done")


if __name__ == "__main__":
    main()
