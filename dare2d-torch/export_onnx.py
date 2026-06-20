"""Export DARE2D Keras checkpoints (.h5) to ONNX. [runs in the TF env]

Piste 1 of PYTORCH_MIGRATION.md: bridge the trained weights to onnxruntime-gpu
WITHOUT reimplementing the models. We rebuild each model through the project's
own Hydra path (``napari_dare2d._api.build_models``) so the architecture is
*byte-for-byte* the one used at inference, ``load_weights``, then trace it to
ONNX with tf2onnx.

Key choices (see PYTORCH_MIGRATION.md §14-A):
  - **dynamic batch axis**: the sliding window feeds a *variable* number of
    256x256 patches per frame (e.g. 49 @ 1024^2). The batch dim MUST be None or
    inference breaks on the patch count. Keras ``Input(shape=...)`` already has
    batch=None; we pin it explicitly in the TensorSpec to be safe.
  - **opset 17**: covers sm.Unet's Resize/UpSampling ops; supported by
    onnxruntime >=1.15.
  - NHWC is preserved (no inputs_as_nchw): the backend passes the same NHWC
    array Keras got, so ``inference_strategy``/``infer_stack`` are reused verbatim.

DARE2d-main/ is never modified; this only READS weights and WRITES .onnx files.

Usage (from the repo root, in the TF env):
    python dare2d-torch/export_onnx.py --sets 8          # one set, quick check
    python dare2d-torch/export_onnx.py --sets 1-8        # full ensemble
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Quiet + force CPU (TF 2.12 on Windows is CPU-only anyway); do this before TF import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
# make the plugin importable without installing anything new
sys.path.insert(0, str(PROJECT_ROOT / "napari-dare2d"))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import tensorflow as tf  # noqa: E402
import tf2onnx  # noqa: E402

from napari_dare2d import _api as api  # noqa: E402

OUT_DIR = _HERE / "weights_onnx"
OPSET = 17


def _export_one(keras_model, hw, out_path: Path, name: str):
    """Trace one Keras model to ONNX with a dynamic batch axis; return I/O info."""
    h, w = hw
    spec = (tf.TensorSpec((None, h, w, 3), tf.float32, name="input"),)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tf2onnx.convert.from_keras(
        keras_model, input_signature=spec, opset=OPSET, output_path=str(out_path)
    )
    # round-trip load to prove the file is valid and capture real I/O names
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ins = [(i.name, i.shape) for i in sess.get_inputs()]
    outs = [(o.name, o.shape) for o in sess.get_outputs()]
    print(f"  [{name}] -> {out_path.name}  in={ins}  out={outs}")
    return ins, outs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", default="1-8", help='model sets, e.g. "8" or "1-8"')
    ap.add_argument("--reg-dir", default=str(api.DEFAULT_REG_DIR))
    ap.add_argument("--seg-dir", default=str(api.DEFAULT_SEG_DIR))
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    sets = api.parse_sets(args.sets)
    reg_ckpts, seg_ckpts = api.find_checkpoints(args.reg_dir, args.seg_dir, sets)
    out_dir = Path(args.out)
    print(f"Exporting sets {sets} -> {out_dir} (opset {OPSET})")

    t0 = time.time()
    for n, reg_ckpt, seg_ckpt in zip(sets, reg_ckpts, seg_ckpts):
        print(f"set {n}:")
        # build BOTH wrappers through the exact Hydra path used at inference
        reg, seg = api.build_models(reg_ckpt, seg_ckpt)
        # regression: 64x64x3 crops; segmentation: 256x256x3 sliding windows
        _export_one(reg.model, (64, 64), out_dir / f"reg_set_{n}.onnx", f"reg{n}")
        _export_one(seg.model, (256, 256), out_dir / f"seg_set_{n}.onnx", f"seg{n}")
        # free the Keras graph between sets (long ensemble export hygiene)
        import tensorflow.keras.backend as K

        del reg, seg
        K.clear_session()
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
