"""Parity gate: PyTorch port vs the ONNX oracle. [torch env]

The ONNX models are validated faithful to TF at ~1e-7 (Piste 1), so they serve as
a *portable oracle inside this env* -- no TF/cross-env bridge needed. We run the
same inputs through onnxruntime (CPU) and the torch port and require small max|Δ|.

Parity is realistic, not bit-exact (torch cuDNN/MKL vs ORT differ ~1e-6..1e-4); the
Flatten-order trap (§14-B #1), if present, would instead give O(0.1-1) errors, so
this gate distinguishes "faithful" from "scrambled" with margin.

Usage (torch env, repo root):
    python dare2d-torch/verify_parity_torch.py --kind reg --sets 1-8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

import backends  # noqa: E402  (also sets up CUDA dll path, harmless on CPU)
import models_torch as M  # noqa: E402

WEIGHTS = _HERE / "weights_pt"
REG_TOL = 1e-3
SEG_TOL = 5e-3        # deep U-Net accumulates more fp drift torch vs ORT
SEG_MASK_AGREE = 0.999


def _onnx_sess(path):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def check_regression(n, onnx_dir, batch=16, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((batch, 64, 64, 3), dtype=np.float32)   # /255-normalised range

    sess = _onnx_sess(Path(onnx_dir) / f"reg_set_{n}.onnx")
    outs = sess.run(None, {sess.get_inputs()[0].name: x})
    # order outputs by last dim: 1 -> length, 2 -> angle (robust to names)
    by_dim = {o.shape[-1] if isinstance(o.shape[-1], int) else o.ndim: arr
              for o, arr in zip(sess.get_outputs(), outs)}
    o_len, o_ang = np.asarray(by_dim[1]).reshape(batch, 1), np.asarray(by_dim[2])

    model = M.Regression2dTorch()
    model.load_state_dict(torch.load(WEIGHTS / f"torch_reg_set_{n}.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
        t_len, t_ang = model(xt)
    t_len, t_ang = t_len.numpy(), t_ang.numpy()

    d_len = float(np.abs(o_len - t_len).max())
    d_ang = float(np.abs(o_ang - t_ang).max())
    ok = max(d_len, d_ang) <= REG_TOL
    print(f"  set {n}: max|d|length={d_len:.2e}  max|d|angle={d_ang:.2e}  "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def check_segmentation(n, onnx_dir, batch=2, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((batch, 256, 256, 3), dtype=np.float32)

    sess = _onnx_sess(Path(onnx_dir) / f"seg_set_{n}.onnx")
    o = np.asarray(sess.run(None, {sess.get_inputs()[0].name: x})[0])  # (B,256,256,1)

    model = M.SegmentationUnetTorch()
    model.load_state_dict(torch.load(WEIGHTS / f"torch_seg_set_{n}.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
        y = model(xt).permute(0, 2, 3, 1).numpy()        # -> (B,256,256,1)

    d = np.abs(o - y)
    agree = float(np.mean((o > 0.5) == (y > 0.5)))
    ok = d.max() <= SEG_TOL and agree >= SEG_MASK_AGREE
    print(f"  set {n}: max|d|={d.max():.2e}  mean|d|={d.mean():.2e}  "
          f"mask-agree={agree*100:.4f}%  {'OK' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=["reg", "seg"], default="reg")
    ap.add_argument("--sets", default="1-8")
    ap.add_argument("--onnx-dir", default=str(backends.DEFAULT_ONNX_DIR))
    args = ap.parse_args()

    sets = []
    for tok in args.sets.replace(" ", "").split(","):
        if "-" in tok:
            a, b = tok.split("-"); sets += list(range(int(a), int(b) + 1))
        elif tok:
            sets.append(int(tok))

    tol = REG_TOL if args.kind == "reg" else SEG_TOL
    print(f"torch <-> ONNX parity ({args.kind}), tol={tol}:")
    ok = True
    for n in sets:
        if args.kind == "reg":
            ok &= check_regression(n, args.onnx_dir)
        else:
            ok &= check_segmentation(n, args.onnx_dir)
    print("PARITY OK" if ok else "PARITY FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
