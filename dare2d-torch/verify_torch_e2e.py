"""End-to-end gate: hybrid (torch reg + ONNX seg) vs the ONNX oracle. [torch env]

P2.3. Runs the *real* ``api.infer_stack`` twice over the same frames:
  - oracle : ONNX reg + ONNX seg  (== TF, validated at ~1e-7 in Piste 1)
  - hybrid : torch reg + ONNX seg  (the Piste 2 deliverable)
Segmentation is identical both ways, so detection *centres* must match exactly;
only the regression (angle/length) can differ, and only at fp level.

Usage (torch env, repo root):
    python dare2d-torch/verify_torch_e2e.py --set 8 --frames 24,26,28,30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(PROJECT_ROOT / "napari-dare2d"))

import _tf_stub  # noqa: F401,E402  installs fake tensorflow before _api
import numpy as np  # noqa: E402
from skimage import io  # noqa: E402

import backends  # noqa: E402
import torch_backend as tb  # noqa: E402
from napari_dare2d import _api as api  # noqa: E402

ANG_TOL = 1e-2   # degrees
LEN_TOL = 1e-2   # px


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", type=int, default=8)
    ap.add_argument("--frames", default="24,26,28,30")
    ap.add_argument("--onnx-dir", default=str(backends.DEFAULT_ONNX_DIR))
    ap.add_argument("--seg-backend", choices=["onnx", "torch"], default="onnx",
                    help="onnx = hybrid (default); torch = 100%-torch seg")
    ap.add_argument("--stack", default=None)
    args = ap.parse_args()

    frames = [int(t) for t in args.frames.split(",") if t.strip()]
    stack_path = Path(args.stack) if args.stack else sorted(
        (PROJECT_ROOT / "set_8").glob("*.tif*"))[0]
    stack = io.imread(str(stack_path))
    if stack.dtype != np.uint8:
        raise SystemExit(f"stack must be 8-bit, got {stack.dtype}")

    # oracle: all-ONNX (CPU is fine; equals TF)
    reg_on, seg_on = backends.find_onnx(args.onnx_dir, [args.set])
    o_reg = backends.OnnxModel(reg_on[0], backends.resolve_providers("cpu"))
    o_seg = backends.OnnxModel(seg_on[0], backends.resolve_providers("cpu"))

    # hybrid: torch reg (GPU if available) + seg via the chosen backend
    h_reg, h_seg = tb.build_hybrid_models(args.set, onnx_dir=args.onnx_dir,
                                          seg_prefer="cpu", seg_backend=args.seg_backend)
    seg_desc = getattr(h_seg, "providers", f"torch:{getattr(h_seg, 'device', '?')}")
    print(f"stack {stack_path.name} {stack.shape}; frames {frames}")
    print(f"seg_backend={args.seg_backend}; torch reg device: {h_reg.device}; seg: {seg_desc}")

    res_o = api.infer_stack(stack, o_reg, o_seg, frames=frames)
    res_h = api.infer_stack(stack, h_reg, h_seg, frames=frames)

    ok = True
    for f in frames:
        a = sorted(res_o[f], key=lambda d: (d["x"], d["y"]))
        b = sorted(res_h[f], key=lambda d: (d["x"], d["y"]))
        centres_ok = ([(d["x"], d["y"]) for d in a] == [(d["x"], d["y"]) for d in b])
        dang = dlen = 0.0
        if centres_ok and a:
            for da, db in zip(a, b):
                dang = max(dang, abs(da["angle"] - db["angle"]))
                dlen = max(dlen, abs(da["length"] - db["length"]))
        bad = (not centres_ok) or dang > ANG_TOL or dlen > LEN_TOL
        ok &= not bad
        print(f"  f{f}: n_oracle={len(a)} n_hybrid={len(b)} centres_match={centres_ok} "
              f"max_dAngle={dang:.2e} max_dLen={dlen:.2e}{'  FAIL' if bad else ''}")

    print("\n" + ("E2E PARITY OK" if ok else "E2E PARITY FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
