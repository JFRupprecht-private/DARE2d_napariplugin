"""GPU vs CPU benchmark for the ONNX inference path. [runs in the dare2d-onnx env]

P1.5 of PYTORCH_MIGRATION.md. Two measurements:

  1. **Model micro-benchmark**: raw onnxruntime forward passes at realistic batch
     shapes (seg: 49x256x256x3 == one 1024^2 frame's sliding window; reg: a batch
     of 64x64x3 crops), CPU provider vs CUDA provider. This isolates the GPU
     speedup of the actual networks.
  2. **End-to-end**: the *real* ``api.infer_stack`` over a frame range, run with a
     CPU-ONNX backend then a CUDA-ONNX backend. Same numpy/cv2 overhead both ways,
     so the delta is the model time. We also assert CPU and GPU produce the SAME
     detections (GPU must not change results).

Requires the dare2d-onnx env (onnxruntime-gpu + the inference deps + tf stub).

Usage:
    python dare2d-torch/bench.py --set 8 --frames 20-35
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(PROJECT_ROOT / "napari-dare2d"))

import _tf_stub  # noqa: F401,E402  -- installs fake tensorflow before _api import
import numpy as np  # noqa: E402
from skimage import io  # noqa: E402

import backends  # noqa: E402
from napari_dare2d import _api as api  # noqa: E402


def _time_session(sess_obj, x, n=15, warmup=3):
    """Return median seconds for one forward pass of x through an OnnxModel."""
    pred = sess_obj.model.predict
    for _ in range(warmup):
        pred(x)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        pred(x)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def _parse_frames(spec):
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(t) for t in spec.split(",") if t.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", type=int, default=8)
    ap.add_argument("--frames", default="20-35", help='e.g. "20-35" or "26,27,28"')
    ap.add_argument("--onnx-dir", default=str(backends.DEFAULT_ONNX_DIR))
    ap.add_argument("--stack", default=None)
    args = ap.parse_args()

    reg_on, seg_on = backends.find_onnx(args.onnx_dir, [args.set])
    reg_on, seg_on = reg_on[0], seg_on[0]

    # build CPU and CUDA backends for both models
    reg_cpu, seg_cpu = backends.build_onnx_models(reg_on, seg_on, prefer="cpu")
    reg_gpu, seg_gpu = backends.build_onnx_models(reg_on, seg_on, prefer="cuda",
                                                  require_cuda=True)
    print(f"CPU providers: {seg_cpu.providers}")
    print(f"GPU providers: {seg_gpu.providers}")

    # ---- 1) model micro-benchmark ----
    print("\n=== model micro-benchmark (median of 15) ===")
    seg_x = np.random.rand(49, 256, 256, 3).astype(np.float32)   # one 1024^2 frame
    reg_x = np.random.rand(32, 64, 64, 3).astype(np.float32)
    for name, xc, mc, mg in [("seg (49x256x256x3)", seg_x, seg_cpu, seg_gpu),
                             ("reg (32x64x64x3)", reg_x, reg_cpu, reg_gpu)]:
        tc = _time_session(mc, xc)
        tg = _time_session(mg, xc)
        print(f"  {name:22s} CPU {tc*1e3:8.1f} ms | GPU {tg*1e3:7.1f} ms | "
              f"speedup {tc/tg:5.1f}x")

    # ---- 2) end-to-end real pipeline ----
    if args.stack:
        stack_path = Path(args.stack)
    else:
        stack_path = sorted((PROJECT_ROOT / "set_8").glob("*.tif*"))[0]
    stack = io.imread(str(stack_path))
    if stack.dtype != np.uint8:
        raise SystemExit(f"stack must be 8-bit, got {stack.dtype}")
    frames = _parse_frames(args.frames)
    print(f"\n=== end-to-end infer_stack: {stack_path.name} {stack.shape}, "
          f"{len(frames)} frames {frames[0]}..{frames[-1]} ===")

    # warm both backends on the first frame (excluded from timing)
    api.infer_stack(stack, reg_cpu, seg_cpu, frames=frames[:1])
    api.infer_stack(stack, reg_gpu, seg_gpu, frames=frames[:1])

    t0 = time.perf_counter()
    res_cpu = api.infer_stack(stack, reg_cpu, seg_cpu, frames=frames)
    tc = time.perf_counter() - t0

    t0 = time.perf_counter()
    res_gpu = api.infer_stack(stack, reg_gpu, seg_gpu, frames=frames)
    tg = time.perf_counter() - t0

    # GPU must yield the same detections as CPU (count + centres)
    same = True
    for f in frames:
        a = sorted((d["x"], d["y"]) for d in res_cpu[f])
        b = sorted((d["x"], d["y"]) for d in res_gpu[f])
        if a != b:
            same = False
            print(f"  WARN frame {f}: CPU {len(a)} vs GPU {len(b)} detections differ")
    ndet = sum(len(v) for v in res_cpu.values())
    print(f"  detections total={ndet}  CPU==GPU centres: {same}")
    print(f"  CPU {tc:7.2f} s ({tc/len(frames)*1e3:6.1f} ms/frame) | "
          f"GPU {tg:7.2f} s ({tg/len(frames)*1e3:6.1f} ms/frame) | "
          f"speedup {tc/tg:4.1f}x")


if __name__ == "__main__":
    main()
