"""Parity gate: TensorFlow oracle vs exported ONNX. [runs in the TF env]

This is THE acceptance check for Piste 1 (PYTORCH_MIGRATION.md §7, §14-C). It runs
the *same* DARE2D pipeline twice on the same mid-stack frames -- once with the
Keras models (oracle), once with the ONNX backend -- and compares at three levels:

  1. raw segmentation probability map (pre-threshold)  -> pure model numerics
  2. raw regression head outputs (length sigmoid, angle tanh) -> pure model numerics
  3. end-to-end detections (centres / angle / length)  -> behaviour parity

Parity is *realistic, not bit-exact* (§14-C): conv kernels differ ~1e-6..1e-4 and
the 0.5 threshold can flip a border pixel, so we tolerate small numeric drift and
+/-1-2 detections at the margin, but require the centres that DO match to coincide.

We compare against the TF *oracle*, never the old ``set_8/*.npy`` reference (a
different pipeline). Both run on CPU here, isolating "is the export faithful?"
from "does the GPU provider work?" (that is P1.5, a separate env).

Usage (TF env, from repo root):
    python dare2d-torch/verify_parity.py --set 8 --frames 26,27,28,29
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

# Windows consoles default to cp1252; force UTF-8 so we can print without crashing.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "napari-dare2d"))
sys.path.insert(0, str(_HERE))

import numpy as np  # noqa: E402
from skimage import io  # noqa: E402

import backends  # noqa: E402
from napari_dare2d import _api as api  # noqa: E402

# tolerances (§14-C) -- tight because both runs are CPU; loose enough for fp drift
SEG_MAXDIFF = 2e-3      # max|Δ| on the [0,1] seg probability map
SEG_MASK_AGREE = 0.999  # fraction of pixels with identical >0.5 decision
REG_MAXDIFF = 2e-3      # max|Δ| on raw length/angle head outputs
MATCH_RADIUS = 5.0      # px: centres closer than this are "the same" detection
CENTER_MEDIAN_PX = 1.0  # matched centres must agree to ~1px (same centroid)
COUNT_TOL = 2           # allow +/- this many detections per frame (threshold flips)


def _triplet(stack, i):
    """Build the equalised (prev,curr,next)/255 float32 input, exactly as infer_stack."""
    n = stack.shape[0]
    prev = stack[i - 1] if i > 0 else stack[i]
    curr = stack[i]
    nxt = stack[i + 1] if i < n - 1 else stack[i]
    prev = api.cv2.equalizeHist(prev)
    curr = api.cv2.equalizeHist(curr)
    nxt = api.cv2.equalizeHist(nxt)
    return np.stack([prev, curr, nxt], axis=-1).astype(np.float32) / 255.0


def _match(a, b, radius):
    """Greedy nearest-neighbour match of two (N,2) centre arrays within radius.

    Returns (matched_pairs_distances, n_a_unmatched, n_b_unmatched).
    """
    if len(a) == 0 or len(b) == 0:
        return np.array([]), len(a), len(b)
    used_b = set()
    dists = []
    for pa in a:
        d = np.hypot(b[:, 0] - pa[0], b[:, 1] - pa[1])
        order = np.argsort(d)
        hit = next((j for j in order if j not in used_b and d[j] <= radius), None)
        if hit is not None:
            used_b.add(hit)
            dists.append(d[hit])
    return np.array(dists), len(a) - len(dists), len(b) - len(dists)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", type=int, default=8)
    ap.add_argument("--frames", default="26,27,28,29",
                    help="comma list of 0-based frame indices (mid-stack)")
    ap.add_argument("--stack", default=None, help="override .tif path")
    ap.add_argument("--reg-dir", default=str(api.DEFAULT_REG_DIR))
    ap.add_argument("--seg-dir", default=str(api.DEFAULT_SEG_DIR))
    ap.add_argument("--onnx-dir", default=str(backends.DEFAULT_ONNX_DIR))
    args = ap.parse_args()

    n = args.set
    frames = [int(t) for t in args.frames.split(",") if t.strip() != ""]

    # locate the stack
    if args.stack:
        stack_path = Path(args.stack)
    else:
        tifs = sorted((PROJECT_ROOT / "set_8").glob("*.tif*"))
        if not tifs:
            raise FileNotFoundError("no .tif in set_8/ ; pass --stack")
        stack_path = tifs[0]
    stack = io.imread(str(stack_path))
    if stack.dtype != np.uint8:
        raise SystemExit(f"stack must be 8-bit, got {stack.dtype}")
    print(f"stack {stack_path.name} {stack.shape} {stack.dtype}; frames {frames}")

    # resolve weights
    reg_ck, seg_ck = api.find_checkpoints(args.reg_dir, args.seg_dir, [n])
    reg_on, seg_on = backends.find_onnx(args.onnx_dir, [n])

    # build both backends
    print("building TF oracle ...")
    reg_k, seg_k = api.build_models(reg_ck[0], seg_ck[0])
    print("building ONNX backend (CPU) ...")
    reg_o, seg_o = backends.build_onnx_models(reg_on[0], seg_on[0], prefer="cpu")
    print(f"  ONNX providers: {seg_o.providers}")

    ok = True
    mid = frames[len(frames) // 2]

    # ---- level 1: raw segmentation probability map on one mid frame ----
    x = _triplet(stack, mid)
    seg_tf = api.inference_strategy(x, seg_k, window_size=256)
    seg_oo = api.inference_strategy(x, seg_o, window_size=256)
    dseg = np.abs(seg_tf - seg_oo)
    agree = np.mean((seg_tf > 0.5) == (seg_oo > 0.5))
    print(f"\n[1] seg prob map (frame {mid}): max|d|={dseg.max():.2e} "
          f"mean|d|={dseg.mean():.2e} mask-agree={agree*100:.4f}%")
    if dseg.max() > SEG_MAXDIFF:
        ok = False; print(f"    FAIL: seg max|d| > {SEG_MAXDIFF}")
    if agree < SEG_MASK_AGREE:
        ok = False; print(f"    FAIL: mask agreement < {SEG_MASK_AGREE*100}%")

    # ---- level 2: raw regression head on this frame's crops ----
    seg_bin = np.where(seg_tf > 0.5, 255, 0).astype(np.uint8)
    centers = api.extract_centers(seg_bin)
    half = 32
    xp = np.pad(x, ((half, half), (half, half), (0, 0)), mode="constant")
    dlen = dang = 0.0
    for c in centers:
        crop = api.crop_img_from_center(xp, (c[1], c[0]), half)
        batch = np.expand_dims(crop, 0)
        lt, at = reg_k.model.predict(batch, verbose=0)
        lo, ao = reg_o.model.predict(batch, verbose=0)
        dlen = max(dlen, float(np.abs(np.asarray(lt) - np.asarray(lo)).max()))
        dang = max(dang, float(np.abs(np.asarray(at) - np.asarray(ao)).max()))
    print(f"[2] reg head ({len(centers)} crops): max|d|length={dlen:.2e} "
          f"max|d|angle={dang:.2e}")
    if max(dlen, dang) > REG_MAXDIFF:
        ok = False; print(f"    FAIL: reg max|d| > {REG_MAXDIFF}")

    # ---- level 3: end-to-end detections over all frames ----
    print("[3] end-to-end detections (TF vs ONNX):")
    tf_res = api.infer_stack(stack, reg_k, seg_k, frames=frames)
    on_res = api.infer_stack(stack, reg_o, seg_o, frames=frames)
    for f in frames:
        A = np.array([[d["y"], d["x"]] for d in tf_res[f]]).reshape(-1, 2)
        B = np.array([[d["y"], d["x"]] for d in on_res[f]]).reshape(-1, 2)
        dists, ua, ub = _match(A, B, MATCH_RADIUS)
        med = float(np.median(dists)) if dists.size else 0.0
        # angle/length diff on matched-by-index when counts equal (quick scalar check)
        amax = lmax = 0.0
        if len(tf_res[f]) == len(on_res[f]) and len(tf_res[f]):
            for da, db in zip(tf_res[f], on_res[f]):
                amax = max(amax, abs(da["angle"] - db["angle"]))
                lmax = max(lmax, abs(da["length"] - db["length"]))
        flag = ""
        if abs(len(A) - len(B)) > COUNT_TOL:
            ok = False; flag += " FAIL:count"
        if dists.size and med > CENTER_MEDIAN_PX:
            ok = False; flag += " FAIL:centre"
        print(f"    f{f}: n_tf={len(A)} n_onnx={len(B)} matched={dists.size} "
              f"unmatched(tf={ua},onnx={ub}) med_d={med:.2f}px "
              f"dAngle={amax:.2f} dLen={lmax:.2f}{flag}")

    print("\n" + ("PARITY OK" if ok else "PARITY FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
