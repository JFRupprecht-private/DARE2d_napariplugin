"""Runnable check for the DARE2D in-process API (_dare2d_api.py).

Proves the pipeline is callable as an API on REAL data:
  - builds the model-set-8 regression + segmentation models from the .h5 weights,
  - runs infer_stack on a few frames of the real (56,1024,1024) stack,
  - HARD-asserts the API contract (structure, dtypes, value ranges),
  - SOFT-compares detected centers against the known set_8 reference .npy,
  - HARD-asserts consensus() produces the expected per-cluster fields.

Run:
    <env>/python.exe napari-dare2d/verify_api.py
Exit code 0 = OK, 1 = FAIL.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from napari_dare2d import _api as api  # noqa: E402

# Canonical layout (derived in _api): the dataset lives under
# data/neuroepithelium/neuroepithelium/set_8 and the curated checkpoints under
# models/best/{regression,segmentation}_checkpoints — both produced by the
# "Download DARE2D data" button. Use those paths so this check matches what ships.
STACK_PATH = api.DATA_DIR / "set_8" / "siractinE2_14-03-23_1_post_z9-celldivisionlevel.tiff"
REF_DIR = api.DATA_DIR / "set_8"  # division_position{frame}.npy (set-8 reference)
REG_CKPT = api.DEFAULT_REG_DIR / "checkpoints_set_8_all_but_target" / "best.h5"
SEG_CKPT = api.DEFAULT_SEG_DIR / "checkpoints_set_8_all_but_target" / "best.h5"

FRAMES = [27, 28, 29]       # 0-based, mid-stack (56 frames); compared to division_position{i+1}.npy
MATCH_TOL = 15.0            # px, for the soft reference comparison

failures = []


def hard(cond, msg):
    print(("  [OK] " if cond else "  [FAIL] ") + msg)
    if not cond:
        failures.append(msg)


def load_ref_centers(frame_0based):
    """set_8 reference rows are [x, y, 2]; return Nx2 (x, y) array."""
    fp = REF_DIR / f"division_position{frame_0based + 1}.npy"
    if not fp.exists():
        return None
    arr = np.load(fp, allow_pickle=True)
    if len(arr) == 0:
        return np.empty((0, 2))
    return np.array([[float(r[0]), float(r[1])] for r in arr], dtype=float)


def main():
    print("== preconditions ==")
    for p in (STACK_PATH, REG_CKPT, SEG_CKPT):
        hard(p.exists(), f"exists: {p.name}")
    if failures:
        return 1

    import tifffile

    stack = tifffile.imread(str(STACK_PATH))
    hard(stack.ndim == 3 and stack.dtype == np.uint8, f"stack (T,Y,X) uint8: {stack.shape} {stack.dtype}")

    print("== build_models (set 8) ==")
    reg, seg = api.build_models(str(REG_CKPT), str(SEG_CKPT))
    hard(hasattr(reg, "model") and hasattr(seg, "model"), "model wrappers built + weights loaded")

    print(f"== infer_stack on frames {FRAMES} (CPU; sliding window 256) ==")
    res = api.infer_stack(stack, reg, seg, frames=FRAMES,
                          progress_cb=lambda d, t: print(f"  frame {d}/{t} done"))

    # --- HARD: API contract -------------------------------------------------
    hard(set(res.keys()) == set(FRAMES), "infer_stack returns requested frames")
    all_dets_flat = [d for f in FRAMES for d in res[f]]
    hard(len(all_dets_flat) > 0, f"produced detections (total={len(all_dets_flat)})")
    sample_ok = all(
        set(d.keys()) == {"x", "y", "angle", "length"}
        and isinstance(d["x"], int) and isinstance(d["y"], int)
        and isinstance(d["angle"], float) and isinstance(d["length"], float)
        for d in all_dets_flat
    )
    hard(sample_ok, "each detection is {x:int, y:int, angle:float, length:float}")
    if all_dets_flat:
        angs = [d["angle"] for d in all_dets_flat]
        lens = [d["length"] for d in all_dets_flat]
        hard(all(-90.1 <= a <= 90.1 for a in angs), f"angles in [-90,90] (deg); range [{min(angs):.1f},{max(angs):.1f}]")
        hard(all(0 <= l <= 64 for l in lens), f"lengths in [0, crop_size]; range [{min(lens):.1f},{max(lens):.1f}]")

    # --- SOFT: alignment with set-8 reference centers -----------------------
    # The set_8 .npy are an OLDER format ([x, y, 2], not dicts) and use the
    # opposite x/y convention from the current script (this API matches the
    # current script: x=column, y=row). So we report BOTH orientations; the
    # swapped one is the meaningful comparison. This is a loose sanity anchor,
    # not ground truth -- a hard assert here would be misleading.
    def nn_stats(ref, pred):
        dmat = np.linalg.norm(ref[:, None, :] - pred[None, :, :], axis=2)
        nn = dmat.min(axis=1)
        return float(np.mean(nn <= MATCH_TOL)), float(np.median(nn))

    print("== reference comparison (soft; ref is old-format, see note) ==")
    for i in FRAMES:
        ref = load_ref_centers(i)
        pred = np.array([[d["x"], d["y"]] for d in res[i]], dtype=float)
        if ref is None or ref.shape[0] == 0:
            print(f"  frame {i}: no reference")
            continue
        if pred.shape[0] == 0:
            print(f"  frame {i}: ref={ref.shape[0]} pred=0  (no match)")
            continue
        m_asis, md_asis = nn_stats(ref, pred)
        m_swap, md_swap = nn_stats(ref, pred[:, ::-1])
        print(f"  frame {i}: ref={ref.shape[0]} pred={pred.shape[0]} | "
              f"as-is match@{MATCH_TOL:g}={m_asis:.0%} med={md_asis:.0f}px | "
              f"x/y-swapped match@{MATCH_TOL:g}={m_swap:.0%} med={md_swap:.0f}px")

    # --- HARD: consensus contract ------------------------------------------
    print("== consensus() ==")
    # Build a synthetic 3-model ensemble from the real frame-1 detections so the
    # consensus path is exercised quickly without running all 8 model sets.
    base = res[FRAMES[0]]
    all_dets = {1: []}
    rng = np.random.default_rng(0)
    for model_id in (1, 2, 3):
        for d in base:
            jx, jy = rng.normal(0, 1.5, 2)
            all_dets[1].append((model_id, d["x"] + jx, d["y"] + jy, d["angle"], d["length"]))
    cons = api.consensus(all_dets, n_frames=1, eps=10, min_models=2, num_models=3)
    hard(1 in cons, "consensus returns frame 1")
    if cons.get(1):
        c = cons[1][0]
        needed = {"x", "y", "angle", "angle_std_deg", "length", "length_std",
                  "pos_std", "n_models", "support_fraction"}
        hard(needed.issubset(c.keys()), "consensus dict has required fields")
        hard(c["n_models"] >= 2, f"cluster supported by >=2 models (n_models={c['n_models']})")
    else:
        hard(False, "consensus produced at least one cluster")

    print()
    if failures:
        print(f"RESULT: FAIL ({len(failures)} hard check(s) failed)")
        return 1
    print("RESULT: OK — DARE2D inference + consensus are callable as an in-process API.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
