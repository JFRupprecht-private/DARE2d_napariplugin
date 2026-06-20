"""Preprocess a raw DARE2D set into the training layout. [TF env / any with skimage]

Raw set  = one movie `.tiff` + `division_position{f}.npy` (int [x,y,frame], paired rows).
Training generators need a per-frame layout:
    <out>/previmg/{i}.tif  currimg/{i}.tif  nextimg/{i}.tif  div_location/{i}.npy

We reuse `annotator.preprocessing.format_gastru`'s exact crop logic so the prepared data
matches what DARE2D was trained on. For each annotated frame `f`: prev=stack[f-2],
curr=stack[f-1], next=stack[f]; `div_location` = the [x,y] pairs (frame column dropped).
`crop_size>0` tiles each frame into crop_size^2 non-overlapping crops (bipoints assigned to
the crop holding their centre). Boundary frames (f<2) are skipped (no valid prev).

Derived data is written to a SEPARATE folder (default data/prepared/<set>/), leaving the raw
set untouched, and cached (skip if already populated unless --force). DARE2d-main is read-only.

Usage:
    python retrain/prepare.py --set data/neuroepithelium/neuroepithelium/set_8 --crop 256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from skimage import io

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
_REPO = PROJECT_ROOT / "DARE2d-main"
sys.path.insert(0, str(_REPO))

# reuse the exact crop logic the original training data was built with
from annotator.preprocessing.format_gastru import (  # noqa: E402
    get_all_annotated_frames,
    image_to_crops,
)

_SUBDIRS = ("previmg", "currimg", "nextimg", "div_location")
# files that are NOT the source movie (overlays/masks from convertomask3.m etc.)
_DERIVED = ("mask", "overlay", "_pred", "result", "shifted")


def find_movie(set_dir: Path) -> Path:
    set_dir = Path(set_dir)
    cands = [p for p in set_dir.glob("*.tif*")
             if not any(k in p.name.lower() for k in _DERIVED)]
    if len(cands) != 1:
        raise ValueError(
            f"expected exactly 1 source movie in {set_dir}, found {[c.name for c in cands]} "
            "(pass the movie explicitly if ambiguous)")
    return cands[0]


def _save(out_dir: Path, index: int, prev, curr, nxt, pts):
    io.imsave(out_dir / "previmg" / f"{index}.tif", prev, check_contrast=False)
    io.imsave(out_dir / "currimg" / f"{index}.tif", curr, check_contrast=False)
    io.imsave(out_dir / "nextimg" / f"{index}.tif", nxt, check_contrast=False)
    np.save(out_dir / "div_location" / f"{index}.npy", np.asarray(pts).reshape(-1, 2))


def prepare_set(set_dir, out_dir, crop_size=256, force=False, progress_cb=None):
    """Build the training layout for one set. Returns (out_dir, n_samples).

    Cache only when ALL subdirs hold the same (non-zero) count -- an interrupted
    prep (e.g. disk full) leaves inconsistent counts and MUST be rebuilt, else the
    generator chokes on mismatched prev/curr/next shapes.
    """
    import shutil

    set_dir, out_dir = Path(set_dir), Path(out_dir)
    counts = {s: len(list((out_dir / s).glob("*.*"))) if (out_dir / s).exists() else 0
              for s in _SUBDIRS}
    if not force and counts["currimg"] > 0 and len(set(counts.values())) == 1:
        return out_dir, counts["currimg"]  # cached & complete
    # (re)build from scratch: clear any partial/stale output first
    for s in _SUBDIRS:
        if (out_dir / s).exists():
            shutil.rmtree(out_dir / s)
        (out_dir / s).mkdir(parents=True, exist_ok=True)

    movie = find_movie(set_dir)
    stack = io.imread(str(movie))
    if stack.ndim != 3:
        raise ValueError(f"expected a (T,Y,X) movie, got {stack.shape} from {movie.name}")
    frames = [f for f in get_all_annotated_frames(str(set_dir)) if 2 <= f < len(stack)]

    index = 0
    for n, f in enumerate(frames):
        prev_im, im, next_im = stack[f - 2], stack[f - 1], stack[f]
        dp = np.load(set_dir / f"division_position{f}.npy")
        bip = np.array([[d[0], d[1]] for d in dp]).reshape(-1, 2)  # drop frame column
        if crop_size and crop_size > 0:
            x = np.stack([prev_im, im, next_im], axis=-1)
            bipoints = [(bip[2 * i], bip[2 * i + 1]) for i in range(len(bip) // 2)]
            for crop, cbp in image_to_crops(x, bipoints, crop_size):
                pts = [p for pair in cbp for p in pair]
                _save(out_dir, index, crop[:, :, 0], crop[:, :, 1], crop[:, :, 2], pts)
                index += 1
        else:
            _save(out_dir, index, prev_im, im, next_im, bip)
            index += 1
        if progress_cb is not None:
            progress_cb(n + 1, len(frames))
    return out_dir, index


def default_out(set_dir, crop_size):
    set_dir = Path(set_dir)
    tag = f"crop_{crop_size}" if crop_size else "full"
    return PROJECT_ROOT / "data" / "prepared" / tag / set_dir.name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", required=True, help="path to a raw set folder")
    ap.add_argument("--out", default=None, help="output dir (default data/prepared/...)")
    ap.add_argument("--crop", type=int, default=256, help="crop size (0 = full frames)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out) if args.out else default_out(args.set, args.crop)
    print(f"preparing {args.set} -> {out} (crop={args.crop})")
    out, n = prepare_set(args.set, out, crop_size=args.crop, force=args.force,
                         progress_cb=lambda d, t: print(f"  frame {d}/{t}", end="\r"))
    print(f"\ndone: {n} samples in {out}")


if __name__ == "__main__":
    main()
