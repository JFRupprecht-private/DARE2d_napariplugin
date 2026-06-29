"""Download DARE2D data/models from Zenodo (record 17442227) and save plugin results.

Stdlib-only download (``urllib`` + ``zipfile``), so no install-time dependency is
added. Used by the napari "Download DARE2D data" and "Save DARE2D results" buttons.
"""
from __future__ import annotations

import csv
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

#: Zenodo record holding the checkpoints + the neuroepithelium dataset.
ZENODO_RECORD = "17442227"
_API = "https://zenodo.org/api/records/{}"

#: Zenodo zip -> destination subdir (relative to the project root), matching the demo
#: layout the plugin/notebooks read (``models/demo/neuroepithelium/...`` and
#: ``data/demo/neuroepithelium/``). The checkpoint zips extract into their full target
#: dir; ``neuroepithelium.zip`` contains a top-level ``neuroepithelium/`` folder, so it
#: extracts into ``data/demo`` (yielding ``data/demo/neuroepithelium/``).
_TARGETS = {
    "regression_checkpoints.zip":   Path("models") / "demo" / "neuroepithelium" / "regression_checkpoints",
    "segmentation_checkpoints.zip": Path("models") / "demo" / "neuroepithelium" / "segmentation_checkpoints",
    "torch_weights.zip":            Path("models") / "demo" / "neuroepithelium",
    "neuroepithelium.zip":          Path("data") / "demo",
}


def zenodo_files(record_id: str = ZENODO_RECORD):
    """Return the record's file list (each has ``key``, ``size``, ``links.self``)."""
    with urllib.request.urlopen(_API.format(record_id), timeout=60) as resp:
        return json.load(resp)["files"]


def _is_junk(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return name.startswith("__MACOSX") or base == ".DS_Store" or base.startswith("._")


def _download(url: str, out: Path, total, progress_cb=None) -> None:
    """Stream ``url`` to ``out`` (atomic via a .part temp), reporting bytes done."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    done = 0
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:
        while True:
            buf = resp.read(1 << 20)  # 1 MiB
            if not buf:
                break
            fh.write(buf)
            done += len(buf)
            if progress_cb is not None:
                progress_cb(done, total or 0)
    tmp.replace(out)


def download_dataset(root, progress_cb=None, log=print) -> Path:
    """Download record 17442227 and extract it into the project layout under ``root``.

    Checkpoints -> ``models/demo/neuroepithelium/{regression,segmentation}_checkpoints``;
    the neuroepithelium dataset -> ``data/demo/neuroepithelium/``. Zips are cached under
    ``root/_zenodo_cache`` and reused if already present at the right size. Extraction
    only ADDS missing files -- existing files are never overwritten, so a populated
    ``models/`` (e.g. retrained checkpoints) is left intact.
    """
    root = Path(root)
    cache = root / "_zenodo_cache"
    cache.mkdir(parents=True, exist_ok=True)
    for f in zenodo_files():
        sub = _TARGETS.get(f["key"])
        if sub is None:
            continue  # skip README.txt etc.
        zpath = cache / f["key"]
        size = f.get("size")
        if zpath.exists() and (size is None or zpath.stat().st_size == size):
            log(f"already downloaded: {f['key']}")
        else:
            log(f"downloading {f['key']} ({(size or 0) / 1e6:.0f} MB)…")
            _download(f["links"]["self"], zpath, size, progress_cb)
        dest = root / sub
        dest.mkdir(parents=True, exist_ok=True)
        log(f"extracting {f['key']} → {sub.as_posix()}…")
        with zipfile.ZipFile(zpath) as z:
            for m in z.namelist():
                # skip junk and NEVER overwrite existing files: the download only fills in
                # what's missing, so it never erases or clobbers models/ (or anything else).
                if _is_junk(m) or (dest / m).exists():
                    continue
                z.extract(m, dest)
    log(f"done → {root}")
    return root


# --------------------------------------------------------------------------- #
# Saving plugin results                                                        #
# --------------------------------------------------------------------------- #
def save_results(image, points, features, out_dir, image_name="dare2d", draw=True):
    """Write DARE2D detections to disk.

    Args:
        image: the analysed ``(T, Y, X)`` stack — used to render the overlay movie.
        points: ``(N, 3)`` array of detection coords ``[t, y, x]`` (napari Points data).
        features: dict of per-detection arrays, e.g. ``{"angle": (N,), "length": (N,)}``.
        out_dir: folder to write into (created if missing).
        image_name: stem used for the output filenames.
        draw: also render a ``(T, Y, X, 3)`` overlay tiff ("result movie").

    Writes per-frame ``division_position{t}.npy`` ([x, y] pairs), a ``*_summary.csv``
    (frame, x, y, angle, length, …) and, if ``draw``, ``*_result.tiff``. Returns out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) == 0:
        raise ValueError("no detections to save (the Points layer is empty)")
    t = points[:, 0].round().astype(int)
    y, x = points[:, 1], points[:, 2]
    n = len(points)
    angle = np.asarray(features.get("angle", np.full(n, np.nan)), float).reshape(-1)
    length = np.asarray(features.get("length", np.full(n, np.nan)), float).reshape(-1)
    extra = [k for k in features if k not in ("angle", "length")]

    # per-frame [x, y] npy (the project's division_position layout)
    for f in np.unique(t):
        sel = t == f
        np.save(out_dir / f"division_position{int(f)}.npy",
                np.stack([x[sel], y[sel]], axis=1))

    # summary csv
    cols = ["frame", "x", "y", "angle", "length"] + extra
    with open(out_dir / f"{image_name}_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i in range(n):
            row = [int(t[i]), float(x[i]), float(y[i]), float(angle[i]), float(length[i])]
            row += [float(np.asarray(features[k], float).reshape(-1)[i]) for k in extra]
            w.writerow(row)

    # overlay "result movie": grayscale -> RGB with a red centre + cyan division axis
    if draw:
        import cv2
        import tifffile

        g = np.asarray(image)
        if g.ndim == 2:
            g = g[None]
        if g.dtype != np.uint8:
            gmin, gptp = float(g.min()), float(np.ptp(g)) or 1.0
            g = (255 * (g.astype(float) - gmin) / gptp).astype(np.uint8)
        movie = np.repeat(g[..., None], 3, axis=-1)  # (T, Y, X, 3)
        T = movie.shape[0]
        for i in range(n):
            ti = int(t[i])
            if not (0 <= ti < T):
                continue
            cx, cy = int(round(x[i])), int(round(y[i]))
            cv2.circle(movie[ti], (cx, cy), 4, (255, 0, 0), -1)  # red centre
            if np.isfinite(angle[i]) and np.isfinite(length[i]):
                th = np.radians(angle[i])
                # match detections_to_vectors: direction (Δy=cos·L, Δx=sin·L)
                dcol, drow = np.sin(th) * length[i] / 2.0, np.cos(th) * length[i] / 2.0
                cv2.line(movie[ti], (int(cx - dcol), int(cy - drow)),
                         (int(cx + dcol), int(cy + drow)), (0, 255, 255), 2)  # cyan axis
        tifffile.imwrite(out_dir / f"{image_name}_result.tiff", movie)
    return out_dir
