"""napari reader for plain ``(T, Y, X)`` ``.tif`` / ``.tiff`` image stacks.

Intentionally **import-light**: this module pulls in nothing from the DARE2D core, no
TensorFlow, and not ``_api`` / ``_widget``. napari probes a plugin's reader on *every* file
open, so doing the heavy DARE2D import here would make opening any file slow (or fail in a
bare env). Reading itself uses ``tifffile`` (imported lazily, with a ``skimage`` fallback).

The reader only claims ``.tif`` / ``.tiff`` (case-insensitive) and returns ``None`` for
anything else, so other readers (napari-builtins, other plugins) keep handling every other
format. A ``.tif`` may therefore be opened by either this reader or napari-builtins —
napari lets the user pick / remembers the choice.
"""
from __future__ import annotations

from pathlib import Path

_EXTS = (".tif", ".tiff")


def napari_get_reader(path):
    """Return a reader callable if ``path`` is a ``.tif``/``.tiff``, else ``None``.

    ``path`` is a single path or a list of paths (napari passes either). We claim the file(s)
    only when the (first) path has a ``.tif``/``.tiff`` suffix, matched case-insensitively so
    ``.TIF`` works on case-sensitive filesystems too.
    """
    first = path[0] if isinstance(path, (list, tuple)) else path
    if not isinstance(first, str):
        return None
    if Path(first).suffix.lower() not in _EXTS:
        return None
    return _read


def _read(path):
    """Read one or more ``.tif``/``.tiff`` paths into napari image layer-data tuples."""
    paths = path if isinstance(path, (list, tuple)) else [path]
    out = []
    for p in paths:
        try:
            import tifffile
            data = tifffile.imread(str(p))
        except Exception:
            from skimage import io as skio
            data = skio.imread(str(p))
        out.append((data, {"name": Path(p).stem}, "image"))
    return out
