"""ONNX inference backend with a Keras-compatible surface. [env-agnostic]

The whole point (PYTORCH_MIGRATION.md §13): expose an object that quacks exactly
like the DARE2D Keras wrapper used in ``napari_dare2d._api`` -- i.e. something
with ``.model.predict(x_nhwc, verbose=0) -> numpy`` -- so that ``infer_stack``
and ``inference_strategy`` are reused **verbatim**, with zero edits to the
plugin or to DARE2d-main. We just hand them ``OnnxModel`` objects instead of the
Keras wrappers.

Two shapes of output, matching the two models:
  - segmentation: a single array ``(N, 256, 256, 1)`` -> ``predict`` returns it as-is.
  - regression: TWO arrays. Keras unpacks ``length_pred, angle_pred = predict(...)``,
    so we MUST return ``[length(N,1), angle(N,2)]`` in that order. We don't trust
    output *names*; we order by last dim (1 == length, 2 == angle) so the mapping
    is self-correcting (§14-A "ordre des 2 sorties").

Provider policy (§14-A "provider CUDA silencieux"): ``onnxruntime-gpu`` silently
falls back to CPU if CUDA/cuDNN are missing. ``build_onnx_models(require_cuda=True)``
*asserts* CUDAExecutionProvider is actually active and raises otherwise.
"""

from __future__ import annotations

import glob
import os
import site
from pathlib import Path


def _setup_cuda_dll_path():
    """Make the pip ``nvidia-*-cu12`` CUDA/cuDNN DLLs discoverable on Windows.

    onnxruntime-gpu's CUDA EP needs cudart/cublas/cufft/cudnn at session time.
    When CUDA is provided via pip wheels (no system CUDA install), their DLLs
    live in ``site-packages/nvidia/*/bin``. ``os.add_dll_directory`` covers the
    DLLs ORT loads directly, but cuDNN resolves its *sublibraries*
    (cudnn_engines_*64_9.dll) through the OS loader -> PATH, so we set both.
    No-op on non-Windows or CPU-only envs (no nvidia/ dir => empty list).
    """
    dirs = []
    for sp in set(site.getsitepackages() + [site.getusersitepackages()]):
        dirs += glob.glob(os.path.join(sp, "nvidia", "*", "bin"))
    for d in dirs:
        try:
            os.add_dll_directory(d)
        except (OSError, AttributeError):
            pass  # AttributeError: add_dll_directory is Windows-only
    if dirs:
        os.environ["PATH"] = os.pathsep.join(dirs) + os.pathsep + os.environ.get("PATH", "")


# must run BEFORE onnxruntime is imported so the CUDA provider can bind its DLLs
_setup_cuda_dll_path()

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

# silence the benign "tf_half_pixel_for_nn is deprecated" Resize warning that the
# sm.Unet upsampling triggers at session creation (cosmetic; results are correct).
ort.set_default_logger_severity(3)

_HERE = Path(__file__).resolve().parent
DEFAULT_ONNX_DIR = _HERE / "weights_onnx"


def resolve_providers(prefer: str = "cuda"):
    """Return an ORT provider list. ``prefer='cuda'`` puts CUDA first if present."""
    avail = ort.get_available_providers()
    if prefer == "cuda" and "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class _Session:
    """Wraps one ``ort.InferenceSession`` with a Keras ``.predict``-like call."""

    def __init__(self, onnx_path: str, providers):
        so = ort.SessionOptions()
        so.log_severity_level = 3  # mute the benign tf_half_pixel_for_nn Resize warning
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        outs = self.sess.get_outputs()
        self.multi = len(outs) > 1
        if self.multi:
            # order so last-dim==1 (length) precedes last-dim==2 (angle), matching
            # Keras' [len_x, angle_x]. Robust to whatever tf2onnx named them.
            def last_dim(o):
                s = o.shape
                return s[-1] if isinstance(s[-1], int) else 10 ** 9

            order = sorted(range(len(outs)), key=lambda i: last_dim(outs[i]))
            self.request = [outs[i].name for i in order]
        else:
            self.request = [outs[0].name]

    @property
    def providers(self):
        return self.sess.get_providers()

    def predict(self, x, verbose=0, **_):
        """NHWC float32 in -> numpy out. Single array (seg) or [length, angle] (reg)."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        outs = self.sess.run(self.request, {self.input_name: x})
        if not self.multi:
            return outs[0]
        # Regression: request[0] is the length head (last-dim 1). DARE2D's
        # convert_values wants length as 1-D (B,) [its docstring], but the Dense(1)
        # head yields (B,1); numpy>=2 refuses to assign that into a scalar slot
        # (numpy<2 silently coerced it). Return the documented (B,) so the pipeline
        # runs identically under numpy 1.x (TF env) and numpy 2.x (this env).
        length, angle = outs[0], outs[1]
        return [length.reshape(length.shape[0]), angle]


class OnnxModel:
    """Drop-in for a DARE2D Keras wrapper: ``onnx_model.model.predict(...)``."""

    def __init__(self, onnx_path: str, providers):
        self.model = _Session(onnx_path, providers)

    @property
    def providers(self):
        return self.model.providers


def find_onnx(weights_dir, sets):
    """Resolve ``reg_set_{n}.onnx`` / ``seg_set_{n}.onnx`` for the given sets."""
    weights_dir = Path(weights_dir)
    reg, seg = [], []
    for n in sets:
        r = weights_dir / f"reg_set_{n}.onnx"
        s = weights_dir / f"seg_set_{n}.onnx"
        if not r.exists():
            raise FileNotFoundError(f"regression ONNX missing: {r}")
        if not s.exists():
            raise FileNotFoundError(f"segmentation ONNX missing: {s}")
        reg.append(str(r))
        seg.append(str(s))
    return reg, seg


def build_onnx_models(reg_onnx, seg_onnx, prefer="cuda", require_cuda=False):
    """Return ``(reg_OnnxModel, seg_OnnxModel)`` ready for ``api.infer_stack``.

    ``require_cuda=True`` raises unless CUDAExecutionProvider is actually active
    (no silent CPU fallback) -- use it for the GPU perf check.
    """
    providers = resolve_providers(prefer)
    reg = OnnxModel(reg_onnx, providers)
    seg = OnnxModel(seg_onnx, providers)
    if require_cuda:
        active = set(reg.providers) | set(seg.providers)
        if "CUDAExecutionProvider" not in active:
            raise RuntimeError(
                "CUDA was required but onnxruntime is running on "
                f"{sorted(active)}. Available providers: {ort.get_available_providers()}. "
                "Install onnxruntime-gpu with matching CUDA/cuDNN."
            )
    return reg, seg
