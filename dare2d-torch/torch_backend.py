"""PyTorch inference backend with a Keras-compatible surface. [torch env only]

Mirror of ``backends.OnnxModel`` but for torch: exposes ``obj.model.predict(
x_nhwc, verbose=0) -> numpy`` so ``napari_dare2d._api.infer_stack`` runs verbatim.
Kept in a SEPARATE module from ``backends.py`` because that one is imported in the
TF / onnx envs which have no torch.

Scope (PYTORCH_MIGRATION.md §10, §14-B): the **regression** head is a faithful
torch port (parity ~1e-7, see verify_parity_torch.py). The **segmentation** U-Net
stays on ONNX (qubvel-TF resnet18 != torchvision/smp resnet18: input ``bn_data``
BN, a stage1 projection shortcut, and TF 'same' asymmetric padding make a 1:1
weight transfer unfaithful). So the recommended path is a HYBRID: torch reg + ONNX
seg -- ``build_hybrid_models`` wires exactly that.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import models_torch as M  # noqa: E402

WEIGHTS = _HERE / "weights_pt"


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class _RegPredictor:
    """Keras-like ``.predict`` for the regression net: NHWC in -> [length, angle]."""

    def __init__(self, net, device):
        self.net = net.to(device).eval()
        self.device = device

    def predict(self, x, verbose=0, **_):
        xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        xt = xt.permute(0, 3, 1, 2).contiguous().to(self.device)  # NHWC -> NCHW
        with torch.no_grad():
            length, angle = self.net(xt)
        # match the Keras/ONNX contract: length 1-D (B,), angle (B,2)
        return [length.cpu().numpy().reshape(length.shape[0]), angle.cpu().numpy()]


class TorchRegModel:
    """Drop-in for a DARE2D Keras wrapper: ``torch_model.model.predict(...)``."""

    def __init__(self, net, device):
        self.model = _RegPredictor(net, device)
        self.device = device


class _SegPredictor:
    """Keras-like ``.predict`` for the seg net: NHWC in -> NHWC (B,256,256,1) out."""

    def __init__(self, net, device):
        self.net = net.to(device).eval()
        self.device = device

    def predict(self, x, verbose=0, **_):
        xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        xt = xt.permute(0, 3, 1, 2).contiguous().to(self.device)  # NHWC -> NCHW
        with torch.no_grad():
            y = self.net(xt)                                      # (B,1,H,W)
        return y.permute(0, 2, 3, 1).cpu().numpy()               # -> (B,H,W,1)


class TorchSegModel:
    """100%-torch segmentation drop-in (faithful qubvel-resnet18-Unet port)."""

    def __init__(self, net, device):
        self.model = _SegPredictor(net, device)
        self.device = device


def load_torch_regression(pt_path, device=None):
    device = device or default_device()
    net = M.Regression2dTorch()
    net.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    return TorchRegModel(net, device)


def load_torch_segmentation(pt_path, device=None):
    device = device or default_device()
    net = M.SegmentationUnetTorch()
    net.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    return TorchSegModel(net, device)


def build_hybrid_models(set_n, weights_pt=WEIGHTS, onnx_dir=None, device=None,
                        seg_prefer="cuda", seg_backend="onnx"):
    """Return ``(reg, seg)`` ready for ``api.infer_stack``.

    Regression always runs in torch (faithful port). Segmentation is **switchable**
    (risk mitigation, see PYTORCH_MIGRATION.md):
      - ``seg_backend="onnx"`` (default): exact ONNX seg -- the proven path.
      - ``seg_backend="torch"``: the 100%-torch ``SegmentationUnetTorch`` (parity ~1e-7).
    """
    device = device or default_device()
    reg = load_torch_regression(Path(weights_pt) / f"torch_reg_set_{set_n}.pt", device)
    if seg_backend == "torch":
        seg = load_torch_segmentation(Path(weights_pt) / f"torch_seg_set_{set_n}.pt", device)
    elif seg_backend == "onnx":
        import backends  # local import: pulls onnxruntime, only needed here
        onnx_dir = onnx_dir or backends.DEFAULT_ONNX_DIR
        _, seg_onnx = backends.find_onnx(onnx_dir, [set_n])
        seg = backends.OnnxModel(seg_onnx[0], backends.resolve_providers(seg_prefer))
    else:
        raise ValueError(f"seg_backend must be 'onnx' or 'torch', got {seg_backend!r}")
    return reg, seg
