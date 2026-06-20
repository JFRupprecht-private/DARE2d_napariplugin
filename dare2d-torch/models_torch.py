"""PyTorch ports of the DARE2D inference models. [torch env]

Piste 2 of PYTORCH_MIGRATION.md. These mirror the Keras architectures in
``DARE2d-main/dare2d/model/`` exactly, so the trained weights transfer 1:1
(see ``convert_to_torch.py``). DARE2d-main is never imported or modified.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# regression hyperparams come from config/experiment/regression2d.yaml (model
# override): n_stages=4, n_start_filters=128, crop=64, channels=[-1,0,1] -> 3.
# NOTE the effective net is 4 stages / 128 filters (the experiment override),
# NOT the 2/64 of the bare model yaml -- the #1 silent-mismatch trap (§13).
REG_FILTERS = (128, 256, 512, 1024)
REG_CROP = 64
REG_IN_CH = 3


class Regression2dTorch(nn.Module):
    """Port of ``Regression2dCNN`` (config/model/regression2d_cnn.py).

    Keras: ``[Conv2D(f,3, valid) -> ReLU -> MaxPool2D(2)] x4 -> Flatten ->
    {Dense(1)->sigmoid (length), Dense(2)->tanh (angle)}``.

    THE trap (§14-B #1): Keras ``Flatten`` on an NHWC tensor orders features
    [h, w, c] (channel fastest); a torch flatten of an NCHW tensor orders them
    [c, h, w] (channel slowest). The Dense weights were trained on the Keras
    order, so we ``permute(0,2,3,1)`` (NCHW->NHWC) BEFORE flattening. Skip this
    and the heads read scrambled features -> wrong outputs, no error.
    """

    def __init__(self, in_channels=REG_IN_CH, filters=REG_FILTERS, crop=REG_CROP):
        super().__init__()
        layers = []
        c = in_channels
        for f in filters:
            layers += [nn.Conv2d(c, f, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2)]
            c = f
        self.features = nn.Sequential(*layers)
        # infer the flattened dim by a dummy forward (avoids hand-computed shapes)
        with torch.no_grad():
            d = self.features(torch.zeros(1, in_channels, crop, crop))
        self._flat = d.permute(0, 2, 3, 1).reshape(1, -1).shape[1]
        self.fc_len = nn.Linear(self._flat, 1)   # length head
        self.fc_ang = nn.Linear(self._flat, 2)   # angle head (cos2θ, sin2θ)

    def forward(self, x):  # x: (B, 3, H, W) float32
        x = self.features(x)
        x = x.permute(0, 2, 3, 1).contiguous()   # NCHW -> NHWC (Keras Flatten order)
        x = torch.flatten(x, 1)
        length = torch.sigmoid(self.fc_len(x))   # (B, 1)
        angle = torch.tanh(self.fc_ang(x))       # (B, 2)
        return length, angle


def build_unet_smp():
    """``smp.Unet('resnet18')`` -- NOT weight-compatible with qubvel-TF (see
    SegmentationUnetTorch). Kept only for reference/experiments."""
    import segmentation_models_pytorch as smp

    return smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3,
                    classes=1, activation="sigmoid")


# ---------------------------------------------------------------------------
# Faithful hand-written port of qubvel sm.Unet('resnet18') (the 100%-torch seg)
# ---------------------------------------------------------------------------
# Built straight from the introspected Keras graph (see dare2d-torch/seg_arch.json),
# so the trained weights transfer 1:1 (convert_to_torch.py --kind seg). Crucial
# faithful details, all verified against that graph:
#   * PREACTIVATION resnet (v2): each unit is bn1->relu1->conv1->bn2->relu2->conv2;
#     the 1x1 shortcut is taken from relu1 (post first BN-ReLU), identity otherwise.
#   * qubvel uses explicit *symmetric* ZeroPadding2D + 'valid' convs in the encoder
#     -> exact via F.pad (NO TF 'same' asymmetry). Decoder convs are 'same' stride-1
#     -> torch padding=1.
#   * MaxPool follows a ZERO-pad (relu outputs >=0) -> F.pad then pool (torch's own
#     pool padding uses -inf, which would differ).
#   * BN eps differs by section: encoder/stem 2e-5, decoder 1e-3 (the eps trap).
#   * bn_data has scale=False (gamma fixed = 1); final_conv has bias.
SEG_ENC_EPS = 2e-5
SEG_DEC_EPS = 1e-3
# (stage_in, stage_out, stride) for resnet18 stages 1..4 (2 basic units each)
_STAGES = [(64, 64, 1), (64, 128, 2), (128, 256, 2), (256, 512, 2)]
# decoder: (stage_idx, in_ch_after_concat, out_ch, skip_name|None)
_DEC = [
    (0, 768, 256, "stage4_unit1_relu1"),
    (1, 384, 128, "stage3_unit1_relu1"),
    (2, 192, 64, "stage2_unit1_relu1"),
    (3, 128, 32, "relu0"),
    (4, 32, 16, None),
]


class SegmentationUnetTorch(nn.Module):
    def __init__(self, enc_eps=SEG_ENC_EPS, dec_eps=SEG_DEC_EPS):
        super().__init__()
        L = nn.ModuleDict()  # keyed by the EXACT Keras layer names (no dots)

        def conv(name, ci, co, k, s, bias=False):
            L[name] = nn.Conv2d(ci, co, k, stride=s, bias=bias)

        def bn(name, ch, eps):
            L[name] = nn.BatchNorm2d(ch, eps=eps)

        # --- stem ---
        bn("bn_data", 3, enc_eps)                 # scale=False handled at load
        conv("conv0", 3, 64, 7, 2)
        bn("bn0", 64, enc_eps)
        # --- stages ---
        for si, (cin, cout, stride) in enumerate(_STAGES, start=1):
            for ui in (1, 2):
                p = f"stage{si}_unit{ui}"
                u_in = cin if ui == 1 else cout
                u_stride = stride if ui == 1 else 1
                bn(f"{p}_bn1", u_in, enc_eps)
                conv(f"{p}_conv1", u_in, cout, 3, u_stride)
                bn(f"{p}_bn2", cout, enc_eps)
                conv(f"{p}_conv2", cout, cout, 3, 1)
                if ui == 1:
                    conv(f"{p}_sc", u_in, cout, 1, u_stride)   # projection shortcut
        bn("bn1", 512, enc_eps)                   # final preactivation BN
        # --- decoder ---
        for stage, cin, cout, _skip in _DEC:
            conv(f"decoder_stage{stage}a_conv", cin, cout, 3, 1)
            bn(f"decoder_stage{stage}a_bn", cout, dec_eps)
            conv(f"decoder_stage{stage}b_conv", cout, cout, 3, 1)
            bn(f"decoder_stage{stage}b_bn", cout, dec_eps)
        conv("final_conv", 16, 1, 3, 1, bias=True)
        self.L = L

    def _unit(self, x, prefix, has_sc):
        L = self.L
        pre = F.relu(L[f"{prefix}_bn1"](x))               # = *_unit*_relu1 (skip source)
        shortcut = L[f"{prefix}_sc"](pre) if has_sc else x
        y = L[f"{prefix}_conv1"](F.pad(pre, (1, 1, 1, 1)))   # stride baked into conv
        y = F.relu(L[f"{prefix}_bn2"](y))
        y = L[f"{prefix}_conv2"](F.pad(y, (1, 1, 1, 1)))
        return y + shortcut, pre

    def forward(self, x):  # x: (B, 3, H, W) float32, H=W=256
        L = self.L
        x = L["bn_data"](x)
        x = L["conv0"](F.pad(x, (3, 3, 3, 3)))
        x = F.relu(L["bn0"](x))
        relu0 = x
        x = F.max_pool2d(F.pad(x, (1, 1, 1, 1)), kernel_size=3, stride=2)  # zero-pad pool
        skips = {"relu0": relu0}
        for si in range(1, 5):
            x, pre1 = self._unit(x, f"stage{si}_unit1", has_sc=True)
            x, _ = self._unit(x, f"stage{si}_unit2", has_sc=False)
            skips[f"stage{si}_unit1_relu1"] = pre1
        x = F.relu(L["bn1"](x))                            # encoder output
        for stage, _cin, _cout, skip in _DEC:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if skip is not None:
                x = torch.cat([x, skips[skip]], dim=1)
            x = F.relu(L[f"decoder_stage{stage}a_bn"](
                L[f"decoder_stage{stage}a_conv"](F.pad(x, (1, 1, 1, 1)))))
            x = F.relu(L[f"decoder_stage{stage}b_bn"](
                L[f"decoder_stage{stage}b_conv"](F.pad(x, (1, 1, 1, 1)))))
        x = torch.sigmoid(L["final_conv"](F.pad(x, (1, 1, 1, 1))))
        return x  # (B, 1, H, W)
