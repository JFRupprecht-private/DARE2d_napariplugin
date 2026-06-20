"""Convert dumped Keras weights -> torch state_dict. [torch env]

Layout remaps (PYTORCH_MIGRATION.md §5, §14-B):
  - Conv2D kernel  HWIO (kh,kw,in,out) -> OIHW (out,in,kh,kw)   transpose(3,2,0,1)
  - Dense  kernel  (in,out)            -> (out,in)              transpose
  - biases unchanged

Every assignment is shape-asserted against the freshly built torch module, so a
hyperparam mismatch (e.g. 4 stages vs 2) fails loudly here, not silently at run
time. Architecture dims (filters, in_channels) are INFERRED from the dump, not
hard-coded -- the dump is the single source of truth.

Usage (torch env, repo root):
    python dare2d-torch/convert_to_torch.py --kind reg --sets 1-8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import models_torch as M  # noqa: E402

WEIGHTS = _HERE / "weights_pt"


def _assign(dst: torch.Tensor, arr: np.ndarray, name: str):
    t = torch.from_numpy(np.ascontiguousarray(arr))
    if tuple(t.shape) != tuple(dst.shape):
        raise ValueError(f"shape mismatch for {name}: keras->{tuple(t.shape)} "
                         f"vs torch {tuple(dst.shape)}")
    dst.copy_(t)


def convert_segmentation(npz):
    """Load qubvel-TF seg weights into SegmentationUnetTorch (by exact Keras name).

    Conv HWIO->OIHW (no bias except final_conv). BN: scale=True layers carry
    [gamma,beta,mean,var]; bn_data (scale=False) carries [beta,mean,var] with gamma
    fixed = 1. Detected by whether the 4th array (|3) exists.
    """
    model = M.SegmentationUnetTorch()
    files = set(npz.files)
    with torch.no_grad():
        for name, mod in model.L.items():
            if isinstance(mod, torch.nn.Conv2d):
                k = npz[f"{name}|0"]  # HWIO
                _assign(mod.weight.data, np.transpose(k, (3, 2, 0, 1)), f"{name}.weight")
                if mod.bias is not None:
                    _assign(mod.bias.data, npz[f"{name}|1"], f"{name}.bias")
            elif isinstance(mod, torch.nn.BatchNorm2d):
                if f"{name}|3" in files:  # scale=True: gamma, beta, mean, var
                    _assign(mod.weight.data, npz[f"{name}|0"], f"{name}.gamma")
                    _assign(mod.bias.data, npz[f"{name}|1"], f"{name}.beta")
                    _assign(mod.running_mean.data, npz[f"{name}|2"], f"{name}.mean")
                    _assign(mod.running_var.data, npz[f"{name}|3"], f"{name}.var")
                else:                     # scale=False (bn_data): gamma fixed = 1
                    mod.weight.data.fill_(1.0)
                    _assign(mod.bias.data, npz[f"{name}|0"], f"{name}.beta")
                    _assign(mod.running_mean.data, npz[f"{name}|1"], f"{name}.mean")
                    _assign(mod.running_var.data, npz[f"{name}|2"], f"{name}.var")
    return model


def convert_regression(npz):
    # infer architecture from the dump (garde-fou §13)
    n_conv = sum(1 for k in npz.files if k.startswith("conv") and k.endswith(".weight"))
    filters = tuple(int(npz[f"conv{i}.weight"].shape[3]) for i in range(n_conv))
    in_ch = int(npz["conv0.weight"].shape[2])
    model = M.Regression2dTorch(in_channels=in_ch, filters=filters, crop=M.REG_CROP)

    with torch.no_grad():
        for i in range(n_conv):
            conv = model.features[3 * i]            # conv, relu, maxpool triples
            k = npz[f"conv{i}.weight"]              # HWIO
            _assign(conv.weight.data, np.transpose(k, (3, 2, 0, 1)), f"conv{i}.weight")
            _assign(conv.bias.data, npz[f"conv{i}.bias"], f"conv{i}.bias")
        for tag, fc in [("fc_len", model.fc_len), ("fc_ang", model.fc_ang)]:
            _assign(fc.weight.data, npz[f"{tag}.weight"].T, f"{tag}.weight")  # (in,out)->(out,in)
            _assign(fc.bias.data, npz[f"{tag}.bias"], f"{tag}.bias")
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=["reg", "seg"], default="reg")
    ap.add_argument("--sets", default="1-8")
    args = ap.parse_args()

    sets = []
    for tok in args.sets.replace(" ", "").split(","):
        if "-" in tok:
            a, b = tok.split("-"); sets += list(range(int(a), int(b) + 1))
        elif tok:
            sets.append(int(tok))

    for n in sets:
        npz_path = WEIGHTS / f"keras_{args.kind}_set_{n}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
        npz = np.load(npz_path)
        model = convert_regression(npz) if args.kind == "reg" else convert_segmentation(npz)
        out = WEIGHTS / f"torch_{args.kind}_set_{n}.pt"
        torch.save(model.state_dict(), out)
        nparam = sum(p.numel() for p in model.parameters())
        extra = f"flatten_dim={model._flat}" if args.kind == "reg" else f"modules={len(model.L)}"
        print(f"set {n}: {out.name}  params={nparam}  {extra}")
    print("done")


if __name__ == "__main__":
    main()
