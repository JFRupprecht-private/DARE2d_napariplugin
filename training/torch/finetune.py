"""PyTorch transfer-learning / fine-tune backend for DARE2D. [TF env]

Loads a pretrained DARE2D ``.pt`` checkpoint, rebuilds the matching architecture
(dare2d-torch: Regression2dTorch / SegmentationUnetTorch), loads the weights, freezes
the backbone, and continues training at a low (optionally discriminative) learning rate
on the same leave-one-out data pipeline as train.py. Saves a state_dict ``best.pt`` in the
exact format the inference backend loads, plus a ``finetune_config.json`` sidecar.

Fine-tuning is PyTorch-only and is the lead path; there is no TF/Keras fine-tune backend.
The DATA + architecture + train loop are reused verbatim from train.py / train_split.py.

Usage:
  python training/torch/finetune.py --experiment regression2d --base-model models/demo/neuroepithelium/regression_checkpoints/checkpoints_set_8_all_but_target/best.pt --test-set 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("SM_FRAMEWORK", "tf.keras")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

_HERE = Path(__file__).resolve().parent                 # training/torch/
PROJECT_ROOT = _HERE.parents[1]                          # repo root
for p in (str(_HERE.parent), str(_HERE.parent / "tf"), str(PROJECT_ROOT / "dare2d-torch")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim.lr_scheduler as S  # noqa: E402

import train as T          # noqa: E402  reuse build_loaders / train_loop / data wiring
import train_split as ts   # noqa: E402  run-dir resolver + default raw root
import models_torch as M   # noqa: E402  faithful torch models

_EXP = T._EXP              # experiment -> (batch_training cfg, output subfolder, kind)


# --------------------------------------------------------------------------- #
# Base-model loading (state_dict vs full nn.Module vs {"state_dict": ...})     #
# --------------------------------------------------------------------------- #
def _extract_state_dict(obj):
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if isinstance(obj, dict):
        return obj["state_dict"] if "state_dict" in obj else obj
    raise ValueError("unrecognized .pt content: expected a state_dict or an nn.Module")


def _guess_stage(state):
    """Best-effort: which DARE2D stage does this state_dict belong to? (seg = U-Net ModuleDict
    'L.*'; reg = 'features.*'/'fc_*'.)"""
    keys = list(state.keys()) if hasattr(state, "keys") else []
    if any(str(k).startswith("L.") for k in keys):
        return "segmentation"
    if any(str(k).startswith(("features.", "fc_len", "fc_ang")) for k in keys):
        return "regression"
    return None


def load_base(model, base_path, experiment):
    """Load pretrained weights into ``model`` (strict). Clear message on a mismatch."""
    obj = torch.load(str(base_path), map_location="cpu", weights_only=False)
    state = _extract_state_dict(obj)
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, KeyError) as e:
        guess = _guess_stage(state)
        want = "regression" if experiment == "regression2d" else "segmentation"
        hint = (f" '{Path(base_path).name}' looks like a {guess.upper()} checkpoint, but you "
                f"selected the {want.upper()} stage." if guess and guess != want else "")
        raise SystemExit(
            f"could not load '{Path(base_path).name}' into the {experiment} model: it does not "
            f"match this architecture.{hint}\nPick the .pt for the stage you selected "
            f"(regression -> torch_reg_*.pt or regression_checkpoints/.../best.pt; "
            f"segmentation -> torch_seg_*.pt or segmentation_checkpoints/.../best.pt)."
        )


# --------------------------------------------------------------------------- #
# Freeze: backbone frozen by default; unfreeze head (+ decoder) + last N blocks #
# --------------------------------------------------------------------------- #
def _reg_groups(model):
    """(ordered backbone blocks input->output, head modules) for Regression2dTorch."""
    convs = [m for m in model.features if isinstance(m, nn.Conv2d)]   # 4 conv blocks
    backbone = [[c] for c in convs]
    head = [model.fc_len, model.fc_ang]
    return backbone, head


def _seg_groups(model):
    """(ordered encoder blocks input->output, decoder+final head modules) for the U-Net."""
    L = model.L
    backbone = [[L["bn_data"], L["conv0"], L["bn0"]]]                 # stem
    for si in (1, 2, 3, 4):
        keys = [k for k in L.keys() if k.startswith(f"stage{si}_")]
        if si == 4:
            keys = keys + ["bn1"]                                      # final encoder BN
        backbone.append([L[k] for k in keys])
    head = [L[k] for k in L.keys() if k.startswith("decoder_") or k == "final_conv"]
    return backbone, head


def apply_freeze(model, kind, unfreeze_last):
    """Freeze everything, then unfreeze the head plus the last ``unfreeze_last`` backbone
    blocks. Returns (head_params, unfrozen_backbone_params) for the optimizer groups."""
    for p in model.parameters():
        p.requires_grad = False
    backbone, head = _reg_groups(model) if kind == "reg" else _seg_groups(model)

    head_params = []
    for m in head:
        for p in m.parameters():
            p.requires_grad = True
            head_params.append(p)

    n = max(0, int(unfreeze_last))
    bb_params = []
    for group in (backbone[-n:] if n > 0 else []):
        for m in group:
            for p in m.parameters():
                p.requires_grad = True
                bb_params.append(p)
    return head_params, bb_params


def make_optimizer(head_params, bb_params, ft_lr, backbone_lr_mult, weight_decay, discriminative):
    groups = [{"params": head_params, "lr": ft_lr}]
    if bb_params:
        bb_lr = ft_lr * backbone_lr_mult if discriminative else ft_lr
        groups.append({"params": bb_params, "lr": bb_lr})
    return torch.optim.AdamW(groups, lr=ft_lr, weight_decay=weight_decay)


def make_scheduler(opt, schedule, warmup_epochs, epochs):
    """Linear warmup -> cosine (default) or constant, stepped once per epoch by train_loop."""
    warmup_epochs = max(0, int(warmup_epochs))
    rest = max(1, int(epochs) - warmup_epochs)
    phases, milestones = [], []
    if warmup_epochs > 0:
        phases.append(S.LinearLR(opt, start_factor=0.1, total_iters=warmup_epochs))
        milestones.append(warmup_epochs)
    phases.append(S.CosineAnnealingLR(opt, T_max=rest) if schedule == "cosine"
                  else S.ConstantLR(opt, factor=1.0, total_iters=0))
    if len(phases) == 1:
        return None if (schedule != "cosine" and warmup_epochs == 0) else phases[0]
    return S.SequentialLR(opt, phases, milestones=milestones)


# --------------------------------------------------------------------------- #
# Frozen-backbone BatchNorm policy                                            #
# --------------------------------------------------------------------------- #
def frozen_bn_modules(model):
    """BatchNorm modules whose affine params are frozen (i.e. part of the frozen backbone).
    Their running stats are BUFFERS, so requires_grad=False does NOT stop them updating in
    train() mode -- these are the modules the bn_mode policy governs. (Trainable head/decoder
    BN keep requires_grad=True and are not returned.)"""
    out = []
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            ps = list(m.parameters(recurse=False))
            if ps and not any(p.requires_grad for p in ps):
                out.append(m)
    return out


def make_train_armer(model, bn_mode):
    """Return ``arm(model)`` that re-arms training honouring the frozen-backbone BN policy:
      "frozen": after model.train(), force frozen BN back to .eval() so running stats AND affine
                stay fixed -- a truly frozen backbone (default).
      "adapt" : leave frozen BN in .train() so their running stats re-estimate on the new data;
                their affine gamma/beta stay requires_grad=False, so they do not *learn*.
    For a model with no frozen BN (e.g. regression) this is just model.train()."""
    frozen_bn = frozen_bn_modules(model)
    def arm(m):
        m.train()
        if bn_mode == "frozen":
            for bn in frozen_bn:
                bn.eval()
    return arm


# --------------------------------------------------------------------------- #
# Sidecar config                                                              #
# --------------------------------------------------------------------------- #
def _fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"name": Path(path).name, "sha256": h.hexdigest(),
            "size_bytes": Path(path).stat().st_size}


def write_sidecar(ckpt_path, args, base_path, kind, best_val):
    cfg = {
        "mode": "finetune",
        "backend": "pytorch",
        "experiment": args.experiment,
        "stage": kind,
        "test_set": args.test_set,
        "train_sets": args.train_sets or "(all but test)",
        "seed": args.seed,
        "hyperparameters": {
            "unfreeze_last": args.unfreeze_last,
            "bn_mode": args.bn_mode,
            "ft_lr": args.ft_lr,
            "backbone_lr_mult": args.backbone_lr_mult,
            "discriminative_lr": not args.no_discriminative,
            "weight_decay": args.weight_decay,
            "lr_schedule": args.lr_schedule,
            "warmup_epochs": args.warmup_epochs,
            "grad_clip": args.grad_clip,
            "epochs": args.epochs,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "crop": args.crop,
            "augment": not args.no_augment,
            "augment_strength": args.augment_strength,
            "patience": args.patience,
        },
        "base_model": _fingerprint(base_path),
        "best_val_loss": float(best_val),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    out = Path(ckpt_path).parent / "finetune_config.json"
    out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"[finetune] wrote config sidecar -> {out}", flush=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="DARE2D PyTorch fine-tuning")
    ap.add_argument("--experiment", choices=list(_EXP), required=True)
    ap.add_argument("--base-model", required=True, help="pretrained .pt checkpoint to fine-tune")
    ap.add_argument("--test-set", required=True, help="held-out set number, e.g. 8")
    ap.add_argument("--train-sets", default=None, help="comma list (default = all other sets)")
    ap.add_argument("--all-sets", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--raw-root", default=str(ts.DEFAULT_RAW_ROOT))
    ap.add_argument("--run-name", default=None, help="default = today's date")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=12345)
    # fine-tuning knobs (advanced)
    ap.add_argument("--unfreeze-last", type=int, default=0,
                    help="unfreeze the last N backbone blocks (0 = head only)")
    ap.add_argument("--bn-mode", choices=["frozen", "adapt"], default="frozen",
                    help="frozen-backbone BatchNorm policy: 'frozen' = running stats fixed "
                         "(BN .eval(); true freeze); 'adapt' = running stats re-estimate on the "
                         "new data (weights incl. BN gamma/beta stay frozen)")
    ap.add_argument("--ft-lr", type=float, default=1e-4, help="fine-tune LR for the head")
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1,
                    help="unfrozen-backbone LR = ft_lr * this (discriminative LR)")
    ap.add_argument("--no-discriminative", action="store_true",
                    help="use a single LR for head + backbone instead of discriminative")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lr-schedule", choices=["cosine", "constant"], default="cosine")
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--grad-clip", type=float, default=1.0, help="max grad-norm; 0 = off")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--augment-strength", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=0, help="early-stop epochs; 0 = off")
    args = ap.parse_args()

    bt_name, sub, kind = _EXP[args.experiment]
    test_set = f"set_{args.test_set}"
    if args.train_sets:
        train_sets = [f"set_{s.strip()}" for s in args.train_sets.split(",")]
    else:
        train_sets = [f"set_{s.strip()}" for s in args.all_sets.split(",")
                      if f"set_{s.strip()}" != test_set]

    base = Path(args.base_model)
    if base.suffix.lower() != ".pt":
        raise SystemExit(f"--base-model must be a .pt checkpoint (got '{base.name}'). "
                         "Fine-tuning runs on PyTorch; convert a .h5 via dare2d-torch/convert_to_torch.py.")
    if not base.exists():
        raise SystemExit(f"base model not found: {base}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_name = args.run_name or datetime.now().date().isoformat()
    out_dir, run_name = ts.resolve_run_dir(run_name, sub, f"{test_set}_all_but_target",
                                           ckpt_name="best.pt")
    ckpt_path = out_dir / f"checkpoints_{test_set}_all_but_target" / "best.pt"
    print(f"[finetune] {args.experiment} test={test_set} train={train_sets} device={device}")
    print(f"[finetune] base={base.name} -> {ckpt_path}")

    # 1) model + load pretrained weights FIRST -- fail fast on a wrong-stage / bad checkpoint,
    #    BEFORE the (minutes-long, first-run) data preprocessing.
    model = (M.Regression2dTorch() if kind == "reg" else M.SegmentationUnetTorch()).to(device)
    load_base(model, base, args.experiment)

    # 2) data (reuse the scratch pipeline verbatim)
    train_loader, val_loader = T.build_loaders(
        args.experiment, kind, train_sets, test_set, crop=args.crop, epochs=args.epochs,
        steps=args.steps, batch_size=args.batch_size, seed=args.seed,
        raw_root=args.raw_root, out_dir=out_dir,
        augment=not args.no_augment, aug_strength=args.augment_strength)

    # 3) freeze + discriminative optimizer + warmup/cosine schedule + frozen-BN policy
    head_params, bb_params = apply_freeze(model, kind, args.unfreeze_last)
    discriminative = not args.no_discriminative
    opt = make_optimizer(head_params, bb_params, args.ft_lr, args.backbone_lr_mult,
                         args.weight_decay, discriminative)
    sched = make_scheduler(opt, args.lr_schedule, args.warmup_epochs, args.epochs)
    arm = make_train_armer(model, args.bn_mode)
    n_train = sum(p.numel() for p in head_params + bb_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[finetune] trainable {n_train:,}/{n_total:,} params "
          f"(unfreeze_last={args.unfreeze_last}, bn_mode={args.bn_mode}, "
          f"discriminative={discriminative}, ft_lr={args.ft_lr}, "
          f"backbone_mult={args.backbone_lr_mult}, schedule={args.lr_schedule}, "
          f"warmup={args.warmup_epochs})", flush=True)

    grad_clip = args.grad_clip if args.grad_clip and args.grad_clip > 0 else None
    patience = args.patience if args.patience and args.patience > 0 else None

    # 4) reuse the shared train loop (saves best.pt; emits the widget's progress markers).
    #    on_train_mode=arm enforces the frozen-BN policy after every model.train() re-arm.
    best = T.train_loop(model, opt, train_loader, val_loader, kind, args.epochs, args.steps,
                        ckpt_path, device, lr_scheduler=sched, patience=patience,
                        grad_clip=grad_clip, on_train_mode=arm)

    write_sidecar(ckpt_path, args, base, kind, best)
    print(f"[finetune] DONE. best val_loss={best:.4f}  checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
