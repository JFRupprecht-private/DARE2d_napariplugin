"""PyTorch training backend for DARE2D retraining (native-Windows GPU). [TF env]

Same leave-one-out task as train_split.py, but trains the faithful torch models
(dare2d-torch: Regression2dTorch / SegmentationUnetTorch) on the GPU. The DATA is
produced by DARE2D's own generators (pure numpy/cv2/albumentations), so the crop /
target / mask / augmentation logic is reused verbatim -> zero data divergence. Only
the model + training loop are torch. Runs in the TF env (has both TF and torch+cu124);
no WSL needed. DARE2d-main is read-only.

Output: models/<run>/<reg|seg>_checkpoints/checkpoints_set_<test>_all_but_target/best.pt
(directly loadable by the torch inference backend). HARD never-overwrite; models/best/
is never written.

Usage:
  python training/torch/train.py --experiment regression2d --test-set 8 --run-name 2026-06-20
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SM_FRAMEWORK", "tf.keras")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

_HERE = Path(__file__).resolve().parent           # training/torch/
PROJECT_ROOT = _HERE.parents[1]                    # repo root (training/torch -> root)
# training/ holds prepare.py; training/tf/ holds train_split.py; dare2d-torch/ holds models_torch.
for p in (str(_HERE.parent), str(_HERE.parent / "tf"), str(PROJECT_ROOT / "dare2d-torch")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import ConcatDataset, DataLoader, Dataset  # noqa: E402

import prepare as prep  # noqa: E402
import train_split as ts  # noqa: E402  (reuse config build + generator wiring + run resolver)
import models_torch as M  # noqa: E402  (faithful torch models)

# experiment -> (batch_training cfg, output subfolder, kind)
_EXP = {
    "regression2d": ("regression2d", "regression_checkpoints", "reg"),
    "segmentation2d": ("center_detection2d", "segmentation_checkpoints", "seg"),
}


class GenDataset(Dataset):
    """Wrap a DARE2D generator (Sequence-style) as a torch Dataset. NHWC -> NCHW."""

    def __init__(self, gen, kind):
        self.gen = gen
        self.kind = kind

    def __len__(self):
        return len(self.gen)

    def __getitem__(self, i):
        X, Y = self.gen[i]
        x = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).permute(2, 0, 1)
        if self.kind == "reg":
            y = (torch.tensor(np.asarray(Y["length_output"], np.float32)),
                 torch.tensor(np.asarray(Y["angle_output"], np.float32)))
        else:  # seg: Y is a (H,W) {0,1} mask
            y = torch.from_numpy(np.ascontiguousarray(Y, dtype=np.float32))[None]  # (1,H,W)
        return x, y


def _cycle(loader):
    while True:
        for b in loader:
            yield b


def reg_loss(pred, target):
    length, angle = pred
    t_len, t_ang = target
    return F.mse_loss(length, t_len) + F.mse_loss(angle, t_ang)


def seg_loss(pred, target, weight_scaling=4.0):
    p = pred.clamp(1e-7, 1 - 1e-7)
    w = target * weight_scaling + 1.0
    return F.binary_cross_entropy(p, target, weight=w)


def evaluate(model, loader, kind, device, max_batches=None):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader):
            if max_batches and bi >= max_batches:
                break
            x = x.to(device)
            if kind == "reg":
                y = (y[0].to(device), y[1].to(device))
                loss = reg_loss(model(x), y)
            else:
                y = y.to(device)
                loss = seg_loss(model(x), y)
            tot += float(loss); n += 1
    model.train()
    return tot / max(n, 1)


def _fmt_eta(seconds: float) -> str:
    """Human-readable ETA, e.g. '1h04m', '12m30s', '45s'."""
    s = int(max(0, seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", choices=list(_EXP), required=True)
    ap.add_argument("--test-set", required=True)
    ap.add_argument("--train-sets", default=None)
    ap.add_argument("--raw-root", default=str(ts.DEFAULT_RAW_ROOT))
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--all-sets", default="1,2,3,4,5,6,7,8")
    args = ap.parse_args()

    bt_name, sub, kind = _EXP[args.experiment]
    test_set = f"set_{args.test_set}"
    if args.train_sets:
        train_sets = [f"set_{s.strip()}" for s in args.train_sets.split(",")]
    else:
        train_sets = [f"set_{s.strip()}" for s in args.all_sets.split(",")
                      if f"set_{s.strip()}" != test_set]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_name = args.run_name or __import__("datetime").date.today().isoformat()
    out_dir, run_name = ts.resolve_run_dir(run_name, sub, f"{test_set}_all_but_target",
                                           ckpt_name="best.pt")
    ckpt_path = out_dir / f"checkpoints_{test_set}_all_but_target" / "best.pt"
    print(f"[torch-train] {args.experiment} test={test_set} train={train_sets} "
          f"device={device}\n[torch-train] -> {ckpt_path}")

    # 1) preprocess (cached)
    print("[phase] preprocessing data (first run only; can take a few minutes)…", flush=True)
    raw_root = Path(args.raw_root)
    prepared_root = PROJECT_ROOT / "data" / "prepared" / f"crop_{args.crop}"
    for name in train_sets + [test_set]:
        _, n = prep.prepare_set(raw_root / name, prepared_root / name, crop_size=args.crop)
        print(f"[prep] {name}: {n} samples", flush=True)

    # 2) build DARE2D generators (reuse train_split wiring) + augmentations
    cfg = ts.build_config(args.experiment, bt_name, prepared_root, out_dir,
                          args.epochs, args.steps, args.batch_size, args.seed)
    proc = ts.SingleSplitProcedure(cfg, out_dir, train_sets, test_set)  # builds proc.sets
    proc.init_datamodule()
    aug = proc.datamodule.get_augmentations()
    train_gens = [proc.sets[n] for n in train_sets]
    for g in train_gens:
        g.set_augmentations(aug)
    test_gen = proc.sets[test_set]

    train_ds = ConcatDataset([GenDataset(g, kind) for g in train_gens])
    val_ds = GenDataset(test_gen, kind)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[torch-train] train samples={len(train_ds)} val samples={len(val_ds)}")

    # 3) model + optimizer
    model = (M.Regression2dTorch() if kind == "reg" else M.SegmentationUnetTorch()).to(device)
    model.train()
    opt = (torch.optim.RMSprop(model.parameters(), lr=1e-4) if kind == "reg"
           else torch.optim.Adam(model.parameters(), lr=1e-4))

    # 4) train loop (steps/epoch caps the pass, cycling the loader like the TF infinite ds)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if ckpt_path.exists():
        raise FileExistsError(f"refusing to overwrite {ckpt_path}")
    best = float("inf")
    it = _cycle(train_loader)
    total_steps = args.epochs * args.steps
    train_t0 = time.time()
    last_log = 0.0
    print(f"[phase] training: {args.epochs} epochs x {args.steps} steps on {device}", flush=True)
    for ep in range(args.epochs):
        t0 = time.time()
        run = 0.0
        for si in range(args.steps):
            x, y = next(it)
            x = x.to(device)
            opt.zero_grad()
            if kind == "reg":
                y = (y[0].to(device), y[1].to(device))
                loss = reg_loss(model(x), y)
            else:
                y = y.to(device)
                loss = seg_loss(model(x), y)
            loss.backward()
            opt.step()
            run += float(loss)
            # throttled heartbeat so the widget bar + terminal advance WITHIN an epoch
            now = time.time()
            if now - last_log >= 2.0:
                done = ep * args.steps + si + 1
                el = now - train_t0
                rate = done / el if el > 0 else 0.0
                eta = (total_steps - done) / rate if rate > 0 else 0.0
                pct = 100.0 * done / total_steps
                nfill = int(round(20 * done / total_steps))
                bar = "#" * nfill + "-" * (20 - nfill)   # ASCII-safe (no cp1252 crash)
                print(f"[step] {ep+1}/{args.epochs} {si+1}/{args.steps} "
                      f"[{bar}] {pct:.0f}% {rate:.1f} it/s "
                      f"eta {_fmt_eta(eta)} (this stage)", flush=True)
                last_log = now
        val = evaluate(model, val_loader, kind, device, max_batches=50)
        improved = val < best
        if improved:
            best = val
            torch.save(model.state_dict(), ckpt_path)
        print(f"[epoch {ep+1}/{args.epochs}] train_loss={run/args.steps:.4f} "
              f"val_loss={val:.4f}{'  *saved' if improved else ''} "
              f"({time.time()-t0:.1f}s)", flush=True)

    print(f"[torch-train] DONE. best val_loss={best:.4f}  checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
