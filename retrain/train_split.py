"""Single leave-one-out retraining: train on a subset of sets, test on one held-out set.

Reuses DARE2D's own training stack (BatchTrainingProcedure / datamodule / trainer /
losses / callbacks) for ONE split, overriding only:
  - which sets to build (just the chosen train + test, not all 8),
  - the datamodule init (use a real placeholder generator, avoid the default
    data_dir/{train,val,test} folders the original code requires),
  - the checkpoint output -> models/<run>/<reg|seg>_checkpoints/
    checkpoints_set_<test>_all_but_target/best.h5, with a HARD never-overwrite guard.

DARE2d-main is imported read-only. Raw data is preprocessed once (retrain.prepare) into
data/prepared/crop_<n>/set_<k>/ before training. models/best/ is never written.

Usage (TF env):
  python retrain/train_split.py --experiment regression2d --test-set 8 \
      --run-name 2026-06-20 --epochs 50 --steps 1000 [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("SM_FRAMEWORK", "tf.keras")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
_REPO = PROJECT_ROOT / "DARE2d-main"
for p in (str(_REPO), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import hydra  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from scripts.batch_train.batch_train_eval import BatchTrainingProcedure  # noqa: E402
import prepare as prep  # noqa: E402  (retrain/prepare.py)

CONFIG_DIR = str(_REPO / "config")
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "neuroepithelium" / "neuroepithelium"
MODELS_DIR = PROJECT_ROOT / "models"

# experiment -> (batch_training config name, output checkpoint subfolder)
_EXP = {
    "regression2d": ("regression2d", "regression_checkpoints"),
    "segmentation2d": ("center_detection2d", "segmentation_checkpoints"),
}


class SingleSplitProcedure(BatchTrainingProcedure):
    def __init__(self, config, out_dir, train_set_names, test_set_name):
        self.out_dir = Path(out_dir)
        self.train_set_names = list(train_set_names)
        self.test_set_name = test_set_name
        self._ckpt_path = None
        super().__init__(config)  # -> init_generators(), load_scores()

    # build only the sets we need (raw->prepared paths already resolved in cfg.batch_training.sets)
    def init_generators(self):
        sets = self.config.batch_training.sets
        gen_cfg = self.config.batch_training.generator
        self.sets = {}
        for name in self.train_set_names + [self.test_set_name]:
            gen_cfg.data_folder = sets[name]
            self.sets[name] = hydra.utils.instantiate(gen_cfg)

    # avoid the default data_dir/{train,val,test} load: use the (already-loaded) test gen
    def init_datamodule(self):
        cls = hydra.utils.get_class(self.config.datamodule._target_)
        ph = self.sets[self.test_set_name]
        self.datamodule = cls(train=ph, val=ph, test=ph, batch_size=self.config.batch_size)

    # checkpoint -> our run folder; NEVER overwrite an existing best.h5
    def update_checkpoint_callback(self, name):
        import tensorflow.keras.callbacks as KC

        ckpt = self.out_dir / f"checkpoints_{name}" / "best.h5"
        if ckpt.exists():
            raise FileExistsError(
                f"refusing to overwrite existing checkpoint: {ckpt}. "
                "Choose another run name."
            )
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        self.callbacks = [c for c in self.callbacks
                          if not isinstance(c, KC.ModelCheckpoint)]
        self.callbacks.append(KC.ModelCheckpoint(
            filepath=str(ckpt), monitor="val_loss", save_best_only=True,
            mode="min", verbose=1))
        self._ckpt_path = str(ckpt)

    # test from our explicit checkpoint path (not hydra's output_dir)
    def test(self, checkpoint_folder="checkpoints"):
        if not self.config.get("test"):
            return
        for cb in self.callbacks:
            if hasattr(cb, "set_generator"):
                cb.set_generator(self.datamodule.test)
            if hasattr(cb, "mode"):
                cb.mode = "test"
        self.trainer.test(model_handler=self.model, datamodule=self.datamodule,
                          checkpoint=self._ckpt_path, callbacks=self.callbacks)

    def run_split(self):
        self.init()
        keys = [self.test_set_name, "all_but_target"]  # -> checkpoints_set_<n>_all_but_target
        train_gens = [self.sets[n] for n in self.train_set_names]
        val_gens = [self.sets[self.test_set_name]]
        self.train_eval(keys, train_gens, val_gens, test_set=val_gens)
        return self.scores, self._ckpt_path


def resolve_run_dir(run_name, sub, exp_name):
    """Return models/<run>/<sub>; auto-suffix run if its best.h5 already exists.
    Never returns models/best/."""
    base = run_name
    i = 1
    while True:
        run = "best_REFUSED" if base == "best" else (base if i == 1 else f"{base}_{i}")
        ckpt = MODELS_DIR / run / sub / f"checkpoints_{exp_name}" / "best.h5"
        if run != "best" and not ckpt.exists():
            return MODELS_DIR / run / sub, run
        i += 1


def build_config(experiment, bt_name, data_dir, out_dir, epochs, steps, batch_size, seed):
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
            cfg = compose(
                config_name="train",
                overrides=[f"experiment={experiment}", f"batch_training={bt_name}",
                           "test=True", "print_config=False"],
                return_hydra_config=True,
            )
    finally:
        GlobalHydra.instance().clear()
    # Rebuild as a FULLY-WRITABLE config (compose returns a read-only tree) and drop the
    # read-only ``hydra`` node -- we override test()/original_work_dir, so it's not needed.
    container = OmegaConf.to_container(cfg, resolve=False, throw_on_missing=False)
    container.pop("hydra", None)
    cfg = OmegaConf.create(container)
    # set absolute paths / hparams AFTER compose (avoids Windows-path override parsing)
    cfg.data_dir = str(data_dir)
    cfg.original_work_dir = str(out_dir)
    cfg.epochs = int(epochs)
    cfg.steps_per_epoch = int(steps)
    if batch_size:
        cfg.batch_size = int(batch_size)
    cfg.seed = int(seed)
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", choices=list(_EXP), required=True)
    ap.add_argument("--test-set", required=True, help="held-out set number, e.g. 8")
    ap.add_argument("--train-sets", default=None,
                    help="comma list (default = all other sets 1..8)")
    ap.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    ap.add_argument("--run-name", default=None, help="default = today's date")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--all-sets", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--dry-run", action="store_true", help="build everything, skip fit")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    all_sets = [s.strip() for s in args.all_sets.split(",") if s.strip()]
    test_set = f"set_{args.test_set}"
    if args.train_sets:
        train_sets = [f"set_{s.strip()}" for s in args.train_sets.split(",")]
    else:
        train_sets = [f"set_{s}" for s in all_sets if f"set_{s}" != test_set]

    bt_name, sub = _EXP[args.experiment]
    run_name = args.run_name or __import__("datetime").date.today().isoformat()
    out_dir, run_name = resolve_run_dir(run_name, sub, f"{test_set}_all_but_target")
    print(f"[retrain] experiment={args.experiment} test={test_set} "
          f"train={train_sets}\n[retrain] -> {out_dir}")

    # 1) preprocess needed sets (cached) into data/prepared/crop_<n>/set_<k>
    prepared_root = PROJECT_ROOT / "data" / "prepared" / f"crop_{args.crop}"
    for name in train_sets + [test_set]:
        out, n = prep.prepare_set(raw_root / name, prepared_root / name, crop_size=args.crop)
        print(f"[prep] {name}: {n} samples")

    # 2) compose config pointing data_dir at the prepared root
    cfg = build_config(args.experiment, bt_name, prepared_root, out_dir,
                       args.epochs, args.steps, args.batch_size, args.seed)

    # 3) run the single split
    proc = SingleSplitProcedure(cfg, out_dir, train_sets, test_set)
    if args.dry_run:
        proc.init()
        print("[dry-run] datamodule/model/trainer built OK; "
              f"train batches sample shape check:")
        for x, y in proc.datamodule.train.take(1):
            import numpy as np
            xs = x.shape if hasattr(x, "shape") else np.asarray(x).shape
            print(f"  X {xs}  y {type(y).__name__}")
        return
    # run INSIDE the run folder so callback artifacts (log.txt, history.log, scores.json,
    # tensorboard logs, ...) land in models/<run>/ (git-ignored), not the repo root.
    run_root = out_dir.parent
    run_root.mkdir(parents=True, exist_ok=True)
    cwd0 = os.getcwd()
    os.chdir(run_root)
    try:
        scores, ckpt = proc.run_split()
    finally:
        os.chdir(cwd0)
    print(f"[retrain] DONE. checkpoint: {ckpt}")
    print(f"[retrain] scores: {scores}")


if __name__ == "__main__":
    main()
