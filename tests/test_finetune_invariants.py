"""Invariant tests for the DARE2D PyTorch fine-tuning path.

Pure-CPU, no data / no GPU / no preprocessing -- these pin the *delicate* fine-tuning invariants that
are otherwise only verified by hand and could regress silently on the next edit:

  1. the optimizer trains EXACTLY the params with requires_grad=True (no frozen param handed to the
     optimizer, no trainable param omitted);
  2. the frozen-backbone BatchNorm policy actually holds after ``arm(model)`` (frozen BN in eval,
     trainable BN in train; ``adapt`` leaves frozen BN in train);
  3. the train/val index split is disjoint, covering, >=1 val per non-empty set, and deterministic;
  4. a fine-tuned checkpoint saved with ``torch.save(state_dict)`` round-trips exactly through the
     inference loaders.

Run:  python -m pytest tests/test_finetune_invariants.py -q
   or: python tests/test_finetune_invariants.py        (plain-script fallback, no pytest needed)
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Make the training + model packages importable regardless of CWD (mirrors the training scripts).
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (_ROOT / "training" / "torch", _ROOT / "training", _ROOT / "training" / "tf",
           _ROOT / "dare2d-torch"):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

import torch  # noqa: E402

import finetune as F        # noqa: E402  apply_freeze / make_optimizer / make_train_armer / frozen_bn_modules
import train as T           # noqa: E402  _per_set_val_split
import models_torch as M    # noqa: E402  faithful torch models
import torch_backend as B   # noqa: E402  inference-side strict loaders

_BN = torch.nn.modules.batchnorm._BatchNorm


def _trainable_bn(model):
    return [m for m in model.modules()
            if isinstance(m, _BN) and any(p.requires_grad for p in m.parameters(recurse=False))]


# --------------------------------------------------------------------------- #
# 1. optimizer param groups == requires_grad params (exactly)                 #
# --------------------------------------------------------------------------- #
def _check_opt_parity(model, kind, unfreeze_last):
    head, bb = F.apply_freeze(model, kind, unfreeze_last)
    opt = F.make_optimizer(head, bb, 1e-4, 0.1, 1e-4, discriminative=True)
    opt_ids = {id(p) for grp in opt.param_groups for p in grp["params"]}
    req_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert opt_ids == req_ids, (kind, unfreeze_last, len(opt_ids), len(req_ids))
    # nothing frozen leaked into the optimizer
    frozen_ids = {id(p) for p in model.parameters() if not p.requires_grad}
    assert opt_ids.isdisjoint(frozen_ids), (kind, unfreeze_last)


def test_optimizer_requires_grad_parity_regression():
    _check_opt_parity(M.Regression2dTorch(), "reg", 0)
    _check_opt_parity(M.Regression2dTorch(), "reg", 2)


def test_optimizer_requires_grad_parity_segmentation():
    _check_opt_parity(M.SegmentationUnetTorch(), "seg", 0)
    _check_opt_parity(M.SegmentationUnetTorch(), "seg", 2)


# --------------------------------------------------------------------------- #
# 2. frozen-backbone BatchNorm policy after arm()                            #
# --------------------------------------------------------------------------- #
def test_frozen_bn_arm_policy_segmentation():
    model = M.SegmentationUnetTorch()
    F.apply_freeze(model, "seg", 0)                 # head only -> encoder BN frozen
    frozen = F.frozen_bn_modules(model)
    assert len(frozen) > 0, "expected frozen encoder BatchNorm layers in the seg U-Net"

    # "frozen": after arming train mode, frozen BN forced back to eval(); trainable BN stay in train()
    F.make_train_armer(model, "frozen")(model)
    assert all(bn.training is False for bn in frozen)
    assert all(bn.training is True for bn in _trainable_bn(model))

    # "adapt": frozen BN left in train() so running stats re-estimate (affine still requires_grad=False)
    F.make_train_armer(model, "adapt")(model)
    assert all(bn.training is True for bn in frozen)


def test_regression_has_no_batchnorm():
    # The frozen-BN policy is a no-op for regression; assert the assumption the code relies on.
    model = M.Regression2dTorch()
    assert not any(isinstance(m, _BN) for m in model.modules())


# --------------------------------------------------------------------------- #
# 3. train/val split invariants (F3 helper)                                  #
# --------------------------------------------------------------------------- #
def test_per_set_val_split_invariants():
    lengths = [100, 50, 3, 1, 0]
    tr, va = T._per_set_val_split(lengths, 0.1, seed=12345)
    assert len(tr) == len(va) == len(lengths)
    for n, ti, vi in zip(lengths, tr, va):
        s_tr, s_va = set(ti), set(vi)
        assert s_tr.isdisjoint(s_va), n                      # disjoint
        assert s_tr | s_va == set(range(n)), n               # covering
        if n > 0:
            assert len(vi) >= 1, n                            # >=1 val per non-empty set
        else:
            assert not ti and not vi                          # empty set contributes nothing

    # determinism: same seed -> identical partition
    tr2, va2 = T._per_set_val_split(lengths, 0.1, seed=12345)
    assert tr == tr2 and va == va2

    # small-data (the F3 case): no globally-empty val split even when every set rounds to zero
    _, va_small = T._per_set_val_split([3, 3, 3], 0.1, seed=7)
    assert sum(len(v) for v in va_small) >= 1


# --------------------------------------------------------------------------- #
# 4. save -> load round-trip fidelity                                        #
# --------------------------------------------------------------------------- #
def _roundtrip(model, loader_fn):
    sd = model.state_dict()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "best.pt")
        torch.save(sd, p)
        loader_fn(p, device="cpu")                            # strict load must not raise
        reloaded = torch.load(p, map_location="cpu", weights_only=True)
    assert set(reloaded.keys()) == set(sd.keys())
    for k in sd:
        assert torch.equal(sd[k], reloaded[k]), k


def test_roundtrip_regression():
    _roundtrip(M.Regression2dTorch(), B.load_torch_regression)


def test_roundtrip_segmentation():
    _roundtrip(M.SegmentationUnetTorch(), B.load_torch_segmentation)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nALL {len(tests)} TESTS PASSED")
