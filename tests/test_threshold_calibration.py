"""Tests for accuracy-maximising decision-threshold calibration (P1.3)."""

from __future__ import annotations

import lightning as L
import numpy as np
import pytest
import torch
from hydra import compose, initialize
from omegaconf import open_dict

from src.lightning.loger_module import LOGERLightningModule
from src.training.metrics import best_accuracy_threshold

TINY_BACKBONE = dict(
    hidden_size=96,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=192,
    patch_size=16,
    image_size=64,
)


def _tiny_ntire_module() -> LOGERLightningModule:
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="loger_fsm_ntire")
    with open_dict(cfg):
        cfg.backbone.pretrained = False
        cfg.backbone.config_overrides = TINY_BACKBONE
        cfg.backbone.image_size = 64
        cfg.data.image_size = 64
    return LOGERLightningModule(cfg)


# ---- src.training.metrics.best_accuracy_threshold --------------------------


def test_best_accuracy_threshold_known_optimum():
    # 90 reals @ 0.3, 5 fakes @ 0.9 (well separated), 5 fakes @ 0.2 (ambiguous,
    # indistinguishable from the reals). The accuracy-maximising threshold
    # accepts only the well-separated fakes rather than lowering the bar and
    # misclassifying all 90 reals to also catch the ambiguous ones.
    labels = np.array([0] * 90 + [1] * 5 + [1] * 5)
    scores = np.array([0.3] * 90 + [0.9] * 5 + [0.2] * 5)

    threshold, acc = best_accuracy_threshold(labels, scores)

    assert acc == pytest.approx(0.95)
    assert threshold == pytest.approx(0.9)


def test_best_accuracy_threshold_perfectly_separated():
    labels = np.array([0] * 20 + [1] * 20)
    scores = np.array([0.1] * 20 + [0.9] * 20)

    threshold, acc = best_accuracy_threshold(labels, scores)

    assert acc == pytest.approx(1.0)
    assert 0.1 < threshold <= 0.9


def test_best_accuracy_threshold_clips_inf_sentinel():
    # roc_curve's first threshold ("predict nothing positive") is `inf` in
    # this sklearn version; it must never leak out as the returned threshold.
    labels = np.array([0, 0, 0, 1])
    scores = np.array([0.1, 0.1, 0.1, 0.1])

    threshold, _ = best_accuracy_threshold(labels, scores)
    assert np.isfinite(threshold)


# ---- LOGERLightningModule integration --------------------------------------


def _prepare_for_direct_finalise_eval_call(module: LOGERLightningModule) -> None:
    """Let `_finalise_eval` run outside an active Trainer loop.

    It reads `self.trainer.world_size` (raises RuntimeError if never attached
    to a Trainer) and calls `self.log`/`self.log_dict` (raises
    MisconfigurationException outside an active fit/validate loop's result
    collection). Both are Lightning's own plumbing, not this feature's logic,
    so they're stubbed out for this direct, non-Trainer-driven call.
    """
    module.trainer = L.Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False)
    module.log = lambda *args, **kwargs: None
    module.log_dict = lambda *args, **kwargs: None


def test_finalise_eval_calibrates_best_threshold_on_val():
    module = _tiny_ntire_module()
    _prepare_for_direct_finalise_eval_call(module)
    assert module.best_threshold == 0.5  # default before any validation epoch

    scores = [np.array([0.3] * 90 + [0.9] * 5 + [0.2] * 5)]
    labels = [np.array([0] * 90 + [1] * 5 + [1] * 5)]
    module._finalise_eval(scores, labels, "val")

    assert module.best_threshold == pytest.approx(0.9)


def test_finalise_eval_skips_calibration_on_single_class_batch():
    module = _tiny_ntire_module()
    _prepare_for_direct_finalise_eval_call(module)
    module.best_threshold = 0.7  # simulate a previously calibrated value

    scores = [np.array([0.1, 0.2, 0.3])]
    labels = [np.array([0, 0, 0])]  # single class: AUC/threshold undefined
    module._finalise_eval(scores, labels, "val")

    assert module.best_threshold == 0.7  # left untouched, not reset to 0.5


def test_predict_step_uses_calibrated_threshold():
    module = _tiny_ntire_module()
    module.eval()
    module.best_threshold = 0.9

    batch = {"pixel_values": torch.randn(2, 3, 64, 64), "label": torch.tensor([0, 1])}
    with torch.no_grad():
        result = module.predict_step(batch, 0)

    assert torch.equal(result["pred"], (result["prob"] >= 0.9).long())


def test_best_threshold_persists_across_checkpoint_hooks():
    module = _tiny_ntire_module()
    module.best_threshold = 0.732

    checkpoint: dict = {}
    module.on_save_checkpoint(checkpoint)
    assert checkpoint["best_threshold"] == pytest.approx(0.732)

    fresh = _tiny_ntire_module()
    assert fresh.best_threshold == 0.5
    fresh.on_load_checkpoint(checkpoint)
    assert fresh.best_threshold == pytest.approx(0.732)


def test_on_load_checkpoint_defaults_to_half_for_checkpoints_without_it():
    """Checkpoints saved before this feature existed have no `best_threshold`."""
    module = _tiny_ntire_module()
    module.best_threshold = 0.9  # simulate a stale in-memory value
    module.on_load_checkpoint({})
    assert module.best_threshold == 0.5
