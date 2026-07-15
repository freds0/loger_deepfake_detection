"""Tests for per-shard/per-manipulation metric breakdown on eval (P1.4)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import lightning as L
import numpy as np
import pytest
from hydra import compose, initialize
from omegaconf import open_dict

from src.lightning.loger_module import LOGERLightningModule

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


def _prepare_for_direct_finalise_eval_call(module: LOGERLightningModule) -> None:
    """See tests/test_threshold_calibration.py for why trainer/log are stubbed."""
    module.trainer = L.Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False)
    module.log = MagicMock()
    module.log_dict = MagicMock()


def _logged_names(mock: MagicMock) -> list[str]:
    return [call.args[0] for call in mock.call_args_list]


def _paths_for_shard(shard: str, n: int) -> list[str]:
    return [f"/root/{shard}/images/img_{i:04d}.jpg" for i in range(n)]


def test_per_group_metrics_isolates_a_bad_shard():
    module = _tiny_ntire_module()
    _prepare_for_direct_finalise_eval_call(module)

    # shard_0: 60 samples, perfectly separated (auc ~= 1).
    # shard_1: 60 samples, scores inverted w.r.t. label (auc ~= 0).
    scores = [np.concatenate([np.array([0.1] * 30 + [0.9] * 30), np.array([0.9] * 30 + [0.1] * 30)])]
    labels = [np.array([0] * 30 + [1] * 30 + [0] * 30 + [1] * 30)]
    paths = [_paths_for_shard("shard_0", 60) + _paths_for_shard("shard_1", 60)]

    module._finalise_eval(scores, labels, "val", paths=paths)

    logged = {call.args[0]: call.args[1] for call in module.log.call_args_list}
    assert logged["val/auc/shard_0"] == pytest.approx(1.0, abs=0.01)
    assert logged["val/auc/shard_1"] == pytest.approx(0.0, abs=0.01)


def test_per_group_metrics_skips_small_groups():
    module = _tiny_ntire_module()
    _prepare_for_direct_finalise_eval_call(module)

    # shard_0: 60 samples (qualifies). shard_1: 10 samples (< 50, skipped).
    scores = [np.concatenate([np.array([0.1] * 30 + [0.9] * 30), np.array([0.2] * 5 + [0.8] * 5)])]
    labels = [np.array([0] * 30 + [1] * 30 + [0] * 5 + [1] * 5)]
    paths = [_paths_for_shard("shard_0", 60) + _paths_for_shard("shard_1", 10)]

    module._finalise_eval(scores, labels, "val", paths=paths)

    names = _logged_names(module.log)
    assert any("val/auc/shard_0" in n for n in names)
    assert not any("shard_1" in n for n in names)


def test_per_group_metrics_skips_single_class_groups():
    module = _tiny_ntire_module()
    _prepare_for_direct_finalise_eval_call(module)

    # shard_0: 60 samples, both classes (qualifies).
    # shard_1: 60 samples, all real -- AUC undefined, must be skipped.
    scores = [np.concatenate([np.array([0.1] * 30 + [0.9] * 30), np.array([0.3] * 60)])]
    labels = [np.array([0] * 30 + [1] * 30 + [0] * 60)]
    paths = [_paths_for_shard("shard_0", 60) + _paths_for_shard("shard_1", 60)]

    module._finalise_eval(scores, labels, "val", paths=paths)

    names = _logged_names(module.log)
    assert any("val/auc/shard_0" in n for n in names)
    assert not any("shard_1" in n for n in names)


def test_per_group_metrics_skipped_entirely_under_ddp():
    module = _tiny_ntire_module()
    module.trainer = SimpleNamespace(world_size=2)
    module.log = MagicMock()
    module.log_dict = MagicMock()
    module.all_gather = lambda tensor, *a, **kw: tensor  # stand-in: already "gathered"

    scores = [np.array([0.1] * 30 + [0.9] * 30)]
    labels = [np.array([0] * 30 + [1] * 30)]
    paths = [_paths_for_shard("shard_0", 60)]

    module._finalise_eval(scores, labels, "val", paths=paths)

    names = _logged_names(module.log)
    assert not any("auc/shard_0" in n for n in names)
    # The pooled metric itself still gets logged (via log_dict), just not the
    # per-shard breakdown.
    module.log_dict.assert_called_once()


def test_per_group_metrics_noop_without_paths():
    module = _tiny_ntire_module()
    _prepare_for_direct_finalise_eval_call(module)

    scores = [np.array([0.1] * 30 + [0.9] * 30)]
    labels = [np.array([0] * 30 + [1] * 30)]

    module._finalise_eval(scores, labels, "val")  # paths=None, must not raise

    names = _logged_names(module.log)
    assert not any("auc/" in n for n in names)
