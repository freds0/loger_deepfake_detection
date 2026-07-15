"""Tests for discriminative backbone/head learning rates (PLAN_v0.2 V8)."""

from __future__ import annotations

from hydra import compose, initialize
from omegaconf import open_dict

from src.lightning.loger_module import LOGERLightningModule

TINY = dict(
    hidden_size=96,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=192,
    patch_size=16,
    image_size=64,
)


def _module(multiplier: float) -> LOGERLightningModule:
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="ntire_v2_base")
    with open_dict(cfg):
        cfg.backbone.pretrained = False
        cfg.backbone.config_overrides = TINY
        cfg.backbone.image_size = 64
        cfg.data.image_size = 64
        cfg.optimizer.head_lr_multiplier = multiplier
    return LOGERLightningModule(cfg)


def _optimizer(module: LOGERLightningModule):
    result = module.configure_optimizers()
    return result["optimizer"] if isinstance(result, dict) else result


def test_single_group_when_multiplier_is_one():
    opt = _optimizer(_module(1.0))
    assert len(opt.param_groups) == 1


def test_two_groups_with_scaled_head_lr():
    module = _module(10.0)
    opt = _optimizer(module)
    assert len(opt.param_groups) == 2

    # initial_lr (set by the scheduler before warmup scaling) holds the base
    # rates: backbone at lr, heads at 10x. The live `lr` is warmup-scaled but
    # keeps the same 10x ratio between the groups.
    base_lrs = sorted(g["initial_lr"] for g in opt.param_groups)
    assert base_lrs[0] == 3.0e-5
    assert abs(base_lrs[1] - 3.0e-4) < 1e-12
    live = sorted(g["lr"] for g in opt.param_groups)
    assert abs(live[1] / live[0] - 10.0) < 1e-9

    # The two groups partition exactly the trainable parameters.
    grouped = sum(len(g["params"]) for g in opt.param_groups)
    assert grouped == len(list(module.model.trainable_parameters()))
