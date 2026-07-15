"""Tests for horizontal-flip TTA at predict time (P2.3)."""

from __future__ import annotations

import torch
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


def _tiny_ntire_module(tta_hflip: bool = False) -> LOGERLightningModule:
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="loger_fsm_ntire")
    with open_dict(cfg):
        cfg.backbone.pretrained = False
        cfg.backbone.config_overrides = TINY_BACKBONE
        cfg.backbone.image_size = 64
        cfg.data.image_size = 64
        cfg.model.tta_hflip = tta_hflip
    return LOGERLightningModule(cfg)


def test_tta_hflip_disabled_by_default_in_ntire_config():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="loger_fsm_ntire")
    assert cfg.model.tta_hflip is False


def test_predict_step_without_tta_matches_plain_eval_logits():
    module = _tiny_ntire_module(tta_hflip=False)
    module.eval()
    x = torch.randn(2, 3, 64, 64)
    batch = {"pixel_values": x}

    with torch.no_grad():
        result = module.predict_step(batch, 0)
        expected_logits = module._eval_logits(x)

    assert torch.equal(result["logit"], expected_logits.detach().cpu())


def test_predict_step_with_tta_averages_flipped_logits():
    module = _tiny_ntire_module(tta_hflip=True)
    module.eval()
    x = torch.randn(2, 3, 64, 64)
    batch = {"pixel_values": x}

    with torch.no_grad():
        result = module.predict_step(batch, 0)
        native = module._eval_logits(x)
        flipped = module._eval_logits(torch.flip(x, dims=[-1]))
        expected = 0.5 * (native + flipped)

    assert torch.allclose(result["logit"], expected.detach().cpu(), atol=1e-6)
    # Sanity: TTA actually changes the result relative to the plain forward
    # (random tiny model has no flip symmetry, so native != flipped in general).
    assert not torch.equal(result["logit"], native.detach().cpu())


def test_predict_step_pred_uses_calibrated_threshold_with_tta():
    module = _tiny_ntire_module(tta_hflip=True)
    module.eval()
    module.best_threshold = 0.9
    x = torch.randn(2, 3, 64, 64)
    batch = {"pixel_values": x}

    with torch.no_grad():
        result = module.predict_step(batch, 0)

    assert torch.equal(result["pred"], (result["prob"] >= 0.9).long())
