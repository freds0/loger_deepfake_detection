"""Tests for the soft log-sum-exp local pooling option (PLAN_v0.2 V9)."""

from __future__ import annotations

import torch

from src.models.loger import LOGERModel, lse_pool, mil_topk_pool

TINY = dict(
    hidden_size=96,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=192,
    patch_size=16,
    image_size=64,
)


def _tiny_model(local_pool: str = "topk", pool_temperature: float = 1.0) -> LOGERModel:
    return LOGERModel(
        backbone_name="siglip2",
        peft_mode="lora",
        lora_r=4,
        lora_alpha=4.0,
        local_pool=local_pool,
        pool_temperature=pool_temperature,
        head_hidden_dim=32,
        pretrained=False,
        backbone_config_overrides=TINY,
    )


def test_lse_approaches_mean_at_high_temperature():
    # Residual is O(var / tau) (the -tau*log(N) term cancels the log(N) offset),
    # so a large tau is needed to hit mean within 1e-3.
    torch.manual_seed(0)
    d = torch.randn(4, 50)
    assert torch.allclose(lse_pool(d, 1000.0), d.mean(dim=1), atol=1e-3)


def test_lse_approaches_max_at_low_temperature():
    # As tau -> 0 the -tau*log(N) correction vanishes and LSE -> max.
    torch.manual_seed(0)
    d = torch.randn(4, 50)
    assert torch.allclose(lse_pool(d, 1e-4), d.max(dim=1).values, atol=1e-3)


def test_lse_gradient_reaches_every_patch():
    d = torch.randn(2, 10, requires_grad=True)
    lse_pool(d, 1.0).sum().backward()
    assert (d.grad.abs() > 0).all()  # unlike top-k, no patch is dropped


def test_topk_pool_output_unchanged_by_default():
    model = _tiny_model(local_pool="topk")
    model.eval()
    x = torch.randn(3, 3, 64, 64)
    with torch.no_grad():
        out = model(x, apply_fsm=False)
    assert torch.allclose(out.local_logits, mil_topk_pool(out.patch_diffs, model.topk_ratio))


def test_lse_pool_used_when_configured():
    model = _tiny_model(local_pool="lse", pool_temperature=0.5)
    model.eval()
    x = torch.randn(3, 3, 64, 64)
    with torch.no_grad():
        out = model(x, apply_fsm=False)
    assert torch.allclose(out.local_logits, lse_pool(out.patch_diffs, 0.5))


def test_unknown_local_pool_rejected():
    import pytest

    with pytest.raises(ValueError):
        _tiny_model(local_pool="bogus")
