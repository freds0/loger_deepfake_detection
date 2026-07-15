"""Tests for the Single-Center Loss option on global features (PLAN_v0.2 V10)."""

from __future__ import annotations

import torch

from src.losses.loger import LOGERLoss
from src.losses.single_center_loss import SingleCenterLoss


def _batch(seed: int = 0):
    torch.manual_seed(seed)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])  # both classes present
    return (
        torch.randn(8),        # fused
        torch.randn(8),        # global
        torch.randn(8),        # local
        torch.randn(8, 20),    # patch diffs
        labels,
        torch.randn(8, 32),    # scl features
    )


def test_scl_weight_zero_ignores_features():
    fused, glob, loc, pd, labels, feats = _batch()
    loss = LOGERLoss()  # scl_weight=0 (default)
    with_feats, parts = loss(fused, glob, loc, pd, labels, scl_features=feats)
    without_feats, _ = loss(fused, glob, loc, pd, labels)
    assert torch.allclose(with_feats, without_feats)
    assert "scl" not in parts


def test_scl_weight_adds_single_center_loss():
    fused, glob, loc, pd, labels, feats = _batch()
    base_total, _ = LOGERLoss()(fused, glob, loc, pd, labels)
    total, parts = LOGERLoss(scl_weight=1.0, scl_margin=0.01)(
        fused, glob, loc, pd, labels, scl_features=feats
    )
    manual = SingleCenterLoss(margin=0.01)(feats, labels)
    assert torch.allclose(total, base_total + manual, atol=1e-5)
    assert "scl" in parts
    assert torch.allclose(parts["scl"], manual.detach(), atol=1e-5)


def test_scl_gradient_flows_to_features():
    fused, glob, loc, pd, labels, _ = _batch()
    feats = torch.randn(8, 32, requires_grad=True)
    total, _ = LOGERLoss(scl_weight=1.0)(fused, glob, loc, pd, labels, scl_features=feats)
    total.backward()
    assert feats.grad is not None
    assert torch.isfinite(feats.grad).all()
