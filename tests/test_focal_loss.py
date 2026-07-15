"""Tests for the Focal Loss image-level option (PLAN_v0.2 V7)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.losses.loger import FocalLoss, LOGERLoss


def _batch(seed: int = 0):
    torch.manual_seed(seed)
    return (
        torch.randn(16),          # fused
        torch.randn(16),          # global
        torch.randn(16),          # local
        torch.randn(16, 20),      # patch diffs
        torch.randint(0, 2, (16,)),
    )


def test_focal_reduces_to_half_bce_at_gamma0_alpha_half():
    torch.manual_seed(1)
    logits = torch.randn(32)
    targets = torch.randint(0, 2, (32,)).float()
    focal = FocalLoss(gamma=0.0, alpha=0.5)
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    assert torch.allclose(focal(logits, targets), 0.5 * bce, atol=1e-6)


def test_focal_downweights_confidently_correct_examples():
    logits = torch.tensor([12.0, 12.0])  # confident, correct positives
    targets = torch.tensor([1.0, 1.0])
    assert FocalLoss(gamma=2.0, alpha=0.5)(logits, targets) < 1e-4


def test_loger_loss_focal_off_is_bit_identical_to_bce():
    fused, glob, loc, pd, labels = _batch()
    bce_loss = LOGERLoss(focal=False)
    total, parts = bce_loss(fused, glob, loc, pd, labels)
    # global/fused terms must equal plain BCE-with-logits.
    labels_f = labels.float()
    assert torch.allclose(parts["global_bce"], F.binary_cross_entropy_with_logits(glob, labels_f))
    assert torch.allclose(parts["fused_bce"], F.binary_cross_entropy_with_logits(fused, labels_f))


def test_loger_loss_focal_only_changes_image_terms():
    fused, glob, loc, pd, labels = _batch()
    focal_loss = LOGERLoss(focal=True, focal_gamma=2.0, focal_alpha=0.25)
    _, parts = focal_loss(fused, glob, loc, pd, labels)
    labels_f = labels.float()
    # local CE stays BCE; global/fused become focal (different from BCE).
    assert torch.allclose(parts["local_ce"], F.binary_cross_entropy_with_logits(loc, labels_f))
    assert not torch.allclose(parts["global_bce"], F.binary_cross_entropy_with_logits(glob, labels_f))
