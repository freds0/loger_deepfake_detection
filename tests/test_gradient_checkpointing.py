"""Tests for backbone gradient checkpointing (P2.2).

Uses the tiny, randomly-initialised SigLIP tower (no checkpoint download) so
these run on CPU in seconds. Correctness is verified deterministically:
checkpointing must not change the forward output at all (it only changes what
gets recomputed for backward), and gradients must match a non-checkpointed
run with identical weights -- a stronger, more portable guarantee than a
GPU memory-delta measurement.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from src.models.backbone import enable_gradient_checkpointing
from src.models.loger import LOGERModel

TINY = dict(
    hidden_size=96,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=192,
    patch_size=16,
    image_size=64,
)


def _tiny_full_model(gradient_checkpointing: bool) -> LOGERModel:
    return LOGERModel(
        backbone_name="siglip2",
        peft_mode="full",  # every backbone param trains -> real checkpointing benefit
        head_hidden_dim=32,
        pretrained=False,
        backbone_config_overrides=TINY,
        gradient_checkpointing=gradient_checkpointing,
    )


def test_enable_gradient_checkpointing_raises_clear_error_when_unsupported():
    plain = nn.Linear(4, 4)  # no `gradient_checkpointing_enable` at all
    with pytest.raises(RuntimeError, match="MyBackbone"):
        enable_gradient_checkpointing(plain, "MyBackbone")


def test_enable_gradient_checkpointing_wraps_unsupported_architecture_error():
    fake = MagicMock()
    fake.gradient_checkpointing_enable.side_effect = ValueError("does not support gradient checkpointing")
    with pytest.raises(RuntimeError, match="MyBackbone"):
        enable_gradient_checkpointing(fake, "MyBackbone")


def test_enable_gradient_checkpointing_uses_non_reentrant():
    fake = MagicMock()
    enable_gradient_checkpointing(fake, "MyBackbone")
    fake.gradient_checkpointing_enable.assert_called_once_with(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )


def test_gradient_checkpointing_flag_reaches_every_encoder_layer():
    model = _tiny_full_model(gradient_checkpointing=True)
    layers = model.backbone.vision_model.encoder.layers
    assert len(layers) == TINY["num_hidden_layers"]
    assert all(layer.gradient_checkpointing for layer in layers)


def test_gradient_checkpointing_disabled_by_default():
    model = _tiny_full_model(gradient_checkpointing=False)
    layers = model.backbone.vision_model.encoder.layers
    assert not any(layer.gradient_checkpointing for layer in layers)


def test_gradient_checkpointing_forward_and_gradients_match_without_it():
    torch.manual_seed(0)
    baseline = _tiny_full_model(gradient_checkpointing=False)
    checkpointed = _tiny_full_model(gradient_checkpointing=True)
    checkpointed.load_state_dict(baseline.state_dict())  # identical weights

    baseline.train()
    checkpointed.train()
    x = torch.randn(4, 3, 64, 64)

    out_baseline = baseline(x, apply_fsm=False)
    out_checkpointed = checkpointed(x, apply_fsm=False)
    # Checkpointing only changes what's recomputed for backward, never the
    # forward numerics -- must match exactly, not just approximately.
    assert torch.equal(out_baseline.logits, out_checkpointed.logits)

    out_baseline.logits.sum().backward()
    out_checkpointed.logits.sum().backward()

    baseline_grads = dict(baseline.named_parameters())
    checkpointed_grads = dict(checkpointed.named_parameters())
    for name, p in baseline_grads.items():
        if p.grad is None:
            assert checkpointed_grads[name].grad is None
            continue
        assert torch.allclose(p.grad, checkpointed_grads[name].grad, atol=1e-5), name
