"""Fast, offline smoke tests for the OSDFD components.

Uses a tiny, randomly-initialised SigLIP vision tower (no checkpoint download)
so the whole model can be exercised on CPU in seconds. Verifies:
  * forward / backward pass and shapes,
  * only PEFT + head params are trainable (frozen backbone),
  * FSM is train-only and mixes only fake samples,
  * SCL / combined loss behave sanely,
  * the metric suite returns the expected keys.
"""

from __future__ import annotations

import numpy as np
import torch

from src.losses.combined import OSDFDLoss
from src.losses.single_center_loss import SingleCenterLoss
from src.models.fsm import ForgeryStyleMixture
from src.models.osdfd import OSDFDModel
from src.models.peft_inject import CDCConfig, LoRAConfig
from src.training.metrics import compute_metrics

# Tiny config: 4 layers, 96-dim, patch16, 64px -> (64/16)^2 = 16 square tokens.
TINY = dict(
    hidden_size=96,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=192,
    patch_size=16,
    image_size=64,
)


def _tiny_model(fsm_prob: float = 0.5) -> OSDFDModel:
    return OSDFDModel(
        lora=LoRAConfig(r=4, alpha=4.0),
        cdc=CDCConfig(bottleneck=16),
        fsm_prob=fsm_prob,
        feature_fusion="global_local",
        head_hidden_dim=32,
        freeze_backbone=True,
        pretrained=False,
        backbone_config_overrides=TINY,
    )


def test_forward_backward_and_trainable_params():
    model = _tiny_model()
    model.train()
    x = torch.randn(6, 3, 64, 64)
    is_fake = torch.tensor([0, 1, 1, 0, 1, 1]).bool()
    domains = torch.tensor([0, 1, 2, 0, 3, 4])

    out = model(x, is_fake=is_fake, domains=domains, apply_fsm=True)
    assert out.logits.shape == (6,)
    assert out.scl_features.shape[0] == 6
    assert out.pooled.shape == (6, 96)

    loss = OSDFDLoss()(out.logits, out.scl_features, is_fake.long())[0]
    loss.backward()

    # Backbone frozen: no grad on vision_model weights; PEFT/head get grads.
    for name, p in model.named_parameters():
        is_peft_or_head = ("lora_" in name) or ("adapter" in name) or name.startswith("head.")
        if is_peft_or_head:
            assert p.requires_grad, f"{name} should be trainable"
        if "vision_model" in name and "lora_" not in name and "adapter" not in name:
            assert not p.requires_grad, f"{name} should be frozen"

    n_train = model.num_trainable_parameters()
    n_total = model.num_total_parameters()
    assert 0 < n_train < n_total


def test_fsm_train_only_and_fake_only():
    fsm = ForgeryStyleMixture(prob=1.0)
    tokens = torch.randn(4, 16, 96)
    is_fake = torch.tensor([0, 1, 0, 1]).bool()
    domains = torch.tensor([0, 1, 0, 2])

    # eval mode: identity
    fsm.eval()
    assert torch.equal(fsm(tokens, is_fake, domains), tokens)

    # train mode: real rows unchanged, at least one fake row changed
    fsm.train()
    torch.manual_seed(0)
    out = fsm(tokens, is_fake, domains)
    assert torch.equal(out[0], tokens[0])  # real
    assert torch.equal(out[2], tokens[2])  # real
    changed = (~torch.isclose(out, tokens).all(dim=(1, 2)))
    assert changed[1] or changed[3]  # some fake row mixed


def test_fsm_single_domain_is_noop():
    fsm = ForgeryStyleMixture(prob=1.0)
    fsm.train()
    tokens = torch.randn(4, 16, 96)
    is_fake = torch.tensor([0, 1, 1, 0]).bool()
    domains = torch.tensor([0, 1, 1, 0])  # only one fake domain -> cannot mix
    assert torch.equal(fsm(tokens, is_fake, domains), tokens)


def test_single_center_loss_pushes_fakes_away():
    scl = SingleCenterLoss(margin=0.01)
    torch.manual_seed(0)
    real = torch.randn(8, 16) * 0.5      # some spread -> Dist_R > 0
    labels = torch.tensor([0] * 8 + [1] * 8)
    # fakes far from the real center -> hinge inactive -> loss == Dist_R
    good = scl(torch.cat([real, torch.ones(8, 16) * 5.0]), labels)
    # fakes at the real center -> hinge active -> larger loss
    bad = scl(torch.cat([real, real.mean(0, keepdim=True).repeat(8, 1)]), labels)
    assert bad > good


def test_metrics_keys():
    labels = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.6])
    m = compute_metrics(labels, scores)
    for k in ("acc", "auc", "f1", "ap", "eer", "fpr", "fnr"):
        assert k in m


def test_inference_disables_fsm():
    model = _tiny_model()
    model.eval()
    x = torch.randn(3, 3, 64, 64)
    with torch.no_grad():
        a = model(x, apply_fsm=False).logits
        b = model(x, apply_fsm=False).logits
    assert torch.allclose(a, b)
