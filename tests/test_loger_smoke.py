"""Fast, offline smoke tests for the LOGER + FSM components.

Uses a tiny, randomly-initialised SigLIP vision tower (no checkpoint download)
so the whole model can be exercised on CPU in seconds. Verifies:
  * dual-branch forward / backward pass and output shapes,
  * MIL top-k pooling (k = floor(0.1 * N)),
  * PEFT modes (frozen / lora / full) trainable-parameter behaviour,
  * FSM is train-only inside the LOGER forward,
  * the LOGER loss components are independent and finite,
  * multi-resolution inference.
"""

from __future__ import annotations

import math

import torch

from src.losses.loger import AUCSurrogateLoss, LOGERLoss, MILLoss
from src.models.loger import LOGERModel, mil_topk_pool

# Tiny config: 4 layers, 96-dim, patch16, 64px -> (64/16)^2 = 16 square tokens.
TINY = dict(
    hidden_size=96,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=192,
    patch_size=16,
    image_size=64,
)


def _tiny_model(peft_mode: str = "lora", fsm_prob: float = 0.5) -> LOGERModel:
    return LOGERModel(
        backbone_name="siglip2",
        peft_mode=peft_mode,
        lora_r=4,
        lora_alpha=4.0,
        fsm_prob=fsm_prob,
        head_hidden_dim=32,
        pretrained=False,
        backbone_config_overrides=TINY,
    )


def test_forward_backward_shapes():
    model = _tiny_model()
    model.train()
    x = torch.randn(6, 3, 64, 64)
    is_fake = torch.tensor([0, 1, 1, 0, 1, 1]).bool()
    domains = torch.tensor([0, 1, 2, 0, 3, 4])

    out = model(x, is_fake=is_fake, domains=domains, apply_fsm=True)
    assert out.logits.shape == (6,)
    assert out.global_logits.shape == (6,)
    assert out.local_logits.shape == (6,)
    assert out.patch_diffs.shape == (6, 16)  # 16 patch tokens
    assert out.pooled.shape == (6, 96)

    loss, parts = LOGERLoss()(
        out.logits, out.global_logits, out.local_logits, out.patch_diffs, is_fake.long()
    )
    assert torch.isfinite(loss)
    loss.backward()
    for key in ("global_bce", "fused_bce", "local_ce", "local_auc", "local_mil", "local_reg"):
        assert torch.isfinite(parts[key]), key


def test_mil_topk_pool_matches_paper_formula():
    torch.manual_seed(0)
    d = torch.randn(2, 50)
    pooled = mil_topk_pool(d, ratio=0.1)
    k = max(1, math.floor(0.1 * 50))  # k = 5
    expected = d.topk(k, dim=1).values.mean(dim=1)
    assert torch.allclose(pooled, expected)
    # Tiny N still keeps at least one patch.
    assert mil_topk_pool(torch.randn(2, 4), ratio=0.1).shape == (2,)


def test_peft_modes():
    frozen = _tiny_model(peft_mode="frozen")
    lora = _tiny_model(peft_mode="lora")
    full = _tiny_model(peft_mode="full")

    def backbone_base_trainable(m: LOGERModel) -> int:
        return sum(
            p.numel()
            for n, p in m.backbone.named_parameters()
            if p.requires_grad and "lora_" not in n
        )

    assert backbone_base_trainable(frozen) == 0
    assert backbone_base_trainable(lora) == 0  # only LoRA params train
    assert any("lora_" in n for n, p in lora.backbone.named_parameters() if p.requires_grad)
    assert backbone_base_trainable(full) > 0
    # Heads always train.
    for m in (frozen, lora, full):
        assert all(p.requires_grad for p in m.global_head.parameters())
        assert all(p.requires_grad for p in m.local_head.parameters())


def test_fsm_train_only_in_loger_forward():
    model = _tiny_model(fsm_prob=1.0)
    x = torch.randn(4, 3, 64, 64)
    is_fake = torch.tensor([0, 1, 0, 1]).bool()
    domains = torch.tensor([0, 1, 0, 2])

    model.eval()
    with torch.no_grad():
        a = model(x, is_fake=is_fake, domains=domains, apply_fsm=True).logits
        b = model(x, apply_fsm=False).logits
    assert torch.allclose(a, b)  # FSM inactive in eval mode


def test_auc_and_mil_losses_behave():
    auc = AUCSurrogateLoss(margin=1.0)
    labels = torch.tensor([0, 0, 1, 1])
    separated = torch.tensor([-5.0, -4.0, 4.0, 5.0])
    collapsed = torch.tensor([1.0, 2.0, -1.0, -2.0])
    assert auc(separated, labels) == 0.0
    assert auc(collapsed, labels) > 0.0
    # Single-class batch -> zero, not NaN.
    assert auc(torch.tensor([1.0, 2.0]), torch.tensor([1, 1])) == 0.0

    mil = MILLoss(topk_ratio=0.1)
    patch = torch.randn(4, 20)
    assert torch.isfinite(mil(patch, labels))


def test_multi_resolution_inference():
    model = _tiny_model()
    model.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        logits = model.forward_multi_resolution(x, resolutions=[64, 96])
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
