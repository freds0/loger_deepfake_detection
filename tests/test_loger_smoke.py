"""Fast, offline smoke tests for the LOGER + FSM components.

Uses a tiny, randomly-initialised SigLIP vision tower (no checkpoint download)
so the whole model can be exercised on CPU in seconds. Verifies:
  * dual-branch forward / backward pass and output shapes,
  * MIL top-k pooling (k = floor(0.1 * N)),
  * PEFT modes (frozen / lora / full) trainable-parameter behaviour,
  * FSM is train-only inside the LOGER forward,
  * the LOGER loss components are independent and finite,
  * multi-resolution inference,
  * a real Trainer.fit() with EMACallback attached (P1.2).
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import lightning as L
import pytest
import torch
from hydra import compose, initialize
from omegaconf import open_dict
from torch.utils.data import DataLoader, Dataset

from src.lightning.loger_module import LOGERLightningModule
from src.losses.loger import AUCSurrogateLoss, LOGERLoss, MILLoss
from src.models.loger import LOGERModel, mil_topk_pool
from src.utils.ema import EMACallback

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


def _tiny_learnable_fusion_model() -> LOGERModel:
    return LOGERModel(
        backbone_name="siglip2",
        peft_mode="lora",
        lora_r=4,
        lora_alpha=4.0,
        head_hidden_dim=32,
        learnable_fusion=True,
        pretrained=False,
        backbone_config_overrides=TINY,
    )


def test_fixed_fusion_state_dict_unchanged_for_checkpoint_compat():
    model = _tiny_model()  # learnable_fusion=False (default)
    sd = model.state_dict()
    assert "fusion_weights" in sd
    assert "fusion_logits" not in sd
    assert torch.equal(sd["fusion_weights"], torch.full((2,), 0.5))


def test_learnable_fusion_adds_parameter_that_receives_gradient():
    model = _tiny_learnable_fusion_model()
    sd = model.state_dict()
    assert "fusion_logits" in sd
    assert "fusion_weights" not in sd
    assert model.fusion_logits.requires_grad
    assert torch.equal(model.fusion_logits, torch.zeros(2))  # softmax(0,0) = (0.5, 0.5)

    model.train()
    x = torch.randn(4, 3, 64, 64)
    out = model(x, apply_fsm=False)
    out.logits.sum().backward()
    assert model.fusion_logits.grad is not None
    assert torch.isfinite(model.fusion_logits.grad).all()


def test_learnable_fusion_and_fixed_weights_are_mutually_exclusive():
    with pytest.raises(ValueError):
        LOGERModel(
            backbone_name="siglip2",
            peft_mode="lora",
            fusion_weights=(0.7, 0.3),
            learnable_fusion=True,
            pretrained=False,
            backbone_config_overrides=TINY,
        )


def test_learnable_fusion_forward_matches_manual_softmax_combination():
    model = _tiny_learnable_fusion_model()
    model.eval()
    with torch.no_grad():
        model.fusion_logits.copy_(torch.tensor([1.0, -1.0]))

    x = torch.randn(3, 3, 64, 64)
    with torch.no_grad():
        out = model(x, apply_fsm=False)

    w = torch.softmax(torch.tensor([1.0, -1.0]), dim=0)
    expected = w[0] * out.global_logits + w[1] * out.local_logits
    assert torch.allclose(out.logits, expected, atol=1e-6)


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


def test_multi_resolution_precomputed_skips_the_native_resolution_forward():
    """P1.5: passing precomputed=(native_res, logits) must match the fully
    recomputed result bit-for-bit while running the backbone one fewer time."""
    model = _tiny_model()
    model.eval()
    x = torch.randn(2, 3, 64, 64)

    calls: list[int] = []
    model.backbone.register_forward_hook(lambda *_: calls.append(1))

    with torch.no_grad():
        full = model.forward_multi_resolution(x, resolutions=[64, 96])
    assert len(calls) == 2  # one backbone forward per resolution
    calls.clear()

    with torch.no_grad():
        native_out = model(x, apply_fsm=False)
    assert len(calls) == 1
    calls.clear()

    with torch.no_grad():
        precomputed_result = model.forward_multi_resolution(
            x, resolutions=[64, 96], precomputed=(64, native_out.logits)
        )
    assert len(calls) == 1  # only the 96px resolution actually ran the backbone
    assert torch.allclose(full, precomputed_result, atol=1e-6)


def test_eval_step_reuses_native_resolution_forward_with_multi_resolution_eval():
    """P1.5, integration level: LOGERLightningModule._eval_step must feed its
    already-computed native-resolution logits into forward_multi_resolution
    instead of letting it recompute that resolution from scratch."""
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="loger_fsm_ntire")
    with open_dict(cfg):
        cfg.backbone.pretrained = False
        cfg.backbone.config_overrides = TINY
        cfg.backbone.image_size = 64
        cfg.data.image_size = 64
        cfg.model.eval_resolutions = [64, 96]

    module = LOGERLightningModule(cfg)
    module.eval()

    calls: list[int] = []
    module.model.backbone.register_forward_hook(lambda *_: calls.append(1))

    batch = {
        "pixel_values": torch.randn(2, 3, 64, 64),
        "label": torch.tensor([0, 1]),
        "domain": torch.tensor([0, 1]),
        "path": ["a.jpg", "b.jpg"],
    }
    with torch.no_grad():
        module._eval_step(batch, [], [])

    # One forward for the loss/native-resolution logits + one more for the
    # 96px resolution -- NOT a second forward at 64px.
    assert len(calls) == 2


def _tiny_ntire_module(learnable_fusion: bool = False) -> LOGERLightningModule:
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="loger_fsm_ntire")
    with open_dict(cfg):
        cfg.backbone.pretrained = False
        cfg.backbone.config_overrides = TINY
        cfg.backbone.image_size = 64
        cfg.data.image_size = 64
        cfg.model.learnable_fusion = learnable_fusion
    return LOGERLightningModule(cfg)


def test_on_train_epoch_end_logs_fusion_weights_when_learnable():
    module = _tiny_ntire_module(learnable_fusion=True)
    module._epoch_start = 0.0
    module.log = MagicMock()

    module.on_train_epoch_end()

    logged_names = [call.args[0] for call in module.log.call_args_list]
    assert "fusion/w_global" in logged_names
    assert "fusion/w_local" in logged_names


def test_on_train_epoch_end_skips_fusion_logging_when_fixed():
    module = _tiny_ntire_module(learnable_fusion=False)
    module._epoch_start = 0.0
    module.log = MagicMock()

    module.on_train_epoch_end()

    logged_names = [call.args[0] for call in module.log.call_args_list]
    assert "fusion/w_global" not in logged_names
    assert "fusion/w_local" not in logged_names


class _RandomFrameDataset(Dataset):
    """8 random 64x64 frames, alternating label/domain -- no disk I/O."""

    def __len__(self) -> int:
        return 8

    def __getitem__(self, idx: int) -> dict:
        return {
            "pixel_values": torch.randn(3, 64, 64),
            "label": torch.tensor(idx % 2, dtype=torch.long),
            "domain": torch.tensor(idx % 2, dtype=torch.long),
            "path": f"img_{idx}.jpg",
        }


def test_ema_callback_integration_with_real_trainer_fit(tmp_path):
    """End-to-end: Trainer.fit() with EMACallback attached to the real
    loger_fsm_ntire config (tiny non-pretrained backbone swapped in so this
    stays a fast, offline test)."""
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="loger_fsm_ntire")
    with open_dict(cfg):
        cfg.backbone.pretrained = False
        cfg.backbone.config_overrides = TINY
        cfg.backbone.image_size = 64
        cfg.data.image_size = 64
        cfg.output_dir = str(tmp_path)

    module = LOGERLightningModule(cfg)
    loader = DataLoader(_RandomFrameDataset(), batch_size=4)
    ema = EMACallback(decay=0.5)

    trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        callbacks=[ema],
        accelerator="cpu",
        devices=1,
    )
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)

    assert ema._shadow is not None
    # Swap is transient: after the last validation epoch, raw weights restored.
    assert ema._backup is None
    for k, v in module.model.state_dict().items():
        assert torch.isfinite(v).all()
        assert torch.isfinite(ema._shadow[k]).all()
