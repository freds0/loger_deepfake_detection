"""Verify the LR warmup + cosine schedule wired for NTIRE training (P1.1).

Builds the real ``loger_fsm_ntire`` Hydra config end-to-end (so a config
regression on ``scheduler.warmup_steps`` or the optimizer/scheduler wiring in
``LOGERLightningModule.configure_optimizers`` is caught), but swaps in a tiny
randomly-initialised SigLIP tower (no checkpoint download) so it runs on CPU
in seconds -- same trick as ``tests/test_loger_smoke.py``.
"""

from __future__ import annotations

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


def test_ntire_config_enables_warmup():
    module = _tiny_ntire_module()
    assert module.cfg.scheduler.warmup_steps == 500


def test_warmup_then_cosine_schedule_shape():
    module = _tiny_ntire_module()
    base_lr = module.cfg.optimizer.lr
    warmup_steps = module.cfg.scheduler.warmup_steps

    result = module.configure_optimizers()
    optimizer, scheduler = result["optimizer"], result["lr_scheduler"]["scheduler"]

    lrs = []
    for _ in range(warmup_steps + 50):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()  # no-op (no grads computed); keeps step ordering PyTorch-correct
        scheduler.step()

    # Starts near-zero (LinearLR start_factor=1e-3), not at base_lr.
    assert lrs[0] < base_lr * 0.01
    # Monotonically increasing throughout warmup.
    assert all(a <= b + 1e-12 for a, b in zip(lrs[:warmup_steps], lrs[1 : warmup_steps + 1]))
    # Warmup ends exactly at base_lr.
    assert lrs[warmup_steps] == base_lr
    # Cosine decay takes over immediately after: lr strictly decreases.
    assert lrs[warmup_steps + 49] < lrs[warmup_steps]
