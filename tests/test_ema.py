"""Tests for the EMA callback and its checkpoint integration (P1.2)."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from hydra import compose, initialize
from omegaconf import open_dict

from src.utils.ema import EMACallback, load_ema_weights
from src.utils.lightning_setup import build_callbacks


def _pl_module(model: torch.nn.Module) -> SimpleNamespace:
    """EMACallback only touches `pl_module.model`; no real LightningModule needed."""
    return SimpleNamespace(model=model)


def test_ema_shadow_matches_manual_computation():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    pl_module = _pl_module(model)

    callback = EMACallback(decay=0.9)
    callback.on_fit_start(trainer=None, pl_module=pl_module)
    manual_shadow = {k: v.clone() for k, v in model.state_dict().items()}

    for step in range(10):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)  # simulate an optimizer step
        callback.on_train_batch_end(None, pl_module, None, None, step)
        for k, v in model.state_dict().items():
            manual_shadow[k].mul_(0.9).add_(v, alpha=0.1)

    for k in manual_shadow:
        assert torch.allclose(callback._shadow[k], manual_shadow[k])


def test_ema_swaps_in_for_validation_and_restores_after():
    model = torch.nn.Linear(4, 2)
    pl_module = _pl_module(model)
    callback = EMACallback(decay=0.5)
    callback.on_fit_start(None, pl_module)

    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    callback.on_train_batch_end(None, pl_module, None, None, 0)

    raw_state = {k: v.clone() for k, v in model.state_dict().items()}
    shadow_state = {k: v.clone() for k, v in callback._shadow.items()}
    assert any(not torch.equal(raw_state[k], shadow_state[k]) for k in raw_state)  # sanity

    callback.on_validation_start(None, pl_module)
    for k, v in model.state_dict().items():
        assert torch.equal(v, shadow_state[k])

    callback.on_validation_end(None, pl_module)
    for k, v in model.state_dict().items():
        assert torch.equal(v, raw_state[k])
    assert callback._backup is None


def test_ema_swaps_in_for_test_and_restores_after():
    model = torch.nn.Linear(4, 2)
    pl_module = _pl_module(model)
    callback = EMACallback(decay=0.5)
    callback.on_fit_start(None, pl_module)

    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    callback.on_train_batch_end(None, pl_module, None, None, 0)
    raw_state = {k: v.clone() for k, v in model.state_dict().items()}
    shadow_state = {k: v.clone() for k, v in callback._shadow.items()}

    callback.on_test_start(None, pl_module)
    for k, v in model.state_dict().items():
        assert torch.equal(v, shadow_state[k])

    callback.on_test_end(None, pl_module)
    for k, v in model.state_dict().items():
        assert torch.equal(v, raw_state[k])


def test_swap_is_a_noop_before_fit_starts():
    """`trainer.validate()` without ever calling `fit()`: no shadow yet."""
    model = torch.nn.Linear(4, 2)
    pl_module = _pl_module(model)
    callback = EMACallback(decay=0.9)

    raw_state = {k: v.clone() for k, v in model.state_dict().items()}
    callback.on_validation_start(None, pl_module)
    for k, v in model.state_dict().items():
        assert torch.equal(v, raw_state[k])
    callback.on_validation_end(None, pl_module)  # must not raise


def test_ema_state_dict_survives_resume_without_reinit():
    model = torch.nn.Linear(4, 2)
    pl_module = _pl_module(model)
    callback = EMACallback(decay=0.9)
    callback.on_fit_start(None, pl_module)

    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    callback.on_train_batch_end(None, pl_module, None, None, 0)
    saved_shadow = {k: v.clone() for k, v in callback._shadow.items()}

    resumed = EMACallback(decay=0.9)
    resumed.load_state_dict(callback.state_dict())
    # Simulates Lightning's resume order: _restore_modules_and_callbacks()
    # (-> load_state_dict) always runs before the on_fit_start hook.
    resumed.on_fit_start(None, pl_module)

    for k, v in resumed._shadow.items():
        assert torch.equal(v, saved_shadow[k])


def test_load_ema_weights_from_checkpoint(tmp_path):
    model = torch.nn.Linear(4, 2)
    shadow = {k: v.clone() + 5.0 for k, v in model.state_dict().items()}
    ckpt_path = tmp_path / "fake.ckpt"
    torch.save({"state_dict": {}, "callbacks": {"EMACallback": {"shadow": shadow}}}, ckpt_path)

    loaded = load_ema_weights(str(ckpt_path), model)
    assert loaded is True
    for k, v in model.state_dict().items():
        assert torch.equal(v, shadow[k])


def test_load_ema_weights_returns_false_without_ema_state(tmp_path):
    model = torch.nn.Linear(4, 2)
    raw_state = {k: v.clone() for k, v in model.state_dict().items()}
    ckpt_path = tmp_path / "no_ema.ckpt"
    torch.save({"state_dict": {}, "callbacks": {}}, ckpt_path)

    loaded = load_ema_weights(str(ckpt_path), model)
    assert loaded is False
    for k, v in model.state_dict().items():
        assert torch.equal(v, raw_state[k])  # untouched


def test_build_callbacks_adds_ema_for_loger_ntire_config(tmp_path):
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="loger_fsm_ntire")
    with open_dict(cfg):
        cfg.output_dir = str(tmp_path)

    callbacks = build_callbacks(cfg)
    ema_callbacks = [c for c in callbacks if isinstance(c, EMACallback)]
    assert len(ema_callbacks) == 1
    assert ema_callbacks[0].decay == 0.999


def test_build_callbacks_no_ema_for_osdfd_default_config(tmp_path):
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")
    with open_dict(cfg):
        cfg.output_dir = str(tmp_path)

    callbacks = build_callbacks(cfg)
    assert not any(isinstance(c, EMACallback) for c in callbacks)
