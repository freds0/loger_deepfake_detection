"""Tests for the `tune_batch_size` config wiring (P3.2).

The Tuner run itself is exercised manually against a real GPU/dataset (slow,
hardware-dependent -- not suitable for the fast CI suite); see the P3.2
summary for that end-to-end verification. These tests just guard the cheap,
deterministic part: the flag exists with the right default in every LOGER
root config and is genuinely overridable from the CLI.
"""

from __future__ import annotations

import pytest
from hydra import compose, initialize

ROOT_CONFIGS = [
    "loger_fsm_siglip2",
    "loger_fsm_baseline",
    "loger_fsm_dinov2",
    "loger_fsm_ntire",
    "loger_fsm_combined",
]


@pytest.mark.parametrize("config_name", ROOT_CONFIGS)
def test_tune_batch_size_defaults_to_false(config_name):
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=config_name)
    assert cfg.tune_batch_size is False


@pytest.mark.parametrize("config_name", ROOT_CONFIGS)
def test_tune_batch_size_overridable_without_plus_prefix(config_name):
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=config_name, overrides=["tune_batch_size=true"])
    assert cfg.tune_batch_size is True
