"""Every PLAN_v0.2 ablation config must compose cleanly (no model load)."""

from __future__ import annotations

import glob
import os

import pytest
from hydra import compose, initialize

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
_NAMES = sorted(
    os.path.splitext(os.path.basename(p))[0]
    for p in glob.glob(os.path.join(_CONFIG_DIR, "ntire_v2_*.yaml"))
)


def test_there_are_ablation_configs():
    assert "ntire_v2_base" in _NAMES
    assert len(_NAMES) >= 20


@pytest.mark.parametrize("name", _NAMES)
def test_config_composes(name):
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=name)
    # Shared ablation regime inherited from ntire_v2_base.
    assert cfg.data.source == "ntire"
    assert cfg.trainer.max_steps == 8000
    assert cfg.resume.enabled is False
