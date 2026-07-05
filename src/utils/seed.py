"""Reproducibility helpers."""

from __future__ import annotations

import lightning as L


def seed_everything(seed: int, workers: bool = True) -> int:
    """Seed Python, NumPy and PyTorch RNGs via Lightning.

    Args:
        seed: The random seed.
        workers: Also seed dataloader workers.

    Returns:
        The seed that was set.
    """
    return L.seed_everything(seed, workers=workers)
