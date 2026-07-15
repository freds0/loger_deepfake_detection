"""Tests for balanced-sampling class rebalancing (P0.3)."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image

from src.data.datamodule import ForgeryDataModule


def _write_manifest(tmp_path, n_real: int, n_fake: int):
    img_path = tmp_path / "x.jpg"
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(img_path, format="JPEG")

    rows = [(str(img_path), 0, 0, "train") for _ in range(n_real)]
    rows += [(str(img_path), 1, 1, "train") for _ in range(n_fake)]
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows, columns=["path", "label", "domain", "split"]).to_csv(manifest_path, index=False)
    return manifest_path


def test_balanced_sampling_and_oversample_are_mutually_exclusive():
    with pytest.raises(ValueError):
        ForgeryDataModule(
            source="manifest", manifest="unused.csv", balanced_sampling=True, real_oversample=4
        )


def test_balanced_sampling_yields_roughly_balanced_batches(tmp_path):
    manifest_path = _write_manifest(tmp_path, n_real=90, n_fake=10)

    dm = ForgeryDataModule(
        source="manifest",
        manifest=str(manifest_path),
        batch_size=20,
        num_workers=0,
        real_oversample=1,
        balanced_sampling=True,
        persistent_workers=False,
    )
    dm.setup("fit")

    fractions = []
    for _ in range(10):  # 10 epochs x 5 batches/epoch (100 records / batch_size 20) = 50 batches
        for batch in dm.train_dataloader():
            fractions.append(batch["label"].float().mean().item())
    assert len(fractions) == 50

    avg = sum(fractions) / len(fractions)
    assert 0.4 <= avg <= 0.6


def test_unweighted_sampling_stays_skewed(tmp_path):
    """Sanity check: without balanced_sampling, batches reflect the raw 90/10 skew."""
    manifest_path = _write_manifest(tmp_path, n_real=90, n_fake=10)

    dm = ForgeryDataModule(
        source="manifest",
        manifest=str(manifest_path),
        batch_size=20,
        num_workers=0,
        real_oversample=1,
        balanced_sampling=False,
        persistent_workers=False,
    )
    dm.setup("fit")

    fractions = []
    for _ in range(10):
        for batch in dm.train_dataloader():
            fractions.append(batch["label"].float().mean().item())

    avg = sum(fractions) / len(fractions)
    assert avg < 0.2  # close to the true 10% positive rate, not rebalanced


def test_balanced_sampling_raises_under_multi_gpu(tmp_path):
    manifest_path = _write_manifest(tmp_path, n_real=9, n_fake=1)

    dm = ForgeryDataModule(
        source="manifest",
        manifest=str(manifest_path),
        batch_size=2,
        num_workers=0,
        real_oversample=1,
        balanced_sampling=True,
        persistent_workers=False,
    )
    dm.setup("fit")
    dm.trainer = SimpleNamespace(world_size=2)

    with pytest.raises(ValueError, match="not DDP-aware"):
        dm.train_dataloader()
