"""Tests for prefetch_factor wiring in ForgeryDataModule (P3.1)."""

from __future__ import annotations

import pandas as pd
from PIL import Image

from src.data.datamodule import ForgeryDataModule


def _write_manifest(tmp_path):
    img_path = tmp_path / "x.jpg"
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(img_path, format="JPEG")
    rows = [(str(img_path), i % 2, i % 2, "train") for i in range(10)]
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows, columns=["path", "label", "domain", "split"]).to_csv(manifest_path, index=False)
    return manifest_path


def test_prefetch_factor_applied_when_workers_enabled(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    dm = ForgeryDataModule(
        source="manifest",
        manifest=str(manifest_path),
        batch_size=2,
        num_workers=2,
        prefetch_factor=4,
        real_oversample=1,
        persistent_workers=False,
    )
    dm.setup("fit")
    loader = dm.train_dataloader()
    assert loader.prefetch_factor == 4


def test_prefetch_factor_omitted_with_zero_workers_no_crash(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    dm = ForgeryDataModule(
        source="manifest",
        manifest=str(manifest_path),
        batch_size=2,
        num_workers=0,
        prefetch_factor=4,  # must be silently ignored, not passed to DataLoader
        real_oversample=1,
        persistent_workers=False,
    )
    dm.setup("fit")
    loader = dm.train_dataloader()  # must not raise
    assert loader.prefetch_factor is None
