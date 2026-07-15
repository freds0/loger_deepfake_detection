"""Tests for NTIRE shard loading and dataset-validation robustness (P0.1, P0.2).

Covers:
  * ``ForgeryFrameDataset`` tolerates truncated JPEGs (``LOAD_TRUNCATED_IMAGES``).
  * ``scripts/validate_dataset.py`` correctly classifies ok / missing / corrupt /
    bad_label images across a synthetic shard.
  * ``ForgeryDataModule(source="ntire", val_fraction=...)`` re-splits every
    train_shards record by filename hash instead of holding out a shard.
"""

from __future__ import annotations

import csv
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.datamodule import ForgeryDataModule
from src.data.dataset import Record
from src.data.dataset import ForgeryFrameDataset
from src.data.transforms import build_transform

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validate_dataset.py"
_spec = importlib.util.spec_from_file_location("validate_dataset", _SCRIPT_PATH)
validate_dataset = importlib.util.module_from_spec(_spec)
sys.modules["validate_dataset"] = validate_dataset
_spec.loader.exec_module(validate_dataset)


def test_forgery_frame_dataset_tolerates_truncated_jpeg(tmp_path):
    img_path = tmp_path / "truncated.jpg"
    full = tmp_path / "full.jpg"
    # Random noise (not a solid color) so the JPEG has substantial scan data;
    # otherwise a 60% cut lands inside the header, which even
    # LOAD_TRUNCATED_IMAGES cannot recover from.
    rng = random.Random(0)
    pixels = np.array(
        [rng.randint(0, 255) for _ in range(128 * 128 * 3)], dtype=np.uint8
    ).reshape(128, 128, 3)
    Image.fromarray(pixels, mode="RGB").save(full, format="JPEG", quality=90)
    data = full.read_bytes()
    img_path.write_bytes(data[: int(len(data) * 0.6)])

    record = Record(path=str(img_path), label=1, domain=1)
    dataset = ForgeryFrameDataset([record], transform=build_transform(32))

    sample = dataset[0]  # must not raise despite the truncated file
    assert sample["pixel_values"].shape == (3, 32, 32)
    assert sample["label"].item() == 1


def _write_shard(root: Path) -> None:
    shard_dir = root / "shard_0"
    images_dir = shard_dir / "images"
    images_dir.mkdir(parents=True)

    Image.new("RGB", (16, 16), color=(0, 200, 0)).save(images_dir / "ok.jpg", format="JPEG")
    (images_dir / "corrupt.jpg").write_bytes(b"not an image at all")
    Image.new("RGB", (16, 16), color=(0, 0, 200)).save(images_dir / "bad_label.jpg", format="JPEG")
    # "missing.jpg" is referenced in labels.csv but never written to disk.

    with open(shard_dir / "labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", "image_name", "label"])
        writer.writerow([0, "ok.jpg", 0])
        writer.writerow([1, "missing.jpg", 1])
        writer.writerow([2, "corrupt.jpg", 1])
        writer.writerow([3, "bad_label.jpg", 2])


def test_validate_dataset_classifies_all_statuses(tmp_path):
    _write_shard(tmp_path)

    results = validate_dataset.validate(tmp_path, shard_nums=[0], workers=1)
    status_by_name = {name: status for _, name, status in results}

    assert status_by_name == {
        "ok.jpg": "ok",
        "missing.jpg": "missing",
        "corrupt.jpg": "corrupt",
        "bad_label.jpg": "bad_label",
    }


def test_validate_dataset_main_writes_report_and_exit_code(tmp_path, monkeypatch, capsys):
    _write_shard(tmp_path)

    monkeypatch.setattr(sys, "argv", ["validate_dataset.py", "--root", str(tmp_path), "--workers", "1"])
    exit_code = validate_dataset.main()

    assert exit_code == 1
    report_path = tmp_path / "validation_report.csv"
    assert report_path.is_file()
    with open(report_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert {r["image_name"] for r in rows} == {"missing.jpg", "corrupt.jpg", "bad_label.jpg"}


def _write_ntire_labels(root: Path, shard_num: int, n: int) -> None:
    """Write only labels.csv for a shard (images are never opened by this test)."""
    shard_dir = root / f"shard_{shard_num}"
    shard_dir.mkdir(parents=True)
    with open(shard_dir / "labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", "image_name", "label"])
        for i in range(n):
            writer.writerow([i, f"s{shard_num}_img{i:04d}.jpg", i % 2])


def test_datamodule_val_fraction_splits_by_hash_not_shard(tmp_path):
    _write_ntire_labels(tmp_path, shard_num=0, n=200)
    _write_ntire_labels(tmp_path, shard_num=1, n=200)

    dm = ForgeryDataModule(
        source="ntire",
        root=str(tmp_path),
        train_shards=[0, 1],
        val_fraction=0.2,
        batch_size=4,
        num_workers=0,
        real_oversample=1,
        persistent_workers=False,
    )
    dm.setup("fit")
    dm.setup("test")

    train_paths = {r.path for r in dm._train.records}
    val_paths = {r.path for r in dm._val.records}
    test_paths = {r.path for r in dm._test.records}

    # val and test resolve to the exact same held-out subset by design.
    assert val_paths == test_paths
    assert train_paths.isdisjoint(val_paths)
    assert train_paths | val_paths == train_paths | test_paths

    fraction = len(val_paths) / (len(train_paths) + len(val_paths))
    assert 0.15 <= fraction <= 0.25

    # both shards contribute to val: this is a filename-hash split, not a
    # shard-level split (shard_1 is not silently treated as "the val shard").
    assert any("shard_0" in p for p in val_paths)
    assert any("shard_1" in p for p in val_paths)
    assert any("shard_0" in p for p in train_paths)
    assert any("shard_1" in p for p in train_paths)
