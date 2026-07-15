"""Tests for the deterministic hash-based train/val split (P0.2)."""

from __future__ import annotations

import pytest

from src.data.dataset import Record, split_records_by_hash


def _synthetic_records(n: int) -> list[Record]:
    return [Record(path=f"/data/images/img_{i:06d}.jpg", label=i % 2, domain=i % 2) for i in range(n)]


def test_val_fraction_is_approximately_respected():
    records = _synthetic_records(10_000)
    val = split_records_by_hash(records, val_fraction=0.05, split="val")
    fraction = len(val) / len(records)
    assert 0.043 <= fraction <= 0.057


def test_train_and_val_partition_all_records():
    records = _synthetic_records(10_000)
    train = split_records_by_hash(records, val_fraction=0.05, split="train")
    val = split_records_by_hash(records, val_fraction=0.05, split="val")

    train_paths = {r.path for r in train}
    val_paths = {r.path for r in val}

    assert train_paths.isdisjoint(val_paths)
    assert train_paths | val_paths == {r.path for r in records}
    assert len(train) + len(val) == len(records)


def test_split_is_deterministic_across_calls_and_order():
    records = _synthetic_records(500)
    val_1 = {r.path for r in split_records_by_hash(records, 0.1, "val")}
    val_2 = {r.path for r in split_records_by_hash(records, 0.1, "val")}
    assert val_1 == val_2

    shuffled = list(reversed(records))
    val_shuffled = {r.path for r in split_records_by_hash(shuffled, 0.1, "val")}
    assert val_1 == val_shuffled


def test_split_keyed_on_basename_not_full_path():
    a = Record(path="/machine_a/data/root/img_1.jpg", label=0, domain=0)
    b = Record(path="/machine_b/other/root/img_1.jpg", label=0, domain=0)
    val_a = split_records_by_hash([a], 0.5, "val")
    val_b = split_records_by_hash([b], 0.5, "val")
    assert bool(val_a) == bool(val_b)


def test_invalid_split_name_raises():
    with pytest.raises(ValueError):
        split_records_by_hash(_synthetic_records(10), 0.05, "test")
