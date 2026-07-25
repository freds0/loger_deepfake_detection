"""LightningDataModule for FaceForensics++-style forgery detection.

Wires the :class:`ForgeryFrameDataset` into PyTorch Lightning, handling the
train / val / test splits, SigLIP 2 preprocessing, optional face cropping and
the paper's 4x real-face oversampling for class balance.
"""

from __future__ import annotations

import lightning as L
from collections import Counter, defaultdict

from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import (
    ForgeryFrameDataset,
    Record,
    oversample_real,
    records_from_faceforensics,
    records_from_manifest,
    records_from_ntire,
    split_records_by_hash,
)
from .face_detection import FaceCropper
from .transforms import build_transform


class ForgeryDataModule(L.LightningDataModule):
    """Data module for cross-manipulation / cross-dataset forgery detection.

    Args:
        source: ``"faceforensics"`` (folder scan), ``"manifest"`` (CSV) or
            ``"ntire"`` (NTIRE 2026 shard layout).
        root: Dataset root (folder mode / ntire mode).
        manifest: CSV path, or a list of CSV paths to combine (manifest mode) —
            e.g. one manifest per dataset for cross-dataset training (see
            ``configs/data/combined.yaml``).
        train_shards/val_shards/test_shards: Shard numbers per split (ntire
            mode). Ignored when ``val_fraction`` is set (see below).
        val_fraction: Ntire mode only. When set, ``train``/``val``/``test``
            are no longer distinct shards: every record from ``train_shards``
            is combined and deterministically re-split by
            :func:`~src.data.dataset.split_records_by_hash` keyed on image
            filename (``val_fraction`` routed to val). ``val`` and ``test``
            resolve to the *same* held-out subset -- the NTIRE release ships
            no separate labelled test set, so a distinct test split would
            just be a second validation split under a different name.
            ``None`` (default) keeps the shard-per-split behaviour.
        image_size: Model input resolution.
        batch_size: Batch size.
        num_workers: Dataloader workers.
        prefetch_factor: Batches pre-loaded per worker (ignored when
            ``num_workers == 0``, where the DataLoader forbids setting it).
        real_oversample: Replication factor for real training frames (paper: 4).
        balanced_sampling: Draw training batches with a
            ``WeightedRandomSampler`` weighted by inverse class frequency,
            instead of physically duplicating records. Mutually exclusive
            with ``real_oversample > 1`` (raises at construction time).
            ``WeightedRandomSampler`` is not DDP-aware: this option raises in
            ``train_dataloader`` if ``trainer.world_size > 1`` (single-GPU
            only in this version).
        use_face_crop: Enable face detection / cropping.
        face_backend: Face-detection backend (see :class:`FaceCropper`).
        face_margin: Crop enlargement factor (paper: 1.3).
        augmentation: Advanced augmentation config (see
            :data:`src.data.transforms.DEFAULT_AUG`). Disabled by default to
            match the paper; only applied to the training split when
            ``augmentation['enabled']`` is True.
        domain_map: Optional ``{subfolder: domain_id}`` override (folder mode).
        normalization_mean/std: Channel stats expected by the selected backbone.
        max_records_per_split: Optional int or {split: int} cap for smoke tests.
        pin_memory: Pin dataloader memory.
        persistent_workers: Keep workers alive between epochs.
    """

    def __init__(
        self,
        source: str = "faceforensics",
        root: str | None = None,
        manifest: str | list[str] | None = None,
        train_shards: list[int] | None = None,
        val_shards: list[int] | None = None,
        test_shards: list[int] | None = None,
        val_fraction: float | None = None,
        image_size: int = 224,
        batch_size: int = 48,
        num_workers: int = 8,
        prefetch_factor: int = 2,
        real_oversample: int = 4,
        balanced_sampling: bool = False,
        use_face_crop: bool = False,
        face_backend: str = "opencv",
        face_margin: float = 1.3,
        augmentation: dict | None = None,
        domain_map: dict[str, int] | None = None,
        normalization_mean: tuple[float, float, float] | list[float] = (0.5, 0.5, 0.5),
        normalization_std: tuple[float, float, float] | list[float] = (0.5, 0.5, 0.5),
        max_records_per_split: int | dict[str, int] | None = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        super().__init__()
        if balanced_sampling and real_oversample > 1:
            raise ValueError(
                "balanced_sampling and real_oversample>1 are mutually exclusive "
                "(both rebalance classes; pick one)."
            )
        self.save_hyperparameters()
        self._train: ForgeryFrameDataset | None = None
        self._val: ForgeryFrameDataset | None = None
        self._test: ForgeryFrameDataset | None = None

    def _split_limit(self, split: str) -> int | None:
        limit = self.hparams.max_records_per_split
        if limit is None:
            return None
        if isinstance(limit, dict):
            value = limit.get(split)
            return int(value) if value is not None else None
        return int(limit)

    @staticmethod
    def _round_robin(records: list[Record], limit: int) -> list[Record]:
        groups: dict[int, list[Record]] = defaultdict(list)
        for rec in records:
            groups[rec.domain].append(rec)
        selected: list[Record] = []
        while len(selected) < limit and groups:
            for domain in sorted(list(groups)):
                bucket = groups[domain]
                if bucket:
                    selected.append(bucket.pop(0))
                    if len(selected) == limit:
                        break
                if not bucket:
                    groups.pop(domain, None)
        return selected

    def _limit_records(self, records: list[Record], split: str) -> list[Record]:
        limit = self._split_limit(split)
        if limit is None or len(records) <= limit:
            return records
        real = [r for r in records if r.label == 0]
        fake = [r for r in records if r.label == 1]
        n_real = min(len(real), max(1, limit // 2))
        n_fake = min(len(fake), limit - n_real)
        real_sel = real[:n_real]
        fake_sel = self._round_robin(fake, n_fake)
        selected: list[Record] = []
        for i in range(max(len(real_sel), len(fake_sel))):
            if i < len(real_sel):
                selected.append(real_sel[i])
            if i < len(fake_sel):
                selected.append(fake_sel[i])
        if len(selected) < limit:
            # id()-based membership, not `r not in selected` (Record is a
            # dataclass, so `in` does an O(len(selected)) value comparison per
            # element -> O(n*m) overall on large splits). `selected` is built
            # from the same `records` objects, so identity is safe here.
            selected_ids = {id(r) for r in selected}
            remaining = [r for r in records if id(r) not in selected_ids]
            selected.extend(remaining[: limit - len(selected)])
        return selected[:limit]

    def _load_records(self, split: str) -> list[Record]:
        h = self.hparams
        if h.source == "faceforensics":
            if h.root is None:
                raise ValueError("`root` is required for source='faceforensics'")
            records = records_from_faceforensics(h.root, split, domain_map=h.domain_map)
        elif h.source == "manifest":
            if h.manifest is None:
                raise ValueError("`manifest` is required for source='manifest'")
            records = records_from_manifest(h.manifest, split=split)
        elif h.source == "ntire":
            if h.root is None:
                raise ValueError("`root` is required for source='ntire'")
            if h.val_fraction is not None:
                # No real held-out shard: re-split every train_shards record by
                # filename hash. val and test intentionally resolve to the same
                # subset (see class docstring).
                if not h.train_shards:
                    raise ValueError(
                        "`train_shards` is required for source='ntire' with val_fraction set"
                    )
                pool = records_from_ntire(h.root, h.train_shards)
                target = "val" if split in ("val", "test") else "train"
                records = split_records_by_hash(pool, h.val_fraction, target)
            else:
                shard_nums = getattr(h, f"{split}_shards")
                if not shard_nums:
                    raise ValueError(f"`{split}_shards` is required for source='ntire'")
                records = records_from_ntire(h.root, shard_nums)
        else:
            raise ValueError(f"Unknown data source: {h.source}")
        return self._limit_records(records, split)

    def _cropper(self) -> FaceCropper | None:
        h = self.hparams
        if not h.use_face_crop:
            return None
        return FaceCropper(backend=h.face_backend, margin=h.face_margin)

    def setup(self, stage: str | None = None) -> None:
        h = self.hparams
        cropper = self._cropper()

        if stage in (None, "fit"):
            train_records = oversample_real(self._load_records("train"), h.real_oversample)
            self._train = ForgeryFrameDataset(
                train_records,
                transform=build_transform(
                    h.image_size,
                    train=True,
                    augmentation=h.augmentation,
                    mean=tuple(h.normalization_mean),
                    std=tuple(h.normalization_std),
                ),
                face_cropper=cropper,
            )
            self._val = ForgeryFrameDataset(
                self._load_records("val"),
                transform=build_transform(
                    h.image_size,
                    train=False,
                    mean=tuple(h.normalization_mean),
                    std=tuple(h.normalization_std),
                ),
                face_cropper=cropper,
            )
        if stage in ("test", "predict") or stage is None:
            self._test = ForgeryFrameDataset(
                self._load_records("test"),
                transform=build_transform(
                    h.image_size,
                    train=False,
                    mean=tuple(h.normalization_mean),
                    std=tuple(h.normalization_std),
                ),
                face_cropper=cropper,
            )
        if stage == "validate" and self._val is None:
            self._val = ForgeryFrameDataset(
                self._load_records("val"),
                transform=build_transform(
                    h.image_size,
                    train=False,
                    mean=tuple(h.normalization_mean),
                    std=tuple(h.normalization_std),
                ),
                face_cropper=cropper,
            )

    def _train_sampler(self) -> WeightedRandomSampler | None:
        if not self.hparams.balanced_sampling:
            return None
        world_size = getattr(getattr(self, "trainer", None), "world_size", 1)
        if world_size > 1:
            raise ValueError(
                "balanced_sampling is not DDP-aware (WeightedRandomSampler has no "
                "distributed variant here); use a single device or disable "
                "balanced_sampling for multi-GPU runs."
            )
        counts = Counter(r.label for r in self._train.records)
        weights = [1.0 / counts[r.label] for r in self._train.records]
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    def _loader(
        self, dataset: ForgeryFrameDataset, shuffle: bool, sampler: WeightedRandomSampler | None = None
    ) -> DataLoader:
        h = self.hparams
        kwargs: dict = dict(
            batch_size=h.batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=h.num_workers,
            pin_memory=h.pin_memory,
            drop_last=shuffle,
            persistent_workers=h.persistent_workers and h.num_workers > 0,
        )
        if h.num_workers > 0:
            # DataLoader raises ValueError if prefetch_factor is set at all
            # (even to its own default) when num_workers == 0.
            kwargs["prefetch_factor"] = h.prefetch_factor
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self._train, shuffle=True, sampler=self._train_sampler())

    def val_dataloader(self) -> DataLoader:
        return self._loader(self._val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self._test, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        # Predict over the test split (used by test.py to write per-image CSVs).
        return self._loader(self._test, shuffle=False)
