"""LightningDataModule for FaceForensics++-style forgery detection.

Wires the :class:`ForgeryFrameDataset` into PyTorch Lightning, handling the
train / val / test splits, SigLIP 2 preprocessing, optional face cropping and
the paper's 4x real-face oversampling for class balance.
"""

from __future__ import annotations

import lightning as L
from collections import defaultdict

from torch.utils.data import DataLoader

from .dataset import (
    ForgeryFrameDataset,
    Record,
    oversample_real,
    records_from_faceforensics,
    records_from_manifest,
)
from .face_detection import FaceCropper
from .transforms import build_transform


class ForgeryDataModule(L.LightningDataModule):
    """Data module for cross-manipulation / cross-dataset forgery detection.

    Args:
        source: ``"faceforensics"`` (folder scan) or ``"manifest"`` (CSV).
        root: Dataset root (folder mode).
        manifest: CSV path (manifest mode).
        image_size: Model input resolution.
        batch_size: Batch size.
        num_workers: Dataloader workers.
        real_oversample: Replication factor for real training frames (paper: 4).
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
        manifest: str | None = None,
        image_size: int = 224,
        batch_size: int = 48,
        num_workers: int = 8,
        real_oversample: int = 4,
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
            remaining = [r for r in records if r not in selected]
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

    def _loader(self, dataset: ForgeryFrameDataset, shuffle: bool) -> DataLoader:
        h = self.hparams
        return DataLoader(
            dataset,
            batch_size=h.batch_size,
            shuffle=shuffle,
            num_workers=h.num_workers,
            pin_memory=h.pin_memory,
            drop_last=shuffle,
            persistent_workers=h.persistent_workers and h.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self._train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self._val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self._test, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        # Predict over the test split (used by test.py to write per-image CSVs).
        return self._loader(self._test, shuffle=False)
