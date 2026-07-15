"""FaceForensics++-style frame dataset.

Supports two ways of specifying data:

1. **Folder layout** (recommended for FF++). Frames are pre-extracted / cropped
   and organised per split and per manipulation type::

       <root>/<split>/real/**/*.png              # bona-fide faces  (label 0)
       <root>/<split>/Deepfakes/**/*.png         # forgery domain 1 (label 1)
       <root>/<split>/Face2Face/**/*.png         # forgery domain 2 (label 1)
       <root>/<split>/FaceSwap/**/*.png          # forgery domain 3 (label 1)
       <root>/<split>/NeuralTextures/**/*.png    # forgery domain 4 (label 1)

   The user pre-splits videos into ``train``/``val``/``test`` following the
   official FF++ protocol (720 / 140 / 140 videos).

2. **Manifest CSV** with columns ``path,label,domain[,split]`` for arbitrary
   datasets (e.g. cross-dataset evaluation on CDF, DFDC, WDF, ...).

Each item is a dict with keys ``pixel_values``, ``label`` (binary),
``domain`` (forgery source-domain id; 0 for real) and ``path``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

# "In the wild" datasets (NTIRE, web-scraped, ...) can contain truncated JPEGs;
# decode as much as is available instead of raising OSError mid-training.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Forgery source-domain ids used by the Forgery Style Mixture module.
# 0 is reserved for real faces; each manipulation type is a distinct domain.
FFPP_DOMAINS: dict[str, int] = {
    "real": 0,
    "Deepfakes": 1,
    "Face2Face": 2,
    "FaceSwap": 3,
    "NeuralTextures": 4,
}

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
_SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation", "dev"),
    "test": ("test", "testing", "eval", "evaluation"),
}
_REAL_ALIASES = (
    "real",
    "original",
    "authentic",
    "bonafide",
    "bona_fide",
    "pristine",
)


@dataclass
class Record:
    """A single frame sample."""

    path: str
    label: int   # 1 = fake, 0 = real
    domain: int  # forgery source-domain id (0 for real)


def _scan_dir(directory: str | Path) -> list[str]:
    root = Path(directory)
    return sorted(str(p) for p in root.rglob("*") if p.suffix.lower() in _IMG_EXTS)


def _resolve_split_dir(root: str | Path, split: str) -> Path:
    """Resolve a split directory using common aliases."""
    root = Path(root)
    candidates = _SPLIT_ALIASES.get(split, (split,))
    for name in candidates:
        path = root / name
        if path.is_dir():
            return path
    available = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    raise FileNotFoundError(
        f"Could not infer split '{split}' under {root}. "
        f"Tried {candidates}; available directories: {available}"
    )


def _is_real_folder(name: str) -> bool:
    lower = name.lower().replace("-", "_")
    return any(alias in lower for alias in _REAL_ALIASES)


def _infer_domain_map(split_dir: Path) -> dict[str, int]:
    """Infer class/domain ids from immediate class folders of a split."""
    class_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    if not class_dirs:
        raise RuntimeError(f"No class folders found under {split_dir}")

    mapping: dict[str, int] = {}
    next_domain = 1
    for class_dir in class_dirs:
        if _is_real_folder(class_dir.name):
            mapping[class_dir.name] = 0
        else:
            mapping[class_dir.name] = next_domain
            next_domain += 1
    if all(domain != 0 for domain in mapping.values()):
        raise RuntimeError(
            f"Could not infer a real class folder under {split_dir}. "
            f"Use data.domain_map to mark the real folder with domain 0."
        )
    return mapping


def records_from_faceforensics(
    root: str,
    split: str,
    domain_map: dict[str, int] | None = None,
) -> list[Record]:
    """Build records by scanning an FF++-style split folder.

    Args:
        root: Dataset root containing per-split subfolders.
        split: ``"train"``, ``"val"`` or ``"test"``.
        domain_map: Mapping ``{subfolder_name: domain_id}``; ``"real"`` (or any
            entry mapped to 0) is treated as bona-fide. Defaults to
            :data:`FFPP_DOMAINS`.

    Returns:
        A list of :class:`Record`.
    """
    split_dir = _resolve_split_dir(root, split)
    domain_map = domain_map or _infer_domain_map(split_dir)

    records: list[Record] = []
    for name, domain in domain_map.items():
        sub = split_dir / name
        if not sub.is_dir():
            continue
        label = 0 if domain == 0 else 1
        for path in _scan_dir(sub):
            records.append(Record(path=path, label=label, domain=domain))
    if not records:
        raise RuntimeError(f"No images found under {split_dir}")
    return records


def records_from_manifest(csv_path: str, split: str | None = None) -> list[Record]:
    """Build records from a manifest CSV (``path,label,domain[,split]``)."""
    df = pd.read_csv(csv_path)
    required = {"path", "label", "domain"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest must contain columns {required}, got {list(df.columns)}")
    if split is not None and "split" in df.columns:
        df = df[df["split"] == split]
    return [
        Record(path=r.path, label=int(r.label), domain=int(r.domain))
        for r in df.itertuples(index=False)
    ]


def records_from_ntire(root: str, shard_nums: list[int]) -> list[Record]:
    """Build records from the NTIRE 2026 shard layout.

    Each ``<root>/shard_<n>/labels.csv`` has columns ``,image_name,label``
    (``label``: 0 = real, 1 = fake) and images live under
    ``<root>/shard_<n>/images/``. The dataset carries no per-generator
    domain metadata, so the binary label doubles as the FSM domain id
    (0 = real, 1 = fake).
    """
    root = Path(root)
    records: list[Record] = []
    for num in shard_nums:
        shard_dir = root / f"shard_{num}"
        csv_path = shard_dir / "labels.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"labels.csv not found under {shard_dir}")
        df = pd.read_csv(csv_path, index_col=0)
        for row in df.itertuples(index=False):
            label = int(row.label)
            records.append(
                Record(
                    path=str(shard_dir / "images" / row.image_name),
                    label=label,
                    domain=label,
                )
            )
    if not records:
        raise RuntimeError(f"No images found for shards {shard_nums} under {root}")
    return records


def split_records_by_hash(
    records: list[Record],
    val_fraction: float,
    split: str,
) -> list[Record]:
    """Deterministic, RNG-free train/val split keyed by image filename.

    A record is routed to ``val`` iff the MD5 hash of its filename (the
    basename, not the full path, so the split is stable across machines/roots
    with the same images) falls in the bottom ``val_fraction`` of the hash
    space. Two calls on the same records always agree, independent of
    ``seed``, process, or call order -- unlike a shuffle-and-slice split,
    nothing here is randomised.

    Args:
        records: Records to split (typically every shard assigned to train).
        val_fraction: Fraction of records routed to ``"val"``.
        split: ``"train"`` (complement) or ``"val"``.

    Returns:
        The matching subset of ``records`` (order preserved).
    """
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    threshold = int(val_fraction * 10_000)

    def _in_val(rec: Record) -> bool:
        digest = hashlib.md5(Path(rec.path).name.encode()).hexdigest()
        return int(digest, 16) % 10_000 < threshold

    if split == "val":
        return [r for r in records if _in_val(r)]
    return [r for r in records if not _in_val(r)]


def oversample_real(records: list[Record], factor: int) -> list[Record]:
    """Replicate real records ``factor`` times to balance positives/negatives.

    Paper (Sec. IV-B): "To balance positive and negative samples during
    training, real faces are augmented fourfold."
    """
    if factor <= 1:
        return records
    reals = [r for r in records if r.label == 0]
    return records + reals * (factor - 1)


class ForgeryFrameDataset(Dataset):
    """Dataset of forgery/real face frames.

    Args:
        records: List of :class:`Record`.
        transform: A callable mapping a PIL image to a tensor.
        face_cropper: Optional :class:`~src.data.face_detection.FaceCropper`.
    """

    def __init__(self, records: list[Record], transform, face_cropper=None) -> None:
        self.records = records
        self.transform = transform
        self.face_cropper = face_cropper

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        image = Image.open(rec.path).convert("RGB")
        if self.face_cropper is not None:
            image = self.face_cropper(image)
        pixel_values = self.transform(image)
        return {
            "pixel_values": pixel_values,
            "label": torch.tensor(rec.label, dtype=torch.long),
            "domain": torch.tensor(rec.domain, dtype=torch.long),
            "path": rec.path,
        }
