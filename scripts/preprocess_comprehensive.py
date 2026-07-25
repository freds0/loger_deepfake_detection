"""Comprehensive Deepfake Detection Dataset (Roop / Akool) preprocessing.

This Kaggle dataset ships as a single zip of already face-cropped frames, so
there is no video/face-detection step here — just extraction and a manifest
CSV (``path,label,domain,split``) consumable by ``source: manifest`` in
:class:`src.data.datamodule.ForgeryDataModule`.

Expected raw layout (inside ``--zip-path``)::

    deepfake_dataset/real/<video>.mp4_frame_<n>_face_0.jpg       # label 0
    deepfake_dataset/deepfake/<video>.mp4_frame_<n>_face_0.jpg   # label 1

No official split is bundled, so frames are routed to train/val/test with a
deterministic hash of their source-video id (the ``<video>`` token before
``.mp4_frame``), keeping every frame of a given video in one split.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.manifest_utils import DomainRegistry, deterministic_split, write_manifest

_JUNK_PREFIXES = ("__MACOSX/", "._")


def video_key(filename: str) -> str:
    return filename.split(".mp4_frame")[0] if ".mp4_frame" in filename else filename


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Comprehensive Roop/Akool dataset preprocessing")
    p.add_argument(
        "--zip-path",
        default="/home/ubuntu/fred-experiments/DATASETS/DeepFake/"
        "Comprehensive Deepfake Detection Dataset Real and Synthetic Frames "
        "from Roop and Akool AI Technologies/deepfake_dataset.zip",
    )
    p.add_argument("--out-root", default="data/comprehensive_frames")
    p.add_argument("--manifest-out", default="data/manifests/comprehensive.csv")
    p.add_argument("--limit", type=int, default=None,
                   help="extract at most N images per class for a dry run")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    registry = DomainRegistry()
    fake_domain = registry.get_id("comprehensive:fake")

    with zipfile.ZipFile(args.zip_path) as zf:
        by_class: dict[str, list[zipfile.ZipInfo]] = {"real": [], "deepfake": []}
        for info in zf.infolist():
            if info.is_dir() or Path(info.filename).name == ".DS_Store":
                continue
            if any(part.startswith(_JUNK_PREFIXES) for part in info.filename.split("/")):
                continue
            for cls in by_class:
                if info.filename.startswith(f"deepfake_dataset/{cls}/"):
                    by_class[cls].append(info)
                    break

        rows = []
        for cls, label, domain in (("real", 0, 0), ("deepfake", 1, fake_domain)):
            members = sorted(by_class[cls], key=lambda i: i.filename)
            if args.limit is not None:
                members = members[: args.limit]
            for info in members:
                name = Path(info.filename).name
                split = deterministic_split(video_key(name), salt="comprehensive")
                dest_dir = out_root / split / cls
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / name
                if not dest_path.exists():
                    with zf.open(info) as src, open(dest_path, "wb") as dst:
                        dst.write(src.read())
                rows.append({"path": str(dest_path), "label": label, "domain": domain, "split": split})
            print(f"[{cls}] extracted {len(members)} images")

    print(f"\ntotal: {len(rows)} images -> {out_root}")
    if rows:
        write_manifest(rows, args.manifest_out)


if __name__ == "__main__":
    main()
