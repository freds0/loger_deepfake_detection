"""HiDF (High-resolution face-swap Deepfake) manifest builder.

Ships as pre-extracted, ready-to-use images: no video/face-detection step
needed. Real and fake images already live in separate folders, so this only
builds a manifest CSV (``path,label,domain,split``), referencing the images
in place rather than copying them (like ``scripts/preprocess_df40.py``).

Expected raw layout (``--root``)::

    <root>/Real-img/<id>.{png,jpg,jpeg}             # bona-fide faces, label 0
    <root>/Image/<source_id>_<target_id>.{jpg,png}  # face-swap fakes, label 1

(``<root>/Fake-img.zip`` / ``Real-img.zip`` are the source archives for
``Image/`` / ``Real-img/`` respectively and are not read directly here since
both are already extracted on disk.)

No official split is bundled, so images are routed to train/val/test with a
deterministic hash of the leading identity token in the filename, keeping a
given identity's bona-fide photo and any fakes swapped from it grouped
together.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.manifest_utils import DomainRegistry, deterministic_split, write_manifest

_IMG_EXTS = {".png", ".jpg", ".jpeg"}


def identity_key(stem: str) -> str:
    return stem.split("_")[0]


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in _IMG_EXTS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HiDF manifest builder")
    p.add_argument("--root", default="/home/ubuntu/fred-experiments/DATASETS/DeepFake/HiDF")
    p.add_argument("--manifest-out", default="data/manifests/hidf.csv")
    p.add_argument("--limit", type=int, default=None,
                   help="cap images per class for a dry run")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    registry = DomainRegistry()
    fake_domain = registry.get_id("hidf:faceswap")

    rows = []
    for subdir, label, domain in (("Real-img", 0, 0), ("Image", 1, fake_domain)):
        files = image_files(root / subdir)
        if args.limit is not None:
            files = files[: args.limit]
        rows.extend(
            {"path": str(f), "label": label, "domain": domain,
             "split": deterministic_split(identity_key(f.stem), salt="hidf")}
            for f in files
        )
        print(f"[{subdir}] {len(files)} images")

    print(f"\ntotal: {len(rows)} images")
    if rows:
        write_manifest(rows, args.manifest_out)


if __name__ == "__main__":
    main()
