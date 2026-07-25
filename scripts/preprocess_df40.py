"""DF40 (40-method deepfake generation benchmark) manifest builder.

DF40 ships pre-extracted frames (no video/face-detection step needed here),
but each of its ~40 generator subfolders uses a slightly different layout.
Two families exist:

* **own** methods shipped their own real+fake reference frames, e.g.
  ``<method>/<method>/real/**`` + ``<method>/<method>/fake/**``.
* **reuse** methods (the majority: face-swap/reenactment/GAN/diffusion
  methods applied on top of Celeb-DF-v2 / FF++) only ship *fake* frames,
  nested under a ``cdf`` and/or ``ff`` branch (real frames for these domains
  come from ``scripts/preprocess_celebdf.py`` / ``scripts/preprocess_ffpp.py``
  instead — this script never emits label=0 rows for "reuse" methods).

``mobileswap`` is excluded: its ``cdf``/``ff`` branches only contain
un-extracted ``frames.zip`` archives (no actual images), unlike every other
method here.

Each method becomes its own FSM domain (``df40:<method>``). No official split
exists, so frames are routed to train/val/test via a deterministic hash of
their enclosing per-video/per-identity folder (or, when frames are flat with
no such folder, the file stem) so a given source video's frames stay in one
split. Output is a manifest CSV (``path,label,domain,split``) consumable by
``source: manifest`` in :class:`src.data.datamodule.ForgeryDataModule`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.manifest_utils import DomainRegistry, deterministic_split, write_manifest

_IMG_EXTS = {".png", ".jpg", ".jpeg"}
_FLAT_PARENT_NAMES = {"real", "fake", "deepfake", "frames"}

# method -> (real_subdir, fake_subdir, wrapper). wrapper is the path from
# DF40/<method>/ down to the directory that directly contains real_subdir/fake_subdir.
OWN_METHODS: dict[str, tuple[str, str, str]] = {
    "CollabDiff": ("real", "fake", "CollabDiff"),
    "MidJourney": ("real", "fake", "MidJourney"),
    "deepfacelab": ("real/frames", "fake/frames", "deepfacelab"),
    "heygen": ("real", "fake/frames", "heygen_new"),
    "stargan": ("real/real", "fake/fake", "stargan"),
    "starganv2": ("real", "fake", "starganv2"),
    "styleclip": ("real", "fake", "styleclip"),
    "whichfaceisreal": ("real", "fake", "whichfaceisreal"),
}

# method -> wrapper (path from DF40/<method>/ down to the dir containing cdf/ff).
REUSE_METHODS: dict[str, str] = {
    "DiT": "DiT",
    "MRAA": "",
    "RDDM": "RDDM",
    "SiT": "SiT",
    "StyleGAN2": "StyleGAN2",
    "StyleGAN3": "StyleGAN3",
    "StyleGANXL": "StyleGANXL",
    "VQGAN": "VQGAN",
    "blendface": "blendface",
    "danet": "",
    "ddim": "ddim",
    "e4e": "e4e/e4e",
    "e4s": "",
    "facedancer": "",
    "faceswap": "faceswap",
    "facevid2vid": "",
    "fomm": "",
    "fsgan": "",
    "hyperreenact": "",
    "inswap": "",
    "lia": "",
    "mcnet": "",
    "one_shot_free": "",
    "pirender": "",
    "pixart": "pixart",
    "sadtalker": "",
    "sd2.1": "sd2.1",
    "simswap": "",
    "tpsm": "",
    "uniface": "uniface",
    "wav2lip": "",
}
# mobileswap intentionally excluded: only un-extracted frames.zip archives, no images.


def _is_junk(p: Path) -> bool:
    return "__MACOSX" in p.parts or "landmarks" in p.parts or p.name == ".DS_Store" or p.name.startswith("._")


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and not _is_junk(p) and p.suffix.lower() in _IMG_EXTS
    )


def split_key(image_path: Path) -> str:
    parent_name = image_path.parent.name
    if parent_name in _FLAT_PARENT_NAMES:
        return image_path.stem
    return parent_name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DF40 manifest builder")
    p.add_argument("--root", default="/home/ubuntu/fred-experiments/DATASETS/DeepFake/DF40")
    p.add_argument("--manifest-out", default="data/manifests/df40.csv")
    p.add_argument("--limit-per-method", type=int, default=None,
                   help="cap images per (method, class) for a dry run")
    p.add_argument("--methods", nargs="+", default=None,
                   help="restrict to a subset of method names (default: all)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    registry = DomainRegistry()
    wanted = set(args.methods) if args.methods else None

    rows = []
    for method, (real_sub, fake_sub, wrapper) in OWN_METHODS.items():
        if wanted is not None and method not in wanted:
            continue
        base = root / method / wrapper
        domain = registry.get_id(f"df40:{method}")
        for subdir, label, dom in ((real_sub, 0, 0), (fake_sub, 1, domain)):
            files = image_files(base / subdir)
            if args.limit_per_method is not None:
                files = files[: args.limit_per_method]
            rows.extend(
                {"path": str(f), "label": label, "domain": dom,
                 "split": deterministic_split(split_key(f), salt="df40")}
                for f in files
            )
        print(f"[{method}] real={len(image_files(base / real_sub))} fake={len(image_files(base / fake_sub))}")

    for method, wrapper in REUSE_METHODS.items():
        if wanted is not None and method not in wanted:
            continue
        base = root / method / wrapper if wrapper else root / method
        domain = registry.get_id(f"df40:{method}")
        method_files: list[Path] = []
        for branch in ("cdf", "ff"):
            files = image_files(base / branch)
            method_files.extend(files)
        if args.limit_per_method is not None:
            method_files = method_files[: args.limit_per_method]
        rows.extend(
            {"path": str(f), "label": 1, "domain": domain,
             "split": deterministic_split(split_key(f), salt="df40")}
            for f in method_files
        )
        print(f"[{method}] fake={len(method_files)} (reuses external real frames)")

    print(f"\ntotal: {len(rows)} images")
    if rows:
        write_manifest(rows, args.manifest_out)


if __name__ == "__main__":
    main()
