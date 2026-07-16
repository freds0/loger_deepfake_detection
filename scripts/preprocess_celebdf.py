"""Celeb-DF-v2 frame extraction + face cropping.

Turns the raw Celeb-DF-v2 videos into face-crop PNGs and a manifest CSV
(``path,label,domain,split``) consumable by ``source: manifest`` in
:class:`src.data.datamodule.ForgeryDataModule`.

Expected raw layout (``--root``)::

    <root>/Celeb-real/id<N>_<k>.mp4          # bona-fide, label 0
    <root>/YouTube-real/<id>.mp4              # bona-fide, label 0
    <root>/Celeb-synthesis/id<A>_id<B>_<k>.mp4  # forgery, label 1

There is no official Celeb-DF-v2 test-list file bundled with this copy of the
dataset, so videos are routed to train/val/test with a deterministic hash of
their identity id (Celeb-real/synthesis: the ``idN`` token; YouTube-real: the
video stem), keeping a given identity's videos in a single split.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.manifest_utils import DomainRegistry, deterministic_split, write_manifest
from src.data.video_extract import build_mtcnn, process_video

_ID_RE = re.compile(r"id\d+")


def identity_key(stem: str) -> str:
    match = _ID_RE.search(stem)
    return match.group(0) if match else stem


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Celeb-DF-v2 frame extraction + face crop")
    p.add_argument("--root", default="/home/ubuntu/fred-experiments/DATASETS/DeepFake")
    p.add_argument("--out-root", default="data/celebdf_frames")
    p.add_argument("--manifest-out", default="data/manifests/celebdf.csv")
    p.add_argument("--frames-train", type=int, default=32)
    p.add_argument("--frames-val", type=int, default=10)
    p.add_argument("--frames-test", type=int, default=10)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--margin", type=float, default=1.3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit-videos", type=int, default=None,
                   help="process at most N videos per class for a dry run")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frames_per_split = {"train": args.frames_train, "val": args.frames_val, "test": args.frames_test}
    mtcnn = build_mtcnn(args.device)
    registry = DomainRegistry()

    classes = [
        ("real", os.path.join(args.root, "Celeb-real"), 0),
        ("real", os.path.join(args.root, "YouTube-real"), 0),
        ("synthesis", os.path.join(args.root, "Celeb-synthesis"), 1),
    ]

    rows = []
    grand_total = 0
    for cls_name, cls_dir, label in classes:
        if not os.path.isdir(cls_dir):
            print(f"[skip] missing dir: {cls_dir}")
            continue
        domain = 0 if label == 0 else registry.get_id("celebdf:synthesis")
        videos = sorted(p for p in os.listdir(cls_dir) if p.endswith(".mp4"))
        if args.limit_videos is not None:
            videos = videos[: args.limit_videos]
        for vid in videos:
            stem = Path(vid).stem
            split = deterministic_split(identity_key(stem), salt="celebdf")
            out_dir = os.path.join(args.out_root, split, cls_name, stem)
            n = process_video(
                os.path.join(cls_dir, vid), out_dir, mtcnn,
                frames_per_split[split], args.margin, args.image_size,
            )
            grand_total += n
            rows.extend(
                {"path": str(p), "label": label, "domain": domain, "split": split}
                for p in sorted(Path(out_dir).glob("*.png"))
            )
        print(f"[{Path(cls_dir).name}] processed {len(videos)} videos")

    print(f"\ntotal: {grand_total} face crops -> {args.out_root}")
    if rows:
        write_manifest(rows, args.manifest_out)


if __name__ == "__main__":
    main()
