"""SDFVD 2.0 (Small-Scale Deep Fake Video Dataset) frame extraction + face cropping.

Turns the raw SDFVD videos into face-crop PNGs and a manifest CSV
(``path,label,domain,split``) consumable by ``source: manifest`` in
:class:`src.data.datamodule.ForgeryDataModule`.

Expected raw layout (``--root``)::

    <root>/SDFVD2.0_real/real_v<N>_aug_<k>.mp4         # bona-fide, label 0
    <root>/SDFVD2.0_fake/fake_vs<N>_aug_<k>[-hash].mp4  # forgery, label 1
    <root>/SDFVD2.0_fake/vs<N>.mp4                       # forgery, label 1 (no aug)

No official split is bundled, so videos are routed to train/val/test with a
deterministic hash of their source-video id (the ``v<N>``/``vs<N>`` token),
keeping augmented copies of the same source video in one split.
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

_ID_RE = re.compile(r"vs?\d+")


def identity_key(stem: str) -> str:
    match = _ID_RE.search(stem)
    return match.group(0) if match else stem


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SDFVD 2.0 frame extraction + face crop")
    p.add_argument(
        "--root",
        default="/home/ubuntu/fred-experiments/DATASETS/DeepFake/"
        "SDFVD2.0 Extension of Small Scale Deep Fake Video Dataset",
    )
    p.add_argument("--out-root", default="data/sdfvd_frames")
    p.add_argument("--manifest-out", default="data/manifests/sdfvd.csv")
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
    fake_domain = registry.get_id("sdfvd:fake")

    classes = [
        ("real", os.path.join(args.root, "SDFVD2.0_real"), 0, 0),
        ("fake", os.path.join(args.root, "SDFVD2.0_fake"), 1, fake_domain),
    ]

    rows = []
    grand_total = 0
    for cls_name, cls_dir, label, domain in classes:
        if not os.path.isdir(cls_dir):
            print(f"[skip] missing dir: {cls_dir}")
            continue
        videos = sorted(p for p in os.listdir(cls_dir) if p.endswith(".mp4"))
        if args.limit_videos is not None:
            videos = videos[: args.limit_videos]
        for vid in videos:
            stem = Path(vid).stem
            split = deterministic_split(identity_key(stem), salt="sdfvd")
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
        print(f"[{cls_name}] processed {len(videos)} videos")

    print(f"\ntotal: {grand_total} face crops -> {args.out_root}")
    if rows:
        write_manifest(rows, args.manifest_out)


if __name__ == "__main__":
    main()
