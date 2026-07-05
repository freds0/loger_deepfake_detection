"""FaceForensics++ frame extraction + face cropping for OSDFD training.

Turns the raw FF++ videos in ``data/FaceForensics++_C23`` into the per-split,
per-manipulation face-crop layout expected by
:class:`src.data.datamodule.ForgeryDataModule`::

    <out>/<split>/real/<video>/<frame>.png                 # label 0, domain 0
    <out>/<split>/Deepfakes/<video>/<frame>.png            # domain 1
    <out>/<split>/Face2Face/<video>/<frame>.png            # domain 2
    <out>/<split>/FaceSwap/<video>/<frame>.png             # domain 3
    <out>/<split>/NeuralTextures/<video>/<frame>.png       # domain 4

Pipeline per video (matches the paper's procedure, Sec. IV-A):
  1. sample ``frames_per_video`` evenly-spaced frames,
  2. detect faces with MTCNN (facenet-pytorch, GPU),
  3. take the largest face, enlarge the box by ``margin`` (paper: 1.3x),
     squared to avoid aspect distortion,
  4. resize to ``image_size`` (224) and save as PNG.

Split assignment uses the official FF++ protocol JSONs (720/140/140 videos):
real ``XXX.mp4`` and fake ``SRC_TGT.mp4`` are routed by their youtube id, so no
identity leaks across train/val/test. The extra ``DeepFakeDetection`` subset
(actor-based naming) is not covered by the youtube-id split and is skipped by
the default manipulation list.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image

# Output subfolder -> forgery source-domain id (must match src/data/dataset.py).
DOMAINS = {
    "real": 0,
    "Deepfakes": 1,
    "Face2Face": 2,
    "FaceSwap": 3,
    "NeuralTextures": 4,
    "FaceShifter": 5,
}
CLASSIC = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def load_id_to_split(splits_dir: str) -> dict[str, str]:
    """Map each youtube video id to its official split.

    Each split JSON is a list of ``[a, b]`` id pairs; both ids of a pair (and
    thus the reenactment videos ``a_b`` / ``b_a``) belong to the same split.
    """
    id_to_split: dict[str, str] = {}
    for split in ("train", "val", "test"):
        pairs = json.load(open(os.path.join(splits_dir, f"{split}.json")))
        for a, b in pairs:
            id_to_split[a] = split
            id_to_split[b] = split
    return id_to_split


def video_split(stem: str, id_to_split: dict[str, str]) -> str | None:
    """Route a video stem (``000`` or ``000_003``) to its split."""
    first_id = stem.split("_")[0]
    return id_to_split.get(first_id)


def sample_frame_indices(total: int, k: int) -> list[int]:
    """Evenly-spaced frame indices (at most ``k``, at least 1)."""
    if total <= 0:
        return []
    k = min(k, total)
    return list(np.linspace(0, total - 1, num=k, dtype=int))


def read_frames(video_path: str, indices: list[int]) -> list[np.ndarray]:
    """Read the requested frame indices as RGB uint8 arrays (sequential scan)."""
    cap = cv2.VideoCapture(video_path)
    wanted = set(indices)
    frames: list[np.ndarray] = []
    idx = 0
    max_idx = max(indices) if indices else -1
    while idx <= max_idx:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return frames


def crop_face(
    frame: np.ndarray,
    box: np.ndarray,
    margin: float,
    size: int,
) -> Image.Image:
    """Square-crop the face with ``margin`` enlargement and resize to ``size``."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * margin  # square box avoids aspect distortion
    half = side / 2.0
    nx1, ny1 = max(0, int(cx - half)), max(0, int(cy - half))
    nx2, ny2 = min(w, int(cx + half)), min(h, int(cy + half))
    crop = frame[ny1:ny2, nx1:nx2]
    return Image.fromarray(crop).resize((size, size), Image.BICUBIC)


def process_video(
    video_path: str,
    out_dir: str,
    mtcnn: MTCNN,
    k: int,
    margin: float,
    size: int,
) -> int:
    """Extract, detect, crop and save faces for one video. Returns #saved."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    indices = sample_frame_indices(total, k)
    frames = read_frames(video_path, indices)
    if not frames:
        return 0

    # MTCNN batches a list of same-sized frames from one video.
    boxes_list, probs_list = mtcnn.detect(frames)

    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for i, (frame, boxes, probs) in enumerate(zip(frames, boxes_list, probs_list)):
        if boxes is None or len(boxes) == 0:
            continue
        # Largest detected face.
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        box = boxes[int(np.argmax(areas))]
        img = crop_face(frame, box, margin, size)
        img.save(os.path.join(out_dir, f"{indices[i]:04d}.png"))
        saved += 1
    return saved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FF++ frame extraction + face crop")
    p.add_argument("--video-root", default="data/FaceForensics++_C23")
    p.add_argument("--splits-dir", default="data/ffpp_splits")
    p.add_argument("--out-root", default="data/ffpp_frames")
    p.add_argument("--manipulations", nargs="+", default=CLASSIC,
                   help=f"subset of {list(DOMAINS)[1:]} (default: 4 classic)")
    p.add_argument("--frames-train", type=int, default=32)
    p.add_argument("--frames-val", type=int, default=10)
    p.add_argument("--frames-test", type=int, default=10)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--margin", type=float, default=1.3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit-videos", type=int, default=None,
                   help="process at most N videos per (split, class) for a dry run")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    id_to_split = load_id_to_split(args.splits_dir)
    frames_per_split = {"train": args.frames_train, "val": args.frames_val, "test": args.frames_test}
    mtcnn = MTCNN(keep_all=True, device=args.device, post_process=False)

    # Classes to process: real + selected manipulations.
    classes = [("real", os.path.join(args.video_root, "real"))]
    for m in args.manipulations:
        classes.append((m, os.path.join(args.video_root, "fake", m)))

    grand_total = 0
    per_split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for cls_name, cls_dir in classes:
        if not os.path.isdir(cls_dir):
            print(f"[skip] missing dir: {cls_dir}")
            continue
        videos = sorted(p for p in os.listdir(cls_dir) if p.endswith(".mp4"))
        # Group by split, then optionally limit per (split, class) for dry runs.
        seen_per_split: dict[str, int] = {"train": 0, "val": 0, "test": 0}
        for vid in videos:
            stem = Path(vid).stem
            split = video_split(stem, id_to_split)
            if split is None:  # e.g. DeepFakeDetection actor ids
                continue
            if args.limit_videos is not None and seen_per_split[split] >= args.limit_videos:
                continue
            seen_per_split[split] += 1
            out_dir = os.path.join(args.out_root, split, cls_name, stem)
            n = process_video(
                os.path.join(cls_dir, vid), out_dir, mtcnn,
                frames_per_split[split], args.margin, args.image_size,
            )
            grand_total += n
            per_split_counts[split] += n
        print(f"[{cls_name}] processed "
              f"{ {s: seen_per_split[s] for s in seen_per_split} } videos")

    print("\n=== Done ===")
    for s in ("train", "val", "test"):
        print(f"{s:>5}: {per_split_counts[s]} face crops")
    print(f"total: {grand_total} face crops -> {args.out_root}")


if __name__ == "__main__":
    main()
