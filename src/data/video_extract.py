"""Shared MTCNN face-crop frame extraction for video-based preprocessing.

Used by ``scripts/preprocess_{ffpp,celebdf,sdfvd}.py``: sample evenly-spaced
frames from a video, detect faces with MTCNN (facenet-pytorch, GPU), take the
largest face per frame, enlarge its box by ``margin`` (squared to avoid
aspect distortion), resize to ``size`` and save as PNG.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image


def build_mtcnn(device: str) -> MTCNN:
    return MTCNN(keep_all=True, device=device, post_process=False)


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
