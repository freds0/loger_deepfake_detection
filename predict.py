"""OSDFD inference entry point for a single image or a folder of images.

FSM is always disabled at inference (paper Sec. III-C).

Usage:
    python predict.py --ckpt model.ckpt --input path/to/image.png
    python predict.py --ckpt model.ckpt --input path/to/folder --csv out.csv
    python predict.py --ckpt model.ckpt --input img.png --face-crop
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from src.inference.predictor import OSDFDPredictor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OSDFD (SigLIP 2) forgery inference")
    p.add_argument("--ckpt", required=True, help="Path to a Lightning .ckpt")
    p.add_argument("--input", required=True, help="Image file or folder of images")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--device", default=None, help="cuda | cpu (auto by default)")
    p.add_argument("--face-crop", action="store_true", help="Detect+crop face first")
    p.add_argument("--face-backend", default="opencv", choices=["opencv", "retinaface", "dlib"])
    p.add_argument("--csv", default=None, help="Optional CSV output path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    predictor = OSDFDPredictor(
        ckpt_path=args.ckpt,
        device=args.device,
        image_size=args.image_size,
        use_face_crop=args.face_crop,
        face_backend=args.face_backend,
    )

    if os.path.isdir(args.input):
        results = predictor.predict_folder(args.input)
    else:
        results = [predictor.predict_image(args.input)]

    for r in results:
        print(
            f"{r.path}\n  -> {r.label.upper():5s} | P(fake)={r.probability:.4f} "
            f"| confidence={r.confidence:.4f} | logit={r.logit:.4f}"
        )

    if args.csv:
        pd.DataFrame([r.__dict__ for r in results]).to_csv(args.csv, index=False)
        print(f"\nWrote {len(results)} predictions to {args.csv}")


if __name__ == "__main__":
    main()
