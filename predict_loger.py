"""LOGER + FSM inference entry point for one image or a folder."""

from __future__ import annotations

import argparse
import os

import pandas as pd

from src.inference.loger_predictor import LOGERPredictor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LOGER + FSM forgery inference")
    p.add_argument("--ckpt", required=True, help="Path to a LOGER Lightning .ckpt")
    p.add_argument("--input", required=True, help="Image file or folder of images")
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--device", default=None, help="cuda | cpu (auto by default)")
    p.add_argument("--face-crop", action="store_true", help="Detect+crop face first")
    p.add_argument("--face-backend", default="opencv", choices=["opencv", "retinaface", "dlib"])
    p.add_argument("--csv", default=None, help="Optional CSV output path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    predictor = LOGERPredictor(
        ckpt_path=args.ckpt,
        device=args.device,
        image_size=args.image_size,
        use_face_crop=args.face_crop,
        face_backend=args.face_backend,
    )
    results = predictor.predict_folder(args.input) if os.path.isdir(args.input) else [
        predictor.predict_image(args.input)
    ]

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
