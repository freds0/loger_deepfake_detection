"""Logit-average two or more LOGER ``predictions.csv`` files (PLAN_v0.2 V12).

Each ``predictions.csv`` (written by ``test_loger.py``) has columns
``path, prob, pred[, label]``. This script aligns the runs on ``path``,
averages their per-sample logits ``logit(prob) = log(p / (1 - p))`` (the
paper's inference-time ensemble is a mean of logits, Eq. 3), converts the mean
back to a probability, recomputes the full metric suite against ``label`` and
writes a fused ``predictions.csv``.

Averaging in logit space rather than probability space matches the LOGER
ensemble formulation; probabilities are clipped to ``[eps, 1-eps]`` first so
saturated scores (0 or 1) give finite logits.

Usage:
    python scripts/fuse_predictions.py run_a/predictions.csv run_b/predictions.csv
    python scripts/fuse_predictions.py a.csv b.csv c.csv --output fused.csv
"""

from __future__ import annotations

import argparse
import sys
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `python scripts/fuse_predictions.py` from the repo root (the script's
# own dir is on sys.path, not the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.metrics import compute_metrics

EPS = 1e-6


def _to_logit(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob.astype(float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def fuse(csv_paths: list[str]) -> pd.DataFrame:
    """Inner-join the runs on ``path`` and add the fused ``prob``/``pred``."""
    frames = []
    for i, path in enumerate(csv_paths):
        df = pd.read_csv(path)
        if "path" not in df.columns or "prob" not in df.columns:
            raise ValueError(f"{path} must have 'path' and 'prob' columns")
        cols = {"prob": f"prob_{i}"}
        if "label" in df.columns:
            cols["label"] = f"label_{i}"
        frames.append(df.rename(columns=cols)[["path", *cols.values()]])

    merged = reduce(lambda a, b: a.merge(b, on="path", how="inner"), frames)
    if merged.empty:
        raise ValueError("No overlapping `path` rows across the given files.")

    logits = np.stack([_to_logit(merged[f"prob_{i}"].to_numpy()) for i in range(len(frames))])
    fused_logit = logits.mean(axis=0)
    fused_prob = 1.0 / (1.0 + np.exp(-fused_logit))

    out = pd.DataFrame({"path": merged["path"], "prob": fused_prob, "pred": (fused_prob >= 0.5).astype(int)})
    label_cols = [c for c in merged.columns if c.startswith("label_")]
    if label_cols:
        # Labels are per-sample ground truth; identical across runs after the
        # join, so any column is the truth.
        out["label"] = merged[label_cols[0]].to_numpy()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_paths", nargs="+", help="Two or more predictions.csv files")
    parser.add_argument("--output", default="fused_predictions.csv")
    args = parser.parse_args()

    if len(args.csv_paths) < 2:
        parser.error("give at least two predictions.csv files to fuse")

    out = fuse(args.csv_paths)
    out.to_csv(args.output, index=False)
    print(f"Fused {len(args.csv_paths)} runs over {len(out)} aligned samples -> {args.output}")

    if "label" in out.columns:
        metrics = compute_metrics(out["label"].to_numpy(), out["prob"].to_numpy())
        width = max(len(k) for k in metrics)
        for name, value in metrics.items():
            print(f"  {name:<{width}}  {value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
