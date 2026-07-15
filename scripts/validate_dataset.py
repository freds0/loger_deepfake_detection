"""Validate a NTIRE-style shard dataset before training.

Checks, for every image referenced by ``<root>/shard_<n>/labels.csv``:
  1. the image file exists under ``<root>/shard_<n>/images/``,
  2. it decodes without error (``Image.open(...).load()``, with
     ``ImageFile.LOAD_TRUNCATED_IMAGES`` enabled to match training-time
     tolerance for truncated JPEGs),
  3. its label is 0 or 1.

Problems are written to ``<root>/validation_report.csv`` (columns
``shard,image_name,status``); per-shard and aggregate counts are printed to
stdout. Exits 0 if every image is ok, 1 otherwise.

Usage:
    python scripts/validate_dataset.py --root data/NTIRE-RobustAIGenDetection-train
    python scripts/validate_dataset.py --root <root> --shards 0 1 2 --workers 32
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFile

# Mirrors src/data/dataset.py so validation reflects exactly what training will
# tolerate: a truncated JPEG loads (partially) rather than raising OSError.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _discover_shards(root: Path) -> list[int]:
    nums = []
    for p in sorted(root.glob("shard_*")):
        if p.is_dir():
            try:
                nums.append(int(p.name.split("_", 1)[1]))
            except ValueError:
                continue
    return sorted(nums)


def _check_one(task: tuple[str, str, int, str]) -> tuple[str, str, str]:
    """Check a single image. Returns ``(shard_name, image_name, status)``."""
    shard_name, image_name, label, path = task
    if label not in (0, 1):
        return shard_name, image_name, "bad_label"
    p = Path(path)
    if not p.is_file():
        return shard_name, image_name, "missing"
    try:
        with Image.open(p) as img:
            img.load()
    except Exception:
        return shard_name, image_name, "corrupt"
    return shard_name, image_name, "ok"


def validate(root: Path, shard_nums: list[int], workers: int) -> list[tuple[str, str, str]]:
    """Check every image of the given shards. Returns a list of
    ``(shard_name, image_name, status)`` for ALL images (not just problems)."""
    tasks: list[tuple[str, str, int, str]] = []
    for num in shard_nums:
        shard_dir = root / f"shard_{num}"
        csv_path = shard_dir / "labels.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"labels.csv not found under {shard_dir}")
        df = pd.read_csv(csv_path, index_col=0)
        for row in df.itertuples(index=False):
            image_path = shard_dir / "images" / row.image_name
            tasks.append((shard_dir.name, row.image_name, row.label, str(image_path)))

    with Pool(workers) as pool:
        return pool.map(_check_one, tasks, chunksize=256)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--shards", type=int, nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    shard_nums = args.shards if args.shards is not None else _discover_shards(args.root)
    if not shard_nums:
        print(f"No shard_* directories found under {args.root}")
        return 1

    results = validate(args.root, shard_nums, args.workers)

    counts_by_shard: dict[str, Counter] = {}
    for shard_name, _, status in results:
        counts_by_shard.setdefault(shard_name, Counter())[status] += 1

    print(f"{'shard':<12} {'ok':>8} {'missing':>8} {'corrupt':>8} {'bad_label':>10}")
    for shard_name in sorted(counts_by_shard, key=lambda s: int(s.split("_", 1)[1])):
        c = counts_by_shard[shard_name]
        print(f"{shard_name:<12} {c['ok']:>8} {c['missing']:>8} {c['corrupt']:>8} {c['bad_label']:>10}")

    total = Counter(status for _, _, status in results)
    print(f"{'TOTAL':<12} {total['ok']:>8} {total['missing']:>8} {total['corrupt']:>8} {total['bad_label']:>10}")

    problems = [(shard, name, status) for shard, name, status in results if status != "ok"]
    if problems:
        report_path = args.root / "validation_report.csv"
        with open(report_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["shard", "image_name", "status"])
            writer.writerows(problems)
        print(f"\n{len(problems)} problem(s) written to {report_path}")
        return 1

    print("\nAll images ok.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
