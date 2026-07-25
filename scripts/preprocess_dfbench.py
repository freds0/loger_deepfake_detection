"""DFBench (general AI-generated image detection) preprocessing.

DFBench is not face-specific (like NTIRE, it's a general real-vs-AI-generated
image benchmark): 8 real-image sources (IQA sets: CLIVE/CSIQ/LIVE/TID2013/
kadid10k/koniq10k/Flick8k, plus ``partial_source``) and 13 generator sources
(Janus, Kandinsky-3, Kolors, LaVi-Bridge, NOVA, PixArt, Playground,
ali_flux_dev, ali_flux_schnell, infinity, sd3_5_large, sd3_medium, edit),
distributed as per-source zips plus ``img_train.jsonl``/``img_test.jsonl``
label files (each record: ``{"image": "DFBench/<prefix>/<name>", ...,
"conversations":[..., {"value": "A"|"B"}]}``, A = real, B = generated).

Each zip's internal layout doesn't match the ``DFBench/<prefix>/...`` paths
used by the label files (e.g. generator zips are flat ``000001.jpg`` at the
zip root, LIVE nests distortion-type subfolders with a different extension
than the json expects), so this script extracts every zip, indexes the
extracted files by ``(subdir, stem)`` ignoring extension, and joins that
index against the json records. A handful of records (~15, mostly PixArt/
ali_flux_schnell) have no matching file and are skipped with a warning.

Output is a manifest CSV (``path,label,domain,split``) consumable by
``source: manifest`` in :class:`src.data.datamodule.ForgeryDataModule`. The
``test`` split comes from ``img_test.jsonl``; ``train``/``val`` are carved
out of ``img_train.jsonl`` via a deterministic hash of the file stem.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.manifest_utils import DomainRegistry, deterministic_split, write_manifest

PREFIX_TO_ZIP = {
    "CLIVE": "CLIVE.zip",
    "CSIQ": "CSIQ.zip",
    "LIVE": "LIVE.zip",
    "TID2013": "TID2013.zip",
    "kadid10k": "kadid10k.zip",
    "koniq10k": "koniq10k_512x384.zip",
    "Flick8k": "Flick8kimg.zip",
    "partial_source": "partial_source.zip",
    "Janus": "Janus_000001-040000.zip",
    "Kandinsky-3": "Kandinsky-3_000001-040000.zip",
    "Kolors": "Kolors_000001-040000.zip",
    "LaVi-Bridge": "LaVi-Bridge_000001-040000.zip",
    "NOVA": "NOVA_000001-040000.zip",
    "PixArt": "PixArt_000001-040000.zip",
    "Playground": "Playground_000001-040000.zip",
    "ali_flux_dev": "ali_flux_dev_000001-040000.zip",
    "ali_flux_schnell": "ali_flux_schnell_000001-040000.zip",
    "infinity": "infinity_000001-040000.zip",
    "sd3_5_large": "sd3_5_large_000001-040000.zip",
    "sd3_medium": "sd3_medium_000001-040000.zip",
    "edit": "edit.zip",
}
_JUNK_NAMES = {"captions.txt", "dmos.csv", ".DS_Store", "info.txt", "train.txt", "test.txt", "Thumbs.db"}


def extract_zip(zip_path: Path, out_dir: Path, limit: int | None) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        if limit is not None:
            members = members[:limit]
        for member in members:
            name = Path(member.filename).name
            if name in _JUNK_NAMES or name.startswith("._"):
                continue
            dest = out_dir / member.filename
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(dest, "wb") as dst:
                dst.write(src.read())


def build_index(extract_dir: Path) -> dict[tuple[str, str], Path]:
    index: dict[tuple[str, str], Path] = {}
    if not extract_dir.is_dir():
        return index
    for f in extract_dir.rglob("*"):
        if not f.is_file() or f.name in _JUNK_NAMES or f.name.startswith("._"):
            continue
        rel_dir = str(f.relative_to(extract_dir).parent)
        index[(rel_dir, f.stem)] = f
        index.setdefault(("", f.stem), f)  # flat fallback for unambiguous stems
    return index


def resolve(index: dict[tuple[str, str], Path], prefix: str, rest: str) -> Path | None:
    rest_path = Path(rest)
    subdir = str(rest_path.parent) if str(rest_path.parent) != "." else ""
    return index.get((subdir, rest_path.stem)) or index.get(("", rest_path.stem))


def _find_answer(turns) -> str | None:
    """Recursively find the 'A'/'B' assistant answer.

    ``img_train.json`` nests a flat ``messages: [{"content": "A"|"B"}, ...]``
    list, while ``img_train.jsonl`` wraps that same list one level deeper
    inside ``conversations: [{"from": "gpt", "value": [...]}]``.
    """
    for turn in turns:
        value = turn.get("content", turn.get("value"))
        if isinstance(value, str) and value in ("A", "B"):
            return value
        if isinstance(value, list):
            answer = _find_answer(value)
            if answer is not None:
                return answer
    return None


def iter_records(jsonl_path: Path):
    with jsonl_path.open() as f:
        for line in f:
            rec = json.loads(line)
            image_field = rec.get("images") or rec.get("image")
            turns = rec.get("messages") or rec.get("conversations") or []
            answer = _find_answer(turns)
            if image_field is None or answer is None:
                continue
            yield image_field, answer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DFBench manifest builder")
    p.add_argument("--root", default="/home/ubuntu/fred-experiments/DATASETS/DeepFake/DFBench/DFBench")
    p.add_argument("--out-root", default="data/dfbench_raw")
    p.add_argument("--manifest-out", default="data/manifests/dfbench.csv")
    p.add_argument("--limit-per-zip", type=int, default=None,
                   help="extract at most N members per zip for a dry run")
    p.add_argument("--prefixes", nargs="+", default=None,
                   help="restrict to a subset of prefixes (default: all 21)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_root = Path(args.out_root)
    registry = DomainRegistry()
    prefixes = args.prefixes or list(PREFIX_TO_ZIP)

    indexes: dict[str, dict[tuple[str, str], Path]] = {}
    for prefix in prefixes:
        zip_path = root / PREFIX_TO_ZIP[prefix]
        extract_dir = out_root / prefix
        extract_zip(zip_path, extract_dir, args.limit_per_zip)
        indexes[prefix] = build_index(extract_dir)
        print(f"[{prefix}] indexed {len(indexes[prefix])} extracted files")

    rows = []
    missing = 0
    for jsonl_name, base_split in (("img_train.jsonl", "train"), ("img_test.jsonl", "test")):
        jsonl_path = root / jsonl_name
        if not jsonl_path.is_file():
            continue
        for image_field, answer in iter_records(jsonl_path):
            parts = image_field.split("/", 2)
            if len(parts) < 3 or parts[1] not in indexes:
                continue
            prefix, rest = parts[1], parts[2]
            resolved = resolve(indexes[prefix], prefix, rest)
            if resolved is None:
                missing += 1
                continue
            label = 0 if answer == "A" else 1
            domain = 0 if label == 0 else registry.get_id(f"dfbench:{prefix}")
            split = base_split
            if base_split == "train":
                split = deterministic_split(rest, ratios={"train": 0.9, "val": 0.1}, salt="dfbench")
            rows.append({"path": str(resolved), "label": label, "domain": domain, "split": split})

    print(f"\ntotal: {len(rows)} images ({missing} json records had no matching extracted file)")
    if rows:
        write_manifest(rows, args.manifest_out)


if __name__ == "__main__":
    main()
