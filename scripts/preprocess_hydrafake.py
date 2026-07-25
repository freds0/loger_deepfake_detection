"""HydraFake (multi-generator deepfake benchmark) preprocessing.

Ships as pre-extracted face crops inside per-split zip archives, so there is
no video/face-detection step here — just extraction into
``data/hydrafake_frames`` and a manifest CSV (``path,label,domain,split``).

Expected raw layout (``--root``)::

    <root>/train/real.zip   -> real/<file>                       (label 0)
    <root>/train/fake.zip   -> fake/<category>/<method>/<file>   (label 1)
    <root>/val/real.zip     -> real/<file>                        (label 0)
    <root>/val/fake.zip     -> fake/<method>_<...>                (label 1, method
                                embedded as a prefix in the flat filename)
    <root>/test/{id,cd,cm,cf}.zip
        -> <source>/0_real/**   (label 0)
        -> <source>/1_fake/**   (label 1, possibly nested one level deeper
                                 per source video/identity)
        A few sources (e.g. ``facevid2vid`` in ``id.zip``) ship fake-only,
        flat under ``<source>/**`` — they reuse another source's real frames
        (mirrors the "reuse" methods in ``scripts/preprocess_df40.py``).

``id``/``cd``/``cm``/``cf`` = in-domain / cross-dataset / cross-manipulation /
cross-frequency test subsets; all map to manifest split "test" (the subset
name is kept in the extracted path for reference). Each generator method
becomes its own FSM domain (``hydrafake:<method>``), shared across
train/val/test so in-domain test frames land on the same domain ids as their
training counterparts, while cd/cm/cf introduce fresh domain ids for
genuinely unseen methods.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.manifest_utils import DomainRegistry, write_manifest

_JUNK_PREFIXES = ("__MACOSX/", "._")
_FLAT_METHOD_RE = re.compile(r"^([A-Za-z0-9]+?)(?:_\d|--)")

# FaceForensics++'s test/*.zip "1_fake" is the only test source that nests by
# manipulation *method* (Deepfakes/Face2Face/FaceSwap/NeuralTextures) rather
# than by per-video/per-sample folder — every other source's 1_fake subfolders
# are per-instance, so only this one needs the extra nesting level for domain.
_METHOD_NESTED_SOURCES = {"FaceForensics++"}


def _is_junk(name: str) -> bool:
    if Path(name).name == ".DS_Store":
        return True
    return any(part.startswith(_JUNK_PREFIXES) for part in name.split("/"))


def flat_method(filename: str) -> str:
    """Recover the generator name from HydraFake's flat val filenames, e.g.
    'AniPortraitVideo_04156--XDmfBTgIRUg_35--AniPortraitVideo_010.png' -> 'AniPortraitVideo'."""
    m = _FLAT_METHOD_RE.match(filename)
    return m.group(1) if m else Path(filename).stem


def extract_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo, dest: Path) -> None:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(dest, "wb") as dst:
            dst.write(src.read())


def iter_members(zf: zipfile.ZipFile, limit: int | None) -> list[zipfile.ZipInfo]:
    infos = [i for i in zf.infolist() if not i.is_dir() and not _is_junk(i.filename)]
    infos.sort(key=lambda i: i.filename)
    return infos[:limit] if limit is not None else infos


def process_train_val(root: Path, split: str, out_root: Path,
                       registry: DomainRegistry, limit: int | None) -> list[dict]:
    rows = []
    with zipfile.ZipFile(root / split / "real.zip") as zf:
        members = iter_members(zf, limit)
        for info in members:
            dest = out_root / split / info.filename
            extract_member(zf, info, dest)
            rows.append({"path": str(dest), "label": 0, "domain": 0, "split": split})
        print(f"[{split}/real] {len(members)} images")

    with zipfile.ZipFile(root / split / "fake.zip") as zf:
        members = iter_members(zf, limit)
        for info in members:
            parts = Path(info.filename).parts  # ("fake", category, method, file) or ("fake", file)
            method = parts[2] if len(parts) >= 4 else flat_method(parts[-1])
            domain = registry.get_id(f"hydrafake:{method}")
            dest = out_root / split / info.filename
            extract_member(zf, info, dest)
            rows.append({"path": str(dest), "label": 1, "domain": domain, "split": split})
        print(f"[{split}/fake] {len(members)} images")
    return rows


def process_test(root: Path, subset: str, out_root: Path,
                  registry: DomainRegistry, limit: int | None) -> list[dict]:
    rows = []
    with zipfile.ZipFile(root / "test" / f"{subset}.zip") as zf:
        members = iter_members(zf, None)  # discover source layout from the full listing first

        sources: dict[str, set[str]] = {}
        for info in members:
            parts = Path(info.filename).parts
            if len(parts) >= 2:
                sources.setdefault(parts[0], set()).add(parts[1])

        def group_key(info: zipfile.ZipInfo) -> tuple[str, str]:
            parts = Path(info.filename).parts
            source = parts[0]
            if {"0_real", "1_fake"} & sources.get(source, set()):
                return (source, parts[1] if len(parts) >= 2 else "")
            return (source, "")  # flat, fake-only source: one group per source

        if limit is not None:
            # cap per (source, class) so a dry run still touches every source
            by_key: dict[tuple[str, str], list[zipfile.ZipInfo]] = {}
            for info in members:
                by_key.setdefault(group_key(info), []).append(info)
            members = [i for group in by_key.values() for i in group[:limit]]

        for info in members:
            parts = Path(info.filename).parts
            source = parts[0]
            has_real_fake_split = {"0_real", "1_fake"} & sources.get(source, set())
            if has_real_fake_split:
                if len(parts) < 3 or parts[1] not in ("0_real", "1_fake"):
                    continue
                label = 0 if parts[1] == "0_real" else 1
            else:
                label = 1  # flat, fake-only source that reuses another source's real frames
            if label == 0:
                domain = 0
            elif source in _METHOD_NESTED_SOURCES and len(parts) >= 3:
                domain = registry.get_id(f"hydrafake:{parts[2]}")
            else:
                domain = registry.get_id(f"hydrafake:{source}")
            dest = out_root / "test" / subset / info.filename
            extract_member(zf, info, dest)
            rows.append({"path": str(dest), "label": label, "domain": domain, "split": "test"})
    print(f"[test/{subset}] {len(rows)} images")
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HydraFake extraction + manifest builder")
    p.add_argument("--root", default="/home/ubuntu/fred-experiments/DATASETS/DeepFake/HydraFake/HydraFake")
    p.add_argument("--out-root", default="data/hydrafake_frames")
    p.add_argument("--manifest-out", default="data/manifests/hydrafake.csv")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   choices=["train", "val", "test"])
    p.add_argument("--test-subsets", nargs="+", default=["id", "cd", "cm", "cf"],
                   choices=["id", "cd", "cm", "cf"])
    p.add_argument("--limit", type=int, default=None,
                   help="cap images per zip (train/val) or per source+class (test) for a dry run")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_root = Path(args.out_root)
    registry = DomainRegistry()

    rows = []
    for split in ("train", "val"):
        if split in args.splits:
            rows.extend(process_train_val(root, split, out_root, registry, args.limit))
    if "test" in args.splits:
        for subset in args.test_subsets:
            rows.extend(process_test(root, subset, out_root, registry, args.limit))

    print(f"\ntotal: {len(rows)} images -> {out_root}")
    if rows:
        write_manifest(rows, args.manifest_out)


if __name__ == "__main__":
    main()
