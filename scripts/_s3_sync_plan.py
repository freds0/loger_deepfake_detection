#!/usr/bin/env python3
"""Compute the minimal set of directories that can be uploaded wholesale
(`aws s3 sync <dir> ...`) for a manifest, without ever pulling in a file the
manifest doesn't reference.

A directory is "safe" to sync as a whole when the number of files on disk
under it (recursively) exactly matches the number of manifest rows that fall
under it. When it doesn't match — some sibling content exists that the
manifest doesn't reference, e.g. HiDF's Real-img.zip sitting next to
Real-img/, or DF40's landmarks/ sibling to frames/ — we descend one path
component at a time and re-check each child, only falling back to per-file
granularity where a directory is genuinely mixed.

Usage: python3 _s3_sync_plan.py <manifest.csv>   # prints one dir per line
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict


def file_count(path: str) -> int:
    n = 0
    for _root, _dirs, files in os.walk(path):
        n += len(files)
    return n


def descend(dirpath: str, paths: list[str]) -> list[str]:
    if not os.path.isdir(dirpath):
        return []
    if file_count(dirpath) == len(paths):
        return [dirpath]

    buckets: dict[str, list[str]] = defaultdict(list)
    for p in paths:
        rel = os.path.relpath(p, dirpath)
        parts = rel.split(os.sep, 1)
        buckets[parts[0] if len(parts) > 1 else ""].append(p)

    result: list[str] = []
    for child, sub_paths in buckets.items():
        if child == "":
            # file lives directly in dirpath -- can't isolate it from siblings
            # by directory alone, so the whole directory is the sync unit.
            result.append(dirpath)
        else:
            result.extend(descend(os.path.join(dirpath, child), sub_paths))
    return result


def plan(paths: list[str]) -> list[str]:
    if not paths:
        return []
    common = os.path.commonpath(paths)
    return sorted(set(descend(common, paths)))


def main() -> None:
    manifest_csv = sys.argv[1]
    paths = []
    with open(manifest_csv) as f:
        next(f)  # header
        for line in f:
            paths.append(line.rstrip("\n").split(",", 1)[0])
    for d in plan(paths):
        print(d)


if __name__ == "__main__":
    main()
