"""Shared helpers for building cross-dataset manifest CSVs.

Every ``scripts/preprocess_*.py`` script turns its raw dataset into a
manifest CSV with columns ``path,label,domain,split`` consumable by
``source: manifest`` in :class:`src.data.datamodule.ForgeryDataModule`. This
module holds the three pieces of logic shared across all of them:

  * :class:`DomainRegistry` -- a persistent name -> forgery-domain-id map
    (``data/manifests/domain_registry.json``) so every manipulation
    method/generator across every dataset gets its own stable FSM domain id,
    never reused across runs or scripts.
  * :func:`deterministic_split` -- a salted hash-based train/val/test
    assignment for datasets that ship no official split file, so re-running a
    preprocessing script never reshuffles an already-processed
    video/identity/image across splits.
  * :func:`write_manifest` -- writes the collected rows to a manifest CSV.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

# Domain ids 0-5 are reserved for FF++'s hardcoded folder-name mapping
# (src/data/dataset.py::FFPP_DOMAINS, scripts/preprocess_ffpp.py::DOMAINS);
# every other dataset's manipulation methods register a fresh id here.
_FIRST_DYNAMIC_DOMAIN = 6


class DomainRegistry:
    """Persistent ``name -> forgery-domain-id`` map shared across preprocessing runs.

    Backed by a JSON file (default ``data/manifests/domain_registry.json``):
    loaded on construction, saved after every newly-assigned id, so ids stay
    stable across scripts and reruns -- required for cross-dataset FSM domain
    consistency (see ``configs/loger_fsm_combined.yaml``).
    """

    def __init__(self, path: str | Path = "data/manifests/domain_registry.json") -> None:
        self.path = Path(path)
        if self.path.is_file():
            self._map: dict[str, int] = json.loads(self.path.read_text())
        else:
            self._map = {}

    def get_id(self, name: str) -> int:
        """Return ``name``'s domain id, assigning and persisting a new one if needed."""
        if name not in self._map:
            next_id = max([*self._map.values(), _FIRST_DYNAMIC_DOMAIN - 1]) + 1
            self._map[name] = next_id
            self._save()
        return self._map[name]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._map, indent=2, sort_keys=True))


def deterministic_split(
    key: str,
    salt: str = "",
    ratios: dict[str, float] | None = None,
) -> str:
    """Hash ``key`` into one of ``ratios``' splits, deterministically.

    Args:
        key: Identity/video/file key to route (e.g. a video stem).
        salt: Per-dataset salt so the same key hashes differently across
            datasets (avoids correlated splits between e.g. FF++ and DF40).
        ratios: ``{split_name: fraction}`` in the order splits should be
            tried, fractions summing to <= 1.0. Defaults to
            ``{"train": 0.8, "val": 0.1, "test": 0.1}``.

    Returns:
        The split name ``key`` was routed to.
    """
    ratios = ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
    digest = hashlib.md5(f"{salt}:{key}".encode()).hexdigest()
    bucket = (int(digest, 16) % 10_000) / 10_000.0
    cumulative = 0.0
    last_split = next(iter(ratios))
    for split, ratio in ratios.items():
        cumulative += ratio
        last_split = split
        if bucket < cumulative:
            return split
    return last_split  # floating-point rounding safety net


def write_manifest(rows: list[dict], path: str | Path) -> None:
    """Write collected ``{path,label,domain,split}`` rows to a manifest CSV."""
    df = pd.DataFrame(rows, columns=["path", "label", "domain", "split"])
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[manifest] wrote {len(df)} rows -> {out}")
