"""Performance regression test for ForgeryDataModule._limit_records (P0.5).

Before the fix, the ``remaining`` fallback used ``r not in selected`` where
``Record`` is a dataclass, so ``in`` performs a value comparison against every
element of ``selected`` -- O(n*m) overall. That fallback only triggers when
the minority class is scarce enough that the round-robin selection alone
can't fill the quota, so a dataset with a large majority class and a small
minority class hits it directly. This asserts a wall-clock bound, not just
correctness, so a regression back to O(n*m) fails the suite long before
anyone hits it on the real 200k+ record NTIRE shards.
"""

from __future__ import annotations

import time

from src.data.dataset import Record
from src.data.datamodule import ForgeryDataModule


def test_limit_records_hits_remaining_fallback_and_stays_fast():
    # 200k real + 50 fake: fake is exhausted well before the quota is filled,
    # forcing the `if len(selected) < limit` fallback (the O(n*m) hot path).
    records = [Record(path=f"/data/r_{i}.jpg", label=0, domain=0) for i in range(200_000)]
    records += [Record(path=f"/data/f_{i}.jpg", label=1, domain=(i % 4) + 1) for i in range(50)]

    dm = ForgeryDataModule(source="manifest", manifest="unused.csv", max_records_per_split=1000)

    start = time.perf_counter()
    limited = dm._limit_records(records, "train")
    elapsed = time.perf_counter() - start

    assert len(limited) == 1000
    assert len({id(r) for r in limited}) == 1000  # no duplicates
    assert sum(1 for r in limited if r.label == 1) == 50  # every scarce fake kept
    assert elapsed < 1.0, f"_limit_records took {elapsed:.2f}s, expected < 1.0s"
