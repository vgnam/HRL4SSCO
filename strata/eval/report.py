"""Regret table of a STRATA benchmark CSV.

Usage::

    python -m strata.eval.report out/strata_im.csv

The report groups the per-episode rows written by :mod:`strata.eval.run` by
``(law, dataset, method)`` and prints the mean relative regret, its standard
error over episodes and the worst episode.  Positive regret means the method
is worse than the oracle-law flat planner.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

GroupKey = Tuple[str, str, str]  # law, dataset, method


def load_rows(path: str) -> List[dict]:
    """Read a benchmark CSV (empty list when the file has no rows)."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarise(rows: Sequence[dict]) -> Dict[GroupKey, dict]:
    """Mean / SE / maximum regret per ``(law, dataset, method)``."""
    groups: Dict[GroupKey, List[float]] = defaultdict(list)
    for row in rows:
        key = (str(row["law"]), str(row["dataset"]), str(row["method"]))
        groups[key].append(float(row["regret"]))
    out: Dict[GroupKey, dict] = {}
    for key, values in groups.items():
        arr = np.asarray(values, dtype=np.float64)
        n = int(arr.size)
        se = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        out[key] = {
            "mean": float(arr.mean()),
            "se": se,
            "max": float(arr.max()),
            "n": n,
            "mean_value": float(
                np.mean([float(r["value"]) for r in rows if _key(r) == key])
            ),
        }
    return out


def _key(row: dict) -> GroupKey:
    return (str(row["law"]), str(row["dataset"]), str(row["method"]))


def format_table(summary: Dict[GroupKey, dict]) -> str:
    """Format the summary as a fixed-width text table."""
    header = (
        "law        dataset    method          n   mean_regret      se   max_regret"
    )
    lines = [header, "-" * len(header)]
    for key in sorted(summary):
        law, dataset, method = key
        stats = summary[key]
        lines.append(
            "%-10s %-10s %-15s %3d %+12.3f%% %7.3f%% %+11.3f%%"
            % (
                law,
                dataset,
                method,
                stats["n"],
                100.0 * stats["mean"],
                100.0 * stats["se"],
                100.0 * stats["max"],
            )
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> str:
    parser = argparse.ArgumentParser(description="Print the STRATA regret table.")
    parser.add_argument("path", help="CSV written by strata.eval.run")
    args = parser.parse_args(argv)
    table = format_table(summarise(load_rows(args.path)))
    print(table)
    return table


if __name__ == "__main__":  # pragma: no cover
    main()
