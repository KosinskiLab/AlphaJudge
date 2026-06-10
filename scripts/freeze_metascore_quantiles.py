#!/usr/bin/env python3
"""Freeze the AlphaJudge metascore calibration deciles from the benchmark.

The percentile sliders in AlphaJudge reports map each raw interface descriptor
onto a frozen percentile scale (``BENCHMARK_QUANTILES`` in
``alphajudge.meta_score``). Those deciles must describe the distribution of
*real, interacting* complexes, so a new prediction is ranked against the
population of true interfaces rather than against a benchmark that is half
non-interacting decoys. This script therefore calibrates on **positive
(interacting) pairs only** by default; the database-negative re-pairings are
excluded.

For each metascore feature the value is oriented "higher is better" using
``FEATURE_DIRECTIONS`` (PAE and solvation energy are sign-flipped), NaNs are
dropped, and ``numpy.quantile`` is evaluated at ``CALIBRATION_LEVELS``
(deciles 0.0..1.0). The printed dictionary can be pasted into
``src/alphajudge/meta_score.py``.

Only ``numpy`` and the standard library are used (no pandas), so the script
runs in a stock AlphaJudge install.

Usage:
    python scripts/freeze_metascore_quantiles.py \
        --input-csv .../benchmark_best....csv \
        [--label-filter positive]   # use "all" to reproduce the legacy scale
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from alphajudge.meta_score import (
    BENCHMARK_QUANTILES,
    CALIBRATION_LEVELS,
    FEATURE_DIRECTIONS,
)

DEFAULT_BENCHMARK_CSV = Path(
    "/scratch/dima/benchmark_26/final_sync_20260523_225722/staged_benchmark/"
    "benchmark_best.final_sync_20260523_225722_force_recompute_nointerfacefix.csv"
)


def _safe_float(value) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def feature_deciles(rows: list[dict[str, str]], feature: str, direction: float) -> tuple[np.ndarray, int]:
    oriented = np.asarray([_safe_float(row.get(feature)) for row in rows], dtype=float) * direction
    oriented = oriented[np.isfinite(oriented)]
    return np.quantile(oriented, list(CALIBRATION_LEVELS)), len(oriented)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=str(DEFAULT_BENCHMARK_CSV))
    parser.add_argument(
        "--label-filter",
        default="positive",
        choices=("positive", "negative", "all"),
        help="Subset of rows to calibrate on (default: positive interacting pairs).",
    )
    args = parser.parse_args()

    with Path(args.input_csv).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"no rows found in {args.input_csv}")

    if args.label_filter != "all":
        rows = [r for r in rows if str(r.get("label", "")).strip().lower() == args.label_filter]
        if not rows:
            raise SystemExit(f"no rows with label '{args.label_filter}' in {args.input_csv}")

    header = set(rows[0])
    print(f"# input: {args.input_csv}")
    print(f"# label-filter: {args.label_filter}  rows: {len(rows)}")
    print("BENCHMARK_QUANTILES = {")
    for feature in BENCHMARK_QUANTILES:
        if feature not in header:
            raise SystemExit(f"missing column in CSV: {feature}")
        quantiles, n_finite = feature_deciles(rows, feature, FEATURE_DIRECTIONS[feature])
        print(f'    "{feature}": (  # n_finite={n_finite}')
        for value in quantiles:
            print(f"        {float(value)!r},")
        print("    ),")
    print("}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
