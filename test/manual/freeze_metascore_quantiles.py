#!/usr/bin/env python3
"""Manually freeze AlphaJudge metascore calibration deciles from a benchmark CSV.

This is a developer calibration helper, not part of the release CLI and not run
by pytest. Invoke it manually with an explicit benchmark CSV path.

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
    python test/manual/freeze_metascore_quantiles.py \
        --input-csv .../benchmark_best....csv \
        --ccc-csv .../confident_contacts_full_v3.csv \
        [--label-filter positive] [--backend-filter pooled|af2|af3]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from alphajudge.meta_score import (
    BENCHMARK_QUANTILES,
    CALIBRATION_LEVELS,
    FEATURE_DIRECTIONS,
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


def add_ccc_values(rows: list[dict[str, str]], ccc_csv: Path) -> int:
    """Join the raw-model CCC extraction onto benchmark interface rows."""
    with ccc_csv.open(newline="") as handle:
        ccc_rows = list(csv.DictReader(handle))
    lookup = {}
    for raw_row in ccc_rows:
        if raw_row.get("status") != "ok":
            continue
        key = tuple(
            raw_row.get(field, "")
            for field in ("organism", "backend", "label", "pair")
        )
        lookup[key] = raw_row.get("ccc_ab_pae4", "")
    joined = 0
    for row in rows:
        backend = row.get("backend") or row.get("model", "")
        key = (
            row.get("organism", ""),
            backend,
            row.get("label", ""),
            row.get("pair", ""),
        )
        value = lookup.get(key, "")
        row["interface_ccc"] = value
        if math.isfinite(_safe_float(value)):
            joined += 1
    return joined


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument(
        "--ccc-csv",
        type=Path,
        help=(
            "Optional raw CCC extraction to join by organism/backend/label/pair; "
            "required when the benchmark CSV has no interface_ccc column."
        ),
    )
    parser.add_argument(
        "--label-filter",
        default="positive",
        choices=("positive", "negative", "all"),
        help="Subset of rows to calibrate on (default: positive interacting pairs).",
    )
    parser.add_argument(
        "--backend-filter",
        default="pooled",
        choices=("pooled", "af2", "af3"),
        help="Pool backends or freeze one backend-specific ladder (default: pooled).",
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

    if args.backend_filter != "pooled":
        rows = [
            r for r in rows
            if (r.get("backend") or r.get("model", "")).strip().lower()
            == args.backend_filter
        ]
        if not rows:
            raise SystemExit(
                f"no rows for backend '{args.backend_filter}' in {args.input_csv}"
            )

    if args.ccc_csv is not None:
        joined = add_ccc_values(rows, args.ccc_csv)
        print(f"# CCC input: {args.ccc_csv}  joined rows: {joined}")

    header = set(rows[0])
    print(f"# input: {args.input_csv}")
    print(f"# label-filter: {args.label_filter}  rows: {len(rows)}")
    print(f"# backend-filter: {args.backend_filter}")
    print("BENCHMARK_QUANTILES = {")
    for feature in BENCHMARK_QUANTILES:
        if feature not in header:
            if feature == "interface_ccc":
                raise SystemExit(
                    "missing column in CSV: interface_ccc (pass --ccc-csv to join "
                    "the raw CCC extraction)"
                )
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
