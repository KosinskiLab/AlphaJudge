#!/usr/bin/env python3
"""Audit pair-matched AlphaFold prediction completion against a frozen pair list."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ORGANISMS = ("arabidopsis", "ecoli", "human", "yeast")
MODELS = ("af2", "af3")
PAIRSETS = (("pos_pairs", "pos", "positive"), ("neg_pairs", "neg", "negative"))

DEFAULT_PAIR_ROOT = Path(
    "/g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26/"
    "frozen_inputs/pairmatched_20260505_fetch3_march2"
)
DEFAULT_PRED_ROOT = Path("/scratch/dima/benchmark_26/predictions")


def read_pairs(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def completed_marker(pred_root: Path, organism: str, model: str, pairset: str, pair: str) -> Path:
    return pred_root / organism / model / pairset / "predictions" / pair / "completed_fold.txt"


def row_for_cell(pair_root: Path, pred_root: Path, organism: str, model: str, pairset: str, short: str, label: str) -> dict[str, int | str]:
    pair_path = pair_root / "by_org" / f"{organism}_{short}.pairs.txt"
    pairs = read_pairs(pair_path)
    target_pairs = set(pairs)
    prediction_dir = pred_root / organism / model / pairset / "predictions"
    dirs = {path.name for path in prediction_dir.iterdir() if path.is_dir()} if prediction_dir.exists() else set()
    completed = sum(completed_marker(pred_root, organism, model, pairset, pair).exists() for pair in pairs)
    target_dirs = len(target_pairs & dirs)
    return {
        "organism": organism,
        "model": model,
        "pairset": pairset,
        "label": label,
        "input_pairs": len(pairs),
        "target_dirs": target_dirs,
        "completed": completed,
        "missing": len(pairs) - completed,
        "dirs_total": len(dirs),
        "dirs_extra": len(dirs - target_pairs),
    }


def write_missing_lists(pair_root: Path, pred_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for organism in ORGANISMS:
        for model in MODELS:
            for pairset, short, _label in PAIRSETS:
                pairs = read_pairs(pair_root / "by_org" / f"{organism}_{short}.pairs.txt")
                missing = [
                    pair
                    for pair in pairs
                    if not completed_marker(pred_root, organism, model, pairset, pair).exists()
                ]
                path = out_dir / f"{organism}_{model}_{pairset}.pairs.txt"
                path.write_text("".join(f"{pair}\n" for pair in missing))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", type=Path, default=DEFAULT_PAIR_ROOT)
    parser.add_argument("--pred-root", type=Path, default=DEFAULT_PRED_ROOT)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--write-missing-dir", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, int | str]] = []
    for organism in ORGANISMS:
        for model in MODELS:
            for pairset, short, label in PAIRSETS:
                rows.append(row_for_cell(args.pair_root, args.pred_root, organism, model, pairset, short, label))

    fields = [
        "organism",
        "model",
        "pairset",
        "label",
        "input_pairs",
        "target_dirs",
        "completed",
        "missing",
        "dirs_total",
        "dirs_extra",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    totals = {field: "" for field in fields}
    totals.update(
        {
            "organism": "TOTAL",
            "input_pairs": sum(int(row["input_pairs"]) for row in rows),
            "target_dirs": sum(int(row["target_dirs"]) for row in rows),
            "completed": sum(int(row["completed"]) for row in rows),
            "missing": sum(int(row["missing"]) for row in rows),
            "dirs_total": sum(int(row["dirs_total"]) for row in rows),
            "dirs_extra": sum(int(row["dirs_extra"]) for row in rows),
        }
    )
    writer.writerow(totals)

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as handle:
            out_writer = csv.DictWriter(handle, fieldnames=fields)
            out_writer.writeheader()
            out_writer.writerows(rows)
            out_writer.writerow(totals)

    if args.write_missing_dir:
        write_missing_lists(args.pair_root, args.pred_root, args.write_missing_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
