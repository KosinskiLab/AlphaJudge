#!/usr/bin/env python3
"""Analyze AlphaJudge score redundancy and transparent metascore candidates."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from alphajudge.meta_score import META_SCORE_FEATURES, interface_meta_score

DEFAULT_BENCHMARK_CSV = Path(
    "/g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26/"
    "merged_best_interfaces_all_models.20260422_111918_904fef5.csv"
)

ANALYSIS_FEATURES = (
    "iptm_ptm",
    "iptm",
    "ptm",
    "confidence_score",
    "pDockQ/mpDockQ",
    "average_interface_pae",
    "interface_average_plddt",
    "interface_num_intf_residues",
    "interface_polar",
    "interface_hydrophobic",
    "interface_charged",
    "interface_contact_pairs",
    "interface_score",
    "interface_pDockQ2",
    "interface_ipSAE",
    "interface_LIS",
    "interface_hb",
    "interface_sb",
    "interface_sc",
    "interface_area",
    "interface_solv_en",
)

RANK_METASCORES = {
    "confidence3": ("interface_LIS", "interface_ipSAE", "interface_pDockQ2"),
    "selected_current": META_SCORE_FEATURES,
    "all_analysis_features": ANALYSIS_FEATURES,
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def safe_float(value) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def labels(rows: list[dict[str, str]]) -> np.ndarray:
    values = []
    for row in rows:
        label = str(row.get("label", "")).strip().lower()
        if label not in {"positive", "negative"}:
            raise ValueError("input CSV must have label values 'positive'/'negative'")
        values.append(1 if label == "positive" else 0)
    return np.asarray(values, dtype=int)


def values_for(rows: list[dict[str, str]], feature: str) -> np.ndarray:
    return np.asarray([safe_float(row.get(feature)) for row in rows], dtype=float)


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def auc_score(y: np.ndarray, scores: np.ndarray) -> float:
    mask = np.isfinite(scores)
    y = y[mask]
    scores = scores[mask]
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(scores)
    sum_pos = float(ranks[y == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    mask = np.isfinite(scores)
    y = y[mask]
    scores = scores[mask]
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_y = y[order]
    sorted_scores = scores[order]
    change_points = np.r_[np.flatnonzero(np.diff(sorted_scores)) + 1, len(sorted_scores)]
    previous_end = 0
    cumulative_pos = 0
    ap = 0.0
    for end in change_points:
        group_pos = int(sorted_y[previous_end:end].sum())
        if group_pos:
            cumulative_pos += group_pos
            precision_at_threshold = cumulative_pos / end
            ap += (group_pos / n_pos) * precision_at_threshold
        previous_end = end
    return float(ap)


def rank01(values: np.ndarray) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    mask = np.isfinite(values)
    if not mask.any():
        return out
    ranks = average_ranks(values[mask])
    out[mask] = (ranks - 1.0) / max(mask.sum() - 1, 1)
    return out


def oriented_feature(rows: list[dict[str, str]], y: np.ndarray, feature: str) -> tuple[np.ndarray, int, float, float]:
    values = values_for(rows, feature)
    auc_pos = auc_score(y, values)
    auc_neg = auc_score(y, -values)
    if np.nan_to_num(auc_neg, nan=-1.0) > np.nan_to_num(auc_pos, nan=-1.0):
        values = -values
        return values, -1, auc_neg, average_precision(y, values)
    return values, 1, auc_pos, average_precision(y, values)


def present_features(rows: list[dict[str, str]], candidates: Iterable[str]) -> list[str]:
    header = set(rows[0]) if rows else set()
    return [feature for feature in candidates if feature in header]


def rank_metascore(rows: list[dict[str, str]], y: np.ndarray, features: Iterable[str]) -> np.ndarray:
    columns = []
    for feature in present_features(rows, features):
        values, _, _, _ = oriented_feature(rows, y, feature)
        ranked = rank01(values)
        if np.isfinite(ranked).any():
            ranked[~np.isfinite(ranked)] = float(np.nanmedian(ranked))
            columns.append(ranked)
    if not columns:
        return np.full(len(rows), np.nan, dtype=float)
    return np.vstack(columns).T.mean(axis=1)


def print_single_feature_table(rows: list[dict[str, str]], y: np.ndarray, top: int) -> None:
    print("\nSingle Features")
    table = []
    for feature in present_features(rows, ANALYSIS_FEATURES):
        values, sign, auc, ap = oriented_feature(rows, y, feature)
        table.append((auc, ap, sign, int(np.isfinite(values).sum()), feature))
    for auc, ap, sign, n, feature in sorted(table, reverse=True)[:top]:
        print(f"{feature:32s} sign={sign:+d} auroc={auc:.3f} ap={ap:.3f} n={n}")


def print_metascore_table(rows: list[dict[str, str]], y: np.ndarray) -> None:
    print("\nMetascores")
    production = np.asarray([interface_meta_score(row) for row in rows], dtype=float)
    print(
        f"{'interface_meta_score':32s} "
        f"auroc={auc_score(y, production):.3f} ap={average_precision(y, production):.3f}"
    )
    for name, features in RANK_METASCORES.items():
        score = rank_metascore(rows, y, features)
        print(f"{name:32s} auroc={auc_score(y, score):.3f} ap={average_precision(y, score):.3f}")


def print_correlations(rows: list[dict[str, str]], y: np.ndarray) -> None:
    features = present_features(rows, META_SCORE_FEATURES)
    if len(features) < 2:
        return
    columns = []
    for feature in features:
        values, _, _, _ = oriented_feature(rows, y, feature)
        ranked = rank01(values)
        ranked[~np.isfinite(ranked)] = float(np.nanmedian(ranked))
        columns.append(ranked)
    matrix = np.vstack(columns).T
    corr = np.corrcoef(matrix, rowvar=False)

    print("\nSpearman-Like Correlations, Oriented Higher-Is-Better")
    print("feature," + ",".join(features))
    for idx, feature in enumerate(features):
        print(feature + "," + ",".join(f"{corr[idx, j]:.2f}" for j in range(len(features))))


def print_pca(rows: list[dict[str, str]], y: np.ndarray) -> None:
    features = present_features(rows, META_SCORE_FEATURES)
    if len(features) < 2:
        return
    columns = []
    for feature in features:
        values, _, _, _ = oriented_feature(rows, y, feature)
        ranked = rank01(values)
        ranked[~np.isfinite(ranked)] = float(np.nanmedian(ranked))
        columns.append(ranked)
    matrix = np.vstack(columns).T
    std = matrix.std(axis=0)
    std[std == 0.0] = 1.0
    matrix = (matrix - matrix.mean(axis=0)) / std
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    explained = singular**2 / max(len(matrix) - 1, 1)
    explained_ratio = explained / explained.sum()
    pcs = u * singular

    print("\nPCA On Oriented Rank Features")
    for idx in range(min(5, pcs.shape[1])):
        pc = pcs[:, idx]
        auc_pos = auc_score(y, pc)
        auc_neg = auc_score(y, -pc)
        best_auc = max(auc_pos, auc_neg)
        sign = 1 if auc_pos >= auc_neg else -1
        loadings = sorted(
            ((abs(vt[idx, j]), vt[idx, j], features[j]) for j in range(len(features))),
            reverse=True,
        )[:5]
        loading_text = "; ".join(f"{feature}:{value:+.2f}" for _, value, feature in loadings)
        print(
            f"PC{idx + 1} explained={explained_ratio[idx]:.3f} "
            f"auroc={best_auc:.3f} sign={sign:+d} loadings={loading_text}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_BENCHMARK_CSV),
        help="Benchmark CSV with AlphaJudge score columns and positive/negative labels.",
    )
    parser.add_argument("--top", type=int, default=20, help="Number of single-feature rows to print.")
    parser.add_argument(
        "--require-meta-auroc",
        type=float,
        default=None,
        help="Exit nonzero if production interface_meta_score AUROC is below this value.",
    )
    args = parser.parse_args()

    rows = read_csv_rows(Path(args.input_csv))
    if not rows:
        raise SystemExit(f"no rows found in {args.input_csv}")
    y = labels(rows)

    print(f"Input: {args.input_csv}")
    print(f"Rows: {len(rows)} positives={int(y.sum())} negatives={int(len(y) - y.sum())}")
    print_single_feature_table(rows, y, top=args.top)
    print_metascore_table(rows, y)
    print_correlations(rows, y)
    print_pca(rows, y)

    if args.require_meta_auroc is not None:
        production = np.asarray([interface_meta_score(row) for row in rows], dtype=float)
        meta_auc = auc_score(y, production)
        if not math.isfinite(meta_auc) or meta_auc < args.require_meta_auroc:
            raise SystemExit(
                f"interface_meta_score AUROC {meta_auc:.3f} is below required {args.require_meta_auroc:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
