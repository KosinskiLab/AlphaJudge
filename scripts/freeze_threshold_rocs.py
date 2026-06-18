#!/usr/bin/env python3
"""Freeze the benchmark ROC curves that back the AlphaJudge threshold calculator.

The threshold calculator (``alphajudge.thresholds``) is a deterministic lookup
over the benchmark ROC, not a model: for a chosen score and AlphaFold version it
inverts the stored ROC at a requested false-positive rate (or precision/recall)
and projects precision to the user's prevalence. This script computes those ROCs
once from the main 4-organism benchmark and writes them to
``src/alphajudge/threshold_rocs.json``, mirroring how
``freeze_metascore_quantiles.py`` freezes ``BENCHMARK_QUANTILES``.

Calibrated on POSITIVE vs DATABASE-NEGATIVE pairs (the same population the paper
ROC/AUROC numbers come from). For each ``score x {af2, af3, both}`` we store a
dense threshold grid with FPR/TPR and pre-computed Wilson 95% half-widths, plus
``fpr_resolution`` (~1/n_neg) so the runtime can flag tail-extrapolation.

``both`` is the AF2-and-AF3 agreement curve: a single shared operating quantile
``q`` sets each version's threshold to its own ``q``-quantile, and a pair is
called positive iff BOTH versions clear their own threshold (the paper's
n=40 / 93% point is one sample on this curve).

numpy + stdlib only (no pandas/sklearn), so it runs in a stock AlphaJudge env.

Usage:
    python scripts/freeze_threshold_rocs.py \
        --input-csv .../benchmark_best.balanced.csv \
        --scores interface_ipSAE interface_LIS \
        --out src/alphajudge/threshold_rocs.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_common as bc  # noqa: E402

DEFAULT_CSV = (
    "/scratch/dima/benchmark_26/final_sync_20260523_225722/"
    "staged_benchmark/benchmark_best.balanced.csv"
)
DEFAULT_SCORES = (
    "interface_ipSAE",
    "interface_LIS",
    "interface_pDockQ2",
    "iptm",
    "interface_sc",
)
GRID_N = 401           # threshold samples per curve (>~5x the ~1/n_neg resolution)
Z = 1.959963984540054  # 95% normal quantile for Wilson interval


def _sha16(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _wilson_halfwidth(p: np.ndarray, n: int) -> np.ndarray:
    """95% Wilson interval half-width for a proportion p estimated from n trials.

    Returned as a single symmetric-ish half-width (mean of the two arms) so the
    frozen artifact stays compact; the runtime treats it as +/-.
    """
    if n <= 0:
        return np.zeros_like(p)
    denom = 1.0 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / denom
    margin = (Z / denom) * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    lo = np.clip(centre - margin, 0.0, 1.0)
    hi = np.clip(centre + margin, 0.0, 1.0)
    return (hi - lo) / 2.0


def _roc_on_grid(y: np.ndarray, s: np.ndarray, grid: np.ndarray):
    """FPR/TPR at each threshold in ``grid`` (call positive iff s >= threshold)."""
    finite = np.isfinite(s)
    y, s = y[finite], s[finite]
    pos = s[y == 1]
    neg = s[y == 0]
    npos, nneg = len(pos), len(neg)
    # fraction of pos/neg with score >= t, vectorised over the grid
    tpr = np.array([(pos >= t).mean() if npos else np.nan for t in grid])
    fpr = np.array([(neg >= t).mean() if nneg else np.nan for t in grid])
    return fpr, tpr, npos, nneg


def _grid_for(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, GRID_N)


def _curve_block(y: np.ndarray, s: np.ndarray) -> dict:
    grid = _grid_for(s)
    fpr, tpr, npos, nneg = _roc_on_grid(y, s, grid)
    auc = bc.auroc(y, s)
    return {
        "auroc": round(float(auc), 4),
        "n_pos": int(npos),
        "n_neg": int(nneg),
        "thresholds": [round(float(t), 6) for t in grid],
        "fpr": [round(float(v), 6) for v in fpr],
        "tpr": [round(float(v), 6) for v in tpr],
        "fpr_ci_hw": [round(float(v), 6) for v in _wilson_halfwidth(fpr, nneg)],
        "tpr_ci_hw": [round(float(v), 6) for v in _wilson_halfwidth(tpr, npos)],
        "fpr_resolution": round(1.0 / nneg, 8) if nneg else None,
    }


def _both_curve(y2, s2, y3, s3) -> dict:
    """AND-rule ROC parameterised by a single shared operating quantile q.

    The AF2 and AF3 rows are aligned pairwise (same pair order in both backends).
    At each q, threshold each version at its own q-quantile and call a pair
    positive iff both clear it. Stored thresholds are (t_af2, t_af3) tuples.
    """
    assert np.array_equal(y2, y3), "af2/af3 label vectors must be pair-aligned"
    y = y2
    pos = y == 1
    neg = y == 0
    npos, nneg = int(pos.sum()), int(neg.sum())
    qs = np.linspace(0.0, 1.0, GRID_N)
    s2f = np.where(np.isfinite(s2), s2, -np.inf)
    s3f = np.where(np.isfinite(s3), s3, -np.inf)
    t2 = np.quantile(s2f[np.isfinite(s2)], qs) if np.isfinite(s2).any() else np.zeros_like(qs)
    t3 = np.quantile(s3f[np.isfinite(s3)], qs) if np.isfinite(s3).any() else np.zeros_like(qs)
    fpr, tpr = [], []
    for a, b in zip(t2, t3):
        called = (s2f >= a) & (s3f >= b)
        tpr.append(called[pos].mean() if npos else np.nan)
        fpr.append(called[neg].mean() if nneg else np.nan)
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)
    # AUROC of the AND-rule scored by the min of the two oriented values
    joint = np.minimum(s2f, s3f)
    auc = bc.auroc(y, joint)
    return {
        "auroc": round(float(auc), 4),
        "n_pos": npos,
        "n_neg": nneg,
        "quantiles": [round(float(q), 6) for q in qs],
        "thresholds_af2": [round(float(a), 6) for a in t2],
        "thresholds_af3": [round(float(b), 6) for b in t3],
        "fpr": [round(float(v), 6) for v in fpr],
        "tpr": [round(float(v), 6) for v in tpr],
        "fpr_ci_hw": [round(float(v), 6) for v in _wilson_halfwidth(fpr, nneg)],
        "tpr_ci_hw": [round(float(v), 6) for v in _wilson_halfwidth(tpr, npos)],
        "fpr_resolution": round(1.0 / nneg, 8) if nneg else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-csv", default=DEFAULT_CSV)
    ap.add_argument("--scores", nargs="+", default=list(DEFAULT_SCORES))
    ap.add_argument("--out", default="src/alphajudge/threshold_rocs.json")
    args = ap.parse_args()

    rows = bc.read_rows(args.input_csv)
    if not rows:
        raise SystemExit(f"no rows in {args.input_csv}")

    def split(backend):
        rs = [r for r in rows if str(r.get("model", "")).strip() == backend]
        return rs, bc.labels(rs)

    af2_rows, y2 = split("af2")
    af3_rows, y3 = split("af3")
    if not af2_rows or not af3_rows:
        raise SystemExit("benchmark must contain both af2 and af3 rows (column 'model')")

    # pair-align af2/af3 by pair id so the 'both' AND-rule is well defined
    def by_pair(rs, y):
        return {bc.pair_id(r): (r, int(yi)) for r, yi in zip(rs, y)}
    m2, m3 = by_pair(af2_rows, y2), by_pair(af3_rows, y3)
    shared = sorted(set(m2) & set(m3))
    print(f"af2 rows={len(af2_rows)}  af3 rows={len(af3_rows)}  shared pairs={len(shared)}")

    curves: dict[str, dict] = {}
    for score in args.scores:
        if score not in af2_rows[0]:
            print(f"[skip] {score}: not a column")
            continue
        block = {}
        # single-version curves
        block["af2"] = _curve_block(y2, bc.oriented_values(af2_rows, score, y2))
        block["af3"] = _curve_block(y3, bc.oriented_values(af3_rows, score, y3))
        # both: aligned on shared pairs
        ys = np.array([m2[p][1] for p in shared])
        a2_rows = [m2[p][0] for p in shared]
        a3_rows = [m3[p][0] for p in shared]
        sa2 = bc.oriented_values(a2_rows, score, ys)
        sa3 = bc.oriented_values(a3_rows, score, ys)
        block["both"] = _both_curve(ys, sa2, ys, sa3)
        curves[score] = block
        print(f"  {score:20s} AUROC af2={block['af2']['auroc']:.3f} "
              f"af3={block['af3']['auroc']:.3f} both={block['both']['auroc']:.3f}")

    artifact = {
        "schema_version": 1,
        "source_csv": Path(args.input_csv).name,
        "source_sha16": _sha16(args.input_csv),
        "orientation": "higher_is_stronger",
        "grid_n": GRID_N,
        "curves": curves,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=1))
    print(f"\nWrote {out} ({len(curves)} scores x 3 versions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
