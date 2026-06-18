"""Tests for the threshold calculator (alphajudge.thresholds).

These exercise the lookup logic against a small synthetic ROC artifact so they
run without the benchmark CSV, plus the precision-projection maths and the
tail-extrapolation guard.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from alphajudge import thresholds as T


@pytest.fixture
def artifact(tmp_path) -> str:
    """A tiny hand-built ROC: a clean separating score on the af2 curve.

    thresholds 0.0 .. 1.0 (11 points); fpr decreases, tpr decreases as the
    cutoff rises. fpr_resolution chosen so 0.05 is resolvable but 0.001 is not.
    """
    grid = [i / 10 for i in range(11)]
    # at threshold t: tpr = 1 - 0.5*t (stays high), fpr = (1-t)^2 (drops fast)
    tpr = [round(1 - 0.5 * t, 4) for t in grid]
    fpr = [round((1 - t) ** 2, 4) for t in grid]
    zeros = [0.0] * 11
    curve = {
        "auroc": 0.9, "n_pos": 100, "n_neg": 100,
        "thresholds": grid, "fpr": fpr, "tpr": tpr,
        "fpr_ci_hw": [0.02] * 11, "tpr_ci_hw": [0.02] * 11,
        "fpr_resolution": 0.01,
    }
    art = {
        "schema_version": 1, "source_csv": "synthetic", "source_sha16": "0",
        "orientation": "higher_is_stronger", "grid_n": 11,
        "curves": {"interface_ipSAE": {"af2": curve, "af3": curve, "both": _both(grid, fpr, tpr)}},
    }
    p = tmp_path / "rocs.json"
    p.write_text(json.dumps(art))
    T.load_rocs.cache_clear()
    return str(p)


def _both(grid, fpr, tpr):
    return {
        "auroc": 0.9, "n_pos": 100, "n_neg": 100,
        "quantiles": grid, "thresholds_af2": grid, "thresholds_af3": grid,
        "fpr": fpr, "tpr": tpr, "fpr_ci_hw": [0.02] * 11, "tpr_ci_hw": [0.02] * 11,
        "fpr_resolution": 0.01,
    }


def test_project_precision_matches_formula():
    # pi=0.05, tpr=0.8, fpr=0.1 -> 0.05*0.8 / (0.05*0.8 + 0.95*0.1)
    got = T.project_precision(0.05, 0.8, 0.1)
    assert got == pytest.approx(0.04 / (0.04 + 0.095))


def test_project_precision_no_calls_is_nan():
    assert math.isnan(T.project_precision(0.05, 0.0, 0.0))


def test_threshold_for_fpr_picks_within_tolerance(artifact):
    op = T.threshold_for_fpr("ipSAE", "af2", prevalence=0.05, max_fpr=0.05, path=artifact)
    assert op.resolved
    assert op.fpr <= 0.05 + 1e-9
    # most permissive point under the tolerance => highest available recall there
    assert 0 <= op.recall <= 1
    assert op.precision_ci[0] <= op.precision <= op.precision_ci[1]


def test_tail_guard_flags_unresolved(artifact):
    op = T.threshold_for_fpr("ipSAE", "af2", prevalence=0.001, max_fpr=0.0001, path=artifact)
    assert op.resolved is False
    assert "below the benchmark resolution" in op.note


def test_lower_prevalence_lowers_precision(artifact):
    hi = T.threshold_for_fpr("ipSAE", "af2", prevalence=0.5, max_fpr=0.05, path=artifact)
    lo = T.threshold_for_fpr("ipSAE", "af2", prevalence=0.01, max_fpr=0.05, path=artifact)
    assert lo.precision < hi.precision


def test_both_returns_threshold_pair(artifact):
    op = T.threshold_for_fpr("ipSAE", "both", prevalence=0.05, max_fpr=0.05, path=artifact)
    assert isinstance(op.threshold, tuple) and len(op.threshold) == 2


def test_precision_for_recall_floor(artifact):
    op = T.threshold_for_recall("ipSAE", "af2", prevalence=0.05, min_recall=0.6, path=artifact)
    assert op.recall >= 0.6 - 1e-9


def test_precision_vs_prevalence_monotone(artifact):
    pts = T.precision_vs_prevalence("ipSAE", "af2", threshold=0.5,
                                    prevalences=[0.5, 0.05, 0.005], path=artifact)
    precs = [p for _, p in pts]
    assert precs[0] > precs[1] > precs[2]


def test_unknown_score_raises(artifact):
    with pytest.raises(KeyError):
        T.threshold_for_fpr("nope", "af2", 0.05, 0.05, path=artifact)


def test_alias_resolution(artifact):
    # "ipSAE" alias resolves to interface_ipSAE without error
    op = T.threshold_for_fpr("ipSAE", "af2", 0.05, 0.05, path=artifact)
    assert op.score == "interface_ipSAE"
