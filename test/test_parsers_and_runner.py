from __future__ import annotations
import csv
import json
from pathlib import Path
import math
import numpy as np
import pytest

from alphajudge.parsers import pick_parser
from alphajudge.runner import process, process_many


@pytest.fixture(scope="module")
def af2_dir() -> Path:
    return Path("test_data/af2/pos_dimers/Q13148+Q92900")


@pytest.fixture(scope="module")
def af3_dir() -> Path:
    return Path("test_data/af3/pos_dimers/Q13148+Q92900")


def read_csv_rows(path: Path):
    with path.open() as f:
        reader = csv.DictReader(f)
        return list(reader)


def nearly_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    if (a is None) or (b is None):
        return a is None and b is None
    if isinstance(a, str):
        try:
            a = float(a)
        except Exception:
            return False
    if isinstance(b, str):
        try:
            b = float(b)
        except Exception:
            return False
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(float(a) - float(b)) <= tol


@pytest.mark.parametrize("models_to_analyse", ["best", "all"])
def test_af2_runner_outputs_have_expected_scores(af2_dir: Path, models_to_analyse: str):
    parser = pick_parser(af2_dir)
    assert parser.name == "af2"
    process(str(af2_dir), 8.0, 100.0, models_to_analyse)

    # PAE heatmaps should be saved as pae_<model_name>.png next to interfaces.csv
    run = parser.parse_run(af2_dir)
    models = [run.order[0]] if models_to_analyse == "best" else run.order
    for m in models:
        png = af2_dir / f"pae_{m}.png"
        assert png.exists() and png.stat().st_size > 0

    out = af2_dir / "interfaces.csv"
    assert out.exists() and out.stat().st_size > 0
    rows = read_csv_rows(out)
    assert rows, "interfaces.csv must contain at least one row"

    # Verify header includes ptm and confidence_score for AF2
    header = list(rows[0].keys())
    assert "ptm" in header
    assert "confidence_score" in header

    # Cross-check against ranking_debug.json: iptm+ptm and iptm present; ptm can be derived
    ranking = json.load((af2_dir / "ranking_debug.json").open())
    order = ranking["order"]
    iptm_map = ranking.get("iptm", {})
    conf_map = ranking.get("iptm+ptm", {})

    # Pick the first row (best model if best) and verify values
    r0 = rows[0]
    model_used = r0["model_used"]
    assert model_used in order
    exp_iptm = float(iptm_map[model_used])
    exp_conf = float(conf_map[model_used])
    exp_ptm = (exp_conf - 0.8 * exp_iptm) / 0.2

    assert nearly_equal(r0["iptm"], exp_iptm)
    assert nearly_equal(r0["iptm_ptm"], exp_conf)
    assert nearly_equal(r0["ptm"], exp_ptm)
    assert nearly_equal(r0["confidence_score"], exp_conf)


@pytest.mark.parametrize("models_to_analyse", ["best", "all"])
def test_af3_runner_outputs_have_expected_scores(af3_dir: Path, models_to_analyse: str):
    parser = pick_parser(af3_dir)
    assert parser.name == "af3"
    process(str(af3_dir), 8.0, 100.0, models_to_analyse)

    # PAE heatmaps should be saved as pae_<model_name>.png next to interfaces.csv
    run = parser.parse_run(af3_dir)
    models = [run.order[0]] if models_to_analyse == "best" else run.order
    for m in models:
        png = af3_dir / f"pae_{m}.png"
        assert png.exists() and png.stat().st_size > 0

    out = af3_dir / "interfaces.csv"
    assert out.exists() and out.stat().st_size > 0
    rows = read_csv_rows(out)
    assert rows, "interfaces.csv must contain at least one row"

    # Header contains AF3 fields too
    header = list(rows[0].keys())
    for col in ("ptm", "confidence_score", "iptm", "iptm_ptm"):
        assert col in header

    # Cross-check first row scores against ranked_0_summary_confidences.json or per-model summary
    # Determine the model directory from model_used
    r0 = rows[0]
    model_used = r0["model_used"]
    model_dir = af3_dir / model_used

    # Prefer per-model summary; fallback to top-level summary
    if (model_dir / "summary_confidences.json").exists():
        summary = json.load((model_dir / "summary_confidences.json").open())
    else:
        summary = json.load((af3_dir / "ranked_0_summary_confidences.json").open())

    got_iptm = float(r0["iptm"]) if r0["iptm"] != "nan" else np.nan
    got_ptm = float(r0["ptm"]) if r0["ptm"] != "nan" else np.nan
    got_conf = float(r0["confidence_score"]) if r0["confidence_score"] != "nan" else np.nan
    got_ipptm = float(r0["iptm_ptm"]) if r0["iptm_ptm"] != "nan" else np.nan

    exp_iptm = summary.get("iptm")
    exp_ptm = summary.get("ptm")
    exp_rank = summary.get("ranking_score") or summary.get("iptm+ptm")
    exp_conf = summary.get("confidence_score")

    # If AF3 omitted ptm but provided iptm+ptm, recompute ptm to compare
    if exp_ptm is None and (exp_rank is not None) and (exp_iptm is not None):
        exp_ptm = (float(exp_rank) - 0.8 * float(exp_iptm)) / 0.2
    if exp_conf is None and (exp_iptm is not None) and (exp_ptm is not None):
        exp_conf = 0.8 * float(exp_iptm) + 0.2 * float(exp_ptm)

    if exp_iptm is not None:
        assert nearly_equal(got_iptm, float(exp_iptm))
    if exp_ptm is not None:
        assert nearly_equal(got_ptm, float(exp_ptm))
    if exp_rank is not None:
        assert nearly_equal(got_ipptm, float(exp_rank))
    if exp_conf is not None:
        assert nearly_equal(got_conf, float(exp_conf))


def test_headers_consistent_between_af2_af3(af2_dir: Path, af3_dir: Path):
    process(str(af2_dir), 8.0, 100.0, "best")
    process(str(af3_dir), 8.0, 100.0, "best")
    h2 = list(read_csv_rows(af2_dir / "interfaces.csv")[0].keys())
    h3 = list(read_csv_rows(af3_dir / "interfaces.csv")[0].keys())
    # Ensure both contain the union of core score fields
    for col in ("iptm_ptm", "iptm", "ptm", "confidence_score"):
        assert col in h2
        assert col in h3


def test_process_many_aggregates_rows(af2_dir: Path, af3_dir: Path, tmp_path: Path):
    # Aggregate two explicit run directories
    summary = tmp_path / "summary.csv"
    got = process_many(
        [str(af2_dir), str(af3_dir)],
        contact_thresh=8.0,
        pae_filter=100.0,
        models_to_analyse="best",
        recursive=False,
        summary_csv=str(summary),
    )
    assert got is not None and summary.exists() and summary.stat().st_size > 0
    rows = read_csv_rows(summary)
    assert rows, "summary.csv must contain at least one row"
    header = list(rows[0].keys())
    for col in ("iptm_ptm", "iptm", "ptm", "confidence_score", "jobs", "model_used"):
        assert col in header
    # The summary row count should equal the sum of the individual per-run rows
    af2_rows = read_csv_rows(af2_dir / "interfaces.csv")
    af3_rows = read_csv_rows(af3_dir / "interfaces.csv")
    assert len(rows) == len(af2_rows) + len(af3_rows)


def test_process_many_recursive_discovers_runs(af2_dir: Path, af3_dir: Path, tmp_path: Path):
    # Recurse from the parent folders; should find at least the two known runs
    summary = tmp_path / "recursive_summary.csv"
    got = process_many(
        [str(af2_dir.parent), str(af3_dir.parent)],
        contact_thresh=8.0,
        pae_filter=100.0,
        models_to_analyse="best",
        recursive=True,
        summary_csv=str(summary),
    )
    assert got is not None and summary.exists() and summary.stat().st_size > 0
    rows = read_csv_rows(summary)
    assert rows, "recursive summary must contain rows"
    # At minimum, recursive should include the rows from both explicit runs
    af2_rows = read_csv_rows(af2_dir / "interfaces.csv")
    af3_rows = read_csv_rows(af3_dir / "interfaces.csv")
    assert len(rows) >= len(af2_rows) + len(af3_rows)

