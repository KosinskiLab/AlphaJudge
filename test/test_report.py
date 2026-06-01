from __future__ import annotations

import csv
from pathlib import Path

import pytest

from alphajudge.report import (
    generate_aggregate_report,
    generate_per_run_report,
)


_BASE_ROW = {
    "jobs": "PROT_A_PROT_B",
    "model_used": "model_1_multimer_v3_pred_0",
    "interface": "A_B",
    "iptm_ptm": "0.55",
    "iptm": "0.55",
    "ptm": "0.60",
    "confidence_score": "0.62",
    "pDockQ/mpDockQ": "0.40",
    "average_interface_pae": "10.0",
    "interface_average_plddt": "78.5",
    "interface_num_intf_residues": "42",
    "interface_polar": "11",
    "interface_hydrophobic": "14",
    "interface_charged": "9",
    "interface_contact_pairs": "82",
    "interface_score": "0.41",
    "interface_pDockQ2": "0.06",
    "interface_ipSAE": "0.45",
    "interface_LIS": "0.30",
    "interface_hb": "5",
    "interface_sb": "2",
    "interface_ss": "0",
    "interface_sc": "0.50",
    "interface_zernike_sc": "0.40",
    "interface_area": "2300.0",
    "interface_solv_en": "-32.0",
    "interface_meta_score": "0.55",
}


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _pdf_page_count(path: Path) -> int:
    """Count '/Type /Page' (not '/Pages') occurrences in a small PDF."""
    data = path.read_bytes()
    count = 0
    idx = 0
    while True:
        i = data.find(b"/Type /Page", idx)
        if i < 0:
            break
        # Skip if this is actually '/Type /Pages'
        if data[i + len(b"/Type /Page") : i + len(b"/Type /Page") + 1] == b"s":
            idx = i + 1
            continue
        count += 1
        idx = i + 1
    return count


def test_per_run_report_produces_a_pdf(tmp_path: Path) -> None:
    rows = [
        dict(_BASE_ROW),
        {**_BASE_ROW, "model_used": "model_2_multimer_v3_pred_0",
         "interface_meta_score": "0.40", "interface_LIS": "0.20"},
    ]
    _write_csv(tmp_path / "interfaces.csv", rows)
    out = generate_per_run_report(tmp_path)
    assert out is not None
    assert out.exists()
    assert out.stat().st_size > 0
    assert _pdf_page_count(out) >= 2  # cover + per-interface table at minimum


def test_per_run_report_returns_none_on_missing_csv(tmp_path: Path) -> None:
    assert generate_per_run_report(tmp_path) is None


def test_aggregate_report_writes_cover_plus_one_page_per_interface(tmp_path: Path) -> None:
    rows = [
        dict(_BASE_ROW),
        {**_BASE_ROW, "jobs": "PROT_C_PROT_D",
         "interface_meta_score": "0.80", "interface_LIS": "0.65"},
        {**_BASE_ROW, "jobs": "PROT_C_PROT_D", "interface": "A_C",
         "interface_meta_score": "0.40"},
    ]
    summary = tmp_path / "summary.csv"
    _write_csv(summary, rows)
    out = tmp_path / "aggregate.pdf"
    result = generate_aggregate_report(summary, out_pdf=out)
    assert result == out
    assert out.exists() and out.stat().st_size > 0
    # cover + one page per scorable interface (3) + one complex-evidence
    # page per unique complex (2 unique complexes in this fixture).
    assert _pdf_page_count(out) == 6


def test_aggregate_report_handles_missing_meta_score_via_recompute(tmp_path: Path) -> None:
    rows = []
    base = dict(_BASE_ROW)
    base.pop("interface_meta_score")
    rows.append(base)
    base2 = dict(_BASE_ROW)
    base2["jobs"] = "PROT_E_PROT_F"
    base2.pop("interface_meta_score")
    rows.append(base2)
    summary = tmp_path / "summary.csv"
    _write_csv(summary, rows)
    out = tmp_path / "agg.pdf"
    result = generate_aggregate_report(summary, out_pdf=out)
    assert result is not None
    assert out.exists()
    # cover + 2 interface pages + 2 complex-evidence pages
    assert _pdf_page_count(out) == 5
