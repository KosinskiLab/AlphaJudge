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


def test_slider_rows_show_ccc_after_full_benchmark_calibration():
    """
    CCC is shown now that its positive-reference deciles are frozen. Features
    without a ladder are still omitted rather than breaking the report.
    """
    from alphajudge import report as rep

    row = {
        "interface_ipSAE": 0.6,
        "interface_ccc": 41,
        "interface_contact_prob_source": "af2_distogram_le_8A",
    }

    rows = rep._metric_rows_for_slider_panel(
        row, include_overall=False, groups=(("af", ("interface_ipSAE", "interface_ccc")),))
    labels = [r[0] for r in rows]
    assert "Interface ipSAE" in labels
    assert "Confident contacts" in labels

    ccc_row = next(r for r in rows if r[0] == "Confident contacts")
    # The AF2 ladder is selected from the row provenance: 41 lies between its
    # median (25) and 60th percentile (42), not on the pooled ladder.
    assert ccc_row[2] == pytest.approx(0.5941176471)

    # An unknown feature must not raise either.
    rep._metric_rows_for_slider_panel(
        row, include_overall=False, groups=(("af", ("not_a_real_feature",)),))


@pytest.fixture
def drawn_text(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every string the report draws, in order."""
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    seen: list[str] = []
    for cls in (Axes, Figure):
        def record(self, x, y, s, *args, _original=cls.text, **kwargs):
            seen.append(str(s))
            return _original(self, x, y, s, *args, **kwargs)

        monkeypatch.setattr(cls, "text", record)
    return seen


def test_find_pae_png_tolerates_empty_and_glob_special_model_names(tmp_path: Path) -> None:
    from alphajudge.report import _find_pae_png

    (tmp_path / "other_m1.png").write_bytes(b"")
    # "[1]" is a character class to glob; the model name must match literally.
    assert _find_pae_png(tmp_path, "m[1]") is None
    (tmp_path / "pae_m[1].png").write_bytes(b"")
    assert _find_pae_png(tmp_path, "m[1]") == tmp_path / "pae_m[1].png"

    # An empty model name used to raise ValueError ("**" glob) on Python < 3.13.
    assert _find_pae_png(tmp_path, "") is None
    fallback = tmp_path / "job_PAE_plot_ranked_0.png"
    fallback.write_bytes(b"")
    assert _find_pae_png(tmp_path, "") == fallback


def test_per_run_report_text_counts_features_and_numbers_appendices(
    tmp_path: Path, drawn_text: list[str]
) -> None:
    from alphajudge.meta_score import META_SCORE_FEATURES

    rows = [
        dict(_BASE_ROW),
        {**_BASE_ROW, "interface": "A_C", "interface_LIS": "0.10"},
        {**_BASE_ROW, "model_used": "model_2_multimer_v3_pred_0", "interface_LIS": "0.05",
         "iptm": "0.20"},
    ]
    _write_csv(tmp_path / "interfaces.csv", rows)
    assert generate_per_run_report(tmp_path) is not None

    assert any(f"up to {len(META_SCORE_FEATURES)} available feature percentiles" in t for t in drawn_text)
    assert "A.1" in drawn_text
    assert "Appendix – model model_2_multimer_v3_pred_0" in drawn_text


def test_aggregate_cover_median_averages_the_middle_pair(tmp_path: Path, drawn_text: list[str]) -> None:
    from alphajudge import report as rep

    rows = [
        {**_BASE_ROW, "jobs": f"JOB_{k}", "interface_LIS": lis}
        for k, lis in enumerate(("0.05", "0.30", "0.55", "0.75"))
    ]
    summary = tmp_path / "summary.csv"
    _write_csv(summary, rows)
    assert generate_aggregate_report(summary, out_pdf=tmp_path / "agg.pdf") is not None

    scores = sorted(rep._row_meta_score(r) for r in rows)
    median = (scores[1] + scores[2]) / 2
    assert f"median   = {median:.3f}" in drawn_text


@pytest.fixture
def captured_pages(monkeypatch):
    """Capture actual PDF page artists and their physical positions before saving."""
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.text import Text

    pages = []
    original = PdfPages.savefig

    def record(pdf, figure, **kwargs):
        texts = []
        for artist in figure.findobj(Text):
            point = artist.get_transform().transform(artist.get_position())
            x, y = figure.transFigure.inverted().transform(point)
            texts.append((artist.get_text(), x, y))
        pages.append(texts)
        return original(pdf, figure, **kwargs)

    monkeypatch.setattr(PdfPages, "savefig", record)
    return pages


def test_interface_table_paginates_without_losing_rows(tmp_path, captured_pages):
    from alphajudge import report as rep
    from matplotlib.backends.backend_pdf import PdfPages

    rows = [{**_BASE_ROW, "model_used": f"R{i:02}"} for i in range(36)]
    with PdfPages(tmp_path / "table.pdf") as pdf:
        rep._per_interface_page(pdf, entry_id="test", section_no="1", rows=rows, page_no=2)
    assert len(captured_pages) == 2
    assert all(any(text == "Model" for text, _, _ in page) for page in captured_pages)
    table_rows = [(text, y) for page in captured_pages for text, _, y in page if text.startswith("R") and text[1:].isdigit()]
    assert sorted(text for text, _ in table_rows) == [f"R{i:02}" for i in range(36)]
    assert all(.05 < y < .85 for _, y in table_rows)
    assert [text for page in captured_pages for text, _, _ in page if text.startswith("Page ")] == ["Page 2", "Page 3"]


def test_large_aggregate_ranking_paginates(tmp_path, captured_pages):
    rows = [{**_BASE_ROW, "jobs": f"JOB_{i:02}"} for i in range(42)]
    summary = tmp_path / "summary.csv"
    _write_csv(summary, rows)
    generate_aggregate_report(summary, out_pdf=tmp_path / "aggregate.pdf", top_n=42, max_complexes=1)
    # Cover with ten entries, two continuation tables, one interface, one complex.
    assert len(captured_pages) == 5
    table_entries = [(text, y) for page in captured_pages[:3] for text, _, y in page if text.startswith("JOB_")]
    assert len(table_entries) == 42
    assert all(.05 < y < .9 for _, y in table_entries)
    assert [text for page in captured_pages for text, _, _ in page if text.startswith("Page ")] == [f"Page {i}" for i in range(2, 6)]


def test_import_preserves_matplotlib_backend_and_figures():
    import subprocess
    import sys

    code = """
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt
fig = plt.figure()
import alphajudge.report
assert matplotlib.get_backend().lower() == 'svg'
assert plt.fignum_exists(fig.number)
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("operation", ["png", "per_run", "aggregate", "error"])
def test_render_restores_matplotlib_settings(tmp_path, operation, monkeypatch):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from alphajudge import report as rep

    _write_csv(tmp_path / "interfaces.csv", [dict(_BASE_ROW)])
    with matplotlib.rc_context({"font.family": ["sans-serif"], "font.size": 17, "pdf.fonttype": 3}):
        figure = plt.figure()
        before = dict(matplotlib.rcParams)
        try:
            if operation == "png":
                rep.render_pae_png(tmp_path / "pae.png", [[1., 2.], [3., 4.]])
            elif operation == "per_run":
                rep.generate_per_run_report(tmp_path)
            elif operation == "aggregate":
                rep.generate_aggregate_report(tmp_path / "interfaces.csv", out_pdf=tmp_path / "a.pdf")
            else:
                def fail(*args, **kwargs):
                    raise OSError("write failed")
                monkeypatch.setattr(Figure, "savefig", fail)
                with pytest.raises(OSError, match="write failed"):
                    rep.render_pae_png(tmp_path / "pae.png", [[1.]])
            assert dict(matplotlib.rcParams) == before
            assert plt.get_fignums() == [figure.number]
        finally:
            plt.close(figure)


def test_report_explains_recalibration_and_custom_cutoffs(tmp_path, drawn_text):
    row = {**_BASE_ROW, "contact_thresh": "12", "pae_filter": "100", "ipsae_pae_cutoff": "10",
           "metascore_calibration": "historical-fit", "interface_meta_score": "0.99"}
    _write_csv(tmp_path / "interfaces.csv", [row])
    generate_per_run_report(tmp_path)
    from alphajudge.meta_score import CALIBRATION_ID
    text = "\n".join(drawn_text)
    assert CALIBRATION_ID in text
    assert "recomputed" in text
    assert "10/11" in text
    assert "CSV score: 0.99" in text
    assert "historical-fit" in text
    assert "Custom cutoffs" in text
    assert "exploratory" in text


def test_legacy_report_marks_unknown_parameters_and_stored_score(tmp_path, drawn_text):
    _write_csv(tmp_path / "interfaces.csv", [{"jobs": "old", "interface": "A_B", "interface_meta_score": "0.6"}])
    generate_per_run_report(tmp_path)
    text = "\n".join(drawn_text)
    assert "Stored CSV score" in text
    assert "cutoffs unknown" in text
