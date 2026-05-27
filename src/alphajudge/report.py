"""RCSB-style validation reports for AlphaJudge interface scores.

The visual layout mirrors the wwPDB / RCSB "Full Validation Report" PDF
(see e.g. ``https://files.rcsb.org/validation/view/<id>_full_validation.pdf``):

* serif typography (DejaVu Serif, the available Computer-Modern lookalike);
* a smooth red -> yellow -> green percentile slider with a single black
  marker for the entry's archive percentile;
* a numbered "Overall quality at a glance" page with a metric/value table;
* a page header rule with title + entry id, and a thin bottom rule
  with the page number.

Two entry points:

* :func:`generate_per_run_report` -- one ``report.pdf`` per run directory.
* :func:`generate_aggregate_report` -- a multi-page PDF over a merged CSV.
"""

from __future__ import annotations

import csv
import logging
import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle

from .meta_score import (
    BENCHMARK_QUANTILES,
    CALIBRATION_LEVELS,
    FEATURE_DIRECTIONS,
    META_SCORE_FEATURES,
    calibrated_feature_percentile,
    interface_meta_score,
)

logger = logging.getLogger(__name__)

_A4 = (8.27, 11.69)
_SLIDER_CMAP = "RdYlGn"
_INFO_BG = "#fbe5e5"
_INFO_EDGE = "#a83232"
_HEADER_RULE = "#3a3a3a"
_TABLE_RULE = "#404040"

_REPORT_TITLE = "AlphaJudge Interface Validation Report"
_BENCHMARK_TAG = "benchmark_26 (final_sync_20260523, n=7,756 AF2/AF3 rows)"

_FEATURE_DISPLAY = {
    "interface_LIS": "Interface LIS",
    "interface_ipSAE": "Interface ipSAE",
    "interface_pDockQ2": "Interface pDockQ2",
    "iptm": "ipTM",
    "confidence_score": "Confidence score",
    "average_interface_pae": "Avg. interface PAE",
    "pDockQ/mpDockQ": "pDockQ / mpDockQ",
    "interface_sc": "Shape complementarity",
    "interface_area": "Interface area",
    "interface_solv_en": "Solvation energy",
}

_FEATURE_UNITS = {
    "average_interface_pae": "Å",
    "interface_area": "Å²",
    "interface_solv_en": "kcal/mol",
}


# ---------------------------------------------------------------------------
# style + utility helpers
# ---------------------------------------------------------------------------

def _setup_rcparams() -> None:
    rcparams = {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.edgecolor": "#222222",
        "savefig.dpi": 200,
    }
    matplotlib.rcParams.update(rcparams)


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _row_meta_score(row: Mapping[str, Any]) -> float | None:
    direct = _safe_float(row.get("interface_meta_score"))
    if direct is not None:
        return direct
    computed = interface_meta_score(row)
    if isinstance(computed, float) and math.isfinite(computed):
        return computed
    return None


def _feature_view(row: Mapping[str, Any]) -> "OrderedDict[str, tuple[float | None, float | None]]":
    view: "OrderedDict[str, tuple[float | None, float | None]]" = OrderedDict()
    for feat in META_SCORE_FEATURES:
        raw = _safe_float(row.get(feat))
        pct = calibrated_feature_percentile(feat, raw) if raw is not None else None
        view[feat] = (raw, pct)
    return view


def _best_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    best: tuple[float, Mapping[str, Any]] | None = None
    for r in rows:
        s = _row_meta_score(r)
        if s is None:
            continue
        if best is None or s > best[0]:
            best = (s, r)
    if best is not None:
        return best[1]
    return rows[0] if rows else None


def _group_complex_rows(rows: Sequence[Mapping[str, Any]]) -> "OrderedDict[str, list[Mapping[str, Any]]]":
    grouped: "OrderedDict[str, list[Mapping[str, Any]]]" = OrderedDict()
    for r in rows:
        key = str(r.get("jobs") or r.get("pair") or r.get("complex") or "")
        if not key:
            continue
        grouped.setdefault(key, []).append(r)
    return grouped


def _format_raw(value: float | None, *, decimals: int = 3) -> str:
    if value is None:
        return "—"
    av = abs(value)
    if av != 0.0 and (av >= 10000 or av < 0.001):
        return f"{value:.2e}"
    if av >= 100:
        return f"{value:.1f}"
    return f"{value:.{decimals}g}"


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _shorten_path(path: str, max_len: int = 64) -> str:
    if len(path) <= max_len:
        return path
    head = path[: max_len // 2 - 1]
    tail = path[-(max_len // 2):]
    return f"{head}…{tail}"


def _detect_backend(rows: Sequence[Mapping[str, Any]]) -> str:
    for r in rows:
        model = str(r.get("model_used") or "")
        if "multimer" in model.lower():
            return "AlphaFold 2"
        if model.startswith("seed-") or "_sample-" in model:
            return "AlphaFold 3"
        if "boltz" in model.lower():
            return "Boltz-2"
    return "unknown"


def _detect_chain_set(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    chains: set[str] = set()
    for r in rows:
        iface = str(r.get("interface") or "")
        for part in iface.split("_"):
            if part:
                chains.add(part)
    return chains


def _decile_label(pct: float | None) -> str:
    if pct is None:
        return "n/a"
    if pct >= 0.9:
        return "Top decile"
    if pct >= 0.75:
        return "Upper quartile"
    if pct >= 0.5:
        return "Above median"
    if pct >= 0.25:
        return "Below median"
    if pct >= 0.1:
        return "Lower quartile"
    return "Bottom decile"


# ---------------------------------------------------------------------------
# page primitives
# ---------------------------------------------------------------------------

def _new_figure() -> plt.Figure:
    return plt.figure(figsize=_A4, facecolor="white")


def _add_page_header(fig: plt.Figure, *, page_no: int, total: int, title: str, entry: str) -> None:
    ax = fig.add_axes((0.07, 0.965, 0.86, 0.018))
    ax.axis("off")
    ax.text(0.0, 0.5, title, fontsize=8, color="#333", va="center", transform=ax.transAxes)
    ax.text(1.0, 0.5, entry, fontsize=8, color="#333", va="center", ha="right", transform=ax.transAxes)
    rule = fig.add_axes((0.07, 0.957, 0.86, 0.003))
    rule.axis("off")
    rule.axhline(0.5, color=_HEADER_RULE, linewidth=0.8)


def _add_page_footer(fig: plt.Figure, *, page_no: int, total: int, last: bool) -> None:
    rule = fig.add_axes((0.07, 0.038, 0.86, 0.003))
    rule.axis("off")
    rule.axhline(0.5, color=_HEADER_RULE, linewidth=0.6)
    ax = fig.add_axes((0.07, 0.018, 0.86, 0.018))
    ax.axis("off")
    ax.text(
        1.0,
        0.5,
        f"{page_no} / {total}",
        ha="right",
        va="center",
        fontsize=8,
        color="#555",
        transform=ax.transAxes,
    )


def _draw_info_box(fig: plt.Figure, *, x: float, y: float, w: float, h: float, lines: Sequence[str]) -> None:
    ax = fig.add_axes((x, y, w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box = FancyBboxPatch(
        (0.005, 0.05),
        0.99,
        0.90,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=0.6,
        edgecolor=_INFO_EDGE,
        facecolor=_INFO_BG,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    n = len(lines)
    if n == 0:
        return
    top = 0.83
    line_h = 0.7 / max(1, n)
    for i, line in enumerate(lines):
        ax.text(
            0.5,
            top - i * line_h,
            line,
            ha="center",
            va="top",
            fontsize=9.5,
            color="#1d1d1d",
            transform=ax.transAxes,
        )


def _draw_meta_block(fig: plt.Figure, *, x: float, y: float, w: float, h: float, pairs: Sequence[tuple[str, str]]) -> None:
    """Right-aligned label, colon, then value (RCSB style)."""
    ax = fig.add_axes((x, y, w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    n = len(pairs)
    if n == 0:
        return
    top = 0.92
    line_h = 0.85 / max(1, n)
    label_x = 0.36
    sep_x = 0.40
    val_x = 0.44
    for i, (label, value) in enumerate(pairs):
        ypos = top - i * line_h
        ax.text(label_x, ypos, label, fontsize=10.5, ha="right", va="top", transform=ax.transAxes)
        ax.text(sep_x, ypos, ":", fontsize=10.5, ha="center", va="top", transform=ax.transAxes)
        ax.text(val_x, ypos, value, fontsize=10.5, ha="left", va="top", transform=ax.transAxes)


def _draw_section_heading(fig: plt.Figure, *, x: float, y: float, w: float, h: float, number: str, title: str) -> None:
    ax = fig.add_axes((x, y, w, h))
    ax.axis("off")
    ax.text(
        0.0,
        0.4,
        f"{number}  {title}",
        fontsize=15,
        fontweight="bold",
        color="#101010",
        transform=ax.transAxes,
    )


# ---------------------------------------------------------------------------
# slider primitive
# ---------------------------------------------------------------------------

_GRADIENT = np.tile(np.linspace(0.02, 0.98, 512), (4, 1))


def _draw_slider_bar(
    ax,
    percentile: float | None,
    *,
    cmap: str = _SLIDER_CMAP,
    show_axis: bool = False,
) -> None:
    """RCSB-style horizontal percentile bar with a single black marker."""

    ax.imshow(
        _GRADIENT,
        aspect="auto",
        cmap=cmap,
        extent=(0.0, 1.0, 0.0, 1.0),
        interpolation="bilinear",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    if show_axis:
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=7)
        ax.tick_params(axis="x", length=2, pad=1, colors="#333333")
    else:
        ax.set_xticks([])

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#1a1a1a")
        spine.set_linewidth(0.6)

    if percentile is not None:
        ax.add_patch(
            Rectangle(
                (max(0.0, min(1.0, percentile)) - 0.005, -0.10),
                0.010,
                1.20,
                color="#0e0e0e",
                clip_on=False,
                zorder=5,
            )
        )


_LAYOUT = {
    "left": 0.085,
    "label_width": 0.190,
    "gap_label_slider": 0.012,
    "slider_width": 0.460,
    "gap_slider_pct": 0.014,
    "pct_width": 0.045,
    "gap_pct_value": 0.010,
    "value_width": 0.115,
}


def _draw_slider_row(
    fig: plt.Figure,
    *,
    bottom: float,
    height: float,
    label: str,
    raw: float | None,
    pct: float | None,
    units: str = "",
    show_axis: bool = False,
) -> None:
    L = _LAYOUT
    x = L["left"]
    label_ax = fig.add_axes((x, bottom, L["label_width"], height))
    label_ax.axis("off")
    label_ax.text(
        1.0,
        0.5,
        label,
        fontsize=10,
        ha="right",
        va="center",
        transform=label_ax.transAxes,
    )
    x += L["label_width"] + L["gap_label_slider"]

    slider_ax = fig.add_axes(
        (
            x,
            bottom + (0.20 if not show_axis else 0.30) * height,
            L["slider_width"],
            (0.55 if not show_axis else 0.42) * height,
        )
    )
    _draw_slider_bar(slider_ax, pct, show_axis=show_axis)
    x += L["slider_width"] + L["gap_slider_pct"]

    pct_ax = fig.add_axes((x, bottom + 0.18 * height, L["pct_width"], 0.64 * height))
    pct_ax.set_xlim(0, 1)
    pct_ax.set_ylim(0, 1)
    pct_ax.axis("off")
    pct_text = "n/a" if pct is None else f"{pct * 100:0.0f}%"
    pct_ax.text(
        0.5,
        0.5,
        pct_text,
        ha="center",
        va="center",
        fontsize=9,
        color="#222222",
        transform=pct_ax.transAxes,
    )
    x += L["pct_width"] + L["gap_pct_value"]

    value_ax = fig.add_axes((x, bottom + 0.18 * height, L["value_width"], 0.64 * height))
    value_ax.set_xlim(0, 1)
    value_ax.set_ylim(0, 1)
    value_ax.set_xticks([])
    value_ax.set_yticks([])
    for spine in value_ax.spines.values():
        spine.set_edgecolor("#5a5a5a")
        spine.set_linewidth(0.6)
    raw_text = _format_raw(raw)
    if units:
        raw_text = f"{raw_text} {units}"
    value_ax.text(
        0.5,
        0.5,
        raw_text,
        ha="center",
        va="center",
        fontsize=9,
        transform=value_ax.transAxes,
    )


def _draw_slider_panel(
    fig: plt.Figure,
    *,
    top: float,
    height: float,
    row: Mapping[str, Any],
    include_overall: bool = True,
) -> None:
    """RCSB-style 'Metric | Percentile rank | %ile | Value' table."""

    rows: list[tuple[str, float | None, float | None, str]] = []
    if include_overall:
        score = _row_meta_score(row)
        rows.append(("Overall meta score", score, score, ""))
    fv = _feature_view(row)
    for feat in META_SCORE_FEATURES:
        raw, pct = fv[feat]
        rows.append((_FEATURE_DISPLAY.get(feat, feat), raw, pct, _FEATURE_UNITS.get(feat, "")))

    L = _LAYOUT
    header_y = top
    body_top = top - 0.024
    n_rows = len(rows)
    body_height = height - 0.024
    row_h = body_height / n_rows

    # Header band
    band = fig.add_axes(
        (
            L["left"],
            header_y - 0.022,
            L["label_width"] + L["gap_label_slider"] + L["slider_width"] + L["gap_slider_pct"]
            + L["pct_width"] + L["gap_pct_value"] + L["value_width"],
            0.022,
        )
    )
    band.set_xlim(0, 1)
    band.set_ylim(0, 1)
    band.axis("off")
    band.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0, color="#efe9d8", transform=band.transAxes))

    band_w = (
        L["label_width"] + L["gap_label_slider"] + L["slider_width"]
        + L["gap_slider_pct"] + L["pct_width"] + L["gap_pct_value"] + L["value_width"]
    )

    def _hx(col_start: float, col_w: float) -> float:
        return (col_start - L["left"] + col_w / 2) / band_w

    # Place header labels at column centers
    band.text(
        _hx(L["left"], L["label_width"]),
        0.5,
        "Metric",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        transform=band.transAxes,
    )
    slider_start = L["left"] + L["label_width"] + L["gap_label_slider"]
    band.text(
        _hx(slider_start, L["slider_width"]),
        0.5,
        "Percentile rank",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        transform=band.transAxes,
    )
    pct_start = slider_start + L["slider_width"] + L["gap_slider_pct"]
    band.text(
        _hx(pct_start, L["pct_width"]),
        0.5,
        "%",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        transform=band.transAxes,
    )
    value_start = pct_start + L["pct_width"] + L["gap_pct_value"]
    band.text(
        _hx(value_start, L["value_width"]),
        0.5,
        "Value",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        transform=band.transAxes,
    )

    for i, (label, raw, pct, units) in enumerate(rows):
        row_bottom = body_top - (i + 1) * row_h
        show_axis = i == n_rows - 1
        _draw_slider_row(
            fig,
            bottom=row_bottom,
            height=row_h * 0.92,
            label=label,
            raw=raw,
            pct=pct,
            units=units,
            show_axis=show_axis,
        )

    bottom_y = body_top - n_rows * row_h
    legend_ax = fig.add_axes((slider_start, bottom_y - 0.022, L["slider_width"], 0.018))
    legend_ax.axis("off")
    legend_ax.annotate(
        "Worse",
        xy=(0.0, 0.5),
        xytext=(0.08, 0.5),
        xycoords="axes fraction",
        textcoords="axes fraction",
        fontsize=8.5,
        color="#7a1d1d",
        va="center",
        ha="left",
        arrowprops=dict(arrowstyle="<-", color="#7a1d1d", lw=0.7),
    )
    legend_ax.annotate(
        "Better",
        xy=(1.0, 0.5),
        xytext=(0.92, 0.5),
        xycoords="axes fraction",
        textcoords="axes fraction",
        fontsize=8.5,
        color="#1d5d1d",
        va="center",
        ha="right",
        arrowprops=dict(arrowstyle="->", color="#1d5d1d", lw=0.7),
    )


# ---------------------------------------------------------------------------
# compact, fixed-width tables (no matplotlib.table -- it truncates labels)
# ---------------------------------------------------------------------------

def _draw_fixed_table(
    fig: plt.Figure,
    *,
    x: float,
    y_top: float,
    w: float,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    col_fracs: Sequence[float],
    row_height: float = 0.024,
    header_color: str = "#efe9d8",
    font_size: float = 8.5,
) -> float:
    """Draw a table anchored at top ``y_top``, growing downward.

    Returns the bottom y of the table (figure fraction).
    """

    assert abs(sum(col_fracs) - 1.0) < 1e-6, "col_fracs must sum to 1"

    n_rows = len(rows)
    table_h = row_height * (n_rows + 1)
    ax = fig.add_axes((x, y_top - table_h, w, table_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if not rows:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=10, color="#555")
        return y_top - table_h

    # Header row at the top of the axes
    cell_h = 1.0 / (n_rows + 1)
    header_top = 1.0
    ax.add_patch(Rectangle((0.0, header_top - cell_h), 1.0, cell_h, color=header_color, zorder=1))
    x_left = 0.0
    for frac, label in zip(col_fracs, headers):
        ax.add_patch(
            Rectangle(
                (x_left, header_top - cell_h),
                frac,
                cell_h,
                fill=False,
                edgecolor=_TABLE_RULE,
                linewidth=0.5,
                zorder=2,
            )
        )
        ax.text(
            x_left + frac / 2,
            header_top - cell_h / 2,
            label,
            ha="center",
            va="center",
            fontsize=font_size + 0.5,
            fontweight="bold",
            color="#111111",
            transform=ax.transAxes,
        )
        x_left += frac

    # Approx max characters per column based on width and font size
    inch_w = w * _A4[0]
    max_chars_per_col = [max(4, int(frac * inch_w * 12)) for frac in col_fracs]

    cur_y = header_top - cell_h
    for r_idx, row_vals in enumerate(rows):
        cell_bot = cur_y - cell_h
        bg = "#ffffff" if r_idx % 2 == 0 else "#f6f6f0"
        ax.add_patch(Rectangle((0.0, cell_bot), 1.0, cell_h, color=bg, zorder=1))
        x_left = 0.0
        for frac, cell, max_chars in zip(col_fracs, row_vals, max_chars_per_col):
            ax.add_patch(
                Rectangle(
                    (x_left, cell_bot),
                    frac,
                    cell_h,
                    fill=False,
                    edgecolor=_TABLE_RULE,
                    linewidth=0.4,
                    zorder=2,
                )
            )
            ax.text(
                x_left + frac / 2,
                cell_bot + cell_h / 2,
                _truncate(str(cell), max_chars),
                ha="center",
                va="center",
                fontsize=font_size,
                color="#1a1a1a",
                transform=ax.transAxes,
            )
            x_left += frac
        cur_y = cell_bot

    return y_top - table_h


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------

def _cover_page(
    pdf: PdfPages,
    *,
    title: str,
    subtitle_lines: Sequence[str],
    entry_id: str,
    meta_pairs: Sequence[tuple[str, str]],
    info_lines: Sequence[str],
    software_lines: Sequence[tuple[str, str]],
    page_no: int,
    total: int,
) -> None:
    fig = _new_figure()
    _add_page_header(fig, page_no=page_no, total=total, title=title, entry=entry_id)

    title_ax = fig.add_axes((0.07, 0.85, 0.86, 0.06))
    title_ax.axis("off")
    title_ax.text(
        0.5,
        0.5,
        title,
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color="#101010",
        transform=title_ax.transAxes,
    )

    sub_ax = fig.add_axes((0.07, 0.815, 0.86, 0.025))
    sub_ax.axis("off")
    sub_ax.text(
        0.5,
        0.5,
        " – ".join(subtitle_lines),
        ha="center",
        va="center",
        fontsize=11,
        color="#1f1f1f",
        transform=sub_ax.transAxes,
    )

    _draw_meta_block(fig, x=0.10, y=0.62, w=0.80, h=0.18, pairs=meta_pairs)
    _draw_info_box(fig, x=0.13, y=0.43, w=0.74, h=0.16, lines=info_lines)

    sw_ax = fig.add_axes((0.10, 0.20, 0.80, 0.18))
    sw_ax.set_xlim(0, 1)
    sw_ax.set_ylim(0, 1)
    sw_ax.axis("off")
    sw_ax.text(
        0.0,
        0.95,
        "The following software and reference data were used in this report:",
        fontsize=10,
        ha="left",
        va="top",
        transform=sw_ax.transAxes,
    )
    n = len(software_lines)
    if n:
        top = 0.80
        line_h = 0.68 / max(1, n)
        for i, (k, v) in enumerate(software_lines):
            ypos = top - i * line_h
            sw_ax.text(0.40, ypos, k, fontsize=10, ha="right", va="top", transform=sw_ax.transAxes)
            sw_ax.text(0.42, ypos, ":", fontsize=10, ha="center", va="top", transform=sw_ax.transAxes)
            sw_ax.text(0.44, ypos, v, fontsize=10, ha="left", va="top", transform=sw_ax.transAxes)

    _add_page_footer(fig, page_no=page_no, total=total, last=False)
    pdf.savefig(fig)
    plt.close(fig)


def _quality_page(
    pdf: PdfPages,
    *,
    title: str,
    entry_id: str,
    section_no: str,
    section_title: str,
    pre_lines: Sequence[str],
    row: Mapping[str, Any],
    page_no: int,
    total: int,
    last: bool = False,
) -> None:
    fig = _new_figure()
    _add_page_header(fig, page_no=page_no, total=total, title=title, entry=entry_id)
    _draw_section_heading(fig, x=0.07, y=0.91, w=0.86, h=0.03, number=section_no, title=section_title)
    intro_ax = fig.add_axes((0.10, 0.79, 0.80, 0.11))
    intro_ax.axis("off")
    for i, line in enumerate(pre_lines):
        intro_ax.text(
            0.0,
            0.95 - i * 0.18,
            line,
            fontsize=10,
            ha="left",
            va="top",
            transform=intro_ax.transAxes,
        )

    _draw_slider_panel(fig, top=0.76, height=0.60, row=row, include_overall=True)

    _add_page_footer(fig, page_no=page_no, total=total, last=last)
    pdf.savefig(fig)
    plt.close(fig)


def _per_interface_page(
    pdf: PdfPages,
    *,
    title: str,
    entry_id: str,
    section_no: str,
    rows: Sequence[Mapping[str, Any]],
    page_no: int,
    total: int,
    last: bool = False,
) -> None:
    fig = _new_figure()
    _add_page_header(fig, page_no=page_no, total=total, title=title, entry=entry_id)
    _draw_section_heading(
        fig, x=0.07, y=0.91, w=0.86, h=0.03,
        number=section_no, title="Per-interface raw scores",
    )

    intro_ax = fig.add_axes((0.10, 0.83, 0.80, 0.06))
    intro_ax.axis("off")
    intro_ax.text(
        0.0,
        1.0,
        "Each row is one chain pair detected by AlphaJudge.",
        fontsize=9,
        ha="left",
        va="top",
        transform=intro_ax.transAxes,
    )
    intro_ax.text(
        0.0,
        0.55,
        "The Meta column is the averaged percentile across the 10 metascore "
        "features (higher is better).",
        fontsize=9,
        ha="left",
        va="top",
        transform=intro_ax.transAxes,
    )

    headers = ["Model", "Interface", "Residues", "Meta", "LIS", "ipSAE", "pDockQ2", "ipTM", "PAE", "Sc"]
    sorted_rows = sorted(
        rows,
        key=lambda r: (_row_meta_score(r) if _row_meta_score(r) is not None else -1.0),
        reverse=True,
    )
    body: list[list[str]] = []
    for r in sorted_rows:
        body.append(
            [
                _truncate(str(r.get("model_used") or ""), 26),
                str(r.get("interface") or ""),
                str(r.get("interface_num_intf_residues") or ""),
                _format_raw(_row_meta_score(r)),
                _format_raw(_safe_float(r.get("interface_LIS"))),
                _format_raw(_safe_float(r.get("interface_ipSAE"))),
                _format_raw(_safe_float(r.get("interface_pDockQ2"))),
                _format_raw(_safe_float(r.get("iptm"))),
                _format_raw(_safe_float(r.get("average_interface_pae"))),
                _format_raw(_safe_float(r.get("interface_sc"))),
            ]
        )

    col_fracs = [0.18, 0.10, 0.10, 0.08, 0.08, 0.09, 0.10, 0.07, 0.08, 0.12]
    _draw_fixed_table(
        fig,
        x=0.07,
        y_top=0.78,
        w=0.86,
        headers=headers,
        rows=body,
        col_fracs=col_fracs,
        row_height=0.024,
    )

    _add_page_footer(fig, page_no=page_no, total=total, last=last)
    pdf.savefig(fig)
    plt.close(fig)


def _pae_page(
    pdf: PdfPages,
    *,
    title: str,
    entry_id: str,
    section_no: str,
    image_path: Path,
    model_label: str,
    page_no: int,
    total: int,
    last: bool = False,
) -> None:
    fig = _new_figure()
    _add_page_header(fig, page_no=page_no, total=total, title=title, entry=entry_id)
    _draw_section_heading(
        fig, x=0.07, y=0.91, w=0.86, h=0.03,
        number=section_no, title=f"Predicted Aligned Error – {model_label}",
    )
    img_ax = fig.add_axes((0.13, 0.10, 0.74, 0.78))
    try:
        img = mpimg.imread(str(image_path))
        img_ax.imshow(img)
    except Exception as e:
        img_ax.text(0.5, 0.5, f"PAE image unavailable\n({e})", ha="center", va="center")
    img_ax.set_xticks([])
    img_ax.set_yticks([])
    for spine in img_ax.spines.values():
        spine.set_visible(False)
    _add_page_footer(fig, page_no=page_no, total=total, last=last)
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

def _aggregate_cover_page(
    pdf: PdfPages,
    *,
    summary_csv: Path,
    n_complexes: int,
    n_interfaces: int,
    scores: Sequence[float],
    top_rows: Sequence[tuple[str, float, Mapping[str, Any]]],
    backends: Mapping[str, int],
    page_no: int,
    total: int,
) -> None:
    fig = _new_figure()
    _add_page_header(
        fig, page_no=page_no, total=total,
        title=_REPORT_TITLE, entry="Aggregate report",
    )

    title_ax = fig.add_axes((0.07, 0.87, 0.86, 0.06))
    title_ax.axis("off")
    title_ax.text(
        0.5,
        0.5,
        _REPORT_TITLE,
        fontsize=22,
        fontweight="bold",
        ha="center",
        va="center",
        transform=title_ax.transAxes,
    )
    sub_ax = fig.add_axes((0.07, 0.835, 0.86, 0.025))
    sub_ax.axis("off")
    sub_ax.text(
        0.5,
        0.5,
        f"Aggregate report – {n_interfaces} interfaces across {n_complexes} complexes",
        ha="center",
        va="center",
        fontsize=11,
        color="#1f1f1f",
        transform=sub_ax.transAxes,
    )

    meta = [
        ("Source", _shorten_path(str(summary_csv), max_len=58)),
        ("Date", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("Complexes", str(n_complexes)),
        ("Interfaces", str(n_interfaces)),
    ]
    if backends:
        meta.append(("Backends", ", ".join(f"{k}={v}" for k, v in backends.items())))
    _draw_meta_block(fig, x=0.10, y=0.68, w=0.80, h=0.13, pairs=meta)

    info = [
        "This report scores AlphaFold-predicted complexes against the",
        "AlphaJudge benchmark_26 reference set.",
        "All percentiles are archive percentiles; higher is better.",
    ]
    _draw_info_box(fig, x=0.13, y=0.54, w=0.74, h=0.11, lines=info)

    hist_ax = fig.add_axes((0.10, 0.36, 0.50, 0.14))
    if scores:
        hist_ax.hist(scores, bins=24, range=(0.0, 1.0), color="#5688c7", edgecolor="white")
    hist_ax.set_xlim(0.0, 1.0)
    hist_ax.set_xlabel("Interface meta score (one point per interface)", fontsize=9, labelpad=2)
    hist_ax.set_ylabel("Interfaces", fontsize=9)
    hist_ax.set_title("Distribution across cohort", fontsize=10, loc="left")
    hist_ax.tick_params(labelsize=8)

    stats_ax = fig.add_axes((0.64, 0.36, 0.26, 0.14))
    stats_ax.axis("off")
    if scores:
        median = sorted(scores)[len(scores) // 2]
        mean = sum(scores) / len(scores)
        stats_ax.text(0.0, 0.95, "Cohort statistics", fontsize=11, fontweight="bold", transform=stats_ax.transAxes)
        lines = [
            f"min      = {min(scores):.3f}",
            f"median   = {median:.3f}",
            f"mean     = {mean:.3f}",
            f"max      = {max(scores):.3f}",
            f"≥ 0.5  = {sum(1 for s in scores if s >= 0.5)} ({100*sum(1 for s in scores if s >= 0.5)/len(scores):.0f}%)",
            f"≥ 0.7  = {sum(1 for s in scores if s >= 0.7)} ({100*sum(1 for s in scores if s >= 0.7)/len(scores):.0f}%)",
        ]
        for i, line in enumerate(lines):
            stats_ax.text(0.0, 0.78 - i * 0.12, line, fontsize=10, family="monospace", transform=stats_ax.transAxes)

    title2_ax = fig.add_axes((0.07, 0.305, 0.86, 0.020))
    title2_ax.axis("off")
    title2_ax.text(
        0.5,
        0.5,
        f"Top {len(top_rows)} interfaces by meta score",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        transform=title2_ax.transAxes,
    )
    headers = ["Rank", "Complex / interface", "Meta", "LIS", "ipSAE", "ipTM", "PAE", "Sc"]
    body: list[list[str]] = []
    for i, (name, score, row) in enumerate(top_rows, start=1):
        body.append(
            [
                str(i),
                _truncate(name, 34),
                _format_raw(score),
                _format_raw(_safe_float(row.get("interface_LIS"))),
                _format_raw(_safe_float(row.get("interface_ipSAE"))),
                _format_raw(_safe_float(row.get("iptm"))),
                _format_raw(_safe_float(row.get("average_interface_pae"))),
                _format_raw(_safe_float(row.get("interface_sc"))),
            ]
        )
    col_fracs = [0.07, 0.34, 0.09, 0.09, 0.10, 0.09, 0.10, 0.12]
    _draw_fixed_table(
        fig,
        x=0.07,
        y_top=0.285,
        w=0.86,
        headers=headers,
        rows=body,
        col_fracs=col_fracs,
        row_height=0.020,
    )

    _add_page_footer(fig, page_no=page_no, total=total, last=False)
    pdf.savefig(fig)
    plt.close(fig)


def _interface_summary_page(
    pdf: PdfPages,
    *,
    complex_name: str,
    interface_label: str,
    row: Mapping[str, Any],
    cohort_position: tuple[int, int] | None,
    page_no: int,
    total: int,
    last: bool,
) -> None:
    fig = _new_figure()
    entry = f"{_truncate(complex_name, 26)} / {interface_label}"
    _add_page_header(
        fig, page_no=page_no, total=total,
        title=_REPORT_TITLE, entry=_truncate(entry, 40),
    )

    title_ax = fig.add_axes((0.07, 0.91, 0.86, 0.05))
    title_ax.axis("off")
    title_ax.text(
        0.5,
        0.5,
        _truncate(complex_name, 60),
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        transform=title_ax.transAxes,
    )

    sub_ax = fig.add_axes((0.07, 0.875, 0.86, 0.025))
    sub_ax.axis("off")
    bits = [
        f"Interface {interface_label}",
        f"Model {row.get('model_used', '?')}",
    ]
    if cohort_position is not None:
        bits.append(f"Rank {cohort_position[0]} of {cohort_position[1]}")
    n_res = row.get("interface_num_intf_residues")
    if n_res:
        bits.append(f"{n_res} interface residues")
    sub_ax.text(0.5, 0.5, "  •  ".join(bits), ha="center", va="center", fontsize=10, color="#222", transform=sub_ax.transAxes)

    _draw_section_heading(
        fig, x=0.07, y=0.83, w=0.86, h=0.025,
        number="1", title="Overall quality at a glance",
    )

    _draw_slider_panel(fig, top=0.79, height=0.62, row=row, include_overall=True)

    note_ax = fig.add_axes((0.10, 0.07, 0.80, 0.06))
    note_ax.axis("off")
    note_ax.text(
        0.5,
        1.0,
        "Black marker shows this interface's percentile rank against the AlphaJudge benchmark "
        "(higher = better).",
        ha="center",
        va="top",
        fontsize=9,
        color="#555",
        transform=note_ax.transAxes,
    )

    _add_page_footer(fig, page_no=page_no, total=total, last=last)
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def _find_pae_png(run_dir: Path, model_used: str) -> Path | None:
    if not run_dir.is_dir():
        return None
    candidates = [
        run_dir / f"pae_{model_used}.png",
        *run_dir.glob(f"*{model_used}*PAE*plot*.png"),
        *run_dir.glob(f"*{model_used}*.png"),
        *run_dir.glob("*PAE*plot*ranked_0*.png"),
    ]
    seen: set[Path] = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if cand.exists() and cand.is_file():
            return cand
    return None


def generate_per_run_report(
    run_dir: str | Path,
    *,
    csv_name: str = "interfaces.csv",
    out_pdf: str | Path | None = None,
) -> Path | None:
    """Build a per-run report.pdf next to ``interfaces.csv``."""

    _setup_rcparams()

    run_dir = Path(run_dir)
    interfaces_csv = run_dir / csv_name
    if not interfaces_csv.exists():
        logger.warning("no %s in %s; skipping report", csv_name, run_dir)
        return None
    rows = _read_csv_rows(interfaces_csv)
    if not rows:
        logger.warning("empty %s in %s; skipping report", csv_name, run_dir)
        return None

    out_pdf = Path(out_pdf) if out_pdf is not None else run_dir / "report.pdf"
    best = _best_row(rows)
    if best is None:
        logger.warning("no usable rows in %s; skipping report", interfaces_csv)
        return None

    by_model: "OrderedDict[str, list[Mapping[str, Any]]]" = OrderedDict()
    for r in rows:
        by_model.setdefault(str(r.get("model_used") or ""), []).append(r)
    best_model = str(best.get("model_used") or "")
    other_models = [m for m in by_model if m and m != best_model]

    pae_png = _find_pae_png(run_dir, best_model)
    # Pick the best model's rows for the per-interface slider pages; sort by
    # metascore descending so the strongest interface comes first.
    best_model_rows = by_model.get(best_model, list(rows))
    interface_rows = sorted(
        best_model_rows,
        key=lambda r: (_row_meta_score(r) if _row_meta_score(r) is not None else -1.0),
        reverse=True,
    )
    show_interface_table = len(interface_rows) > 1

    total = (
        1  # cover
        + (1 if show_interface_table else 0)  # overview table
        + len(interface_rows)  # one slider page per interface
        + (1 if pae_png else 0)  # PAE heatmap
        + len(other_models)  # non-best-model appendix
    )

    entry_id = _truncate(run_dir.name, 36)
    chains = _detect_chain_set(rows)
    backend = _detect_backend(rows)
    score = _row_meta_score(best)
    score_label = "n/a" if score is None else f"{score:.3f} ({_decile_label(score)})"

    meta_pairs: list[tuple[str, str]] = [
        ("Complex", run_dir.name),
        ("Date", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("Backend", backend),
        ("Chains", ", ".join(sorted(chains)) or "?"),
        ("Interface rows", str(len(rows))),
        ("Best model", best_model or "?"),
        ("Best meta score", score_label),
    ]
    info_lines = [
        "AlphaJudge interface validation report.",
        "Each metric is converted to its archive percentile against the frozen",
        "benchmark distribution; the overall meta score is the unweighted mean over",
        "available features.",
    ]
    software_lines: list[tuple[str, str]] = [
        ("Reference distribution", _BENCHMARK_TAG),
        ("Source CSV", _shorten_path(str(interfaces_csv), max_len=62)),
        ("Models analysed", _truncate(", ".join(by_model.keys()) or "?", 60)),
    ]

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(out_pdf)) as pdf:
        page_no = 1
        _cover_page(
            pdf,
            title=_REPORT_TITLE,
            subtitle_lines=[run_dir.name, backend],
            entry_id=entry_id,
            meta_pairs=meta_pairs,
            info_lines=info_lines,
            software_lines=software_lines,
            page_no=page_no,
            total=total,
        )

        next_section = 1
        if show_interface_table:
            page_no += 1
            _per_interface_page(
                pdf,
                title=_REPORT_TITLE,
                entry_id=entry_id,
                section_no=str(next_section),
                rows=rows,
                page_no=page_no,
                total=total,
                last=(page_no == total),
            )
            next_section += 1

        quality_section_no = next_section
        for i, row in enumerate(interface_rows):
            page_no += 1
            iface_label = str(row.get("interface") or "?")
            n_res = row.get("interface_num_intf_residues") or "?"
            if show_interface_table:
                section_title = f"Interface {iface_label}"
                section_no = f"{quality_section_no}.{i + 1}"
            else:
                section_title = "Overall quality at a glance"
                section_no = str(quality_section_no)
            _quality_page(
                pdf,
                title=_REPORT_TITLE,
                entry_id=entry_id,
                section_no=section_no,
                section_title=section_title,
                pre_lines=[
                    f"Model: {row.get('model_used', best_model)}",
                    f"Chain pair: {iface_label}    Residues at interface: {n_res}",
                ],
                row=row,
                page_no=page_no,
                total=total,
                last=(page_no == total),
            )
        next_section = quality_section_no + 1

        if pae_png is not None:
            page_no += 1
            _pae_page(
                pdf,
                title=_REPORT_TITLE,
                entry_id=entry_id,
                section_no=str(next_section),
                image_path=pae_png,
                model_label=best_model,
                page_no=page_no,
                total=total,
                last=(page_no == total),
            )
            next_section += 1

        for m in other_models:
            m_rows = by_model[m]
            m_best = _best_row(m_rows) or m_rows[0]
            page_no += 1
            _quality_page(
                pdf,
                title=_REPORT_TITLE,
                entry_id=entry_id,
                section_no=f"A.{m}",
                section_title=f"Appendix – model {m}",
                pre_lines=[
                    f"Interface: {m_best.get('interface', '?')}",
                    f"Residues at interface: {m_best.get('interface_num_intf_residues', '?')}",
                ],
                row=m_best,
                page_no=page_no,
                total=total,
                last=(page_no == total),
            )

    logger.info("wrote %s", out_pdf)
    return out_pdf


def generate_aggregate_report(
    summary_csv: str | Path,
    *,
    out_pdf: str | Path,
    top_n: int = 10,
    max_complexes: int | None = None,
) -> Path | None:
    """Build a multi-page aggregate validation PDF from a merged interfaces CSV.

    Statistics are computed **per interface** (one data point per chain pair
    in the merged CSV). A multimer with 15 interfaces contributes 15 points.
    """

    _setup_rcparams()

    summary_csv = Path(summary_csv)
    if not summary_csv.exists():
        logger.warning("summary CSV not found: %s", summary_csv)
        return None
    rows = _read_csv_rows(summary_csv)
    if not rows:
        logger.warning("empty summary CSV: %s", summary_csv)
        return None

    # One entry per scorable interface row.
    ranked: list[tuple[str, str, str, float, Mapping[str, Any]]] = []
    for r in rows:
        cname = str(r.get("jobs") or r.get("pair") or r.get("complex") or "")
        iface = str(r.get("interface") or "")
        if not cname:
            continue
        score = _row_meta_score(r)
        if score is None:
            continue
        label = f"{cname} · {iface}" if iface else cname
        ranked.append((label, cname, iface, score, r))
    if not ranked:
        logger.warning("no scorable interface rows in %s", summary_csv)
        return None
    ranked.sort(key=lambda t: t[3], reverse=True)

    top_rows = [(label, score, r) for label, _, _, score, r in ranked[:top_n]]
    ranked_per_page = ranked if max_complexes is None else ranked[:max_complexes]

    # Backends counted per complex (so a multimer doesn't multi-count).
    seen_backend: dict[str, str] = {}
    for _label, cname, _iface, _score, r in ranked:
        if cname not in seen_backend:
            seen_backend[cname] = _detect_backend([r])
    backends: dict[str, int] = {}
    for b in seen_backend.values():
        backends[b] = backends.get(b, 0) + 1

    scores = [s for _, _, _, s, _ in ranked]
    n_complexes = len(seen_backend)
    n_interfaces = len(ranked)
    total = 1 + len(ranked_per_page)

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(out_pdf)) as pdf:
        _aggregate_cover_page(
            pdf,
            summary_csv=summary_csv,
            n_complexes=n_complexes,
            n_interfaces=n_interfaces,
            scores=scores,
            top_rows=top_rows,
            backends=backends,
            page_no=1,
            total=total,
        )
        for rank, (_label, cname, iface, _score, r) in enumerate(ranked_per_page, start=1):
            _interface_summary_page(
                pdf,
                complex_name=cname,
                interface_label=iface or "?",
                row=r,
                cohort_position=(rank, len(ranked_per_page)),
                page_no=1 + rank,
                total=total,
                last=(rank == len(ranked_per_page)),
            )

    logger.info("wrote %s", out_pdf)
    return out_pdf


def main_aggregate(argv: list[str] | None = None) -> None:
    """Console entry point for ``alphajudge-report``."""
    import argparse

    parser = argparse.ArgumentParser(
        "alphajudge-report",
        description="Generate an RCSB-style validation PDF from an AlphaJudge interfaces CSV.",
    )
    parser.add_argument(
        "input",
        help="Either a run directory (with interfaces.csv) or a merged summary CSV.",
    )
    parser.add_argument("--out-pdf", required=True, help="Output PDF path.")
    parser.add_argument(
        "--csv-name",
        default="interfaces.csv",
        help="CSV filename inside a run directory (default: interfaces.csv).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top-N rows shown on the aggregate cover (aggregate mode only).",
    )
    parser.add_argument(
        "--max-complexes",
        type=int,
        default=None,
        help="Optional cap on per-complex pages in aggregate mode.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    src = Path(args.input)
    if src.is_dir():
        result = generate_per_run_report(
            src, csv_name=args.csv_name, out_pdf=args.out_pdf
        )
    else:
        result = generate_aggregate_report(
            src,
            out_pdf=args.out_pdf,
            top_n=args.top_n,
            max_complexes=args.max_complexes,
        )
    if result is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main_aggregate()
