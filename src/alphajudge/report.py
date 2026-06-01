"""AlphaJudge validation reports for AlphaJudge interface scores.

The layout uses a compact scientific validation-report format with only
AlphaJudge branding. No external organisation logo, PDB/wwPDB wordmark,
AlphaFold logo, or EMBL-EBI logo is embedded.

The percentile pages use a compact red -> white -> blue percentile graphic.
The PAE page is rendered, when raw PAE values are available, in the visual
style of the AlphaFold Database PAE panel: a green square heatmap with
Scored residue / Aligned residue axes and a horizontal expected-position-error
colour bar.
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator

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

# Percentile graphic: red -> pale centre -> blue.
_SLIDER_CMAP = LinearSegmentedColormap.from_list(
    "alphajudge_percentile",
    [
        (0.00, "#ff1a1a"),
        (0.35, "#ffd1d1"),
        (0.50, "#f4f0f0"),
        (0.65, "#d8d8ff"),
        (1.00, "#171cff"),
    ],
)

# AlphaFold-DB-like PAE palette: low error = dark green, high error = pale.
_PAE_CMAP = LinearSegmentedColormap.from_list(
    "alphafold_db_like_pae",
    [
        (0.00, "#005f2f"),
        (0.20, "#16813e"),
        (0.45, "#56ad55"),
        (0.72, "#cdebc5"),
        (1.00, "#f7fbf1"),
    ],
)

_INFO_BG = "#ffb3b3"
_INFO_EDGE = "#ff0000"
_HEADER_RULE = "#303030"
_TABLE_RULE = "#202020"

_AJ_BLUE = "#1f4e79"
_AJ_GREEN = "#2e8b57"
_AJ_GOLD = "#d08c00"
_AJ_DARK = "#111111"

_REPORT_TITLE = "AlphaJudge Interface validation Report"
_BENCHMARK_TAG = "benchmark_26 (final_sync_20260523, n=7,756 AF2/AF3 rows)"

_GRADIENT = np.tile(np.linspace(0.0, 1.0, 1024), (2, 1))

_FEATURE_DISPLAY = {
    "interface_LIS": "Interface LIS",
    "interface_ipSAE": "Interface ipSAE",
    "interface_pDockQ2": "Interface pDockQ2",
    "iptm": "ipTM",
    "confidence_score": "Confidence score",
    "average_interface_pae": "Avg. interface PAE",
    "pDockQ/mpDockQ": "pDockQ / mpDockQ",
    "interface_sc": "Shape complementarity",
    "interface_hb": "Hydrogen bonds",
    "interface_area": "Interface area",
    "interface_solv_en": "Solvation energy",
}

_FEATURE_UNITS = {
    "average_interface_pae": "Å",
    "interface_area": "Å²",
    "interface_solv_en": "kcal/mol",
}

# Metric grouping for the slider panel. Lines are drawn only WITHIN each group
# (AF-derived vs. biophysical); the Meta-score row stays separate and is never
# joined to a polyline.
#
# Per-interface vs. complex-level: features that are scalars per predicted
# complex (not per chain pair) are pulled out of the per-interface slider
# panel and shown together with the PAE on a dedicated end-of-report page.
# In AF3 iptm is per chain pair (chain_pair_iptm), so it stays in the
# AF-derived group; confidence_score and pDockQ/mpDockQ are global to the
# complex and live in COMPLEX_LEVEL_FEATURES.
_AF_DERIVED_FEATURES = (
    "interface_LIS",
    "interface_ipSAE",
    "interface_pDockQ2",
    "iptm",
    "average_interface_pae",
)
_BIOPHYSICAL_FEATURES = (
    "interface_sc",
    "interface_hb",
    "interface_solv_en",
)
_COMPLEX_LEVEL_FEATURES = (
    "confidence_score",
    "pDockQ/mpDockQ",
)


# ---------------------------------------------------------------------------
# style + utility helpers
# ---------------------------------------------------------------------------

def _setup_rcparams() -> None:
    """Use a Computer-Modern-like serif PDF look, close to wwPDB reports."""
    rcparams = {
        "font.family": "serif",
        "font.serif": [
            "CMU Serif",
            "Computer Modern Roman",
            "Latin Modern Roman",
            "STIXGeneral",
            "DejaVu Serif",
            "Times New Roman",
            "Times",
        ],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.edgecolor": "#202020",
        "axes.linewidth": 0.6,
        "savefig.dpi": 300,
        # Keep text as searchable TrueType text in the PDF.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
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


def _draw_info_icon(fig: plt.Figure, *, x: float, y: float, r: float = 0.010) -> None:
    """Small circled 'i' marker using AlphaJudge brand colour."""
    ax = fig.add_axes((x - r, y - r, 2 * r, 2 * r))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(
        Circle(
            (0.5, 0.5),
            0.47,
            facecolor="white",
            edgecolor=_AJ_BLUE,
            linewidth=1.0,
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.5,
        0.48,
        "i",
        ha="center",
        va="center",
        fontsize=8,
        color=_AJ_BLUE,
        fontweight="bold",
        transform=ax.transAxes,
    )


def _draw_alphajudge_logo(
    fig: plt.Figure,
    *,
    x: float = 0.5,
    y: float = 0.93,
    w: float = 0.30,
    h: float = 0.080,
    compact: bool = False,
) -> None:
    """Plain-text AlphaJudge mark.

    Renders just "AlphaJudge report" (and a small "interface validation"
    sub-line in the non-compact form). Intentionally text-only to avoid
    any resemblance to third-party logos.
    """
    ax = fig.add_axes((x - w / 2, y - h / 2, w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if compact:
        ax.text(
            0.5,
            0.5,
            "AlphaJudge report",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=_AJ_DARK,
            transform=ax.transAxes,
        )
        return

    ax.text(
        0.5,
        0.62,
        "AlphaJudge report",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=_AJ_DARK,
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.28,
        "interface validation",
        ha="center",
        va="center",
        fontsize=9,
        color="#444444",
        transform=ax.transAxes,
    )


def _add_page_header(fig: plt.Figure, *, page_no: int, total: int, title: str, entry: str) -> None:
    """RCSB-style running header.

    The cover page in wwPDB reports has no running header; page 2 onward does.
    """
    if page_no <= 1:
        return

    ax = fig.add_axes((0.07, 0.952, 0.86, 0.036))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.0,
        0.62,
        f"Page {page_no}",
        fontsize=10,
        ha="left",
        va="center",
        color="#111111",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.62,
        title,
        fontsize=10,
        ha="center",
        va="center",
        color="#111111",
        transform=ax.transAxes,
    )
    ax.text(
        1.0,
        0.62,
        entry,
        fontsize=10,
        ha="right",
        va="center",
        color="#111111",
        transform=ax.transAxes,
    )
    ax.plot([0.0, 1.0], [0.18, 0.18], color=_HEADER_RULE, linewidth=0.6, transform=ax.transAxes)


def _add_page_footer(fig: plt.Figure, *, page_no: int, total: int, last: bool) -> None:
    """No footer mark; the running header already identifies the report."""
    return


def _draw_info_box(fig: plt.Figure, *, x: float, y: float, w: float, h: float, lines: Sequence[str]) -> None:
    """Square-corner pink/red cover callout, closer to wwPDB style."""
    ax = fig.add_axes((x, y, w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            linewidth=0.8,
            edgecolor=_INFO_EDGE,
            facecolor=_INFO_BG,
            transform=ax.transAxes,
        )
    )
    if not lines:
        return

    n = len(lines)
    top = 0.83
    line_h = 0.68 / max(1, n - 1) if n > 1 else 0.0
    for i, line in enumerate(lines):
        ax.text(
            0.5,
            top - i * line_h,
            line,
            ha="center",
            va="top",
            fontsize=10.5,
            color="#111111",
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


def _draw_section_heading(
    fig: plt.Figure,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    number: str,
    title: str,
    show_info: bool = False,
) -> None:
    """Large numbered section heading with RCSB-like spacing."""
    ax = fig.add_axes((x, y, w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    number_text = _truncate(str(number), 8)
    title_x = 0.060 + max(0, len(number_text) - 2) * 0.010

    ax.text(
        0.0,
        0.50,
        number_text,
        fontsize=17,
        fontweight="bold",
        ha="left",
        va="center",
        color="#101010",
        transform=ax.transAxes,
    )
    ax.text(
        title_x,
        0.50,
        title,
        fontsize=17,
        fontweight="bold",
        ha="left",
        va="center",
        color="#101010",
        transform=ax.transAxes,
    )

    if show_info:
        # Approximate icon placement immediately after the heading.
        icon_x = min(x + w - 0.020, x + title_x * w + 0.0120 * len(title) + 0.020)
        _draw_info_icon(fig, x=icon_x, y=y + h * 0.52, r=0.011)


# ---------------------------------------------------------------------------
# slider primitive
# ---------------------------------------------------------------------------

# Compact chart layout in figure coordinates.
_RCSB_SLIDER_LAYOUT = {
    "label_right": 0.235,
    "bar_x": 0.240,
    "bar_width": 0.382,
    "value_x": 0.632,
    "value_width": 0.120,
    "bar_height": 0.0105,
    "row_height": 0.0315,
}


def _clip_pct(pct: float | None) -> float | None:
    if pct is None or not math.isfinite(pct):
        return None
    return max(0.0, min(1.0, pct))


def _draw_slider_bar(
    ax,
    percentile: float | None = None,
    *,
    cmap=_SLIDER_CMAP,
    draw_marker: bool = True,
) -> None:
    """Thin wwPDB-style red-white-blue percentile bar."""
    ax.imshow(
        _GRADIENT,
        aspect="auto",
        cmap=cmap,
        extent=(0.0, 1.0, 0.0, 1.0),
        interpolation="bilinear",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    pct = _clip_pct(percentile)
    if draw_marker and pct is not None:
        ax.add_patch(
            Rectangle(
                (pct - 0.006, -0.15),
                0.012,
                1.30,
                facecolor="#0b0b0b",
                edgecolor="#0b0b0b",
                linewidth=0.4,
                clip_on=False,
                zorder=5,
            )
        )


def _metric_rows_for_slider_panel(
    row: Mapping[str, Any],
    *,
    include_overall: bool,
    groups: Sequence[tuple[str, Sequence[str]]] | None = None,
) -> list[tuple[str, float | None, float | None, str, str]]:
    """Return (label, raw, percentile, units, group) per slider row.

    Group is one of "overall" (the Meta-score row), "af" (AlphaFold-
    derived confidence features), "biophys" (biophysical features), or
    "complex" (per-complex scalars). The grouping is used by
    ``_draw_slider_panel`` to add vertical spacing between groups and to
    draw polylines only within a group.

    ``groups`` lets callers swap the per-interface feature list for a
    different set (e.g. just complex-level metrics on the end-of-report
    PAE page); when ``None`` the per-interface layout is used.
    """
    if groups is None:
        groups = (("af", _AF_DERIVED_FEATURES), ("biophys", _BIOPHYSICAL_FEATURES))

    rows: list[tuple[str, float | None, float | None, str, str]] = []

    if include_overall:
        score = _row_meta_score(row)
        rows.append(("Meta score", score, score, "", "overall"))

    fv = _feature_view(row)
    for group_tag, features in groups:
        for feat in features:
            if feat in fv:
                raw, pct = fv[feat]
            else:
                # Compute percentile even when feature isn't in METASCORE
                # (e.g. complex-level features were dropped from the metascore
                # but still need a slider bar).
                raw = _safe_float(row.get(feat))
                pct = calibrated_feature_percentile(feat, raw) if raw is not None else None
            rows.append(
                (
                    _FEATURE_DISPLAY.get(feat, feat),
                    raw,
                    pct,
                    _FEATURE_UNITS.get(feat, ""),
                    group_tag,
                )
            )

    return rows


def _draw_percentile_legend(
    fig: plt.Figure,
    *,
    x: float,
    y: float,
    w: float,
    label: str = "Percentile relative to AlphaJudge benchmark",
) -> None:
    ax = fig.add_axes((x, y, w, 0.032))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(
        Rectangle(
            (0.000, 0.55),
            0.010,
            0.30,
            facecolor="#0b0b0b",
            edgecolor="#0b0b0b",
            linewidth=0.4,
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.018,
        0.70,
        label,
        ha="left",
        va="center",
        fontsize=7.2,
        color="#111111",
        transform=ax.transAxes,
    )


def _draw_slider_panel(
    fig: plt.Figure,
    *,
    top: float,
    height: float,
    row: Mapping[str, Any],
    include_overall: bool = True,
    groups: Sequence[tuple[str, Sequence[str]]] | None = None,
) -> float:
    """Draw a compact wwPDB-style percentile graphic.

    The Meta-score row (if included) is rendered first and visually offset
    from the rest. Each group passed in ``groups`` is rendered as its own
    block, with its own connecting polyline; lines never cross the
    Meta-score row or a group boundary. When ``groups`` is ``None`` the
    standard per-interface layout (AF-derived + biophysical) is used.

    Returns the bottom y coordinate of the graphic.
    """
    rows = _metric_rows_for_slider_panel(
        row, include_overall=include_overall, groups=groups
    )
    n_rows = len(rows)
    if n_rows == 0:
        return top

    L = _RCSB_SLIDER_LAYOUT
    label_right = L["label_right"]
    bar_x = L["bar_x"]
    bar_w = L["bar_width"]
    value_x = L["value_x"]
    bar_h = L["bar_height"]

    # Vertical layout: row height shrinks if the panel has to fit many rows.
    # Inter-group gap pushes Q-score / AF / biophys apart.
    group_gap = 0.012
    n_group_changes = sum(
        1 for i in range(1, n_rows) if rows[i][4] != rows[i - 1][4]
    )
    available = height - 0.075 - n_group_changes * group_gap
    row_h = min(L["row_height"], max(0.026, available / max(1, n_rows)))
    header_y = top - 0.012

    # Column headers - no beige band, no boxed cells.
    fig.text(
        label_right - 0.020,
        header_y,
        "Metric",
        ha="center",
        va="center",
        fontsize=10,
        color="#111111",
    )
    fig.text(
        bar_x + bar_w / 2,
        header_y,
        "Percentile Ranks",
        ha="center",
        va="center",
        fontsize=10,
        color="#111111",
    )
    fig.text(
        value_x + 0.035,
        header_y,
        "Value",
        ha="center",
        va="center",
        fontsize=10,
        color="#111111",
    )

    # Compute per-row centres with extra spacing at group transitions.
    first_center = top - 0.048
    centers: list[float] = []
    cur_y = first_center
    prev_group: str | None = None
    for _label, _raw, _pct, _units, group in rows:
        if prev_group is not None and group != prev_group:
            cur_y -= group_gap
        centers.append(cur_y)
        cur_y -= row_h
        prev_group = group

    # Rows: label, thin gradient bar, raw value. All rows share the same
    # typography (PDB-validation-style uniform treatment); the inter-group
    # gap is what separates the overall metascore from the feature rows.
    pct_positions: list[tuple[int, float, str]] = []
    for i, ((label, raw, pct, units, group), center_y) in enumerate(zip(rows, centers)):
        pct_clipped = _clip_pct(pct)

        fig.text(
            label_right,
            center_y,
            label,
            ha="right",
            va="center",
            fontsize=9.2,
            color="#111111",
        )

        bar_ax = fig.add_axes((bar_x, center_y - bar_h / 2, bar_w, bar_h), zorder=2)
        _draw_slider_bar(bar_ax, None, draw_marker=False)

        raw_text = _format_raw(raw)
        if units and raw_text != "—":
            raw_text = f"{raw_text} {units}"

        fig.text(
            value_x,
            center_y,
            raw_text,
            ha="left",
            va="center",
            fontsize=9.2,
            color="#111111",
        )

        if pct_clipped is not None:
            pct_positions.append((i, pct_clipped, group))

    chart_top = centers[0] + row_h * 0.50
    chart_bottom = centers[-1] - row_h * 0.50

    line_ax = fig.add_axes((bar_x, chart_bottom, bar_w, chart_top - chart_bottom), zorder=20)
    line_ax.set_xlim(0.0, 1.0)
    line_ax.set_ylim(chart_bottom, chart_top)
    line_ax.axis("off")
    line_ax.patch.set_alpha(0.0)

    def _row_y(idx: int) -> float:
        return centers[idx]

    # Polyline segments per metric group (skip "overall" - the Meta-score row
    # is intentionally not connected to any feature row).
    by_group: dict[str, list[tuple[float, float]]] = {}
    for idx, pct, group in pct_positions:
        if group == "overall":
            continue
        by_group.setdefault(group, []).append((pct, _row_y(idx)))

    for points in by_group.values():
        if len(points) >= 2:
            xs = [p for p, _y in points]
            ys = [y for _p, y in points]
            line_ax.plot(xs, ys, color="#0b0b0b", linewidth=0.75, zorder=4)

    marker_w = 0.012
    marker_h = max(0.0042, min(0.0070, bar_h * 1.35))
    for idx, pct, group in pct_positions:
        y = _row_y(idx)
        line_ax.add_patch(
            Rectangle(
                (pct - marker_w / 2, y - marker_h / 2),
                marker_w,
                marker_h,
                facecolor="#0b0b0b",
                edgecolor="#0b0b0b",
                linewidth=0.45,
                zorder=6,
                clip_on=False,
            )
        )

    # Worse / Better labels directly beneath the bars.
    wb_y = chart_bottom - 0.011
    fig.text(
        bar_x,
        wb_y,
        "Worse",
        ha="left",
        va="center",
        fontsize=6.8,
        fontstyle="italic",
        color="#111111",
    )
    fig.text(
        bar_x + bar_w,
        wb_y,
        "Better",
        ha="right",
        va="center",
        fontsize=6.8,
        fontstyle="italic",
        color="#111111",
    )

    legend_y = chart_bottom - 0.045
    _draw_percentile_legend(fig, x=bar_x - 0.002, y=legend_y, w=0.55)

    return legend_y


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

    # Cover: no running header. Just the report title (no separate logo/wordmark).
    title_ax = fig.add_axes((0.07, 0.830, 0.86, 0.060))
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

    sub_ax = fig.add_axes((0.07, 0.690, 0.86, 0.040))
    sub_ax.axis("off")
    sub_ax.text(
        0.5,
        0.5,
        " - ".join(subtitle_lines),
        ha="center",
        va="center",
        fontsize=13,
        color="#1f1f1f",
        transform=sub_ax.transAxes,
    )

    _draw_meta_block(fig, x=0.10, y=0.535, w=0.80, h=0.135, pairs=meta_pairs)

    _draw_info_box(fig, x=0.09, y=0.350, w=0.82, h=0.135, lines=info_lines)

    sw_ax = fig.add_axes((0.10, 0.090, 0.80, 0.210))
    sw_ax.set_xlim(0, 1)
    sw_ax.set_ylim(0, 1)
    sw_ax.axis("off")

    # Short horizontal rule above the software block, as on the wwPDB cover.
    sw_ax.plot([0.0, 0.42], [0.98, 0.98], color=_HEADER_RULE, linewidth=0.6, transform=sw_ax.transAxes)

    sw_ax.text(
        0.0,
        0.84,
        "The following software and reference data were used in this report:",
        fontsize=10,
        ha="left",
        va="top",
        transform=sw_ax.transAxes,
    )

    n = len(software_lines)
    if n:
        top = 0.66
        line_h = 0.56 / max(1, n - 1) if n > 1 else 0.0
        for i, (k, v) in enumerate(software_lines):
            ypos = top - i * line_h
            sw_ax.text(0.39, ypos, k, fontsize=10, ha="right", va="top", transform=sw_ax.transAxes)
            sw_ax.text(0.415, ypos, ":", fontsize=10, ha="center", va="top", transform=sw_ax.transAxes)
            sw_ax.text(0.445, ypos, v, fontsize=10, ha="left", va="top", transform=sw_ax.transAxes)

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

    _draw_section_heading(
        fig,
        x=0.07,
        y=0.895,
        w=0.86,
        h=0.045,
        number=section_no,
        title=section_title,
    )

    intro_ax = fig.add_axes((0.10, 0.810, 0.80, 0.070))
    intro_ax.axis("off")
    for i, line in enumerate(pre_lines):
        intro_ax.text(
            0.0,
            0.95 - i * 0.32,
            line,
            fontsize=10,
            ha="left",
            va="top",
            transform=intro_ax.transAxes,
        )

    intro_ax.text(
        0.0,
        0.05,
        "Percentile scores ranging between 0-100 for AlphaJudge interface metrics are shown in "
        "the following graphic.",
        fontsize=10,
        ha="left",
        va="bottom",
        transform=intro_ax.transAxes,
    )

    _draw_slider_panel(fig, top=0.775, height=0.56, row=row, include_overall=True)

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


def _format_residue_tick(value: float, _pos: int | None = None) -> str:
    if not math.isfinite(value):
        return ""
    v = int(round(value))
    if v >= 1000:
        text = f"{v / 1000:g}k"
        return text.replace(".0k", "k")
    return str(v)


def _pae_vmax(matrix: np.ndarray, max_error: float | None) -> float:
    if max_error is not None and math.isfinite(max_error) and max_error > 0:
        # AlphaFold DB commonly displays a 0-30 Å scale for static examples.
        if 28.0 <= max_error <= 32.5:
            return 30.0
        return float(math.ceil(max_error / 5.0) * 5.0)

    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return 30.0

    observed = float(np.nanmax(finite))
    if observed <= 32.5:
        return 30.0
    return float(math.ceil(observed / 5.0) * 5.0)


def render_pae_png(
    out_path: str | Path,
    pae_matrix: Any,
    *,
    max_error: float | None = None,
    model_label: str | None = None,
    chain_boundaries: Sequence[float] | None = None,
    figsize: tuple[float, float] = (8.0, 8.6),
    dpi: int = 200,
) -> Path | None:
    """Write a standalone AFDB-style PAE heatmap PNG.

    Used both by the scoring runner (so per-model ``pae_<model>.png`` files
    look like the in-report graphic) and indirectly by reports that embed
    the resulting PNG.
    """
    _setup_rcparams()

    try:
        matrix = pae_matrix if isinstance(pae_matrix, np.ndarray) else np.asarray(pae_matrix, dtype=float)
    except Exception as e:
        logger.error("PAE PNG: could not coerce input to array (%s)", e)
        return None
    if matrix.ndim == 3 and matrix.shape[0] == 1:
        matrix = matrix[0]
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        logger.warning("PAE PNG: matrix shape %s is not a square 2D array", matrix.shape)
        return None
    matrix = np.where(np.isfinite(matrix), matrix, np.nan)

    n_res = int(matrix.shape[0])
    vmax = _pae_vmax(matrix, max_error)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        matrix,
        cmap=_PAE_CMAP,
        vmin=0.0,
        vmax=vmax,
        origin="upper",
        interpolation="nearest",
        extent=(0.0, float(n_res), float(n_res), 0.0),
        aspect="equal",
    )
    ax.set_xlim(0.0, float(n_res))
    ax.set_ylim(float(n_res), 0.0)
    ax.set_xlabel("Scored residue", fontsize=12, labelpad=8)
    ax.set_ylabel("Aligned residue", fontsize=12, labelpad=8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_residue_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_residue_tick))
    ax.tick_params(axis="both", labelsize=10, length=3, width=0.7, colors="#111111")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#777777")

    if chain_boundaries:
        for b in chain_boundaries:
            ax.axhline(b, color="black", linewidth=0.8)
            ax.axvline(b, color="black", linewidth=0.8)

    title = "Predicted aligned error (PAE)"
    if model_label:
        title = f"{title} – {model_label}"
    ax.set_title(title, fontsize=14, pad=12)

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.10)
    ticks = np.arange(0.0, vmax + 0.1, 5.0)
    if len(ticks) > 8:
        ticks = np.linspace(0.0, vmax, 7)
    cbar.set_ticks(ticks)
    cbar.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
    cbar.ax.tick_params(labelsize=10, length=0, pad=3)
    cbar.outline.set_linewidth(0.7)
    cbar.outline.set_edgecolor("#777777")
    cbar.ax.set_xlabel("Expected position error (Ångströms)", fontsize=10, labelpad=7)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _complex_evidence_page(
    pdf: PdfPages,
    *,
    title: str,
    entry_id: str,
    section_no: str,
    row: Mapping[str, Any] | None,
    pae_path: Path | None,
    model_label: str,
    page_no: int,
    total: int,
    last: bool = False,
    complex_label: str | None = None,
) -> None:
    """One end-of-report page that combines:

    - Complex-level slider rows (confidence_score, pDockQ/mpDockQ).
      These are scalars per predicted complex/model rather than per chain
      pair, so showing them on every interface page was misleading.
    - The PAE heatmap for the same model (when a PNG is available).
    """
    fig = _new_figure()
    _add_page_header(fig, page_no=page_no, total=total, title=title, entry=entry_id)

    _draw_section_heading(
        fig,
        x=0.07,
        y=0.895,
        w=0.86,
        h=0.045,
        number=section_no,
        title="Complex-level confidence & PAE",
        show_info=False,
    )

    sub_bits: list[str] = []
    if complex_label:
        sub_bits.append(complex_label)
    if model_label:
        sub_bits.append(f"Model {model_label}")
    if sub_bits:
        sub_ax = fig.add_axes((0.10, 0.855, 0.80, 0.030))
        sub_ax.axis("off")
        sub_ax.text(
            0.5,
            0.5,
            "  •  ".join(sub_bits),
            ha="center",
            va="center",
            fontsize=10,
            color="#1f1f1f",
            transform=sub_ax.transAxes,
        )

    # Top half: complex-level slider mini-panel.
    if row is not None:
        _draw_slider_panel(
            fig,
            top=0.815,
            height=0.180,
            row=row,
            include_overall=False,
            groups=[("complex", _COMPLEX_LEVEL_FEATURES)],
        )

    # Bottom half: PAE heatmap, or a small inline note if no PNG was found.
    img_ax = fig.add_axes((0.10, 0.075, 0.80, 0.530))
    if pae_path is not None and Path(pae_path).exists():
        try:
            img = mpimg.imread(str(pae_path))
            img_ax.imshow(img)
        except Exception as e:
            img_ax.text(0.5, 0.5, f"PAE image unavailable\n({e})",
                        ha="center", va="center", fontsize=10, color="#666")
    else:
        img_ax.text(
            0.5,
            0.5,
            "No PAE heatmap available for this model.",
            ha="center",
            va="center",
            fontsize=10,
            color="#666",
        )
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

    pae_path = _find_pae_png(run_dir, best_model)
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
        + 1  # complex-level confidence + PAE evidence
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

        page_no += 1
        _complex_evidence_page(
            pdf,
            title=_REPORT_TITLE,
            entry_id=entry_id,
            section_no=str(next_section),
            row=best,
            pae_path=pae_path,
            model_label=best_model,
            page_no=page_no,
            total=total,
            last=(page_no == total),
            complex_label=run_dir.name,
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
    if max_complexes is None:
        ranked_per_page = ranked
    else:
        # Cap the number of DISTINCT complexes (not raw interface rows).
        # Walk metascore-sorted; keep every interface row whose complex is
        # among the first `max_complexes` complexes encountered.
        ranked_per_page = []
        seen_complex: set[str] = set()
        for entry in ranked:
            cname = entry[1]
            if cname in seen_complex:
                ranked_per_page.append(entry)
            elif len(seen_complex) < max_complexes:
                seen_complex.add(cname)
                ranked_per_page.append(entry)

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

    # Pick a best-row per complex for the "Per-complex evidence" section so
    # we can render one PAE+complex-level slider page per complex. Limit to
    # the same top_n the cover table shows so the aggregate PDF stays bounded.
    best_per_complex: "OrderedDict[str, tuple[float, Mapping[str, Any]]]" = OrderedDict()
    for _label, cname, _iface, score, r in ranked:
        cur = best_per_complex.get(cname)
        if cur is None or score > cur[0]:
            best_per_complex[cname] = (score, r)
    evidence_cap = top_n if max_complexes is None else min(top_n, max_complexes)
    complex_evidence = sorted(
        best_per_complex.items(),
        key=lambda kv: kv[1][0],
        reverse=True,
    )[:evidence_cap]

    total = 1 + len(ranked_per_page) + len(complex_evidence)

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
                last=False,  # not last; complex evidence pages follow
            )

        ev_page = 1 + len(ranked_per_page)
        for ev_rank, (cname, (cscore, crow)) in enumerate(complex_evidence, start=1):
            ev_page += 1
            source_dir = str(crow.get("source_dir") or "")
            model_label = str(crow.get("model_used") or "")
            pae_path = None
            if source_dir:
                pae_path = _find_pae_png(Path(source_dir), model_label)
            _complex_evidence_page(
                pdf,
                title=_REPORT_TITLE,
                entry_id=_truncate(cname, 40),
                section_no=f"{ev_rank}",
                row=crow,
                pae_path=pae_path,
                model_label=model_label,
                page_no=ev_page,
                total=total,
                last=(ev_rank == len(complex_evidence)),
                complex_label=cname,
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
