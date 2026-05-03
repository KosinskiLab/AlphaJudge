#!/usr/bin/env python3
"""Plot key findings from the human AF3 normal-gap Zernike diagnostic."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
SUMMARY_CSV = DOCS / "zernike_human_normal_gap_quick_candidate_summary.csv"
FIELD_CSV = DOCS / "zernike_human_normal_gap_quick_field_summary.csv"
OUT_BASE = DOCS / "zernike_human_normal_gap_key_findings"

SC = "interface_sc"
ATOM_GAP = "atom_gaussian__g32__o4__s1.5__mgapband__f12"
NORMAL_RATIO = "surface_normal_gap__g24__o4__s1.5__d1.5__tr3__pr2.3__mnormalgap__f12"
NORMAL_CONTACT = "surface_normal_gap__g24__o4__s1.5__d1.5__tr3__pr2.3__mnormalcontact__f12"

METHOD_LABELS = {
    SC: "SC baseline",
    ATOM_GAP: "Atom gap-band\n(best current)",
    NORMAL_RATIO: "Normal-gap\nratio",
    NORMAL_CONTACT: "Normal-gap\ncontact-gated",
}

METHOD_COLORS = {
    SC: "#7a7f87",
    ATOM_GAP: "#1f6f8b",
    NORMAL_RATIO: "#c76f38",
    NORMAL_CONTACT: "#258f67",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _bar_labels(ax, bars, *, fmt: str = "{:.2f}", dy: float = 0.015) -> None:
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for bar in bars:
        value = bar.get_height()
        va = "bottom" if value >= 0 else "top"
        offset = dy * span if value >= 0 else -dy * span
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            fmt.format(value),
            ha="center",
            va=va,
            fontsize=8,
            color="#25313b",
        )


def main() -> int:
    summary = {row["candidate_id"]: row for row in _read_csv(SUMMARY_CSV)}
    field = {row["candidate_id"]: row for row in _read_csv(FIELD_CSV)}
    methods = [SC, ATOM_GAP, NORMAL_RATIO, NORMAL_CONTACT]

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2))
    fig.patch.set_facecolor("#fbfaf6")
    for ax in axes.ravel():
        ax.set_facecolor("#fbfaf6")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#d9d5c9", linewidth=0.8, alpha=0.75)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9)

    fig.suptitle(
        "Human AF3 hard diagnostic: contact-gated Zernike fixes the normal-gap failure mode",
        fontsize=16,
        fontweight="bold",
        color="#18222d",
        y=0.98,
    )
    fig.text(
        0.5,
        0.944,
        "30 rows: 3 lowest-SC positives + 3 highest-SC negatives + 24 AF3 mixed cases. Higher is better unless noted.",
        ha="center",
        fontsize=10,
        color="#51606d",
    )

    # Panel A: global discrimination.
    ax = axes[0, 0]
    x = list(range(len(methods)))
    width = 0.36
    auroc = [_float(summary[mid], "pooled_af3_auroc") for mid in methods]
    ap = [_float(summary[mid], "pooled_af3_average_precision") for mid in methods]
    bars1 = ax.bar(
        [pos - width / 2 for pos in x],
        auroc,
        width,
        color=[METHOD_COLORS[mid] for mid in methods],
        alpha=0.92,
        label="AUROC",
    )
    bars2 = ax.bar(
        [pos + width / 2 for pos in x],
        ap,
        width,
        color=[METHOD_COLORS[mid] for mid in methods],
        alpha=0.42,
        hatch="//",
        label="Average precision",
    )
    ax.axhline(0.5, color="#8a6f2a", linestyle="--", linewidth=1.1)
    ax.text(3.45, 0.515, "random AUROC", ha="right", va="bottom", fontsize=8, color="#8a6f2a")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("A. Pos-vs-neg discrimination", loc="left", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[mid] for mid in methods])
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _bar_labels(ax, bars1)

    # Panel B: rescue rate on SC-failure positives.
    ax = axes[0, 1]
    rescue = [_float(summary[mid], "af3_failure_rescue_rate") for mid in methods]
    bars = ax.bar(x, rescue, color=[METHOD_COLORS[mid] for mid in methods], width=0.62)
    ax.set_ylim(0, 0.55)
    ax.set_ylabel("Rescue rate")
    ax.set_title("B. Rescue of low-SC positives", loc="left", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[mid] for mid in methods])
    _bar_labels(ax, bars)

    # Panel C: median score separation.
    ax = axes[1, 0]
    sep = [_float(summary[mid], "positive_minus_negative_median") for mid in methods]
    colors = ["#258f67" if value > 0 else "#b75c48" for value in sep]
    bars = ax.bar(x, sep, color=colors, width=0.62)
    ax.axhline(0, color="#25313b", linewidth=1.0)
    ax.text(-0.42, 0.066, "above 0: positives higher", ha="left", va="top", fontsize=8, color="#258f67")
    ax.text(-0.42, -0.060, "below 0: negatives higher", ha="left", va="bottom", fontsize=8, color="#b75c48")
    ax.set_ylim(-0.065, 0.075)
    ax.set_ylabel("Positive median - negative median")
    ax.set_title("C. Direction and separation of scores", loc="left", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[mid] for mid in methods])
    _bar_labels(ax, bars, fmt="{:+.3f}", dy=0.02)

    # Panel D: why contact-gating helps.
    ax = axes[1, 1]
    ratio_row = field[NORMAL_RATIO]
    contact_row = field[NORMAL_CONTACT]
    categories = [
        ("Quality ratio\nold score", "normal_gap_score_from_fields", ratio_row),
        ("Contact amount\nnew gate", "normal_gap_contact_amount", contact_row),
        ("Contact-gated\nnew score", "normal_gap_contact_score_from_fields", contact_row),
    ]
    x2 = list(range(len(categories)))
    pos_values = [_float(row, f"{field_name}_pos_median") for _, field_name, row in categories]
    neg_values = [_float(row, f"{field_name}_neg_median") for _, field_name, row in categories]
    pos_bars = ax.bar([pos - width / 2 for pos in x2], pos_values, width, color="#2878a8", label="positive median")
    neg_bars = ax.bar([pos + width / 2 for pos in x2], neg_values, width, color="#c76f38", label="negative median")
    ax.set_ylim(0, 0.70)
    ax.set_ylabel("Median component value")
    ax.set_title("D. Normal-gap ratio vs contact-gated score", loc="left", fontsize=12, fontweight="bold")
    ax.set_xticks(x2)
    ax.set_xticklabels([label for label, _, _ in categories])
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _bar_labels(ax, pos_bars)
    _bar_labels(ax, neg_bars)

    handles = [
        Patch(facecolor=METHOD_COLORS[SC], label="SC baseline"),
        Patch(facecolor=METHOD_COLORS[ATOM_GAP], label="Atom gap-band"),
        Patch(facecolor=METHOD_COLORS[NORMAL_RATIO], label="Old normal-gap ratio"),
        Patch(facecolor=METHOD_COLORS[NORMAL_CONTACT], label="Contact-gated normal-gap"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.008))
    fig.tight_layout(rect=(0.02, 0.055, 0.98, 0.92))

    for suffix in (".svg", ".png"):
        out = OUT_BASE.with_suffix(suffix)
        fig.savefig(out, dpi=220, bbox_inches="tight")
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
