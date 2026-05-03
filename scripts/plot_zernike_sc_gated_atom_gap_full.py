#!/usr/bin/env python3
"""Plot the full benchmark result for SC-gated atom-gap rescue."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
HYBRID_SUMMARY = DOCS / "zernike_sc_gated_atom_gap_tuned_full_candidate_summary.csv"
HYBRID_METRICS = DOCS / "zernike_sc_gated_atom_gap_tuned_full_candidate_metrics.csv"
PURE_SUMMARY = DOCS / "zernike_all_organisms_atom_gap_penalty_candidate_summary.csv"
PURE_METRICS = DOCS / "zernike_all_organisms_atom_gap_penalty_candidate_metrics.csv"
OUT_BASE = DOCS / "zernike_sc_gated_atom_gap_full_comparison"

SC_ID = "interface_sc"
PURE_ID = "atom_gaussian__g32__o4__s1.5__mgapband__f12"
HYBRID_ID = "atom_gaussian__g32__o0__s1.5__mscoverlap__rs0.4__rf0.01__f12"

METHODS = [
    (SC_ID, "SC", "#6f7378"),
    (PURE_ID, "Pure atom gap\nbest old", "#1f6f8b"),
    (HYBRID_ID, "SC-gated\natom gap", "#258f67"),
]
ORGANISMS = ("arabidopsis", "ecoli", "human", "yeast")
ORG_LABELS = {
    "arabidopsis": "Arabidopsis",
    "ecoli": "E. coli",
    "human": "Human",
    "yeast": "Yeast",
}
BACKENDS = ("af2", "af3")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def row_by_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["candidate_id"]: row for row in rows}


def metric_map(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], dict[str, str]]:
    return {
        (row["candidate_id"], row["scope"], row.get("organism", ""), row.get("backend", "")): row
        for row in rows
    }


def main() -> int:
    hybrid_summary = row_by_id(read_csv_rows(HYBRID_SUMMARY))
    pure_summary = row_by_id(read_csv_rows(PURE_SUMMARY))
    hybrid_metrics = metric_map(read_csv_rows(HYBRID_METRICS))
    pure_metrics = metric_map(read_csv_rows(PURE_METRICS))

    summary_by_id = {
        SC_ID: hybrid_summary[SC_ID],
        PURE_ID: pure_summary[PURE_ID],
        HYBRID_ID: hybrid_summary[HYBRID_ID],
    }
    metrics_by_id = {
        SC_ID: hybrid_metrics,
        PURE_ID: pure_metrics,
        HYBRID_ID: hybrid_metrics,
    }

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.6), constrained_layout=True)
    fig.patch.set_facecolor("#fbfaf6")
    for ax in axes.ravel():
        ax.set_facecolor("#fbfaf6")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#d8d3c7", linewidth=0.8)

    # A: pooled AUROC by backend.
    ax = axes[0, 0]
    x = np.arange(len(METHODS))
    width = 0.24
    for offset, scope, backend, label, color in [
        (-width, "af2", "af2", "AF2", "#577590"),
        (0.0, "af3", "af3", "AF3", "#f3722c"),
        (width, "global", "", "All", "#43aa8b"),
    ]:
        values = [
            safe_float(metrics_by_id[candidate_id].get((candidate_id, scope, "", backend), {}).get("auroc"))
            for candidate_id, _, _ in METHODS
        ]
        ax.bar(x + offset, values, width=width, color=color, label=label)
        for xpos, value in zip(x + offset, values):
            if math.isfinite(value):
                ax.text(xpos, value + 0.008, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, color="#555", linestyle="--", linewidth=1.0)
    ax.set_ylim(0.42, 0.84)
    ax.set_ylabel("AUROC")
    ax.set_title("A. Pooled discrimination", loc="left", fontweight="bold")
    ax.set_xticks(x, [label for _, label, _ in METHODS])
    ax.legend(frameon=False, ncol=3, fontsize=9)

    # B: rescue rate and AP.
    ax = axes[0, 1]
    rescue = [safe_float(summary_by_id[candidate_id].get("af3_failure_rescue_rate")) for candidate_id, _, _ in METHODS]
    bars = ax.bar(x, rescue, color=[color for _, _, color in METHODS], width=0.58)
    ax.set_ylim(0, max(rescue) * 1.35)
    ax.set_ylabel("Rescue rate")
    ax.set_title("B. AF3 low-SC positives rescued", loc="left", fontweight="bold")
    ax.set_xticks(x, [label for _, label, _ in METHODS])
    for bar, value in zip(bars, rescue):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.006, f"{100 * value:.0f}%", ha="center", fontsize=9)

    # C: per-cell AUROC delta versus SC.
    ax = axes[1, 0]
    rows = []
    row_labels = []
    for organism in ORGANISMS:
        for backend in BACKENDS:
            row_labels.append(f"{ORG_LABELS[organism]}\n{backend.upper()}")
            pure_delta = safe_float(
                pure_metrics.get((PURE_ID, "cell", organism, backend), {}).get("delta_auroc_vs_sc")
            )
            hybrid_delta = safe_float(
                hybrid_metrics.get((HYBRID_ID, "cell", organism, backend), {}).get("delta_auroc_vs_sc")
            )
            rows.append([pure_delta, hybrid_delta])
    matrix = np.asarray(rows, dtype=float)
    image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-0.36, vmax=0.12)
    ax.set_title("C. Per-cell AUROC delta vs SC", loc="left", fontweight="bold")
    ax.set_xticks([0, 1], ["Pure atom gap", "SC-gated atom gap"])
    ax.set_yticks(np.arange(len(row_labels)), row_labels, fontsize=9)
    ax.tick_params(length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, f"{value:+.02f}", ha="center", va="center", fontsize=9, color="#111")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, shrink=0.82, label="Delta AUROC")

    # D: score direction.
    ax = axes[1, 1]
    med_sep = [
        safe_float(summary_by_id[candidate_id].get("positive_minus_negative_median"))
        for candidate_id, _, _ in METHODS
    ]
    colors = ["#258f67" if value >= 0 else "#b75c48" for value in med_sep]
    bars = ax.bar(x, med_sep, color=colors, width=0.58)
    ax.axhline(0, color="#333", linewidth=1.0)
    ax.set_ylabel("Positive median - negative median")
    ax.set_title("D. Score direction", loc="left", fontweight="bold")
    ax.set_xticks(x, [label for _, label, _ in METHODS])
    for bar, value in zip(bars, med_sep):
        va = "bottom" if value >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:+.3f}", ha="center", va=va, fontsize=9)

    fig.suptitle(
        "SC-gated atom-gap rescue beats SC on full benchmark AUROC while preserving AF2",
        fontsize=15,
        fontweight="bold",
    )
    OUT_BASE.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".svg", ".png"):
        out = OUT_BASE.with_suffix(suffix)
        fig.savefig(out, dpi=240, bbox_inches="tight")
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
