#!/usr/bin/env python3
"""Plot atom gap-band Zernike order sweeps from a benchmark output directory."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SC_ID = "interface_sc"
ATOM_GAUSSIAN = "atom_gaussian"
SCORE_MODES = (
    "gap_zernike_bandpass",
    "gap_zernike_excess_bandpass",
    "gap_zernike_soft_bandpass",
    "gap_zernike_excess_contact_bandpass",
)
MODE_LABELS = {
    "gap_zernike_bandpass": "Hard bandpass",
    "gap_zernike_excess_bandpass": "Excess bandpass",
    "gap_zernike_soft_bandpass": "Soft bandpass",
    "gap_zernike_excess_contact_bandpass": "Contact-gated excess",
}
MODE_COLORS = {
    "gap_zernike_bandpass": "#1f6f8b",
    "gap_zernike_excess_bandpass": "#258f67",
    "gap_zernike_soft_bandpass": "#c76f38",
    "gap_zernike_excess_contact_bandpass": "#8a5fbf",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    seen = set(fieldnames)
    for row in rows[1:]:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _rank_value(row: dict[str, str]) -> tuple[float, float, float, float]:
    rescue = safe_float(row.get("af3_failure_rescue_rate"))
    af3 = safe_float(row.get("pooled_af3_auroc"))
    all_auroc = safe_float(row.get("pooled_all_auroc"))
    runtime = safe_float(row.get("median_runtime_sec"))
    return (
        rescue if math.isfinite(rescue) else -1.0,
        af3 if math.isfinite(af3) else -1.0,
        all_auroc if math.isfinite(all_auroc) else -1.0,
        -(runtime if math.isfinite(runtime) else float("inf")),
    )


def best_atom_gap_rows(summary_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best: dict[tuple[str, int], dict[str, str]] = {}
    for row in summary_rows:
        if row.get("representation") != ATOM_GAUSSIAN:
            continue
        score_mode = row.get("score_mode", "")
        if score_mode not in SCORE_MODES:
            continue
        order = safe_float(row.get("order"))
        if not math.isfinite(order):
            continue
        key = (score_mode, int(order))
        if key not in best or _rank_value(row) > _rank_value(best[key]):
            best[key] = row
    return [best[key] for key in sorted(best, key=lambda item: (SCORE_MODES.index(item[0]), item[1]))]


def _plot_metric(
    ax,
    rows: list[dict[str, str]],
    metric: str,
    title: str,
    ylabel: str,
    *,
    sc_value: float | None = None,
    ylim: tuple[float, float] | None = None,
):
    for score_mode in SCORE_MODES:
        mode_rows = [row for row in rows if row.get("score_mode") == score_mode]
        if not mode_rows:
            continue
        mode_rows.sort(key=lambda row: safe_float(row.get("order")))
        x = [safe_float(row["order"]) for row in mode_rows]
        y = [safe_float(row.get(metric)) for row in mode_rows]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.2,
            markersize=5,
            color=MODE_COLORS[score_mode],
            label=MODE_LABELS[score_mode],
        )
    if sc_value is not None and math.isfinite(sc_value):
        ax.axhline(sc_value, color="#5f6368", linestyle="--", linewidth=1.3)
        ax.text(10.15, sc_value, "SC", va="center", ha="left", fontsize=8, color="#5f6368")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("Zernike order N")
    ax.set_ylabel(ylabel)
    ax.set_xticks([2, 3, 4, 5, 6, 8, 10])
    ax.set_xlim(1.6, 10.4)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#d6d1c6", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def plot_order_curve(bench_out_dir: Path, out_prefix: Path) -> None:
    summary_rows = read_csv_rows(bench_out_dir / "candidate_summary.csv")
    summary_by_id = {row["candidate_id"]: row for row in summary_rows}
    sc_row = summary_by_id.get(SC_ID, {})
    rows = best_atom_gap_rows(summary_rows)
    write_csv(out_prefix.with_name(out_prefix.name + "_selected_candidates.csv"), rows)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), constrained_layout=True)
    fig.patch.set_facecolor("#fbfaf6")
    for ax in axes.ravel():
        ax.set_facecolor("#fbfaf6")

    _plot_metric(
        axes[0, 0],
        rows,
        "pooled_af3_auroc",
        "A. AF3 discrimination",
        "Pooled AF3 AUROC",
        sc_value=safe_float(sc_row.get("pooled_af3_auroc")),
        ylim=(0.0, 0.74),
    )
    _plot_metric(
        axes[0, 1],
        rows,
        "af3_failure_rescue_rate",
        "B. Low-SC AF3 positive rescue",
        "Rescue rate",
        sc_value=safe_float(sc_row.get("af3_failure_rescue_rate")),
        ylim=(-0.01, 0.34),
    )
    _plot_metric(
        axes[1, 0],
        rows,
        "pooled_all_auroc",
        "C. Global guardrail",
        "Pooled all-data AUROC",
        sc_value=safe_float(sc_row.get("pooled_all_auroc")),
        ylim=(0.0, 0.74),
    )
    _plot_metric(
        axes[1, 1],
        rows,
        "positive_minus_negative_median",
        "D. Score direction",
        "Positive median - negative median",
        sc_value=safe_float(sc_row.get("positive_minus_negative_median")),
        ylim=(-0.04, 0.05),
    )
    axes[0, 0].legend(frameon=False, fontsize=9, loc="best")

    fig.suptitle(
        "Atom gap-band Zernike order sweep on the AF3 hard diagnostic",
        fontsize=15,
        fontweight="bold",
    )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".svg"))
    fig.savefig(out_prefix.with_suffix(".png"), dpi=240)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-out-dir", required=True, type=Path)
    parser.add_argument("--out-prefix", default="docs/zernike_atom_gap_order_curve", type=Path)
    args = parser.parse_args()
    plot_order_curve(args.bench_out_dir, args.out_prefix)
    print(args.out_prefix.with_suffix(".svg"))
    print(args.out_prefix.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
