#!/usr/bin/env python3
"""Plot all-organism Zernike benchmark comparisons against interface_sc."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

SC_ID = "interface_sc"
ATOM_COSINE_ID = "atom_gaussian__g32__o10__s1.5__f12"
ATOM_GAP_ID = "atom_gaussian__g32__o0__s1.5__moverlap__f12"

METHOD_LABELS = {
    SC_ID: "SC baseline",
    ATOM_COSINE_ID: "Atom Zernike cosine\n(old, per-side)",
    ATOM_GAP_ID: "Atom Gap overlap\n(new, shared grid)",
}
METHOD_ORDER = [SC_ID, ATOM_COSINE_ID, ATOM_GAP_ID]
ORGANISM_LABELS = {
    "arabidopsis": "Arabidopsis",
    "ecoli": "E. coli",
    "human": "Human",
    "yeast": "Yeast",
}
BACKEND_LABELS = {
    "af2": "AF2",
    "af3": "AF3",
}
LABEL_COLORS = {
    "positive": "#276FBF",
    "negative": "#D95D39",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def finite(values) -> list[float]:
    return [safe_float(value) for value in values if math.isfinite(safe_float(value))]


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


def method_label(candidate_id: str) -> str:
    return METHOD_LABELS.get(candidate_id, candidate_id)


def method_order(candidate_ids: list[str]) -> list[str]:
    ordered = [candidate_id for candidate_id in METHOD_ORDER if candidate_id in candidate_ids]
    ordered.extend(sorted(candidate_id for candidate_id in candidate_ids if candidate_id not in ordered))
    return ordered


def metric_matrix(
    metric_rows: list[dict[str, str]],
    *,
    candidate_ids: list[str],
    metric: str,
) -> tuple[list[str], np.ndarray]:
    scopes: list[tuple[str, str, str]] = []
    for row in metric_rows:
        scope = row["scope"]
        organism = row.get("organism", "")
        backend = row.get("backend", "")
        if scope == "cell":
            scopes.append((scope, organism, backend))
    scopes = sorted(
        set(scopes),
        key=lambda item: (
            list(ORGANISM_LABELS).index(item[1]) if item[1] in ORGANISM_LABELS else 999,
            list(BACKEND_LABELS).index(item[2]) if item[2] in BACKEND_LABELS else 999,
        ),
    )
    pooled_scopes = [("af2", "", "af2"), ("af3", "", "af3"), ("global", "", "")]
    scopes.extend([scope for scope in pooled_scopes if any(row["scope"] == scope[0] for row in metric_rows)])

    values_by_key = {}
    for row in metric_rows:
        key = (row["candidate_id"], row["scope"], row.get("organism", ""), row.get("backend", ""))
        values_by_key[key] = safe_float(row[metric])

    matrix = np.full((len(scopes), len(candidate_ids)), np.nan, dtype=float)
    row_labels = []
    for row_idx, (scope, organism, backend) in enumerate(scopes):
        if scope == "cell":
            row_labels.append(f"{ORGANISM_LABELS.get(organism, organism)} {BACKEND_LABELS.get(backend, backend.upper())}")
        elif scope == "global":
            row_labels.append("All pooled")
        else:
            row_labels.append(f"Pooled {BACKEND_LABELS.get(backend, backend.upper())}")
        for col_idx, candidate_id in enumerate(candidate_ids):
            matrix[row_idx, col_idx] = values_by_key.get((candidate_id, scope, organism, backend), np.nan)
    return row_labels, matrix


def plot_heatmap(ax, matrix: np.ndarray, row_labels: list[str], col_labels: list[str], title: str, *, cmap, norm):
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, aspect="auto", cmap=cmap, norm=norm)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_xticks(np.arange(len(col_labels)), col_labels, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(row_labels)), row_labels, fontsize=9)
    ax.tick_params(length=0)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if math.isfinite(value):
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#151515",
                )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    return image


def plot_summary_figure(bench_out_dir: Path, out_prefix: Path) -> None:
    summary_rows = read_csv_rows(bench_out_dir / "candidate_summary.csv")
    metric_rows = read_csv_rows(bench_out_dir / "candidate_metrics.csv")
    candidate_ids = method_order(sorted({row["candidate_id"] for row in summary_rows}))
    col_labels = [method_label(candidate_id) for candidate_id in candidate_ids]

    row_labels, auroc = metric_matrix(metric_rows, candidate_ids=candidate_ids, metric="auroc")
    _, ap = metric_matrix(metric_rows, candidate_ids=candidate_ids, metric="average_precision")

    summary_by_id = {row["candidate_id"]: row for row in summary_rows}
    rescue = [safe_float(summary_by_id.get(candidate_id, {}).get("af3_failure_rescue_rate", float("nan"))) for candidate_id in candidate_ids]
    af3_auroc_delta = [
        safe_float(summary_by_id.get(candidate_id, {}).get("delta_af3_auroc_vs_sc", float("nan")))
        for candidate_id in candidate_ids
    ]
    all_auroc_delta = [
        safe_float(summary_by_id.get(candidate_id, {}).get("delta_all_auroc_vs_sc", float("nan")))
        for candidate_id in candidate_ids
    ]

    fig = plt.figure(figsize=(13.8, 9.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0])
    ax_auroc = fig.add_subplot(grid[0, 0])
    ax_ap = fig.add_subplot(grid[0, 1])
    ax_rescue = fig.add_subplot(grid[1, 0])
    ax_delta = fig.add_subplot(grid[1, 1])

    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    image = plot_heatmap(
        ax_auroc,
        auroc,
        row_labels,
        col_labels,
        "A. AUROC by organism/backend",
        cmap="RdYlGn",
        norm=norm,
    )
    cbar = fig.colorbar(image, ax=ax_auroc, shrink=0.82)
    cbar.set_label("AUROC, higher is better; 0.5 = random", fontsize=9)

    image = plot_heatmap(
        ax_ap,
        ap,
        row_labels,
        col_labels,
        "B. Average precision by organism/backend",
        cmap="YlGnBu",
        norm=None,
    )
    cbar = fig.colorbar(image, ax=ax_ap, shrink=0.82)
    cbar.set_label("Average precision, higher is better", fontsize=9)

    x = np.arange(len(candidate_ids))
    colors = ["#5B5B5B", "#9E9E9E", "#2A9D8F"][: len(candidate_ids)]
    ax_rescue.bar(x, rescue, color=colors, width=0.65)
    ax_rescue.set_title("C. AF3 low-SC positive rescue rate", loc="left", fontsize=12, fontweight="bold")
    ax_rescue.set_xticks(x, col_labels, rotation=20, ha="right", fontsize=9)
    ax_rescue.set_ylabel("Fraction rescued above candidate negative p90")
    ax_rescue.set_ylim(0.0, max(0.05, np.nanmax(rescue) * 1.2 if np.any(np.isfinite(rescue)) else 1.0))
    ax_rescue.grid(axis="y", color="#D8D8D8", linewidth=0.8)
    ax_rescue.spines[["top", "right"]].set_visible(False)
    for idx, value in enumerate(rescue):
        if math.isfinite(value):
            ax_rescue.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    width = 0.35
    ax_delta.axhline(0.0, color="#333333", linewidth=1.0)
    ax_delta.bar(x - width / 2, all_auroc_delta, width=width, color="#577590", label="All pooled")
    ax_delta.bar(x + width / 2, af3_auroc_delta, width=width, color="#F3722C", label="AF3 pooled")
    ax_delta.set_title("D. AUROC delta versus SC", loc="left", fontsize=12, fontweight="bold")
    ax_delta.set_xticks(x, col_labels, rotation=20, ha="right", fontsize=9)
    ax_delta.set_ylabel("Delta AUROC")
    ax_delta.grid(axis="y", color="#D8D8D8", linewidth=0.8)
    ax_delta.legend(frameon=False, fontsize=9)
    ax_delta.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "All-organism benchmark: SC versus low-pass Zernike interface scores",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.01,
        "Direction: higher score means more interaction-like. New Atom Gap uses shared-grid low-resolution overlap; old cosine is the saturated per-side descriptor diagnostic.",
        fontsize=9,
        color="#333333",
    )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_name(out_prefix.name + "_metrics.svg"))
    fig.savefig(out_prefix.with_name(out_prefix.name + "_metrics.png"), dpi=220)
    plt.close(fig)


def score_rows_by_candidate(bench_out_dir: Path, candidate_ids: list[str]) -> dict[str, list[dict[str, str]]]:
    out = {}
    for candidate_id in candidate_ids:
        path = bench_out_dir / "scores" / f"{candidate_id}.csv"
        if path.exists():
            out[candidate_id] = read_csv_rows(path)
    return out


def distribution_rows(score_rows: dict[str, list[dict[str, str]]]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for candidate_id, rows in score_rows.items():
        for row in rows:
            score = safe_float(row.get("candidate_score"))
            if not math.isfinite(score):
                continue
            key = (candidate_id, row["organism"], row["backend"], row["label"])
            grouped[key].append(score)

    out = []
    for (candidate_id, organism, backend, label), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=float)
        out.append(
            {
                "candidate_id": candidate_id,
                "method": method_label(candidate_id).replace("\n", " "),
                "organism": organism,
                "backend": backend,
                "label": label,
                "n": len(arr),
                "q25": float(np.quantile(arr, 0.25)),
                "median": float(np.median(arr)),
                "q75": float(np.quantile(arr, 0.75)),
                "mean": float(np.mean(arr)),
            }
        )
    return out


def plot_distribution_figure(bench_out_dir: Path, out_prefix: Path) -> None:
    summary_rows = read_csv_rows(bench_out_dir / "candidate_summary.csv")
    candidate_ids = method_order(sorted({row["candidate_id"] for row in summary_rows}))
    scores = score_rows_by_candidate(bench_out_dir, candidate_ids)
    dist_rows = distribution_rows(scores)
    write_csv(out_prefix.with_name(out_prefix.name + "_score_distribution_summary.csv"), dist_rows)

    cell_order = []
    for organism in ORGANISM_LABELS:
        for backend in BACKEND_LABELS:
            cell_order.append((organism, backend))
    cell_labels = [
        f"{ORGANISM_LABELS[organism]}\n{BACKEND_LABELS[backend]}"
        for organism, backend in cell_order
    ]

    fig, axes = plt.subplots(
        len(candidate_ids),
        1,
        figsize=(13.8, 3.25 * len(candidate_ids)),
        sharex=True,
        constrained_layout=True,
    )
    if len(candidate_ids) == 1:
        axes = [axes]

    dist_by_key = {
        (row["candidate_id"], row["organism"], row["backend"], row["label"]): row
        for row in dist_rows
    }

    for ax, candidate_id in zip(axes, candidate_ids):
        x = np.arange(len(cell_order), dtype=float)
        for label, offset, marker in [("negative", -0.12, "v"), ("positive", 0.12, "^")]:
            medians = []
            lows = []
            highs = []
            positions = []
            for idx, (organism, backend) in enumerate(cell_order):
                row = dist_by_key.get((candidate_id, organism, backend, label))
                if not row:
                    continue
                median = safe_float(row["median"])
                q25 = safe_float(row["q25"])
                q75 = safe_float(row["q75"])
                positions.append(idx + offset)
                medians.append(median)
                lows.append(median - q25)
                highs.append(q75 - median)
            if medians:
                ax.errorbar(
                    positions,
                    medians,
                    yerr=np.vstack([lows, highs]),
                    fmt=marker,
                    markersize=7,
                    linewidth=1.5,
                    capsize=3,
                    color=LABEL_COLORS[label],
                    label=label.capitalize(),
                )
        ax.set_title(method_label(candidate_id), loc="left", fontsize=12, fontweight="bold")
        ax.set_ylabel("Raw score\nhigher = more interacting")
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, ncol=2, loc="upper right")

    axes[-1].set_xticks(np.arange(len(cell_order)), cell_labels, fontsize=9)
    fig.suptitle(
        "Score separation across all organisms and AlphaFold backends",
        fontsize=15,
        fontweight="bold",
    )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_name(out_prefix.name + "_score_distributions.svg"))
    fig.savefig(out_prefix.with_name(out_prefix.name + "_score_distributions.png"), dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-out-dir", required=True, type=Path)
    parser.add_argument("--out-prefix", default="docs/zernike_all_organisms_atom_gap", type=Path)
    args = parser.parse_args()

    plot_summary_figure(args.bench_out_dir, args.out_prefix)
    plot_distribution_figure(args.bench_out_dir, args.out_prefix)
    print(f"wrote {args.out_prefix.with_name(args.out_prefix.name + '_metrics.svg')}")
    print(f"wrote {args.out_prefix.with_name(args.out_prefix.name + '_score_distributions.svg')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
