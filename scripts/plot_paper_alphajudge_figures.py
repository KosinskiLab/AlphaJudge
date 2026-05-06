#!/usr/bin/env python3
"""Generate paper-ready AlphaJudge manuscript figures from benchmark_26 outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from scipy.stats import rankdata

from alphajudge.meta_score import (
    CALIBRATION_LEVELS,
    FEATURE_DIRECTIONS,
    META_SCORE_FEATURES,
    interface_meta_score,
)


BENCHMARK_ROOT = Path("/g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26")
LATEST_MERGED = BENCHMARK_ROOT / "merged_best_interfaces_all_models.20260422_111918_904fef5.csv"
IPSAE_CUTOFF_SUMMARY = BENCHMARK_ROOT / "ipsae_scan" / "roc" / "roc_summary_by_cutoff.csv"
CLASSIFIER_SUMMARY = (
    BENCHMARK_ROOT / "classifier" / "clf_out_repeats_all" / "auc_summary_all_repeats.csv"
)
BENCHMARK_EXPORTS = BENCHMARK_ROOT / "data" / "exports"
BENCHMARK_CACHE = BENCHMARK_ROOT / "data" / "cache"
IPSAE_SUMMARY_DIR = BENCHMARK_ROOT / "ipsae_scan" / "summaries"
PULLDOWN_ROOT = Path("/g/transform/kosinski/dima/IntAct_BioGRID_STRING/AlphaPulldownSnakemake")
PULLDOWN_REPORT = (
    PULLDOWN_ROOT / "human" / "nonparalog_strong_weak_1000_af2" / "reports" / "all_interfaces.csv"
)
PULLDOWN_PANELS = {
    "strong": {
        "label": "Strong FNTA-FNTB",
        "positive": "P49354+P49356",
        "labels": PULLDOWN_ROOT
        / "config"
        / "pulldown_nonparalog_strong_weak_seed26"
        / "strong_FNTA_FNTB"
        / "target1000_labels.tsv",
    },
    "weak": {
        "label": "Weak TAPBP-CALR",
        "positive": "O15533+P27797",
        "labels": PULLDOWN_ROOT
        / "config"
        / "pulldown_nonparalog_strong_weak_seed26"
        / "weak_TAPBP_CALR"
        / "target1000_labels.tsv",
    },
}
DEFAULT_OUTDIR = Path("docs/figures")
HARMONIZATION_SEED = "alphajudge-paper-coverage-harmonized-20260505"

OKABE_ITO = {
    "black": "#111827",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

POS_COLOR = OKABE_ITO["blue"]
NEG_COLOR = "#9BA3AF"
CONF_COLOR = OKABE_ITO["blue"]
GEOM_COLOR = OKABE_ITO["green"]
TEXT_COLOR = "#202333"
GRID_COLOR = "#D8DDE5"

SCORE_COLORS = {
    "interface_meta_score": OKABE_ITO["black"],
    "interface_LIS": OKABE_ITO["blue"],
    "interface_ipSAE": OKABE_ITO["green"],
    "interface_pDockQ2": OKABE_ITO["vermillion"],
    "iptm": OKABE_ITO["purple"],
    "average_interface_pae": OKABE_ITO["sky"],
    "interface_score": OKABE_ITO["orange"],
    "interface_sc": OKABE_ITO["purple"],
    "interface_area": OKABE_ITO["green"],
    "interface_solv_en": OKABE_ITO["vermillion"],
}
SCORE_LINESTYLES = {
    "interface_meta_score": "-",
    "interface_LIS": "--",
    "interface_ipSAE": "-.",
    "interface_pDockQ2": ":",
    "interface_sc": (0, (5, 1.5)),
}
SCORE_HATCHES = {
    "interface_meta_score": "",
    "interface_LIS": "///",
    "interface_ipSAE": "\\\\\\",
    "interface_pDockQ2": "...",
    "interface_sc": "xx",
}
ORGANISM_COLORS = {
    "human": OKABE_ITO["blue"],
    "yeast": OKABE_ITO["purple"],
    "arabidopsis": OKABE_ITO["green"],
    "ecoli": OKABE_ITO["vermillion"],
}

ORGANISM_LABELS = {
    "arabidopsis": "A. thaliana",
    "ecoli": "E. coli",
    "human": "H. sapiens",
    "yeast": "S. cerevisiae",
}

SOURCE_COLORS = {
    "IntAct": OKABE_ITO["blue"],
    "BioGRID": OKABE_ITO["green"],
    "STRING": OKABE_ITO["orange"],
}

SCORE_FEATURES = [
    ("interface_meta_score", "Meta", True, "aggregate"),
    ("interface_LIS", "LIS", True, "PAE-derived"),
    ("interface_ipSAE", "ipSAE", True, "PAE-derived"),
    ("interface_pDockQ2", "pDockQ2", True, "PAE-derived"),
    ("iptm", "ipTM", True, "confidence"),
    ("average_interface_pae", "avg interface PAE", False, "PAE-derived"),
    ("iptm_ptm", "ipTM+PTM", True, "confidence"),
    ("confidence_score", "native rank score", True, "confidence"),
    ("pDockQ/mpDockQ", "pDockQ/mpDockQ", True, "PAE-derived"),
    ("interface_score", "interface score", True, "composite"),
    ("interface_sc", "SC", True, "biophysical"),
    ("interface_solv_en", "solvation", False, "biophysical"),
    ("interface_hb", "H-bonds", True, "biophysical"),
    ("interface_area", "area", True, "biophysical"),
    ("interface_sb", "salt bridges", True, "biophysical"),
]

HEATMAP_FEATURES = [
    ("interface_meta_score", "Meta", True),
    ("interface_LIS", "LIS", True),
    ("interface_ipSAE", "ipSAE", True),
    ("interface_pDockQ2", "pDockQ2", True),
    ("iptm", "ipTM", True),
    ("interface_sc", "SC", True),
    ("interface_area", "Area", True),
    ("interface_solv_en", "Solv.", False),
]

PULLDOWN_FEATURES = [
    ("interface_meta_score", "Meta", True),
    ("interface_LIS", "LIS", True),
    ("interface_ipSAE", "ipSAE", True),
    ("interface_pDockQ2", "pDockQ2", True),
    ("iptm", "ipTM", True),
    ("average_interface_pae", "avg PAE", False),
    ("interface_score", "interface score", True),
    ("interface_sc", "SC", True),
    ("interface_area", "area", True),
    ("interface_solv_en", "solvation", False),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def stable_row_digest(row: dict[str, str]) -> str:
    parts = [
        HARMONIZATION_SEED,
        row.get("organism", ""),
        row.get("label", ""),
        row.get("model", ""),
        row.get("jobs", ""),
        row.get("model_used", ""),
        row.get("interface", ""),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def harmonize_backend_coverage(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, int | str]]]:
    """Downsample the higher-coverage backend within each organism/label cell."""
    grouped: dict[tuple[str, str], dict[str, list[tuple[int, dict[str, str]]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for idx, row in enumerate(rows):
        grouped[(row["organism"], row["label"])][row["model"]].append((idx, row))

    keep_indices: set[int] = set()
    audit_records: list[dict[str, int | str]] = []
    for organism in ["arabidopsis", "ecoli", "human", "yeast"]:
        for label in ["positive", "negative"]:
            model_groups = grouped[(organism, label)]
            raw_counts = {model: len(model_groups.get(model, [])) for model in ["af2", "af3"]}
            keep_n = min(raw_counts.values())
            for model in ["af2", "af3"]:
                candidates = sorted(
                    model_groups.get(model, []),
                    key=lambda item: stable_row_digest(item[1]),
                )
                selected = candidates[:keep_n]
                keep_indices.update(idx for idx, _ in selected)
                audit_records.append(
                    {
                        "organism": organism,
                        "label": label,
                        "model": model,
                        "raw_count": raw_counts[model],
                        "kept_count": keep_n,
                        "dropped_count": raw_counts[model] - keep_n,
                    }
                )

    return [row for idx, row in enumerate(rows) if idx in keep_indices], audit_records


def safe_float(value) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def labels(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([1 if row["label"] == "positive" else 0 for row in rows], dtype=int)


def values(rows: list[dict[str, str]], column: str) -> np.ndarray:
    if column == "interface_meta_score":
        return np.asarray([interface_meta_score(row) for row in rows], dtype=float)
    return np.asarray([safe_float(row.get(column)) for row in rows], dtype=float)


def average_ranks(x: np.ndarray) -> np.ndarray:
    return rankdata(x, method="average")


def auc_score(y: np.ndarray, scores: np.ndarray, *, higher_is_better: bool = True) -> float:
    scores = scores if higher_is_better else -scores
    mask = np.isfinite(scores)
    y = y[mask]
    scores = scores[mask]
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(scores)
    sum_pos = float(ranks[y == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(y: np.ndarray, scores: np.ndarray, *, higher_is_better: bool = True) -> float:
    scores = scores if higher_is_better else -scores
    mask = np.isfinite(scores)
    y = y[mask]
    scores = scores[mask]
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ranked_y = y[order]
    ranked_scores = scores[order]
    change_points = np.r_[np.flatnonzero(np.diff(ranked_scores)) + 1, len(ranked_scores)]
    prev_end = 0
    cum_pos = 0
    ap = 0.0
    for end in change_points:
        group_pos = int(ranked_y[prev_end:end].sum())
        if group_pos:
            new_cum_pos = cum_pos + group_pos
            precision_at_threshold = new_cum_pos / end
            ap += (group_pos / n_pos) * precision_at_threshold
            cum_pos = new_cum_pos
        prev_end = end
    return float(ap)


def precision_recall_curve_points(
    y: np.ndarray, scores: np.ndarray, *, higher_is_better: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    scores = scores if higher_is_better else -scores
    mask = np.isfinite(scores)
    y = y[mask]
    scores = scores[mask]
    n_pos = int(y.sum())
    if n_pos == 0:
        return np.asarray([1.0]), np.asarray([0.0])
    order = np.argsort(-scores, kind="mergesort")
    ranked_y = y[order]
    ranked_scores = scores[order]
    change_points = np.r_[np.flatnonzero(np.diff(ranked_scores)) + 1, len(ranked_scores)]
    cum_pos = np.cumsum(ranked_y)[change_points - 1]
    precision = cum_pos / change_points
    recall = cum_pos / n_pos
    return precision, recall


def top_fraction_metrics(
    y: np.ndarray,
    scores: np.ndarray,
    fraction: float,
    *,
    higher_is_better: bool = True,
) -> tuple[float, float, float]:
    scores = scores if higher_is_better else -scores
    mask = np.isfinite(scores)
    y = y[mask]
    scores = scores[mask]
    if len(y) == 0:
        return float("nan"), float("nan"), float("nan")
    k = max(1, int(math.ceil(len(y) * fraction)))
    order = np.argsort(-scores, kind="mergesort")[:k]
    precision = float(y[order].mean())
    recall = float(y[order].sum() / max(y.sum(), 1))
    prevalence = float(y.mean())
    enrichment = precision / prevalence if prevalence else float("nan")
    return precision, recall, enrichment


def stratified_bootstrap_metrics(
    y: np.ndarray,
    scores: np.ndarray,
    *,
    higher_is_better: bool = True,
    n_bootstrap: int = 500,
    seed: int = 7,
) -> dict[str, float]:
    oriented = scores if higher_is_better else -scores
    mask = np.isfinite(oriented)
    y = y[mask]
    oriented = oriented[mask]
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return {
            "auroc": float("nan"),
            "auroc_ci_low": float("nan"),
            "auroc_ci_high": float("nan"),
            "ap": float("nan"),
            "ap_ci_low": float("nan"),
            "ap_ci_high": float("nan"),
            "n": int(len(y)),
        }

    rng = np.random.default_rng(seed)
    auc_values: list[float] = []
    ap_values: list[float] = []
    for _ in range(n_bootstrap):
        sample_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        sample_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        sample_idx = np.concatenate([sample_pos, sample_neg])
        yy = y[sample_idx]
        ss = oriented[sample_idx]
        auc_values.append(auc_score(yy, ss, higher_is_better=True))
        ap_values.append(average_precision(yy, ss, higher_is_better=True))

    return {
        "auroc": auc_score(y, oriented, higher_is_better=True),
        "auroc_ci_low": float(np.nanpercentile(auc_values, 2.5)),
        "auroc_ci_high": float(np.nanpercentile(auc_values, 97.5)),
        "ap": average_precision(y, oriented, higher_is_better=True),
        "ap_ci_low": float(np.nanpercentile(ap_values, 2.5)),
        "ap_ci_high": float(np.nanpercentile(ap_values, 97.5)),
        "n": int(len(y)),
    }


def format_ci(center: float, low: float, high: float) -> str:
    return f"{center:.3f} [{low:.3f}, {high:.3f}]"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": "#3B3E47",
            "axes.linewidth": 0.8,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig, outdir: Path, stem: str) -> None:
    """Save reviewer-facing raster and vector versions of a figure."""
    fig.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")


def add_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    fc: str,
    ec: str,
    fontsize: int = 10,
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=1.4,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.18,
    )


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    rad: float = 0.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color="#3B3E47",
            shrinkA=6,
            shrinkB=6,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def plot_flowchart(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.4, 6.1))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    blue_fc, blue_ec = "#E8F1FA", "#477CA8"
    green_fc, green_ec = "#EAF5EA", "#4F8A58"
    orange_fc, orange_ec = "#FFF1DD", "#B97828"
    gray_fc, gray_ec = "#F3F4F6", "#6B717C"

    purple_fc, purple_ec = "#F8F2F8", "#8A5D8E"

    boxes = {
        "intact": ((0.04, 0.76), 0.15, 0.11, "IntAct\nphysical PPIs", blue_fc, blue_ec, 10),
        "biogrid": ((0.04, 0.60), 0.15, 0.11, "BioGRID\nphysical PPIs", blue_fc, blue_ec, 10),
        "string": ((0.04, 0.44), 0.15, 0.11, "STRING\nphysical links", blue_fc, blue_ec, 10),
        "canon": ((0.27, 0.58), 0.18, 0.14, "Canonicalize\nUniProt IDs", gray_fc, gray_ec, 10),
        "pos": ((0.53, 0.76), 0.20, 0.14, "Three-way ALL3\npositive pairs", green_fc, green_ec, 10),
        "hub": ((0.79, 0.76), 0.18, 0.14, "Hubness panels\nwith and without\nhub filtering", green_fc, green_ec, 8),
        "union": ((0.53, 0.55), 0.20, 0.14, "Union blacklist\nall observed pairs", orange_fc, orange_ec, 10),
        "neg": ((0.53, 0.34), 0.20, 0.14, "Same-protein\nre-pairings\nblacklist excluded", purple_fc, purple_ec, 9),
        "score": ((0.79, 0.48), 0.18, 0.16, "AF2/AF3 prediction\nbest model\nAlphaJudge scores", gray_fc, gray_ec, 8),
        "eval": ((0.79, 0.26), 0.18, 0.13, "AUROC, AP,\ntop-k,\nbootstrap CIs", gray_fc, gray_ec, 9),
        "pulldown": ((0.27, 0.14), 0.23, 0.15, "Sparse pulldown\nFNTA-FNTB / TAPBP-CALR\n1 positive + 1000 negatives", purple_fc, purple_ec, 8),
        "pd_eval": ((0.56, 0.14), 0.19, 0.15, "AF2 pulldown\nrank audit", purple_fc, purple_ec, 10),
    }
    for xy, width, height, label, fc, ec, fontsize in boxes.values():
        add_box(ax, xy, width, height, label, fc=fc, ec=ec, fontsize=fontsize)

    def left(name: str) -> tuple[float, float]:
        (x, y), _, h, *_ = boxes[name]
        return x, y + h / 2

    def right(name: str) -> tuple[float, float]:
        (x, y), w, h, *_ = boxes[name]
        return x + w, y + h / 2

    def top(name: str) -> tuple[float, float]:
        (x, y), w, h, *_ = boxes[name]
        return x + w / 2, y + h

    def bottom(name: str) -> tuple[float, float]:
        (x, y), w, _, *_ = boxes[name]
        return x + w / 2, y

    for source in ("intact", "biogrid", "string"):
        add_arrow(ax, right(source), left("canon"))
    add_arrow(ax, right("canon"), left("pos"), rad=0.05)
    add_arrow(ax, right("canon"), left("union"))
    add_arrow(ax, bottom("union"), top("neg"))
    add_arrow(ax, right("pos"), left("hub"))
    add_arrow(ax, bottom("hub"), top("score"))
    add_arrow(ax, right("neg"), left("score"))
    add_arrow(ax, bottom("score"), top("eval"))
    add_arrow(ax, bottom("canon"), top("pulldown"))
    add_arrow(ax, right("pulldown"), left("pd_eval"))
    add_arrow(ax, right("pd_eval"), left("eval"), rad=-0.12)

    ax.text(
        0.5,
        0.965,
        "Benchmark construction",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    save_figure(fig, outdir, "flowchart")
    plt.close(fig)


def plot_benchmark_organism_pie(rows: list[dict[str, str]], outdir: Path) -> None:
    orgs = ["arabidopsis", "ecoli", "human", "yeast"]
    counts = np.asarray([sum(1 for row in rows if row["organism"] == org) for org in orgs])
    labels_ = [ORGANISM_LABELS[org] for org in orgs]
    colors = [ORGANISM_COLORS[org] for org in orgs]

    fig, ax = plt.subplots(figsize=(6.9, 5.1))
    wedges, _ = ax.pie(
        counts,
        startangle=95,
        counterclock=False,
        colors=colors,
        wedgeprops={"width": 0.44, "edgecolor": "white", "linewidth": 2.2},
    )
    total = int(counts.sum())
    ax.text(0, 0.05, f"{total:,}", ha="center", va="center", fontsize=18, fontweight="bold")
    ax.text(0, -0.14, "interfaces", ha="center", va="center", fontsize=10)
    legend_labels = [
        f"{label}: {count:,} ({count / total:.1%})"
        for label, count in zip(labels_, counts)
    ]
    ax.legend(
        wedges,
        legend_labels,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
        fontsize=9,
    )
    ax.set_title(
        "Coverage-harmonized benchmark by organism",
        loc="left",
        fontweight="bold",
        pad=12,
    )
    save_figure(fig, outdir, "benchmark_organism_pie")
    plt.close(fig)


def read_edge_set(path: Path) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) < 2:
                continue
            a, b = fields[:2]
            if a == b:
                continue
            if a > b:
                a, b = b, a
            edges.add((a, b))
    return edges


def database_overlap_counts(organism: str) -> dict[str, int]:
    intact = read_edge_set(
        BENCHMARK_CACHE / f"{organism}.intact_union.edges.v14_primarycanon.txt"
    )
    biogrid = read_edge_set(
        BENCHMARK_CACHE / f"{organism}.biogrid_union.edges.v14_primarycanon.txt"
    )
    string = read_edge_set(
        BENCHMARK_CACHE
        / f"{organism}.string_union.edges.v14_primarycanon.c400_e400_tm200_db1000_exp_and_tm.txt"
    )

    ib = intact & biogrid
    is_ = intact & string
    bs = biogrid & string
    all3 = ib & string

    return {
        "intact_total": len(intact),
        "biogrid_total": len(biogrid),
        "string_total": len(string),
        "intact_only": len(intact - biogrid - string),
        "biogrid_only": len(biogrid - intact - string),
        "string_only": len(string - intact - biogrid),
        "intact_biogrid_only": len(ib - string),
        "intact_string_only": len(is_ - biogrid),
        "biogrid_string_only": len(bs - intact),
        "all3": len(all3),
        "union_total": len(intact | biogrid | string),
    }


def draw_database_venn_panel(
    ax,
    organism: str,
    counts: dict[str, int],
    *,
    show_source_labels: bool,
) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    circles = [
        ("IntAct", (0.42, 0.58), 0.285),
        ("BioGRID", (0.58, 0.58), 0.285),
        ("STRING", (0.50, 0.37), 0.285),
    ]
    for name, center, radius in circles:
        ax.add_patch(
            Circle(
                center,
                radius,
                facecolor=SOURCE_COLORS[name],
                edgecolor=SOURCE_COLORS[name],
                alpha=0.27,
                linewidth=2.0,
            )
        )

    ax.text(
        0.02,
        0.98,
        ORGANISM_LABELS[organism],
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )

    if show_source_labels:
        labels = [
            ("IntAct", (0.18, 0.88), SOURCE_COLORS["IntAct"]),
            ("BioGRID", (0.82, 0.88), SOURCE_COLORS["BioGRID"]),
            ("STRING", (0.50, 0.035), SOURCE_COLORS["STRING"]),
        ]
        for label, xy, color in labels:
            ax.text(
                *xy,
                label,
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color=color,
            )

    positions = {
        "intact_only": (0.285, 0.63),
        "biogrid_only": (0.715, 0.63),
        "string_only": (0.50, 0.16),
        "intact_biogrid_only": (0.50, 0.70),
        "intact_string_only": (0.375, 0.415),
        "biogrid_string_only": (0.625, 0.415),
        "all3": (0.50, 0.505),
    }
    region_labels = {
        "intact_only": counts["intact_only"],
        "biogrid_only": counts["biogrid_only"],
        "string_only": counts["string_only"],
        "intact_biogrid_only": counts["intact_biogrid_only"],
        "intact_string_only": counts["intact_string_only"],
        "biogrid_string_only": counts["biogrid_string_only"],
        "all3": counts["all3"],
    }
    for key, value in region_labels.items():
        bbox = {
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#D7DCE5",
            "linewidth": 0.6,
            "alpha": 0.88,
        }
        ax.text(
            *positions[key],
            f"{value:,}",
            ha="center",
            va="center",
            fontsize=8.2 if value < 100000 else 7.4,
            fontweight="bold" if key == "all3" else "normal",
            bbox=bbox,
        )

    ax.text(
        0.50,
        0.86,
        f"union {counts['union_total']:,} pairs",
        ha="center",
        va="center",
        fontsize=7.8,
        color="#4B5563",
    )


def plot_database_overlap_venn(outdir: Path) -> None:
    orgs = ["arabidopsis", "ecoli", "human", "yeast"]
    records: list[dict[str, int | str]] = []
    counts_by_org = {}
    for organism in orgs:
        counts = database_overlap_counts(organism)
        counts_by_org[organism] = counts
        records.append({"organism": organism, **counts})

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.7))
    for ax, organism in zip(axes.ravel(), orgs):
        draw_database_venn_panel(
            ax,
            organism,
            counts_by_org[organism],
            show_source_labels=True,
        )
    fig.suptitle(
        "Canonical source-database overlap before hub filtering",
        x=0.03,
        y=0.978,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.03,
        0.028,
        "Numbers are unique canonical protein pairs after the paper-style STRING physical-link filter. "
        "Circle areas are schematic rather than count-proportional.",
        ha="left",
        va="bottom",
        fontsize=7.8,
        color="#4B5563",
    )
    fig.subplots_adjust(left=0.04, right=0.98, top=0.91, bottom=0.075, wspace=0.06, hspace=0.10)
    save_figure(fig, outdir, "benchmark_database_overlap_venn")
    pd.DataFrame.from_records(records).to_csv(
        outdir / "benchmark_database_overlap_venn_counts_latest.csv",
        index=False,
    )
    plt.close(fig)


def plot_score_histograms(rows: list[dict[str, str]], outdir: Path) -> None:
    y = labels(rows)
    panels = [
        ("interface_meta_score", "Meta score", True, (0, 1), SCORE_COLORS["interface_meta_score"]),
        ("interface_LIS", "LIS", True, (0, 0.8), SCORE_COLORS["interface_LIS"]),
        ("interface_ipSAE", "ipSAE", True, (0, 1), SCORE_COLORS["interface_ipSAE"]),
        ("interface_pDockQ2", "pDockQ2", True, (0, 1), SCORE_COLORS["interface_pDockQ2"]),
        ("average_interface_pae", "Average interface PAE", False, (0, 32), SCORE_COLORS["average_interface_pae"]),
        ("interface_sc", "Shape complementarity", True, (0, 0.8), SCORE_COLORS["interface_sc"]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), sharey=False)
    axes = axes.ravel()

    for ax, (column, title, higher, forced_range, color) in zip(axes, panels):
        x = values(rows, column)
        auc = auc_score(y, x, higher_is_better=higher)
        pos = x[(y == 1) & np.isfinite(x)]
        neg = x[(y == 0) & np.isfinite(x)]
        xmin, xmax = forced_range
        bins = np.linspace(xmin, xmax, 36)
        neg_weights = np.full(len(neg), 1.0 / max(len(neg), 1))
        pos_weights = np.full(len(pos), 1.0 / max(len(pos), 1))
        ax.hist(neg, bins=bins, weights=neg_weights, color=NEG_COLOR, alpha=0.42, label="Negative")
        ax.hist(pos, bins=bins, weights=pos_weights, color=color, alpha=0.50, label="Positive")
        ax.set_xlim(xmin, xmax)
        ax.set_title(f"{title}  AUROC={auc:.3f}", loc="left", fontweight="bold")
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        if column == "average_interface_pae":
            ax.set_xlabel("Score value (lower is better)")
        else:
            ax.set_xlabel("Score value")
        ax.set_ylabel("Fraction of class")

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Representative score distributions from the April 22 best-interface table", y=1.06)
    fig.tight_layout()
    save_figure(fig, outdir, "scores_histos")
    plt.close(fig)


def plot_score_heatmap(rows: list[dict[str, str]], outdir: Path) -> None:
    cells: list[tuple[str, str]] = []
    for org in ["arabidopsis", "ecoli", "human", "yeast"]:
        for model in ["af2", "af3"]:
            cells.append((org, model))

    matrix = np.full((len(HEATMAP_FEATURES), len(cells)), np.nan)
    for row_idx, (feature, _, higher) in enumerate(HEATMAP_FEATURES):
        for col_idx, (org, model) in enumerate(cells):
            subset = [row for row in rows if row["organism"] == org and row["model"] == model]
            matrix[row_idx, col_idx] = auc_score(labels(subset), values(subset, feature), higher_is_better=higher)

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    image = ax.imshow(matrix, cmap="viridis", vmin=0.55, vmax=0.95, aspect="auto")
    ax.set_xticks(
        np.arange(len(cells)),
        [f"{ORGANISM_LABELS[org]}\n{model.upper()}" for org, model in cells],
        rotation=0,
        ha="center",
    )
    ax.set_yticks(np.arange(len(HEATMAP_FEATURES)), [label for _, label, _ in HEATMAP_FEATURES])
    ax.set_title("Single-score AUROC by organism and backend", loc="left", fontweight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            norm_value = (value - 0.55) / (0.95 - 0.55)
            text_color = "white" if norm_value < 0.45 else "#14213D"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color=text_color)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("AUROC")
    save_figure(fig, outdir, "score_heatmap")
    plt.close(fig)


def plot_retrieval_summary(rows: list[dict[str, str]], outdir: Path) -> None:
    y = labels(rows)
    panels = [
        ("interface_meta_score", "Meta", True, SCORE_COLORS["interface_meta_score"]),
        ("interface_LIS", "LIS", True, SCORE_COLORS["interface_LIS"]),
        ("interface_ipSAE", "ipSAE", True, SCORE_COLORS["interface_ipSAE"]),
        ("interface_pDockQ2", "pDockQ2", True, SCORE_COLORS["interface_pDockQ2"]),
        ("interface_sc", "SC", True, SCORE_COLORS["interface_sc"]),
    ]

    fig, (ax_pr, ax_topk) = plt.subplots(1, 2, figsize=(10.8, 4.4))
    top_precisions = []
    top_labels = []
    for feature, label, higher, color in panels:
        x = values(rows, feature)
        precision, recall = precision_recall_curve_points(y, x, higher_is_better=higher)
        ap = average_precision(y, x, higher_is_better=higher)
        ax_pr.plot(
            recall,
            precision,
            linewidth=2.2,
            linestyle=SCORE_LINESTYLES.get(feature, "-"),
            color=color,
            label=f"{label} AP={ap:.3f}",
        )
        precision5, _, enrichment5 = top_fraction_metrics(y, x, 0.05, higher_is_better=higher)
        top_precisions.append((precision5, enrichment5, color))
        top_labels.append(label)

    prevalence = float(y.mean())
    ax_pr.axhline(prevalence, color="#9BA3AF", linestyle="--", linewidth=1.2, label=f"Prevalence={prevalence:.2f}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0.45, 1.02)
    ax_pr.set_title("Precision-recall", loc="left", fontweight="bold")
    ax_pr.grid(color=GRID_COLOR, linewidth=0.8)
    ax_pr.spines[["top", "right"]].set_visible(False)
    ax_pr.legend(frameon=False, fontsize=8, loc="lower left")

    x_pos = np.arange(len(top_labels))
    bars = ax_topk.bar(
        x_pos,
        [value for value, _, _ in top_precisions],
        color=[color for _, _, color in top_precisions],
        width=0.72,
    )
    for bar, (_, enrichment, _), feature_info in zip(bars, top_precisions, panels):
        feature = feature_info[0]
        bar.set_hatch(SCORE_HATCHES.get(feature, ""))
        bar.set_edgecolor("#202333")
        bar.set_linewidth(0.6)
        ax_topk.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{enrichment:.1f}x",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax_topk.axhline(prevalence, color="#9BA3AF", linestyle="--", linewidth=1.2)
    ax_topk.set_xticks(x_pos, top_labels)
    ax_topk.set_ylim(0.45, 1.02)
    ax_topk.set_ylabel("Precision in top 5%")
    ax_topk.set_title("Top-ranked enrichment", loc="left", fontweight="bold")
    ax_topk.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax_topk.spines[["top", "right"]].set_visible(False)
    save_figure(fig, outdir, "retrieval_summary")
    plt.close(fig)


def load_pulldown_panels() -> dict[str, pd.DataFrame]:
    report = pd.read_csv(PULLDOWN_REPORT)
    report["interface_meta_score"] = report.apply(lambda row: interface_meta_score(row.to_dict()), axis=1)

    panels: dict[str, pd.DataFrame] = {}
    for panel, spec in PULLDOWN_PANELS.items():
        labels_frame = pd.read_csv(spec["labels"], sep="\t")
        labeled = labels_frame.merge(report, left_on="pair", right_on="jobs", how="left", indicator=True)
        panels[panel] = labeled
    return panels


def pulldown_rank_summary(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for panel, frame in panels.items():
        scored = frame.loc[frame["_merge"] == "both"].copy()
        positive = scored.loc[scored["label"] == "positive"]
        if len(positive) != 1:
            continue
        positive_row = positive.iloc[0]
        for feature, label, higher in PULLDOWN_FEATURES:
            values_series = pd.to_numeric(scored[feature], errors="coerce")
            finite = values_series.notna() & np.isfinite(values_series)
            score_frame = scored.loc[finite, ["pair", "label", feature]].copy()
            if score_frame.empty or positive_row["pair"] not in set(score_frame["pair"]):
                continue
            oriented = score_frame[feature].astype(float).to_numpy()
            if not higher:
                oriented = -oriented
            positive_mask = score_frame["label"].eq("positive").to_numpy()
            positive_oriented = float(oriented[positive_mask][0])
            negative_mask = score_frame["label"].eq("negative").to_numpy()
            negative_oriented = oriented[negative_mask]
            better_negatives = int(np.sum(negative_oriented > positive_oriented))
            better_or_equal_negatives = int(np.sum(negative_oriented >= positive_oriented))
            negative_scores = score_frame.loc[negative_mask, feature].astype(float).to_numpy()
            if len(negative_scores):
                best_negative = float(np.nanmax(negative_scores) if higher else np.nanmin(negative_scores))
            else:
                best_negative = float("nan")
            n_negative = int(negative_mask.sum())
            percentile = 1.0 - better_or_equal_negatives / max(n_negative, 1)
            records.append(
                {
                    "panel": panel,
                    "panel_label": PULLDOWN_PANELS[panel]["label"],
                    "positive_pair": positive_row["pair"],
                    "positive_gene": positive_row.get("target_gene", ""),
                    "score": feature,
                    "score_label": label,
                    "direction": "higher" if higher else "lower",
                    "positive_score": float(positive_row[feature]),
                    "best_negative_score": best_negative,
                    "rank": better_negatives + 1,
                    "scored_pair_count": int(len(score_frame)),
                    "scored_negative_count": n_negative,
                    "labeled_pair_count": int(len(frame)),
                    "missing_labeled_scores": int(len(frame) - len(scored)),
                    "negative_scores_better_than_positive": better_negatives,
                    "negative_scores_better_or_equal_positive": better_or_equal_negatives,
                    "percentile_vs_scored_negatives": percentile,
                }
            )
    return pd.DataFrame.from_records(records)


def plot_pulldown_rank_summary(panels: dict[str, pd.DataFrame], outdir: Path) -> None:
    summary = pulldown_rank_summary(panels)
    score_order = [feature for feature, _, _ in PULLDOWN_FEATURES]
    panel_order = ["strong", "weak"]
    matrix = np.full((len(score_order), len(panel_order)), np.nan)
    annotations = [["" for _ in panel_order] for _ in score_order]

    for row_idx, feature in enumerate(score_order):
        for col_idx, panel in enumerate(panel_order):
            match = summary[(summary["score"] == feature) & (summary["panel"] == panel)]
            if match.empty:
                continue
            row = match.iloc[0]
            matrix[row_idx, col_idx] = float(row["percentile_vs_scored_negatives"])
            annotations[row_idx][col_idx] = f"rank {int(row['rank'])}/{int(row['scored_pair_count'])}"

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(panel_order)), [PULLDOWN_PANELS[p]["label"] for p in panel_order])
    ax.set_yticks(np.arange(len(score_order)), [label for _, label, _ in PULLDOWN_FEATURES])
    ax.set_title("Pulldown positive rank among database-negative candidates", loc="left", fontweight="bold")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if math.isfinite(value):
                color = "white" if value < 0.55 else "#14213D"
                ax.text(j, i, annotations[i][j], ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Positive percentile vs. scored negatives")
    save_figure(fig, outdir, "pulldown_rank_summary")
    plt.close(fig)


def plot_pulldown_meta_rank_curves(panels: dict[str, pd.DataFrame], outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), sharey=True)
    for ax, panel in zip(axes, ["strong", "weak"]):
        scored = panels[panel].loc[panels[panel]["_merge"] == "both"].copy()
        scored = scored[np.isfinite(pd.to_numeric(scored["interface_meta_score"], errors="coerce"))].copy()
        scored = scored.sort_values("interface_meta_score", ascending=False).reset_index(drop=True)
        ranks = np.arange(1, len(scored) + 1)
        negatives = scored["label"].eq("negative").to_numpy()
        positives = scored["label"].eq("positive").to_numpy()
        ax.scatter(
            ranks[negatives],
            scored.loc[negatives, "interface_meta_score"],
            s=14,
            alpha=0.45,
            color="#9BA3AF",
            edgecolors="none",
            label="Database-negative",
        )
        ax.scatter(
            ranks[positives],
            scored.loc[positives, "interface_meta_score"],
            s=140,
            marker="*",
            color=SCORE_COLORS["interface_meta_score"],
            edgecolors="white",
            linewidth=0.8,
            zorder=4,
            label="Known positive",
        )
        positive_rank = int(ranks[positives][0]) if positives.any() else -1
        positive_score = float(scored.loc[positives, "interface_meta_score"].iloc[0])
        if positive_rank <= 20:
            ax.annotate(
                f"rank {positive_rank}/{len(scored)}",
                xy=(positive_rank, positive_score),
                xytext=(positive_rank + 70, min(positive_score + 0.04, 0.97)),
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
                arrowprops={"arrowstyle": "-", "color": "#3B3E47", "linewidth": 1.0},
            )
        else:
            ax.text(
                positive_rank,
                min(positive_score + 0.035, 0.97),
                f"rank {positive_rank}/{len(scored)}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        ax.set_title(PULLDOWN_PANELS[panel]["label"], loc="left", fontweight="bold")
        ax.set_xlabel("Rank by interface meta score")
        ax.set_xlim(0, len(scored) + 25)
        ax.set_ylim(0.0, 1.02)
        ax.grid(color=GRID_COLOR, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Interface meta score")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    save_figure(fig, outdir, "pulldown_meta_rank_curves")
    plt.close(fig)


def plot_pulldown_score_histograms(panels: dict[str, pd.DataFrame], outdir: Path) -> None:
    features = [
        ("interface_meta_score", "Meta", True),
        ("interface_LIS", "LIS", True),
        ("interface_ipSAE", "ipSAE", True),
        ("interface_pDockQ2", "pDockQ2", True),
        ("interface_sc", "SC", True),
    ]
    panel_order = ["strong", "weak"]
    fig, axes = plt.subplots(
        len(panel_order),
        len(features),
        figsize=(12.4, 5.2),
        sharey=False,
    )

    for row_idx, panel in enumerate(panel_order):
        scored = panels[panel].loc[panels[panel]["_merge"] == "both"].copy()
        positive = scored.loc[scored["label"] == "positive"]
        if positive.empty:
            continue
        positive_row = positive.iloc[0]
        negatives = scored.loc[scored["label"] == "negative"]
        for col_idx, (feature, label, higher) in enumerate(features):
            ax = axes[row_idx, col_idx]
            neg_values = pd.to_numeric(negatives[feature], errors="coerce").dropna().to_numpy(dtype=float)
            pos_value = safe_float(positive_row.get(feature))
            finite_values = neg_values[np.isfinite(neg_values)]
            if math.isfinite(pos_value):
                finite_values = np.concatenate([finite_values, [pos_value]])
            if len(finite_values) == 0:
                ax.set_axis_off()
                continue
            xmin, xmax = float(np.nanmin(finite_values)), float(np.nanmax(finite_values))
            if math.isclose(xmin, xmax):
                xmin -= 0.5
                xmax += 0.5
            pad = (xmax - xmin) * 0.08
            bins = np.linspace(xmin - pad, xmax + pad, 28)
            weights = np.full(len(neg_values), 1.0 / max(len(neg_values), 1))
            ax.hist(
                neg_values,
                bins=bins,
                weights=weights,
                color=NEG_COLOR,
                alpha=0.42,
                edgecolor="white",
                linewidth=0.4,
            )
            if math.isfinite(pos_value):
                ax.axvline(pos_value, color=POS_COLOR, linewidth=2.4)
                ax.scatter(
                    [pos_value],
                    [ax.get_ylim()[1] * 0.92],
                    color=POS_COLOR,
                    s=28,
                    zorder=5,
                )
            direction = "higher is better" if higher else "lower is better"
            if row_idx == 0:
                ax.set_title(f"{label}\n{direction}", fontsize=9, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(PULLDOWN_PANELS[panel]["label"] + "\nnegative fraction")
            else:
                ax.set_ylabel("")
            ax.set_xlabel("score")
            ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)

    handles = [
        plt.Line2D([0], [0], color=NEG_COLOR, linewidth=8, alpha=0.48, label="database-negative targets"),
        plt.Line2D([0], [0], color=POS_COLOR, linewidth=2.4, label="known positive"),
    ]
    fig.legend(handles=handles, frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Pulldown score distributions with known positives marked", y=1.09)
    fig.tight_layout()
    save_figure(fig, outdir, "pulldown_score_histograms")
    plt.close(fig)


def plot_geometry_backend_distributions(rows: list[dict[str, str]], outdir: Path) -> None:
    y_all = labels(rows)
    panels = [
        ("interface_sc", "Shape complementarity", True, (0, 0.75)),
        ("interface_area", "Interface area", True, None),
        ("interface_solv_en", "Solvation energy", False, None),
    ]

    fig, axes = plt.subplots(len(panels), 2, figsize=(10.4, 7.4))
    for row_idx, (feature, title, higher, forced_range) in enumerate(panels):
        all_values = values(rows, feature)
        finite = all_values[np.isfinite(all_values)]
        if forced_range is None:
            xmin, xmax = np.nanpercentile(finite, [1, 99])
        else:
            xmin, xmax = forced_range
        bins = np.linspace(xmin, xmax, 36)
        for col_idx, model in enumerate(["af2", "af3"]):
            subset = [row for row in rows if row["model"] == model]
            y = labels(subset)
            x = values(subset, feature)
            pos = x[(y == 1) & np.isfinite(x)]
            neg = x[(y == 0) & np.isfinite(x)]
            ax = axes[row_idx, col_idx]
            ax.hist(
                neg,
                bins=bins,
                weights=np.full(len(neg), 1.0 / max(len(neg), 1)),
                color=NEG_COLOR,
                alpha=0.42,
                label="Negative",
            )
            ax.hist(
                pos,
                bins=bins,
                weights=np.full(len(pos), 1.0 / max(len(pos), 1)),
                color=GEOM_COLOR,
                alpha=0.52,
                label="Positive",
            )
            auc = auc_score(y, x, higher_is_better=higher)
            ax.set_xlim(xmin, xmax)
            ax.set_title(f"{model.upper()} {title}  AUROC={auc:.3f}", loc="left", fontweight="bold")
            ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_ylabel("Fraction of class")
            ax.set_xlabel("Value")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Geometry-only descriptors by backend and label", y=1.04)
    fig.tight_layout()
    save_figure(fig, outdir, "geometry_backend_distributions")
    plt.close(fig)


def plot_score_correlation(rows: list[dict[str, str]], outdir: Path) -> None:
    labels_for_plot = []
    ranked = []
    for feature, label, higher, _ in SCORE_FEATURES:
        x = values(rows, feature)
        x = x if higher else -x
        mask = np.isfinite(x)
        if mask.sum() < 10:
            continue
        ranks = np.full(len(x), np.nan)
        ranks[mask] = average_ranks(x[mask])
        ranked.append(ranks)
        labels_for_plot.append(label)
    matrix = np.vstack(ranked)
    row_means = np.nanmean(matrix, axis=1)
    filled = matrix.copy()
    missing_rows, missing_cols = np.where(~np.isfinite(filled))
    filled[missing_rows, missing_cols] = row_means[missing_rows]
    corr = np.corrcoef(filled)

    fig, ax = plt.subplots(figsize=(8.8, 7.4))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(labels_for_plot)), labels_for_plot, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels_for_plot)), labels_for_plot)
    ax.set_title("Rank correlation among oriented scores", loc="left", fontweight="bold")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="Spearman-like correlation")
    save_figure(fig, outdir, "score_correlation")
    plt.close(fig)


def load_ipsae_cutoff_scores(summary_dir: Path, organism: str, cutoff: int) -> tuple[np.ndarray, np.ndarray]:
    y_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    for label_name, y_value in [("pos_pairs", 1), ("neg_pairs", 0)]:
        path = summary_dir / f"{organism}__{label_name}__pae{cutoff}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        scores = pd.to_numeric(frame["interface_ipSAE"], errors="coerce").to_numpy(dtype=float)
        y_parts.append(np.full(len(scores), y_value, dtype=int))
        score_parts.append(scores)
    if not y_parts:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)
    return np.concatenate(y_parts), np.concatenate(score_parts)


def plot_ipsae_cutoff_summary(summary_path: Path, summary_dir: Path, outdir: Path) -> None:
    rows = read_rows(summary_path)
    by_org: dict[str, list[tuple[int, float]]] = defaultdict(list)
    ci_by_org_cutoff: dict[tuple[str, int], tuple[float, float]] = {}
    for row in rows:
        org = row["organism"]
        cutoff = int(row["pae_cutoff"])
        y, scores = load_ipsae_cutoff_scores(summary_dir, org, cutoff)
        if len(y):
            stats = stratified_bootstrap_metrics(y, scores, n_bootstrap=300, seed=17 + cutoff)
            auc_value = stats["auroc"]
            ci_by_org_cutoff[(org, cutoff)] = (stats["auroc_ci_low"], stats["auroc_ci_high"])
        else:
            auc_value = safe_float(row["auc"])
        by_org[org].append((cutoff, auc_value))

    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    colors = ORGANISM_COLORS
    for org in ["human", "yeast", "arabidopsis", "ecoli"]:
        data = sorted(by_org[org])
        cutoffs = np.asarray([x for x, _ in data])
        aucs = np.asarray([x for _, x in data])
        ci_lows = np.asarray([ci_by_org_cutoff.get((org, int(cutoff)), (auc, auc))[0] for cutoff, auc in data])
        ci_highs = np.asarray([ci_by_org_cutoff.get((org, int(cutoff)), (auc, auc))[1] for cutoff, auc in data])
        best_idx = int(np.nanargmax(aucs))
        ax.errorbar(
            cutoffs,
            aucs,
            yerr=np.vstack([np.maximum(0, aucs - ci_lows), np.maximum(0, ci_highs - aucs)]),
            marker="o",
            linewidth=2.0,
            elinewidth=0.9,
            capsize=2.5,
            markersize=5,
            color=colors[org],
            label=f"{ORGANISM_LABELS[org]} (best {cutoffs[best_idx]} A)",
        )
        ax.scatter(cutoffs[best_idx], aucs[best_idx], color=colors[org], s=70, zorder=4)

    ax.set_xlabel("ipSAE PAE cutoff (A)")
    ax.set_ylabel("AUROC")
    ax.set_title("ipSAE cutoff sweep on AF3 predictions", loc="left", fontweight="bold")
    ax.set_xticks([5, 10, 15, 20, 25, 30])
    ax.set_ylim(0.74, 0.95)
    ax.grid(axis="both", color=GRID_COLOR, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    save_figure(fig, outdir, "ipsae_cutoff_summary")
    plt.close(fig)


def plot_classifier_summary(summary_path: Path, outdir: Path) -> None:
    rows = read_rows(summary_path)
    order = ["LogReg", "SGD-log", "HistGB", "RF", "MLP", "ipSAE-only", "ipTM-only", "Chance"]
    by_model = {row["model"]: row for row in rows}
    models = [model for model in order if model in by_model]
    means = np.asarray([safe_float(by_model[model]["auc_mean"]) for model in models])
    stds = np.asarray([safe_float(by_model[model]["auc_std"]) for model in models])
    model_colors = {
        "LogReg": OKABE_ITO["blue"],
        "SGD-log": OKABE_ITO["sky"],
        "HistGB": OKABE_ITO["green"],
        "RF": OKABE_ITO["orange"],
        "MLP": OKABE_ITO["purple"],
        "ipSAE-only": OKABE_ITO["vermillion"],
        "ipTM-only": OKABE_ITO["yellow"],
        "Chance": "#9BA3AF",
    }
    colors = [model_colors.get(model, CONF_COLOR) for model in models]

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    x = np.arange(len(models))
    ax.bar(x, means, yerr=stds, capsize=3, color=colors, width=0.72, edgecolor="white", linewidth=0.7)
    for xx, mean in zip(x, means):
        ax.text(xx, mean + 0.006, f"{mean:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, models, rotation=25, ha="right")
    ax.set_ylabel("Validation AUROC")
    ax.set_ylim(0.48, 0.91)
    ax.set_title("Classifier comparison over 20 grouped splits", loc="left", fontweight="bold")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    save_figure(fig, outdir, "classifier_auc_summary")
    plt.close(fig)


def write_score_audit(rows: list[dict[str, str]], outdir: Path) -> None:
    y = labels(rows)
    out_path = outdir / "paper_score_metrics_latest.csv"
    fieldnames = [
        "score",
        "label",
        "family",
        "higher_is_better",
        "auroc",
        "auroc_ci_low",
        "auroc_ci_high",
        "average_precision",
        "ap_ci_low",
        "ap_ci_high",
        "precision_top_1pct",
        "recall_top_1pct",
        "enrichment_top_1pct",
        "precision_top_5pct",
        "recall_top_5pct",
        "enrichment_top_5pct",
        "precision_top_10pct",
        "recall_top_10pct",
        "enrichment_top_10pct",
        "n",
    ]
    rows_out = []
    for feature, label, higher, family in SCORE_FEATURES:
        x = values(rows, feature)
        stats = stratified_bootstrap_metrics(y, x, higher_is_better=higher, n_bootstrap=300)
        top1 = top_fraction_metrics(y, x, 0.01, higher_is_better=higher)
        top5 = top_fraction_metrics(y, x, 0.05, higher_is_better=higher)
        top10 = top_fraction_metrics(y, x, 0.10, higher_is_better=higher)
        rows_out.append(
            {
                "score": feature,
                "label": label,
                "family": family,
                "higher_is_better": int(higher),
                "auroc": f"{stats['auroc']:.6f}",
                "auroc_ci_low": f"{stats['auroc_ci_low']:.6f}",
                "auroc_ci_high": f"{stats['auroc_ci_high']:.6f}",
                "average_precision": f"{stats['ap']:.6f}",
                "ap_ci_low": f"{stats['ap_ci_low']:.6f}",
                "ap_ci_high": f"{stats['ap_ci_high']:.6f}",
                "precision_top_1pct": f"{top1[0]:.6f}",
                "recall_top_1pct": f"{top1[1]:.6f}",
                "enrichment_top_1pct": f"{top1[2]:.6f}",
                "precision_top_5pct": f"{top5[0]:.6f}",
                "recall_top_5pct": f"{top5[1]:.6f}",
                "enrichment_top_5pct": f"{top5[2]:.6f}",
                "precision_top_10pct": f"{top10[0]:.6f}",
                "recall_top_10pct": f"{top10[1]:.6f}",
                "enrichment_top_10pct": f"{top10[2]:.6f}",
                "n": stats["n"],
            }
        )

    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    legacy_path = outdir / "paper_score_audit_latest.csv"
    with legacy_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["score", "higher_is_better", "auroc", "n"])
        writer.writeheader()
        for row in rows_out:
            writer.writerow(
                {
                    "score": row["score"],
                    "higher_is_better": row["higher_is_better"],
                    "auroc": row["auroc"],
                    "n": row["n"],
                }
            )


def write_ipsae_cutoff_bootstrap(summary_path: Path, summary_dir: Path, outdir: Path) -> None:
    rows = read_rows(summary_path)
    out_path = outdir / "ipsae_cutoff_bootstrap_latest.csv"
    fieldnames = [
        "organism",
        "pae_cutoff",
        "n_pos",
        "n_neg",
        "auroc",
        "auroc_ci_low",
        "auroc_ci_high",
        "average_precision",
        "ap_ci_low",
        "ap_ci_high",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            org = row["organism"]
            cutoff = int(row["pae_cutoff"])
            y, scores = load_ipsae_cutoff_scores(summary_dir, org, cutoff)
            stats = stratified_bootstrap_metrics(y, scores, n_bootstrap=300, seed=101 + cutoff)
            writer.writerow(
                {
                    "organism": org,
                    "pae_cutoff": cutoff,
                    "n_pos": row["n_pos"],
                    "n_neg": row["n_neg"],
                    "auroc": f"{stats['auroc']:.6f}",
                    "auroc_ci_low": f"{stats['auroc_ci_low']:.6f}",
                    "auroc_ci_high": f"{stats['auroc_ci_high']:.6f}",
                    "average_precision": f"{stats['ap']:.6f}",
                    "ap_ci_low": f"{stats['ap_ci_low']:.6f}",
                    "ap_ci_high": f"{stats['ap_ci_high']:.6f}",
                }
            )


def split_pair(pair: str) -> tuple[str, str]:
    parts = str(pair).split("+")
    if len(parts) != 2:
        return str(pair), ""
    return parts[0], parts[1]


def protein_set_from_pairs(pairs: pd.Series) -> set[str]:
    proteins: set[str] = set()
    for pair in pairs.dropna().astype(str):
        a, b = split_pair(pair)
        if a:
            proteins.add(a)
        if b:
            proteins.add(b)
    return proteins


def load_uniprot_lengths(cache_dir: Path) -> dict[str, int]:
    lengths: dict[str, int] = {}
    for path in cache_dir.glob("uniprot_lengths.*.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        for accession, length in data.items():
            try:
                lengths[str(accession)] = int(length)
            except (TypeError, ValueError):
                continue
    return lengths


def pair_length_frame(pairs: pd.Series, lengths: dict[str, int]) -> pd.DataFrame:
    records = []
    for pair in pairs.dropna().astype(str):
        a, b = split_pair(pair)
        len_a = lengths.get(a)
        len_b = lengths.get(b)
        if len_a is None or len_b is None:
            continue
        records.append(
            {
                "pair": pair,
                "lenA": len_a,
                "lenB": len_b,
                "mean_len": (len_a + len_b) / 2.0,
                "abs_len_diff": abs(len_a - len_b),
            }
        )
    return pd.DataFrame.from_records(records)


def write_benchmark_audit(rows: list[dict[str, str]], outdir: Path) -> None:
    scored = pd.DataFrame(rows)
    lengths = load_uniprot_lengths(BENCHMARK_ROOT / "data" / "cache")
    out_path = outdir / "benchmark_audit_latest.csv"
    fieldnames = [
        "organism",
        "scored_rows",
        "scored_unique_pairs",
        "scored_positive_pairs",
        "scored_negative_pairs",
        "scored_unique_proteins",
        "proteins_in_positive_pairs",
        "proteins_in_negative_pairs",
        "proteins_in_both_pos_and_neg",
        "construction_export_pairs",
        "scored_pairs_found_in_construction_export",
        "median_chain_length_positive",
        "median_chain_length_negative",
        "median_mean_pair_length_positive",
        "median_mean_pair_length_negative",
        "median_abs_length_difference_positive",
        "median_abs_length_difference_negative",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for org in ["arabidopsis", "ecoli", "human", "yeast"]:
            scored_org = scored[scored["organism"] == org]
            scored_pairs = scored_org[["jobs", "label"]].drop_duplicates()
            pos_pairs = scored_pairs.loc[scored_pairs["label"] == "positive", "jobs"]
            neg_pairs = scored_pairs.loc[scored_pairs["label"] == "negative", "jobs"]
            pos_proteins = protein_set_from_pairs(pos_pairs)
            neg_proteins = protein_set_from_pairs(neg_pairs)
            pos_lengths = pair_length_frame(pos_pairs, lengths)
            neg_lengths = pair_length_frame(neg_pairs, lengths)
            construction_path = BENCHMARK_EXPORTS / f"benchmark_{org}_all.csv"
            if construction_path.exists():
                construction_pairs = set(pd.read_csv(construction_path)["pair"].astype(str))
                construction_count = len(construction_pairs)
                scored_in_export = len(set(scored_pairs["jobs"].astype(str)) & construction_pairs)
            else:
                construction_count = ""
                scored_in_export = ""

            writer.writerow(
                {
                    "organism": org,
                    "scored_rows": len(scored_org),
                    "scored_unique_pairs": len(scored_pairs),
                    "scored_positive_pairs": len(pos_pairs),
                    "scored_negative_pairs": len(neg_pairs),
                    "scored_unique_proteins": len(pos_proteins | neg_proteins),
                    "proteins_in_positive_pairs": len(pos_proteins),
                    "proteins_in_negative_pairs": len(neg_proteins),
                    "proteins_in_both_pos_and_neg": len(pos_proteins & neg_proteins),
                    "construction_export_pairs": construction_count,
                    "scored_pairs_found_in_construction_export": scored_in_export,
                    "median_chain_length_positive": f"{float(pos_lengths[['lenA', 'lenB']].stack().median()):.1f}",
                    "median_chain_length_negative": f"{float(neg_lengths[['lenA', 'lenB']].stack().median()):.1f}",
                    "median_mean_pair_length_positive": f"{float(pos_lengths['mean_len'].median()):.1f}",
                    "median_mean_pair_length_negative": f"{float(neg_lengths['mean_len'].median()):.1f}",
                    "median_abs_length_difference_positive": f"{float(pos_lengths['abs_len_diff'].median()):.1f}",
                    "median_abs_length_difference_negative": f"{float(neg_lengths['abs_len_diff'].median()):.1f}",
                }
            )


def _meta_percentile_from_quantiles(value: float, quantiles: np.ndarray) -> float:
    if not math.isfinite(value):
        return float("nan")
    levels = np.asarray(CALIBRATION_LEVELS, dtype=float)
    if value <= quantiles[0]:
        return float(levels[0])
    if value >= quantiles[-1]:
        return float(levels[-1])
    lower_idx = int(np.searchsorted(quantiles, value, side="right") - 1)
    lower_idx = max(0, min(lower_idx, len(quantiles) - 2))
    q0, q1 = quantiles[lower_idx], quantiles[lower_idx + 1]
    p0, p1 = levels[lower_idx], levels[lower_idx + 1]
    if value == q0 or q1 <= q0:
        return float(p0)
    return float(p0 + (value - q0) / (q1 - q0) * (p1 - p0))


def meta_scores_from_reference(frame: pd.DataFrame, reference: pd.DataFrame) -> np.ndarray:
    quantiles: dict[str, np.ndarray] = {}
    levels = np.asarray(CALIBRATION_LEVELS, dtype=float)
    for feature in META_SCORE_FEATURES:
        oriented = pd.to_numeric(reference[feature], errors="coerce").to_numpy(dtype=float)
        oriented = oriented * FEATURE_DIRECTIONS[feature]
        quantiles[feature] = np.nanquantile(oriented, levels)

    scores = np.full(len(frame), np.nan)
    for row_idx, (_, row) in enumerate(frame.iterrows()):
        parts = []
        for feature in META_SCORE_FEATURES:
            raw = safe_float(row.get(feature))
            if not math.isfinite(raw):
                continue
            oriented = raw * FEATURE_DIRECTIONS[feature]
            parts.append(_meta_percentile_from_quantiles(oriented, quantiles[feature]))
        if parts:
            scores[row_idx] = float(np.nanmean(parts))
    return scores


def write_metascore_holdout_audit(rows: list[dict[str, str]], outdir: Path) -> None:
    frame = pd.DataFrame(rows)
    out_path = outdir / "metascore_leave_organism_out_latest.csv"
    fieldnames = [
        "heldout_organism",
        "n",
        "production_meta_auroc",
        "production_meta_ap",
        "leave_organism_reference_auroc",
        "leave_organism_reference_ap",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for org in ["arabidopsis", "ecoli", "human", "yeast"]:
            test = frame[frame["organism"] == org].copy()
            train = frame[frame["organism"] != org].copy()
            y = (test["label"] == "positive").astype(int).to_numpy()
            production = test.apply(lambda row: interface_meta_score(row.to_dict()), axis=1).to_numpy(dtype=float)
            heldout = meta_scores_from_reference(test, train)
            writer.writerow(
                {
                    "heldout_organism": org,
                    "n": len(test),
                    "production_meta_auroc": f"{auc_score(y, production):.6f}",
                    "production_meta_ap": f"{average_precision(y, production):.6f}",
                    "leave_organism_reference_auroc": f"{auc_score(y, heldout):.6f}",
                    "leave_organism_reference_ap": f"{average_precision(y, heldout):.6f}",
                }
            )


def write_coverage_harmonization_audit(
    records: list[dict[str, int | str]],
    outdir: Path,
) -> None:
    out_path = outdir / "coverage_harmonization_latest.csv"
    fieldnames = ["organism", "label", "model", "raw_count", "kept_count", "dropped_count"]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_pairwise_auc_comparisons(rows: list[dict[str, str]], outdir: Path) -> None:
    y = labels(rows)
    comparisons = [
        ("interface_meta_score", "interface_LIS"),
        ("interface_meta_score", "interface_ipSAE"),
        ("interface_meta_score", "interface_pDockQ2"),
        ("interface_meta_score", "iptm"),
        ("interface_meta_score", "interface_sc"),
        ("interface_LIS", "interface_ipSAE"),
    ]
    score_cache = {feature: values(rows, feature) for feature, _, _, _ in SCORE_FEATURES}
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    rng = np.random.default_rng(77)

    out_path = outdir / "pairwise_auc_bootstrap_latest.csv"
    fieldnames = ["score_a", "score_b", "auroc_a", "auroc_b", "auroc_diff", "diff_ci_low", "diff_ci_high", "bootstrap_prob_diff_gt_zero"]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for score_a, score_b in comparisons:
            a = score_cache[score_a]
            b = score_cache[score_b]
            auc_a = auc_score(y, a)
            auc_b = auc_score(y, b)
            diffs = []
            for _ in range(500):
                sample_idx = np.concatenate(
                    [
                        rng.choice(pos_idx, size=len(pos_idx), replace=True),
                        rng.choice(neg_idx, size=len(neg_idx), replace=True),
                    ]
                )
                diffs.append(auc_score(y[sample_idx], a[sample_idx]) - auc_score(y[sample_idx], b[sample_idx]))
            diff_values = np.asarray(diffs)
            writer.writerow(
                {
                    "score_a": score_a,
                    "score_b": score_b,
                    "auroc_a": f"{auc_a:.6f}",
                    "auroc_b": f"{auc_b:.6f}",
                    "auroc_diff": f"{(auc_a - auc_b):.6f}",
                    "diff_ci_low": f"{float(np.nanpercentile(diff_values, 2.5)):.6f}",
                    "diff_ci_high": f"{float(np.nanpercentile(diff_values, 97.5)):.6f}",
                    "bootstrap_prob_diff_gt_zero": f"{float(np.mean(diff_values > 0)):.6f}",
                }
            )


def write_pulldown_audit(panels: dict[str, pd.DataFrame], outdir: Path) -> None:
    summary = pulldown_rank_summary(panels)
    summary.to_csv(outdir / "pulldown_rank_summary_latest.csv", index=False)

    labeled_frames = []
    for panel, frame in panels.items():
        scored = frame.loc[frame["_merge"] == "both"].copy()
        scored["panel"] = panel
        scored["panel_label"] = PULLDOWN_PANELS[panel]["label"]
        columns = [
            "panel",
            "panel_label",
            "pair",
            "label",
            "role",
            "bait",
            "target",
            "target_gene",
            "target_length",
            "length_delta_vs_positive",
            "model_used",
            "interface",
            "interface_meta_score",
        ]
        for feature, _, _ in PULLDOWN_FEATURES:
            if feature not in columns:
                columns.append(feature)
        labeled_frames.append(scored[[column for column in columns if column in scored.columns]])
    if labeled_frames:
        pd.concat(labeled_frames, ignore_index=True).to_csv(
            outdir / "pulldown_labeled_scores_latest.csv",
            index=False,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=LATEST_MERGED)
    parser.add_argument("--ipsae-summary", type=Path, default=IPSAE_CUTOFF_SUMMARY)
    parser.add_argument("--ipsae-summary-dir", type=Path, default=IPSAE_SUMMARY_DIR)
    parser.add_argument("--classifier-summary", type=Path, default=CLASSIFIER_SUMMARY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    setup_style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    raw_rows = read_rows(args.input_csv)
    if not raw_rows:
        raise SystemExit(f"no rows in {args.input_csv}")
    rows, harmonization_records = harmonize_backend_coverage(raw_rows)
    if not rows:
        raise SystemExit("coverage harmonization removed all rows")

    plot_flowchart(args.outdir)
    plot_database_overlap_venn(args.outdir)
    plot_score_histograms(rows, args.outdir)
    plot_score_heatmap(rows, args.outdir)
    plot_retrieval_summary(rows, args.outdir)
    pulldown_panels = load_pulldown_panels()
    plot_pulldown_rank_summary(pulldown_panels, args.outdir)
    plot_pulldown_meta_rank_curves(pulldown_panels, args.outdir)
    plot_pulldown_score_histograms(pulldown_panels, args.outdir)
    plot_geometry_backend_distributions(rows, args.outdir)
    plot_score_correlation(rows, args.outdir)
    plot_ipsae_cutoff_summary(args.ipsae_summary, args.ipsae_summary_dir, args.outdir)
    plot_classifier_summary(args.classifier_summary, args.outdir)
    write_score_audit(rows, args.outdir)
    write_ipsae_cutoff_bootstrap(args.ipsae_summary, args.ipsae_summary_dir, args.outdir)
    write_benchmark_audit(rows, args.outdir)
    write_metascore_holdout_audit(rows, args.outdir)
    write_coverage_harmonization_audit(harmonization_records, args.outdir)
    write_pairwise_auc_comparisons(rows, args.outdir)
    write_pulldown_audit(pulldown_panels, args.outdir)

    print(f"Wrote paper figures to {args.outdir}")
    print(f"Score source: {args.input_csv}")
    print(f"Coverage harmonized rows: {len(rows)} / raw rows: {len(raw_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
