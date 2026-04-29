#!/usr/bin/env python3
"""Plot real-structure benchmark scores for a small human AF3 subset."""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from matplotlib.colors import LinearSegmentedColormap

from alphajudge.biophysics.zernike import (
    ATOM_GAUSSIAN,
    GAUSSIAN_WEIGHTED_SCORE,
    HARD_CUTOFF_SCORE,
    JOINT_LOW_ORDER_RATIO_SCORE,
    JOINT_RESIDUE_BEAD_GAUSSIAN,
    RESIDUE_BEAD_GAUSSIAN,
    ZernikeSpec,
    zernike_shape_complementarity,
)

BENCH_ROOT_DEFAULT = Path(
    "/g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26/predictions"
)

CANDIDATES = OrderedDict(
    [
        (
            "interface_sc",
            {
                "label": "SC baseline",
                "kind": "csv",
                "panel_title": "interface_sc\n(from best_interfaces.csv)",
            },
        ),
        (
            "atom_gaussian",
            {
                "label": "Atom Zernike",
                "kind": "zernike",
                "panel_title": "Atom Gaussian\n32^3, N=10, sigma=1.5",
                "spec": ZernikeSpec(
                    representation=ATOM_GAUSSIAN,
                    grid_size=32,
                    order=10,
                    sigma=1.5,
                    score_mode=HARD_CUTOFF_SCORE,
                    fit_order=12,
                ),
            },
        ),
        (
            "residue_bead_weighted",
            {
                "label": "Residue Zernike",
                "kind": "zernike",
                "panel_title": "Residue Bead Weighted\n24^3, N=8, sigma=2.0, n0=4",
                "spec": ZernikeSpec(
                    representation=RESIDUE_BEAD_GAUSSIAN,
                    grid_size=24,
                    order=8,
                    sigma=2.0,
                    score_mode=GAUSSIAN_WEIGHTED_SCORE,
                    fit_order=12,
                    order_decay_n0=4.0,
                ),
            },
        ),
        (
            "joint_residue_ratio",
            {
                "label": "Joint Zernike",
                "kind": "zernike",
                "panel_title": "Joint Residue Ratio\n24^3, N=6, sigma=2.0",
                "spec": ZernikeSpec(
                    representation=JOINT_RESIDUE_BEAD_GAUSSIAN,
                    grid_size=24,
                    order=6,
                    sigma=2.0,
                    score_mode=JOINT_LOW_ORDER_RATIO_SCORE,
                    fit_order=12,
                ),
            },
        ),
    ]
)

POS_COLOR = "#2E7D4D"
NEG_COLOR = "#B04E37"
FACE_BG = "#F6F4EF"
GRID = "#D7DDD8"
TEXT = "#21332D"
MUTED = "#5B6D65"


def subset_auroc(rows: list[dict], field: str) -> float:
    pos = [float(row[field]) for row in rows if row["label"] == "positive"]
    neg = [float(row[field]) for row in rows if row["label"] == "negative"]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    total = 0
    for p_score in pos:
        for n_score in neg:
            total += 1
            if p_score > n_score:
                wins += 1.0
            elif p_score == n_score:
                wins += 0.5
    return wins / total


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def find_manifest(group_root: Path, manifest_tag: str | None) -> Path:
    if manifest_tag:
        path = group_root / f"manifest.{manifest_tag}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    manifests = sorted(group_root.glob("manifest.*.csv"), key=lambda p: (p.stat().st_mtime, str(p)))
    if not manifests:
        raise FileNotFoundError(f"No manifest.*.csv found in {group_root}")
    return manifests[-1]


def parser_for_model(path: str):
    model_path = Path(path)
    return MMCIFParser(QUIET=True) if model_path.suffix.lower() == ".cif" else PDBParser(QUIET=True)


def load_interface_residues(model_file: str, interface_label: str):
    structure = parser_for_model(model_file).get_structure("model", model_file)
    model = next(structure.get_models())
    chains = {chain.id: tuple(chain) for chain in model.get_chains()}
    chain1_id, chain2_id = interface_label.split("_", 1)
    return chains[chain1_id], chains[chain2_id]


def select_group_rows(
    bench_root: Path,
    *,
    backend: str,
    pairset: str,
    organism: str,
    per_class: int,
    manifest_tag: str | None,
) -> list[dict]:
    group_root = bench_root / organism / backend / pairset
    best_path = group_root / "best_interfaces.csv"
    manifest_path = find_manifest(group_root, manifest_tag)
    manifest_rows = {
        row["pair"]: row
        for row in read_csv_rows(manifest_path)
        if row.get("pair")
    }

    seen: set[str] = set()
    selected = []
    for row in read_csv_rows(best_path):
        pair = row.get("jobs", "").strip()
        interface = row.get("interface", "").strip()
        if not pair or pair in seen or not interface:
            continue
        seen.add(pair)
        manifest_row = manifest_rows.get(pair)
        if manifest_row is None:
            continue
        model_file = str(manifest_row.get("model_file", "")).strip()
        if not model_file or not Path(model_file).exists():
            continue
        try:
            sc = float(row["interface_sc"])
        except Exception:
            continue
        selected.append(
            {
                "pair": pair,
                "interface": interface,
                "model_file": model_file,
                "label": "positive" if pairset == "pos_pairs" else "negative",
                "interface_sc": sc,
                "average_interface_pae": row.get("average_interface_pae", ""),
                "interface_area": row.get("interface_area", ""),
                "backend": backend,
                "organism": organism,
            }
        )

    selected.sort(key=lambda row: row["interface_sc"], reverse=(pairset == "neg_pairs"))
    return selected[:per_class]


def score_rows(rows: list[dict]) -> list[dict]:
    scored = []
    for row in rows:
        residues1, residues2 = load_interface_residues(row["model_file"], row["interface"])
        out = dict(row)
        for key, config in CANDIDATES.items():
            if config["kind"] == "csv":
                out[key] = float(row["interface_sc"])
            else:
                out[key] = float(
                    zernike_shape_complementarity(
                        residues1,
                        residues2,
                        representation=config["spec"].representation,
                        distance=config["spec"].distance,
                        grid_size=config["spec"].grid_size,
                        order=config["spec"].order,
                        sigma=config["spec"].sigma,
                        padding=config["spec"].padding,
                        surface_density=config["spec"].surface_density,
                        surface_trim_cutoff=config["spec"].surface_trim_cutoff,
                        surface_probe_radius=config["spec"].surface_probe_radius,
                        proximity_length_scale=config["spec"].proximity_length_scale,
                        score_mode=config["spec"].score_mode,
                        fit_order=config["spec"].fit_order,
                        order_decay_n0=config["spec"].order_decay_n0,
                    )
                )
        scored.append(out)
    return scored


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows: list[dict], out_path: Path, *, organism: str, backend: str, per_class: int) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": FACE_BG,
            "figure.facecolor": FACE_BG,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
        }
    )

    candidate_keys = list(CANDIDATES)
    candidate_labels = [CANDIDATES[key]["label"] for key in candidate_keys]
    row_labels = [
        f"{'POS' if row['label'] == 'positive' else 'NEG'}  {row['pair']}"
        for row in rows
    ]

    raw = np.asarray(
        [[float(row[key]) for key in candidate_keys] for row in rows],
        dtype=float,
    )
    normalized = np.zeros_like(raw)
    for col_idx in range(raw.shape[1]):
        col = raw[:, col_idx]
        lo = float(np.min(col))
        hi = float(np.max(col))
        normalized[:, col_idx] = 0.5 if hi == lo else (col - lo) / (hi - lo)

    aucs = [subset_auroc(rows, key) for key in candidate_keys]
    fig = plt.figure(figsize=(14.5, max(9.2, 0.44 * len(rows) + 4.8)))
    grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[0.92, 3.35])
    ax_auc = fig.add_subplot(grid[0, 0])
    ax_heat = fig.add_subplot(grid[1, 0])
    fig.subplots_adjust(left=0.22, right=0.93, top=0.84, bottom=0.12, hspace=0.48)

    bar_y = np.arange(len(candidate_keys))
    bar_colors = ["#314D45", "#2E7D4D", "#789C55", "#8D7A45"]
    ax_auc.barh(bar_y, aucs, color=bar_colors, height=0.58)
    ax_auc.axvline(0.5, color="#9EA9A2", linewidth=1.4, linestyle=(0, (4, 4)))
    ax_auc.text(0.505, -0.62, "random", color=MUTED, fontsize=9, ha="left", va="center")
    ax_auc.set_xlim(0.0, 1.0)
    ax_auc.set_yticks(bar_y, candidate_labels, fontsize=11)
    ax_auc.invert_yaxis()
    ax_auc.set_xlabel("subset AUROC", fontsize=11)
    ax_auc.set_title("Separation summary: subset AUROC, higher is better", fontsize=13, fontweight="bold", pad=10)
    ax_auc.grid(axis="x", color=GRID, linewidth=1.0)
    ax_auc.set_axisbelow(True)
    for y_pos, auc in zip(bar_y, aucs):
        ax_auc.text(
            min(auc + 0.035, 0.96),
            y_pos,
            f"{auc:.2f}",
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
            color=TEXT,
        )
    for spine in ("top", "right", "left"):
        ax_auc.spines[spine].set_visible(False)
    ax_auc.tick_params(axis="y", length=0)

    cmap = LinearSegmentedColormap.from_list("interaction_like", ["#F3EFE8", "#BBD7BC", "#2E7D4D"])
    image = ax_heat.imshow(normalized, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax_heat.set_xticks(np.arange(len(candidate_keys)), candidate_labels, fontsize=11)
    ax_heat.set_yticks(np.arange(len(row_labels)), row_labels, fontsize=10)
    ax_heat.tick_params(axis="both", length=0)
    ax_heat.set_title(
        "Scores by complex: higher/darker within each score = more interaction-like; POS high, NEG low",
        fontsize=13,
        fontweight="bold",
        pad=16,
    )
    ax_heat.set_xticks(np.arange(-0.5, len(candidate_keys), 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax_heat.grid(which="minor", color=FACE_BG, linewidth=2.4)
    ax_heat.tick_params(which="minor", bottom=False, left=False)

    split = sum(1 for row in rows if row["label"] == "positive") - 0.5
    if split >= 0:
        ax_heat.axhline(split, color="#FFFFFF", linewidth=5)

    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            value = raw[i, j]
            label = f"{value:.3f}" if value < 0.2 else f"{value:.2f}"
            ax_heat.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color="#0F241D" if normalized[i, j] < 0.72 else "white",
                fontweight="bold" if normalized[i, j] > 0.72 else "normal",
            )

    for idx, row in enumerate(rows):
        ax_heat.get_yticklabels()[idx].set_color(POS_COLOR if row["label"] == "positive" else NEG_COLOR)
        ax_heat.get_yticklabels()[idx].set_fontweight("bold")
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(image, ax=ax_heat, fraction=0.03, pad=0.025)
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(["low", "high"])
    cbar.ax.tick_params(labelsize=9, length=0)
    cbar.outline.set_visible(False)

    fig.suptitle(
        (
            f"Real {organism} {backend.upper()} subset: can Zernike rescue low-SC positives?\n"
            f"{per_class} lowest-SC positives + {per_class} highest-SC negatives"
        ),
        fontsize=18,
        fontweight="bold",
        y=0.965,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if out_path.suffix.lower() != ".png":
        png_path = out_path.with_suffix(".png")
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", default=str(BENCH_ROOT_DEFAULT))
    parser.add_argument("--organism", default="human")
    parser.add_argument("--backend", default="af3")
    parser.add_argument("--per-class", type=int, default=6)
    parser.add_argument("--manifest-tag", default=None)
    parser.add_argument("--out-svg", default="docs/zernike_real_subset_scores.svg")
    parser.add_argument("--out-csv", default="docs/zernike_real_subset_scores.csv")
    args = parser.parse_args()

    bench_root = Path(args.bench_root)
    positives = select_group_rows(
        bench_root,
        backend=args.backend,
        pairset="pos_pairs",
        organism=args.organism,
        per_class=args.per_class,
        manifest_tag=args.manifest_tag,
    )
    negatives = select_group_rows(
        bench_root,
        backend=args.backend,
        pairset="neg_pairs",
        organism=args.organism,
        per_class=args.per_class,
        manifest_tag=args.manifest_tag,
    )

    rows = positives + negatives
    scored = score_rows(rows)
    write_csv(Path(args.out_csv), scored)
    plot_rows(
        scored,
        Path(args.out_svg),
        organism=args.organism,
        backend=args.backend,
        per_class=args.per_class,
    )
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
