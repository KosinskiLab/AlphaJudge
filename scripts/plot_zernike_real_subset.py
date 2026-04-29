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

    ylabels = []
    for row in rows:
        prefix = "POS" if row["label"] == "positive" else "NEG"
        ylabels.append(f"{prefix}  {row['pair']}")
    y = np.arange(len(rows))
    colors = [POS_COLOR if row["label"] == "positive" else NEG_COLOR for row in rows]

    fig, axes = plt.subplots(
        1,
        len(CANDIDATES),
        figsize=(16, max(7, 0.55 * len(rows) + 2.6)),
        sharey=True,
        constrained_layout=True,
    )

    for idx, (key, config) in enumerate(CANDIDATES.items()):
        ax = axes[idx]
        x = [float(row[key]) for row in rows]
        ax.scatter(x, y, s=70, c=colors, edgecolors="white", linewidths=0.8, zorder=3)
        ax.set_title(config["panel_title"], fontsize=12, fontweight="bold", pad=12)
        ax.grid(axis="x", color=GRID, linewidth=1.0, alpha=0.9)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if x:
            lo = min(x)
            hi = max(x)
            span = max(hi - lo, 1e-6)
            pad = 0.12 * span
            left = max(0.0, lo - pad)
            right = hi + pad
            if key != "interface_sc":
                right = min(1.02, right)
            ax.set_xlim(left, right)
        ax.set_xlabel("score", fontsize=11)
        if idx == 0:
            ax.set_yticks(y, ylabels, fontsize=10)
        else:
            ax.tick_params(axis="y", length=0)

    if rows:
        split = sum(1 for row in rows if row["label"] == "positive") - 0.5
        if split >= 0:
            for ax in axes:
                ax.axhline(split, color=GRID, linewidth=1.6, linestyle=(0, (4, 4)))
                ax.invert_yaxis()

    fig.suptitle(
        f"Real benchmark subset on {organism} {backend.upper()} structures",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        0.02,
        (
            f"{per_class} lowest-SC positives and {per_class} highest-SC negatives from best_interfaces.csv; "
            "complex names shown on the y-axis."
        ),
        ha="center",
        fontsize=10.5,
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
