from __future__ import annotations

from pathlib import Path
import csv
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .parsers import pick_parser
from .complex import Complex

logger = logging.getLogger(__name__)


def _save_pae_heatmap(
    pae_matrix,
    out_file: Path,
    chain_boundaries: list[float] | None = None,
    figsize: tuple[int, int] = (10, 10),
) -> None:
    """
    Save a PAE heatmap PNG for a given residue×residue PAE matrix.

    Optionally draw vertical and horizontal separator lines at positions
    provided in `chain_boundaries` (typically between different chains).
    """
    try:
        # pae_matrix is already a numpy array for memory efficiency
        mtx = pae_matrix if isinstance(pae_matrix, np.ndarray) else np.array(pae_matrix, dtype=float)
        if mtx.size == 0:
            logger.warning(f"empty PAE matrix; skipping heatmap for {out_file}")
            return

        # Ensure parent exists (safe no-op if already exists)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(mtx, cmap="Greens_r", vmin=0, vmax=30)
        ax.set_xlabel("Scored residue", fontsize=12)
        ax.set_ylabel("Aligned residue", fontsize=12)
        ax.set_title("Predicted Aligned Error (PAE)", fontsize=14, fontweight="bold")

        if chain_boundaries:
            for b in chain_boundaries:
                ax.axhline(b, color="black", linewidth=0.5)
                ax.axvline(b, color="black", linewidth=0.5)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Expected position error (Å)", rotation=270, labelpad=20)

        fig.tight_layout()
        fig.savefig(out_file, dpi=300)
        plt.close(fig)
        logger.info(f"wrote {out_file}")
    except Exception as e:
        logger.error(f"Could not create PAE heatmap {out_file}: {e}")


def process(
    directory: str,
    contact_thresh: float,
    pae_filter: float,
    models_to_analyse: str,
    ipsae_pae_cutoff: float = 10.0,
    *,
    per_run_csv_name: str = "interfaces.csv",
    skip_pae_png: bool = False,
) -> Path | None:
    d = Path(directory)
    parser = pick_parser(d)
    run = parser.parse_run(d)
    models = [run.order[0]] if models_to_analyse == "best" else run.order
    job = d.resolve().name

    rows: list[dict] = []
    for m in models:
        try:
            structure, confidence = run.load_model(m)
            comp = Complex(structure, confidence, contact_thresh, pae_filter, ipsae_pae_cutoff)

            global_score = (
                comp.mpDockQ
                if comp.num_chains > 2
                else (comp.interfaces[0].pDockQ if comp.interfaces else float("nan"))
            )

            for iface in comp.interfaces:
                if iface.num_intf_residues == 0:
                    continue
                if iface.average_interface_pae > pae_filter:
                    continue
                pd2, _ = iface.pDockQ2()
                label = (
                    f"{iface.chain1[0].get_parent().id}_{iface.chain2[0].get_parent().id}"
                )
                iptm_val = iface.iptm_chainpair if iface.iptm_chainpair is not None else confidence.iptm
                rows.append(
                    {
                        "jobs": job,
                        "model_used": m,
                        "interface": label,
                        "iptm_ptm": float(confidence.iptm_ptm)
                        if confidence.iptm_ptm is not None
                        else float("nan"),
                        "iptm": float(iptm_val) if iptm_val is not None else float("nan"),
                        "ptm": float(confidence.ptm)
                        if confidence.ptm is not None
                        else float("nan"),
                        "confidence_score": float(confidence.confidence_score)
                        if confidence.confidence_score is not None
                        else float("nan"),
                        "pDockQ/mpDockQ": global_score,
                        "average_interface_pae": iface.average_interface_pae,
                        "interface_average_plddt": iface.average_interface_plddt,
                        "interface_num_intf_residues": iface.num_intf_residues,
                        "interface_polar": iface.polar,
                        "interface_hydrophobic": iface.hydrophobic,
                        "interface_charged": iface.charged,
                        "interface_contact_pairs": iface.contact_pairs,
                        "interface_score": iface.score_complex,
                        "interface_pDockQ2": pd2,
                        "interface_ipSAE": iface.ipsae(),
                        "interface_LIS": iface.lis(),
                        "interface_hb": iface.hb,
                        "interface_sb": iface.sb,
                        "interface_ss": iface.ss,
                        "interface_sc": iface.sc,
                        "interface_area": iface.int_area,
                        "interface_solv_en": iface.int_solv_en,
                    }
                )

            # Compute chain boundaries for separator lines on PAE heatmap
            chain_boundaries: list[float] = []
            try:
                idx_lists = [
                    idxs
                    for idxs in comp._chain_indices_by_id.values()  # type: ignore[attr-defined]
                    if idxs
                ]
                if idx_lists:
                    sorted_bounds = sorted(max(idxs) + 0.5 for idxs in idx_lists)
                    chain_boundaries = sorted_bounds[:-1] if len(sorted_bounds) > 1 else []
            except Exception:
                chain_boundaries = []

            if not skip_pae_png:
                pae_png = d / f"pae_{m}.png"
                _save_pae_heatmap(
                    confidence.pae_matrix, pae_png, chain_boundaries=chain_boundaries
                )

            logger.info(f"processed model: {m} via {parser.name}")
        except Exception as e:
            logger.error(f"error processing model {m}: {e}")

    out = d / per_run_csv_name
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        else:
            f.write("")
    logger.info(f"wrote {out}")
    return out


def _discover_run_dirs(root: Path) -> list[Path]:
    """Walk a directory tree and collect directories that look like supported runs."""
    results: list[Path] = []

    # include root too (rglob doesn't include it)
    candidates = [root] if root.is_dir() else []
    if root.is_dir():
        candidates.extend([p for p in root.rglob("*") if p.is_dir()])

    for d in candidates:
        try:
            pick_parser(d)  # raises if not supported
            results.append(d)
        except Exception:
            continue

    # De-dupe + stable sort (shallow first)
    uniq = sorted(set(p.resolve() for p in results), key=lambda p: (len(p.parts), str(p)))
    return [Path(p) for p in uniq]


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _process_one_run(
    d_str: str,
    contact_thresh: float,
    pae_filter: float,
    models_to_analyse: str,
    summary_csv: str | None,
    ipsae_pae_cutoff: float,
    force_recompute: bool,
    per_run_csv_name: str,
    skip_pae_png: bool,
) -> tuple[str, list[dict]]:
    """
    Worker: process a single run dir (or reuse interfaces.csv) and optionally return rows for aggregation.
    Returns (run_dir, rows_for_summary).
    """
    d = Path(d_str)
    existing_csv = d / per_run_csv_name

    want_summary = summary_csv is not None

    # When building a summary, prefer reusing precomputed interfaces.csv
    if want_summary and existing_csv.exists() and not force_recompute:
        try:
            rows = _read_csv_rows(existing_csv)
            if rows:
                logger.info(f"reused existing {existing_csv} for aggregation")
                return (d_str, rows)
            logger.info(f"existing {existing_csv} is empty; recomputing")
        except Exception as e:
            logger.warning(f"could not reuse {existing_csv}; recomputing: {e}")

    # If no summary requested but interfaces.csv exists, skip recompute
    if not want_summary and existing_csv.exists() and not force_recompute:
        logger.info(f"reused existing {existing_csv}; skipping recompute")
        return (d_str, [])

    out_path = process(
        d_str,
        contact_thresh,
        pae_filter,
        models_to_analyse,
        ipsae_pae_cutoff,
        per_run_csv_name=per_run_csv_name,
        skip_pae_png=skip_pae_png,
    )

    if want_summary and out_path is not None:
        try:
            return (d_str, _read_csv_rows(Path(out_path)))
        except Exception as e:
            logger.error(f"failed reading {out_path} for aggregation: {e}")

    return (d_str, [])


def process_many(
    paths: list[str],
    contact_thresh: float,
    pae_filter: float,
    models_to_analyse: str,
    recursive: bool = False,
    summary_csv: str | None = None,
    cores: int = 1,
    ipsae_pae_cutoff: float = 10.0,
    force_recompute: bool = False,
    per_run_csv_name: str = "interfaces.csv",
    skip_pae_png: bool = False,
) -> Path | None:
    """
    Process one or more directories. Optionally recurse into nested directories
    to find supported runs. If summary_csv is provided, aggregate all per-run
    interface rows into a single CSV at that path and return it.

    cores:
      - 1 = serial
      - >1 = process pool over run directories
      - 0 or <0 = use os.cpu_count()
    """
    if not paths:
        logger.warning("no input paths provided")
        return None

    # Resolve set of run directories to process
    run_dirs: list[Path] = []
    for p in paths:
        rp = Path(p).resolve()
        if not rp.exists():
            logger.warning(f"path does not exist: {rp}")
            continue

        if recursive and rp.is_dir():
            run_dirs.extend(_discover_run_dirs(rp))
            continue

        # Try direct path as a run directory
        try:
            pick_parser(rp)
            run_dirs.append(rp)
        except Exception:
            logger.warning(
                f"no supported run detected at {rp} (use --recursive to search within)"
            )

    # Deduplicate
    seen: set[Path] = set()
    unique_run_dirs: list[Path] = []
    for d in run_dirs:
        r = d.resolve()
        if r not in seen:
            seen.add(r)
            unique_run_dirs.append(d)

    if not unique_run_dirs:
        logger.warning("no runnable directories found")
        return None

    # Normalize cores
    if cores <= 0:
        cores = os.cpu_count() or 1
    if cores > len(unique_run_dirs):
        cores = len(unique_run_dirs)

    aggregated_rows: list[dict] = []
    logger.info(f"Processing {len(unique_run_dirs)} runs with {cores} cores")
    # Serial
    if cores == 1:
        for d in unique_run_dirs:
            try:
                _, rows = _process_one_run(
                    str(d),
                    contact_thresh,
                    pae_filter,
                    models_to_analyse,
                    summary_csv,
                    ipsae_pae_cutoff,
                    force_recompute,
                    per_run_csv_name,
                    skip_pae_png,
                )
                if summary_csv and rows:
                    aggregated_rows.extend(rows)
            except Exception as e:
                logger.error(f"failed processing {d}: {e}")

    # Parallel
    else:
        # Important: logging from multiple processes can interleave; acceptable.
        with ProcessPoolExecutor(max_workers=cores) as ex:
            futures = [
                ex.submit(
                    _process_one_run,
                    str(d),
                    contact_thresh,
                    pae_filter,
                    models_to_analyse,
                    summary_csv,
                    ipsae_pae_cutoff,
                    force_recompute,
                    per_run_csv_name,
                    skip_pae_png,
                )
                for d in unique_run_dirs
            ]
            for fut in as_completed(futures):
                try:
                    _, rows = fut.result()
                    if summary_csv and rows:
                        aggregated_rows.extend(rows)
                except Exception as e:
                    logger.error(f"worker failed: {e}")

    if summary_csv:
        if not aggregated_rows:
            logger.info("no rows to write to summary; skipping creation")
            return None

        # Compute union of all keys to accommodate AF2/AF3 variations
        fieldnames: list[str] = []
        seen_fields: set[str] = set()
        for row in aggregated_rows:
            for k in row.keys():
                if k not in seen_fields:
                    seen_fields.add(k)
                    fieldnames.append(k)

        def _iptm_sort_key(row: dict) -> float:
            v = row.get("iptm")
            if v is None or v == "":
                return float("-inf")
            try:
                val = float(v)
            except Exception:
                return float("-inf")
            return val if val == val else float("-inf")  # NaN -> -inf

        aggregated_rows = sorted(aggregated_rows, key=_iptm_sort_key, reverse=True)

        summary_path = Path(summary_csv).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        with summary_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in aggregated_rows:
                w.writerow({k: row.get(k, "") for k in fieldnames})

        logger.info(
            f"wrote summary {summary_path} ({len(aggregated_rows)} rows from {len(unique_run_dirs)} runs)"
        )
        return summary_path

    return None
