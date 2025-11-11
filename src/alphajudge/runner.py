from __future__ import annotations
from pathlib import Path
import csv, logging
from .parsers import pick_parser
from .core import Complex

def process(directory: str, contact_thresh: float, pae_filter: float, models_to_analyse: str) -> Path | None:
    d = Path(directory)
    parser = pick_parser(d)
    run = parser.parse_run(d)
    models = [run.order[0]] if models_to_analyse == "best" else run.order
    job = d.resolve().name

    rows = []
    for m in models:
        try:
            structure, confidence = run.load_model(m)
            comp = Complex(structure, confidence, contact_thresh, pae_filter)
            global_score = comp.mpDockQ if comp.num_chains > 2 else (
                comp.interfaces[0].pDockQ if comp.interfaces else float('nan')
            )
            for iface in comp.interfaces:
                if iface.num_intf_residues == 0: continue
                if iface.average_interface_pae > pae_filter: continue
                pd2, _ = iface.pDockQ2()
                label = f"{iface.chain1[0].get_parent().id}_{iface.chain2[0].get_parent().id}"
                rows.append({
                    "jobs": job,
                    "model_used": m,
                    "interface": label,
                    "iptm_ptm": float(confidence.iptm_ptm) if confidence.iptm_ptm is not None else float('nan'),
                    "iptm": float(confidence.iptm) if confidence.iptm is not None else float('nan'),
                    "ptm": float(confidence.ptm) if confidence.ptm is not None else float('nan'),
                    "confidence_score": float(confidence.confidence_score) if confidence.confidence_score is not None else float('nan'),
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
                    "interface_sc": iface.sc,
                    "interface_area": iface.int_area,
                    "interface_solv_en": iface.int_solv_en,
                })
            logging.info(f"processed model: {m} via {parser.name}")
        except Exception as e:
            logging.error(f"error processing model {m}: {e}")

    out = d / "interfaces.csv"
    with out.open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
        else:
            f.write("")
    logging.info(f"wrote {out}")
    return out

def _discover_run_dirs(root: Path) -> list[Path]:
    """Walk a directory tree and collect directories that look like supported runs."""
    results: list[Path] = []
    for d in [p for p in root.rglob("*") if p.is_dir()] + ([root] if root.is_dir() else []):
        try:
            # pick_parser raises if not supported; we do not need the result here
            pick_parser(d)
            results.append(d)
        except Exception:
            continue
    uniq = sorted(set(p.resolve() for p in results), key=lambda p: (len(p.parts), str(p)))
    return [Path(p) for p in uniq]

def process_many(
    paths: list[str],
    contact_thresh: float,
    pae_filter: float,
    models_to_analyse: str,
    recursive: bool = False,
    summary_csv: str | None = None,
) -> Path | None:
    """
    Process one or more directories. Optionally recurse into nested directories
    to find supported runs. If summary_csv is provided, aggregate all per-run
    interface rows into a single CSV at that path and return it.
    """
    if not paths:
        logging.warning("no input paths provided")
        return None

    # Resolve set of run directories to process
    run_dirs: list[Path] = []
    for p in paths:
        rp = Path(p).resolve()
        if not rp.exists():
            logging.warning(f"path does not exist: {rp}")
            continue
        if recursive and rp.is_dir():
            run_dirs.extend(_discover_run_dirs(rp))
        else:
            # Try direct path as a run directory
            try:
                pick_parser(rp)
                run_dirs.append(rp)
            except Exception:
                # If not a direct run dir and recursive not requested, skip
                logging.warning(f"no supported run detected at {rp} (use --recursive to search within)")
                continue

    # Deduplicate
    seen = set()
    unique_run_dirs: list[Path] = []
    for d in run_dirs:
        r = d.resolve()
        if r not in seen:
            seen.add(r); unique_run_dirs.append(d)

    if not unique_run_dirs:
        logging.warning("no runnable directories found")
        return None

    # Process each run and collect rows for summary if requested
    aggregated_rows: list[dict] = []
    for d in unique_run_dirs:
        try:
            out_path = process(str(d), contact_thresh, pae_filter, models_to_analyse)
            if summary_csv:
                # Read rows back to aggregate
                try:
                    with Path(out_path).open() as f:
                        reader = csv.DictReader(f)
                        aggregated_rows.extend(list(reader))
                except Exception as e:
                    logging.error(f"failed reading {out_path} for aggregation: {e}")
        except Exception as e:
            logging.error(f"failed processing {d}: {e}")

    if summary_csv:
        if not aggregated_rows:
            logging.info("no rows to write to summary; skipping creation")
            return None
        # Compute union of all keys to accommodate AF2/AF3 variations
        fieldnames: list[str] = []
        seen_fields = set()
        for row in aggregated_rows:
            for k in row.keys():
                if k not in seen_fields:
                    seen_fields.add(k); fieldnames.append(k)
        summary_path = Path(summary_csv).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in aggregated_rows:
                w.writerow({k: row.get(k, "") for k in fieldnames})
        logging.info(f"wrote summary {summary_path} ({len(aggregated_rows)} rows from {len(unique_run_dirs)} runs)")
        return summary_path

    return None
