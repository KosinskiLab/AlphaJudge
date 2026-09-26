from __future__ import annotations

from pathlib import Path
import csv
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from .parsers import pick_parser
from .complex import Complex
from .confidence import IPTM_SCOPE_CHAIN_PAIR, IPTM_SCOPE_GLOBAL
from .meta_score import interface_meta_score
from .report import generate_per_run_report, render_pae_png

logger = logging.getLogger(__name__)


# Cached per-run CSVs from older releases are safe to reuse only when they
# contain every field introduced by the current output contract.
_REQUIRED_CACHE_COLUMNS = frozenset({
    "interface_ccc", "interface_expected_contacts", "global_confidence_scope", "iptm_scope",
})


def _float_or_nan(value) -> float:
    return float(value) if value is not None else float("nan")


def _interface_row(
    job: str, model: str, iface, confidence, global_score: float, skip_biophysical_scores: bool
) -> dict:
    """One output CSV row for a scored interface."""
    pd2, _ = iface.pDockQ2()
    pair_iptm = iface.iptm_chainpair
    row = {
        "jobs": job,
        "model_used": model,
        "interface": iface.label,
        "global_confidence_scope": confidence.global_confidence_scope,
        "iptm_scope": IPTM_SCOPE_CHAIN_PAIR if pair_iptm is not None else IPTM_SCOPE_GLOBAL,
        "iptm_ptm": _float_or_nan(confidence.iptm_ptm),
        "iptm": _float_or_nan(pair_iptm if pair_iptm is not None else confidence.iptm),
        "ptm": _float_or_nan(confidence.ptm),
        "confidence_score": _float_or_nan(confidence.confidence_score),
        "pDockQ/mpDockQ": global_score,
        "average_interface_pae": iface.average_interface_pae,
        "interface_average_plddt": iface.average_interface_plddt,
        "interface_num_intf_residues": iface.num_intf_residues,
        "interface_polar": iface.polar,
        "interface_hydrophobic": iface.hydrophobic,
        "interface_charged": iface.charged,
        "interface_contact_pairs": iface.contact_pairs,
        "interface_contact_prob_source": confidence.contact_prob_source or "",
        "interface_contact_prob_max": iface.contact_prob_max,
        "interface_contact_prob_top10_mean": iface.contact_prob_top10_mean,
        "interface_expected_contacts": iface.expected_contacts,
        "interface_ccc": iface.ccc,
        "interface_score": iface.score_complex,
        "interface_pDockQ2": pd2,
        "interface_ipSAE": iface.ipsae(),
        "interface_LIS": iface.lis(),
        "interface_cLIS": iface.clis(),
        "interface_iLIS": iface.ilis(),
    }
    if not skip_biophysical_scores:
        row.update({
            "interface_hb": iface.hb,
            "interface_sb": iface.sb,
            "interface_ss": iface.ss,
            "interface_sc": iface.sc,
            "interface_area": iface.int_area,
            "interface_solv_en": iface.int_solv_en,
        })
    row["interface_meta_score"] = interface_meta_score(row)
    return row


def process(
    directory: str,
    contact_thresh: float,
    pae_filter: float,
    models_to_analyse: str,
    ipsae_pae_cutoff: float = 10.0,
    *,
    per_run_csv_name: str = "interfaces.csv",
    skip_pae_png: bool = False,
    skip_biophysical_scores: bool = False,
) -> Path:
    d = Path(directory)
    parser = pick_parser(d)
    run = parser.parse_run(d)
    models = [run.order[0]] if models_to_analyse == "best" else run.order
    job = d.resolve().name

    rows: list[dict] = []
    # Diagnostics so an empty CSV can be explained rather than written silently.
    models_processed = 0
    total_interfaces = 0
    dropped_by_pae = 0
    for m in models:
        try:
            structure, confidence = run.load_model(m)
            comp = Complex(structure, confidence, contact_thresh, pae_filter, ipsae_pae_cutoff)
            models_processed += 1
            total_interfaces += len(comp.interfaces)

            global_score = (
                comp.mpDockQ
                if comp.num_chains > 2
                else (comp.interfaces[0].pDockQ if comp.interfaces else float("nan"))
            )

            for iface in comp.interfaces:
                if iface.num_intf_residues == 0:
                    continue
                if iface.average_interface_pae > pae_filter:
                    dropped_by_pae += 1
                    continue
                rows.append(
                    _interface_row(job, m, iface, confidence, global_score, skip_biophysical_scores)
                )

            if not skip_pae_png:
                pae_png = d / f"pae_{m}.png"
                try:
                    render_pae_png(
                        pae_png,
                        confidence.pae_matrix,
                        chain_boundaries=comp.chain_boundaries,
                        figsize=(8, 8),
                    )
                    logger.info(f"wrote {pae_png}")
                except Exception as e:
                    logger.error(f"Could not create PAE heatmap {pae_png}: {e}")

            logger.info(f"processed model: {m} via {parser.name}")
        except Exception as e:
            logger.error(f"error processing model {m}: {e}")

    out = d / per_run_csv_name
    out.parent.mkdir(parents=True, exist_ok=True)  # per_run_csv_name may hold a subdir
    if not rows:
        # Explain *why* the CSV is empty instead of writing a silent zero-byte file
        # (see https://github.com/KosinskiLab/AlphaJudge/issues/17). The common case
        # for heterodimers is that AlphaFold placed the chains without any inter-chain
        # contact within --contact_thresh, so no interface is detected.
        if models_processed == 0:
            reason = "no model could be loaded/processed"
        elif total_interfaces == 0:
            reason = (
                f"no inter-chain contacts within contact_thresh={contact_thresh} A "
                f"(chains have no detectable interface); try a larger --contact_thresh "
                f"or check that the model is actually a complex"
            )
        elif dropped_by_pae:
            reason = (
                f"all {dropped_by_pae} detected interface(s) were filtered out by "
                f"pae_filter={pae_filter}; try a larger --pae_filter"
            )
        else:
            reason = "all detected interfaces had zero interface residues"
        logger.warning(f"no interface rows for {job}: {reason}; writing empty {out}")

    with out.open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    logger.info(f"wrote {out}")
    return out


def _is_run_dir(d: Path) -> bool:
    try:
        pick_parser(d)
    except Exception:
        return False
    return True


def _discover_run_dirs(root: Path) -> list[Path]:
    """Walk a directory tree and collect directories that look like supported runs."""
    if not root.is_dir():
        return []
    candidates = [root, *(p for p in root.rglob("*") if p.is_dir())]
    runs = {p.resolve() for p in candidates if _is_run_dir(p)}
    # Stable order, shallow first
    return sorted(runs, key=lambda p: (len(p.parts), str(p)))


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _read_reusable_csv(path: Path) -> list[dict] | None:
    """Return cached rows only when the file satisfies the current schema."""
    rows = _read_csv_rows(path)
    if not rows:
        logger.info(f"existing {path} is empty; recomputing")
        return None

    missing = sorted(_REQUIRED_CACHE_COLUMNS - set(rows[0]))
    if missing:
        logger.info(
            f"existing {path} is missing required column(s) "
            f"{', '.join(missing)}; recomputing"
        )
        return None
    return rows


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
    skip_biophysical_scores: bool,
    write_per_run_report: bool = False,
) -> tuple[str, list[dict]]:
    """
    Worker: process a single run dir (or reuse its per-run CSV) and return
    (run_dir, rows_for_summary); rows are only collected when summary_csv is set.
    """
    d = Path(d_str)
    csv_path = d / per_run_csv_name

    # Reuse a precomputed CSV only if its header satisfies the current output
    # contract. This prevents an upgrade from silently aggregating stale rows.
    rows = None
    if csv_path.exists() and not force_recompute:
        try:
            rows = _read_reusable_csv(csv_path)
        except Exception as e:
            logger.warning(f"could not reuse {csv_path}; recomputing: {e}")
        if rows is not None and summary_csv is not None:
            logger.info(f"reused existing {csv_path} for aggregation")
        elif rows is not None:
            logger.info(f"reused existing {csv_path}; skipping recompute")

    if rows is None:
        csv_path = process(
            d_str,
            contact_thresh,
            pae_filter,
            models_to_analyse,
            ipsae_pae_cutoff,
            per_run_csv_name=per_run_csv_name,
            skip_pae_png=skip_pae_png,
            skip_biophysical_scores=skip_biophysical_scores,
        )

    if write_per_run_report:
        try:
            generate_per_run_report(d, csv_name=per_run_csv_name)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"per-run report failed for {d}: {e}")

    if summary_csv is None:
        return d_str, []
    if rows is None:
        try:
            rows = _read_csv_rows(csv_path)
        except Exception as e:
            logger.error(f"failed reading {csv_path} for aggregation: {e}")
            return d_str, []
    # Absolute source_dir lets the aggregate report locate per-run side files
    # (PAE PNGs, etc.) from the summary CSV.
    source_dir = str(d.resolve())
    for r in rows:
        r.setdefault("source_dir", source_dir)
    return d_str, rows


def _iptm_sort_key(row: dict) -> float:
    try:
        val = float(row.get("iptm"))
    except (TypeError, ValueError):
        return float("-inf")
    return val if val == val else float("-inf")  # NaN -> -inf


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
    skip_biophysical_scores: bool = False,
    write_per_run_report: bool = False,
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

    # Resolve set of run directories to process (dict keeps first-seen order)
    run_dirs: dict[Path, None] = {}
    for p in paths:
        rp = Path(p).resolve()
        if not rp.exists():
            logger.warning(f"path does not exist: {rp}")
        elif recursive and rp.is_dir():
            run_dirs.update(dict.fromkeys(_discover_run_dirs(rp)))
        elif _is_run_dir(rp):
            run_dirs[rp] = None
        else:
            logger.warning(
                f"no supported run detected at {rp} (use --recursive to search within)"
            )

    if not run_dirs:
        logger.warning("no runnable directories found")
        return None

    if cores <= 0:
        cores = os.cpu_count() or 1
    cores = min(cores, len(run_dirs))

    worker = partial(
        _process_one_run,
        contact_thresh=contact_thresh,
        pae_filter=pae_filter,
        models_to_analyse=models_to_analyse,
        summary_csv=summary_csv,
        ipsae_pae_cutoff=ipsae_pae_cutoff,
        force_recompute=force_recompute,
        per_run_csv_name=per_run_csv_name,
        skip_pae_png=skip_pae_png,
        skip_biophysical_scores=skip_biophysical_scores,
        write_per_run_report=write_per_run_report,
    )
    aggregated_rows: list[dict] = []
    logger.info(f"Processing {len(run_dirs)} runs with {cores} cores")
    if cores == 1:
        for d in run_dirs:
            try:
                aggregated_rows.extend(worker(str(d))[1])
            except Exception as e:
                logger.error(f"failed processing {d}: {e}")
    else:
        # Important: logging from multiple processes can interleave; acceptable.
        with ProcessPoolExecutor(max_workers=cores) as ex:
            for fut in as_completed([ex.submit(worker, str(d)) for d in run_dirs]):
                try:
                    aggregated_rows.extend(fut.result()[1])
                except Exception as e:
                    logger.error(f"worker failed: {e}")

    if not summary_csv:
        return None
    if not aggregated_rows:
        logger.info("no rows to write to summary; skipping creation")
        return None

    # Union of all keys to accommodate AF2/AF3 variations
    fieldnames = list(dict.fromkeys(k for row in aggregated_rows for k in row))
    aggregated_rows.sort(key=_iptm_sort_key, reverse=True)

    summary_path = Path(summary_csv).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        w.writeheader()
        w.writerows(aggregated_rows)

    logger.info(
        f"wrote summary {summary_path} ({len(aggregated_rows)} rows from {len(run_dirs)} runs)"
    )
    return summary_path
