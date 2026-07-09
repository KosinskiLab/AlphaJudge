#!/usr/bin/env python3
"""Evaluate AlphaJudge contact-probability scores on a labelled prediction tree.

The benchmark tree is expected to look like:

    <root>/<organism>/<af2|af3>/<pos_pairs|neg_pairs>/predictions/<run_dir>

Rows are written without modifying the prediction directories. By default the
script scores all chain pairs, including pairs without coordinate-detected
contacts, because contact probability is a model-confidence quantity and can be
defined even when the sampled structure has no interface.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from alphajudge.complex import Complex
from alphajudge.interface import Interface
from alphajudge.meta_score import interface_meta_score
from alphajudge.parsers import pick_parser

logger = logging.getLogger("contact_probability_benchmark")

DEFAULT_ROOT = Path(
    "/g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26/predictions"
)
DEFAULT_SCORES = (
    "interface_contact_prob_max",
    "interface_contact_prob_top10_mean",
    "interface_expected_contacts",
    "iptm",
    "iptm_ptm",
    "confidence_score",
    "pDockQ/mpDockQ",
    "interface_pDockQ2",
    "interface_ipSAE",
    "interface_LIS",
    "interface_cLIS",
    "interface_iLIS",
    "interface_contact_pairs",
    "average_interface_pae",
)
LOWER_IS_BETTER = {"average_interface_pae"}


def _safe_float(value) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def _infer_metadata(run_dir: Path, root: Path) -> dict[str, str | int]:
    parts = run_dir.resolve().relative_to(root.resolve()).parts
    organism = parts[0] if len(parts) > 0 else ""
    backend = parts[1] if len(parts) > 1 else ""
    pair_set = parts[2] if len(parts) > 2 else ""
    if pair_set.startswith("pos"):
        label = 1
    elif pair_set.startswith("neg"):
        label = 0
    else:
        label = -1
    return {
        "organism": organism,
        "backend": backend,
        "pair_set": pair_set,
        "label": label,
        "pair_id": run_dir.name,
        "source_dir": str(run_dir.resolve()),
    }


def _discover_run_dirs(
    root: Path,
    limit_per_group: int | None = None,
    *,
    organism_filter: str | None = None,
    backend_filter: str | None = None,
    pair_set_filter: str | None = None,
) -> list[Path]:
    candidates: list[Path] = []
    for predictions_dir in sorted(root.glob("*/*/*/predictions")):
        if not predictions_dir.is_dir():
            continue
        try:
            organism, backend, pair_set, _ = predictions_dir.relative_to(root).parts
        except ValueError:
            organism, backend, pair_set = "", "", ""
        if organism_filter and organism != organism_filter:
            continue
        if backend_filter and backend != backend_filter:
            continue
        if pair_set_filter and pair_set != pair_set_filter:
            continue
        group_hits: list[Path] = []
        for child in sorted(p for p in predictions_dir.iterdir() if p.is_dir()):
            try:
                pick_parser(child)
            except Exception:
                continue
            group_hits.append(child)
            if limit_per_group is not None and len(group_hits) >= limit_per_group:
                break
        candidates.extend(group_hits)

    if candidates:
        return candidates

    for child in sorted(p for p in root.rglob("*") if p.is_dir()):
        try:
            pick_parser(child)
        except Exception:
            continue
        metadata = _infer_metadata(child, root)
        if organism_filter and metadata["organism"] != organism_filter:
            continue
        if backend_filter and metadata["backend"] != backend_filter:
            continue
        if pair_set_filter and metadata["pair_set"] != pair_set_filter:
            continue
        candidates.append(child)
    return candidates[:limit_per_group] if limit_per_group is not None else candidates


def _interface_label(iface: Interface) -> str:
    return f"{iface.chain1[0].get_parent().id}_{iface.chain2[0].get_parent().id}"


def _all_chain_pair_interfaces(comp: Complex) -> Iterable[Interface]:
    for i in range(len(comp._chains)):  # noqa: SLF001 - benchmark helper
        for j in range(i + 1, len(comp._chains)):  # noqa: SLF001
            yield Interface(comp._chains[i], comp._chains[j], comp)  # noqa: SLF001


def _row_for_interface(job: str, model: str, confidence, global_score: float, iface: Interface) -> dict:
    pd2, _ = iface.pDockQ2()
    iptm_val = iface.iptm_chainpair if iface.iptm_chainpair is not None else confidence.iptm
    row = {
        "jobs": job,
        "model_used": model,
        "interface": _interface_label(iface),
        "iptm_ptm": float(confidence.iptm_ptm) if confidence.iptm_ptm is not None else float("nan"),
        "iptm": float(iptm_val) if iptm_val is not None else float("nan"),
        "ptm": float(confidence.ptm) if confidence.ptm is not None else float("nan"),
        "confidence_score": float(confidence.confidence_score)
        if confidence.confidence_score is not None
        else float("nan"),
        "pDockQ/mpDockQ": global_score,
        "average_interface_pae": iface.average_interface_pae,
        "interface_average_plddt": iface.average_interface_plddt,
        "interface_num_intf_residues": iface.num_intf_residues,
        "interface_contact_pairs": iface.contact_pairs,
        "interface_contact_prob_source": confidence.contact_prob_source or "",
        "interface_contact_prob_max": iface.contact_prob_max,
        "interface_contact_prob_top10_mean": iface.contact_prob_top10_mean,
        "interface_expected_contacts": iface.expected_contacts,
        "interface_score": iface.score_complex,
        "interface_pDockQ2": pd2,
        "interface_ipSAE": iface.ipsae(),
        "interface_LIS": iface.lis(),
        "interface_cLIS": iface.clis(),
        "interface_iLIS": iface.ilis(),
    }
    row["interface_meta_score"] = interface_meta_score(row)
    return row


def score_run(
    run_dir: Path,
    root: Path,
    *,
    contact_thresh: float,
    pae_filter: float,
    ipsae_pae_cutoff: float,
    models_to_analyse: str,
    chain_pairs: str,
) -> tuple[list[dict], str | None]:
    metadata = _infer_metadata(run_dir, root)
    try:
        parser = pick_parser(run_dir)
        run = parser.parse_run(run_dir)
        models = [run.order[0]] if models_to_analyse == "best" else list(run.order)
    except Exception as e:
        return [], f"{run_dir}: parser setup failed: {e}"

    rows: list[dict] = []
    for model in models:
        try:
            structure, confidence = run.load_model(model)
            comp = Complex(structure, confidence, contact_thresh, pae_filter, ipsae_pae_cutoff)
            global_score = (
                comp.mpDockQ
                if comp.num_chains > 2
                else (comp.interfaces[0].pDockQ if comp.interfaces else float("nan"))
            )
            interfaces = (
                list(_all_chain_pair_interfaces(comp))
                if chain_pairs == "all"
                else list(comp.interfaces)
            )
            for iface in interfaces:
                if chain_pairs == "detected" and iface.num_intf_residues == 0:
                    continue
                if (
                    iface.num_intf_residues > 0
                    and math.isfinite(iface.average_interface_pae)
                    and iface.average_interface_pae > pae_filter
                ):
                    continue
                row = {**metadata, "parser": parser.name}
                row.update(_row_for_interface(run_dir.name, model, confidence, global_score, iface))
                rows.append(row)
        except Exception as e:
            return rows, f"{run_dir} {model}: scoring failed: {e}"
    return rows, None


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(values.size, dtype=float)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    n_pos = int(np.count_nonzero(labels == 1))
    n_neg = int(np.count_nonzero(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    sum_pos = float(np.sum(ranks[labels == 1]))
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    n_pos = int(np.count_nonzero(labels == 1))
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores)
    y = labels[order]
    precision = np.cumsum(y == 1) / (np.arange(y.size) + 1)
    return float(np.sum(precision[y == 1]) / n_pos)


def benchmark_metrics(rows: list[dict], score_names: Iterable[str]) -> list[dict]:
    groups: list[tuple[str, str, list[dict]]] = [("all", "all", rows)]
    for backend in sorted({str(r.get("backend", "")) for r in rows}):
        groups.append((backend, "all", [r for r in rows if r.get("backend") == backend]))
    for organism in sorted({str(r.get("organism", "")) for r in rows}):
        groups.append(("all", organism, [r for r in rows if r.get("organism") == organism]))
    for backend in sorted({str(r.get("backend", "")) for r in rows}):
        for organism in sorted({str(r.get("organism", "")) for r in rows}):
            subset = [
                r
                for r in rows
                if r.get("backend") == backend and r.get("organism") == organism
            ]
            groups.append((backend, organism, subset))

    metrics: list[dict] = []
    for backend, organism, subset in groups:
        labels_all = np.asarray([int(r.get("label", -1)) for r in subset], dtype=int)
        valid_labels = (labels_all == 0) | (labels_all == 1)
        for score_name in score_names:
            raw_scores = np.asarray([_safe_float(r.get(score_name)) for r in subset], dtype=float)
            valid = valid_labels & np.isfinite(raw_scores)
            labels = labels_all[valid]
            scores = raw_scores[valid]
            if score_name in LOWER_IS_BETTER:
                scores = -scores
            metrics.append(
                {
                    "backend": backend,
                    "organism": organism,
                    "score": score_name,
                    "n": int(valid.sum()),
                    "n_pos": int(np.count_nonzero(labels == 1)),
                    "n_neg": int(np.count_nonzero(labels == 0)),
                    "auroc": _auroc(labels, scores),
                    "average_precision": _average_precision(labels, scores),
                }
            )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--out", required=True, help="Output per-interface/per-chain-pair CSV.")
    parser.add_argument("--metrics-out", required=True, help="Output AUROC/AP summary CSV.")
    parser.add_argument("--models-to-analyse", default="best", choices=("best", "all"))
    parser.add_argument("--chain-pairs", default="all", choices=("all", "detected"))
    parser.add_argument("--contact-thresh", type=float, default=8.0)
    parser.add_argument("--pae-filter", type=float, default=100.0)
    parser.add_argument("--ipsae-pae-cutoff", type=float, default=10.0)
    parser.add_argument("--limit-per-group", type=int, default=None)
    parser.add_argument("--organism", default=None)
    parser.add_argument("--backend", default=None, choices=("af2", "af3"))
    parser.add_argument("--pair-set", default=None, choices=("pos_pairs", "neg_pairs"))
    parser.add_argument("--scores", nargs="*", default=list(DEFAULT_SCORES))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s: %(message)s")

    root = Path(args.predictions_root)
    run_dirs = _discover_run_dirs(
        root,
        args.limit_per_group,
        organism_filter=args.organism,
        backend_filter=args.backend,
        pair_set_filter=args.pair_set,
    )
    logger.info("found %d run directories under %s", len(run_dirs), root)

    rows: list[dict] = []
    errors: list[str] = []
    for i, run_dir in enumerate(run_dirs, start=1):
        logger.info("scoring %d/%d %s", i, len(run_dirs), run_dir)
        scored, error = score_run(
            run_dir,
            root,
            contact_thresh=args.contact_thresh,
            pae_filter=args.pae_filter,
            ipsae_pae_cutoff=args.ipsae_pae_cutoff,
            models_to_analyse=args.models_to_analyse,
            chain_pairs=args.chain_pairs,
        )
        rows.extend(scored)
        if error:
            logger.warning(error)
            errors.append(error)

    _write_rows(Path(args.out), rows)
    _write_rows(Path(args.metrics_out), benchmark_metrics(rows, args.scores))

    if errors:
        errors_path = Path(args.metrics_out).with_name(Path(args.metrics_out).stem + "_errors.txt")
        errors_path.write_text("\n".join(errors) + "\n")
        logger.warning("wrote %d errors to %s", len(errors), errors_path)
    logger.info("wrote %d score rows to %s", len(rows), args.out)
    logger.info("wrote metric summary to %s", args.metrics_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
