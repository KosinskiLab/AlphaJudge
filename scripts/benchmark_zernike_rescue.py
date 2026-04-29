#!/usr/bin/env python3
"""Benchmark pure Zernike candidates against the AlphaJudge benchmark tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from scipy.stats import rankdata

from alphajudge.biophysics.zernike import (
    ATOM_GAUSSIAN,
    DEFAULT_DISTANCE,
    DEFAULT_PADDING,
    DEFAULT_PROXIMITY_LENGTH_SCALE,
    DEFAULT_SIGMA,
    DEFAULT_SURFACE_DENSITY,
    DEFAULT_SURFACE_TRIM_CUTOFF,
    GAUSSIAN_REPRESENTATIONS,
    SURFACE_BINARY,
    SURFACE_GAUSSIAN,
    SURFACE_PROXIMITY_GAUSSIAN,
    ZernikeSpec,
    zernike_coefficients,
    zernike_descriptor_prefix_length,
    zernike_grids,
    zernike_shape_complementarity,
    zernike_similarity_from_coefficients,
)

BENCH_ROOT_DEFAULT = Path(
    "/g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26/predictions"
)
ORGANISMS = ("arabidopsis", "ecoli", "human", "yeast")
BACKENDS = ("af2", "af3")
PAIRSETS = ("pos_pairs", "neg_pairs")
SMOKE_SAMPLE_SIZE = 100
RUNTIME_SAMPLE_SIZE = 200
MAX_SWEEP_ORDER = 12
AF3_FAILURE_QUANTILE = 0.90
TOP_N = 50


class GridCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _hash_payload(payload: dict) -> str:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _cache_path(self, row: dict, spec: ZernikeSpec) -> Path:
        payload = {
            "model_file": str(Path(str(row["model_file"])).resolve()),
            "interface": str(row["interface"]),
            "representation": spec.representation,
            "grid_size": int(spec.grid_size),
            "sigma": float(spec.sigma) if spec.representation in GAUSSIAN_REPRESENTATIONS else None,
            "padding": float(spec.padding),
            "distance": float(spec.distance),
            "surface_density": float(spec.surface_density),
            "surface_trim_cutoff": float(spec.surface_trim_cutoff),
            "proximity_length_scale": float(spec.proximity_length_scale)
            if spec.representation == SURFACE_PROXIMITY_GAUSSIAN
            else None,
        }
        return self.cache_dir / f"{self._hash_payload(payload)}.npz"

    def get_or_build(self, row: dict, spec: ZernikeSpec) -> tuple[np.ndarray, np.ndarray]:
        path = self._cache_path(row, spec)
        if path.exists():
            with np.load(path) as payload:
                self.hits += 1
                return payload["grid1"], payload["grid2"]

        residues1, residues2 = load_interface_residues(str(row["model_file"]), str(row["interface"]))
        grid1, grid2 = zernike_grids(
            residues1,
            residues2,
            representation=spec.representation,
            distance=spec.distance,
            grid_size=spec.grid_size,
            sigma=spec.sigma,
            padding=spec.padding,
            surface_density=spec.surface_density,
            surface_trim_cutoff=spec.surface_trim_cutoff,
            proximity_length_scale=spec.proximity_length_scale,
        )
        np.savez_compressed(path, grid1=grid1, grid2=grid2)
        self.misses += 1
        return grid1, grid2


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


def coerce_best_row(row: dict[str, str]) -> dict:
    out: dict[str, object] = {}
    for key, value in row.items():
        if key in {"jobs", "model_used", "interface"}:
            out[key] = value
            continue
        numeric = safe_float(value)
        out[key] = numeric if math.isfinite(numeric) or str(value).strip().lower() in {"nan", ""} else value
    return out


def candidate_groups(candidates: Iterable[ZernikeSpec]) -> dict[tuple, list[ZernikeSpec]]:
    groups: dict[tuple, list[ZernikeSpec]] = defaultdict(list)
    for spec in candidates:
        key = (
            spec.representation,
            int(spec.grid_size),
            float(spec.sigma) if spec.representation in GAUSSIAN_REPRESENTATIONS else None,
            float(spec.padding),
            float(spec.distance),
            float(spec.surface_density),
            float(spec.surface_trim_cutoff),
            float(spec.proximity_length_scale)
            if spec.representation == SURFACE_PROXIMITY_GAUSSIAN
            else None,
        )
        groups[key].append(spec)
    return groups


def build_full_candidates() -> list[ZernikeSpec]:
    candidates = [ZernikeSpec()]
    for grid_size in (32, 48):
        for order in (8, 10, 12):
            candidates.append(
                ZernikeSpec(
                    representation=SURFACE_BINARY,
                    grid_size=grid_size,
                    order=order,
                    sigma=DEFAULT_SIGMA,
                )
            )
            for sigma in (1.0, 1.5):
                candidates.append(
                    ZernikeSpec(
                        representation=SURFACE_GAUSSIAN,
                        grid_size=grid_size,
                        order=order,
                        sigma=sigma,
                    )
                )
                candidates.append(
                    ZernikeSpec(
                        representation=SURFACE_PROXIMITY_GAUSSIAN,
                        grid_size=grid_size,
                        order=order,
                        sigma=sigma,
                    )
                )
    return candidates


def build_smoke_candidates() -> list[ZernikeSpec]:
    return [
        ZernikeSpec(),
        ZernikeSpec(
            representation=SURFACE_GAUSSIAN,
            grid_size=32,
            order=8,
            sigma=1.5,
        ),
    ]


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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_benchmark_rows(
    bench_root: Path,
    *,
    manifest_tag: str | None = None,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    skipped: list[dict] = []

    for organism in ORGANISMS:
        for backend in BACKENDS:
            for pairset in PAIRSETS:
                group_root = bench_root / organism / backend / pairset
                best_csv = group_root / "best_interfaces.csv"
                if not best_csv.exists():
                    continue

                try:
                    manifest_path = find_manifest(group_root, manifest_tag)
                    manifest_rows = {
                        row["pair"]: row
                        for row in read_csv_rows(manifest_path)
                        if row.get("pair")
                    }
                except Exception as exc:
                    skipped.append(
                        {
                            "organism": organism,
                            "backend": backend,
                            "pairset": pairset,
                            "pair": "",
                            "reason": f"manifest_error: {exc}",
                        }
                    )
                    continue

                seen_pairs: set[str] = set()
                for best_row_raw in read_csv_rows(best_csv):
                    pair = str(best_row_raw.get("jobs", "")).strip()
                    if not pair or pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)

                    manifest_row = manifest_rows.get(pair)
                    if manifest_row is None:
                        skipped.append(
                            {
                                "organism": organism,
                                "backend": backend,
                                "pairset": pairset,
                                "pair": pair,
                                "reason": "missing_manifest_row",
                            }
                        )
                        continue

                    model_file = str(manifest_row.get("model_file", "")).strip()
                    interface_label = str(best_row_raw.get("interface", "")).strip()
                    if not model_file or not interface_label:
                        skipped.append(
                            {
                                "organism": organism,
                                "backend": backend,
                                "pairset": pairset,
                                "pair": pair,
                                "reason": "missing_model_or_interface",
                            }
                        )
                        continue

                    model_path = Path(model_file)
                    if not model_path.exists():
                        skipped.append(
                            {
                                "organism": organism,
                                "backend": backend,
                                "pairset": pairset,
                                "pair": pair,
                                "reason": f"missing_model_file:{model_path}",
                            }
                        )
                        continue

                    row = coerce_best_row(best_row_raw)
                    row.update(
                        {
                            "pair": pair,
                            "jobs": pair,
                            "organism": organism,
                            "backend": backend,
                            "pairset": pairset,
                            "label": "positive" if pairset == "pos_pairs" else "negative",
                            "group_root": str(group_root),
                            "manifest_path": str(manifest_path),
                            "pair_dir": str(manifest_row.get("pair_dir", "")),
                            "model_file": str(model_path),
                            "selected_model_used": str(manifest_row.get("model_used", "")).strip(),
                            "manifest_run_status": str(manifest_row.get("run_status", "")).strip(),
                        }
                    )
                    rows.append(row)
    rows.sort(key=lambda row: (row["organism"], row["backend"], row["pairset"], row["pair"]))
    return rows, skipped


def dataset_cell_counts(rows: Iterable[dict]) -> list[dict]:
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        counts[(str(row["organism"]), str(row["backend"]), str(row["pairset"]))] += 1
    out = []
    for (organism, backend, pairset), count in sorted(counts.items()):
        out.append(
            {
                "organism": organism,
                "backend": backend,
                "pairset": pairset,
                "count": count,
            }
        )
    return out


def balanced_sample(rows: list[dict], sample_size: int) -> list[dict]:
    if sample_size <= 0 or len(rows) <= sample_size:
        return list(rows)

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (str(row["organism"]), str(row["backend"]), str(row["pairset"]))
        grouped[key].append(row)
    ordered_keys = sorted(grouped)
    for key in ordered_keys:
        grouped[key].sort(key=lambda row: (str(row["pair"]), str(row["interface"])))

    base = sample_size // len(ordered_keys)
    remainder = sample_size % len(ordered_keys)
    selected: list[dict] = []
    leftovers: dict[tuple[str, str, str], list[dict]] = {}

    for key in ordered_keys:
        take = min(base, len(grouped[key]))
        selected.extend(grouped[key][:take])
        leftovers[key] = grouped[key][take:]

    key_index = 0
    while len(selected) < sample_size:
        key = ordered_keys[key_index % len(ordered_keys)]
        if leftovers[key]:
            selected.append(leftovers[key].pop(0))
        key_index += 1
        if key_index > len(ordered_keys) * max(1, sample_size):
            break

    if len(selected) < sample_size:
        remaining = []
        for key in ordered_keys:
            remaining.extend(leftovers[key])
        remaining.sort(key=lambda row: (str(row["organism"]), str(row["backend"]), str(row["pairset"]), str(row["pair"])))
        selected.extend(remaining[: sample_size - len(selected)])

    return selected[:sample_size]


def parser_for_model(path: str):
    model_path = Path(path)
    return MMCIFParser(QUIET=True) if model_path.suffix.lower() == ".cif" else PDBParser(QUIET=True)


@lru_cache(maxsize=128)
def load_interface_residues(model_file: str, interface_label: str) -> tuple[tuple, tuple]:
    structure = parser_for_model(model_file).get_structure("model", model_file)
    model = next(structure.get_models())
    chains = {chain.id: tuple(chain) for chain in model.get_chains()}
    if "_" not in interface_label:
        raise ValueError(f"interface label must contain '_': {interface_label}")
    chain1_id, chain2_id = interface_label.split("_", 1)
    if chain1_id not in chains or chain2_id not in chains:
        raise KeyError(
            f"Interface {interface_label} not present in {model_file}; found {sorted(chains)}"
        )
    return chains[chain1_id], chains[chain2_id]


def descriptor_prefix(coeffs: np.ndarray, order: int) -> np.ndarray:
    return coeffs[: zernike_descriptor_prefix_length(order)]


def evaluate_candidates(
    rows: list[dict],
    candidates: list[ZernikeSpec],
    *,
    cache_dir: Path,
) -> tuple[dict[str, list[dict]], dict]:
    grouped_candidates = candidate_groups(candidates)
    cache = GridCache(cache_dir)
    results: dict[str, list[dict]] = {spec.candidate_id(): [] for spec in candidates}
    status_counts: dict[str, int] = defaultdict(int)

    for row in rows:
        row_id = {
            "pair": row["pair"],
            "organism": row["organism"],
            "backend": row["backend"],
            "pairset": row["pairset"],
            "label": row["label"],
            "interface": row["interface"],
            "jobs": row["jobs"],
            "model_file": row["model_file"],
            "interface_sc": row.get("interface_sc", float("nan")),
            "average_interface_pae": row.get("average_interface_pae", float("nan")),
            "interface_area": row.get("interface_area", float("nan")),
            "interface_contact_pairs": row.get("interface_contact_pairs", float("nan")),
            "interface_num_intf_residues": row.get("interface_num_intf_residues", float("nan")),
            "interface_average_plddt": row.get("interface_average_plddt", float("nan")),
        }

        for _, spec_group in grouped_candidates.items():
            anchor = spec_group[0]
            try:
                grid1, grid2 = cache.get_or_build(row, anchor)
                coeff1 = zernike_coefficients(grid1, MAX_SWEEP_ORDER)
                coeff2 = zernike_coefficients(grid2, MAX_SWEEP_ORDER)
                for spec in spec_group:
                    score = zernike_similarity_from_coefficients(
                        descriptor_prefix(coeff1, spec.order),
                        descriptor_prefix(coeff2, spec.order),
                    )
                    out_row = dict(row_id)
                    out_row.update(
                        {
                            "candidate_id": spec.candidate_id(),
                            "representation": spec.representation,
                            "grid_size": spec.grid_size,
                            "order": spec.order,
                            "sigma": spec.sigma if spec.representation in GAUSSIAN_REPRESENTATIONS else "",
                            "surface_density": spec.surface_density
                            if spec.representation in {SURFACE_BINARY, SURFACE_GAUSSIAN, SURFACE_PROXIMITY_GAUSSIAN}
                            else "",
                            "surface_trim_cutoff": spec.surface_trim_cutoff
                            if spec.representation in {SURFACE_BINARY, SURFACE_GAUSSIAN, SURFACE_PROXIMITY_GAUSSIAN}
                            else "",
                            "proximity_length_scale": spec.proximity_length_scale
                            if spec.representation == SURFACE_PROXIMITY_GAUSSIAN
                            else "",
                            "padding": spec.padding,
                            "distance": spec.distance,
                            "candidate_score": score,
                            "candidate_status": "success",
                        }
                    )
                    results[spec.candidate_id()].append(out_row)
                    status_counts["success"] += 1
            except Exception as exc:
                for spec in spec_group:
                    out_row = dict(row_id)
                    out_row.update(
                        {
                            "candidate_id": spec.candidate_id(),
                            "representation": spec.representation,
                            "grid_size": spec.grid_size,
                            "order": spec.order,
                            "sigma": spec.sigma if spec.representation in GAUSSIAN_REPRESENTATIONS else "",
                            "surface_density": spec.surface_density
                            if spec.representation in {SURFACE_BINARY, SURFACE_GAUSSIAN, SURFACE_PROXIMITY_GAUSSIAN}
                            else "",
                            "surface_trim_cutoff": spec.surface_trim_cutoff
                            if spec.representation in {SURFACE_BINARY, SURFACE_GAUSSIAN, SURFACE_PROXIMITY_GAUSSIAN}
                            else "",
                            "proximity_length_scale": spec.proximity_length_scale
                            if spec.representation == SURFACE_PROXIMITY_GAUSSIAN
                            else "",
                            "padding": spec.padding,
                            "distance": spec.distance,
                            "candidate_score": float("nan"),
                            "candidate_status": f"error:{exc}",
                        }
                    )
                    results[spec.candidate_id()].append(out_row)
                    status_counts["error"] += 1

    cache_meta = {
        "grid_cache_hits": cache.hits,
        "grid_cache_misses": cache.misses,
        "status_counts": dict(sorted(status_counts.items())),
    }
    return results, cache_meta


def auroc_from_rows(rows: list[dict], field: str) -> float:
    pos = np.asarray(
        [safe_float(row[field]) for row in rows if row["label"] == "positive" and math.isfinite(safe_float(row[field]))],
        dtype=float,
    )
    neg = np.asarray(
        [safe_float(row[field]) for row in rows if row["label"] == "negative" and math.isfinite(safe_float(row[field]))],
        dtype=float,
    )
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    all_scores = np.concatenate([pos, neg])
    ranks = np.asarray(rankdata(all_scores, method="average"), dtype=float)
    rank_sum_pos = np.sum(ranks[: len(pos)])
    return float((rank_sum_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def average_precision_from_rows(rows: list[dict], field: str) -> float:
    values = [
        (1 if row["label"] == "positive" else 0, safe_float(row[field]))
        for row in rows
        if math.isfinite(safe_float(row[field]))
    ]
    if not values:
        return float("nan")
    positives = sum(label for label, _ in values)
    negatives = len(values) - positives
    if positives == 0 or negatives == 0:
        return float("nan")

    values.sort(key=lambda item: item[1], reverse=True)
    tp = 0
    fp = 0
    precision_sum = 0.0
    for label, _ in values:
        if label == 1:
            tp += 1
            precision_sum += tp / (tp + fp)
        else:
            fp += 1
    return float(precision_sum / positives)


def finite_scores(rows: list[dict], field: str) -> list[float]:
    return [safe_float(row[field]) for row in rows if math.isfinite(safe_float(row[field]))]


def score_quantile(rows: list[dict], field: str, q: float) -> float:
    values = np.asarray(finite_scores(rows, field), dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


def compute_scope_metrics(rows: list[dict], field: str) -> tuple[float, float, int, int]:
    auroc = auroc_from_rows(rows, field)
    ap = average_precision_from_rows(rows, field)
    n_pos = sum(1 for row in rows if row["label"] == "positive" and math.isfinite(safe_float(row[field])))
    n_neg = sum(1 for row in rows if row["label"] == "negative" and math.isfinite(safe_float(row[field])))
    return auroc, ap, n_pos, n_neg


def grouped_by(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        out[tuple(row[key] for key in keys)].append(row)
    return out


def annotate_failure_slice(candidate_rows: list[dict]) -> list[dict]:
    negatives_by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in candidate_rows:
        if row["label"] == "negative":
            negatives_by_cell[(row["organism"], row["backend"])].append(row)

    negative_sc_thresholds = {
        key: score_quantile(group, "interface_sc", AF3_FAILURE_QUANTILE)
        for key, group in negatives_by_cell.items()
    }
    negative_candidate_thresholds = {
        key: score_quantile(group, "candidate_score", AF3_FAILURE_QUANTILE)
        for key, group in negatives_by_cell.items()
    }

    annotated = []
    for row in candidate_rows:
        key = (row["organism"], row["backend"])
        baseline_neg90 = negative_sc_thresholds.get(key, float("nan"))
        candidate_neg90 = negative_candidate_thresholds.get(key, float("nan"))
        in_failure_slice = (
            row["backend"] == "af3"
            and row["label"] == "positive"
            and math.isfinite(safe_float(row["interface_sc"]))
            and math.isfinite(baseline_neg90)
            and safe_float(row["interface_sc"]) <= baseline_neg90
        )
        candidate_score = safe_float(row["candidate_score"])
        rescued = (
            in_failure_slice
            and math.isfinite(candidate_score)
            and math.isfinite(candidate_neg90)
            and candidate_score > candidate_neg90
        )
        out_row = dict(row)
        out_row.update(
            {
                "baseline_sc_neg90": baseline_neg90,
                "candidate_neg90": candidate_neg90,
                "is_af3_sc_failure_slice": int(in_failure_slice),
                "rescued_af3_failure_positive": int(rescued),
                "candidate_margin_to_neg90": candidate_score - candidate_neg90
                if math.isfinite(candidate_score) and math.isfinite(candidate_neg90)
                else float("nan"),
            }
        )
        annotated.append(out_row)
    return annotated


def runtime_sample_rows(rows: list[dict], sample_size: int) -> list[dict]:
    return balanced_sample(rows, sample_size)


def measure_candidate_runtime(rows: list[dict], spec: ZernikeSpec) -> dict[str, float]:
    timings = []
    residues_cache: dict[tuple[str, str], tuple[tuple, tuple]] = {}
    for row in rows:
        key = (str(row["model_file"]), str(row["interface"]))
        if key not in residues_cache:
            residues_cache[key] = load_interface_residues(*key)
        residues1, residues2 = residues_cache[key]
        start = time.perf_counter()
        _ = zernike_shape_complementarity(
            residues1,
            residues2,
            representation=spec.representation,
            distance=spec.distance,
            grid_size=spec.grid_size,
            order=spec.order,
            sigma=spec.sigma,
            padding=spec.padding,
            surface_density=spec.surface_density,
            surface_trim_cutoff=spec.surface_trim_cutoff,
            proximity_length_scale=spec.proximity_length_scale,
        )
        timings.append(time.perf_counter() - start)

    if not timings:
        return {
            "median_runtime_sec": float("nan"),
            "mean_runtime_sec": float("nan"),
            "max_runtime_sec": float("nan"),
            "runtime_n": 0,
        }
    return {
        "median_runtime_sec": float(np.median(timings)),
        "mean_runtime_sec": float(np.mean(timings)),
        "max_runtime_sec": float(np.max(timings)),
        "runtime_n": len(timings),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    seen = set(fieldnames)
    for row in rows[1:]:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def summarize_candidates(
    candidate_results: dict[str, list[dict]],
    candidates: list[ZernikeSpec],
    *,
    runtime_rows: list[dict],
) -> tuple[list[dict], list[dict], list[dict], dict[str, list[dict]]]:
    baseline_sc_rows = next(
        rows
        for spec, rows in candidate_results.items()
        if spec == ZernikeSpec().candidate_id()
    )

    baseline_cell_aurocs = {
        key: auroc_from_rows(group, "interface_sc")
        for key, group in grouped_by(baseline_sc_rows, ("organism", "backend")).items()
    }
    baseline_sc_global = auroc_from_rows(baseline_sc_rows, "interface_sc")
    baseline_sc_af3 = auroc_from_rows(
        [row for row in baseline_sc_rows if row["backend"] == "af3"],
        "interface_sc",
    )

    summary_rows: list[dict] = []
    metric_rows: list[dict] = []
    annotated_outputs: dict[str, list[dict]] = {}
    runtime_rows_out: list[dict] = []

    baseline_failure_slice = [
        row
        for row in annotate_failure_slice(baseline_sc_rows)
        if row["is_af3_sc_failure_slice"] == 1
    ]
    baseline_atom_failure_rescue_rate = float(
        np.mean([row["rescued_af3_failure_positive"] for row in baseline_failure_slice])
    ) if baseline_failure_slice else 0.0
    baseline_sc_failure_rescue_rate = 0.0

    runtime_by_candidate: dict[str, dict[str, float]] = {}
    for spec in candidates:
        stats = measure_candidate_runtime(runtime_rows, spec)
        runtime_by_candidate[spec.candidate_id()] = stats
        runtime_rows_out.append({"candidate_id": spec.candidate_id(), **stats})

    baseline_runtime = runtime_by_candidate[ZernikeSpec().candidate_id()]["median_runtime_sec"]

    for spec in candidates:
        candidate_id = spec.candidate_id()
        rows = annotate_failure_slice(candidate_results[candidate_id])
        annotated_outputs[candidate_id] = rows

        pooled_global = compute_scope_metrics(rows, "candidate_score")
        pooled_af2 = compute_scope_metrics(
            [row for row in rows if row["backend"] == "af2"],
            "candidate_score",
        )
        pooled_af3 = compute_scope_metrics(
            [row for row in rows if row["backend"] == "af3"],
            "candidate_score",
        )

        cell_deltas = []
        for key, group in grouped_by(rows, ("organism", "backend")).items():
            auroc, ap, n_pos, n_neg = compute_scope_metrics(group, "candidate_score")
            baseline_auroc = baseline_cell_aurocs.get(key, float("nan"))
            delta = auroc - baseline_auroc if math.isfinite(auroc) and math.isfinite(baseline_auroc) else float("nan")
            metric_rows.append(
                {
                    "candidate_id": candidate_id,
                    "scope": "cell",
                    "organism": key[0],
                    "backend": key[1],
                    "auroc": auroc,
                    "average_precision": ap,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "baseline_sc_auroc": baseline_auroc,
                    "delta_auroc_vs_sc": delta,
                }
            )
            if math.isfinite(delta):
                cell_deltas.append(delta)

        for scope_name, scope_rows in (
            ("global", rows),
            ("af2", [row for row in rows if row["backend"] == "af2"]),
            ("af3", [row for row in rows if row["backend"] == "af3"]),
        ):
            auroc, ap, n_pos, n_neg = compute_scope_metrics(scope_rows, "candidate_score")
            baseline_auroc = (
                baseline_sc_global
                if scope_name == "global"
                else auroc_from_rows(scope_rows, "interface_sc")
            )
            metric_rows.append(
                {
                    "candidate_id": candidate_id,
                    "scope": scope_name,
                    "organism": "",
                    "backend": scope_name if scope_name in {"af2", "af3"} else "",
                    "auroc": auroc,
                    "average_precision": ap,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "baseline_sc_auroc": baseline_auroc,
                    "delta_auroc_vs_sc": auroc - baseline_auroc
                    if math.isfinite(auroc) and math.isfinite(baseline_auroc)
                    else float("nan"),
                }
            )

        failure_slice_rows = [
            row
            for row in rows
            if row["is_af3_sc_failure_slice"] == 1
        ]
        rescue_rate = float(
            np.mean([row["rescued_af3_failure_positive"] for row in failure_slice_rows])
        ) if failure_slice_rows else float("nan")

        runtime_stats = runtime_by_candidate[candidate_id]
        runtime_ok = (
            math.isfinite(baseline_runtime)
            and math.isfinite(runtime_stats["median_runtime_sec"])
            and runtime_stats["median_runtime_sec"] <= 5.0 * baseline_runtime
        )
        guardrail_pass = (
            math.isfinite(pooled_global[0])
            and pooled_global[0] >= baseline_sc_global
            and all(delta >= -0.01 for delta in cell_deltas)
        )
        accepted_for_production = (
            guardrail_pass
            and runtime_ok
            and math.isfinite(rescue_rate)
            and rescue_rate > baseline_sc_failure_rescue_rate
            and math.isfinite(pooled_af3[0])
            and pooled_af3[0] > baseline_sc_af3
        )

        summary_rows.append(
            {
                "candidate_id": candidate_id,
                "representation": spec.representation,
                "grid_size": spec.grid_size,
                "order": spec.order,
                "sigma": spec.sigma if spec.representation in GAUSSIAN_REPRESENTATIONS else "",
                "surface_density": spec.surface_density
                if spec.representation in {SURFACE_BINARY, SURFACE_GAUSSIAN, SURFACE_PROXIMITY_GAUSSIAN}
                else "",
                "surface_trim_cutoff": spec.surface_trim_cutoff
                if spec.representation in {SURFACE_BINARY, SURFACE_GAUSSIAN, SURFACE_PROXIMITY_GAUSSIAN}
                else "",
                "proximity_length_scale": spec.proximity_length_scale
                if spec.representation == SURFACE_PROXIMITY_GAUSSIAN
                else "",
                "padding": spec.padding,
                "distance": spec.distance,
                "af3_failure_rescue_rate": rescue_rate,
                "pooled_af3_auroc": pooled_af3[0],
                "pooled_af3_average_precision": pooled_af3[1],
                "pooled_all_auroc": pooled_global[0],
                "pooled_all_average_precision": pooled_global[1],
                "pooled_af2_auroc": pooled_af2[0],
                "median_runtime_sec": runtime_stats["median_runtime_sec"],
                "mean_runtime_sec": runtime_stats["mean_runtime_sec"],
                "runtime_ok": int(runtime_ok),
                "guardrail_pass": int(guardrail_pass),
                "accepted_for_production": int(accepted_for_production),
                "baseline_sc_global_auroc": baseline_sc_global,
                "baseline_sc_af3_auroc": baseline_sc_af3,
                "baseline_sc_failure_rescue_rate": baseline_sc_failure_rescue_rate,
                "baseline_atom_failure_rescue_rate": baseline_atom_failure_rescue_rate,
            }
        )

    summary_rows.sort(
        key=lambda row: (
            -(safe_float(row["af3_failure_rescue_rate"]) if math.isfinite(safe_float(row["af3_failure_rescue_rate"])) else -1.0),
            -(safe_float(row["pooled_af3_auroc"]) if math.isfinite(safe_float(row["pooled_af3_auroc"])) else -1.0),
            -(safe_float(row["pooled_all_auroc"]) if math.isfinite(safe_float(row["pooled_all_auroc"])) else -1.0),
            safe_float(row["median_runtime_sec"]) if math.isfinite(safe_float(row["median_runtime_sec"])) else float("inf"),
        )
    )

    first_pass = True
    for idx, row in enumerate(summary_rows, start=1):
        row["rank"] = idx
        if first_pass and row["accepted_for_production"] == 1:
            row["recommended_candidate"] = 1
            first_pass = False
        else:
            row["recommended_candidate"] = 0

    return summary_rows, metric_rows, runtime_rows_out, annotated_outputs


def write_candidate_reports(
    out_dir: Path,
    annotated_outputs: dict[str, list[dict]],
) -> None:
    score_dir = out_dir / "scores"
    top_dir = out_dir / "top_hits"
    for candidate_id, rows in annotated_outputs.items():
        write_csv(score_dir / f"{candidate_id}.csv", rows)

        failure_slice = [
            row for row in rows if row["is_af3_sc_failure_slice"] == 1
        ]
        rescued = [
            row for row in failure_slice if row["rescued_af3_failure_positive"] == 1
        ]
        rescued.sort(
            key=lambda row: (
                -safe_float(row["candidate_margin_to_neg90"]),
                -safe_float(row["candidate_score"]),
                str(row["pair"]),
            )
        )
        missed = [
            row for row in failure_slice if row["rescued_af3_failure_positive"] == 0
        ]
        missed.sort(
            key=lambda row: (
                -safe_float(row["candidate_margin_to_neg90"]),
                -safe_float(row["candidate_score"]),
                str(row["pair"]),
            )
        )
        write_csv(top_dir / f"{candidate_id}__rescued.csv", rescued[:TOP_N])
        write_csv(top_dir / f"{candidate_id}__still_missed.csv", missed[:TOP_N])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", default=str(BENCH_ROOT_DEFAULT))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manifest-tag", default=None)
    parser.add_argument(
        "--mode",
        choices=["full", "smoke"],
        default="full",
        help="Smoke mode uses a fixed mixed sample and a smaller candidate set.",
    )
    parser.add_argument("--smoke-sample-size", type=int, default=SMOKE_SAMPLE_SIZE)
    parser.add_argument("--runtime-sample-size", type=int, default=RUNTIME_SAMPLE_SIZE)
    args = parser.parse_args()

    bench_root = Path(args.bench_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows, skipped = load_benchmark_rows(bench_root, manifest_tag=args.manifest_tag)
    if not all_rows:
        raise SystemExit("No benchmark rows discovered.")

    candidates = build_smoke_candidates() if args.mode == "smoke" else build_full_candidates()
    benchmark_rows = (
        balanced_sample(all_rows, args.smoke_sample_size)
        if args.mode == "smoke"
        else all_rows
    )
    runtime_rows = runtime_sample_rows(all_rows, args.runtime_sample_size)

    candidate_results, cache_meta = evaluate_candidates(
        benchmark_rows,
        candidates,
        cache_dir=out_dir / "cache" / "grids",
    )
    summary_rows, metric_rows, runtime_summary_rows, annotated_outputs = summarize_candidates(
        candidate_results,
        candidates,
        runtime_rows=runtime_rows,
    )

    write_csv(out_dir / "dataset_rows.csv", benchmark_rows)
    write_csv(out_dir / "skipped_rows.csv", skipped)
    write_csv(out_dir / "dataset_cell_counts.csv", dataset_cell_counts(benchmark_rows))
    write_csv(out_dir / "candidate_summary.csv", summary_rows)
    write_csv(out_dir / "candidate_metrics.csv", metric_rows)
    write_csv(out_dir / "candidate_runtime_summary.csv", runtime_summary_rows)
    write_candidate_reports(out_dir, annotated_outputs)
    write_json(
        out_dir / "run_metadata.json",
        {
            "bench_root": str(bench_root),
            "mode": args.mode,
            "manifest_tag": args.manifest_tag,
            "rows_total_discovered": len(all_rows),
            "rows_scored": len(benchmark_rows),
            "runtime_rows": len(runtime_rows),
            "candidate_count": len(candidates),
            "candidates": [asdict(spec) for spec in candidates],
            **cache_meta,
        },
    )

    print(f"wrote {out_dir / 'candidate_summary.csv'}")
    print(f"wrote {out_dir / 'candidate_metrics.csv'}")
    print(f"wrote {out_dir / 'candidate_runtime_summary.csv'}")
    print(f"wrote {out_dir / 'run_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
