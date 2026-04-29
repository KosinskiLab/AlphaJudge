#!/usr/bin/env python3
"""Benchmark low-pass Zernike candidates head-to-head against interface_sc."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from scipy.stats import rankdata

from alphajudge.biophysics.sc import shape_complementarity
from alphajudge.biophysics.zernike import (
    ATOM_GAUSSIAN,
    DEFAULT_SURFACE_DENSITY,
    DEFAULT_SURFACE_PROBE_RADIUS,
    GAUSSIAN_REPRESENTATIONS,
    GAUSSIAN_WEIGHTED_SCORE,
    GAP_ZERNIKE_BANDPASS_SCORE,
    GAP_ZERNIKE_NONUNIFORM_SCORE,
    GAP_ZERNIKE_RATIO_SCORE,
    GAP_ZERNIKE_WEIGHTED_SCORE,
    GRID_SCORE_MODES,
    HARD_CUTOFF_SCORE,
    JOINT_LOW_ORDER_RATIO_SCORE,
    JOINT_REPRESENTATIONS,
    JOINT_RESIDUE_BEAD_GAUSSIAN,
    NORMAL_GAP_FIELD_SCORE,
    NORMAL_GAP_REPRESENTATIONS,
    NORMAL_GAP_SCORE_MODES,
    RESIDUE_BEAD_GAUSSIAN,
    SURFACE_GAUSSIAN,
    SURFACE_NORMAL_GAP,
    SURFACE_REPRESENTATIONS,
    SHARED_GRID_OVERLAP_SCORE,
    WeightedPointCloud,
    ZernikeSpec,
    fit_order_value,
    zernike_candidate_family,
    zernike_coefficient_bundle_from_grids,
    zernike_gap_coefficient_bundle_from_grids,
    zernike_grids_from_point_clouds,
    zernike_normal_gap_coefficient_bundle,
    zernike_normal_gap_field_bundle,
    zernike_point_clouds,
    zernike_score_from_coefficients,
    zernike_score_from_gap_coefficients,
    zernike_score_from_normal_gap_coefficients,
    zernike_source_representation,
)

BENCH_ROOT_DEFAULT = Path(
    "/g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26/predictions"
)
ORGANISMS = ("arabidopsis", "ecoli", "human", "yeast")
BACKENDS = ("af2", "af3")
PAIRSETS = ("pos_pairs", "neg_pairs")
SMOKE_SAMPLE_SIZE = 100
RUNTIME_SAMPLE_SIZE = 200
ROBUSTNESS_SAMPLE_SIZE = 50
ROBUSTNESS_JITTER_STD = 1.0
MAX_SWEEP_ORDER = 12
AF3_FAILURE_QUANTILE = 0.90
TOP_N = 50
SC_BASELINE_ID = "interface_sc"
DIAGNOSTIC_HARD_PER_CLASS = 6
DIAGNOSTIC_AF3_SAMPLE_SIZE = 200
SATURATION_SCORE_THRESHOLD = 0.95
SATURATION_FRACTION_THRESHOLD = 0.80
MIN_MEDIAN_SEPARATION = 0.02

PROTEIN_BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}
NA_BACKBONE_ATOMS = {
    "P",
    "OP1",
    "OP2",
    "OP3",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "C1'",
    "O2'",
    "O5*",
    "C5*",
    "C4*",
    "O4*",
    "C3*",
    "O3*",
    "C2*",
    "C1*",
    "O2*",
}


def hash_payload(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


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


def unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def atomic_savez_compressed(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        tmp_path.replace(path)
    finally:
        unlink_if_exists(tmp_path)


def git_metadata(repo_root: Path) -> dict:
    def run_git(args: list[str]) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()

    try:
        commit = run_git(["rev-parse", "HEAD"])
    except Exception:
        commit = ""
    try:
        branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        branch = ""
    try:
        tracked_status = run_git(["status", "--short", "--untracked-files=no"]).splitlines()
    except Exception:
        tracked_status = []
    try:
        full_status = run_git(["status", "--short", "--untracked-files=normal"]).splitlines()
    except Exception:
        full_status = []

    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_tracked_dirty": int(bool(tracked_status)),
        "git_tracked_status": tracked_status,
        "git_untracked_count": len([line for line in full_status if line.startswith("?? ")]),
    }


def _point_cloud_payload(row: dict, spec: ZernikeSpec) -> dict:
    source = zernike_source_representation(spec.representation)
    return {
        "model_file": str(Path(str(row["model_file"])).resolve()),
        "interface": str(row["interface"]),
        "representation": source,
        "distance": float(spec.distance),
        "surface_density": float(spec.surface_density) if source in SURFACE_REPRESENTATIONS else None,
        "surface_trim_cutoff": float(spec.surface_trim_cutoff) if source in SURFACE_REPRESENTATIONS else None,
        "surface_probe_radius": float(spec.surface_probe_radius) if source in SURFACE_REPRESENTATIONS else None,
        "proximity_length_scale": float(spec.proximity_length_scale)
        if source == "surface_proximity_gaussian"
        else None,
    }


def _grid_payload(row: dict, spec: ZernikeSpec) -> dict:
    payload = _point_cloud_payload(row, spec)
    payload.update(
        {
            "grid_size": int(spec.grid_size),
            "sigma": float(spec.sigma) if spec.representation in GAUSSIAN_REPRESENTATIONS else None,
            "padding": float(spec.padding),
            "grid_builder": "binary"
            if zernike_source_representation(spec.representation) == "surface_binary"
            else "density",
        }
    )
    return payload


def _coeff_payload(row: dict, spec: ZernikeSpec) -> dict:
    payload = _grid_payload(row, spec)
    payload.update(
        {
            "fit_order": fit_order_value(spec),
            "coefficient_builder": "zernike_coeff_gap_bundle_v1",
        }
    )
    return payload


class PointCloudCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _cache_path(self, row: dict, spec: ZernikeSpec) -> Path:
        return self.cache_dir / f"{hash_payload(_point_cloud_payload(row, spec))}.npz"

    def get_or_build(self, row: dict, spec: ZernikeSpec) -> tuple[WeightedPointCloud, WeightedPointCloud]:
        path = self._cache_path(row, spec)
        if path.exists():
            try:
                with np.load(path) as payload:
                    self.hits += 1
                    return (
                        WeightedPointCloud(
                            np.asarray(payload["points1"], dtype=float),
                            np.asarray(payload["weights1"], dtype=float),
                        ),
                        WeightedPointCloud(
                            np.asarray(payload["points2"], dtype=float),
                            np.asarray(payload["weights2"], dtype=float),
                        ),
                    )
            except Exception:
                unlink_if_exists(path)

        residues1, residues2 = load_interface_residues(str(row["model_file"]), str(row["interface"]))
        side1, side2 = zernike_point_clouds(
            residues1,
            residues2,
            representation=spec.representation,
            distance=spec.distance,
            surface_density=spec.surface_density,
            surface_trim_cutoff=spec.surface_trim_cutoff,
            surface_probe_radius=spec.surface_probe_radius,
            proximity_length_scale=spec.proximity_length_scale,
        )
        atomic_savez_compressed(
            path,
            points1=side1.points,
            weights1=side1.weights,
            points2=side2.points,
            weights2=side2.weights,
        )
        self.misses += 1
        return side1, side2


class GridCache:
    def __init__(self, cache_dir: Path, point_cloud_cache: PointCloudCache):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.point_cloud_cache = point_cloud_cache
        self.hits = 0
        self.misses = 0

    def _cache_path(self, row: dict, spec: ZernikeSpec) -> Path:
        return self.cache_dir / f"{hash_payload(_grid_payload(row, spec))}.npz"

    def get_or_build(self, row: dict, spec: ZernikeSpec) -> tuple[np.ndarray, np.ndarray]:
        path = self._cache_path(row, spec)
        if path.exists():
            try:
                with np.load(path) as payload:
                    self.hits += 1
                    return payload["grid1"], payload["grid2"]
            except Exception:
                unlink_if_exists(path)

        side1, side2 = self.point_cloud_cache.get_or_build(row, spec)
        grid1, grid2 = zernike_grids_from_point_clouds(
            side1,
            side2,
            representation=spec.representation,
            grid_size=spec.grid_size,
            sigma=spec.sigma,
            padding=spec.padding,
        )
        atomic_savez_compressed(path, grid1=grid1, grid2=grid2)
        self.misses += 1
        return grid1, grid2


class CoefficientCache:
    def __init__(self, cache_dir: Path, grid_cache: GridCache):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.grid_cache = grid_cache
        self.hits = 0
        self.misses = 0

    def _cache_path(self, row: dict, spec: ZernikeSpec) -> Path:
        return self.cache_dir / f"{hash_payload(_coeff_payload(row, spec))}.npz"

    def get_or_build(self, row: dict, spec: ZernikeSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        path = self._cache_path(row, spec)
        if path.exists():
            try:
                with np.load(path) as payload:
                    self.hits += 1
                    return (
                        payload["coeff1"],
                        payload["coeff2"],
                        payload["joint_coeff"],
                        payload["gap_coeff"],
                        float(payload["grid_overlap"]),
                    )
            except Exception:
                unlink_if_exists(path)

        grid1, grid2 = self.grid_cache.get_or_build(row, spec)
        coeff1, coeff2, joint_coeff = zernike_coefficient_bundle_from_grids(
            grid1,
            grid2,
            fit_order_value(spec),
        )
        gap_coeff, grid_overlap = zernike_gap_coefficient_bundle_from_grids(
            grid1,
            grid2,
            fit_order_value(spec),
        )
        atomic_savez_compressed(
            path,
            coeff1=coeff1,
            coeff2=coeff2,
            joint_coeff=joint_coeff,
            gap_coeff=gap_coeff,
            grid_overlap=np.asarray(grid_overlap, dtype=float),
        )
        self.misses += 1
        return coeff1, coeff2, joint_coeff, gap_coeff, grid_overlap


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
        source = zernike_source_representation(spec.representation)
        key = (
            source,
            int(spec.grid_size),
            float(spec.sigma) if spec.representation in GAUSSIAN_REPRESENTATIONS else None,
            float(spec.padding),
            float(spec.distance),
            float(spec.surface_density) if source in SURFACE_REPRESENTATIONS else None,
            float(spec.surface_trim_cutoff) if source in SURFACE_REPRESENTATIONS else None,
            float(spec.surface_probe_radius) if source in SURFACE_REPRESENTATIONS else None,
            float(spec.proximity_length_scale) if source == "surface_proximity_gaussian" else None,
            float(spec.normal_gap_good_scale) if source in NORMAL_GAP_REPRESENTATIONS else None,
            float(spec.normal_gap_far_scale) if source in NORMAL_GAP_REPRESENTATIONS else None,
            fit_order_value(spec),
        )
        groups[key].append(spec)
    return groups


def _atom_cosine_baseline_spec() -> ZernikeSpec:
    return ZernikeSpec(
        representation=ATOM_GAUSSIAN,
        grid_size=32,
        order=10,
        sigma=1.5,
        score_mode=HARD_CUTOFF_SCORE,
        fit_order=MAX_SWEEP_ORDER,
    )


def _tuned_atom_gap_overlap_spec() -> ZernikeSpec:
    return ZernikeSpec(
        representation=ATOM_GAUSSIAN,
        grid_size=32,
        order=0,
        sigma=1.5,
        score_mode=SHARED_GRID_OVERLAP_SCORE,
        fit_order=MAX_SWEEP_ORDER,
    )


def tuned_atom_gap_penalty_specs() -> list[ZernikeSpec]:
    common = {
        "representation": ATOM_GAUSSIAN,
        "grid_size": 32,
        "sigma": 1.5,
        "fit_order": MAX_SWEEP_ORDER,
    }
    return [
        ZernikeSpec(
            **common,
            order=0,
            score_mode=GAP_ZERNIKE_NONUNIFORM_SCORE,
        ),
        ZernikeSpec(
            **common,
            order=2,
            score_mode=GAP_ZERNIKE_NONUNIFORM_SCORE,
        ),
        ZernikeSpec(
            **common,
            order=4,
            score_mode=GAP_ZERNIKE_BANDPASS_SCORE,
        ),
        ZernikeSpec(
            **common,
            order=6,
            score_mode=GAP_ZERNIKE_BANDPASS_SCORE,
        ),
        ZernikeSpec(
            **common,
            order=8,
            score_mode=GAP_ZERNIKE_BANDPASS_SCORE,
        ),
    ]


def tuned_surface_normal_gap_specs() -> list[ZernikeSpec]:
    common = {
        "representation": SURFACE_NORMAL_GAP,
        "grid_size": 32,
        "sigma": 1.5,
        "surface_density": 3.0,
        "surface_probe_radius": 2.3,
        "surface_trim_cutoff": 3.0,
        "score_mode": NORMAL_GAP_FIELD_SCORE,
        "fit_order": MAX_SWEEP_ORDER,
    }
    return [
        ZernikeSpec(**common, order=4),
        ZernikeSpec(**common, order=6),
        ZernikeSpec(**common, order=8),
    ]


def build_cosine_diagnostic_candidates() -> list[ZernikeSpec]:
    return [
        _atom_cosine_baseline_spec(),
        ZernikeSpec(
            representation=RESIDUE_BEAD_GAUSSIAN,
            grid_size=24,
            order=8,
            sigma=2.0,
            score_mode=GAUSSIAN_WEIGHTED_SCORE,
            fit_order=MAX_SWEEP_ORDER,
            order_decay_n0=4.0,
        ),
        ZernikeSpec(
            representation=SURFACE_GAUSSIAN,
            grid_size=24,
            order=6,
            sigma=1.5,
            surface_density=3.0,
            surface_probe_radius=2.3,
            score_mode=HARD_CUTOFF_SCORE,
            fit_order=MAX_SWEEP_ORDER,
        ),
        ZernikeSpec(
            representation=JOINT_RESIDUE_BEAD_GAUSSIAN,
            grid_size=24,
            order=6,
            sigma=2.0,
            score_mode=JOINT_LOW_ORDER_RATIO_SCORE,
            fit_order=MAX_SWEEP_ORDER,
        ),
    ]


def _append_gap_candidates(
    candidates: list[ZernikeSpec],
    *,
    representation: str,
    grid_size: int,
    sigma: float,
    surface_density: float = DEFAULT_SURFACE_DENSITY,
    surface_probe_radius: float = DEFAULT_SURFACE_PROBE_RADIUS,
) -> None:
    common = {
        "representation": representation,
        "grid_size": grid_size,
        "sigma": sigma,
        "surface_density": surface_density,
        "surface_probe_radius": surface_probe_radius,
        "fit_order": MAX_SWEEP_ORDER,
    }
    candidates.append(
        ZernikeSpec(
            **common,
            order=0,
            score_mode=SHARED_GRID_OVERLAP_SCORE,
        )
    )
    for order in (4, 6, 8):
        candidates.append(
            ZernikeSpec(
                **common,
                order=order,
                score_mode=GAP_ZERNIKE_RATIO_SCORE,
            )
        )
        candidates.append(
            ZernikeSpec(
                **common,
                order=order,
                score_mode=GAP_ZERNIKE_WEIGHTED_SCORE,
                order_decay_n0=4.0,
            )
        )


def build_full_candidates() -> list[ZernikeSpec]:
    candidates: list[ZernikeSpec] = build_cosine_diagnostic_candidates()
    candidates.append(_tuned_atom_gap_overlap_spec())
    candidates.extend(tuned_atom_gap_penalty_specs())
    candidates.extend(tuned_surface_normal_gap_specs())
    for grid_size in (24, 32):
        for sigma in (2.0, 3.0):
            _append_gap_candidates(candidates, representation=ATOM_GAUSSIAN, grid_size=grid_size, sigma=sigma)
            _append_gap_candidates(
                candidates,
                representation=RESIDUE_BEAD_GAUSSIAN,
                grid_size=grid_size,
                sigma=sigma,
            )
        for sigma in (1.5, 2.5):
            _append_gap_candidates(
                candidates,
                representation=SURFACE_GAUSSIAN,
                grid_size=grid_size,
                sigma=sigma,
                surface_density=3.0,
                surface_probe_radius=2.3,
            )

    return candidates


def build_smoke_candidates() -> list[ZernikeSpec]:
    return [
        _atom_cosine_baseline_spec(),
        _tuned_atom_gap_overlap_spec(),
        *tuned_atom_gap_penalty_specs(),
        *tuned_surface_normal_gap_specs(),
        ZernikeSpec(
            representation=RESIDUE_BEAD_GAUSSIAN,
            grid_size=24,
            order=8,
            sigma=2.0,
            score_mode=GAUSSIAN_WEIGHTED_SCORE,
            fit_order=MAX_SWEEP_ORDER,
            order_decay_n0=4.0,
        ),
        ZernikeSpec(
            representation=SURFACE_GAUSSIAN,
            grid_size=24,
            order=6,
            sigma=1.5,
            surface_density=3.0,
            surface_probe_radius=2.3,
            score_mode=HARD_CUTOFF_SCORE,
            fit_order=MAX_SWEEP_ORDER,
        ),
        ZernikeSpec(
            representation=JOINT_RESIDUE_BEAD_GAUSSIAN,
            grid_size=24,
            order=6,
            sigma=2.0,
            score_mode=JOINT_LOW_ORDER_RATIO_SCORE,
            fit_order=MAX_SWEEP_ORDER,
        ),
        ZernikeSpec(
            representation=ATOM_GAUSSIAN,
            grid_size=24,
            order=6,
            sigma=2.0,
            score_mode=GAP_ZERNIKE_RATIO_SCORE,
            fit_order=MAX_SWEEP_ORDER,
        ),
        ZernikeSpec(
            representation=RESIDUE_BEAD_GAUSSIAN,
            grid_size=24,
            order=6,
            sigma=2.0,
            score_mode=GAP_ZERNIKE_WEIGHTED_SCORE,
            fit_order=MAX_SWEEP_ORDER,
            order_decay_n0=4.0,
        ),
        ZernikeSpec(
            representation=SURFACE_GAUSSIAN,
            grid_size=24,
            order=0,
            sigma=1.5,
            surface_density=3.0,
            surface_probe_radius=2.3,
            score_mode=SHARED_GRID_OVERLAP_SCORE,
            fit_order=MAX_SWEEP_ORDER,
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
    organisms: Iterable[str] = ORGANISMS,
    backends: Iterable[str] = BACKENDS,
    pairsets: Iterable[str] = PAIRSETS,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    skipped: list[dict] = []

    for organism in organisms:
        for backend in backends:
            for pairset in pairsets:
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
    if sample_size <= 0:
        return []
    if len(rows) <= sample_size:
        return list(rows)

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (str(row["organism"]), str(row["backend"]), str(row["pairset"]))
        grouped[key].append(row)
    ordered_keys = sorted(grouped)
    for key in ordered_keys:
        grouped[key].sort(key=lambda row: (str(row["pair"]), str(row["interface"])))

    base = sample_size // len(ordered_keys)
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


def _copy_for_diagnostic_slice(row: dict, slice_name: str) -> dict:
    out = dict(row)
    out["diagnostic_slice"] = slice_name
    return out


def human_af3_hard_slice(rows: list[dict], per_class: int = DIAGNOSTIC_HARD_PER_CLASS) -> list[dict]:
    positives = [
        row
        for row in rows
        if row["organism"] == "human" and row["backend"] == "af3" and row["label"] == "positive"
    ]
    negatives = [
        row
        for row in rows
        if row["organism"] == "human" and row["backend"] == "af3" and row["label"] == "negative"
    ]
    positives.sort(key=lambda row: (safe_float(row.get("interface_sc", float("nan"))), str(row["pair"])))
    negatives.sort(key=lambda row: (-safe_float(row.get("interface_sc", float("nan"))), str(row["pair"])))
    selected = positives[:per_class] + negatives[:per_class]
    return [_copy_for_diagnostic_slice(row, "human_af3_hard_low_sc_pos_high_sc_neg") for row in selected]


def af3_mixed_diagnostic_sample(rows: list[dict], sample_size: int = DIAGNOSTIC_AF3_SAMPLE_SIZE) -> list[dict]:
    af3_rows = [row for row in rows if row["backend"] == "af3"]
    if sample_size <= 0 or len(af3_rows) <= sample_size:
        return [_copy_for_diagnostic_slice(row, "af3_mixed_200") for row in af3_rows]

    positives = [row for row in af3_rows if row["label"] == "positive"]
    negatives = [row for row in af3_rows if row["label"] == "negative"]
    quarter = max(1, sample_size // 4)

    low_sc_pos = sorted(positives, key=lambda row: (safe_float(row.get("interface_sc", float("nan"))), str(row["pair"])))[:quarter]
    high_sc_neg = sorted(negatives, key=lambda row: (-safe_float(row.get("interface_sc", float("nan"))), str(row["pair"])))[:quarter]

    used_keys = {(row["organism"], row["pairset"], row["pair"], row["interface"]) for row in low_sc_pos + high_sc_neg}
    ordinary_pos = [
        row
        for row in sorted(positives, key=lambda row: (str(row["organism"]), str(row["pair"]), str(row["interface"])))
        if (row["organism"], row["pairset"], row["pair"], row["interface"]) not in used_keys
    ][:quarter]
    used_keys.update((row["organism"], row["pairset"], row["pair"], row["interface"]) for row in ordinary_pos)
    ordinary_neg = [
        row
        for row in sorted(negatives, key=lambda row: (str(row["organism"]), str(row["pair"]), str(row["interface"])))
        if (row["organism"], row["pairset"], row["pair"], row["interface"]) not in used_keys
    ][:quarter]
    selected = low_sc_pos + ordinary_pos + high_sc_neg + ordinary_neg

    if len(selected) < sample_size:
        used_keys = {(row["organism"], row["pairset"], row["pair"], row["interface"]) for row in selected}
        remainder = [
            row
            for row in sorted(af3_rows, key=lambda row: (str(row["organism"]), str(row["pairset"]), str(row["pair"])))
            if (row["organism"], row["pairset"], row["pair"], row["interface"]) not in used_keys
        ]
        selected.extend(remainder[: sample_size - len(selected)])

    return [_copy_for_diagnostic_slice(row, "af3_mixed_200") for row in selected[:sample_size]]


def diagnostic_benchmark_rows(rows: list[dict]) -> list[dict]:
    return human_af3_hard_slice(rows) + af3_mixed_diagnostic_sample(rows)


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


def candidate_row_metadata(row: dict, *, candidate_id: str, representation: str, candidate_family: str) -> dict:
    out = {
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
        "candidate_id": candidate_id,
        "candidate_family": candidate_family,
        "representation": representation,
    }
    if "diagnostic_slice" in row:
        out["diagnostic_slice"] = row["diagnostic_slice"]
    return out


def baseline_sc_results(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        out_row = candidate_row_metadata(
            row,
            candidate_id=SC_BASELINE_ID,
            representation=SC_BASELINE_ID,
            candidate_family="sc_baseline",
        )
        out_row.update(
            {
                "grid_size": "",
                "order": "",
                "sigma": "",
                "surface_density": "",
                "surface_trim_cutoff": "",
                "surface_probe_radius": "",
                "proximity_length_scale": "",
                "padding": "",
                "distance": "",
                "score_mode": "baseline",
                "fit_order": "",
                "order_decay_n0": "",
                "normal_gap_good_scale": "",
                "normal_gap_far_scale": "",
                "normal_gap_clash_weight": "",
                "normal_gap_far_weight": "",
                "candidate_score": safe_float(row.get("interface_sc", float("nan"))),
                "candidate_status": "baseline",
            }
        )
        out.append(out_row)
    return out


def _candidate_result_row(row: dict, spec: ZernikeSpec, score: float, status: str) -> dict:
    out_row = candidate_row_metadata(
        row,
        candidate_id=spec.candidate_id(),
        representation=spec.representation,
        candidate_family=zernike_candidate_family(spec.representation, spec.score_mode),
    )
    out_row.update(
        {
            "grid_size": spec.grid_size,
            "order": spec.order,
            "sigma": spec.sigma if spec.representation in GAUSSIAN_REPRESENTATIONS else "",
            "surface_density": spec.surface_density
            if zernike_source_representation(spec.representation) in SURFACE_REPRESENTATIONS
            else "",
            "surface_trim_cutoff": spec.surface_trim_cutoff
            if zernike_source_representation(spec.representation) in SURFACE_REPRESENTATIONS
            else "",
            "surface_probe_radius": spec.surface_probe_radius
            if zernike_source_representation(spec.representation) in SURFACE_REPRESENTATIONS
            else "",
            "proximity_length_scale": spec.proximity_length_scale
            if zernike_source_representation(spec.representation) == "surface_proximity_gaussian"
            else "",
            "padding": spec.padding,
            "distance": spec.distance,
            "score_mode": spec.score_mode,
            "fit_order": fit_order_value(spec),
            "order_decay_n0": spec.order_decay_n0
            if spec.score_mode in {GAUSSIAN_WEIGHTED_SCORE, GAP_ZERNIKE_WEIGHTED_SCORE}
            else "",
            "normal_gap_good_scale": spec.normal_gap_good_scale
            if spec.score_mode in NORMAL_GAP_SCORE_MODES
            else "",
            "normal_gap_far_scale": spec.normal_gap_far_scale
            if spec.score_mode in NORMAL_GAP_SCORE_MODES
            else "",
            "normal_gap_clash_weight": spec.normal_gap_clash_weight
            if spec.score_mode in NORMAL_GAP_SCORE_MODES
            else "",
            "normal_gap_far_weight": spec.normal_gap_far_weight
            if spec.score_mode in NORMAL_GAP_SCORE_MODES
            else "",
            "candidate_score": score,
            "candidate_status": status,
        }
    )
    return out_row


def _score_spec_from_coefficients(
    spec: ZernikeSpec,
    coeff1: np.ndarray,
    coeff2: np.ndarray,
    joint_coeff: np.ndarray,
    gap_coeff: np.ndarray,
    grid_overlap: float,
) -> float:
    if spec.score_mode in GRID_SCORE_MODES:
        return zernike_score_from_gap_coefficients(
            gap_coeff,
            grid_overlap,
            order=spec.order,
            score_mode=spec.score_mode,
            fit_order=fit_order_value(spec),
            order_decay_n0=spec.order_decay_n0,
        )
    if spec.representation in JOINT_REPRESENTATIONS:
        return zernike_score_from_coefficients(
            joint_coeff,
            None,
            order=spec.order,
            score_mode=spec.score_mode,
            fit_order=fit_order_value(spec),
            order_decay_n0=spec.order_decay_n0,
        )
    return zernike_score_from_coefficients(
        coeff1,
        coeff2,
        order=spec.order,
        score_mode=spec.score_mode,
        fit_order=fit_order_value(spec),
        order_decay_n0=spec.order_decay_n0,
    )


def _evaluate_row_candidate_groups(
    row: dict,
    grouped_candidates: list[list[ZernikeSpec]],
    cache_dir: Path,
) -> tuple[dict[str, list[dict]], dict]:
    point_cloud_cache = PointCloudCache(cache_dir / "point_clouds")
    grid_cache = GridCache(cache_dir / "grids", point_cloud_cache)
    coeff_cache = CoefficientCache(cache_dir / "coefficients", grid_cache)
    rows_by_candidate: dict[str, list[dict]] = defaultdict(list)
    status_counts: dict[str, int] = defaultdict(int)

    for spec_group in grouped_candidates:
        anchor = spec_group[0]
        try:
            if anchor.representation in NORMAL_GAP_REPRESENTATIONS or anchor.score_mode in NORMAL_GAP_SCORE_MODES:
                residues1, residues2 = load_interface_residues(str(row["model_file"]), str(row["interface"]))
                fields = zernike_normal_gap_field_bundle(
                    residues1,
                    residues2,
                    distance=anchor.distance,
                    grid_size=anchor.grid_size,
                    sigma=anchor.sigma,
                    padding=anchor.padding,
                    surface_density=anchor.surface_density,
                    surface_trim_cutoff=anchor.surface_trim_cutoff,
                    surface_probe_radius=anchor.surface_probe_radius,
                    normal_gap_good_scale=anchor.normal_gap_good_scale,
                    normal_gap_far_scale=anchor.normal_gap_far_scale,
                )
                coeffs = zernike_normal_gap_coefficient_bundle(fields, fit_order_value(anchor))
                scores = {
                    spec.candidate_id(): zernike_score_from_normal_gap_coefficients(
                        coeffs,
                        order=spec.order,
                        fit_order=fit_order_value(spec),
                        normal_gap_clash_weight=spec.normal_gap_clash_weight,
                        normal_gap_far_weight=spec.normal_gap_far_weight,
                    )
                    for spec in spec_group
                }
            else:
                coeff1, coeff2, joint_coeff, gap_coeff, grid_overlap = coeff_cache.get_or_build(row, anchor)
                scores = {
                    spec.candidate_id(): _score_spec_from_coefficients(
                        spec,
                        coeff1,
                        coeff2,
                        joint_coeff,
                        gap_coeff,
                        grid_overlap,
                    )
                    for spec in spec_group
                }

            for spec in spec_group:
                rows_by_candidate[spec.candidate_id()].append(
                    _candidate_result_row(row, spec, scores[spec.candidate_id()], "success")
                )
                status_counts["success"] += 1
        except Exception as exc:
            for spec in spec_group:
                rows_by_candidate[spec.candidate_id()].append(
                    _candidate_result_row(row, spec, float("nan"), f"error:{exc}")
                )
                status_counts["error"] += 1

    cache_meta = {
        "point_cloud_cache_hits": point_cloud_cache.hits,
        "point_cloud_cache_misses": point_cloud_cache.misses,
        "grid_cache_hits": grid_cache.hits,
        "grid_cache_misses": grid_cache.misses,
        "coefficient_cache_hits": coeff_cache.hits,
        "coefficient_cache_misses": coeff_cache.misses,
        "status_counts": dict(sorted(status_counts.items())),
    }
    return dict(rows_by_candidate), cache_meta


def _evaluate_row_candidate_groups_task(task: tuple[dict, list[list[ZernikeSpec]], str]) -> tuple[dict[str, list[dict]], dict]:
    row, grouped_candidates, cache_dir = task
    return _evaluate_row_candidate_groups(row, grouped_candidates, Path(cache_dir))


def _merge_cache_meta(target: dict, source: dict) -> None:
    for key, value in source.items():
        if key == "status_counts":
            counts = target.setdefault("status_counts", {})
            for status, count in value.items():
                counts[status] = counts.get(status, 0) + count
        elif isinstance(value, (int, float)):
            target[key] = target.get(key, 0) + value


def evaluate_candidates(
    rows: list[dict],
    candidates: list[ZernikeSpec],
    *,
    cache_dir: Path,
    jobs: int = 1,
    progress_every: int = 0,
) -> tuple[dict[str, list[dict]], dict]:
    grouped_candidates = candidate_groups(candidates)
    grouped_candidate_lists = list(grouped_candidates.values())
    results: dict[str, list[dict]] = {spec.candidate_id(): [] for spec in candidates}
    results[SC_BASELINE_ID] = baseline_sc_results(rows)
    worker_count = min(max(1, int(jobs)), max(1, len(rows)))
    cache_meta = {
        "point_cloud_cache_hits": 0,
        "point_cloud_cache_misses": 0,
        "grid_cache_hits": 0,
        "grid_cache_misses": 0,
        "coefficient_cache_hits": 0,
        "coefficient_cache_misses": 0,
        "worker_count": worker_count,
        "status_counts": {},
    }

    started = time.perf_counter()

    def report_progress(done: int) -> None:
        if progress_every <= 0:
            return
        if done != len(rows) and done % progress_every != 0:
            return
        elapsed = max(time.perf_counter() - started, 1e-9)
        rate = done / elapsed
        print(
            f"[benchmark_zernike] scored {done}/{len(rows)} rows "
            f"with {worker_count} workers ({rate:.2f} rows/s)",
            flush=True,
        )

    if worker_count == 1 or len(rows) <= 1:
        for done, row in enumerate(rows, start=1):
            row_results, row_meta = _evaluate_row_candidate_groups(row, grouped_candidate_lists, cache_dir)
            for candidate_id, candidate_rows in row_results.items():
                results[candidate_id].extend(candidate_rows)
            _merge_cache_meta(cache_meta, row_meta)
            report_progress(done)
    else:
        tasks = ((row, grouped_candidate_lists, str(cache_dir)) for row in rows)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for done, (row_results, row_meta) in enumerate(
                executor.map(_evaluate_row_candidate_groups_task, tasks, chunksize=1),
                start=1,
            ):
                for candidate_id, candidate_rows in row_results.items():
                    results[candidate_id].extend(candidate_rows)
                _merge_cache_meta(cache_meta, row_meta)
                report_progress(done)

    cache_meta["status_counts"] = dict(sorted(cache_meta.get("status_counts", {}).items()))
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


def _median_or_nan(values: list[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def score_distribution_summary(rows: list[dict], field: str = "candidate_score") -> dict[str, float | int]:
    pos = [
        safe_float(row[field])
        for row in rows
        if row["label"] == "positive" and math.isfinite(safe_float(row[field]))
    ]
    neg = [
        safe_float(row[field])
        for row in rows
        if row["label"] == "negative" and math.isfinite(safe_float(row[field]))
    ]
    all_scores = pos + neg
    pos_median = _median_or_nan(pos)
    neg_median = _median_or_nan(neg)
    median_separation = (
        pos_median - neg_median
        if math.isfinite(pos_median) and math.isfinite(neg_median)
        else float("nan")
    )
    pos_high_fraction = (
        float(np.mean([score >= SATURATION_SCORE_THRESHOLD for score in pos]))
        if pos
        else float("nan")
    )
    neg_high_fraction = (
        float(np.mean([score >= SATURATION_SCORE_THRESHOLD for score in neg]))
        if neg
        else float("nan")
    )
    score_min = min(all_scores) if all_scores else float("nan")
    score_max = max(all_scores) if all_scores else float("nan")
    dynamic_range = score_max - score_min if all_scores else float("nan")
    saturation_reject = (
        math.isfinite(pos_high_fraction)
        and math.isfinite(neg_high_fraction)
        and pos_high_fraction >= SATURATION_FRACTION_THRESHOLD
        and neg_high_fraction >= SATURATION_FRACTION_THRESHOLD
    ) or (
        math.isfinite(median_separation)
        and median_separation < MIN_MEDIAN_SEPARATION
    )
    return {
        "positive_median_score": pos_median,
        "negative_median_score": neg_median,
        "positive_minus_negative_median": median_separation,
        "positive_fraction_ge_0_95": pos_high_fraction,
        "negative_fraction_ge_0_95": neg_high_fraction,
        "score_min": score_min,
        "score_max": score_max,
        "score_dynamic_range": dynamic_range,
        "saturation_reject": int(saturation_reject),
    }


def is_production_candidate(spec: ZernikeSpec) -> bool:
    return spec.score_mode in GRID_SCORE_MODES or spec.score_mode in NORMAL_GAP_SCORE_MODES


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
        _ = zernike_shape_from_spec(residues1, residues2, spec)
        timings.append(time.perf_counter() - start)
    return summarize_timings(timings)


def measure_sc_runtime(rows: list[dict]) -> dict[str, float]:
    timings = []
    residues_cache: dict[tuple[str, str], tuple[tuple, tuple]] = {}
    for row in rows:
        key = (str(row["model_file"]), str(row["interface"]))
        if key not in residues_cache:
            residues_cache[key] = load_interface_residues(*key)
        residues1, residues2 = residues_cache[key]
        start = time.perf_counter()
        _ = shape_complementarity(residues1, residues2)
        timings.append(time.perf_counter() - start)
    return summarize_timings(timings)


def summarize_timings(timings: list[float]) -> dict[str, float]:
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


def _is_sidechain_heavy_atom(atom_name: str, element: str) -> bool:
    if not element or element.upper() == "H":
        return False
    name = atom_name.strip().upper()
    return name not in PROTEIN_BACKBONE_ATOMS and name not in NA_BACKBONE_ATOMS


def jitter_sidechains(residues, rng: np.random.Generator, std: float):
    moved = copy.deepcopy(residues)
    for residue in moved:
        for atom in residue:
            if _is_sidechain_heavy_atom(atom.id, atom.element or ""):
                atom.coord = atom.coord + rng.normal(0.0, float(std), size=3)
    return moved


def zernike_shape_from_spec(residues1, residues2, spec: ZernikeSpec) -> float:
    from alphajudge.biophysics.zernike import zernike_shape_complementarity

    return zernike_shape_complementarity(
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
        surface_probe_radius=spec.surface_probe_radius,
        proximity_length_scale=spec.proximity_length_scale,
        score_mode=spec.score_mode,
        fit_order=spec.fit_order,
        order_decay_n0=spec.order_decay_n0,
        normal_gap_good_scale=spec.normal_gap_good_scale,
        normal_gap_far_scale=spec.normal_gap_far_scale,
        normal_gap_clash_weight=spec.normal_gap_clash_weight,
        normal_gap_far_weight=spec.normal_gap_far_weight,
    )


def robustness_sample_rows(rows: list[dict], sample_size: int) -> list[dict]:
    positives = [row for row in rows if row["label"] == "positive"]
    return balanced_sample(positives, sample_size)


def measure_robustness(
    rows: list[dict],
    candidates: list[ZernikeSpec],
    *,
    jitter_std: float,
) -> list[dict]:
    deltas_by_id: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        residues1, residues2 = load_interface_residues(str(row["model_file"]), str(row["interface"]))
        seed = int(hash_payload({"pair": row["pair"], "interface": row["interface"]})[:16], 16) % (2**32)
        rng = np.random.default_rng(seed)
        jittered1 = jitter_sidechains(residues1, rng, jitter_std)
        jittered2 = jitter_sidechains(residues2, rng, jitter_std)

        baseline = shape_complementarity(residues1, residues2)
        baseline_jitter = shape_complementarity(jittered1, jittered2)
        deltas_by_id[SC_BASELINE_ID].append(abs(baseline_jitter - baseline))

        for spec in candidates:
            score = zernike_shape_from_spec(residues1, residues2, spec)
            score_jitter = zernike_shape_from_spec(jittered1, jittered2, spec)
            deltas_by_id[spec.candidate_id()].append(abs(score_jitter - score))

    out = []
    baseline_median = float(np.median(deltas_by_id[SC_BASELINE_ID])) if deltas_by_id[SC_BASELINE_ID] else float("nan")
    for candidate_id, deltas in sorted(deltas_by_id.items()):
        median_delta = float(np.median(deltas)) if deltas else float("nan")
        out.append(
            {
                "candidate_id": candidate_id,
                "jitter_std_angstrom": float(jitter_std),
                "median_abs_delta": median_delta,
                "mean_abs_delta": float(np.mean(deltas)) if deltas else float("nan"),
                "max_abs_delta": float(np.max(deltas)) if deltas else float("nan"),
                "robustness_n": len(deltas),
                "delta_vs_sc": median_delta - baseline_median
                if math.isfinite(median_delta) and math.isfinite(baseline_median)
                else float("nan"),
                "robustness_pass": int(
                    candidate_id == SC_BASELINE_ID
                    or (
                        math.isfinite(median_delta)
                        and math.isfinite(baseline_median)
                        and median_delta <= baseline_median
                    )
                ),
            }
        )
    return out


def summarize_candidates(
    candidate_results: dict[str, list[dict]],
    candidates: list[ZernikeSpec],
    *,
    runtime_rows: list[dict],
    robustness_rows: list[dict],
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict[str, list[dict]]]:
    baseline_rows = annotate_failure_slice(candidate_results[SC_BASELINE_ID])

    baseline_cell_metrics = {
        key: compute_scope_metrics(group, "candidate_score")
        for key, group in grouped_by(baseline_rows, ("organism", "backend")).items()
    }
    baseline_global = compute_scope_metrics(baseline_rows, "candidate_score")
    baseline_af2 = compute_scope_metrics([row for row in baseline_rows if row["backend"] == "af2"], "candidate_score")
    baseline_af3 = compute_scope_metrics([row for row in baseline_rows if row["backend"] == "af3"], "candidate_score")
    baseline_distribution = score_distribution_summary(baseline_rows)
    baseline_failure_slice = [
        row
        for row in baseline_rows
        if row["is_af3_sc_failure_slice"] == 1
    ]
    baseline_rescue_rate = float(
        np.mean([row["rescued_af3_failure_positive"] for row in baseline_failure_slice])
    ) if baseline_failure_slice else 0.0

    runtime_by_candidate: dict[str, dict[str, float]] = {
        SC_BASELINE_ID: measure_sc_runtime(runtime_rows)
    }
    runtime_rows_out: list[dict] = [{ "candidate_id": SC_BASELINE_ID, **runtime_by_candidate[SC_BASELINE_ID]}]
    for spec in candidates:
        stats = measure_candidate_runtime(runtime_rows, spec)
        runtime_by_candidate[spec.candidate_id()] = stats
        runtime_rows_out.append({"candidate_id": spec.candidate_id(), **stats})

    robustness_by_candidate = {row["candidate_id"]: row for row in robustness_rows}

    atom_baseline_id = next(
        spec.candidate_id()
        for spec in candidates
        if spec.representation == ATOM_GAUSSIAN
        and spec.grid_size == 32
        and spec.order == 10
        and math.isclose(float(spec.sigma), 1.5, rel_tol=0.0, abs_tol=1e-6)
        and spec.score_mode == HARD_CUTOFF_SCORE
    )
    atom_baseline_runtime = runtime_by_candidate[atom_baseline_id]["median_runtime_sec"]
    sc_runtime = runtime_by_candidate[SC_BASELINE_ID]["median_runtime_sec"]
    baseline_jitter = safe_float(robustness_by_candidate.get(SC_BASELINE_ID, {}).get("median_abs_delta", float("nan")))

    summary_rows: list[dict] = []
    metric_rows: list[dict] = []
    annotated_outputs: dict[str, list[dict]] = {SC_BASELINE_ID: baseline_rows}

    def append_scope_rows(candidate_id: str, rows: list[dict], baseline_scope: tuple[float, float, int, int], scope: str, organism: str = "", backend: str = "") -> None:
        auroc, ap, n_pos, n_neg = compute_scope_metrics(rows, "candidate_score")
        metric_rows.append(
            {
                "candidate_id": candidate_id,
                "scope": scope,
                "organism": organism,
                "backend": backend,
                "auroc": auroc,
                "average_precision": ap,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "sc_auroc": baseline_scope[0],
                "sc_average_precision": baseline_scope[1],
                "delta_auroc_vs_sc": auroc - baseline_scope[0]
                if math.isfinite(auroc) and math.isfinite(baseline_scope[0])
                else float("nan"),
                "delta_average_precision_vs_sc": ap - baseline_scope[1]
                if math.isfinite(ap) and math.isfinite(baseline_scope[1])
                else float("nan"),
            }
        )

    for key, group in grouped_by(baseline_rows, ("organism", "backend")).items():
        append_scope_rows(SC_BASELINE_ID, group, baseline_cell_metrics[key], "cell", key[0], key[1])
    append_scope_rows(SC_BASELINE_ID, baseline_rows, baseline_global, "global")
    append_scope_rows(SC_BASELINE_ID, [row for row in baseline_rows if row["backend"] == "af2"], baseline_af2, "af2", backend="af2")
    append_scope_rows(SC_BASELINE_ID, [row for row in baseline_rows if row["backend"] == "af3"], baseline_af3, "af3", backend="af3")

    summary_rows.append(
        {
            "candidate_id": SC_BASELINE_ID,
            "candidate_family": "sc_baseline",
            "representation": SC_BASELINE_ID,
            "grid_size": "",
            "order": "",
            "sigma": "",
            "surface_density": "",
            "surface_trim_cutoff": "",
            "surface_probe_radius": "",
            "proximity_length_scale": "",
            "padding": "",
            "distance": "",
            "score_mode": "baseline",
            "fit_order": "",
            "order_decay_n0": "",
            "af3_failure_rescue_rate": baseline_rescue_rate,
            "delta_rescue_vs_sc": 0.0,
            "pooled_af3_auroc": baseline_af3[0],
            "delta_af3_auroc_vs_sc": 0.0,
            "pooled_af3_average_precision": baseline_af3[1],
            "delta_af3_average_precision_vs_sc": 0.0,
            "pooled_all_auroc": baseline_global[0],
            "delta_all_auroc_vs_sc": 0.0,
            "pooled_all_average_precision": baseline_global[1],
            "delta_all_average_precision_vs_sc": 0.0,
            "pooled_af2_auroc": baseline_af2[0],
            "median_runtime_sec": runtime_by_candidate[SC_BASELINE_ID]["median_runtime_sec"],
            "runtime_ratio_vs_sc": 1.0,
            "runtime_ratio_vs_atom_baseline": runtime_by_candidate[SC_BASELINE_ID]["median_runtime_sec"] / atom_baseline_runtime
            if math.isfinite(runtime_by_candidate[SC_BASELINE_ID]["median_runtime_sec"]) and math.isfinite(atom_baseline_runtime) and atom_baseline_runtime > 0.0
            else float("nan"),
            "sidechain_jitter_median_abs_delta": baseline_jitter,
            "delta_sidechain_jitter_vs_sc": 0.0,
            **baseline_distribution,
            "production_eligible": 0,
            "diagnostic_only": 0,
            "diagnostic_pass": 0,
            "runtime_ok": 1,
            "robustness_pass": 1,
            "guardrail_pass": 1,
            "accepted_for_production": 0,
            "rank": 0,
            "recommended_candidate": 0,
        }
    )

    ranked_rows: list[dict] = []
    for spec in candidates:
        candidate_id = spec.candidate_id()
        rows = annotate_failure_slice(candidate_results[candidate_id])
        annotated_outputs[candidate_id] = rows

        pooled_global = compute_scope_metrics(rows, "candidate_score")
        pooled_af2 = compute_scope_metrics([row for row in rows if row["backend"] == "af2"], "candidate_score")
        pooled_af3 = compute_scope_metrics([row for row in rows if row["backend"] == "af3"], "candidate_score")
        distribution = score_distribution_summary(rows)
        production_eligible = is_production_candidate(spec)
        diagnostic_only = not production_eligible
        diagnostic_pass = production_eligible and int(distribution["saturation_reject"]) == 0

        cell_deltas = []
        for key, group in grouped_by(rows, ("organism", "backend")).items():
            append_scope_rows(candidate_id, group, baseline_cell_metrics[key], "cell", key[0], key[1])
            auroc = metric_rows[-1]["auroc"]
            delta = metric_rows[-1]["delta_auroc_vs_sc"]
            if math.isfinite(safe_float(delta)):
                cell_deltas.append(float(delta))

        append_scope_rows(candidate_id, rows, baseline_global, "global")
        append_scope_rows(candidate_id, [row for row in rows if row["backend"] == "af2"], baseline_af2, "af2", backend="af2")
        append_scope_rows(candidate_id, [row for row in rows if row["backend"] == "af3"], baseline_af3, "af3", backend="af3")

        failure_slice_rows = [row for row in rows if row["is_af3_sc_failure_slice"] == 1]
        rescue_rate = float(
            np.mean([row["rescued_af3_failure_positive"] for row in failure_slice_rows])
        ) if failure_slice_rows else float("nan")

        runtime_stats = runtime_by_candidate[candidate_id]
        runtime_ok = (
            math.isfinite(atom_baseline_runtime)
            and math.isfinite(runtime_stats["median_runtime_sec"])
            and atom_baseline_runtime > 0.0
            and runtime_stats["median_runtime_sec"] <= 5.0 * atom_baseline_runtime
        )
        guardrail_pass = (
            math.isfinite(pooled_global[0])
            and math.isfinite(baseline_global[0])
            and pooled_global[0] >= baseline_global[0]
            and all(delta >= -0.01 for delta in cell_deltas)
        )
        robustness_row = robustness_by_candidate.get(candidate_id, {})
        robustness_pass = int(robustness_row.get("robustness_pass", 0)) == 1
        accepted_for_production = (
            production_eligible
            and diagnostic_pass
            and runtime_ok
            and robustness_pass
            and guardrail_pass
            and math.isfinite(rescue_rate)
            and rescue_rate > baseline_rescue_rate
            and math.isfinite(pooled_af3[0])
            and math.isfinite(baseline_af3[0])
            and pooled_af3[0] > baseline_af3[0]
        )

        ranked_rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_family": zernike_candidate_family(spec.representation, spec.score_mode),
                "representation": spec.representation,
                "grid_size": spec.grid_size,
                "order": spec.order,
                "sigma": spec.sigma if spec.representation in GAUSSIAN_REPRESENTATIONS else "",
                "surface_density": spec.surface_density
                if zernike_source_representation(spec.representation) in SURFACE_REPRESENTATIONS
                else "",
                "surface_trim_cutoff": spec.surface_trim_cutoff
                if zernike_source_representation(spec.representation) in SURFACE_REPRESENTATIONS
                else "",
                "surface_probe_radius": spec.surface_probe_radius
                if zernike_source_representation(spec.representation) in SURFACE_REPRESENTATIONS
                else "",
                "proximity_length_scale": spec.proximity_length_scale
                if zernike_source_representation(spec.representation) == "surface_proximity_gaussian"
                else "",
                "padding": spec.padding,
                "distance": spec.distance,
                "score_mode": spec.score_mode,
                "fit_order": fit_order_value(spec),
                "order_decay_n0": spec.order_decay_n0
                if spec.score_mode in {GAUSSIAN_WEIGHTED_SCORE, GAP_ZERNIKE_WEIGHTED_SCORE}
                else "",
                "af3_failure_rescue_rate": rescue_rate,
                "delta_rescue_vs_sc": rescue_rate - baseline_rescue_rate
                if math.isfinite(rescue_rate)
                else float("nan"),
                "pooled_af3_auroc": pooled_af3[0],
                "delta_af3_auroc_vs_sc": pooled_af3[0] - baseline_af3[0]
                if math.isfinite(pooled_af3[0]) and math.isfinite(baseline_af3[0])
                else float("nan"),
                "pooled_af3_average_precision": pooled_af3[1],
                "delta_af3_average_precision_vs_sc": pooled_af3[1] - baseline_af3[1]
                if math.isfinite(pooled_af3[1]) and math.isfinite(baseline_af3[1])
                else float("nan"),
                "pooled_all_auroc": pooled_global[0],
                "delta_all_auroc_vs_sc": pooled_global[0] - baseline_global[0]
                if math.isfinite(pooled_global[0]) and math.isfinite(baseline_global[0])
                else float("nan"),
                "pooled_all_average_precision": pooled_global[1],
                "delta_all_average_precision_vs_sc": pooled_global[1] - baseline_global[1]
                if math.isfinite(pooled_global[1]) and math.isfinite(baseline_global[1])
                else float("nan"),
                "pooled_af2_auroc": pooled_af2[0],
                "median_runtime_sec": runtime_stats["median_runtime_sec"],
                "runtime_ratio_vs_sc": runtime_stats["median_runtime_sec"] / sc_runtime
                if math.isfinite(runtime_stats["median_runtime_sec"]) and math.isfinite(sc_runtime) and sc_runtime > 0.0
                else float("nan"),
                "runtime_ratio_vs_atom_baseline": runtime_stats["median_runtime_sec"] / atom_baseline_runtime
                if math.isfinite(runtime_stats["median_runtime_sec"]) and math.isfinite(atom_baseline_runtime) and atom_baseline_runtime > 0.0
                else float("nan"),
                "sidechain_jitter_median_abs_delta": safe_float(robustness_row.get("median_abs_delta", float("nan"))),
                "delta_sidechain_jitter_vs_sc": safe_float(robustness_row.get("delta_vs_sc", float("nan"))),
                **distribution,
                "production_eligible": int(production_eligible),
                "diagnostic_only": int(diagnostic_only),
                "diagnostic_pass": int(diagnostic_pass),
                "runtime_ok": int(runtime_ok),
                "robustness_pass": int(robustness_pass),
                "guardrail_pass": int(guardrail_pass),
                "accepted_for_production": int(accepted_for_production),
            }
        )

    ranked_rows.sort(
        key=lambda row: (
            -(safe_float(row["af3_failure_rescue_rate"]) if math.isfinite(safe_float(row["af3_failure_rescue_rate"])) else -1.0),
            -(safe_float(row["pooled_af3_auroc"]) if math.isfinite(safe_float(row["pooled_af3_auroc"])) else -1.0),
            -(safe_float(row["pooled_all_auroc"]) if math.isfinite(safe_float(row["pooled_all_auroc"])) else -1.0),
            safe_float(row["median_runtime_sec"]) if math.isfinite(safe_float(row["median_runtime_sec"])) else float("inf"),
        )
    )
    first_pass = True
    for idx, row in enumerate(ranked_rows, start=1):
        row["rank"] = idx
        row["recommended_candidate"] = int(first_pass and row["accepted_for_production"] == 1)
        if row["recommended_candidate"] == 1:
            first_pass = False
    summary_rows.extend(ranked_rows)
    return summary_rows, metric_rows, runtime_rows_out, robustness_rows, annotated_outputs


def write_candidate_reports(
    out_dir: Path,
    annotated_outputs: dict[str, list[dict]],
) -> None:
    score_dir = out_dir / "scores"
    top_dir = out_dir / "top_hits"
    sc_rows = annotated_outputs.get(SC_BASELINE_ID, [])
    sc_by_key = {
        (
            row["pair"],
            row["organism"],
            row["backend"],
            row["interface"],
            row.get("diagnostic_slice", ""),
        ): row
        for row in sc_rows
    }

    for candidate_id, rows in annotated_outputs.items():
        write_csv(score_dir / f"{candidate_id}.csv", rows)

        if candidate_id == SC_BASELINE_ID:
            continue

        failure_slice = [row for row in rows if row["is_af3_sc_failure_slice"] == 1]
        rescued_by_candidate = []
        rescued_by_sc = []
        still_missed = []
        for row in failure_slice:
            key = (
                row["pair"],
                row["organism"],
                row["backend"],
                row["interface"],
                row.get("diagnostic_slice", ""),
            )
            sc_row = sc_by_key.get(key, {})
            merged = dict(row)
            merged.update(
                {
                    "sc_candidate_score": safe_float(sc_row.get("candidate_score", float("nan"))),
                    "sc_candidate_neg90": safe_float(sc_row.get("candidate_neg90", float("nan"))),
                    "sc_rescued_af3_failure_positive": int(sc_row.get("rescued_af3_failure_positive", 0)),
                }
            )
            if row["rescued_af3_failure_positive"] == 1 and int(sc_row.get("rescued_af3_failure_positive", 0)) == 0:
                rescued_by_candidate.append(merged)
            if row["rescued_af3_failure_positive"] == 0 and int(sc_row.get("rescued_af3_failure_positive", 0)) == 1:
                rescued_by_sc.append(merged)
            if row["rescued_af3_failure_positive"] == 0:
                still_missed.append(merged)

        rescued_by_candidate.sort(
            key=lambda row: (
                -safe_float(row["candidate_margin_to_neg90"]),
                -safe_float(row["candidate_score"]),
                str(row["pair"]),
            )
        )
        rescued_by_sc.sort(
            key=lambda row: (
                -safe_float(row["sc_candidate_score"]),
                str(row["pair"]),
            )
        )
        still_missed.sort(
            key=lambda row: (
                -safe_float(row["candidate_score"]),
                str(row["pair"]),
            )
        )

        write_csv(top_dir / f"{candidate_id}__rescued_vs_sc.csv", rescued_by_candidate[:TOP_N])
        write_csv(top_dir / f"{candidate_id}__rescued_by_sc_only.csv", rescued_by_sc[:TOP_N])
        write_csv(top_dir / f"{candidate_id}__still_missed.csv", still_missed[:TOP_N])


def diagnostic_summary_rows(annotated_outputs: dict[str, list[dict]]) -> list[dict]:
    out: list[dict] = []
    for candidate_id, rows in sorted(annotated_outputs.items()):
        rows = [row for row in rows if "diagnostic_slice" in row]
        if not rows:
            continue
        for (diagnostic_slice,), group in sorted(grouped_by(rows, ("diagnostic_slice",)).items()):
            auroc, ap, n_pos, n_neg = compute_scope_metrics(group, "candidate_score")
            distribution = score_distribution_summary(group)
            out.append(
                {
                    "candidate_id": candidate_id,
                    "diagnostic_slice": diagnostic_slice,
                    "auroc": auroc,
                    "average_precision": ap,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    **distribution,
                }
            )
    return out


def filter_candidates_from_summary(candidates: list[ZernikeSpec], summary_path: Path) -> list[ZernikeSpec]:
    rows = read_csv_rows(summary_path)
    allowed = {
        row["candidate_id"]
        for row in rows
        if row.get("candidate_id") != SC_BASELINE_ID
        and row.get("diagnostic_pass") in {"1", "1.0", "true", "True"}
    }
    atom_baseline = _atom_cosine_baseline_spec()
    atom_baseline_id = atom_baseline.candidate_id()
    filtered = [spec for spec in candidates if spec.candidate_id() in allowed or spec.candidate_id() == atom_baseline_id]
    if not any(spec.candidate_id() != atom_baseline_id for spec in filtered):
        raise SystemExit(f"No diagnostic-pass Zernike candidates found in {summary_path}")
    return filtered


def filter_candidates_by_id(candidates: list[ZernikeSpec], candidate_ids: list[str]) -> list[ZernikeSpec]:
    allowed = set(candidate_ids)
    atom_baseline = _atom_cosine_baseline_spec()
    atom_baseline_id = atom_baseline.candidate_id()
    filtered = [spec for spec in candidates if spec.candidate_id() in allowed or spec.candidate_id() == atom_baseline_id]
    missing = sorted(allowed - {spec.candidate_id() for spec in candidates})
    if missing:
        raise SystemExit(f"Unknown candidate id(s): {', '.join(missing)}")
    if not any(spec.candidate_id() != atom_baseline_id for spec in filtered):
        raise SystemExit("Candidate filter removed all non-baseline Zernike candidates.")
    return filtered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", default=str(BENCH_ROOT_DEFAULT))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manifest-tag", default=None)
    parser.add_argument(
        "--mode",
        choices=["full", "diagnostic", "smoke"],
        default="full",
        help="Diagnostic mode uses fixed AF3 slices; smoke mode uses a small mixed sample.",
    )
    parser.add_argument("--smoke-sample-size", type=int, default=SMOKE_SAMPLE_SIZE)
    parser.add_argument("--runtime-sample-size", type=int, default=RUNTIME_SAMPLE_SIZE)
    parser.add_argument(
        "--robustness-sample-size",
        type=int,
        default=None,
        help="Override side-chain jitter robustness sample size; use 0 to skip.",
    )
    parser.add_argument(
        "--survivors-from",
        default=None,
        help="Candidate summary CSV from diagnostic mode; full mode will score only diagnostic-pass candidates plus the atom baseline.",
    )
    parser.add_argument(
        "--candidate-id",
        action="append",
        default=[],
        help="Restrict scoring to one candidate id; may be passed multiple times. The atom cosine baseline is kept for runtime ratios.",
    )
    parser.add_argument(
        "--organism",
        action="append",
        choices=ORGANISMS,
        default=[],
        help="Restrict benchmark loading to one organism; may be passed multiple times.",
    )
    parser.add_argument(
        "--backend",
        action="append",
        choices=BACKENDS,
        default=[],
        help="Restrict benchmark loading to one backend; may be passed multiple times.",
    )
    parser.add_argument(
        "--pairset",
        action="append",
        choices=PAIRSETS,
        default=[],
        help="Restrict benchmark loading to one pair set; may be passed multiple times.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of local worker processes for candidate scoring.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print scoring progress every N benchmark rows; use 0 to disable.",
    )
    args = parser.parse_args()

    bench_root = Path(args.bench_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    organisms = tuple(args.organism) if args.organism else ORGANISMS
    backends = tuple(args.backend) if args.backend else BACKENDS
    pairsets = tuple(args.pairset) if args.pairset else PAIRSETS

    all_rows, skipped = load_benchmark_rows(
        bench_root,
        manifest_tag=args.manifest_tag,
        organisms=organisms,
        backends=backends,
        pairsets=pairsets,
    )
    if not all_rows:
        raise SystemExit("No benchmark rows discovered.")

    candidates = build_smoke_candidates() if args.mode == "smoke" else build_full_candidates()
    if args.survivors_from:
        candidates = filter_candidates_from_summary(candidates, Path(args.survivors_from))
    if args.candidate_id:
        candidates = filter_candidates_by_id(candidates, args.candidate_id)

    if args.mode == "smoke":
        benchmark_rows = balanced_sample(all_rows, args.smoke_sample_size)
    elif args.mode == "diagnostic":
        benchmark_rows = diagnostic_benchmark_rows(all_rows)
    else:
        benchmark_rows = all_rows

    runtime_source_rows = [row for row in all_rows if row["backend"] == "af3"] if args.mode == "diagnostic" else all_rows
    runtime_rows = runtime_sample_rows(runtime_source_rows, args.runtime_sample_size)
    default_robustness_sample_size = (
        min(10, len([row for row in all_rows if row["label"] == "positive"]))
        if args.mode == "smoke"
        else ROBUSTNESS_SAMPLE_SIZE
    )
    robustness_sample_size = (
        default_robustness_sample_size
        if args.robustness_sample_size is None
        else args.robustness_sample_size
    )
    robustness_rows_sample = robustness_sample_rows(all_rows, robustness_sample_size)

    candidate_results, cache_meta = evaluate_candidates(
        benchmark_rows,
        candidates,
        cache_dir=out_dir / "cache",
        jobs=args.jobs,
        progress_every=args.progress_every,
    )
    robustness_summary = measure_robustness(
        robustness_rows_sample,
        candidates,
        jitter_std=ROBUSTNESS_JITTER_STD,
    )
    summary_rows, metric_rows, runtime_summary_rows, robustness_summary_rows, annotated_outputs = summarize_candidates(
        candidate_results,
        candidates,
        runtime_rows=runtime_rows,
        robustness_rows=robustness_summary,
    )

    write_csv(out_dir / "dataset_rows.csv", benchmark_rows)
    write_csv(out_dir / "skipped_rows.csv", skipped)
    write_csv(out_dir / "dataset_cell_counts.csv", dataset_cell_counts(benchmark_rows))
    write_csv(out_dir / "runtime_rows.csv", runtime_rows)
    write_csv(out_dir / "robustness_rows.csv", robustness_rows_sample)
    write_csv(out_dir / "candidate_summary.csv", summary_rows)
    write_csv(out_dir / "candidate_metrics.csv", metric_rows)
    write_csv(out_dir / "candidate_runtime_summary.csv", runtime_summary_rows)
    write_csv(out_dir / "candidate_robustness_summary.csv", robustness_summary_rows)
    diagnostic_rows = diagnostic_summary_rows(annotated_outputs)
    if diagnostic_rows:
        write_csv(out_dir / "candidate_diagnostic_summary.csv", diagnostic_rows)
    write_candidate_reports(out_dir, annotated_outputs)
    write_json(
        out_dir / "run_metadata.json",
        {
            "bench_root": str(bench_root),
            "mode": args.mode,
            "manifest_tag": args.manifest_tag,
            "organisms": list(organisms),
            "backends": list(backends),
            "pairsets": list(pairsets),
            "jobs": args.jobs,
            "progress_every": args.progress_every,
            "rows_total_discovered": len(all_rows),
            "rows_scored": len(benchmark_rows),
            "runtime_rows": len(runtime_rows),
            "robustness_rows": len(robustness_rows_sample),
            "candidate_count": len(candidates),
            "reported_score_count": len(candidates) + 1,
            "survivors_from": args.survivors_from,
            "candidates": [asdict(spec) for spec in candidates],
            **git_metadata(Path(__file__).resolve().parents[1]),
            **cache_meta,
        },
    )

    print(f"wrote {out_dir / 'candidate_summary.csv'}")
    print(f"wrote {out_dir / 'candidate_metrics.csv'}")
    print(f"wrote {out_dir / 'candidate_runtime_summary.csv'}")
    print(f"wrote {out_dir / 'candidate_robustness_summary.csv'}")
    print(f"wrote {out_dir / 'run_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
