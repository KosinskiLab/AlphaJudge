from __future__ import annotations

import csv
import json
import subprocess
import sys
from io import StringIO
from pathlib import Path

from Bio.PDB import MMCIFIO, PDBIO, PDBParser

from scripts.benchmark_zernike_rescue import score_distribution_summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _script_path() -> Path:
    return _repo_root() / "scripts" / "benchmark_zernike_rescue.py"


def test_zernike_benchmark_smoke_uses_cache_and_scores_pdb_and_cif(tmp_path: Path):
    bench_root = tmp_path / "bench" / "predictions"
    model_root = tmp_path / "models"
    model_root.mkdir(parents=True, exist_ok=True)

    pdb_text = """\
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C
ATOM      2  CB  ALA A   1       1.500   0.500   0.200  1.00 50.00           C
ATOM      3  CA  LEU A   2       0.000   2.000   0.500  1.00 50.00           C
ATOM      4  CB  LEU A   2       1.200   2.600   0.800  1.00 50.00           C
ATOM      5  CA  GLY B   1       3.000   0.200   0.000  1.00 50.00           C
ATOM      6  O   GLY B   1       3.800   0.700   0.100  1.00 50.00           O
ATOM      7  CA  SER B   2       3.200   2.100   0.600  1.00 50.00           C
ATOM      8  OG  SER B   2       4.000   2.700   1.000  1.00 50.00           O
TER
END
"""
    structure = PDBParser(QUIET=True).get_structure("toy", StringIO(pdb_text))
    pdb_path = model_root / "toy_interface.pdb"
    cif_path = model_root / "toy_interface.cif"
    pdb_io = PDBIO()
    pdb_io.set_structure(structure)
    pdb_io.save(str(pdb_path))
    cif_io = MMCIFIO()
    cif_io.set_structure(structure)
    cif_io.save(str(cif_path))

    rows = [
        {
            "organism": "human",
            "backend": "af2",
            "pairset": "pos_pairs",
            "pair": "Q13148+Q92900",
            "interface": "A_B",
            "model_file": str(pdb_path),
            "pair_dir": str(model_root),
            "interface_sc": 0.12,
            "average_interface_pae": 4.0,
            "interface_area": 2500.0,
            "interface_contact_pairs": 120,
            "interface_num_intf_residues": 80,
            "interface_average_plddt": 90.0,
        },
        {
            "organism": "human",
            "backend": "af2",
            "pairset": "neg_pairs",
            "pair": "Q14974+Q13033",
            "interface": "A_B",
            "model_file": str(pdb_path),
            "pair_dir": str(model_root),
            "interface_sc": 0.04,
            "average_interface_pae": 8.0,
            "interface_area": 900.0,
            "interface_contact_pairs": 35,
            "interface_num_intf_residues": 28,
            "interface_average_plddt": 82.0,
        },
        {
            "organism": "human",
            "backend": "af3",
            "pairset": "pos_pairs",
            "pair": "Q13148+Q92900",
            "interface": "A_B",
            "model_file": str(cif_path),
            "pair_dir": str(model_root),
            "interface_sc": 0.07,
            "average_interface_pae": 12.0,
            "interface_area": 2100.0,
            "interface_contact_pairs": 65,
            "interface_num_intf_residues": 54,
            "interface_average_plddt": 78.0,
        },
        {
            "organism": "human",
            "backend": "af3",
            "pairset": "neg_pairs",
            "pair": "Q14974+Q13033",
            "interface": "A_B",
            "model_file": str(cif_path),
            "pair_dir": str(model_root),
            "interface_sc": 0.06,
            "average_interface_pae": 13.0,
            "interface_area": 950.0,
            "interface_contact_pairs": 22,
            "interface_num_intf_residues": 24,
            "interface_average_plddt": 75.0,
        },
    ]

    minimal_best_header = [
        "jobs",
        "model_used",
        "interface",
        "interface_sc",
        "average_interface_pae",
        "interface_area",
        "interface_contact_pairs",
        "interface_num_intf_residues",
        "interface_average_plddt",
    ]
    minimal_manifest_header = ["pair", "pair_dir", "model_file", "model_used", "run_status"]

    for row in rows:
        group_root = bench_root / row["organism"] / row["backend"] / row["pairset"]
        best_row = {key: "" for key in minimal_best_header}
        best_row.update(
            {
                "jobs": row["pair"],
                "model_used": "best_model",
                "interface": row["interface"],
                "interface_sc": row["interface_sc"],
                "average_interface_pae": row["average_interface_pae"],
                "interface_area": row["interface_area"],
                "interface_contact_pairs": row["interface_contact_pairs"],
                "interface_num_intf_residues": row["interface_num_intf_residues"],
                "interface_average_plddt": row["interface_average_plddt"],
            }
        )
        manifest_row = {key: "" for key in minimal_manifest_header}
        manifest_row.update(
            {
                "pair": row["pair"],
                "pair_dir": row["pair_dir"],
                "model_file": row["model_file"],
                "model_used": "best_model",
                "run_status": "complete_with_interface",
            }
        )
        _write_csv(group_root / "best_interfaces.csv", [best_row])
        _write_csv(group_root / "manifest.test.csv", [manifest_row])

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(_script_path()),
        "--bench-root",
        str(bench_root),
        "--out-dir",
        str(out_dir),
        "--manifest-tag",
        "test",
        "--mode",
        "smoke",
        "--smoke-sample-size",
        "4",
        "--runtime-sample-size",
        "1",
    ]

    first = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    meta1 = json.loads((out_dir / "run_metadata.json").read_text())
    assert meta1["grid_cache_misses"] > 0

    second = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert second.returncode == 0, second.stderr
    meta2 = json.loads((out_dir / "run_metadata.json").read_text())
    assert (
        meta2["point_cloud_cache_hits"] > meta1["point_cloud_cache_hits"]
        or meta2["grid_cache_hits"] > meta1["grid_cache_hits"]
        or meta2["coefficient_cache_hits"] > meta1["coefficient_cache_hits"]
    )

    summary_rows = list(csv.DictReader((out_dir / "candidate_summary.csv").open()))
    assert len(summary_rows) == 8
    assert {row["candidate_family"] for row in summary_rows} == {"sc_baseline", "per_side", "joint_volume", "grid_gap"}
    baseline_row = next(row for row in summary_rows if row["candidate_id"] == "interface_sc")
    assert baseline_row["delta_all_auroc_vs_sc"] == "0.0"
    assert "positive_fraction_ge_0_95" in baseline_row
    assert "negative_fraction_ge_0_95" in baseline_row
    assert "saturation_reject" in baseline_row
    assert any(row["production_eligible"] == "1" for row in summary_rows)

    metric_rows = list(csv.DictReader((out_dir / "candidate_metrics.csv").open()))
    assert "delta_auroc_vs_sc" in metric_rows[0]
    assert "delta_average_precision_vs_sc" in metric_rows[0]

    runtime_rows = list(csv.DictReader((out_dir / "candidate_runtime_summary.csv").open()))
    assert any(row["candidate_id"] == "interface_sc" for row in runtime_rows)

    robustness_rows = list(csv.DictReader((out_dir / "candidate_robustness_summary.csv").open()))
    assert any(row["candidate_id"] == "interface_sc" for row in robustness_rows)

    score_rows = list(csv.DictReader((out_dir / "scores" / "interface_sc.csv").open()))
    assert {row["backend"] for row in score_rows} == {"af2", "af3"}
    assert {row["candidate_status"] for row in score_rows} == {"baseline"}


def test_saturation_diagnostics_reject_always_high_scores():
    rows = [
        {"label": "positive", "candidate_score": 0.98},
        {"label": "positive", "candidate_score": 0.99},
        {"label": "negative", "candidate_score": 0.97},
        {"label": "negative", "candidate_score": 0.98},
    ]

    summary = score_distribution_summary(rows)

    assert summary["positive_fraction_ge_0_95"] == 1.0
    assert summary["negative_fraction_ge_0_95"] == 1.0
    assert summary["saturation_reject"] == 1
