"""Regression coverage for AF3/AF3x mixtures of residue and ligand tokens."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import Atom, Chain, MMCIFIO, Model, Residue, Structure

from alphajudge.complex import Complex
from alphajudge.parsers.af3 import AF3Parser
from alphajudge.runner import process


def _write_run(tmp_path: Path, chain_order=("A", "L", "B")):
    """Two contacting proteins and an atom-tokenized ligand, with asymmetric PAE."""
    run_dir = tmp_path / "mixed"
    model_dir = run_dir / "seed-1_sample-0"
    model_dir.mkdir(parents=True)
    structure = Structure.Structure("mixed")
    model = Model.Model(0)
    structure.add(model)
    token_chains, token_resids, protein_indices = [], [], []
    for chain_id in chain_order:
        chain = Chain.Chain(chain_id)
        model.add(chain)
        for number in range(1, 2 if chain_id == "L" else 3):
            ligand = chain_id == "L"
            residue = Residue.Residue(
                ("H_LIG" if ligand else " ", number, " "),
                "LIG" if ligand else "ALA", " ",
            )
            chain.add(residue)
            for atom_name in (["C1", "C2", "C3"] if ligand else ["CA"]):
                residue.add(Atom.Atom(
                    atom_name, np.array([number * 2., 3. if chain_id == "B" else 0., 0.]),
                    90., 1., " ", atom_name, len(token_chains) + 1, element="C",
                ))
                if not ligand:
                    protein_indices.append(len(token_chains))
                token_chains.append(chain_id)
                token_resids.append(number)
    writer = MMCIFIO()
    writer.set_structure(structure)
    writer.save(str(model_dir / "model.cif"))
    n = len(token_chains)
    pae = np.arange(n * n, dtype=float).reshape(n, n) / 2
    matrix = {
        "pae": pae.tolist(), "contact_probs": (pae / (n * n)).tolist(),
        "token_chain_ids": token_chains, "token_res_ids": token_resids,
    }
    summary = {
        "iptm": 0.4, "ptm": 0.6, "ranking_score": 0.5,
        "chain_pair_iptm": [[0.9, 0.12, 0.83], [0.12, 0.7, 0.34], [0.83, 0.34, 0.8]],
    }
    (model_dir / "confidences.json").write_text(json.dumps(matrix))
    (model_dir / "summary_confidences.json").write_text(json.dumps(summary))
    (run_dir / "ranking_scores.csv").write_text("seed,sample,ranking_score\n1,0,0.5\n")
    return run_dir, matrix, protein_indices


@pytest.mark.parametrize("chain_order", [("A", "L", "B"), ("L", "A", "B"), ("A", "B", "L")])
def test_af3x_parser_preserves_residue_pae_and_contact_probs(tmp_path, chain_order):
    run_dir, matrix, keep = _write_run(tmp_path, chain_order)
    _, conf = AF3Parser().parse_run(run_dir).load_model("seed-1_sample-0")
    np.testing.assert_array_equal(conf.pae_matrix, np.asarray(matrix["pae"])[np.ix_(keep, keep)])
    np.testing.assert_array_equal(conf.contact_prob_matrix, np.asarray(matrix["contact_probs"])[np.ix_(keep, keep)])


def test_af3x_unmappable_pae_is_rejected_instead_of_minimum_filled(tmp_path):
    run_dir, matrix, _ = _write_run(tmp_path)
    matrix["token_res_ids"][0] = 999
    path = run_dir / "seed-1_sample-0/confidences.json"
    path.write_text(json.dumps(matrix))
    with pytest.raises(ValueError, match="align.*PAE|PAE.*align"):
        AF3Parser().parse_run(run_dir).load_model("seed-1_sample-0")


def test_af3x_missing_residue_ids_with_extra_same_chain_tokens_is_rejected(tmp_path):
    run_dir, matrix, _ = _write_run(tmp_path)
    matrix.pop("token_res_ids")
    matrix["token_chain_ids"] = ["A"] * 5 + ["B"] * 2
    (run_dir / "seed-1_sample-0/confidences.json").write_text(json.dumps(matrix))
    with pytest.raises(ValueError, match="align.*PAE|PAE.*align"):
        AF3Parser().parse_run(run_dir).load_model("seed-1_sample-0")


def test_af3x_summary_minima_cannot_replace_missing_full_pae(tmp_path, caplog):
    run_dir, _, _ = _write_run(tmp_path)
    model_dir = run_dir / "seed-1_sample-0"
    (model_dir / "confidences.json").unlink()
    summary = json.loads((model_dir / "summary_confidences.json").read_text())
    summary["chain_pair_pae_min"] = [[1.] * 3] * 3
    (model_dir / "summary_confidences.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="residue-level PAE"):
        AF3Parser().parse_run(run_dir).load_model("seed-1_sample-0")
    output = process(str(run_dir), 8., 100., "best", skip_pae_png=True, skip_biophysical_scores=True)
    assert output.read_text() == ""
    assert "residue-level PAE" in caplog.text


@pytest.mark.parametrize("field", ["pae", "predicted_aligned_error"])
def test_plain_af3_pae_is_unchanged_and_token_permutations_are_aligned(tmp_path, field):
    run_dir, matrix, _ = _write_run(tmp_path, ("A", "B"))
    expected = np.asarray(matrix.pop("pae"))
    matrix[field] = expected.tolist()
    model_dir = run_dir / "seed-1_sample-0"
    summary = json.loads((model_dir / "summary_confidences.json").read_text())
    summary["chain_pair_iptm"] = [[.9, .83], [.83, .8]]
    (model_dir / "summary_confidences.json").write_text(json.dumps(summary))
    for order in [[0, 1, 2, 3], [2, 3, 0, 1]]:
        permuted = {**matrix, field: expected[np.ix_(order, order)].tolist(),
                    "token_chain_ids": [matrix["token_chain_ids"][i] for i in order],
                    "token_res_ids": [matrix["token_res_ids"][i] for i in order]}
        (model_dir / "confidences.json").write_text(json.dumps(permuted))
        _, conf = AF3Parser().parse_run(run_dir).load_model("seed-1_sample-0")
        np.testing.assert_array_equal(conf.pae_matrix, expected)


@pytest.mark.parametrize("residue_ids", [None, "token_residue_ids"])
def test_af3x_legacy_chain_only_and_alternate_residue_ids(tmp_path, residue_ids):
    run_dir, matrix, keep = _write_run(tmp_path)
    ids = matrix.pop("token_res_ids")
    if residue_ids:
        matrix[residue_ids] = ids
    (run_dir / "seed-1_sample-0/confidences.json").write_text(json.dumps(matrix))
    _, conf = AF3Parser().parse_run(run_dir).load_model("seed-1_sample-0")
    np.testing.assert_array_equal(conf.pae_matrix, np.asarray(matrix["pae"])[np.ix_(keep, keep)])


@pytest.mark.parametrize("bad_pae", [[1., 2.], [[1., 2.]], [], [[float("nan")]*7]*7])
def test_af3x_malformed_pae_fails_explicitly(tmp_path, bad_pae):
    run_dir, matrix, _ = _write_run(tmp_path)
    matrix["pae"] = bad_pae
    (run_dir / "seed-1_sample-0/confidences.json").write_text(json.dumps(matrix))
    with pytest.raises(ValueError, match="PAE"):
        AF3Parser().parse_run(run_dir).load_model("seed-1_sample-0")


def test_af3x_duplicate_scored_residue_ids_are_rejected(tmp_path):
    run_dir, matrix, _ = _write_run(tmp_path)
    matrix["token_chain_ids"][2] = "A"
    matrix["token_res_ids"][2] = 1
    (run_dir / "seed-1_sample-0/confidences.json").write_text(json.dumps(matrix))
    with pytest.raises(ValueError, match="unambiguously"):
        AF3Parser().parse_run(run_dir).load_model("seed-1_sample-0")


@pytest.mark.parametrize("chain_order", [("A", "L", "B"), ("L", "A", "B"), ("A", "B", "L")])
def test_af3x_pair_iptm_follows_summary_chain_order(tmp_path, chain_order):
    run_dir, _, _ = _write_run(tmp_path, chain_order)
    structure, conf = AF3Parser().parse_run(run_dir).load_model("seed-1_sample-0")
    interface = Complex(structure, conf, 8., 100.).interfaces[0]
    summary = json.loads((run_dir / "seed-1_sample-0/summary_confidences.json").read_text())
    assert interface.iptm_chainpair == summary["chain_pair_iptm"][chain_order.index("A")][chain_order.index("B")]


def test_af3x_pair_iptm_with_wrong_dimensions_is_dropped(tmp_path, caplog):
    run_dir, _, _ = _write_run(tmp_path)
    path = run_dir / "seed-1_sample-0/summary_confidences.json"
    summary = json.loads(path.read_text())
    summary["chain_pair_iptm"] = [[.9, .83], [.83, .8]]
    path.write_text(json.dumps(summary))
    structure, conf = AF3Parser().parse_run(run_dir).load_model("seed-1_sample-0")
    assert conf.chain_pair_iptm is None
    assert Complex(structure, conf, 8., 100.).interfaces[0].iptm_chainpair is None
    assert "chain_pair_iptm dimensions" in caplog.text
