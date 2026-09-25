"""Regression coverage for AF3/AF3x mixtures of residue and ligand tokens."""
from __future__ import annotations

import csv
import json
import lzma
import shutil
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import Atom, Chain, MMCIFIO, Model, Residue, Structure

from alphajudge.complex import Complex
from alphajudge.meta_score import interface_meta_score
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


@pytest.mark.parametrize("chain_order, scope", [
    (("A", "L", "B"), "includes_excluded_tokens"),
    (("A", "B"), "scored_residues"),
])
def test_af3_global_confidence_scope_reports_excluded_tokens(tmp_path, chain_order, scope):
    run_dir, _, _ = _write_run(tmp_path, chain_order)
    _, conf = AF3Parser().parse_run(run_dir).load_model("seed-1_sample-0")
    assert conf.global_confidence_scope == scope


def test_af3x_runner_preserves_global_values_but_excludes_them_from_metascore(tmp_path):
    run_dir, _, _ = _write_run(tmp_path)
    path = process(str(run_dir), 8., 100., "best", skip_pae_png=True, skip_biophysical_scores=True)
    with path.open() as fh:
        row = next(csv.DictReader(fh))
    assert row["global_confidence_scope"] == "includes_excluded_tokens"
    assert row["iptm_scope"] == "chain_pair"
    assert float(row["iptm"]) == 0.83
    assert float(row["ptm"]) == 0.6
    assert float(row["iptm_ptm"]) == pytest.approx(0.44)
    assert float(row["confidence_score"]) == 0.5
    without_global = {k: v for k, v in row.items() if k != "confidence_score"}
    assert float(row["interface_meta_score"]) == pytest.approx(interface_meta_score(without_global))


def test_mixed_token_metascore_excludes_global_iptm_fallback():
    row = {"model_used": "seed-1_sample-0", "interface_LIS": .3,
           "confidence_score": .9, "iptm": .9, "iptm_scope": "global",
           "global_confidence_scope": "includes_excluded_tokens"}
    expected = interface_meta_score({"model_used": row["model_used"], "interface_LIS": .3})
    assert interface_meta_score(row) == expected
    row["iptm_scope"] = "chain_pair"
    assert interface_meta_score(row) != expected


def test_mixed_token_report_does_not_show_global_confidence_percentile():
    from alphajudge.report import _feature_view, _row_meta_score
    row = {"model_used": "seed-1_sample-0", "interface_LIS": .3, "confidence_score": .9,
           "global_confidence_scope": "includes_excluded_tokens"}
    assert _feature_view(row)["confidence_score"] == (.9, None)
    assert _row_meta_score(row) == interface_meta_score({"model_used": row["model_used"], "interface_LIS": .3})


def test_af3x_invalid_pair_iptm_uses_labelled_global_fallback(tmp_path, caplog):
    run_dir, _, _ = _write_run(tmp_path)
    path = run_dir / "seed-1_sample-0/summary_confidences.json"
    summary = json.loads(path.read_text())
    summary["chain_pair_iptm"] = [[.9, .83], [.83, .8]]
    path.write_text(json.dumps(summary))
    output = process(str(run_dir), 8., 100., "best", skip_pae_png=True, skip_biophysical_scores=True)
    with output.open() as fh:
        row = next(csv.DictReader(fh))
    assert row["iptm_scope"] == "global"
    assert float(row["iptm"]) == .4
    assert "chain_pair_iptm dimensions" in caplog.text
    expected = {k: v for k, v in row.items() if k not in {"confidence_score", "iptm"}}
    assert float(row["interface_meta_score"]) == pytest.approx(interface_meta_score(expected))


def test_af3x_report_does_not_restore_stale_metascore():
    from alphajudge.report import _row_meta_score
    assert _row_meta_score({"confidence_score": .9, "interface_meta_score": .9,
                           "global_confidence_scope": "includes_excluded_tokens"}) is None


def test_plain_af3_global_confidence_still_contributes_to_metascore():
    row = {"model_used": "seed-1_sample-0", "interface_LIS": .3, "confidence_score": .9}
    assert interface_meta_score({**row, "global_confidence_scope": "scored_residues"}) == interface_meta_score(row)
    assert interface_meta_score(row) != interface_meta_score({"model_used": row["model_used"], "interface_LIS": .3})


def test_cached_scores_without_scope_metadata_are_recomputed(tmp_path, monkeypatch):
    from alphajudge import runner
    stale = tmp_path / "interfaces.csv"
    stale.write_text("jobs,interface_ccc,interface_expected_contacts\nold,1,2\n")
    calls = []

    def recompute(directory, *args, **kwargs):
        calls.append(directory)
        stale.write_text(
            "jobs,interface_ccc,interface_expected_contacts,global_confidence_scope,iptm_scope\n"
            "new,1,2,includes_excluded_tokens,chain_pair\n"
        )
        return stale

    monkeypatch.setattr(runner, "process", recompute)
    for _ in range(2):
        _, rows = runner._process_one_run(
            str(tmp_path), contact_thresh=8., pae_filter=100., models_to_analyse="best",
            summary_csv="summary.csv", ipsae_pae_cutoff=10., force_recompute=False,
            per_run_csv_name="interfaces.csv", skip_pae_png=True, skip_biophysical_scores=True,
        )
        assert rows[0]["jobs"] == "new"
    assert calls == [str(tmp_path)]


@pytest.mark.parametrize("ligand_first", [False, True])
def test_real_af3x_prediction_matches_raw_confidences(tmp_path, ligand_first):
    """A fresh AF3x inference, plus a reordered copy, guards the complete path."""
    fixture = Path(__file__).parent / "fixtures" / "af3x_4g3y_bc_dsso"
    run_dir = tmp_path / "af3x"
    shutil.copytree(fixture, run_dir)
    model_dir = run_dir / "seed-30_sample-0"
    with lzma.open(model_dir / "confidences.json.xz", "rt") as fh:
        raw = json.load(fh)
    summary_path = model_dir / "summary_confidences.json"
    summary = json.loads(summary_path.read_text())
    chain_ids = list(dict.fromkeys(raw["token_chain_ids"]))
    assert len(chain_ids) == 3
    if ligand_first:
        # Reorder the confidence tensors and their summary consistently. The
        # structure keeps its original order, so positional ipTM lookup fails.
        chain_order = [2, 0, 1]
        token_order = [i for c in chain_order for i, cid in enumerate(raw["token_chain_ids"])
                       if cid == chain_ids[c]]
        for key in ["pae", "contact_probs"]:
            raw[key] = np.asarray(raw[key])[np.ix_(token_order, token_order)].tolist()
        for key in ["token_chain_ids", "token_res_ids"]:
            raw[key] = [raw[key][i] for i in token_order]
        summary["chain_pair_iptm"] = np.asarray(summary["chain_pair_iptm"])[np.ix_(chain_order, chain_order)].tolist()
        with lzma.open(model_dir / "confidences.json.xz", "wt") as fh:
            json.dump(raw, fh)
        summary_path.write_text(json.dumps(summary))
        chain_ids = [chain_ids[i] for i in chain_order]

    structure, confidence = AF3Parser().parse_run(run_dir).load_model("seed-30_sample-0")
    # Independently extract the protein block from AF3x's original token matrix.
    lookup = {(c, r): i for i, (c, r) in enumerate(zip(raw["token_chain_ids"], raw["token_res_ids"]))
              if c in {"B", "C"}}
    keep = [lookup[(chain, residue)] for chain, length in [("B", 226), ("C", 158)]
            for residue in range(1, length + 1)]
    expected_pae = np.asarray(raw["pae"])[np.ix_(keep, keep)]
    expected_contacts = np.asarray(raw["contact_probs"])[np.ix_(keep, keep)]
    assert len(raw["pae"]) > len(keep) == 384
    assert np.unique(expected_pae[:226, 226:]).size > 10
    np.testing.assert_array_equal(confidence.pae_matrix, expected_pae)
    np.testing.assert_array_equal(confidence.contact_prob_matrix, expected_contacts)
    assert confidence.global_confidence_scope == "includes_excluded_tokens"
    complex_ = Complex(structure, confidence, 8., 100.)
    assert len(complex_.interfaces) == 1
    interface = complex_.interfaces[0]
    assert interface.iptm_chainpair == summary["chain_pair_iptm"][chain_ids.index("B")][chain_ids.index("C")]

    output = process(str(run_dir), 8., 100., "best", skip_pae_png=True, skip_biophysical_scores=True)
    with output.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    row = rows[0]
    assert row["interface"] == "B_C"
    assert row["iptm_scope"] == "chain_pair"
    assert float(row["iptm"]) == interface.iptm_chainpair
    assert float(row["confidence_score"]) == summary["ranking_score"]
    token_pae = np.asarray(raw["pae"])
    contact_paes = []
    for res1, res2 in interface._pairs:
        i = lookup[(res1.get_parent().id, res1.id[1])]
        j = lookup[(res2.get_parent().id, res2.id[1])]
        contact_paes.extend([token_pae[i, j], token_pae[j, i]])
    assert float(row["average_interface_pae"]) == pytest.approx(np.mean(contact_paes))
    assert float(row["interface_pDockQ2"]) == pytest.approx(interface.pDockQ2()[0])
    assert float(row["interface_ipSAE"]) == pytest.approx(interface.ipsae())
    assert float(row["interface_LIS"]) == pytest.approx(interface.lis())
    assert float(row["interface_ccc"]) == interface.ccc
    assert float(row["interface_meta_score"]) == pytest.approx(interface_meta_score(
        {k: v for k, v in row.items() if k != "confidence_score"}
    ))
