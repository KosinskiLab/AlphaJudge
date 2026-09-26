"""Boltz-2 confidence arrays are aligned to scored residues by token, not position."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import Atom, Chain, MMCIFIO, Model, Residue, Structure

from alphajudge.parsers.boltz import Boltz2Parser
from alphajudge.runner import process

MODEL = "toy_model_0"


def _write_run(tmp_path: Path, chain_order: tuple[str, ...]) -> tuple[Path, list[int], list[str]]:
    """Two contacting two-residue proteins (A, B) and a three-atom ligand (L).

    Boltz-2 gives each standard residue one token and each ligand atom one
    token, chain by chain, so the ligand's position in ``chain_order`` decides
    where the protein tokens sit in the PAE and pLDDT arrays.
    """
    run_dir = tmp_path / "boltz"
    run_dir.mkdir()
    structure = Structure.Structure("toy")
    model = Model.Model(0)
    structure.add(model)
    protein_tokens, serial = [], 1
    for chain_id in chain_order:
        chain = Chain.Chain(chain_id)
        model.add(chain)
        if chain_id == "L":
            residue = Residue.Residue(("H_LIG", 1, " "), "LIG", " ")
            chain.add(residue)
            for k, name in enumerate(("C1", "C2", "O1")):
                residue.add(Atom.Atom(name, np.array([20.0 + k, 20.0, 20.0]), 50.0, 1.0, " ",
                                      name, serial, element=name[0]))
                serial += 1
            continue
        for number in (1, 2):
            residue = Residue.Residue((" ", number, " "), "ALA", " ")
            chain.add(residue)
            y = 0.0 if chain_id == "A" else 5.0
            for name, dx in (("CA", 0.0), ("CB", 0.5)):
                residue.add(Atom.Atom(name, np.array([3.8 * number + dx, y, 0.0]), 80.0, 1.0, " ",
                                      name, serial, element="C"))
                serial += 1
    n_tokens = 0
    for chain_id in chain_order:
        size = 3 if chain_id == "L" else 2
        if chain_id != "L":
            protein_tokens.extend(range(n_tokens, n_tokens + size))
        n_tokens += size

    writer = MMCIFIO()
    writer.set_structure(structure)
    writer.save(str(run_dir / f"{MODEL}.cif"))
    pae = np.arange(n_tokens * n_tokens, dtype=float).reshape(n_tokens, n_tokens) / 4
    np.savez(run_dir / f"pae_{MODEL}.npz", pae=pae)
    np.savez(run_dir / f"plddt_{MODEL}.npz", plddt=np.linspace(0.5, 0.9, n_tokens))
    pair = {str(i): {str(j): round(0.1 * (i + 1) + 0.01 * (j + 1), 3) for j in range(3)} for i in range(3)}
    (run_dir / f"confidence_{MODEL}.json").write_text(json.dumps(
        {"iptm": 0.4, "ptm": 0.6, "confidence_score": 0.5, "pair_chains_iptm": pair}
    ))
    return run_dir, protein_tokens, list(chain_order)


@pytest.mark.parametrize("chain_order", [("A", "B", "L"), ("L", "A", "B"), ("A", "L", "B")])
def test_boltz2_arrays_follow_structure_tokens(tmp_path, chain_order):
    run_dir, keep, order = _write_run(tmp_path, chain_order)
    _, conf = Boltz2Parser().parse_run(run_dir).load_model(MODEL)

    pae = np.load(run_dir / f"pae_{MODEL}.npz")["pae"]
    plddt = np.load(run_dir / f"plddt_{MODEL}.npz")["plddt"]
    np.testing.assert_array_equal(conf.pae_matrix, pae[np.ix_(keep, keep)])
    np.testing.assert_array_equal(conf.plddt_residue, plddt[keep])

    pair = json.loads((run_dir / f"confidence_{MODEL}.json").read_text())["pair_chains_iptm"]
    expected = pair[str(order.index("A"))][str(order.index("B"))]
    assert conf.pair_iptm("A", "B", ["A", "B"]) == expected

    out = process(str(run_dir), 8.0, 100.0, "best", skip_pae_png=True, skip_biophysical_scores=True)
    with out.open() as fh:
        (row,) = list(csv.DictReader(fh))
    assert row["interface"] == "A_B"
    assert row["iptm_scope"] == "chain_pair"
    assert float(row["iptm"]) == pytest.approx(expected)


def test_boltz2_pair_iptm_with_wrong_dimensions_is_dropped(tmp_path, caplog):
    run_dir, _, _ = _write_run(tmp_path, ("L", "A", "B"))
    confidence = run_dir / f"confidence_{MODEL}.json"
    payload = json.loads(confidence.read_text())
    payload["pair_chains_iptm"] = {"0": {"0": 0.9, "1": 0.2}, "1": {"0": 0.2, "1": 0.8}}
    confidence.write_text(json.dumps(payload))

    _, conf = Boltz2Parser().parse_run(run_dir).load_model(MODEL)
    assert conf.chain_pair_iptm is None
    assert "pair_chains_iptm dimensions" in caplog.text
