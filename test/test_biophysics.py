from __future__ import annotations

from io import StringIO

from Bio.PDB import PDBParser

from alphajudge.biophysics import (
    buried_surface_area,
    disulfide_bonds,
    hydrogen_bonds,
    salt_bridges,
)


def _parse_residues(pdb_text: str):
    structure = PDBParser(QUIET=True).get_structure("x", StringIO(pdb_text))
    return list(structure[0]["A"]), list(structure[0]["B"])


def test_pisa_style_bond_counters_count_cross_chain_pairs_once():
    residues_a, residues_b = _parse_residues(
        """\
ATOM      1  NE  ARG A   1       0.000   0.000   0.000  1.00 50.00           N
ATOM      2  SG  CYS A   2      20.000   0.000   0.000  1.00 50.00           S
ATOM      3  OD1 ASP B   1       3.000   0.000   0.000  1.00 50.00           O
ATOM      4  SG  CYS B   2      22.030   0.000   0.000  1.00 50.00           S
TER
END
"""
    )

    assert hydrogen_bonds(residues_a, residues_b) == 0
    assert salt_bridges(residues_a, residues_b) == 1
    assert disulfide_bonds(residues_a, residues_b) == 1


def test_ccp4srs_style_hbond_requires_monomer_bond_geometry():
    residues_a, residues_b = _parse_residues(
        """\
ATOM      1  CG  ASN A   1      -1.000   0.000   0.000  1.00 50.00           C
ATOM      2  ND2 ASN A   1       0.000   0.000   0.000  1.00 50.00           N
ATOM      3  O   ALA B   1       3.000   0.000   0.000  1.00 50.00           O
ATOM      4  C   ALA B   1       4.000   0.000   0.000  1.00 50.00           C
TER
END
"""
    )

    assert hydrogen_bonds(residues_a, residues_b) == 1


def test_pisa_style_buried_surface_area_is_positive_for_contacting_atoms():
    residues_a, residues_b = _parse_residues(
        """\
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C
ATOM      2  CA  ALA B   1       3.200   0.000   0.000  1.00 50.00           C
TER
END
"""
    )

    assert buried_surface_area(residues_a, residues_b) > 0.0
