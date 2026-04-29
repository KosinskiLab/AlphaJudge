from __future__ import annotations

from io import StringIO

import pytest
from Bio.PDB import PDBParser

from alphajudge.biophysics import (
    buried_surface_area,
    disulfide_bonds,
    hydrogen_bonds,
    salt_bridges,
    zernike_shape_complementarity,
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


def test_pisa_interface_residue_filter_survives_cached_reparse():
    pdb_text = """\
ATOM      1  NE  ARG A   1       0.000   0.000   0.000  1.00 50.00           N
ATOM      2  OD1 ASP B   1       3.000   0.000   0.000  1.00 50.00           O
TER
END
"""
    first_a, first_b = _parse_residues(pdb_text)
    assert salt_bridges(first_a, first_b) == 1

    second_a, second_b = _parse_residues(pdb_text)
    assert salt_bridges(second_a, second_b) == 1


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


def test_zernike_shape_complementarity_is_symmetric_and_bounded():
    residues_a, residues_b = _parse_residues(
        """\
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
    )

    ab = zernike_shape_complementarity(residues_a, residues_b, grid_size=16, order=4, sigma=1.0)
    ba = zernike_shape_complementarity(residues_b, residues_a, grid_size=16, order=4, sigma=1.0)

    assert ab == pytest.approx(ba, abs=1e-6)
    assert 0.0 <= ab <= 1.0


def test_zernike_shape_complementarity_returns_zero_without_contacts():
    residues_a, residues_b = _parse_residues(
        """\
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C
ATOM      2  CA  GLY B   1      20.000   0.000   0.000  1.00 50.00           C
TER
END
"""
    )

    assert zernike_shape_complementarity(residues_a, residues_b, grid_size=16, order=4) == 0.0
