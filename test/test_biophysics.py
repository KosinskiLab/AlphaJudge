from __future__ import annotations

import copy
from io import StringIO

import numpy as np
import pytest
from Bio.PDB import PDBParser

from alphajudge.biophysics import (
    buried_surface_area,
    disulfide_bonds,
    hydrogen_bonds,
    salt_bridges,
    zernike_shape_complementarity,
)
from alphajudge.biophysics.zernike import SURFACE_BINARY, zernike_grids


def _parse_residues(pdb_text: str):
    structure = PDBParser(QUIET=True).get_structure("x", StringIO(pdb_text))
    return list(structure[0]["A"]), list(structure[0]["B"])


def _transform_residues(residues, *, scale: float = 1.0, shift: tuple[float, float, float] = (0, 0, 0)):
    moved = copy.deepcopy(residues)
    delta = np.asarray(shift, dtype=float)
    for residue in moved:
        for atom in residue:
            atom.coord = atom.coord * float(scale) + delta
    return moved


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
    assert -1.0 <= ab <= 1.0


def test_zernike_shape_complementarity_is_translation_and_scale_invariant():
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

    translated_a = _transform_residues(residues_a, shift=(12.5, -7.0, 3.5))
    translated_b = _transform_residues(residues_b, shift=(12.5, -7.0, 3.5))
    scaled_a = _transform_residues(residues_a, scale=2.75, shift=(-4.0, 1.0, 8.0))
    scaled_b = _transform_residues(residues_b, scale=2.75, shift=(-4.0, 1.0, 8.0))

    baseline = zernike_shape_complementarity(residues_a, residues_b, grid_size=16, order=4, sigma=1.0)
    translated = zernike_shape_complementarity(translated_a, translated_b, grid_size=16, order=4, sigma=1.0)
    scaled = zernike_shape_complementarity(scaled_a, scaled_b, grid_size=16, order=4, sigma=1.0)

    assert translated == pytest.approx(baseline, abs=1e-6)
    assert scaled == pytest.approx(baseline, abs=5e-3)


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


def test_surface_binary_zernike_grids_are_deterministic():
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

    grid1_a, grid2_a = zernike_grids(
        residues_a,
        residues_b,
        representation=SURFACE_BINARY,
        grid_size=16,
        sigma=1.0,
    )
    grid1_b, grid2_b = zernike_grids(
        residues_a,
        residues_b,
        representation=SURFACE_BINARY,
        grid_size=16,
        sigma=1.0,
    )

    assert np.array_equal(grid1_a, grid1_b)
    assert np.array_equal(grid2_a, grid2_b)
    assert float(np.sum(grid1_a)) > 0.0
    assert float(np.sum(grid2_a)) > 0.0
