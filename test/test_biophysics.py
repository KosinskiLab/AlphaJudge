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
    shape_complementarity,
    zernike_shape_complementarity,
)
from alphajudge.biophysics.sc import interface_surface_dots
from alphajudge.biophysics.zernike import (
    GAUSSIAN_WEIGHTED_SCORE,
    GAP_ZERNIKE_BANDPASS_SCORE,
    GAP_ZERNIKE_EXCESS_BANDPASS_SCORE,
    GAP_ZERNIKE_EXCESS_CONTACT_SCORE,
    GAP_ZERNIKE_NONUNIFORM_SCORE,
    GAP_ZERNIKE_RATIO_SCORE,
    GAP_ZERNIKE_SOFT_BANDPASS_SCORE,
    GAP_ZERNIKE_WEIGHTED_SCORE,
    HARD_CUTOFF_SCORE,
    JOINT_LOW_ORDER_RATIO_SCORE,
    JOINT_RESIDUE_BEAD_GAUSSIAN,
    NORMAL_GAP_FIELD_SCORE,
    RESIDUE_BEAD_GAUSSIAN,
    SHARED_GRID_OVERLAP_SCORE,
    SURFACE_BINARY,
    SURFACE_GAUSSIAN,
    SURFACE_NORMAL_GAP,
    zernike_grids,
    zernike_gap_coefficient_bundle_from_grids,
    zernike_gap_score_diagnostics,
    zernike_score_from_grids,
)


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


def _jitter_distal_sidechains(residues, shift: tuple[float, float, float]):
    moved = copy.deepcopy(residues)
    delta = np.asarray(shift, dtype=float)
    for residue in moved:
        for atom in residue:
            if atom.id.strip().upper() not in {"N", "CA", "C", "O", "CB"}:
                atom.coord = atom.coord + delta
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


def test_residue_bead_zernike_is_translation_and_scale_invariant():
    residues_a, residues_b = _parse_residues(
        """\
ATOM      1  CA  LYS A   1       0.000   0.000   0.000  1.00 50.00           C
ATOM      2  CB  LYS A   1       1.400   0.200   0.200  1.00 50.00           C
ATOM      3  CG  LYS A   1       2.300   0.900   0.300  1.00 50.00           C
ATOM      4  CA  GLU A   2       0.200   2.100   0.300  1.00 50.00           C
ATOM      5  CB  GLU A   2       1.300   2.700   0.500  1.00 50.00           C
ATOM      6  CG  GLU A   2       2.300   3.300   0.600  1.00 50.00           C
ATOM      7  CA  ARG B   1       3.300   0.300   0.100  1.00 50.00           C
ATOM      8  CB  ARG B   1       4.400   0.900   0.200  1.00 50.00           C
ATOM      9  CG  ARG B   1       5.400   1.300   0.300  1.00 50.00           C
ATOM     10  CA  ASP B   2       3.200   2.200   0.400  1.00 50.00           C
ATOM     11  CB  ASP B   2       4.200   2.800   0.500  1.00 50.00           C
ATOM     12  CG  ASP B   2       5.100   3.300   0.700  1.00 50.00           C
TER
END
"""
    )

    translated_a = _transform_residues(residues_a, shift=(5.0, -3.0, 7.0))
    translated_b = _transform_residues(residues_b, shift=(5.0, -3.0, 7.0))
    scaled_a = _transform_residues(residues_a, scale=1.75, shift=(-8.0, 2.0, 1.5))
    scaled_b = _transform_residues(residues_b, scale=1.75, shift=(-8.0, 2.0, 1.5))

    baseline = zernike_shape_complementarity(
        residues_a,
        residues_b,
        representation=RESIDUE_BEAD_GAUSSIAN,
        grid_size=16,
        order=4,
        sigma=2.0,
    )
    translated = zernike_shape_complementarity(
        translated_a,
        translated_b,
        representation=RESIDUE_BEAD_GAUSSIAN,
        grid_size=16,
        order=4,
        sigma=2.0,
    )
    scaled = zernike_shape_complementarity(
        scaled_a,
        scaled_b,
        representation=RESIDUE_BEAD_GAUSSIAN,
        grid_size=16,
        order=4,
        sigma=2.0,
    )

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


def test_joint_low_order_ratio_returns_zero_without_contacts():
    residues_a, residues_b = _parse_residues(
        """\
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C
ATOM      2  CA  GLY B   1      20.000   0.000   0.000  1.00 50.00           C
TER
END
"""
    )

    assert (
        zernike_shape_complementarity(
            residues_a,
            residues_b,
            representation=JOINT_RESIDUE_BEAD_GAUSSIAN,
            score_mode=JOINT_LOW_ORDER_RATIO_SCORE,
            grid_size=16,
            order=4,
            sigma=2.0,
            fit_order=12,
        )
        == 0.0
    )


def test_gap_zernike_scores_are_bounded_symmetric_and_zero_without_contacts():
    residues_a, residues_b = _parse_residues(
        """\
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C
ATOM      2  CB  ALA A   1       1.400   0.400   0.200  1.00 50.00           C
ATOM      3  CA  LEU A   2       0.100   2.000   0.300  1.00 50.00           C
ATOM      4  CB  LEU A   2       1.300   2.500   0.500  1.00 50.00           C
ATOM      5  CA  GLY B   1       2.900   0.200   0.100  1.00 50.00           C
ATOM      6  O   GLY B   1       3.700   0.600   0.100  1.00 50.00           O
ATOM      7  CA  SER B   2       3.000   2.100   0.400  1.00 50.00           C
ATOM      8  OG  SER B   2       3.900   2.600   0.800  1.00 50.00           O
TER
END
"""
    )
    far_b = _transform_residues(residues_b, shift=(30.0, 0.0, 0.0))

    for score_mode in (
        SHARED_GRID_OVERLAP_SCORE,
        GAP_ZERNIKE_RATIO_SCORE,
        GAP_ZERNIKE_WEIGHTED_SCORE,
        GAP_ZERNIKE_NONUNIFORM_SCORE,
        GAP_ZERNIKE_BANDPASS_SCORE,
        GAP_ZERNIKE_EXCESS_BANDPASS_SCORE,
        GAP_ZERNIKE_SOFT_BANDPASS_SCORE,
        GAP_ZERNIKE_EXCESS_CONTACT_SCORE,
    ):
        ab = zernike_shape_complementarity(
            residues_a,
            residues_b,
            representation=RESIDUE_BEAD_GAUSSIAN,
            grid_size=16,
            order=4,
            sigma=2.0,
            score_mode=score_mode,
            fit_order=12,
        )
        ba = zernike_shape_complementarity(
            residues_b,
            residues_a,
            representation=RESIDUE_BEAD_GAUSSIAN,
            grid_size=16,
            order=4,
            sigma=2.0,
            score_mode=score_mode,
            fit_order=12,
        )
        no_contact = zernike_shape_complementarity(
            residues_a,
            far_b,
            representation=RESIDUE_BEAD_GAUSSIAN,
            grid_size=16,
            order=4,
            sigma=2.0,
            score_mode=score_mode,
            fit_order=12,
        )

        assert 0.0 <= ab <= 1.0
        assert ab == pytest.approx(ba, abs=1e-6)
        assert no_contact == 0.0


def test_gap_zernike_scores_are_translation_and_scale_stable():
    residues_a, residues_b = _parse_residues(
        """\
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C
ATOM      2  CB  ALA A   1       1.400   0.400   0.200  1.00 50.00           C
ATOM      3  CA  LEU A   2       0.100   2.000   0.300  1.00 50.00           C
ATOM      4  CB  LEU A   2       1.300   2.500   0.500  1.00 50.00           C
ATOM      5  CA  GLY B   1       2.900   0.200   0.100  1.00 50.00           C
ATOM      6  O   GLY B   1       3.700   0.600   0.100  1.00 50.00           O
ATOM      7  CA  SER B   2       3.000   2.100   0.400  1.00 50.00           C
ATOM      8  OG  SER B   2       3.900   2.600   0.800  1.00 50.00           O
TER
END
"""
    )
    translated_a = _transform_residues(residues_a, shift=(11.0, -3.0, 4.0))
    translated_b = _transform_residues(residues_b, shift=(11.0, -3.0, 4.0))
    scaled_a = _transform_residues(residues_a, scale=1.8, shift=(-5.0, 2.0, 0.5))
    scaled_b = _transform_residues(residues_b, scale=1.8, shift=(-5.0, 2.0, 0.5))

    baseline = zernike_shape_complementarity(
        residues_a,
        residues_b,
        representation=RESIDUE_BEAD_GAUSSIAN,
        grid_size=16,
        order=4,
        sigma=2.0,
        score_mode=GAP_ZERNIKE_RATIO_SCORE,
        fit_order=12,
    )
    translated = zernike_shape_complementarity(
        translated_a,
        translated_b,
        representation=RESIDUE_BEAD_GAUSSIAN,
        grid_size=16,
        order=4,
        sigma=2.0,
        score_mode=GAP_ZERNIKE_RATIO_SCORE,
        fit_order=12,
    )
    scaled = zernike_shape_complementarity(
        scaled_a,
        scaled_b,
        representation=RESIDUE_BEAD_GAUSSIAN,
        grid_size=16,
        order=4,
        sigma=3.6,
        distance=14.4,
        padding=3.6,
        score_mode=GAP_ZERNIKE_RATIO_SCORE,
        fit_order=12,
    )

    assert translated == pytest.approx(baseline, abs=1e-6)
    assert scaled == pytest.approx(baseline, abs=5e-2)


def test_gap_zernike_contact_scores_higher_than_separated_toy_interface():
    residues_a, residues_b = _parse_residues(
        """\
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C
ATOM      2  CB  ALA A   1       1.300   0.300   0.100  1.00 50.00           C
ATOM      3  CA  SER B   1       2.700   0.100   0.100  1.00 50.00           C
ATOM      4  CB  SER B   1       3.800   0.700   0.300  1.00 50.00           C
TER
END
"""
    )
    far_b = _transform_residues(residues_b, shift=(25.0, 0.0, 0.0))
    contacting = zernike_shape_complementarity(
        residues_a,
        residues_b,
        representation=RESIDUE_BEAD_GAUSSIAN,
        grid_size=16,
        order=4,
        sigma=2.0,
        score_mode=GAP_ZERNIKE_WEIGHTED_SCORE,
        fit_order=12,
    )
    separated = zernike_shape_complementarity(
        residues_a,
        far_b,
        representation=RESIDUE_BEAD_GAUSSIAN,
        grid_size=16,
        order=4,
        sigma=2.0,
        score_mode=GAP_ZERNIKE_WEIGHTED_SCORE,
        fit_order=12,
    )

    assert contacting > separated


def test_calibrated_atom_gap_scores_are_translation_and_scale_stable():
    residues_a, residues_b = _parse_residues(
        """\
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C
ATOM      2  CB  ALA A   1       1.300   0.300   0.100  1.00 50.00           C
ATOM      3  CA  SER B   1       2.700   0.100   0.100  1.00 50.00           C
ATOM      4  CB  SER B   1       3.800   0.700   0.300  1.00 50.00           C
TER
END
"""
    )
    translated_a = _transform_residues(residues_a, shift=(6.0, -4.0, 2.5))
    translated_b = _transform_residues(residues_b, shift=(6.0, -4.0, 2.5))
    scaled_a = _transform_residues(residues_a, scale=2.0, shift=(-3.0, 1.0, 0.5))
    scaled_b = _transform_residues(residues_b, scale=2.0, shift=(-3.0, 1.0, 0.5))

    for score_mode in (
        GAP_ZERNIKE_EXCESS_BANDPASS_SCORE,
        GAP_ZERNIKE_SOFT_BANDPASS_SCORE,
        GAP_ZERNIKE_EXCESS_CONTACT_SCORE,
    ):
        kwargs = {
            "representation": "atom_gaussian",
            "grid_size": 16,
            "order": 4,
            "sigma": 1.5,
            "score_mode": score_mode,
            "fit_order": 12,
        }
        baseline = zernike_shape_complementarity(residues_a, residues_b, **kwargs)
        translated = zernike_shape_complementarity(translated_a, translated_b, **kwargs)
        scaled = zernike_shape_complementarity(
            scaled_a,
            scaled_b,
            **{
                **kwargs,
                "sigma": 3.0,
                "distance": 16.0,
                "padding": 4.0,
            },
        )

        assert 0.0 <= baseline <= 1.0
        assert translated == pytest.approx(baseline, abs=1e-6)
        assert scaled == pytest.approx(baseline, abs=5e-2)


def test_calibrated_gap_diagnostics_reuse_order12_coefficients():
    residues_a, residues_b = _parse_residues(
        """\
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C
ATOM      2  CB  ALA A   1       1.400   0.400   0.200  1.00 50.00           C
ATOM      3  CA  SER B   1       2.900   0.200   0.100  1.00 50.00           C
ATOM      4  CB  SER B   1       3.900   0.700   0.400  1.00 50.00           C
TER
END
"""
    )
    grid1, grid2 = zernike_grids(
        residues_a,
        residues_b,
        representation="atom_gaussian",
        grid_size=16,
        sigma=1.5,
    )
    gap_coeff, overlap, volume1, volume2, voxel_count = zernike_gap_coefficient_bundle_from_grids(
        grid1,
        grid2,
        fit_order=12,
    )
    from_grids = zernike_score_from_grids(
        grid1,
        grid2,
        order=4,
        score_mode=GAP_ZERNIKE_EXCESS_BANDPASS_SCORE,
        fit_order=12,
    )
    diagnostics = zernike_gap_score_diagnostics(
        gap_coeff,
        overlap,
        order=4,
        score_mode=GAP_ZERNIKE_EXCESS_BANDPASS_SCORE,
        fit_order=12,
        side1_effective_volume=volume1,
        side2_effective_volume=volume2,
        voxel_count=voxel_count,
    )

    assert diagnostics["gap_final_score"] == pytest.approx(from_grids, abs=1e-8)
    assert 0.0 <= diagnostics["gap_excess_overlap"] <= diagnostics["gap_raw_overlap"] <= 1.0
    assert diagnostics["gap_expected_random_overlap"] >= 0.0


def test_lower_n_atom_gap_is_less_sidechain_sensitive_than_higher_n():
    residues_a, residues_b = _parse_residues(
        """\
ATOM      1  N   LYS A   1      -0.200  -0.300   0.000  1.00 50.00           N
ATOM      2  CA  LYS A   1       0.000   0.000   0.000  1.00 50.00           C
ATOM      3  C   LYS A   1       0.500   1.300   0.100  1.00 50.00           C
ATOM      4  O   LYS A   1       0.300   2.300   0.000  1.00 50.00           O
ATOM      5  CB  LYS A   1       1.400  -0.400   0.300  1.00 50.00           C
ATOM      6  CG  LYS A   1       2.400   0.300   0.700  1.00 50.00           C
ATOM      7  CD  LYS A   1       3.500  -0.200   1.100  1.00 50.00           C
ATOM      8  CE  LYS A   1       4.500   0.500   1.500  1.00 50.00           C
ATOM      9  NZ  LYS A   1       5.500   0.000   1.800  1.00 50.00           N
ATOM     10  N   GLU B   1       2.000  -0.200   0.200  1.00 50.00           N
ATOM     11  CA  GLU B   1       3.000   0.200   0.300  1.00 50.00           C
ATOM     12  C   GLU B   1       3.400   1.500   0.500  1.00 50.00           C
ATOM     13  O   GLU B   1       3.200   2.500   0.400  1.00 50.00           O
ATOM     14  CB  GLU B   1       4.000  -0.500   0.700  1.00 50.00           C
ATOM     15  CG  GLU B   1       4.900   0.200   1.300  1.00 50.00           C
ATOM     16  CD  GLU B   1       5.800  -0.500   1.900  1.00 50.00           C
ATOM     17  OE1 GLU B   1       6.700   0.200   2.300  1.00 50.00           O
ATOM     18  OE2 GLU B   1       5.700  -1.700   2.000  1.00 50.00           O
TER
END
"""
    )
    jittered_a = _jitter_distal_sidechains(residues_a, shift=(1.2, -0.7, 0.5))
    jittered_b = _jitter_distal_sidechains(residues_b, shift=(-1.0, 0.8, -0.4))

    kwargs = {
        "representation": "atom_gaussian",
        "grid_size": 16,
        "sigma": 1.5,
        "score_mode": GAP_ZERNIKE_BANDPASS_SCORE,
        "fit_order": 12,
    }
    low_delta = abs(
        zernike_shape_complementarity(residues_a, residues_b, order=2, **kwargs)
        - zernike_shape_complementarity(jittered_a, jittered_b, order=2, **kwargs)
    )
    high_delta = abs(
        zernike_shape_complementarity(residues_a, residues_b, order=8, **kwargs)
        - zernike_shape_complementarity(jittered_a, jittered_b, order=8, **kwargs)
    )

    assert low_delta <= high_delta


def test_normal_gap_zernike_is_bounded_symmetric_and_zero_without_contacts():
    residues_a, residues_b = _parse_residues(
        """\
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C
ATOM      2  CB  ALA A   1       1.300   0.300   0.100  1.00 50.00           C
ATOM      3  CA  SER B   1       2.700   0.100   0.100  1.00 50.00           C
ATOM      4  CB  SER B   1       3.800   0.700   0.300  1.00 50.00           C
TER
END
"""
    )
    far_b = _transform_residues(residues_b, shift=(25.0, 0.0, 0.0))

    ab = zernike_shape_complementarity(
        residues_a,
        residues_b,
        representation=SURFACE_NORMAL_GAP,
        score_mode=NORMAL_GAP_FIELD_SCORE,
        grid_size=16,
        order=4,
        sigma=1.5,
        surface_density=3.0,
        surface_probe_radius=2.3,
        surface_trim_cutoff=3.0,
        fit_order=8,
    )
    ba = zernike_shape_complementarity(
        residues_b,
        residues_a,
        representation=SURFACE_NORMAL_GAP,
        score_mode=NORMAL_GAP_FIELD_SCORE,
        grid_size=16,
        order=4,
        sigma=1.5,
        surface_density=3.0,
        surface_probe_radius=2.3,
        surface_trim_cutoff=3.0,
        fit_order=8,
    )
    no_contact = zernike_shape_complementarity(
        residues_a,
        far_b,
        representation=SURFACE_NORMAL_GAP,
        score_mode=NORMAL_GAP_FIELD_SCORE,
        grid_size=16,
        order=4,
        sigma=1.5,
        surface_density=3.0,
        surface_probe_radius=2.3,
        surface_trim_cutoff=3.0,
        fit_order=8,
    )

    assert 0.0 <= ab <= 1.0
    assert ab == pytest.approx(ba, abs=1e-6)
    assert no_contact == 0.0


def test_normal_gap_zernike_is_translation_stable():
    residues_a, residues_b = _parse_residues(
        """\
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C
ATOM      2  CB  ALA A   1       1.300   0.300   0.100  1.00 50.00           C
ATOM      3  CA  SER B   1       2.700   0.100   0.100  1.00 50.00           C
ATOM      4  CB  SER B   1       3.800   0.700   0.300  1.00 50.00           C
TER
END
"""
    )
    translated_a = _transform_residues(residues_a, shift=(11.0, -3.0, 4.0))
    translated_b = _transform_residues(residues_b, shift=(11.0, -3.0, 4.0))

    kwargs = {
        "representation": SURFACE_NORMAL_GAP,
        "score_mode": NORMAL_GAP_FIELD_SCORE,
        "grid_size": 16,
        "order": 4,
        "sigma": 1.5,
        "surface_density": 3.0,
        "surface_probe_radius": 2.3,
        "surface_trim_cutoff": 3.0,
        "fit_order": 8,
    }
    baseline = zernike_shape_complementarity(residues_a, residues_b, **kwargs)
    translated = zernike_shape_complementarity(translated_a, translated_b, **kwargs)

    assert translated == pytest.approx(baseline, abs=1e-6)


def test_zernike_fit_order_reuses_lower_order_for_hard_and_weighted_scores():
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

    grid1, grid2 = zernike_grids(
        residues_a,
        residues_b,
        representation=RESIDUE_BEAD_GAUSSIAN,
        grid_size=16,
        sigma=2.0,
    )

    direct_hard = zernike_score_from_grids(
        grid1,
        grid2,
        order=4,
        score_mode=HARD_CUTOFF_SCORE,
    )
    reused_hard = zernike_score_from_grids(
        grid1,
        grid2,
        order=4,
        score_mode=HARD_CUTOFF_SCORE,
        fit_order=12,
    )
    direct_weighted = zernike_score_from_grids(
        grid1,
        grid2,
        order=4,
        score_mode=GAUSSIAN_WEIGHTED_SCORE,
        order_decay_n0=4.0,
    )
    reused_weighted = zernike_score_from_grids(
        grid1,
        grid2,
        order=4,
        score_mode=GAUSSIAN_WEIGHTED_SCORE,
        fit_order=12,
        order_decay_n0=4.0,
    )

    assert reused_hard == pytest.approx(direct_hard, abs=1e-6)
    assert reused_weighted == pytest.approx(direct_weighted, abs=1e-6)


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


def test_surface_probe_override_is_deterministic_and_preserves_sc_defaults():
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

    d1a, n1a, d2a, n2a = interface_surface_dots(residues_a, residues_b)
    d1b, n1b, d2b, n2b = interface_surface_dots(residues_a, residues_b, probe_radius=1.7)

    assert np.array_equal(d1a, d1b)
    assert np.array_equal(n1a, n1b)
    assert np.array_equal(d2a, d2b)
    assert np.array_equal(n2a, n2b)
    assert shape_complementarity(residues_a, residues_b) == pytest.approx(
        shape_complementarity(residues_a, residues_b),
        abs=1e-12,
    )

    grid1_a, grid2_a = zernike_grids(
        residues_a,
        residues_b,
        representation=SURFACE_GAUSSIAN,
        grid_size=16,
        sigma=1.5,
        surface_density=3.0,
        surface_probe_radius=2.3,
    )
    grid1_b, grid2_b = zernike_grids(
        residues_a,
        residues_b,
        representation=SURFACE_GAUSSIAN,
        grid_size=16,
        sigma=1.5,
        surface_density=3.0,
        surface_probe_radius=2.3,
    )

    assert np.array_equal(grid1_a, grid1_b)
    assert np.array_equal(grid2_a, grid2_b)


def test_sidechain_jitter_perturbs_residue_bead_less_than_atom_and_sc():
    residues_a, residues_b = _parse_residues(
        """\
ATOM      1  N   LYS A   1      -0.200  -0.300   0.000  1.00 50.00           N
ATOM      2  CA  LYS A   1       0.000   0.000   0.000  1.00 50.00           C
ATOM      3  C   LYS A   1       0.500   1.300   0.100  1.00 50.00           C
ATOM      4  O   LYS A   1       0.300   2.300   0.000  1.00 50.00           O
ATOM      5  CB  LYS A   1       1.400  -0.400   0.300  1.00 50.00           C
ATOM      6  CG  LYS A   1       2.400   0.300   0.700  1.00 50.00           C
ATOM      7  CD  LYS A   1       3.500  -0.200   1.100  1.00 50.00           C
ATOM      8  CE  LYS A   1       4.500   0.500   1.500  1.00 50.00           C
ATOM      9  NZ  LYS A   1       5.500   0.000   1.800  1.00 50.00           N
ATOM     10  N   GLU B   1       2.000  -0.200   0.200  1.00 50.00           N
ATOM     11  CA  GLU B   1       3.000   0.200   0.300  1.00 50.00           C
ATOM     12  C   GLU B   1       3.400   1.500   0.500  1.00 50.00           C
ATOM     13  O   GLU B   1       3.200   2.500   0.400  1.00 50.00           O
ATOM     14  CB  GLU B   1       4.000  -0.500   0.700  1.00 50.00           C
ATOM     15  CG  GLU B   1       4.900   0.200   1.300  1.00 50.00           C
ATOM     16  CD  GLU B   1       5.800  -0.500   1.900  1.00 50.00           C
ATOM     17  OE1 GLU B   1       6.700   0.200   2.300  1.00 50.00           O
ATOM     18  OE2 GLU B   1       5.700  -1.700   2.000  1.00 50.00           O
TER
END
"""
    )

    jittered_a = _jitter_distal_sidechains(residues_a, shift=(1.2, -0.7, 0.5))
    jittered_b = _jitter_distal_sidechains(residues_b, shift=(-1.0, 0.8, -0.4))

    sc_delta = abs(shape_complementarity(jittered_a, jittered_b) - shape_complementarity(residues_a, residues_b))
    atom_delta = abs(
        zernike_shape_complementarity(residues_a, residues_b, representation="atom_gaussian", grid_size=16, order=4, sigma=1.5)
        - zernike_shape_complementarity(jittered_a, jittered_b, representation="atom_gaussian", grid_size=16, order=4, sigma=1.5)
    )
    residue_delta = abs(
        zernike_shape_complementarity(
            residues_a,
            residues_b,
            representation=RESIDUE_BEAD_GAUSSIAN,
            grid_size=16,
            order=4,
            sigma=2.0,
        )
        - zernike_shape_complementarity(
            jittered_a,
            jittered_b,
            representation=RESIDUE_BEAD_GAUSSIAN,
            grid_size=16,
            order=4,
            sigma=2.0,
        )
    )

    assert residue_delta < atom_delta
    assert residue_delta < sc_delta


def test_sidechain_jitter_perturbs_residue_gap_score_less_than_atom_and_sc():
    residues_a, residues_b = _parse_residues(
        """\
ATOM      1  N   LYS A   1      -0.200  -0.300   0.000  1.00 50.00           N
ATOM      2  CA  LYS A   1       0.000   0.000   0.000  1.00 50.00           C
ATOM      3  C   LYS A   1       0.500   1.300   0.100  1.00 50.00           C
ATOM      4  O   LYS A   1       0.300   2.300   0.000  1.00 50.00           O
ATOM      5  CB  LYS A   1       1.400  -0.400   0.300  1.00 50.00           C
ATOM      6  CG  LYS A   1       2.400   0.300   0.700  1.00 50.00           C
ATOM      7  CD  LYS A   1       3.500  -0.200   1.100  1.00 50.00           C
ATOM      8  CE  LYS A   1       4.500   0.500   1.500  1.00 50.00           C
ATOM      9  NZ  LYS A   1       5.500   0.000   1.800  1.00 50.00           N
ATOM     10  N   GLU B   1       2.000  -0.200   0.200  1.00 50.00           N
ATOM     11  CA  GLU B   1       3.000   0.200   0.300  1.00 50.00           C
ATOM     12  C   GLU B   1       3.400   1.500   0.500  1.00 50.00           C
ATOM     13  O   GLU B   1       3.200   2.500   0.400  1.00 50.00           O
ATOM     14  CB  GLU B   1       4.000  -0.500   0.700  1.00 50.00           C
ATOM     15  CG  GLU B   1       4.900   0.200   1.300  1.00 50.00           C
ATOM     16  CD  GLU B   1       5.800  -0.500   1.900  1.00 50.00           C
ATOM     17  OE1 GLU B   1       6.700   0.200   2.300  1.00 50.00           O
ATOM     18  OE2 GLU B   1       5.700  -1.700   2.000  1.00 50.00           O
TER
END
"""
    )

    jittered_a = _jitter_distal_sidechains(residues_a, shift=(1.2, -0.7, 0.5))
    jittered_b = _jitter_distal_sidechains(residues_b, shift=(-1.0, 0.8, -0.4))

    sc_delta = abs(shape_complementarity(jittered_a, jittered_b) - shape_complementarity(residues_a, residues_b))
    atom_delta = abs(
        zernike_shape_complementarity(
            residues_a,
            residues_b,
            representation="atom_gaussian",
            grid_size=16,
            order=4,
            sigma=1.5,
            score_mode=GAP_ZERNIKE_RATIO_SCORE,
            fit_order=12,
        )
        - zernike_shape_complementarity(
            jittered_a,
            jittered_b,
            representation="atom_gaussian",
            grid_size=16,
            order=4,
            sigma=1.5,
            score_mode=GAP_ZERNIKE_RATIO_SCORE,
            fit_order=12,
        )
    )
    residue_delta = abs(
        zernike_shape_complementarity(
            residues_a,
            residues_b,
            representation=RESIDUE_BEAD_GAUSSIAN,
            grid_size=16,
            order=4,
            sigma=2.0,
            score_mode=GAP_ZERNIKE_RATIO_SCORE,
            fit_order=12,
        )
        - zernike_shape_complementarity(
            jittered_a,
            jittered_b,
            representation=RESIDUE_BEAD_GAUSSIAN,
            grid_size=16,
            order=4,
            sigma=2.0,
            score_mode=GAP_ZERNIKE_RATIO_SCORE,
            fit_order=12,
        )
    )

    assert residue_delta < atom_delta
    assert residue_delta < sc_delta
