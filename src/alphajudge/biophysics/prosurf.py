"""PISA ProSurf-style solvent-accessible and buried interface area."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.spatial import cKDTree

from .connolly import get_radius
from .pisa_molref import PISA_STANDARD_AA_RADII

PISA_PROBE_RADIUS = 1.4  # Angstrom, default solvent probe in pisa_prosurf.cpp
PISA_CODE_NO = 36        # default spherical code size in pisa_prosurf.cpp

_PISA_ELEMENT_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "P": 1.80,
    "S": 1.80,
    "SE": 1.80,
    "F": 1.47,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
}


def _atom_element(atom) -> str:
    element = (getattr(atom, "element", "") or "").strip().upper()
    if element and element != "X":
        return element
    name = atom.id.strip().upper()
    if not name:
        return ""
    if len(name) >= 2 and name[:2] in _PISA_ELEMENT_RADII:
        return name[:2]
    return name[0]


def _pisa_radius(atom) -> float:
    """PISA-style VdW radius used by ProSurf for SAS calculations."""
    residue = atom.get_parent()
    resname = residue.get_resname().strip().upper()
    atom_name = atom.id.strip().upper()
    molref_radius = PISA_STANDARD_AA_RADII.get(resname, {}).get(atom_name)
    if molref_radius is not None:
        return molref_radius

    element = _atom_element(atom)
    if element in _PISA_ELEMENT_RADII:
        return _PISA_ELEMENT_RADII[element]
    return get_radius(resname, atom_name)


def _collect_surface_atoms(residues: Iterable) -> tuple[np.ndarray, np.ndarray, list]:
    """Heavy atoms of ``residues`` with their coordinates and PISA radii."""
    atoms = [atom for residue in residues for atom in residue if _atom_element(atom) != "H"]
    if not atoms:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float), []
    coords = np.asarray([atom.coord for atom in atoms], dtype=float)
    radii = np.asarray([_pisa_radius(atom) for atom in atoms], dtype=float)
    return coords, radii, atoms


def _mround(value: float) -> int:
    return int(math.floor(value + 0.5))


@lru_cache(maxsize=8)
def _pisa_spherical_code(code_no: int = PISA_CODE_NO) -> tuple[np.ndarray, np.ndarray]:
    """Port of ProSurf::calcSphericalCode from CCP4 PISA."""
    min_code_no = 6
    code_no = max(min_code_no, int(code_no))
    dalpha = math.pi / (code_no - 1)

    points = []
    areas = []
    for i in range(1, code_no + 1):
        if i in (1, code_no):
            points.append((0.0, 0.0, 1.0 if i == 1 else -1.0))
            areas.append(2.0 * math.pi * (1.0 - math.cos(dalpha / 2.0)))
        else:
            beta = (i - 1) * dalpha
            nr = max(min_code_no, _mround(code_no * math.sin(beta)))
            dbeta = 2.0 * math.pi / nr
            z = math.cos(beta)
            xy = math.sin(beta)
            a = 2.0 * dbeta * math.sin(beta) * math.sin(dalpha / 2.0)
            for j in range(nr):
                alpha = j * dbeta
                points.append((xy * math.cos(alpha), xy * math.sin(alpha), z))
                areas.append(a)

    return np.asarray(points, dtype=float), np.asarray(areas, dtype=float)


@dataclass(frozen=True)
class _PisaInterfaceResult:
    area: float
    residue_keys1: frozenset[tuple]
    residue_keys2: frozenset[tuple]
    atoms1: tuple
    atom_sas1: tuple[float, ...]
    atom_int_sas1: tuple[float, ...]
    atoms2: tuple
    atom_sas2: tuple[float, ...]
    atom_int_sas2: tuple[float, ...]


_PISA_INTERFACE_CACHE: dict[tuple, _PisaInterfaceResult] = {}


def _residue_fingerprint(residue) -> tuple:
    """Stable identity for a residue across GC — id() alone is reused by CPython."""
    first_atom = next(iter(residue), None)
    if first_atom is None:
        return (residue.full_id, None)
    coord = first_atom.coord
    return (residue.full_id, (float(coord[0]), float(coord[1]), float(coord[2])))


def _interface_cache_key(residues1, residues2, probe_radius: float, code_no: int) -> tuple:
    return (
        tuple(_residue_fingerprint(residue) for residue in residues1),
        tuple(_residue_fingerprint(residue) for residue in residues2),
        float(probe_radius),
        int(code_no),
    )


def _pisa_interface_result(
    residues1,
    residues2,
    probe_radius: float = PISA_PROBE_RADIUS,
    code_no: int = PISA_CODE_NO,
) -> _PisaInterfaceResult:
    """
    PISA ProSurf interface area plus residue selections.

    ProSurf selects atoms with nonzero interface area into ``selHndInt1/2``.
    PISA later expands those atom selections to residues before calling
    ccp4srs::CalcHBonds. Keeping stable selected residue keys here lets area and
    bond scoring share the same source of truth.
    """
    residues1 = tuple(residues1)
    residues2 = tuple(residues2)
    key = _interface_cache_key(residues1, residues2, probe_radius, code_no)
    cached = _PISA_INTERFACE_CACHE.get(key)
    if cached is not None:
        return cached

    coords1, radii1, atoms1 = _collect_surface_atoms(residues1)
    coords2, radii2, atoms2 = _collect_surface_atoms(residues2)
    if len(coords1) == 0 or len(coords2) == 0:
        result = _PisaInterfaceResult(0.0, frozenset(), frozenset(), tuple(atoms1), (), (), tuple(atoms2), (), ())
    else:
        code_points, code_areas = _pisa_spherical_code(code_no)
        side1 = (coords1, radii1 + float(probe_radius), atoms1)
        side2 = (coords2, radii2 + float(probe_radius), atoms2)
        area1, keys1, sas1, int_sas1 = _side_interface_area(side1, side2, code_points, code_areas)
        area2, keys2, sas2, int_sas2 = _side_interface_area(side2, side1, code_points, code_areas)
        result = _PisaInterfaceResult(
            float((area1 + area2) / 2.0),
            keys1, keys2,
            tuple(atoms1), sas1, int_sas1,
            tuple(atoms2), sas2, int_sas2,
        )
    if len(_PISA_INTERFACE_CACHE) > 32:
        _PISA_INTERFACE_CACHE.clear()
    _PISA_INTERFACE_CACHE[key] = result
    return result


def _uncovered_code_points(coord, ri, code_points, coords, radii, neighbours, skip: int = -1) -> np.ndarray:
    """Mask of the atom's spherical-code points not covered by any neighbour sphere."""
    mask = np.ones(len(code_points), dtype=bool)
    for j in neighbours:
        if j == skip:
            continue
        rj = float(radii[j])
        delta = coord - coords[j]
        if float(np.dot(delta, delta)) > (ri + rj) ** 2:
            continue
        surface_vectors = delta + ri * code_points
        mask &= ~(np.einsum("ij,ij->i", surface_vectors, surface_vectors) <= (rj * rj + 0.00001))
        if not mask.any():
            break
    return mask


def _side_interface_area(own, other, code_points, code_areas):
    """Interface area of one side: its accessible surface that the other side covers.

    Each side is (coords, probe-extended radii, atoms). Returns the area, the
    fingerprints of residues with interface area, and per-atom SAS and
    interface SAS.

    Like ProSurf, SAS is computed for every atom of a residue that has any atom
    near the other side, not only for the atoms that are: the solvation energy
    types charged groups by comparing SAS between sibling atoms of a residue.
    Residues wholly away from the interface keep SAS 0; their contribution is
    the same in the free and bound states and cancels.
    """
    own_coords, own_radii, own_atoms = own
    other_coords, other_radii, _ = other
    own_tree, other_tree = cKDTree(own_coords), cKDTree(other_coords)
    own_max_radius, other_max_radius = float(np.max(own_radii)), float(np.max(other_radii))
    other_neighbours = [
        other_tree.query_ball_point(coord, float(ri) + other_max_radius)
        for coord, ri in zip(own_coords, own_radii)
    ]
    near_residues = {id(atom.get_parent()) for atom, nbrs in zip(own_atoms, other_neighbours) if nbrs}

    area = 0.0
    interface_residue_keys: set[tuple] = set()
    atom_sas = [0.0] * len(own_coords)
    atom_int_sas = [0.0] * len(own_coords)
    for i, coord in enumerate(own_coords):
        if id(own_atoms[i].get_parent()) not in near_residues:
            continue
        ri = float(own_radii[i])
        own_neighbours = own_tree.query_ball_point(coord, ri + own_max_radius)
        own_mask = _uncovered_code_points(coord, ri, code_points, own_coords, own_radii, own_neighbours, skip=i)
        if not own_mask.any():
            continue
        other_mask = _uncovered_code_points(coord, ri, code_points, other_coords, other_radii, other_neighbours[i])

        atom_sas[i] = float(np.sum(code_areas[own_mask]) * ri * ri)
        atom_int_sas[i] = float(np.sum(code_areas[own_mask & ~other_mask]) * ri * ri)
        area += atom_int_sas[i]
        if atom_int_sas[i] > 0.0:
            interface_residue_keys.add(_residue_fingerprint(own_atoms[i].get_parent()))
    return area, frozenset(interface_residue_keys), tuple(atom_sas), tuple(atom_int_sas)


def _pisa_interface_residues(residues1, residues2) -> tuple[list, list]:
    residues1 = list(residues1)
    residues2 = list(residues2)
    result = _pisa_interface_result(residues1, residues2)
    return (
        [residue for residue in residues1 if _residue_fingerprint(residue) in result.residue_keys1],
        [residue for residue in residues2 if _residue_fingerprint(residue) in result.residue_keys2],
    )


def buried_surface_area(
    residues1,
    residues2,
    probe_radius: float = PISA_PROBE_RADIUS,
    code_no: int = PISA_CODE_NO,
) -> float:
    """
    PISA-style interface area in Angstrom^2.

    This is a direct Python port of the area logic in ProSurf::calcInterface:
    each atom receives a spherical code, code segments accessible in the atom's
    own molecule but covered by the opposing molecule are summed, and PISA's
    reported interface area is (area_side_1 + area_side_2) / 2.
    """
    result = _pisa_interface_result(residues1, residues2, probe_radius, code_no)
    return result.area


__all__ = ["PISA_CODE_NO", "PISA_PROBE_RADIUS", "buried_surface_area"]
