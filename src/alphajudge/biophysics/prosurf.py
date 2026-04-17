"""PISA ProSurf-style solvent-accessible and buried interface area."""

from __future__ import annotations

import math
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


def _collect_surface_atoms_with_residues(
    residues: Iterable,
) -> tuple[np.ndarray, np.ndarray, list, list]:
    coords, radii, atom_residues, atoms = [], [], [], []
    for residue in residues:
        for atom in residue:
            if _atom_element(atom) == "H":
                continue
            coords.append(atom.coord)
            radii.append(_pisa_radius(atom))
            atom_residues.append(residue)
            atoms.append(atom)
    if not coords:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float), [], []
    return np.asarray(coords, dtype=float), np.asarray(radii, dtype=float), atom_residues, atoms


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
        if i == 1:
            points.append((0.0, 0.0, 1.0))
            areas.append(2.0 * math.pi * (1.0 - math.cos(dalpha / 2.0)))
        elif i == code_no:
            points.append((0.0, 0.0, -1.0))
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
    residue_ids1: frozenset[int]
    residue_ids2: frozenset[int]
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
    ccp4srs::CalcHBonds. Keeping the selected residue ids here lets area and
    bond scoring share the same source of truth.
    """
    residues1 = tuple(residues1)
    residues2 = tuple(residues2)
    key = _interface_cache_key(residues1, residues2, probe_radius, code_no)
    cached = _PISA_INTERFACE_CACHE.get(key)
    if cached is not None:
        return cached

    coords1, radii1, atom_residues1, atoms1 = _collect_surface_atoms_with_residues(residues1)
    coords2, radii2, atom_residues2, atoms2 = _collect_surface_atoms_with_residues(residues2)
    if len(coords1) == 0 or len(coords2) == 0:
        result = _PisaInterfaceResult(
            0.0,
            frozenset(),
            frozenset(),
            tuple(atoms1),
            tuple(),
            tuple(),
            tuple(atoms2),
            tuple(),
            tuple(),
        )
        _PISA_INTERFACE_CACHE[key] = result
        return result

    code_points, code_areas = _pisa_spherical_code(code_no)
    radii1 = radii1 + float(probe_radius)
    radii2 = radii2 + float(probe_radius)

    tree1 = cKDTree(coords1)
    tree2 = cKDTree(coords2)
    max_r1 = float(np.max(radii1))
    max_r2 = float(np.max(radii2))

    def side_area(
        own_coords: np.ndarray,
        own_radii: np.ndarray,
        own_tree: cKDTree,
        other_coords: np.ndarray,
        other_radii: np.ndarray,
        other_tree: cKDTree,
        other_max_radius: float,
        own_atom_residues: list,
    ) -> tuple[float, frozenset[int], tuple[float, ...], tuple[float, ...]]:
        area = 0.0
        interface_residue_ids: set[int] = set()
        own_max_radius = float(np.max(own_radii))
        atom_sas = [0.0] * len(own_coords)
        atom_int_sas = [0.0] * len(own_coords)
        for i, coord in enumerate(own_coords):
            ri = float(own_radii[i])
            other_neighbours = other_tree.query_ball_point(coord, ri + other_max_radius)
            if not other_neighbours:
                continue

            own_mask = np.ones(len(code_points), dtype=bool)
            other_mask = np.ones(len(code_points), dtype=bool)

            for j in own_tree.query_ball_point(coord, ri + own_max_radius):
                if j == i:
                    continue
                rj = float(own_radii[j])
                delta = coord - own_coords[j]
                if float(np.dot(delta, delta)) > (ri + rj) ** 2:
                    continue
                surface_vectors = delta + ri * code_points
                covered = (
                    np.einsum("ij,ij->i", surface_vectors, surface_vectors)
                    <= (rj * rj + 0.00001)
                )
                own_mask &= ~covered
                if not own_mask.any():
                    break

            if not own_mask.any():
                continue

            for j in other_neighbours:
                rj = float(other_radii[j])
                delta = coord - other_coords[j]
                if float(np.dot(delta, delta)) > (ri + rj) ** 2:
                    continue
                surface_vectors = delta + ri * code_points
                covered = (
                    np.einsum("ij,ij->i", surface_vectors, surface_vectors)
                    <= (rj * rj + 0.00001)
                )
                other_mask &= ~covered
                if not other_mask.any():
                    break

            atom_surface_area = float(np.sum(code_areas[own_mask]) * ri * ri)
            atom_interface_area = float(np.sum(code_areas[own_mask & ~other_mask]) * ri * ri)
            atom_sas[i] = atom_surface_area
            atom_int_sas[i] = atom_interface_area
            area += atom_interface_area
            if atom_interface_area > 0.0:
                interface_residue_ids.add(id(own_atom_residues[i]))
        return area, frozenset(interface_residue_ids), tuple(atom_sas), tuple(atom_int_sas)

    int_area1, int_residue_ids1, atom_sas1, atom_int_sas1 = side_area(
        coords1, radii1, tree1, coords2, radii2, tree2, max_r2, atom_residues1
    )
    int_area2, int_residue_ids2, atom_sas2, atom_int_sas2 = side_area(
        coords2, radii2, tree2, coords1, radii1, tree1, max_r1, atom_residues2
    )

    result = _PisaInterfaceResult(
        float((int_area1 + int_area2) / 2.0),
        int_residue_ids1,
        int_residue_ids2,
        tuple(atoms1),
        atom_sas1,
        atom_int_sas1,
        tuple(atoms2),
        atom_sas2,
        atom_int_sas2,
    )
    if len(_PISA_INTERFACE_CACHE) > 32:
        _PISA_INTERFACE_CACHE.clear()
    _PISA_INTERFACE_CACHE[key] = result
    return result


def _pisa_interface_residues(residues1, residues2) -> tuple[list, list]:
    residues1 = list(residues1)
    residues2 = list(residues2)
    result = _pisa_interface_result(residues1, residues2)
    return (
        [residue for residue in residues1 if id(residue) in result.residue_ids1],
        [residue for residue in residues2 if id(residue) in result.residue_ids2],
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
