"""
Interface biophysical scores aligned with CCP4 PISA (Krissinel & Henrick 2007)
and SCASA (Lawrence & Colman shape complementarity via Connolly surface).

- shape_complementarity: port of SCASA's SC implementation (CCP4 SC algorithm).
- hydrogen_bonds / salt_bridges / disulfide_bonds: PISA-style distance criteria
  over residue-specific donor/acceptor atom sets. No hydrogens are required;
  matches CCP4-PISA web-server output closely for AlphaFold-predicted complexes.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Iterable, List, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .connolly import (
    BURIED_FLAG,
    PROBE_RADIUS,
    get_radius,
    mds as _connolly_mds,
)

# ---------------------------------------------------------------------------
# PISA bond-detection constants (Krissinel & Henrick, J. Mol. Biol. 2007)
# ---------------------------------------------------------------------------

HB_MAX_DIST = 3.9   # Angstrom, donor-acceptor
HB_MIN_DIST = 2.5   # Angstrom
SB_MAX_DIST = 4.0   # Angstrom, charged-atom pair
SS_MAX_DIST = 2.5   # Angstrom, Cys SG-SG
PISA_PROBE_RADIUS = 1.4  # Angstrom, default solvent probe in pisa_prosurf.cpp
PISA_CODE_NO = 36        # default spherical code size in pisa_prosurf.cpp

# Residue-specific donor/acceptor atom names (heavy atoms only; hydrogens
# optional). Backbone N is donor and backbone O is acceptor for all standard
# amino acids - added automatically in _atom_is_donor / _atom_is_acceptor.
_SIDECHAIN_DONORS = {
    "ARG": {"NE", "NH1", "NH2"},
    "LYS": {"NZ"},
    "HIS": {"ND1", "NE2"},
    "ASN": {"ND2"},
    "GLN": {"NE2"},
    "TRP": {"NE1"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
    "CYS": {"SG"},
}
_SIDECHAIN_ACCEPTORS = {
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
    "ASN": {"OD1"},
    "GLN": {"OE1"},
    "HIS": {"ND1", "NE2"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
    "CYS": {"SG"},
    "MET": {"SD"},
}

_POS_CHARGED = {
    "ARG": {"NE", "NH1", "NH2"},
    "LYS": {"NZ"},
    "HIS": {"ND1", "NE2"},
}
_NEG_CHARGED = {
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
}

_STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}

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


def _atom_is_donor(resname: str, atom_name: str) -> bool:
    if resname in _STANDARD_AA and atom_name == "N" and resname != "PRO":
        return True
    return atom_name in _SIDECHAIN_DONORS.get(resname, ())


def _atom_is_acceptor(resname: str, atom_name: str) -> bool:
    if resname in _STANDARD_AA and atom_name == "O":
        return True
    return atom_name in _SIDECHAIN_ACCEPTORS.get(resname, ())


def _collect_atoms(residues: Iterable) -> Tuple[List, np.ndarray, List[str], List[str]]:
    atoms, coords, resnames, atom_names = [], [], [], []
    for r in residues:
        rn = r.get_resname().strip().upper()
        for a in r:
            atoms.append(a)
            coords.append(a.coord)
            resnames.append(rn)
            atom_names.append(a.id.strip().upper())
    arr = np.asarray(coords, dtype=float) if coords else np.empty((0, 3))
    return atoms, arr, resnames, atom_names


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
    element = _atom_element(atom)
    if element in _PISA_ELEMENT_RADII:
        return _PISA_ELEMENT_RADII[element]
    return get_radius(atom.get_parent().get_resname(), atom.id)


def _collect_surface_atoms(residues: Iterable) -> Tuple[np.ndarray, np.ndarray]:
    coords, radii = [], []
    for residue in residues:
        for atom in residue:
            if _atom_element(atom) == "H":
                continue
            coords.append(atom.coord)
            radii.append(_pisa_radius(atom))
    if not coords:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float)
    return np.asarray(coords, dtype=float), np.asarray(radii, dtype=float)


def _mround(value: float) -> int:
    return int(math.floor(value + 0.5))


@lru_cache(maxsize=8)
def _pisa_spherical_code(code_no: int = PISA_CODE_NO) -> Tuple[np.ndarray, np.ndarray]:
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
    coords1, radii1 = _collect_surface_atoms(residues1)
    coords2, radii2 = _collect_surface_atoms(residues2)
    if len(coords1) == 0 or len(coords2) == 0:
        return 0.0

    coords = np.vstack([coords1, coords2])
    radii = np.concatenate([radii1, radii2]) + float(probe_radius)
    sides = np.concatenate([
        np.ones(len(coords1), dtype=int),
        np.full(len(coords2), 2, dtype=int),
    ])
    code_points, code_areas = _pisa_spherical_code(code_no)

    tree = cKDTree(coords)
    max_r = float(np.max(radii))
    int_area1 = 0.0
    int_area2 = 0.0

    for i, coord in enumerate(coords):
        ri = float(radii[i])
        own_side = int(sides[i])
        own_mask = np.ones(len(code_points), dtype=bool)
        other_mask = np.ones(len(code_points), dtype=bool)

        for j in tree.query_ball_point(coord, ri + max_r):
            if j == i:
                continue
            rj = float(radii[j])
            delta = coord - coords[j]
            if float(np.dot(delta, delta)) > (ri + rj) ** 2:
                continue

            surface_vectors = delta + ri * code_points
            covered = np.einsum("ij,ij->i", surface_vectors, surface_vectors) <= (rj * rj + 0.00001)
            if int(sides[j]) == 1:
                if own_side == 1:
                    own_mask &= ~covered
                else:
                    other_mask &= ~covered
            else:
                if own_side == 2:
                    own_mask &= ~covered
                else:
                    other_mask &= ~covered

            if not own_mask.any() and not other_mask.any():
                break

        if not own_mask.any():
            continue

        atom_area = float(np.sum(code_areas[own_mask & ~other_mask]) * ri * ri)
        if own_side == 1:
            int_area1 += atom_area
        else:
            int_area2 += atom_area

    return float((int_area1 + int_area2) / 2.0)


# ---------------------------------------------------------------------------
# Shape complementarity (SCASA port)
# ---------------------------------------------------------------------------

def shape_complementarity(
    residues1,
    residues2,
    distance: float = 8.0,
    density: float = 15.0,
    weight: float = 0.5,
    trim_cutoff: float = 1.4,
) -> float:
    """
    CCP4-SC shape complementarity via Connolly molecular surface.

    Ported from SCASA (Lawrence & Colman, 1993; Connolly, 1983).
    Returns SC in [-1, 1]; 0 on failure.
    """
    _, coords1, rn1, an1 = _collect_atoms(residues1)
    _, coords2, rn2, an2 = _collect_atoms(residues2)
    if coords1.size == 0 or coords2.size == 0:
        return 0.0

    # Interface filter: keep atoms within `distance` of the other surface.
    t2 = cKDTree(coords2)
    t1 = cKDTree(coords1)
    mask1 = np.zeros(len(coords1), dtype=bool)
    mask2 = np.zeros(len(coords2), dtype=bool)
    for i, c in enumerate(coords1):
        if t2.query_ball_point(c, distance):
            mask1[i] = True
    for i, c in enumerate(coords2):
        if t1.query_ball_point(c, distance):
            mask2[i] = True

    c1 = coords1[mask1]; n1_rn = [rn1[i] for i in np.where(mask1)[0]]
    n1_an = [an1[i] for i in np.where(mask1)[0]]
    c2 = coords2[mask2]; n2_rn = [rn2[i] for i in np.where(mask2)[0]]
    n2_an = [an2[i] for i in np.where(mask2)[0]]
    if c1.size == 0 or c2.size == 0:
        return 0.0

    atoms = np.vstack([c1, c2])
    radii = np.array(
        [get_radius(r, a) for r, a in zip(n1_rn + n2_rn, n1_an + n2_an)],
        dtype=float,
    )
    mol = np.array([1] * len(c1) + [2] * len(c2), dtype=int)

    dots, normals, flags, dot_mol = _connolly_mds(
        PROBE_RADIUS, atoms, radii, mol, density=density
    )
    if len(dots) == 0:
        return 0.0

    buried = flags == BURIED_FLAG
    d1 = dots[(dot_mol == 1) & buried]; nA = normals[(dot_mol == 1) & buried]
    d2 = dots[(dot_mol == 2) & buried]; nB = normals[(dot_mol == 2) & buried]
    if len(d1) == 0 or len(d2) == 0:
        return 0.0

    # Distance-based trim (equivalent to CCP4 SC peripheral trim band).
    _, i2 = cKDTree(d2).query(d1)
    _, i1 = cKDTree(d1).query(d2)
    dist1 = np.linalg.norm(d1 - d2[i2], axis=1)
    dist2 = np.linalg.norm(d2 - d1[i1], axis=1)
    m1 = dist1 <= trim_cutoff
    m2 = dist2 <= trim_cutoff
    d1 = d1[m1]; nA = nA[m1]
    d2 = d2[m2]; nB = nB[m2]
    if len(d1) == 0 or len(d2) == 0:
        return 0.0

    _, i2 = cKDTree(d2).query(d1)
    _, i1 = cKDTree(d1).query(d2)
    dist1 = np.linalg.norm(d1 - d2[i2], axis=1)
    dist2 = np.linalg.norm(d2 - d1[i1], axis=1)
    dot1 = -(np.einsum("ij,ij->i", nA, nB[i2]))
    dot2 = -(np.einsum("ij,ij->i", nB, nA[i1]))

    if weight > 0:
        s1 = dot1 * np.exp(-dist1 ** 2 * weight)
        s2 = dot2 * np.exp(-dist2 ** 2 * weight)
    else:
        s1, s2 = dot1, dot2

    return float((np.median(s1) + np.median(s2)) / 2)


# ---------------------------------------------------------------------------
# PISA-style bond counters
# ---------------------------------------------------------------------------

def _select_coords(residues, predicate) -> np.ndarray:
    """Coords of atoms in `residues` matching predicate(resname, atom_name)."""
    out = []
    for r in residues:
        rn = r.get_resname().strip().upper()
        for a in r:
            if predicate(rn, a.id.strip().upper()):
                out.append(a.coord)
    return np.asarray(out, dtype=float) if out else np.empty((0, 3))


def _select_atom_coords(residues, predicate) -> Tuple[List, np.ndarray]:
    atoms = []
    coords = []
    for r in residues:
        rn = r.get_resname().strip().upper()
        for a in r:
            if predicate(rn, a.id.strip().upper()):
                atoms.append(a)
                coords.append(a.coord)
    arr = np.asarray(coords, dtype=float) if coords else np.empty((0, 3))
    return atoms, arr


def _count_pairs_within(pts_a: np.ndarray, pts_b: np.ndarray,
                        dmin: float, dmax: float) -> int:
    if len(pts_a) == 0 or len(pts_b) == 0:
        return 0
    tree = cKDTree(pts_b)
    cnt = 0
    for c in pts_a:
        for j in tree.query_ball_point(c, dmax):
            d = float(np.linalg.norm(c - pts_b[j]))
            if dmin <= d <= dmax:
                cnt += 1
    return cnt


def _add_pairs_within(
    atoms_a: List,
    pts_a: np.ndarray,
    atoms_b: List,
    pts_b: np.ndarray,
    dmin: float,
    dmax: float,
    pairs: set[Tuple[int, int]],
) -> None:
    if len(pts_a) == 0 or len(pts_b) == 0:
        return
    tree = cKDTree(pts_b)
    for i, coord in enumerate(pts_a):
        for j in tree.query_ball_point(coord, dmax):
            d = float(np.linalg.norm(coord - pts_b[j]))
            if dmin <= d <= dmax:
                a_id = id(atoms_a[i])
                b_id = id(atoms_b[j])
                pairs.add((a_id, b_id) if a_id < b_id else (b_id, a_id))


def hydrogen_bonds(residues1, residues2) -> int:
    """Count PISA-style inter-chain hydrogen bonds (heavy-atom D...A criterion)."""
    donors1, d1 = _select_atom_coords(residues1, _atom_is_donor)
    acceptors2, a2 = _select_atom_coords(residues2, _atom_is_acceptor)
    donors2, d2 = _select_atom_coords(residues2, _atom_is_donor)
    acceptors1, a1 = _select_atom_coords(residues1, _atom_is_acceptor)
    pairs: set[Tuple[int, int]] = set()
    _add_pairs_within(donors1, d1, acceptors2, a2, HB_MIN_DIST, HB_MAX_DIST, pairs)
    _add_pairs_within(donors2, d2, acceptors1, a1, HB_MIN_DIST, HB_MAX_DIST, pairs)
    return len(pairs)


def salt_bridges(residues1, residues2) -> int:
    """Count PISA-style inter-chain salt bridges."""
    pos = lambda rn, an: an in _POS_CHARGED.get(rn, ())
    neg = lambda rn, an: an in _NEG_CHARGED.get(rn, ())
    p1 = _select_coords(residues1, pos); n2 = _select_coords(residues2, neg)
    p2 = _select_coords(residues2, pos); n1 = _select_coords(residues1, neg)
    c = _count_pairs_within(p1, n2, 0.0, SB_MAX_DIST)
    c += _count_pairs_within(p2, n1, 0.0, SB_MAX_DIST)
    return int(c)


def disulfide_bonds(residues1, residues2) -> int:
    """Count inter-chain Cys SG-Cys SG disulfide bonds."""
    sg = lambda rn, an: rn == "CYS" and an == "SG"
    s1 = _select_coords(residues1, sg)
    s2 = _select_coords(residues2, sg)
    return _count_pairs_within(s1, s2, 0.0, SS_MAX_DIST)
