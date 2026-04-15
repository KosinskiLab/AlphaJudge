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
import os
import struct
from functools import lru_cache
from pathlib import Path
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

HB_MAX_DIST = 3.6   # Angstrom, calibrated against ccp4srs::CalcHBonds
HB_MIN_DIST = 2.0   # Angstrom, CCP4 SRS accepts very short H-bonds in clashes
SB_MAX_DIST = 4.0   # Angstrom, charged-atom pair
SB_MIN_DIST = 2.0   # Angstrom, suppress severe atom clashes not reported by PISA
SS_MAX_DIST = 2.3   # Angstrom, DSBondThresh in pisa_interface.cpp
PISA_PROBE_RADIUS = 1.4  # Angstrom, default solvent probe in pisa_prosurf.cpp
PISA_CODE_NO = 36        # default spherical code size in pisa_prosurf.cpp
HB_DONOR_ANGLE_MIN = 90.0
HB_ACCEPTOR_ANGLE_MIN = 90.0

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

_MOLREF_SEARCH_PATHS = (
    Path(os.environ["ALPHAJUDGE_PISA_MOLREF"]) if "ALPHAJUDGE_PISA_MOLREF" in os.environ else None,
    Path("/g/kosinski/dima/PycharmProjects/pisa/molref"),
    Path(__file__).resolve().parents[4] / "pisa" / "molref",
)

_STANDARD_BONDS = {
    "ALA": (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB")),
    "ARG": (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"),
            ("CG", "CD"), ("CD", "NE"), ("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2")),
    "ASN": (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"),
            ("CG", "OD1"), ("CG", "ND2")),
    "ASP": (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"),
            ("CG", "OD1"), ("CG", "OD2")),
    "CYS": (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "SG")),
    "GLN": (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"),
            ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2")),
    "GLU": (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"),
            ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")),
    "GLY": (("N", "CA"), ("CA", "C"), ("C", "O")),
    "HIS": (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"),
            ("CG", "ND1"), ("ND1", "CE1"), ("CE1", "NE2"), ("NE2", "CD2"),
            ("CD2", "CG")),
    "ILE": (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG1"),
            ("CG1", "CD1"), ("CB", "CG2")),
    "LEU": (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"),
            ("CG", "CD1"), ("CG", "CD2")),
    "LYS": (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"),
            ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")),
    "MET": (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"),
            ("CG", "SD"), ("SD", "CE")),
    "PHE": (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"),
            ("CG", "CD1"), ("CD1", "CE1"), ("CE1", "CZ"), ("CZ", "CE2"),
            ("CE2", "CD2"), ("CD2", "CG")),
    "PRO": (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"),
            ("CG", "CD"), ("CD", "N")),
    "SER": (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "OG")),
    "THR": (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "OG1"),
            ("CB", "CG2")),
    "TRP": (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"),
            ("CG", "CD1"), ("CD1", "NE1"), ("NE1", "CE2"), ("CE2", "CD2"),
            ("CD2", "CG"), ("CE2", "CZ2"), ("CZ2", "CH2"), ("CH2", "CZ3"),
            ("CZ3", "CE3"), ("CE3", "CD2")),
    "TYR": (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"),
            ("CG", "CD1"), ("CD1", "CE1"), ("CE1", "CZ"), ("CZ", "OH"),
            ("CZ", "CE2"), ("CE2", "CD2"), ("CD2", "CG")),
    "VAL": (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG1"),
            ("CB", "CG2")),
}


def _bond_neighbours() -> dict[str, dict[str, set[str]]]:
    neighbours: dict[str, dict[str, set[str]]] = {}
    for resname, bonds in _STANDARD_BONDS.items():
        residue_neighbours: dict[str, set[str]] = {}
        for atom1, atom2 in bonds:
            residue_neighbours.setdefault(atom1, set()).add(atom2)
            residue_neighbours.setdefault(atom2, set()).add(atom1)
        neighbours[resname] = residue_neighbours
    return neighbours


_STANDARD_NEIGHBOURS = _bond_neighbours()


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


def _clean_molref_string(raw: bytes) -> str:
    return raw.decode("ascii", errors="ignore").replace("\0", "").strip()


def _decode_mmdb_float(raw: bytes) -> float:
    """Decode MMDB's 5-byte UniBin float used in PISA molref.rdt."""
    # Constants from mmdb_mattype.cpp for UseDoubleFloat builds.
    base = 256.0
    power0 = 127
    power4 = 130
    powers = [0.0] * 256
    powers[power0] = 1.0
    for i in range(1, power0 + 1):
        powers[power0 + i] = powers[power0 + i - 1] * base
        powers[power0 - i] = powers[power0 - i + 1] / base

    data = list(raw)
    sign = bool(data[1] & 0x80)
    data[1] &= 0x7F
    value = float(data[1])
    for j in range(2, 5):
        value = value * base + float(data[j])
    value = (value / powers[power4]) * powers[data[0]]
    return -value if sign else value


def _find_molref_dir() -> Path | None:
    for candidate in _MOLREF_SEARCH_PATHS:
        if candidate and (candidate / "molref.idx").exists() and (candidate / "molref.rdt").exists():
            return candidate
    return None


@lru_cache(maxsize=1)
def _pisa_molref_radii() -> dict[Tuple[str, str], float]:
    """Read PISA's molref atom radii for standard residues when available."""
    molref_dir = _find_molref_dir()
    if molref_dir is None:
        return {}

    idx = (molref_dir / "molref.idx").read_bytes()
    data = (molref_dir / "molref.rdt").read_bytes()
    n_entries = struct.unpack_from("<i", idx, 0)[0]
    offsets: dict[str, int] = {}
    for i in range(n_entries):
        offset = 4 + i * 16
        residue_name = _clean_molref_string(idx[offset:offset + 4])
        data_offset = struct.unpack_from("<i", idx, offset + 12)[0]
        offsets[residue_name] = data_offset

    radii: dict[Tuple[str, str], float] = {}
    for residue_name in _STANDARD_AA:
        offset = offsets.get(residue_name)
        if offset is None:
            continue
        atom_count = struct.unpack_from("<i", data, offset + 16)[0]
        atom_offset = offset + 20
        for _ in range(atom_count):
            atom_name = _clean_molref_string(data[atom_offset:atom_offset + 5])
            radius = _decode_mmdb_float(data[atom_offset + 10:atom_offset + 15])
            radii[(residue_name, atom_name)] = radius
            atom_offset += 29
    return radii


def _pisa_radius(atom) -> float:
    """PISA-style VdW radius used by ProSurf for SAS calculations."""
    residue = atom.get_parent()
    resname = residue.get_resname().strip().upper()
    atom_name = atom.id.strip().upper()
    molref_radius = _pisa_molref_radii().get((resname, atom_name))
    if molref_radius is not None:
        return molref_radius

    element = _atom_element(atom)
    if element in _PISA_ELEMENT_RADII:
        return _PISA_ELEMENT_RADII[element]
    return get_radius(resname, atom_name)


def _collect_surface_atoms_with_residues(residues: Iterable) -> Tuple[np.ndarray, np.ndarray, List]:
    coords, radii, atom_residues = [], [], []
    for residue in residues:
        for atom in residue:
            if _atom_element(atom) == "H":
                continue
            coords.append(atom.coord)
            radii.append(_pisa_radius(atom))
            atom_residues.append(residue)
    if not coords:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float), []
    return np.asarray(coords, dtype=float), np.asarray(radii, dtype=float), atom_residues


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
    coords1, radii1, _ = _collect_surface_atoms_with_residues(residues1)
    coords2, radii2, _ = _collect_surface_atoms_with_residues(residues2)
    if len(coords1) == 0 or len(coords2) == 0:
        return 0.0

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
    ) -> float:
        area = 0.0
        own_max_radius = float(np.max(own_radii))
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
                covered = np.einsum("ij,ij->i", surface_vectors, surface_vectors) <= (rj * rj + 0.00001)
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
                covered = np.einsum("ij,ij->i", surface_vectors, surface_vectors) <= (rj * rj + 0.00001)
                other_mask &= ~covered
                if not other_mask.any():
                    break

            area += float(np.sum(code_areas[own_mask & ~other_mask]) * ri * ri)
        return area

    int_area1 = side_area(coords1, radii1, tree1, coords2, radii2, tree2, max_r2)
    int_area2 = side_area(coords2, radii2, tree2, coords1, radii1, tree1, max_r1)

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

    # SCASA mirrors CCP4 SC's peripheral trim by keeping buried dots whose
    # nearest opposing buried dot lies within the practical trim distance.
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


def _residue_seqid(residue) -> str:
    seqid = str(residue.id[1])
    insertion = residue.id[2].strip()
    return seqid + insertion if insertion else seqid


def _atom_pair_key(atom1, atom2) -> Tuple[Tuple[str, str, str, str], Tuple[str, str, str, str]]:
    def one(atom):
        residue = atom.get_parent()
        chain = residue.get_parent()
        return (
            chain.id,
            residue.get_resname().strip().upper(),
            _residue_seqid(residue),
            atom.id.strip().upper(),
        )

    key1 = one(atom1)
    key2 = one(atom2)
    return (key1, key2) if key1 <= key2 else (key2, key1)


def _angle_degrees(point1: np.ndarray, vertex: np.ndarray, point2: np.ndarray) -> float:
    vec1 = point1 - vertex
    vec2 = point2 - vertex
    norm1 = float(np.linalg.norm(vec1))
    norm2 = float(np.linalg.norm(vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 180.0
    cosine = float(np.dot(vec1, vec2)) / (norm1 * norm2)
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))


def _atom_by_name(residue) -> dict[str, object]:
    return {atom.id.strip().upper(): atom for atom in residue}


def _max_bond_angle(pivot_atom, target_atom) -> float:
    """Largest angle between a bonded heavy neighbour, pivot, and target."""
    residue = pivot_atom.get_parent()
    resname = residue.get_resname().strip().upper()
    atom_name = pivot_atom.id.strip().upper()
    neighbours = _STANDARD_NEIGHBOURS.get(resname, {}).get(atom_name, ())
    residue_atoms = _atom_by_name(residue)
    angles = [
        _angle_degrees(residue_atoms[name].coord, pivot_atom.coord, target_atom.coord)
        for name in neighbours
        if name in residue_atoms
    ]
    return max(angles, default=180.0)


def _salt_bridge_pairs(residues1, residues2) -> set[Tuple[Tuple[str, str, str, str], Tuple[str, str, str, str]]]:
    pos = lambda rn, an: an in _POS_CHARGED.get(rn, ())
    neg = lambda rn, an: an in _NEG_CHARGED.get(rn, ())
    pairs = set()
    for atoms_a, pts_a, atoms_b, pts_b in (
        (*_select_atom_coords(residues1, pos), *_select_atom_coords(residues2, neg)),
        (*_select_atom_coords(residues2, pos), *_select_atom_coords(residues1, neg)),
    ):
        if len(pts_a) == 0 or len(pts_b) == 0:
            continue
        tree = cKDTree(pts_b)
        for i, coord in enumerate(pts_a):
            for j in tree.query_ball_point(coord, SB_MAX_DIST):
                dist = float(np.linalg.norm(coord - pts_b[j]))
                if SB_MIN_DIST <= dist <= SB_MAX_DIST:
                    pairs.add(_atom_pair_key(atoms_a[i], atoms_b[j]))
    return pairs


def hydrogen_bonds(residues1, residues2) -> int:
    """Count PISA-style inter-chain hydrogen bonds.

    CCP4 PISA delegates this to ccp4srs::CalcHBonds. In pure Python we mirror
    the observable SRS behavior with heavy-atom donor/acceptor chemistry,
    monomer-bond angular filters, and removal of pairs that PISA reports as
    salt bridges rather than hydrogen bonds.
    """
    donors1, d1 = _select_atom_coords(residues1, _atom_is_donor)
    acceptors2, a2 = _select_atom_coords(residues2, _atom_is_acceptor)
    donors2, d2 = _select_atom_coords(residues2, _atom_is_donor)
    acceptors1, a1 = _select_atom_coords(residues1, _atom_is_acceptor)
    salt_pairs = _salt_bridge_pairs(residues1, residues2)
    pairs: set[Tuple[Tuple[str, str, str, str], Tuple[str, str, str, str]]] = set()

    for donor_atoms, donor_coords, acceptor_atoms, acceptor_coords in (
        (donors1, d1, acceptors2, a2),
        (donors2, d2, acceptors1, a1),
    ):
        if len(donor_coords) == 0 or len(acceptor_coords) == 0:
            continue
        tree = cKDTree(acceptor_coords)
        for i, donor_coord in enumerate(donor_coords):
            donor_atom = donor_atoms[i]
            for j in tree.query_ball_point(donor_coord, HB_MAX_DIST):
                acceptor_atom = acceptor_atoms[j]
                dist = float(np.linalg.norm(donor_coord - acceptor_coords[j]))
                if not (HB_MIN_DIST <= dist <= HB_MAX_DIST):
                    continue
                pair_key = _atom_pair_key(donor_atom, acceptor_atom)
                if pair_key in salt_pairs:
                    continue
                donor_angle = _max_bond_angle(donor_atom, acceptor_atom)
                acceptor_angle = _max_bond_angle(acceptor_atom, donor_atom)
                if donor_angle < HB_DONOR_ANGLE_MIN or acceptor_angle < HB_ACCEPTOR_ANGLE_MIN:
                    continue
                pairs.add(pair_key)
    return len(pairs)


def salt_bridges(residues1, residues2) -> int:
    """Count PISA-style inter-chain salt bridges."""
    return len(_salt_bridge_pairs(residues1, residues2))


def disulfide_bonds(residues1, residues2) -> int:
    """Count inter-chain Cys SG-Cys SG disulfide bonds."""
    sg = lambda rn, an: rn == "CYS" and an == "SG"
    s1 = _select_coords(residues1, sg)
    s2 = _select_coords(residues2, sg)
    return _count_pairs_within(s1, s2, 0.0, SS_MAX_DIST)
