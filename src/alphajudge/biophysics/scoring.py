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
from .pisa_radii import PISA_STANDARD_AA_RADII
from .srs_chemistry import SRS_ATOM_HB_TYPES, SRS_NEIGHBOURS, SRS_STANDARD_AA

# ---------------------------------------------------------------------------
# PISA bond-detection constants (Krissinel & Henrick, J. Mol. Biol. 2007)
# ---------------------------------------------------------------------------

HB_MAX_DIST = 3.9   # Angstrom, ccp4srs::Chem::maxDAdist
HB_MIN_DIST = 2.0   # Angstrom, CCP4 SRS accepts very short H-bonds in clashes
SB_MAX_DIST = 4.0   # Angstrom, charged-atom pair
SB_MIN_DIST = 2.0   # Angstrom, suppress severe atom clashes not reported by PISA
SS_MAX_DIST = 2.3   # Angstrom, DSBondThresh in pisa_interface.cpp
PISA_PROBE_RADIUS = 1.4  # Angstrom, default solvent probe in pisa_prosurf.cpp
PISA_CODE_NO = 36        # default spherical code size in pisa_prosurf.cpp
HB_MAX_HA_DIST2 = 2.5 * 2.5
HB_MAX_COSINE = 0.0  # equivalent to all ccp4srs h-bond angle thresholds of 90 deg

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

def _srs_hb_type(resname: str, atom_name: str) -> str:
    return SRS_ATOM_HB_TYPES.get(resname, {}).get(atom_name, "N")


def _atom_is_donor(resname: str, atom_name: str) -> bool:
    return _srs_hb_type(resname, atom_name) in {"D", "B"}


def _atom_is_acceptor(resname: str, atom_name: str) -> bool:
    return _srs_hb_type(resname, atom_name) in {"A", "B"}


def _atom_is_hydrogen_candidate(resname: str, atom_name: str) -> bool:
    return _srs_hb_type(resname, atom_name) == "H"


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


_PISA_INTERFACE_CACHE: dict[tuple, tuple[float, frozenset[int], frozenset[int]]] = {}


def _interface_cache_key(residues1, residues2, probe_radius: float, code_no: int) -> tuple:
    return (
        tuple(id(residue) for residue in residues1),
        tuple(id(residue) for residue in residues2),
        float(probe_radius),
        int(code_no),
    )


def _pisa_interface_result(
    residues1,
    residues2,
    probe_radius: float = PISA_PROBE_RADIUS,
    code_no: int = PISA_CODE_NO,
) -> tuple[float, frozenset[int], frozenset[int]]:
    """
    PISA ProSurf interface area plus residue selections.

    ProSurf selects atoms with nonzero interface area into ``selHndInt1/2``.
    PISA later expands those atom selections to residues before calling
    ccp4srs::CalcHBonds.  Keeping the selected residue ids here lets area and
    bond scoring share the same source of truth.
    """
    residues1 = tuple(residues1)
    residues2 = tuple(residues2)
    key = _interface_cache_key(residues1, residues2, probe_radius, code_no)
    cached = _PISA_INTERFACE_CACHE.get(key)
    if cached is not None:
        return cached

    coords1, radii1, atom_residues1 = _collect_surface_atoms_with_residues(residues1)
    coords2, radii2, atom_residues2 = _collect_surface_atoms_with_residues(residues2)
    if len(coords1) == 0 or len(coords2) == 0:
        result = (0.0, frozenset(), frozenset())
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
    ) -> tuple[float, frozenset[int]]:
        area = 0.0
        interface_residue_ids: set[int] = set()
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

            atom_interface_area = float(np.sum(code_areas[own_mask & ~other_mask]) * ri * ri)
            area += atom_interface_area
            if atom_interface_area > 0.0:
                interface_residue_ids.add(id(own_atom_residues[i]))
        return area, frozenset(interface_residue_ids)

    int_area1, int_residue_ids1 = side_area(
        coords1, radii1, tree1, coords2, radii2, tree2, max_r2, atom_residues1
    )
    int_area2, int_residue_ids2 = side_area(
        coords2, radii2, tree2, coords1, radii1, tree1, max_r1, atom_residues2
    )

    result = (float((int_area1 + int_area2) / 2.0), int_residue_ids1, int_residue_ids2)
    if len(_PISA_INTERFACE_CACHE) > 32:
        _PISA_INTERFACE_CACHE.clear()
    _PISA_INTERFACE_CACHE[key] = result
    return result


def _pisa_interface_residues(residues1, residues2) -> tuple[list, list]:
    residues1 = list(residues1)
    residues2 = list(residues2)
    _, ids1, ids2 = _pisa_interface_result(residues1, residues2)
    return (
        [residue for residue in residues1 if id(residue) in ids1],
        [residue for residue in residues2 if id(residue) in ids2],
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
    area, _, _ = _pisa_interface_result(residues1, residues2, probe_radius, code_no)
    return area


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


def _cosine_at_vertex(vertex_atom, atom1, atom2) -> float:
    """Cosine of the atom1-vertex-atom2 angle, matching mmdb::Atom::GetCosine."""
    vec1 = atom1.coord - vertex_atom.coord
    vec2 = atom2.coord - vertex_atom.coord
    norm1 = float(np.linalg.norm(vec1))
    norm2 = float(np.linalg.norm(vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return -1.0
    return float(np.dot(vec1, vec2)) / (norm1 * norm2)


def _atom_by_name(residue) -> dict[str, object]:
    return {atom.id.strip().upper(): atom for atom in residue}


def _bonded_atoms(atom) -> list:
    residue = atom.get_parent()
    residue_atoms = _atom_by_name(residue)
    resname = residue.get_resname().strip().upper()
    atom_name = atom.id.strip().upper()
    return [
        residue_atoms[name]
        for name in SRS_NEIGHBOURS.get(resname, {}).get(atom_name, ())
        if name in residue_atoms
    ]


def _all_cosines_ok(vertex_atom, atom1, atoms2) -> bool:
    return all(
        _cosine_at_vertex(vertex_atom, atom1, atom2) <= HB_MAX_COSINE
        for atom2 in atoms2
    )


def _standard_chain_residues(chain) -> list:
    return [
        residue for residue in chain
        if residue.id[0] == " " and residue.get_resname().strip().upper() in SRS_STANDARD_AA
    ]


def _is_n_terminus(atom) -> bool:
    residue = atom.get_parent()
    if atom.id.strip().upper() != "N":
        return False
    residues = _standard_chain_residues(residue.get_parent())
    return bool(residues) and residues[0] is residue


def _is_c_terminus(atom) -> bool:
    residue = atom.get_parent()
    if atom.id.strip().upper() not in {"O", "OXT"}:
        return False
    residues = _standard_chain_residues(residue.get_parent())
    return bool(residues) and residues[-1] is residue


def _is_salt_bridge_pair(donor_atom, acceptor_atom) -> bool:
    if donor_atom.get_parent() is acceptor_atom.get_parent():
        return False
    if _atom_element(donor_atom) != "N" or _atom_element(acceptor_atom) != "O":
        return False

    donor_resname = donor_atom.get_parent().get_resname().strip().upper()
    donor_name = donor_atom.id.strip().upper()
    if donor_name == "N":
        donor_ok = _is_n_terminus(donor_atom)
    else:
        donor_ok = donor_resname in {"LYS", "ARG", "HIS"}
    if not donor_ok:
        return False

    acceptor_resname = acceptor_atom.get_parent().get_resname().strip().upper()
    acceptor_name = acceptor_atom.id.strip().upper()
    if acceptor_name in {"O", "OXT"}:
        return _is_c_terminus(acceptor_atom)
    return acceptor_resname in {"GLU", "ASP"}


def _hydrogen_bond_pairs_for_contact(donor_atom, acceptor_atom) -> list[tuple[object, object]]:
    """Return the atom pairs ccp4srs would report for one D-A contact."""
    acceptor_bonds = _bonded_atoms(acceptor_atom)
    if not acceptor_bonds:
        return []

    donor_bonds = _bonded_atoms(donor_atom)
    donor_hydrogens = [
        atom for atom in donor_bonds
        if (
            atom.occupancy > 0.0
            and _atom_is_hydrogen_candidate(
                atom.get_parent().get_resname().strip().upper(),
                atom.id.strip().upper(),
            )
        )
    ]

    if donor_hydrogens:
        pairs = []
        for hydrogen in donor_hydrogens:
            ha_dist2 = float(np.dot(hydrogen.coord - acceptor_atom.coord, hydrogen.coord - acceptor_atom.coord))
            if ha_dist2 >= HB_MAX_HA_DIST2:
                continue
            if _cosine_at_vertex(hydrogen, donor_atom, acceptor_atom) > HB_MAX_COSINE:
                continue
            if _all_cosines_ok(acceptor_atom, hydrogen, acceptor_bonds):
                pairs.append((hydrogen, acceptor_atom))
        return pairs

    if not donor_bonds:
        return []
    if not _all_cosines_ok(donor_atom, acceptor_atom, donor_bonds):
        return []
    if not _all_cosines_ok(acceptor_atom, donor_atom, acceptor_bonds):
        return []
    return [(donor_atom, acceptor_atom)]


def _donor_acceptor_contacts(residues1, residues2, max_dist: float):
    donors1, d1 = _select_atom_coords(residues1, _atom_is_donor)
    acceptors2, a2 = _select_atom_coords(residues2, _atom_is_acceptor)
    donors2, d2 = _select_atom_coords(residues2, _atom_is_donor)
    acceptors1, a1 = _select_atom_coords(residues1, _atom_is_acceptor)

    for donor_atoms, donor_coords, acceptor_atoms, acceptor_coords in (
        (donors1, d1, acceptors2, a2),
        (donors2, d2, acceptors1, a1),
    ):
        if len(donor_coords) == 0 or len(acceptor_coords) == 0:
            continue
        tree = cKDTree(acceptor_coords)
        for i, donor_coord in enumerate(donor_coords):
            for j in tree.query_ball_point(donor_coord, max_dist):
                dist = float(np.linalg.norm(donor_coord - acceptor_coords[j]))
                if HB_MIN_DIST <= dist <= max_dist:
                    yield donor_atoms[i], acceptor_atoms[j], dist


def _salt_bridge_pairs(residues1, residues2) -> set[Tuple[Tuple[str, str, str, str], Tuple[str, str, str, str]]]:
    pairs = set()
    for donor_atom, acceptor_atom, dist in _donor_acceptor_contacts(residues1, residues2, SB_MAX_DIST):
        if SB_MIN_DIST <= dist <= SB_MAX_DIST and _is_salt_bridge_pair(donor_atom, acceptor_atom):
            pairs.add(_atom_pair_key(donor_atom, acceptor_atom))
    return pairs


def hydrogen_bonds(residues1, residues2) -> int:
    """Count PISA-style inter-chain hydrogen bonds.

    CCP4 PISA delegates this to ccp4srs::CalcHBonds. In pure Python we mirror
    the observable SRS behavior with heavy-atom donor/acceptor chemistry,
    monomer-bond angular filters, and removal of pairs that PISA reports as
    salt bridges rather than hydrogen bonds.
    """
    residues1, residues2 = _pisa_interface_residues(residues1, residues2)
    salt_pairs = _salt_bridge_pairs(residues1, residues2)
    pairs: set[Tuple[Tuple[str, str, str, str], Tuple[str, str, str, str]]] = set()

    for donor_atom, acceptor_atom, dist in _donor_acceptor_contacts(residues1, residues2, HB_MAX_DIST):
        if not (HB_MIN_DIST <= dist <= HB_MAX_DIST):
            continue
        pair_key = _atom_pair_key(donor_atom, acceptor_atom)
        if pair_key in salt_pairs:
            continue
        for atom1, atom2 in _hydrogen_bond_pairs_for_contact(donor_atom, acceptor_atom):
            pairs.add(_atom_pair_key(atom1, atom2))
    return len(pairs)


def salt_bridges(residues1, residues2) -> int:
    """Count PISA-style inter-chain salt bridges."""
    residues1, residues2 = _pisa_interface_residues(residues1, residues2)
    return len(_salt_bridge_pairs(residues1, residues2))


def disulfide_bonds(residues1, residues2) -> int:
    """Count inter-chain Cys SG-Cys SG disulfide bonds."""
    sg = lambda rn, an: rn == "CYS" and an == "SG"
    s1 = _select_coords(residues1, sg)
    s2 = _select_coords(residues2, sg)
    return _count_pairs_within(s1, s2, 0.0, SS_MAX_DIST)
