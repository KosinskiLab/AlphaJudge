"""ccp4srs/PISA-like hydrogen-bond, salt-bridge, and disulfide counts."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .prosurf import _atom_element, _pisa_interface_residues
from .srs_chemistry import SRS_ATOM_HB_TYPES, SRS_NEIGHBOURS, SRS_STANDARD_AA

HB_MAX_DIST = 3.9   # Angstrom, ccp4srs::Chem::maxDAdist
HB_MIN_DIST = 2.0   # Angstrom, CCP4 SRS accepts very short H-bonds in clashes
SB_MAX_DIST = 4.0   # Angstrom, charged-atom pair
SB_MIN_DIST = 2.0   # Angstrom, suppress severe atom clashes not reported by PISA
SS_MAX_DIST = 2.3   # Angstrom, DSBondThresh in pisa_interface.cpp
HB_MAX_HA_DIST2 = 2.5 * 2.5
HB_MAX_COSINE = 0.0  # equivalent to all ccp4srs h-bond angle thresholds of 90 deg


def _srs_hb_type(resname: str, atom_name: str) -> str:
    return SRS_ATOM_HB_TYPES.get(resname, {}).get(atom_name, "N")


def _atom_is_donor(resname: str, atom_name: str) -> bool:
    return _srs_hb_type(resname, atom_name) in {"D", "B"}


def _atom_is_acceptor(resname: str, atom_name: str) -> bool:
    return _srs_hb_type(resname, atom_name) in {"A", "B"}


def _atom_is_hydrogen_candidate(resname: str, atom_name: str) -> bool:
    return _srs_hb_type(resname, atom_name) == "H"


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


def _count_pairs_within(pts_a: np.ndarray, pts_b: np.ndarray, dmin: float, dmax: float) -> int:
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
    return all(_cosine_at_vertex(vertex_atom, atom1, atom2) <= HB_MAX_COSINE for atom2 in atoms2)


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
            ha_delta = hydrogen.coord - acceptor_atom.coord
            ha_dist2 = float(np.dot(ha_delta, ha_delta))
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


def _salt_bridge_pairs(residues1, residues2) -> set[
    Tuple[Tuple[str, str, str, str], Tuple[str, str, str, str]]
]:
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


__all__ = ["disulfide_bonds", "hydrogen_bonds", "salt_bridges"]
