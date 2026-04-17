"""
Connolly molecular dot surface (MDS) — Python translation of sc.f mds subroutine.

Translated from Fortran by Michael Connolly (Copyright 1986 Michael L. Connolly),
as distributed in the CCP4 SC program (Copyright 1999-2004 Michael Lawrence).

Generates three types of surface dots for each interface atom:
    1. Convex   — contact surface on each atom's VdW shell (outward normal)
    2. Toroidal — probe rolling between two atoms (inward normal from probe centre)
    3. Concave  — re-entrant patch where probe nestles between three atoms

All normals point away from the protein bulk, matching CCP4 SC convention.

Reference:
    Connolly, M.L. (1983) Science 221:709-713
    Lawrence, M.C. & Colman, P.M. (1993) J. Mol. Biol. 234:946-950
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

# CCP4 SC defaults (from defaults.fh)
PROBE_RADIUS = 1.7
DOT_DENSITY = 15.0
BURIED_FLAG = 1
ACCESSIBLE_FLAG = 0

# ---------------------------------------------------------------------------
# Radii table — exact values from CCP4 sc_radii.lib
#
# Format mirrors the Fortran match() logic:
#   - residue "***" = wildcard, matches any residue
#   - atom name ending in "*" = prefix match (e.g. "H*" matches "HA", "HB", etc.)
#   - later entries overwrite earlier wildcard matches (priority scheme)
#
# Stored as list of (residue, atom_pattern, radius) in file order so that
# specific entries correctly override wildcards.
# ---------------------------------------------------------------------------

_SC_RADII_TABLE = [
    ("***", "H",    0.50),
    ("***", "H*",   0.50),
    ("***", "H**",  0.50),
    ("***", "H***", 0.50),
    ("***", "CA",   1.85),
    ("***", "C",    1.80),
    ("***", "O",    1.60),
    ("***", "N",    1.65),
    ("***", "CB",   1.90),
    ("***", "OT*",  1.60),
    ("***", "OXT",  1.60),
    ("***", "S*",   1.90),
    ("***", "P",    1.80),
    ("ALA", "CB",   1.95),
    ("ARG", "NH*",  1.70),
    ("ARG", "CZ",   1.80),
    ("ARG", "NE",   1.65),
    ("ARG", "CD",   1.90),
    ("ARG", "CG",   1.90),
    ("ASN", "ND2",  1.70),
    ("ASN", "OD1",  1.60),
    ("ASN", "CG",   1.80),
    ("ASP", "OD*",  1.60),
    ("ASP", "CG",   1.80),
    ("GLN", "NE2",  1.70),
    ("GLN", "OE1",  1.60),
    ("GLN", "CD",   1.80),
    ("GLN", "CG",   1.90),
    ("GLU", "OE*",  1.60),
    ("GLU", "CD",   1.80),
    ("GLU", "CG",   1.90),
    ("GLY", "CA",   1.90),
    ("HIS", "CD2",  1.90),
    ("HIS", "NE2",  1.65),
    ("HIS", "CE1",  1.90),
    ("HIS", "ND1",  1.65),
    ("HIS", "CG",   1.80),
    ("HOH", "O**",  1.70),
    ("ILE", "CD1",  1.95),
    ("ILE", "CG1",  1.90),
    ("ILE", "CB",   1.85),
    ("ILE", "CG2",  1.95),
    ("LEU", "CD*",  1.95),
    ("LEU", "CG",   1.85),
    ("LYS", "NZ",   1.75),
    ("LYS", "CE",   1.90),
    ("LYS", "CD",   1.90),
    ("LYS", "CG",   1.90),
    ("MET", "CE",   1.95),
    ("MET", "CG",   1.90),
    ("PHE", "CD*",  1.90),
    ("PHE", "CE*",  1.90),
    ("PHE", "CZ",   1.90),
    ("PHE", "CG",   1.80),
    ("PRO", "CD",   1.90),
    ("PRO", "CG",   1.90),
    ("SER", "OG",   1.70),
    ("SUL", "S",    1.90),
    ("SUL", "O***", 1.65),
    ("THR", "CG2",  1.95),
    ("THR", "OG1",  1.70),
    ("THR", "CB",   1.85),
    ("TRP", "CE2",  1.80),
    ("TRP", "CE3",  1.90),
    ("TRP", "CD1",  1.90),
    ("TRP", "CD2",  1.80),
    ("TRP", "CZ*",  1.90),
    ("TRP", "CH2",  1.90),
    ("TRP", "NE1",  1.65),
    ("TRP", "CG",   1.80),
    ("TYR", "OH",   1.70),
    ("TYR", "CD*",  1.90),
    ("TYR", "CE*",  1.90),
    ("TYR", "CZ",   1.80),
    ("TYR", "CG",   1.80),
    ("VAL", "CG*",  1.95),
    ("VAL", "CB",   1.85),
    ("WAT", "O",    1.70),
    ("WAT", "O*",   1.70),
]


def _sc_match_atom(atom_name: str, pattern: str) -> bool:
    """
    Match atom_name against pattern, where trailing '*' is a prefix wildcard.
    Mirrors the Fortran match() function.
    """
    if pattern == "*":
        return True
    star = pattern.find("*")
    if star == -1:
        return atom_name == pattern
    return atom_name.startswith(pattern[:star])


@lru_cache(maxsize=None)
def get_radius(residue_name: str, atom_name: str) -> float:
    """
    Look up the CCP4 sc_radii radius for a (residue, atom) pair. Later entries
    in `_SC_RADII_TABLE` overwrite earlier ones, matching the Fortran priority.
    """
    res = residue_name.strip().upper()
    atom = atom_name.strip().upper()

    radius = 1.80
    for res_pat, atom_pat, r in _SC_RADII_TABLE:
        if (res_pat == "***" or res_pat.upper() == res) and _sc_match_atom(atom, atom_pat.upper()):
            radius = r
    return radius


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def _dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # np.cross has high dispatch overhead for the many scalar 3-vector calls in
    # the Connolly loops. Keep this helper explicit and local to the 3D case.
    return np.array(
        (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ),
        dtype=float,
    )


def _dis2(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.dot(d, d))


def _dis(a: np.ndarray, b: np.ndarray) -> float:
    return math.sqrt(max(0.0, _dis2(a, b)))


def _norm(a: np.ndarray) -> np.ndarray:
    n = math.sqrt(float(np.dot(a, a)))
    if n <= 0.0:
        raise ValueError("zero vector in _norm")
    return a / n


def _disptl(cen: np.ndarray, axis: np.ndarray, pnt: np.ndarray) -> float:
    vec = pnt - cen
    dt = _dot(vec, axis)
    return math.sqrt(max(0.0, float(np.dot(vec, vec)) - dt * dt))


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _empty_points() -> np.ndarray:
    return np.empty((0, 3), dtype=float)


def _subdiv(
    cen: np.ndarray,
    rad: float,
    x: np.ndarray,
    y: np.ndarray,
    angle: float,
    density: float,
) -> tuple[np.ndarray, float]:
    """
    Sample points along an arc in the plane spanned by orthonormal x/y.

    Returns
    -------
    points : (n, 3) ndarray
    spacing : float
        Arc length divided by number of returned points.

    Notes
    -----
    This preserves the original Fortran/Python sampling phase offset:
    points are placed at delta/2, 3*delta/2, 5*delta/2, ...
    """
    if density <= 0.0 or rad <= 0.0 or angle <= 0.0:
        return _empty_points(), 0.0

    delta = 1.0 / (math.sqrt(density) * rad)
    n_points = int(math.floor(angle / delta + 0.5))
    if n_points <= 0:
        return _empty_points(), 0.0

    angles = delta * (0.5 + np.arange(n_points, dtype=float))
    cos_a = np.cos(angles)[:, None]
    sin_a = np.sin(angles)[:, None]
    points = cen + rad * (cos_a * x + sin_a * y)
    return points, (rad * angle / n_points)


def _subarc(
    cen: np.ndarray,
    rad: float,
    axis: np.ndarray,
    density: float,
    x: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, float]:
    y = _cross(axis, x)
    angle = math.atan2(_dot(v, y), _dot(v, x))
    if angle < 0.0:
        angle += 2.0 * math.pi
    return _subdiv(cen, rad, x, y, angle, density)


def _axis_seed(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    v1 = np.array(
        [
            axis[1] ** 2 + axis[2] ** 2,
            axis[0] ** 2 + axis[2] ** 2,
            axis[0] ** 2 + axis[1] ** 2,
        ],
        dtype=float,
    )
    n = math.sqrt(float(np.dot(v1, v1)))
    if n > 0.0:
        v1 /= n
    if abs(_dot(v1, axis)) > 0.99:
        v1 = np.array([1.0, 0.0, 0.0], dtype=float)
    return v1


def _basis_from_axis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(axis, dtype=float)
    v1 = _axis_seed(axis)
    v2 = _norm(_cross(axis, v1))
    x = _norm(_cross(axis, v2))
    y = _cross(axis, x)
    return x, y


def _equatorial_vector(axis: np.ndarray) -> np.ndarray:
    return _norm(_cross(axis, _axis_seed(axis)))


def _subcir(
    cen: np.ndarray,
    rad: float,
    axis: np.ndarray,
    density: float,
) -> tuple[np.ndarray, float]:
    x, y = _basis_from_axis(axis)
    return _subdiv(cen, rad, x, y, 2.0 * math.pi, density)


# ---------------------------------------------------------------------------
# Occlusion / burial helpers
# ---------------------------------------------------------------------------

def _points_outside_spheres(
    points: np.ndarray,
    centers: np.ndarray,
    radii2: np.ndarray,
    chunk_size: int = 256,
    inclusive: bool = True,
) -> np.ndarray:
    """
    Return a mask selecting points that are outside all blocker spheres.

    Parameters
    ----------
    points : (n, 3) ndarray
    centers : (m, 3) ndarray
    radii2 : (m,) ndarray
        Squared blocker radii.
    inclusive : bool
        If True, block on distance^2 <= radius^2.
        If False, block on distance^2 < radius^2.
    """
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.zeros(0, dtype=bool)

    centers = np.asarray(centers, dtype=float)
    radii2 = np.asarray(radii2, dtype=float)

    if centers.size == 0:
        return np.ones(len(points), dtype=bool)

    center_norms = np.einsum("ij,ij->i", centers, centers)
    keep = np.ones(len(points), dtype=bool)

    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        chunk = points[start:stop]
        chunk_norms = np.einsum("ij,ij->i", chunk, chunk)
        sqdist = chunk_norms[:, None] + center_norms[None, :] - 2.0 * (chunk @ centers.T)
        if inclusive:
            blocked = sqdist <= radii2[None, :]
        else:
            blocked = sqdist < radii2[None, :]
        keep[start:stop] = ~blocked.any(axis=1)

    return keep


def _point_outside_spheres(
    point: np.ndarray,
    centers: np.ndarray,
    radii2: np.ndarray,
    inclusive: bool = True,
) -> bool:
    """Single-point fast path for `_points_outside_spheres`."""
    if centers.size == 0:
        return True
    diff = centers - point
    d2 = np.einsum("ij,ij->i", diff, diff)
    if inclusive:
        return bool(np.all(d2 > radii2))
    return bool(np.all(d2 >= radii2))


def _check_buried(
    pcen: np.ndarray,
    atom_idx: int,
    burco: list[np.ndarray],
    burrad2_with_probe: list[np.ndarray],
) -> bool:
    """
    A dot is buried (interface-facing) if its probe centre is within
    (burrad + rp) of any atom in the opposing molecule.
    """
    centers = burco[atom_idx]
    if centers.size == 0:
        return False

    diff = centers - pcen
    d2 = np.einsum("ij,ij->i", diff, diff)
    return bool(np.any(d2 <= burrad2_with_probe[atom_idx]))


def _check_buried_many(
    pcen: np.ndarray,
    atom_indices: np.ndarray,
    burco: list[np.ndarray],
    burrad2_with_probe: list[np.ndarray],
) -> np.ndarray:
    """
    Vectorized form of `_check_buried` for many probe centres.

    Probe centres are grouped by their supporting atom. That keeps the logic
    identical to the scalar helper, but avoids thousands of tiny Python calls
    in the convex-surface pass.
    """
    pcen = np.asarray(pcen, dtype=float)
    atom_indices = np.asarray(atom_indices, dtype=int)
    buried = np.zeros(len(pcen), dtype=bool)
    if len(pcen) == 0:
        return buried

    for atom_idx in np.unique(atom_indices):
        mask = atom_indices == atom_idx
        centers = burco[atom_idx]
        if centers.size == 0:
            continue
        radii2 = burrad2_with_probe[atom_idx]

        # Keep the same arithmetic as `_check_buried` (`diff dot diff`) rather
        # than the expanded distance formula used by `_points_outside_spheres`.
        # Some Connolly dots lie exactly on sphere boundaries, so tiny
        # round-off changes can flip ACCESSIBLE/BURIED labels.
        pts = pcen[mask]
        point_buried = np.zeros(len(pts), dtype=bool)
        for start in range(0, len(pts), 256):
            stop = min(start + 256, len(pts))
            diff = pts[start:stop, None, :] - centers[None, :, :]
            d2 = np.einsum("ijk,ijk->ij", diff, diff)
            point_buried[start:stop] = np.any(d2 <= radii2[None, :], axis=1)
        buried[mask] = point_buried
    return buried


# ---------------------------------------------------------------------------
# Surface append helper
# ---------------------------------------------------------------------------

def _append_points(
    out_dots: list[np.ndarray],
    out_normals: list[np.ndarray],
    out_flags: list[np.ndarray],
    out_mol: list[np.ndarray],
    dots: np.ndarray,
    normals: np.ndarray,
    buried_mask: np.ndarray,
    mol_label: int,
) -> None:
    if len(dots) == 0:
        return
    out_dots.append(np.asarray(dots, dtype=float))
    out_normals.append(np.asarray(normals, dtype=float))
    out_flags.append(np.where(buried_mask, BURIED_FLAG, ACCESSIBLE_FLAG).astype(int, copy=False))
    out_mol.append(np.full(len(dots), int(mol_label), dtype=int))


# ---------------------------------------------------------------------------
# Main Connolly surface generator
# ---------------------------------------------------------------------------

def mds(
    probe_radius: float,
    atoms: np.ndarray,
    radii: np.ndarray,
    mol: np.ndarray,
    density: float = DOT_DENSITY,
    residue_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a Connolly molecular dot surface.

    Parameters
    ----------
    probe_radius   : float, probe radius in Å (CCP4 default: 1.7)
    atoms          : np.array (N, 3)
    radii          : np.array (N,) VdW radii
    mol            : np.array (N,) int, molecule label (1 or 2)
    density        : float, dots per Å²
    residue_names  : list of str, residue names per atom (unused, kept for
                     API compatibility — radii are passed in directly)

    Returns
    -------
    dots     : np.array (M, 3)
    normals  : np.array (M, 3)  outward unit normals
    flags    : np.array (M,) int  BURIED_FLAG or ACCESSIBLE_FLAG
    dot_mol  : np.array (M,) int  molecule label for each dot
    """
    rp = float(probe_radius)
    atoms = np.asarray(atoms, dtype=float)
    radii = np.asarray(radii, dtype=float)
    mol = np.asarray(mol, dtype=int)

    natom = len(atoms)
    if natom == 0:
        return _empty_points(), _empty_points(), np.empty(0, dtype=int), np.empty(0, dtype=int)

    eradii = radii + rp
    atom_tree = cKDTree(atoms)
    max_radius = float(np.max(radii))

    out_dots: list[np.ndarray] = []
    out_normals: list[np.ndarray] = []
    out_flags: list[np.ndarray] = []
    out_mol: list[np.ndarray] = []

    burco: list[np.ndarray] = [_empty_points() for _ in range(natom)]
    burrad2_with_probe: list[np.ndarray] = [np.empty(0, dtype=float) for _ in range(natom)]
    access = np.zeros(natom, dtype=bool)
    probes: list[tuple[np.ndarray, np.ndarray, float, int, int, int]] = []
    same_neighbors: list[np.ndarray] = []
    candidate_neighbors: list[np.ndarray] = []

    for i1 in range(natom):
        reach = float(radii[i1] + max_radius + 2.0 * rp)
        candidates = np.asarray(atom_tree.query_ball_point(atoms[i1], reach), dtype=int)
        candidate_neighbors.append(candidates)

        same_mask = (candidates != i1) & (mol[candidates] == mol[i1])
        same = candidates[same_mask]
        if same.size:
            bridge2 = (radii[i1] + radii[same] + 2.0 * rp) ** 2
            diff = atoms[same] - atoms[i1]
            d2 = np.einsum("ij,ij->i", diff, diff)
            within = d2 < bridge2
            same = same[within]
            if same.size:
                same = same[np.argsort(d2[within])]
        same_neighbors.append(same)

        other_mask = (candidates != i1) & (mol[candidates] != mol[i1])
        other = candidates[other_mask]
        if other.size:
            bridge2 = (radii[i1] + radii[other] + 2.0 * rp) ** 2
            diff = atoms[other] - atoms[i1]
            d2 = np.einsum("ij,ij->i", diff, diff)
            other = other[d2 < bridge2]
        if other.size:
            burco[i1] = atoms[other].copy()
            burrad2_with_probe[i1] = (radii[other] + rp) ** 2

    # ── First loop: build probes and toroidal surface ────────────────────────
    for i1 in range(natom):
        ri = float(radii[i1])
        eri = float(eradii[i1])
        m1 = int(mol[i1])

        same_nbrs = same_neighbors[i1]
        nnbr = len(same_nbrs)
        if nnbr == 0:
            access[i1] = True

        for i2 in same_nbrs:
            if i2 <= i1:
                continue

            rj = float(radii[i2])
            erj = float(eradii[i2])
            dij = _dis(atoms[i1], atoms[i2])
            uij = (atoms[i2] - atoms[i1]) / dij
            asymm = (eri * eri - erj * erj) / dij
            between = abs(asymm) < dij
            tij = 0.5 * (atoms[i1] + atoms[i2]) + 0.5 * asymm * uij

            far = (eri + erj) ** 2 - dij ** 2
            if far <= 0.0:
                continue
            contain = dij ** 2 - (ri - rj) ** 2
            if contain <= 0.0:
                continue
            rij = 0.5 * math.sqrt(far) * math.sqrt(contain) / dij

            if nnbr <= 1:
                access[i1] = access[i2] = True
            else:
                for i3 in same_nbrs:
                    if i3 <= i2:
                        continue

                    rk = float(radii[i3])
                    erk = float(eradii[i3])

                    if _dis(atoms[i2], atoms[i3]) >= erj + erk:
                        continue
                    dik = _dis(atoms[i1], atoms[i3])
                    if dik >= eri + erk:
                        continue

                    uik = (atoms[i3] - atoms[i1]) / dik
                    dt = float(np.clip(_dot(uij, uik), -1.0, 1.0))
                    wijk = math.acos(dt)
                    if wijk <= 0.0:
                        continue

                    swijk = math.sin(wijk)
                    if swijk <= 0.0:
                        continue

                    uijk = _cross(uij, uik) / swijk
                    utb = _cross(uijk, uij)

                    asymm_ik = (eri * eri - erk * erk) / dik
                    tik = 0.5 * (atoms[i1] + atoms[i3]) + 0.5 * asymm_ik * uik
                    dt_p = _dot(uik, tik - tij)
                    bijk = tij + utb * (dt_p / swijk)

                    h2 = eri * eri - _dis2(bijk, atoms[i1])
                    if h2 <= 0.0:
                        dt2 = _dis2(tij, atoms[i3])
                        if dt2 < erk * erk - rij * rij:
                            break
                        continue

                    hijk = math.sqrt(h2)
                    blockers = np.array([i4 for i4 in same_nbrs if i4 != i2 and i4 != i3], dtype=int)
                    blocker_centers = atoms[blockers] if blockers.size else _empty_points()
                    blocker_radii2 = (eradii[blockers] ** 2) if blockers.size else np.empty(0, dtype=float)

                    for isign in (1, -1):
                        pijk = bijk + isign * hijk * uijk
                        if blockers.size and not _point_outside_spheres(
                            pijk,
                            blocker_centers,
                            blocker_radii2,
                        ):
                            continue

                        pi1_, pi2_, pi3_ = (i1, i2, i3) if isign > 0 else (i2, i1, i3)
                        probes.append((pijk.copy(), isign * uijk.copy(), hijk, pi1_, pi2_, pi3_))
                        access[i1] = access[i2] = access[i3] = True

            # Toroidal surface
            rci = rij * ri / eri
            rcj = rij * rj / erj
            rb = max(0.0, rij - rp)
            edens = ((rci + 2.0 * rb + rcj) / (4.0 * rij)) ** 2 * density

            circ_pnts, _ = _subcir(tij, rij, uij, edens)
            if len(circ_pnts) == 0:
                continue

            blockers = np.array([i4 for i4 in same_nbrs if i4 != i2], dtype=int)
            if blockers.size:
                keep = _points_outside_spheres(circ_pnts, atoms[blockers], eradii[blockers] ** 2)
                circ_pnts = circ_pnts[keep]
            if len(circ_pnts) == 0:
                continue

            access[i1] = access[i2] = True

            pi_vecs = (atoms[i1] - circ_pnts) / eri
            pj_vecs = (atoms[i2] - circ_pnts) / erj
            axes = np.cross(pi_vecs, pj_vecs)
            axis_norms = np.linalg.norm(axes, axis=1)
            valid_axes = axis_norms > 0.0
            if not np.any(valid_axes):
                continue

            circ_pnts = circ_pnts[valid_axes]
            pi_vecs = pi_vecs[valid_axes]
            pj_vecs = pj_vecs[valid_axes]
            axes = axes[valid_axes] / axis_norms[valid_axes][:, None]

            dtq = rp * rp - rij * rij
            pcusp = dtq > 0.0 and between
            if pcusp:
                dtq_sqrt = math.sqrt(dtq)
                pqi = (tij - dtq_sqrt * uij - circ_pnts) / rp
                pqj = (tij + dtq_sqrt * uij - circ_pnts) / rp
            else:
                sum_vecs = pi_vecs + pj_vecs
                sum_norms = np.linalg.norm(sum_vecs, axis=1)
                valid_sum = sum_norms > 0.0
                if not np.any(valid_sum):
                    continue

                circ_pnts = circ_pnts[valid_sum]
                pi_vecs = pi_vecs[valid_sum]
                pj_vecs = pj_vecs[valid_sum]
                axes = axes[valid_sum]
                pqi = sum_vecs[valid_sum] / sum_norms[valid_sum][:, None]
                pqj = pqi.copy()

            for half_index, (start_vecs, end_vecs, atom_i) in enumerate((
                (pi_vecs, pqi, i1),
                (pqj, pj_vecs, i2),
            )):
                # SCASA / CCP4 sc.f build burco[j] only when the outer atom
                # loop reaches j, so at this point burco[i2] is still empty
                # and the i2 half of the torus is always accessible in the
                # reference. Mirror that here so flag output matches sc.f.
                check_burial = half_index == 0
                for center, axis, start_v, end_v in zip(circ_pnts, axes, start_vecs, end_vecs):
                    dt = float(np.clip(_dot(start_v, end_v), -1.0, 1.0))
                    if abs(dt) >= 1.0:
                        continue

                    arc, _ = _subarc(center, rp, axis, density, start_v, end_v)
                    if len(arc) == 0:
                        continue

                    normals = (center - arc) / rp
                    is_buried = (
                        _check_buried(center, atom_i, burco, burrad2_with_probe)
                        if check_burial
                        else False
                    )
                    buried = np.full(len(arc), is_buried, dtype=bool)
                    _append_points(out_dots, out_normals, out_flags, out_mol, arc, normals, buried, m1)

    # ── Concave (re-entrant) surface ──────────────────────────────────────────
    if probes:
        probe_coords = np.asarray([p[0] for p in probes], dtype=float)
        probe_heights = np.asarray([p[2] for p in probes], dtype=float)
        low_mask = probe_heights < rp
        low_idx = np.where(low_mask)[0]
        low_coords = probe_coords[low_mask]
    else:
        low_idx = np.empty(0, dtype=int)
        low_coords = _empty_points()

    for iprb, (pijk, uijk, hijk, i1, i2, i3) in enumerate(probes):
        m1 = int(mol[i1])

        if hijk < rp and len(low_coords):
            near_mask = np.einsum("ij,ij->i", low_coords - pijk, low_coords - pijk) < (4.0 * rp * rp)
            near_mask &= low_idx != iprb
            near_low = low_coords[near_mask]
        else:
            near_low = _empty_points()

        vp = [_norm(atoms[i] - pijk) for i in (i1, i2, i3)]
        cv = [
            _norm(_cross(vp[0], vp[1])),
            _norm(_cross(vp[1], vp[2])),
            _norm(_cross(vp[2], vp[0])),
        ]

        dm = -1.0
        mm = 0
        for m_idx in range(3):
            dt = _dot(uijk, vp[m_idx])
            if dt > dm:
                dm = dt
                mm = m_idx

        south = -uijk
        try:
            axis = _norm(_cross(vp[mm], south))
        except ValueError:
            continue

        lats, _ = _subarc(np.zeros(3), rp, axis, density, vp[mm], south)
        for lat in lats:
            dt = _dot(lat, south)
            cen = dt * south
            rad2 = rp * rp - dt * dt
            if rad2 <= 0.0:
                continue

            circle_pnts, _ = _subcir(cen, math.sqrt(rad2), south, density)
            if len(circle_pnts) == 0:
                continue

            cv_dot = circle_pnts @ np.vstack(cv).T
            valid = ~(cv_dot >= 0.0).any(axis=1)
            circle_pnts = circle_pnts[valid]
            if len(circle_pnts) == 0:
                continue

            pts = pijk + circle_pnts
            if len(near_low):
                keep = _points_outside_spheres(
                    pts,
                    near_low,
                    np.full(len(near_low), rp * rp, dtype=float),
                    inclusive=False,
                )
                pts = pts[keep]
                circle_pnts = circle_pnts[keep]
            if len(pts) == 0:
                continue

            support_idx = np.empty(len(pts), dtype=int)
            support_atoms = (i1, i2, i3)
            for j, pt in enumerate(pts):
                mc = 0
                dmin = 2.0 * rp
                for atom_idx in support_atoms:
                    d = _dis(pt, atoms[atom_idx]) - float(radii[atom_idx])
                    if d < dmin:
                        dmin = d
                        mc = atom_idx
                support_idx[j] = mc

            normals = (pijk - pts) / rp
            buried = np.array(
                [_check_buried(pijk, atom_idx, burco, burrad2_with_probe) for atom_idx in support_idx],
                dtype=bool,
            )
            _append_points(out_dots, out_normals, out_flags, out_mol, pts, normals, buried, m1)

    # ── Convex (contact) surface ──────────────────────────────────────────────
    for i1 in range(natom):
        if not access[i1]:
            continue

        ri = float(radii[i1])
        eri = float(eradii[i1])
        m1 = int(mol[i1])
        same_nbrs = same_neighbors[i1]

        if len(same_nbrs) == 0:
            north = np.array([0.0, 0.0, 1.0], dtype=float)
            south = np.array([0.0, 0.0, -1.0], dtype=float)
            eqvec = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            i2 = int(same_nbrs[0])
            north = _norm(atoms[i1] - atoms[i2])
            eqvec = _equatorial_vector(north)
            vql = _cross(eqvec, north)

            rj = float(radii[i2])
            erj = float(eradii[i2])
            dij = _dis(atoms[i1], atoms[i2])
            uij = (atoms[i2] - atoms[i1]) / dij
            asymm = (eri * eri - erj * erj) / dij
            tij = 0.5 * (atoms[i1] + atoms[i2]) + 0.5 * asymm * uij

            far = (eri + erj) ** 2 - dij ** 2
            if far <= 0.0:
                continue
            contain = dij ** 2 - (ri - rj) ** 2
            if contain <= 0.0:
                continue
            rij = 0.5 * math.sqrt(far) * math.sqrt(contain) / dij
            south = (tij + rij * vql - atoms[i1]) / eri

        lats, _ = _subarc(np.zeros(3), ri, eqvec, density, north, south)
        for lat in lats:
            dt = _dot(lat, north)
            cen = atoms[i1] + dt * north
            rad2 = ri * ri - dt * dt
            if rad2 <= 0.0:
                continue

            circle_pnts, _ = _subcir(cen, math.sqrt(rad2), north, density)
            if len(circle_pnts) == 0:
                continue

            pcen = atoms[i1] + (eri / ri) * (circle_pnts - atoms[i1])

            blockers = same_nbrs[1:]
            if len(blockers):
                keep = _points_outside_spheres(pcen, atoms[blockers], eradii[blockers] ** 2)
                circle_pnts = circle_pnts[keep]
                pcen = pcen[keep]
            if len(circle_pnts) == 0:
                continue

            normals = circle_pnts - atoms[i1]
            normals /= np.linalg.norm(normals, axis=1)[:, None]

            buried = np.zeros(len(circle_pnts), dtype=bool)
            if burco[i1].size:
                buried[:] = _check_buried_many(
                    pcen,
                    np.full(len(circle_pnts), i1, dtype=int),
                    burco,
                    burrad2_with_probe,
                )

            _append_points(out_dots, out_normals, out_flags, out_mol, circle_pnts, normals, buried, m1)

    if not out_dots:
        return _empty_points(), _empty_points(), np.empty(0, dtype=int), np.empty(0, dtype=int)

    return (
        np.concatenate(out_dots, axis=0),
        np.concatenate(out_normals, axis=0),
        np.concatenate(out_flags, axis=0).astype(int, copy=False),
        np.concatenate(out_mol, axis=0).astype(int, copy=False),
    )


# ---------------------------------------------------------------------------
# Trim band (mirrors Fortran trim subroutine exactly)
# ---------------------------------------------------------------------------

def trim(dots: np.ndarray, flags: np.ndarray, band: float) -> np.ndarray:
    """
    Remove buried dots within `band` Å of any accessible dot on the same surface.
    Returns boolean mask — True = dot survives.
    """
    dots = np.asarray(dots, dtype=float)
    flags = np.asarray(flags)

    buried_mask = flags == BURIED_FLAG
    acc_mask = flags == ACCESSIBLE_FLAG

    keep = np.zeros(len(dots), dtype=bool)
    if not buried_mask.any():
        return keep
    if not acc_mask.any():
        keep[buried_mask] = True
        return keep

    acc_tree = cKDTree(dots[acc_mask])
    buried_idx = np.flatnonzero(buried_mask)
    neighbour_lists = acc_tree.query_ball_point(dots[buried_mask], band)
    keep[buried_idx] = np.fromiter(
        (len(nbrs) == 0 for nbrs in neighbour_lists),
        dtype=bool,
        count=len(neighbour_lists),
    )
    return keep
