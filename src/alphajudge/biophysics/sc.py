"""SCASA/CCP4-SC shape-complementarity scoring."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree

from .connolly import BURIED_FLAG, PROBE_RADIUS, get_radius, mds as _connolly_mds


def _collect_atoms(residues: Iterable) -> tuple[list, np.ndarray, list[str], list[str]]:
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


def shape_complementarity(
    residues1,
    residues2,
    distance: float = 8.0,
    density: float = 15.0,
    weight: float = 0.5,
    trim_cutoff: float = 1.6,
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

    # SCASA filters side 1 against side 2, then side 2 against the filtered
    # side 1. This slightly asymmetric ordering matches its CLI/reference path.
    t2 = cKDTree(coords2)
    neighbours1 = t2.query_ball_point(coords1, distance)
    mask1 = np.fromiter((len(nbrs) > 0 for nbrs in neighbours1), dtype=bool, count=len(coords1))

    c1 = coords1[mask1]
    n1_rn = [rn1[i] for i in np.where(mask1)[0]]
    n1_an = [an1[i] for i in np.where(mask1)[0]]
    if c1.size == 0:
        return 0.0

    t1_filtered = cKDTree(c1)
    neighbours2 = t1_filtered.query_ball_point(coords2, distance)
    mask2 = np.fromiter((len(nbrs) > 0 for nbrs in neighbours2), dtype=bool, count=len(coords2))

    c2 = coords2[mask2]
    n2_rn = [rn2[i] for i in np.where(mask2)[0]]
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
    d1 = dots[(dot_mol == 1) & buried]
    nA = normals[(dot_mol == 1) & buried]
    d2 = dots[(dot_mol == 2) & buried]
    nB = normals[(dot_mol == 2) & buried]
    if len(d1) == 0 or len(d2) == 0:
        return 0.0

    # This is the SCASA/CCP4-compatible edge trim used by the frozen
    # references. Applying Connolly's same-surface trim directly removes too
    # many buried dots on AlphaFold interfaces.
    _, i2 = cKDTree(d2).query(d1)
    _, i1 = cKDTree(d1).query(d2)
    dist1 = np.linalg.norm(d1 - d2[i2], axis=1)
    dist2 = np.linalg.norm(d2 - d1[i1], axis=1)
    m1 = dist1 <= trim_cutoff
    m2 = dist2 <= trim_cutoff
    d1 = d1[m1]
    nA = nA[m1]
    d2 = d2[m2]
    nB = nB[m2]
    if len(d1) == 0 or len(d2) == 0:
        return 0.0

    _, i2 = cKDTree(d2).query(d1)
    _, i1 = cKDTree(d1).query(d2)
    dist1 = np.linalg.norm(d1 - d2[i2], axis=1)
    dist2 = np.linalg.norm(d2 - d1[i1], axis=1)
    dot1 = -(np.einsum("ij,ij->i", nA, nB[i2]))
    dot2 = -(np.einsum("ij,ij->i", nB, nA[i1]))

    if weight > 0:
        s1 = dot1 * np.exp(-(dist1**2) * weight)
        s2 = dot2 * np.exp(-(dist2**2) * weight)
    else:
        s1, s2 = dot1, dot2

    return float((np.median(s1) + np.median(s2)) / 2)


__all__ = ["shape_complementarity"]
