"""SCASA/CCP4-SC shape-complementarity scoring."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.spatial import cKDTree

from .connolly import BURIED_FLAG, PROBE_RADIUS, get_radius, mds as _connolly_mds


def _collect_atoms(residues: Iterable) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates and CCP4 SC radii of every atom in ``residues``."""
    coords, radii = [], []
    for r in residues:
        rn = r.get_resname().strip().upper()
        for a in r:
            coords.append(a.coord)
            radii.append(get_radius(rn, a.id.strip().upper()))
    if not coords:
        return np.empty((0, 3)), np.empty(0)
    return np.asarray(coords, dtype=float), np.asarray(radii, dtype=float)


def _has_neighbour(points: np.ndarray, others: np.ndarray, distance: float) -> np.ndarray:
    return cKDTree(others).query_ball_point(points, distance, return_length=True) > 0


def _nearest(points: np.ndarray, others: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of, and distance to, each point's nearest neighbour in ``others``."""
    _, idx = cKDTree(others).query(points)
    return idx, np.linalg.norm(points - others[idx], axis=1)


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
    coords1, radii1 = _collect_atoms(residues1)
    coords2, radii2 = _collect_atoms(residues2)
    if coords1.size == 0 or coords2.size == 0:
        return 0.0

    # SCASA filters side 1 against side 2, then side 2 against the filtered
    # side 1. This slightly asymmetric ordering matches its CLI/reference path.
    mask1 = _has_neighbour(coords1, coords2, distance)
    c1 = coords1[mask1]
    if c1.size == 0:
        return 0.0
    mask2 = _has_neighbour(coords2, c1, distance)
    c2 = coords2[mask2]
    if c2.size == 0:
        return 0.0

    atoms = np.vstack([c1, c2])
    radii = np.concatenate([radii1[mask1], radii2[mask2]])
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
    _, dist1 = _nearest(d1, d2)
    _, dist2 = _nearest(d2, d1)
    m1 = dist1 <= trim_cutoff
    m2 = dist2 <= trim_cutoff
    d1, nA = d1[m1], nA[m1]
    d2, nB = d2[m2], nB[m2]
    if len(d1) == 0 or len(d2) == 0:
        return 0.0

    i2, dist1 = _nearest(d1, d2)
    i1, dist2 = _nearest(d2, d1)
    dot1 = -(np.einsum("ij,ij->i", nA, nB[i2]))
    dot2 = -(np.einsum("ij,ij->i", nB, nA[i1]))

    if weight > 0:
        s1 = dot1 * np.exp(-(dist1**2) * weight)
        s2 = dot2 * np.exp(-(dist2**2) * weight)
    else:
        s1, s2 = dot1, dot2

    return float((np.median(s1) + np.median(s2)) / 2)


__all__ = ["shape_complementarity"]
