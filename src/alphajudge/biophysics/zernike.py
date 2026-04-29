"""3D Zernike interface similarity on voxelized interface densities."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Iterable

import numpy as np
from pyzernike import ZernikeDescriptor
from scipy.spatial import cKDTree

DEFAULT_DISTANCE = 8.0
DEFAULT_GRID_SIZE = 32
DEFAULT_ORDER = 10
DEFAULT_SIGMA = 1.5
DEFAULT_PADDING = 2.0


def _collect_coords(residues: Iterable) -> np.ndarray:
    coords = [atom.coord for residue in residues for atom in residue]
    return np.asarray(coords, dtype=float) if coords else np.empty((0, 3), dtype=float)


def _filter_interface_coords(
    coords1: np.ndarray,
    coords2: np.ndarray,
    distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    if coords1.size == 0 or coords2.size == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float)

    tree2 = cKDTree(coords2)
    neighbours1 = tree2.query_ball_point(coords1, distance)
    mask1 = np.fromiter((len(nbrs) > 0 for nbrs in neighbours1), dtype=bool, count=len(coords1))
    side1 = coords1[mask1]
    if side1.size == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float)

    tree1 = cKDTree(side1)
    neighbours2 = tree1.query_ball_point(coords2, distance)
    mask2 = np.fromiter((len(nbrs) > 0 for nbrs in neighbours2), dtype=bool, count=len(coords2))
    side2 = coords2[mask2]
    if side2.size == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float)

    return side1, side2


@lru_cache(maxsize=None)
def _unit_ball_mask(grid_size: int) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    return (xx * xx + yy * yy + zz * zz) <= 1.0


def _density_grid(
    coords: np.ndarray,
    centroid: np.ndarray,
    radius: float,
    grid_size: int,
    sigma: float,
) -> np.ndarray:
    volume = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    if coords.size == 0:
        return volume

    spacing = (2.0 * radius) / max(grid_size - 1, 1)
    sigma_vox = max(float(sigma) / max(spacing, 1e-6), 0.5)
    support = max(1, int(math.ceil(3.0 * sigma_vox)))

    normalized = np.clip((coords - centroid) / radius, -1.0, 1.0)
    positions = (normalized + 1.0) * 0.5 * (grid_size - 1)

    for pos in positions:
        center = np.rint(pos).astype(int)
        x0 = max(0, center[0] - support)
        x1 = min(grid_size - 1, center[0] + support)
        y0 = max(0, center[1] - support)
        y1 = min(grid_size - 1, center[1] + support)
        z0 = max(0, center[2] - support)
        z1 = min(grid_size - 1, center[2] + support)

        xs = np.arange(x0, x1 + 1, dtype=np.float32) - np.float32(pos[0])
        ys = np.arange(y0, y1 + 1, dtype=np.float32) - np.float32(pos[1])
        zs = np.arange(z0, z1 + 1, dtype=np.float32) - np.float32(pos[2])
        dist2 = xs[:, None, None] ** 2 + ys[None, :, None] ** 2 + zs[None, None, :] ** 2
        blob = np.exp(-0.5 * dist2 / (sigma_vox * sigma_vox)).astype(np.float32)
        volume[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1] += blob

    volume[~_unit_ball_mask(grid_size)] = 0.0
    return volume


def zernike_descriptors(
    residues1,
    residues2,
    distance: float = DEFAULT_DISTANCE,
    grid_size: int = DEFAULT_GRID_SIZE,
    order: int = DEFAULT_ORDER,
    sigma: float = DEFAULT_SIGMA,
    padding: float = DEFAULT_PADDING,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build per-side 3D Zernike descriptors for the contacting atom clouds.

    The two sides are centered and scaled together so their descriptors are
    comparable under the same unit-ball normalization.
    """
    coords1 = _collect_coords(residues1)
    coords2 = _collect_coords(residues2)
    side1, side2 = _filter_interface_coords(coords1, coords2, float(distance))
    if side1.size == 0 or side2.size == 0:
        raise ValueError("No contacting interface atoms found for Zernike descriptor.")

    combined = np.vstack([side1, side2])
    centroid = np.mean(combined, axis=0)
    radius = float(np.max(np.linalg.norm(combined - centroid, axis=1)) + float(padding))
    if not np.isfinite(radius) or radius <= 0.0:
        radius = 1.0

    grid1 = _density_grid(side1, centroid, radius, int(grid_size), float(sigma))
    grid2 = _density_grid(side2, centroid, radius, int(grid_size), float(sigma))

    desc1 = ZernikeDescriptor.fit(data=grid1, order=int(order)).get_coefficients()
    desc2 = ZernikeDescriptor.fit(data=grid2, order=int(order)).get_coefficients()
    return np.asarray(desc1, dtype=float), np.asarray(desc2, dtype=float)


def zernike_shape_complementarity(
    residues1,
    residues2,
    distance: float = DEFAULT_DISTANCE,
    grid_size: int = DEFAULT_GRID_SIZE,
    order: int = DEFAULT_ORDER,
    sigma: float = DEFAULT_SIGMA,
    padding: float = DEFAULT_PADDING,
) -> float:
    """
    Compare per-side 3D Zernike descriptors with cosine similarity.

    Returns 0.0 when no interface can be voxelized robustly.
    """
    try:
        desc1, desc2 = zernike_descriptors(
            residues1,
            residues2,
            distance=distance,
            grid_size=grid_size,
            order=order,
            sigma=sigma,
            padding=padding,
        )
    except Exception:
        return 0.0

    norm1 = float(np.linalg.norm(desc1))
    norm2 = float(np.linalg.norm(desc2))
    if norm1 <= 0.0 or norm2 <= 0.0:
        return 0.0

    similarity = float(np.dot(desc1, desc2) / (norm1 * norm2))
    return float(np.clip(similarity, -1.0, 1.0))


__all__ = [
    "DEFAULT_DISTANCE",
    "DEFAULT_GRID_SIZE",
    "DEFAULT_ORDER",
    "DEFAULT_PADDING",
    "DEFAULT_SIGMA",
    "zernike_descriptors",
    "zernike_shape_complementarity",
]
