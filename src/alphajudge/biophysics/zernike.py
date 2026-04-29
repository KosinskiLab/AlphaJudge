"""Low-pass 3D Zernike interface similarity on voxelized interface densities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
from pyzernike import ZernikeDescriptor
from scipy.spatial import cKDTree

from ..geometry import representative_atom
from .connolly import DOT_DENSITY, PROBE_RADIUS
from .sc import interface_surface_dots

ATOM_GAUSSIAN = "atom_gaussian"
RESIDUE_BEAD_GAUSSIAN = "residue_bead_gaussian"
SURFACE_BINARY = "surface_binary"
SURFACE_GAUSSIAN = "surface_gaussian"
SURFACE_PROXIMITY_GAUSSIAN = "surface_proximity_gaussian"
JOINT_RESIDUE_BEAD_GAUSSIAN = "joint_residue_bead_gaussian"
JOINT_SURFACE_GAUSSIAN = "joint_surface_gaussian"

HARD_CUTOFF_SCORE = "hard_cutoff"
GAUSSIAN_WEIGHTED_SCORE = "gaussian_weighted"
JOINT_LOW_ORDER_RATIO_SCORE = "joint_low_order_ratio"

JOINT_REPRESENTATIONS = {
    JOINT_RESIDUE_BEAD_GAUSSIAN,
    JOINT_SURFACE_GAUSSIAN,
}
GAUSSIAN_REPRESENTATIONS = {
    ATOM_GAUSSIAN,
    RESIDUE_BEAD_GAUSSIAN,
    SURFACE_GAUSSIAN,
    SURFACE_PROXIMITY_GAUSSIAN,
    JOINT_RESIDUE_BEAD_GAUSSIAN,
    JOINT_SURFACE_GAUSSIAN,
}
SURFACE_REPRESENTATIONS = {
    SURFACE_BINARY,
    SURFACE_GAUSSIAN,
    SURFACE_PROXIMITY_GAUSSIAN,
    JOINT_SURFACE_GAUSSIAN,
}
PER_SIDE_REPRESENTATIONS = {
    ATOM_GAUSSIAN,
    RESIDUE_BEAD_GAUSSIAN,
    SURFACE_BINARY,
    SURFACE_GAUSSIAN,
    SURFACE_PROXIMITY_GAUSSIAN,
}

DEFAULT_REPRESENTATION = ATOM_GAUSSIAN
DEFAULT_SCORE_MODE = HARD_CUTOFF_SCORE
DEFAULT_DISTANCE = 8.0
DEFAULT_GRID_SIZE = 32
DEFAULT_ORDER = 10
DEFAULT_SIGMA = 1.5
DEFAULT_PADDING = 2.0
# Surface Zernike is intentionally coarser than SC's Connolly-dot path so it
# smooths away local side-chain noise instead of preserving it.
DEFAULT_SURFACE_DENSITY = DOT_DENSITY / 3.0
DEFAULT_SURFACE_TRIM_CUTOFF = 1.6
DEFAULT_SURFACE_PROBE_RADIUS = PROBE_RADIUS
DEFAULT_PROXIMITY_LENGTH_SCALE = 2.5
DEFAULT_ORDER_DECAY_N0 = 4.0

_SHORT_SCORE_TAGS = {
    HARD_CUTOFF_SCORE: "hard",
    GAUSSIAN_WEIGHTED_SCORE: "weighted",
    JOINT_LOW_ORDER_RATIO_SCORE: "jointratio",
}


@dataclass(frozen=True, slots=True)
class WeightedPointCloud:
    points: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True, slots=True)
class ZernikeSpec:
    representation: str = DEFAULT_REPRESENTATION
    grid_size: int = DEFAULT_GRID_SIZE
    order: int = DEFAULT_ORDER
    sigma: float = DEFAULT_SIGMA
    padding: float = DEFAULT_PADDING
    distance: float = DEFAULT_DISTANCE
    surface_density: float = DEFAULT_SURFACE_DENSITY
    surface_trim_cutoff: float = DEFAULT_SURFACE_TRIM_CUTOFF
    surface_probe_radius: float = DEFAULT_SURFACE_PROBE_RADIUS
    proximity_length_scale: float = DEFAULT_PROXIMITY_LENGTH_SCALE
    score_mode: str = DEFAULT_SCORE_MODE
    fit_order: int | None = None
    order_decay_n0: float = DEFAULT_ORDER_DECAY_N0

    def candidate_id(self) -> str:
        parts = [self.representation, f"g{self.grid_size}", f"o{self.order}"]
        if self.representation in GAUSSIAN_REPRESENTATIONS:
            parts.append(f"s{_tag_float(self.sigma)}")
        if self.representation in SURFACE_REPRESENTATIONS:
            parts.append(f"d{_tag_float(self.surface_density)}")
            if not math.isclose(
                float(self.surface_probe_radius),
                float(DEFAULT_SURFACE_PROBE_RADIUS),
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                parts.append(f"pr{_tag_float(self.surface_probe_radius)}")
        if self.score_mode != HARD_CUTOFF_SCORE:
            parts.append(f"m{_SHORT_SCORE_TAGS[self.score_mode]}")
        if self.score_mode == GAUSSIAN_WEIGHTED_SCORE:
            parts.append(f"n{_tag_float(self.order_decay_n0)}")
        fit_to = fit_order_value(self)
        if fit_to != int(self.order):
            parts.append(f"f{fit_to}")
        return "__".join(parts)


def _tag_float(value: float) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def fit_order_value(spec_or_order: ZernikeSpec | int, fit_order: int | None = None) -> int:
    if isinstance(spec_or_order, ZernikeSpec):
        order = int(spec_or_order.order)
        fit_to = spec_or_order.fit_order
    else:
        order = int(spec_or_order)
        fit_to = fit_order
    if fit_to is None:
        return order
    return max(order, int(fit_to))


def zernike_source_representation(representation: str) -> str:
    if representation == JOINT_RESIDUE_BEAD_GAUSSIAN:
        return RESIDUE_BEAD_GAUSSIAN
    if representation == JOINT_SURFACE_GAUSSIAN:
        return SURFACE_GAUSSIAN
    return representation


def zernike_candidate_family(representation: str) -> str:
    return "joint_volume" if representation in JOINT_REPRESENTATIONS else "per_side"


def _collect_atom_coords(residues: Iterable) -> np.ndarray:
    coords = [atom.coord for residue in residues for atom in residue]
    return np.asarray(coords, dtype=float) if coords else np.empty((0, 3), dtype=float)


def _collect_residue_bead_coords(residues: Iterable) -> np.ndarray:
    coords = []
    for residue in residues:
        try:
            coords.append(representative_atom(residue).coord)
        except Exception:
            continue
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


def zernike_descriptor_prefix_length(order: int) -> int:
    """Return the number of invariant coefficients up to a given order."""
    order = int(order)
    if order < 0:
        raise ValueError("order must be non-negative")
    if order == 0:
        return 1
    half = order // 2
    if order % 2 == 0:
        return (half + 1) ** 2
    return (half + 1) * (half + 2)


@lru_cache(maxsize=None)
def zernike_descriptor_orders(order: int) -> np.ndarray:
    """Return the Zernike order associated with each invariant coefficient."""
    order = int(order)
    if order < 0:
        raise ValueError("order must be non-negative")
    out: list[int] = []
    prev = 0
    for n in range(order + 1):
        prefix = zernike_descriptor_prefix_length(n)
        out.extend([n] * (prefix - prev))
        prev = prefix
    return np.asarray(out, dtype=int)


def zernike_order_weights(order: int, order_decay_n0: float) -> np.ndarray:
    n0 = max(float(order_decay_n0), 1e-6)
    orders = zernike_descriptor_orders(int(order))
    return np.exp(-((orders / n0) ** 2)).astype(float)


def _shared_normalization(
    side1: WeightedPointCloud,
    side2: WeightedPointCloud,
    padding: float,
) -> tuple[np.ndarray, float]:
    if side1.points.size == 0 or side2.points.size == 0:
        raise ValueError("No interface points found for Zernike descriptor.")

    combined = np.vstack([side1.points, side2.points])
    centroid = np.mean(combined, axis=0)
    radius = float(np.max(np.linalg.norm(combined - centroid, axis=1)) + float(padding))
    if not np.isfinite(radius) or radius <= 0.0:
        radius = 1.0
    return centroid, radius


def _normalized_positions(
    points: np.ndarray,
    centroid: np.ndarray,
    radius: float,
    grid_size: int,
) -> np.ndarray:
    normalized = np.clip((points - centroid) / radius, -1.0, 1.0)
    return (normalized + 1.0) * 0.5 * (grid_size - 1)


def _binary_grid(
    cloud: WeightedPointCloud,
    centroid: np.ndarray,
    radius: float,
    grid_size: int,
) -> np.ndarray:
    volume = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    if cloud.points.size == 0:
        return volume

    positions = _normalized_positions(cloud.points, centroid, radius, grid_size)
    voxels = np.rint(positions).astype(int)
    voxels = np.clip(voxels, 0, grid_size - 1)
    for idx, weight in zip(voxels, cloud.weights.tolist()):
        if weight > 0.0:
            volume[idx[0], idx[1], idx[2]] = 1.0

    volume[~_unit_ball_mask(grid_size)] = 0.0
    return volume


def _density_grid(
    cloud: WeightedPointCloud,
    centroid: np.ndarray,
    radius: float,
    grid_size: int,
    sigma: float,
) -> np.ndarray:
    volume = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    if cloud.points.size == 0:
        return volume

    spacing = (2.0 * radius) / max(grid_size - 1, 1)
    sigma_vox = max(float(sigma) / max(spacing, 1e-6), 0.5)
    support = max(1, int(math.ceil(3.0 * sigma_vox)))
    positions = _normalized_positions(cloud.points, centroid, radius, grid_size)

    for pos, weight in zip(positions, cloud.weights.tolist()):
        if weight <= 0.0:
            continue
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
        volume[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1] += np.float32(weight) * blob

    volume[~_unit_ball_mask(grid_size)] = 0.0
    return volume


def _surface_point_clouds(
    residues1,
    residues2,
    *,
    distance: float,
    density: float,
    trim_cutoff: float,
    probe_radius: float,
    representation: str,
    proximity_length_scale: float,
) -> tuple[WeightedPointCloud, WeightedPointCloud]:
    d1, _, d2, _ = interface_surface_dots(
        residues1,
        residues2,
        distance=distance,
        density=density,
        trim_cutoff=trim_cutoff,
        probe_radius=probe_radius,
    )
    if d1.size == 0 or d2.size == 0:
        return (
            WeightedPointCloud(np.empty((0, 3), dtype=float), np.empty(0, dtype=float)),
            WeightedPointCloud(np.empty((0, 3), dtype=float), np.empty(0, dtype=float)),
        )

    if representation == SURFACE_PROXIMITY_GAUSSIAN:
        tree2 = cKDTree(d2)
        tree1 = cKDTree(d1)
        dist1, _ = tree2.query(d1)
        dist2, _ = tree1.query(d2)
        scale = max(float(proximity_length_scale), 1e-6)
        w1 = np.exp(-((dist1 / scale) ** 2))
        w2 = np.exp(-((dist2 / scale) ** 2))
    else:
        w1 = np.ones(len(d1), dtype=float)
        w2 = np.ones(len(d2), dtype=float)

    return WeightedPointCloud(d1, w1), WeightedPointCloud(d2, w2)


def zernike_point_clouds(
    residues1,
    residues2,
    *,
    representation: str = DEFAULT_REPRESENTATION,
    distance: float = DEFAULT_DISTANCE,
    surface_density: float = DEFAULT_SURFACE_DENSITY,
    surface_trim_cutoff: float = DEFAULT_SURFACE_TRIM_CUTOFF,
    surface_probe_radius: float = DEFAULT_SURFACE_PROBE_RADIUS,
    proximity_length_scale: float = DEFAULT_PROXIMITY_LENGTH_SCALE,
) -> tuple[WeightedPointCloud, WeightedPointCloud]:
    """Build weighted interface point clouds for a Zernike representation."""
    source_representation = zernike_source_representation(representation)

    if source_representation == ATOM_GAUSSIAN:
        coords1 = _collect_atom_coords(residues1)
        coords2 = _collect_atom_coords(residues2)
        side1, side2 = _filter_interface_coords(coords1, coords2, float(distance))
        return WeightedPointCloud(side1, np.ones(len(side1), dtype=float)), WeightedPointCloud(
            side2,
            np.ones(len(side2), dtype=float),
        )

    if source_representation == RESIDUE_BEAD_GAUSSIAN:
        coords1 = _collect_residue_bead_coords(residues1)
        coords2 = _collect_residue_bead_coords(residues2)
        side1, side2 = _filter_interface_coords(coords1, coords2, float(distance))
        return WeightedPointCloud(side1, np.ones(len(side1), dtype=float)), WeightedPointCloud(
            side2,
            np.ones(len(side2), dtype=float),
        )

    if source_representation in SURFACE_REPRESENTATIONS:
        return _surface_point_clouds(
            residues1,
            residues2,
            distance=float(distance),
            density=float(surface_density),
            trim_cutoff=float(surface_trim_cutoff),
            probe_radius=float(surface_probe_radius),
            representation=source_representation,
            proximity_length_scale=float(proximity_length_scale),
        )

    raise ValueError(f"Unknown Zernike representation {representation!r}")


def zernike_grids_from_point_clouds(
    side1: WeightedPointCloud,
    side2: WeightedPointCloud,
    *,
    representation: str = DEFAULT_REPRESENTATION,
    grid_size: int = DEFAULT_GRID_SIZE,
    sigma: float = DEFAULT_SIGMA,
    padding: float = DEFAULT_PADDING,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-side normalized voxel grids from weighted point clouds."""
    centroid, radius = _shared_normalization(side1, side2, float(padding))
    source_representation = zernike_source_representation(representation)

    if source_representation == SURFACE_BINARY:
        grid1 = _binary_grid(side1, centroid, radius, int(grid_size))
        grid2 = _binary_grid(side2, centroid, radius, int(grid_size))
    else:
        grid1 = _density_grid(side1, centroid, radius, int(grid_size), float(sigma))
        grid2 = _density_grid(side2, centroid, radius, int(grid_size), float(sigma))
    return grid1, grid2


def zernike_grids(
    residues1,
    residues2,
    *,
    representation: str = DEFAULT_REPRESENTATION,
    distance: float = DEFAULT_DISTANCE,
    grid_size: int = DEFAULT_GRID_SIZE,
    sigma: float = DEFAULT_SIGMA,
    padding: float = DEFAULT_PADDING,
    surface_density: float = DEFAULT_SURFACE_DENSITY,
    surface_trim_cutoff: float = DEFAULT_SURFACE_TRIM_CUTOFF,
    surface_probe_radius: float = DEFAULT_SURFACE_PROBE_RADIUS,
    proximity_length_scale: float = DEFAULT_PROXIMITY_LENGTH_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-side normalized voxel grids for a chosen interface representation."""
    side1, side2 = zernike_point_clouds(
        residues1,
        residues2,
        representation=representation,
        distance=distance,
        surface_density=surface_density,
        surface_trim_cutoff=surface_trim_cutoff,
        surface_probe_radius=surface_probe_radius,
        proximity_length_scale=proximity_length_scale,
    )
    return zernike_grids_from_point_clouds(
        side1,
        side2,
        representation=representation,
        grid_size=grid_size,
        sigma=sigma,
        padding=padding,
    )


def zernike_coefficients(grid: np.ndarray, order: int) -> np.ndarray:
    coeffs = ZernikeDescriptor.fit(data=np.asarray(grid, dtype=np.float32), order=int(order)).get_coefficients()
    return np.asarray(coeffs, dtype=float)


def zernike_similarity_from_coefficients(coeff1: np.ndarray, coeff2: np.ndarray) -> float:
    norm1 = float(np.linalg.norm(coeff1))
    norm2 = float(np.linalg.norm(coeff2))
    if norm1 <= 0.0 or norm2 <= 0.0:
        return 0.0
    similarity = float(np.dot(coeff1, coeff2) / (norm1 * norm2))
    return float(np.clip(similarity, -1.0, 1.0))


def zernike_joint_grid(grid1: np.ndarray, grid2: np.ndarray) -> np.ndarray:
    joint = np.asarray(grid1, dtype=np.float32) + np.asarray(grid2, dtype=np.float32)
    mass = float(np.sum(joint))
    if mass > 0.0:
        joint = joint / np.float32(mass)
    return joint.astype(np.float32, copy=False)


def zernike_coefficient_bundle_from_grids(
    grid1: np.ndarray,
    grid2: np.ndarray,
    fit_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit_to = int(fit_order)
    coeff1 = zernike_coefficients(grid1, fit_to)
    coeff2 = zernike_coefficients(grid2, fit_to)
    joint_grid = zernike_joint_grid(grid1, grid2)
    if float(np.sum(joint_grid)) <= 0.0:
        joint_coeff = np.zeros(zernike_descriptor_prefix_length(fit_to), dtype=float)
    else:
        joint_coeff = zernike_coefficients(joint_grid, fit_to)
    return coeff1, coeff2, joint_coeff


def zernike_score_from_coefficients(
    coeff1: np.ndarray,
    coeff2: np.ndarray | None,
    *,
    order: int,
    score_mode: str = DEFAULT_SCORE_MODE,
    fit_order: int | None = None,
    order_decay_n0: float = DEFAULT_ORDER_DECAY_N0,
) -> float:
    fit_to = fit_order_value(int(order), fit_order)
    prefix = zernike_descriptor_prefix_length(int(order))

    if score_mode == HARD_CUTOFF_SCORE:
        if coeff2 is None:
            raise ValueError("hard_cutoff score requires two coefficient vectors")
        return zernike_similarity_from_coefficients(coeff1[:prefix], coeff2[:prefix])

    if score_mode == GAUSSIAN_WEIGHTED_SCORE:
        if coeff2 is None:
            raise ValueError("gaussian_weighted score requires two coefficient vectors")
        weights = zernike_order_weights(int(order), float(order_decay_n0))
        return zernike_similarity_from_coefficients(coeff1[:prefix] * weights, coeff2[:prefix] * weights)

    if score_mode == JOINT_LOW_ORDER_RATIO_SCORE:
        full_prefix = zernike_descriptor_prefix_length(fit_to)
        denom = float(np.dot(coeff1[:full_prefix], coeff1[:full_prefix]))
        if denom <= 0.0:
            return 0.0
        numer = float(np.dot(coeff1[:prefix], coeff1[:prefix]))
        return float(np.clip(numer / denom, 0.0, 1.0))

    raise ValueError(f"Unknown Zernike score mode {score_mode!r}")


def zernike_descriptors_from_grids(
    grid1: np.ndarray,
    grid2: np.ndarray,
    order: int,
    *,
    fit_order: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    fit_to = fit_order_value(int(order), fit_order)
    coeff1 = zernike_coefficients(grid1, fit_to)
    coeff2 = zernike_coefficients(grid2, fit_to)
    prefix = zernike_descriptor_prefix_length(int(order))
    return coeff1[:prefix], coeff2[:prefix]


def zernike_score_from_grids(
    grid1: np.ndarray,
    grid2: np.ndarray,
    *,
    order: int,
    score_mode: str = DEFAULT_SCORE_MODE,
    fit_order: int | None = None,
    order_decay_n0: float = DEFAULT_ORDER_DECAY_N0,
) -> float:
    fit_to = fit_order_value(int(order), fit_order)
    coeff1, coeff2, joint_coeff = zernike_coefficient_bundle_from_grids(grid1, grid2, fit_to)
    if score_mode == JOINT_LOW_ORDER_RATIO_SCORE:
        return zernike_score_from_coefficients(
            joint_coeff,
            None,
            order=order,
            score_mode=score_mode,
            fit_order=fit_to,
            order_decay_n0=order_decay_n0,
        )
    return zernike_score_from_coefficients(
        coeff1,
        coeff2,
        order=order,
        score_mode=score_mode,
        fit_order=fit_to,
        order_decay_n0=order_decay_n0,
    )


def zernike_similarity_from_grids(
    grid1: np.ndarray,
    grid2: np.ndarray,
    order: int,
    *,
    fit_order: int | None = None,
) -> float:
    return zernike_score_from_grids(
        grid1,
        grid2,
        order=order,
        score_mode=HARD_CUTOFF_SCORE,
        fit_order=fit_order,
    )


def zernike_descriptors(
    residues1,
    residues2,
    *,
    representation: str = DEFAULT_REPRESENTATION,
    distance: float = DEFAULT_DISTANCE,
    grid_size: int = DEFAULT_GRID_SIZE,
    order: int = DEFAULT_ORDER,
    sigma: float = DEFAULT_SIGMA,
    padding: float = DEFAULT_PADDING,
    surface_density: float = DEFAULT_SURFACE_DENSITY,
    surface_trim_cutoff: float = DEFAULT_SURFACE_TRIM_CUTOFF,
    surface_probe_radius: float = DEFAULT_SURFACE_PROBE_RADIUS,
    proximity_length_scale: float = DEFAULT_PROXIMITY_LENGTH_SCALE,
    fit_order: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-side 3D Zernike descriptors for a chosen interface representation."""
    grid1, grid2 = zernike_grids(
        residues1,
        residues2,
        representation=representation,
        distance=distance,
        grid_size=grid_size,
        sigma=sigma,
        padding=padding,
        surface_density=surface_density,
        surface_trim_cutoff=surface_trim_cutoff,
        surface_probe_radius=surface_probe_radius,
        proximity_length_scale=proximity_length_scale,
    )
    return zernike_descriptors_from_grids(grid1, grid2, order=order, fit_order=fit_order)


def zernike_shape_complementarity(
    residues1,
    residues2,
    *,
    representation: str = DEFAULT_REPRESENTATION,
    distance: float = DEFAULT_DISTANCE,
    grid_size: int = DEFAULT_GRID_SIZE,
    order: int = DEFAULT_ORDER,
    sigma: float = DEFAULT_SIGMA,
    padding: float = DEFAULT_PADDING,
    surface_density: float = DEFAULT_SURFACE_DENSITY,
    surface_trim_cutoff: float = DEFAULT_SURFACE_TRIM_CUTOFF,
    surface_probe_radius: float = DEFAULT_SURFACE_PROBE_RADIUS,
    proximity_length_scale: float = DEFAULT_PROXIMITY_LENGTH_SCALE,
    score_mode: str = DEFAULT_SCORE_MODE,
    fit_order: int | None = None,
    order_decay_n0: float = DEFAULT_ORDER_DECAY_N0,
) -> float:
    """
    Compare low-pass 3D Zernike interface descriptors.

    Returns 0.0 when no interface can be voxelized robustly.
    """
    try:
        grid1, grid2 = zernike_grids(
            residues1,
            residues2,
            representation=representation,
            distance=distance,
            grid_size=grid_size,
            sigma=sigma,
            padding=padding,
            surface_density=surface_density,
            surface_trim_cutoff=surface_trim_cutoff,
            surface_probe_radius=surface_probe_radius,
            proximity_length_scale=proximity_length_scale,
        )
        return zernike_score_from_grids(
            grid1,
            grid2,
            order=order,
            score_mode=score_mode,
            fit_order=fit_order,
            order_decay_n0=order_decay_n0,
        )
    except Exception:
        return 0.0


__all__ = [
    "ATOM_GAUSSIAN",
    "DEFAULT_DISTANCE",
    "DEFAULT_GRID_SIZE",
    "DEFAULT_ORDER",
    "DEFAULT_ORDER_DECAY_N0",
    "DEFAULT_PADDING",
    "DEFAULT_PROXIMITY_LENGTH_SCALE",
    "DEFAULT_REPRESENTATION",
    "DEFAULT_SCORE_MODE",
    "DEFAULT_SIGMA",
    "DEFAULT_SURFACE_DENSITY",
    "DEFAULT_SURFACE_PROBE_RADIUS",
    "DEFAULT_SURFACE_TRIM_CUTOFF",
    "GAUSSIAN_REPRESENTATIONS",
    "GAUSSIAN_WEIGHTED_SCORE",
    "HARD_CUTOFF_SCORE",
    "JOINT_LOW_ORDER_RATIO_SCORE",
    "JOINT_REPRESENTATIONS",
    "JOINT_RESIDUE_BEAD_GAUSSIAN",
    "JOINT_SURFACE_GAUSSIAN",
    "PER_SIDE_REPRESENTATIONS",
    "RESIDUE_BEAD_GAUSSIAN",
    "SURFACE_BINARY",
    "SURFACE_GAUSSIAN",
    "SURFACE_PROXIMITY_GAUSSIAN",
    "SURFACE_REPRESENTATIONS",
    "WeightedPointCloud",
    "ZernikeSpec",
    "fit_order_value",
    "zernike_candidate_family",
    "zernike_coefficient_bundle_from_grids",
    "zernike_coefficients",
    "zernike_descriptor_orders",
    "zernike_descriptor_prefix_length",
    "zernike_descriptors",
    "zernike_descriptors_from_grids",
    "zernike_grids",
    "zernike_grids_from_point_clouds",
    "zernike_joint_grid",
    "zernike_order_weights",
    "zernike_point_clouds",
    "zernike_score_from_coefficients",
    "zernike_score_from_grids",
    "zernike_shape_complementarity",
    "zernike_similarity_from_coefficients",
    "zernike_similarity_from_grids",
    "zernike_source_representation",
]
