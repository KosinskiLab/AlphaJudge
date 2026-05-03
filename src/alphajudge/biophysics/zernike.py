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
SURFACE_NORMAL_GAP = "surface_normal_gap"
JOINT_RESIDUE_BEAD_GAUSSIAN = "joint_residue_bead_gaussian"
JOINT_SURFACE_GAUSSIAN = "joint_surface_gaussian"

HARD_CUTOFF_SCORE = "hard_cutoff"
GAUSSIAN_WEIGHTED_SCORE = "gaussian_weighted"
JOINT_LOW_ORDER_RATIO_SCORE = "joint_low_order_ratio"
SHARED_GRID_OVERLAP_SCORE = "shared_grid_overlap"
GAP_ZERNIKE_RATIO_SCORE = "gap_zernike_ratio"
GAP_ZERNIKE_WEIGHTED_SCORE = "gap_zernike_weighted"
GAP_ZERNIKE_NONUNIFORM_SCORE = "gap_zernike_nonuniform"
GAP_ZERNIKE_BANDPASS_SCORE = "gap_zernike_bandpass"
GAP_ZERNIKE_EXCESS_BANDPASS_SCORE = "gap_zernike_excess_bandpass"
GAP_ZERNIKE_SOFT_BANDPASS_SCORE = "gap_zernike_soft_bandpass"
GAP_ZERNIKE_EXCESS_CONTACT_SCORE = "gap_zernike_excess_contact_bandpass"
SC_GATED_OVERLAP_SCORE = "sc_gated_overlap_rescue"
SC_GATED_GAP_NONUNIFORM_SCORE = "sc_gated_gap_nonuniform_rescue"
SC_GATED_GAP_BANDPASS_SCORE = "sc_gated_gap_bandpass_rescue"
SC_GATED_GAP_SOFT_SCORE = "sc_gated_gap_soft_rescue"
NORMAL_GAP_FIELD_SCORE = "normal_gap_field"
NORMAL_GAP_CONTACT_SCORE = "normal_gap_contact_field"

NORMAL_GAP_SCORE_MODES = {
    NORMAL_GAP_FIELD_SCORE,
    NORMAL_GAP_CONTACT_SCORE,
}

GAP_SCORE_MODES = {
    GAP_ZERNIKE_RATIO_SCORE,
    GAP_ZERNIKE_WEIGHTED_SCORE,
    GAP_ZERNIKE_NONUNIFORM_SCORE,
    GAP_ZERNIKE_BANDPASS_SCORE,
    GAP_ZERNIKE_EXCESS_BANDPASS_SCORE,
    GAP_ZERNIKE_SOFT_BANDPASS_SCORE,
    GAP_ZERNIKE_EXCESS_CONTACT_SCORE,
}
GRID_SCORE_MODES = {
    SHARED_GRID_OVERLAP_SCORE,
    GAP_ZERNIKE_RATIO_SCORE,
    GAP_ZERNIKE_WEIGHTED_SCORE,
    GAP_ZERNIKE_NONUNIFORM_SCORE,
    GAP_ZERNIKE_BANDPASS_SCORE,
    GAP_ZERNIKE_EXCESS_BANDPASS_SCORE,
    GAP_ZERNIKE_SOFT_BANDPASS_SCORE,
    GAP_ZERNIKE_EXCESS_CONTACT_SCORE,
}
SC_GATED_SCORE_MODES = {
    SC_GATED_OVERLAP_SCORE,
    SC_GATED_GAP_NONUNIFORM_SCORE,
    SC_GATED_GAP_BANDPASS_SCORE,
    SC_GATED_GAP_SOFT_SCORE,
}

JOINT_REPRESENTATIONS = {
    JOINT_RESIDUE_BEAD_GAUSSIAN,
    JOINT_SURFACE_GAUSSIAN,
}
GAUSSIAN_REPRESENTATIONS = {
    ATOM_GAUSSIAN,
    RESIDUE_BEAD_GAUSSIAN,
    SURFACE_GAUSSIAN,
    SURFACE_PROXIMITY_GAUSSIAN,
    SURFACE_NORMAL_GAP,
    JOINT_RESIDUE_BEAD_GAUSSIAN,
    JOINT_SURFACE_GAUSSIAN,
}
SURFACE_REPRESENTATIONS = {
    SURFACE_BINARY,
    SURFACE_GAUSSIAN,
    SURFACE_PROXIMITY_GAUSSIAN,
    SURFACE_NORMAL_GAP,
    JOINT_SURFACE_GAUSSIAN,
}
NORMAL_GAP_REPRESENTATIONS = {
    SURFACE_NORMAL_GAP,
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
DEFAULT_NORMAL_GAP_GOOD_SCALE = 1.0
DEFAULT_NORMAL_GAP_FAR_SCALE = 2.5
DEFAULT_NORMAL_GAP_CLASH_WEIGHT = 0.75
DEFAULT_NORMAL_GAP_FAR_WEIGHT = 0.5
DEFAULT_NORMAL_GAP_CONTACT_SCALE = 500.0
DEFAULT_GAP_BAND_SOFT_WIDTH = 1.0
DEFAULT_GAP_CONTACT_SCALE = 0.05
DEFAULT_SC_RESCUE_WEIGHT = 0.2
DEFAULT_SC_RESCUE_SCALE = 0.2
DEFAULT_SC_RESCUE_FLOOR = 0.0

_SHORT_SCORE_TAGS = {
    HARD_CUTOFF_SCORE: "hard",
    GAUSSIAN_WEIGHTED_SCORE: "weighted",
    JOINT_LOW_ORDER_RATIO_SCORE: "jointratio",
    SHARED_GRID_OVERLAP_SCORE: "overlap",
    GAP_ZERNIKE_RATIO_SCORE: "gapratio",
    GAP_ZERNIKE_WEIGHTED_SCORE: "gapweighted",
    GAP_ZERNIKE_NONUNIFORM_SCORE: "gapnonuniform",
    GAP_ZERNIKE_BANDPASS_SCORE: "gapband",
    GAP_ZERNIKE_EXCESS_BANDPASS_SCORE: "gapexcess",
    GAP_ZERNIKE_SOFT_BANDPASS_SCORE: "gapsoft",
    GAP_ZERNIKE_EXCESS_CONTACT_SCORE: "gapcontact",
    SC_GATED_OVERLAP_SCORE: "scoverlap",
    SC_GATED_GAP_NONUNIFORM_SCORE: "scnonuniform",
    SC_GATED_GAP_BANDPASS_SCORE: "scgapband",
    SC_GATED_GAP_SOFT_SCORE: "scgapsoft",
    NORMAL_GAP_FIELD_SCORE: "normalgap",
    NORMAL_GAP_CONTACT_SCORE: "normalcontact",
}


@dataclass(frozen=True, slots=True)
class WeightedPointCloud:
    points: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True, slots=True)
class NormalGapFieldBundle:
    good_grid: np.ndarray
    clash_grid: np.ndarray
    far_grid: np.ndarray
    good_mass: float
    clash_mass: float
    far_mass: float


@dataclass(frozen=True, slots=True)
class NormalGapCoefficientBundle:
    good_coeff: np.ndarray
    clash_coeff: np.ndarray
    far_coeff: np.ndarray
    good_mass: float
    clash_mass: float
    far_mass: float
    fit_order: int


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
    normal_gap_good_scale: float = DEFAULT_NORMAL_GAP_GOOD_SCALE
    normal_gap_far_scale: float = DEFAULT_NORMAL_GAP_FAR_SCALE
    normal_gap_clash_weight: float = DEFAULT_NORMAL_GAP_CLASH_WEIGHT
    normal_gap_far_weight: float = DEFAULT_NORMAL_GAP_FAR_WEIGHT
    normal_gap_contact_scale: float = DEFAULT_NORMAL_GAP_CONTACT_SCALE
    gap_band_soft_width: float = DEFAULT_GAP_BAND_SOFT_WIDTH
    gap_contact_scale: float = DEFAULT_GAP_CONTACT_SCALE
    sc_rescue_weight: float = DEFAULT_SC_RESCUE_WEIGHT
    sc_rescue_scale: float = DEFAULT_SC_RESCUE_SCALE
    sc_rescue_floor: float = DEFAULT_SC_RESCUE_FLOOR

    def candidate_id(self) -> str:
        parts = [self.representation, f"g{self.grid_size}", f"o{self.order}"]
        if self.representation in GAUSSIAN_REPRESENTATIONS:
            parts.append(f"s{_tag_float(self.sigma)}")
        if self.representation in SURFACE_REPRESENTATIONS:
            parts.append(f"d{_tag_float(self.surface_density)}")
            if not math.isclose(
                float(self.surface_trim_cutoff),
                float(DEFAULT_SURFACE_TRIM_CUTOFF),
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                parts.append(f"tr{_tag_float(self.surface_trim_cutoff)}")
            if not math.isclose(
                float(self.surface_probe_radius),
                float(DEFAULT_SURFACE_PROBE_RADIUS),
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                parts.append(f"pr{_tag_float(self.surface_probe_radius)}")
        if self.score_mode != HARD_CUTOFF_SCORE:
            parts.append(f"m{_SHORT_SCORE_TAGS[self.score_mode]}")
        if self.score_mode in {GAUSSIAN_WEIGHTED_SCORE, GAP_ZERNIKE_WEIGHTED_SCORE}:
            parts.append(f"n{_tag_float(self.order_decay_n0)}")
        if self.score_mode == GAP_ZERNIKE_SOFT_BANDPASS_SCORE and not math.isclose(
            float(self.gap_band_soft_width),
            DEFAULT_GAP_BAND_SOFT_WIDTH,
        ):
            parts.append(f"w{_tag_float(self.gap_band_soft_width)}")
        if self.score_mode == GAP_ZERNIKE_EXCESS_CONTACT_SCORE and not math.isclose(
            float(self.gap_contact_scale),
            DEFAULT_GAP_CONTACT_SCALE,
        ):
            parts.append(f"cs{_tag_float(self.gap_contact_scale)}")
        if self.score_mode in SC_GATED_SCORE_MODES:
            if not math.isclose(float(self.sc_rescue_weight), DEFAULT_SC_RESCUE_WEIGHT):
                parts.append(f"rw{_tag_float(self.sc_rescue_weight)}")
            if not math.isclose(float(self.sc_rescue_scale), DEFAULT_SC_RESCUE_SCALE):
                parts.append(f"rs{_tag_float(self.sc_rescue_scale)}")
            if not math.isclose(float(self.sc_rescue_floor), DEFAULT_SC_RESCUE_FLOOR):
                parts.append(f"rf{_tag_float(self.sc_rescue_floor)}")
        if self.score_mode in NORMAL_GAP_SCORE_MODES:
            if not math.isclose(float(self.normal_gap_good_scale), DEFAULT_NORMAL_GAP_GOOD_SCALE):
                parts.append(f"gs{_tag_float(self.normal_gap_good_scale)}")
            if not math.isclose(float(self.normal_gap_far_scale), DEFAULT_NORMAL_GAP_FAR_SCALE):
                parts.append(f"fs{_tag_float(self.normal_gap_far_scale)}")
            if not math.isclose(float(self.normal_gap_clash_weight), DEFAULT_NORMAL_GAP_CLASH_WEIGHT):
                parts.append(f"cw{_tag_float(self.normal_gap_clash_weight)}")
            if not math.isclose(float(self.normal_gap_far_weight), DEFAULT_NORMAL_GAP_FAR_WEIGHT):
                parts.append(f"fw{_tag_float(self.normal_gap_far_weight)}")
            if self.score_mode == NORMAL_GAP_CONTACT_SCORE and not math.isclose(
                float(self.normal_gap_contact_scale),
                DEFAULT_NORMAL_GAP_CONTACT_SCALE,
            ):
                parts.append(f"cs{_tag_float(self.normal_gap_contact_scale)}")
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


def zernike_candidate_family(representation: str, score_mode: str | None = None) -> str:
    if representation in NORMAL_GAP_REPRESENTATIONS or score_mode in NORMAL_GAP_SCORE_MODES:
        return "normal_gap"
    if score_mode in SC_GATED_SCORE_MODES:
        return "sc_gated_rescue"
    if score_mode in GRID_SCORE_MODES:
        return "grid_gap"
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


def _normal_gap_midpoint_clouds(
    dots_a: np.ndarray,
    normals_a: np.ndarray,
    dots_b: np.ndarray,
    normals_b: np.ndarray,
    *,
    good_scale: float,
    far_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if dots_a.size == 0 or dots_b.size == 0:
        empty_points = np.empty((0, 3), dtype=float)
        empty_weights = np.empty(0, dtype=float)
        return empty_points, empty_weights, empty_weights, empty_weights

    dist, nearest = cKDTree(dots_b).query(dots_a)
    opposite = dots_b[nearest]
    opposite_normals = normals_b[nearest]
    midpoints = 0.5 * (dots_a + opposite)

    normal_complement = np.clip(
        -np.einsum("ij,ij->i", normals_a, opposite_normals),
        0.0,
        1.0,
    )
    good_scale = max(float(good_scale), 1e-6)
    far_scale = max(float(far_scale), 1e-6)
    close = np.exp(-((dist / good_scale) ** 2))
    far = 1.0 - np.exp(-((dist / far_scale) ** 2))

    good_weights = normal_complement * close
    # Close surface samples with poor opposing normals are likely side-chain
    # clashes or tangential/nonspecific contacts after smoothing.
    clash_weights = (1.0 - normal_complement) * close
    # Opposite-facing but distant samples represent an interface gap.
    far_weights = normal_complement * far
    return midpoints, good_weights, clash_weights, far_weights


def zernike_normal_gap_field_bundle(
    residues1,
    residues2,
    *,
    distance: float = DEFAULT_DISTANCE,
    grid_size: int = DEFAULT_GRID_SIZE,
    sigma: float = DEFAULT_SIGMA,
    padding: float = DEFAULT_PADDING,
    surface_density: float = DEFAULT_SURFACE_DENSITY,
    surface_trim_cutoff: float = DEFAULT_SURFACE_TRIM_CUTOFF,
    surface_probe_radius: float = DEFAULT_SURFACE_PROBE_RADIUS,
    normal_gap_good_scale: float = DEFAULT_NORMAL_GAP_GOOD_SCALE,
    normal_gap_far_scale: float = DEFAULT_NORMAL_GAP_FAR_SCALE,
) -> NormalGapFieldBundle:
    """Build smoothed Connolly midpoint fields for good contact, clash, and gap."""
    d1, n1, d2, n2 = interface_surface_dots(
        residues1,
        residues2,
        distance=float(distance),
        density=float(surface_density),
        trim_cutoff=float(surface_trim_cutoff),
        probe_radius=float(surface_probe_radius),
    )
    if d1.size == 0 or d2.size == 0:
        raise ValueError("No interface surface dots found for normal-gap Zernike descriptor.")

    centroid, radius = _shared_normalization(
        WeightedPointCloud(d1, np.ones(len(d1), dtype=float)),
        WeightedPointCloud(d2, np.ones(len(d2), dtype=float)),
        float(padding),
    )

    mid12, good12, clash12, far12 = _normal_gap_midpoint_clouds(
        d1,
        n1,
        d2,
        n2,
        good_scale=float(normal_gap_good_scale),
        far_scale=float(normal_gap_far_scale),
    )
    mid21, good21, clash21, far21 = _normal_gap_midpoint_clouds(
        d2,
        n2,
        d1,
        n1,
        good_scale=float(normal_gap_good_scale),
        far_scale=float(normal_gap_far_scale),
    )
    midpoints = np.vstack([mid12, mid21])
    good_weights = np.concatenate([good12, good21])
    clash_weights = np.concatenate([clash12, clash21])
    far_weights = np.concatenate([far12, far21])

    good_grid = _density_grid(
        WeightedPointCloud(midpoints, good_weights),
        centroid,
        radius,
        int(grid_size),
        float(sigma),
    )
    clash_grid = _density_grid(
        WeightedPointCloud(midpoints, clash_weights),
        centroid,
        radius,
        int(grid_size),
        float(sigma),
    )
    far_grid = _density_grid(
        WeightedPointCloud(midpoints, far_weights),
        centroid,
        radius,
        int(grid_size),
        float(sigma),
    )
    return NormalGapFieldBundle(
        good_grid=good_grid,
        clash_grid=clash_grid,
        far_grid=far_grid,
        good_mass=float(np.sum(good_weights)),
        clash_mass=float(np.sum(clash_weights)),
        far_mass=float(np.sum(far_weights)),
    )


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


def zernike_l2_normalized_grid(grid: np.ndarray) -> np.ndarray:
    """Return a non-negative L2-normalized grid, or all zeros for empty input."""
    arr = np.maximum(np.asarray(grid, dtype=np.float32), 0.0)
    norm = float(np.linalg.norm(arr.ravel()))
    if norm <= 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / np.float32(norm)).astype(np.float32, copy=False)


def zernike_shared_grid_overlap(grid1: np.ndarray, grid2: np.ndarray) -> float:
    """L2-normalized density overlap in the shared interface box."""
    norm1 = zernike_l2_normalized_grid(grid1)
    norm2 = zernike_l2_normalized_grid(grid2)
    if float(np.sum(norm1)) <= 0.0 or float(np.sum(norm2)) <= 0.0:
        return 0.0
    overlap = float(np.dot(norm1.ravel(), norm2.ravel()))
    return float(np.clip(overlap, 0.0, 1.0))


def zernike_effective_grid_volume(grid: np.ndarray) -> float:
    """Participation-ratio volume of a non-negative density grid."""
    norm = zernike_l2_normalized_grid(grid)
    if float(np.sum(norm)) <= 0.0:
        return 0.0
    effective = float(np.sum(norm) ** 2)
    return float(np.clip(effective, 0.0, float(norm.size)))


def zernike_expected_random_overlap(
    side1_effective_volume: float,
    side2_effective_volume: float,
    voxel_count: int,
) -> float:
    """Expected L2 overlap for two independently placed densities in one box."""
    n_voxels = max(int(voxel_count), 1)
    volume1 = max(float(side1_effective_volume), 0.0)
    volume2 = max(float(side2_effective_volume), 0.0)
    expected = math.sqrt(volume1 * volume2) / float(n_voxels)
    return float(np.clip(expected, 0.0, 1.0))


def zernike_excess_grid_overlap(
    grid_overlap: float,
    side1_effective_volume: float,
    side2_effective_volume: float,
    voxel_count: int,
) -> float:
    """Overlap above the random-density expectation, normalized to [0, 1]."""
    overlap = float(np.clip(float(grid_overlap), 0.0, 1.0))
    expected = zernike_expected_random_overlap(
        side1_effective_volume,
        side2_effective_volume,
        voxel_count,
    )
    if overlap <= expected:
        return 0.0
    return float(np.clip((overlap - expected) / max(1.0 - expected, 1e-12), 0.0, 1.0))


def zernike_gap_grid(grid1: np.ndarray, grid2: np.ndarray) -> np.ndarray:
    """Build a soft shared-contact density from two L2-normalized side grids."""
    norm1 = zernike_l2_normalized_grid(grid1)
    norm2 = zernike_l2_normalized_grid(grid2)
    if float(np.sum(norm1)) <= 0.0 or float(np.sum(norm2)) <= 0.0:
        return np.zeros_like(norm1, dtype=np.float32)
    return np.sqrt(norm1 * norm2).astype(np.float32, copy=False)


def zernike_low_order_energy_ratio(coeff: np.ndarray, order: int, fit_order: int | None = None) -> float:
    fit_to = fit_order_value(int(order), fit_order)
    prefix = zernike_descriptor_prefix_length(int(order))
    full_prefix = zernike_descriptor_prefix_length(fit_to)
    full_coeff = np.asarray(coeff[:full_prefix], dtype=float)
    denom = float(np.dot(full_coeff, full_coeff))
    if denom <= 0.0:
        return 0.0
    low_coeff = np.asarray(coeff[:prefix], dtype=float)
    numer = float(np.dot(low_coeff, low_coeff))
    return float(np.clip(numer / denom, 0.0, 1.0))


def zernike_weighted_energy_ratio(
    coeff: np.ndarray,
    order: int,
    fit_order: int | None = None,
    order_decay_n0: float = DEFAULT_ORDER_DECAY_N0,
) -> float:
    fit_to = fit_order_value(int(order), fit_order)
    full_prefix = zernike_descriptor_prefix_length(fit_to)
    full_coeff = np.asarray(coeff[:full_prefix], dtype=float)
    denom = float(np.dot(full_coeff, full_coeff))
    if denom <= 0.0:
        return 0.0
    weights = zernike_order_weights(fit_to, float(order_decay_n0))
    weighted = full_coeff * weights
    numer = float(np.dot(weighted, weighted))
    return float(np.clip(numer / denom, 0.0, 1.0))


def zernike_band_energy_ratio(
    coeff: np.ndarray,
    min_order: int,
    max_order: int,
    fit_order: int | None = None,
) -> float:
    """Fraction of descriptor energy in a low/mid-order Zernike band."""
    fit_to = fit_order_value(int(max_order), fit_order)
    full_prefix = zernike_descriptor_prefix_length(fit_to)
    full_coeff = np.asarray(coeff[:full_prefix], dtype=float)
    denom = float(np.dot(full_coeff, full_coeff))
    if denom <= 0.0:
        return 0.0
    orders = zernike_descriptor_orders(fit_to)[:full_prefix]
    mask = (orders >= int(min_order)) & (orders <= int(max_order))
    band_coeff = full_coeff[mask]
    numer = float(np.dot(band_coeff, band_coeff))
    return float(np.clip(numer / denom, 0.0, 1.0))


def zernike_soft_band_energy_ratio(
    coeff: np.ndarray,
    min_order: int,
    cutoff_order: int,
    fit_order: int | None = None,
    soft_width: float = DEFAULT_GAP_BAND_SOFT_WIDTH,
) -> float:
    """Band energy with a hard low-order floor and smooth high-order taper."""
    fit_to = fit_order_value(int(cutoff_order), fit_order)
    full_prefix = zernike_descriptor_prefix_length(fit_to)
    full_coeff = np.asarray(coeff[:full_prefix], dtype=float)
    denom = float(np.dot(full_coeff, full_coeff))
    if denom <= 0.0:
        return 0.0
    orders = zernike_descriptor_orders(fit_to)[:full_prefix]
    width = max(float(soft_width), 1e-6)
    weights = np.zeros_like(full_coeff, dtype=float)
    in_band = (orders >= int(min_order)) & (orders <= int(cutoff_order))
    high = orders > int(cutoff_order)
    weights[in_band] = 1.0
    weights[high] = np.exp(-(((orders[high] - int(cutoff_order)) / width) ** 2))
    weighted = full_coeff * weights
    numer = float(np.dot(weighted, weighted))
    return float(np.clip(numer / denom, 0.0, 1.0))


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


def zernike_gap_coefficient_bundle_from_grids(
    grid1: np.ndarray,
    grid2: np.ndarray,
    fit_order: int,
) -> tuple[np.ndarray, float, float, float, int]:
    overlap = zernike_shared_grid_overlap(grid1, grid2)
    side1_effective_volume = zernike_effective_grid_volume(grid1)
    side2_effective_volume = zernike_effective_grid_volume(grid2)
    voxel_count = int(np.asarray(grid1).size)
    gap_grid = zernike_gap_grid(grid1, grid2)
    if float(np.sum(gap_grid)) <= 0.0:
        gap_coeff = np.zeros(zernike_descriptor_prefix_length(int(fit_order)), dtype=float)
    else:
        gap_coeff = zernike_coefficients(gap_grid, int(fit_order))
    return gap_coeff, overlap, side1_effective_volume, side2_effective_volume, voxel_count


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
        return zernike_low_order_energy_ratio(coeff1, int(order), fit_to)

    raise ValueError(f"Unknown Zernike score mode {score_mode!r}")


def zernike_score_from_gap_coefficients(
    gap_coeff: np.ndarray,
    grid_overlap: float,
    *,
    order: int,
    score_mode: str,
    fit_order: int | None = None,
    order_decay_n0: float = DEFAULT_ORDER_DECAY_N0,
    side1_effective_volume: float | None = None,
    side2_effective_volume: float | None = None,
    voxel_count: int | None = None,
    gap_band_soft_width: float = DEFAULT_GAP_BAND_SOFT_WIDTH,
    gap_contact_scale: float = DEFAULT_GAP_CONTACT_SCALE,
) -> float:
    diagnostics = zernike_gap_score_diagnostics(
        gap_coeff,
        grid_overlap,
        order=order,
        score_mode=score_mode,
        fit_order=fit_order,
        order_decay_n0=order_decay_n0,
        side1_effective_volume=side1_effective_volume,
        side2_effective_volume=side2_effective_volume,
        voxel_count=voxel_count,
        gap_band_soft_width=gap_band_soft_width,
        gap_contact_scale=gap_contact_scale,
    )
    if "gap_final_score" in diagnostics:
        return diagnostics["gap_final_score"]
    raise ValueError(f"Unknown Zernike gap score mode {score_mode!r}")


def zernike_gap_score_diagnostics(
    gap_coeff: np.ndarray,
    grid_overlap: float,
    *,
    order: int,
    score_mode: str,
    fit_order: int | None = None,
    order_decay_n0: float = DEFAULT_ORDER_DECAY_N0,
    side1_effective_volume: float | None = None,
    side2_effective_volume: float | None = None,
    voxel_count: int | None = None,
    gap_band_soft_width: float = DEFAULT_GAP_BAND_SOFT_WIDTH,
    gap_contact_scale: float = DEFAULT_GAP_CONTACT_SCALE,
) -> dict[str, float]:
    """Return atom/residue/surface gap terms used by shared-box scores."""
    overlap = float(np.clip(float(grid_overlap), 0.0, 1.0))
    volume1 = 0.0 if side1_effective_volume is None else max(float(side1_effective_volume), 0.0)
    volume2 = 0.0 if side2_effective_volume is None else max(float(side2_effective_volume), 0.0)
    n_voxels = 0 if voxel_count is None else int(voxel_count)
    expected_overlap = (
        zernike_expected_random_overlap(volume1, volume2, n_voxels)
        if n_voxels > 0
        else 0.0
    )
    excess_overlap = (
        zernike_excess_grid_overlap(overlap, volume1, volume2, n_voxels)
        if n_voxels > 0
        else overlap
    )
    low_order_ratio = zernike_low_order_energy_ratio(gap_coeff, int(order), fit_order)
    weighted_ratio = zernike_weighted_energy_ratio(
        gap_coeff,
        int(order),
        fit_order,
        order_decay_n0=float(order_decay_n0),
    )
    band_ratio = zernike_band_energy_ratio(gap_coeff, 2, int(order), fit_order)
    soft_band_ratio = zernike_soft_band_energy_ratio(
        gap_coeff,
        2,
        int(order),
        fit_order,
        soft_width=float(gap_band_soft_width),
    )
    contact_scale = max(float(gap_contact_scale), 1e-12)
    contact_amount = 1.0 - math.exp(-(excess_overlap / contact_scale)) if excess_overlap > 0.0 else 0.0

    if score_mode == SHARED_GRID_OVERLAP_SCORE:
        final_score = overlap
    elif score_mode == GAP_ZERNIKE_RATIO_SCORE:
        final_score = overlap * low_order_ratio
    elif score_mode == GAP_ZERNIKE_WEIGHTED_SCORE:
        final_score = overlap * weighted_ratio
    elif score_mode == GAP_ZERNIKE_NONUNIFORM_SCORE:
        final_score = overlap * (1.0 - low_order_ratio)
    elif score_mode == GAP_ZERNIKE_BANDPASS_SCORE:
        final_score = overlap * band_ratio
    elif score_mode == GAP_ZERNIKE_EXCESS_BANDPASS_SCORE:
        final_score = excess_overlap * band_ratio
    elif score_mode == GAP_ZERNIKE_SOFT_BANDPASS_SCORE:
        final_score = overlap * soft_band_ratio
    elif score_mode == GAP_ZERNIKE_EXCESS_CONTACT_SCORE:
        final_score = contact_amount * band_ratio
    else:
        raise ValueError(f"Unknown Zernike gap score mode {score_mode!r}")

    return {
        "gap_raw_overlap": overlap,
        "gap_expected_random_overlap": expected_overlap,
        "gap_excess_overlap": excess_overlap,
        "gap_side1_effective_volume": volume1,
        "gap_side2_effective_volume": volume2,
        "gap_voxel_count": float(n_voxels),
        "gap_low_order_ratio": low_order_ratio,
        "gap_weighted_ratio": weighted_ratio,
        "gap_band_energy_ratio": band_ratio,
        "gap_soft_band_energy_ratio": soft_band_ratio,
        "gap_contact_amount": contact_amount,
        "gap_final_score": float(np.clip(final_score, 0.0, 1.0)),
    }


def zernike_score_from_normal_gap_fields(
    bundle: NormalGapFieldBundle,
    *,
    order: int,
    fit_order: int | None = None,
    normal_gap_clash_weight: float = DEFAULT_NORMAL_GAP_CLASH_WEIGHT,
    normal_gap_far_weight: float = DEFAULT_NORMAL_GAP_FAR_WEIGHT,
    normal_gap_contact_scale: float = DEFAULT_NORMAL_GAP_CONTACT_SCALE,
    score_mode: str = NORMAL_GAP_FIELD_SCORE,
) -> float:
    """Score normal-aware contact after low-pass Zernike filtering of all fields."""
    coeffs = zernike_normal_gap_coefficient_bundle(
        bundle,
        fit_order_value(int(order), fit_order),
    )
    return zernike_score_from_normal_gap_coefficients(
        coeffs,
        order=order,
        fit_order=fit_order,
        normal_gap_clash_weight=normal_gap_clash_weight,
        normal_gap_far_weight=normal_gap_far_weight,
        normal_gap_contact_scale=normal_gap_contact_scale,
        score_mode=score_mode,
    )


def zernike_normal_gap_coefficient_bundle(
    bundle: NormalGapFieldBundle,
    fit_order: int,
) -> NormalGapCoefficientBundle:
    """Fit Zernike coefficients for good-contact, clash, and far-gap fields."""
    fit_to = int(fit_order)
    length = zernike_descriptor_prefix_length(fit_to)

    def _coeff_or_zero(grid: np.ndarray) -> np.ndarray:
        if float(np.sum(grid)) <= 0.0:
            return np.zeros(length, dtype=float)
        return zernike_coefficients(grid, fit_to)

    return NormalGapCoefficientBundle(
        good_coeff=_coeff_or_zero(bundle.good_grid),
        clash_coeff=_coeff_or_zero(bundle.clash_grid),
        far_coeff=_coeff_or_zero(bundle.far_grid),
        good_mass=float(bundle.good_mass),
        clash_mass=float(bundle.clash_mass),
        far_mass=float(bundle.far_mass),
        fit_order=fit_to,
    )


def _normal_gap_field_ratio(
    coeff: np.ndarray,
    mass: float,
    *,
    order: int,
    fit_order: int,
) -> float:
    if max(float(mass), 0.0) <= 0.0:
        return 0.0
    if int(order) >= 2:
        return zernike_band_energy_ratio(coeff, 2, int(order), fit_order)
    return zernike_low_order_energy_ratio(coeff, int(order), fit_order)


def zernike_normal_gap_score_diagnostics(
    coeffs: NormalGapCoefficientBundle,
    *,
    order: int,
    fit_order: int | None = None,
    normal_gap_clash_weight: float = DEFAULT_NORMAL_GAP_CLASH_WEIGHT,
    normal_gap_far_weight: float = DEFAULT_NORMAL_GAP_FAR_WEIGHT,
    normal_gap_contact_scale: float = DEFAULT_NORMAL_GAP_CONTACT_SCALE,
) -> dict[str, float]:
    """Return field-level terms used by the normal-aware gap score."""
    requested_fit = fit_order_value(int(order), fit_order if fit_order is not None else coeffs.fit_order)
    fit_to = min(requested_fit, int(coeffs.fit_order))
    score_order = min(int(order), fit_to)
    good_ratio = _normal_gap_field_ratio(
        coeffs.good_coeff,
        coeffs.good_mass,
        order=score_order,
        fit_order=fit_to,
    )
    clash_ratio = _normal_gap_field_ratio(
        coeffs.clash_coeff,
        coeffs.clash_mass,
        order=score_order,
        fit_order=fit_to,
    )
    far_ratio = _normal_gap_field_ratio(
        coeffs.far_coeff,
        coeffs.far_mass,
        order=score_order,
        fit_order=fit_to,
    )
    good_mass = max(float(coeffs.good_mass), 0.0)
    clash_mass = max(float(coeffs.clash_mass), 0.0)
    far_mass = max(float(coeffs.far_mass), 0.0)
    good_signal = good_mass * good_ratio
    clash_signal = clash_mass * clash_ratio
    far_signal = far_mass * far_ratio
    weighted_clash_signal = max(float(normal_gap_clash_weight), 0.0) * clash_signal
    weighted_far_signal = max(float(normal_gap_far_weight), 0.0) * far_signal
    denominator = good_signal + weighted_clash_signal + weighted_far_signal
    quality_score = good_signal / denominator if denominator > 0.0 else 0.0
    contact_scale = max(float(normal_gap_contact_scale), 1e-6)
    contact_amount = 1.0 - math.exp(-(good_signal / contact_scale)) if good_signal > 0.0 else 0.0
    contact_score = quality_score * contact_amount
    return {
        "normal_gap_good_mass": good_mass,
        "normal_gap_clash_mass": clash_mass,
        "normal_gap_far_mass": far_mass,
        "normal_gap_good_structured_ratio": good_ratio,
        "normal_gap_clash_structured_ratio": clash_ratio,
        "normal_gap_far_structured_ratio": far_ratio,
        "normal_gap_good_signal": good_signal,
        "normal_gap_clash_signal": clash_signal,
        "normal_gap_far_signal": far_signal,
        "normal_gap_weighted_clash_signal": weighted_clash_signal,
        "normal_gap_weighted_far_signal": weighted_far_signal,
        "normal_gap_signal_denominator": denominator,
        "normal_gap_contact_amount": float(np.clip(contact_amount, 0.0, 1.0)),
        "normal_gap_score_from_fields": float(np.clip(quality_score, 0.0, 1.0)),
        "normal_gap_contact_score_from_fields": float(np.clip(contact_score, 0.0, 1.0)),
    }


def zernike_score_from_normal_gap_coefficients(
    coeffs: NormalGapCoefficientBundle,
    *,
    order: int,
    fit_order: int | None = None,
    normal_gap_clash_weight: float = DEFAULT_NORMAL_GAP_CLASH_WEIGHT,
    normal_gap_far_weight: float = DEFAULT_NORMAL_GAP_FAR_WEIGHT,
    normal_gap_contact_scale: float = DEFAULT_NORMAL_GAP_CONTACT_SCALE,
    score_mode: str = NORMAL_GAP_FIELD_SCORE,
) -> float:
    """Score good-contact signal against Zernike-smoothed clash and far-gap fields."""
    diagnostics = zernike_normal_gap_score_diagnostics(
        coeffs,
        order=order,
        fit_order=fit_order,
        normal_gap_clash_weight=normal_gap_clash_weight,
        normal_gap_far_weight=normal_gap_far_weight,
        normal_gap_contact_scale=normal_gap_contact_scale,
    )
    if score_mode == NORMAL_GAP_CONTACT_SCORE:
        return float(diagnostics["normal_gap_contact_score_from_fields"])
    return float(diagnostics["normal_gap_score_from_fields"])


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
    gap_band_soft_width: float = DEFAULT_GAP_BAND_SOFT_WIDTH,
    gap_contact_scale: float = DEFAULT_GAP_CONTACT_SCALE,
) -> float:
    fit_to = fit_order_value(int(order), fit_order)
    if score_mode in GRID_SCORE_MODES:
        (
            gap_coeff,
            grid_overlap,
            side1_effective_volume,
            side2_effective_volume,
            voxel_count,
        ) = zernike_gap_coefficient_bundle_from_grids(grid1, grid2, fit_to)
        return zernike_score_from_gap_coefficients(
            gap_coeff,
            grid_overlap,
            order=order,
            score_mode=score_mode,
            fit_order=fit_to,
            order_decay_n0=order_decay_n0,
            side1_effective_volume=side1_effective_volume,
            side2_effective_volume=side2_effective_volume,
            voxel_count=voxel_count,
            gap_band_soft_width=gap_band_soft_width,
            gap_contact_scale=gap_contact_scale,
        )

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
    normal_gap_good_scale: float = DEFAULT_NORMAL_GAP_GOOD_SCALE,
    normal_gap_far_scale: float = DEFAULT_NORMAL_GAP_FAR_SCALE,
    normal_gap_clash_weight: float = DEFAULT_NORMAL_GAP_CLASH_WEIGHT,
    normal_gap_far_weight: float = DEFAULT_NORMAL_GAP_FAR_WEIGHT,
    normal_gap_contact_scale: float = DEFAULT_NORMAL_GAP_CONTACT_SCALE,
    gap_band_soft_width: float = DEFAULT_GAP_BAND_SOFT_WIDTH,
    gap_contact_scale: float = DEFAULT_GAP_CONTACT_SCALE,
) -> float:
    """
    Compare low-pass 3D Zernike interface descriptors.

    Returns 0.0 when no interface can be voxelized robustly.
    """
    try:
        if representation in NORMAL_GAP_REPRESENTATIONS or score_mode in NORMAL_GAP_SCORE_MODES:
            bundle = zernike_normal_gap_field_bundle(
                residues1,
                residues2,
                distance=distance,
                grid_size=grid_size,
                sigma=sigma,
                padding=padding,
                surface_density=surface_density,
                surface_trim_cutoff=surface_trim_cutoff,
                surface_probe_radius=surface_probe_radius,
                normal_gap_good_scale=normal_gap_good_scale,
                normal_gap_far_scale=normal_gap_far_scale,
            )
            return zernike_score_from_normal_gap_fields(
                bundle,
                order=order,
                fit_order=fit_order,
                normal_gap_clash_weight=normal_gap_clash_weight,
                normal_gap_far_weight=normal_gap_far_weight,
                normal_gap_contact_scale=normal_gap_contact_scale,
                score_mode=score_mode,
            )

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
            gap_band_soft_width=gap_band_soft_width,
            gap_contact_scale=gap_contact_scale,
        )
    except Exception:
        return 0.0


__all__ = [
    "ATOM_GAUSSIAN",
    "DEFAULT_DISTANCE",
    "DEFAULT_GAP_BAND_SOFT_WIDTH",
    "DEFAULT_GAP_CONTACT_SCALE",
    "DEFAULT_GRID_SIZE",
    "DEFAULT_ORDER",
    "DEFAULT_ORDER_DECAY_N0",
    "DEFAULT_NORMAL_GAP_CLASH_WEIGHT",
    "DEFAULT_NORMAL_GAP_CONTACT_SCALE",
    "DEFAULT_NORMAL_GAP_FAR_SCALE",
    "DEFAULT_NORMAL_GAP_FAR_WEIGHT",
    "DEFAULT_NORMAL_GAP_GOOD_SCALE",
    "DEFAULT_PADDING",
    "DEFAULT_PROXIMITY_LENGTH_SCALE",
    "DEFAULT_REPRESENTATION",
    "DEFAULT_SC_RESCUE_FLOOR",
    "DEFAULT_SC_RESCUE_SCALE",
    "DEFAULT_SC_RESCUE_WEIGHT",
    "DEFAULT_SCORE_MODE",
    "DEFAULT_SIGMA",
    "DEFAULT_SURFACE_DENSITY",
    "DEFAULT_SURFACE_PROBE_RADIUS",
    "DEFAULT_SURFACE_TRIM_CUTOFF",
    "GAUSSIAN_REPRESENTATIONS",
    "GAUSSIAN_WEIGHTED_SCORE",
    "GAP_SCORE_MODES",
    "GAP_ZERNIKE_BANDPASS_SCORE",
    "GAP_ZERNIKE_EXCESS_BANDPASS_SCORE",
    "GAP_ZERNIKE_EXCESS_CONTACT_SCORE",
    "GAP_ZERNIKE_NONUNIFORM_SCORE",
    "GAP_ZERNIKE_RATIO_SCORE",
    "GAP_ZERNIKE_SOFT_BANDPASS_SCORE",
    "GAP_ZERNIKE_WEIGHTED_SCORE",
    "GRID_SCORE_MODES",
    "HARD_CUTOFF_SCORE",
    "JOINT_LOW_ORDER_RATIO_SCORE",
    "JOINT_REPRESENTATIONS",
    "JOINT_RESIDUE_BEAD_GAUSSIAN",
    "JOINT_SURFACE_GAUSSIAN",
    "NORMAL_GAP_FIELD_SCORE",
    "NORMAL_GAP_CONTACT_SCORE",
    "NORMAL_GAP_REPRESENTATIONS",
    "NORMAL_GAP_SCORE_MODES",
    "PER_SIDE_REPRESENTATIONS",
    "RESIDUE_BEAD_GAUSSIAN",
    "SC_GATED_GAP_BANDPASS_SCORE",
    "SC_GATED_GAP_NONUNIFORM_SCORE",
    "SC_GATED_GAP_SOFT_SCORE",
    "SC_GATED_OVERLAP_SCORE",
    "SC_GATED_SCORE_MODES",
    "SURFACE_BINARY",
    "SURFACE_GAUSSIAN",
    "SURFACE_NORMAL_GAP",
    "SURFACE_PROXIMITY_GAUSSIAN",
    "SURFACE_REPRESENTATIONS",
    "SHARED_GRID_OVERLAP_SCORE",
    "NormalGapCoefficientBundle",
    "NormalGapFieldBundle",
    "WeightedPointCloud",
    "ZernikeSpec",
    "fit_order_value",
    "zernike_candidate_family",
    "zernike_band_energy_ratio",
    "zernike_coefficient_bundle_from_grids",
    "zernike_coefficients",
    "zernike_descriptor_orders",
    "zernike_descriptor_prefix_length",
    "zernike_descriptors",
    "zernike_descriptors_from_grids",
    "zernike_effective_grid_volume",
    "zernike_excess_grid_overlap",
    "zernike_expected_random_overlap",
    "zernike_gap_coefficient_bundle_from_grids",
    "zernike_gap_grid",
    "zernike_gap_score_diagnostics",
    "zernike_grids",
    "zernike_grids_from_point_clouds",
    "zernike_joint_grid",
    "zernike_normal_gap_coefficient_bundle",
    "zernike_normal_gap_field_bundle",
    "zernike_normal_gap_score_diagnostics",
    "zernike_order_weights",
    "zernike_point_clouds",
    "zernike_score_from_coefficients",
    "zernike_score_from_gap_coefficients",
    "zernike_score_from_grids",
    "zernike_score_from_normal_gap_coefficients",
    "zernike_score_from_normal_gap_fields",
    "zernike_shape_complementarity",
    "zernike_shared_grid_overlap",
    "zernike_similarity_from_coefficients",
    "zernike_similarity_from_grids",
    "zernike_source_representation",
    "zernike_soft_band_energy_ratio",
    "zernike_weighted_energy_ratio",
    "zernike_l2_normalized_grid",
    "zernike_low_order_energy_ratio",
]
