from __future__ import annotations

import numpy as np

AF2_DISTOGRAM_CONTACT_CUTOFF = 8.0


def contact_probs_from_distogram(
    logits: np.ndarray,
    bin_edges: np.ndarray,
    contact_cutoff: float = AF2_DISTOGRAM_CONTACT_CUTOFF,
) -> np.ndarray:
    """
    Convert AF2 distogram logits to P(distance < contact_cutoff).

    This follows the AlphaPulldown diagnostics fallback convention: append an
    infinite final upper bound, clip the requested cutoff to 3-20 A, softmax
    over bins, then sum bins whose upper bound is strictly below the cutoff.
    """
    logits_arr = np.asarray(logits)
    if logits_arr.ndim != 3:
        raise ValueError(f"distogram logits must be 3D, got shape {logits_arr.shape}")

    edges = np.asarray(bin_edges, dtype=float).ravel()
    if edges.size != logits_arr.shape[-1] - 1:
        raise ValueError(
            f"distogram bin_edges length {edges.size} does not match logits bins "
            f"{logits_arr.shape[-1]}"
        )

    dtype = logits_arr.dtype if np.issubdtype(logits_arr.dtype, np.floating) else np.float32
    work = logits_arr.astype(dtype, copy=False)
    shifted = work - np.max(work, axis=-1, keepdims=True)
    np.exp(shifted, out=shifted)

    upper_bounds = np.concatenate([edges, [np.inf]])
    clipped_cutoff = float(np.clip(float(contact_cutoff), 3.0, 20.0))
    contact_bins = upper_bounds < clipped_cutoff
    if not np.any(contact_bins):
        contact_bins[0] = True

    numerator = shifted[..., contact_bins].sum(axis=-1)
    denominator = shifted.sum(axis=-1)
    return numerator / denominator


def symmetrize_contact_probs(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Return a nan-aware symmetric copy and the maximum pre-symmetry delta."""
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"contact probability matrix must be square, got {m.shape}")

    mt = m.T
    sym = (m + mt) / 2.0

    left_nan = np.isnan(m)
    right_nan = np.isnan(mt)
    sym[left_nan & ~right_nan] = mt[left_nan & ~right_nan]
    sym[right_nan & ~left_nan] = m[right_nan & ~left_nan]
    sym[left_nan & right_nan] = np.nan

    delta = np.abs(m - mt)
    finite = np.isfinite(delta)
    max_delta = float(np.nanmax(delta[finite])) if np.any(finite) else 0.0
    return sym, max_delta


def summarize_contact_prob_block(
    matrix: np.ndarray | None,
    idx1: np.ndarray,
    idx2: np.ndarray,
    top_n: int = 10,
) -> tuple[float, float, float]:
    """Summarize inter-chain contact probabilities as max, top-N mean, and sum."""
    if matrix is None or idx1.size == 0 or idx2.size == 0:
        nan = float("nan")
        return nan, nan, nan

    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2:
        nan = float("nan")
        return nan, nan, nan
    if idx1.max(initial=-1) >= m.shape[0] or idx2.max(initial=-1) >= m.shape[1]:
        nan = float("nan")
        return nan, nan, nan

    block = m[np.ix_(idx1, idx2)].ravel()
    finite = block[np.isfinite(block)]
    if finite.size == 0:
        nan = float("nan")
        return nan, nan, nan

    n = max(1, min(int(top_n), finite.size))
    top = np.partition(finite, finite.size - n)[-n:]
    return float(np.max(finite)), float(np.mean(top)), float(np.sum(finite))
