from __future__ import annotations

import numpy as np


AF2_DISTOGRAM_CONTACT_CUTOFF = 8.0


def contact_probs_from_distogram(
    logits,
    bin_edges,
    contact_cutoff: float = AF2_DISTOGRAM_CONTACT_CUTOFF,
) -> np.ndarray:
    """
    Convert AlphaFold distogram logits into P(distance < contact_cutoff).

    AlphaFold distogram ``bin_edges`` are the finite upper bin boundaries and
    the final logit is the overflow bin above the last boundary. The bin mask
    mirrors AlphaPulldown diagnostics: upper bounds are filtered with a strict
    distance cutoff, clipped to the 3-20 A range used for contact maps.
    """
    logits_arr = np.asarray(logits)
    if not np.issubdtype(logits_arr.dtype, np.floating):
        logits_arr = logits_arr.astype(np.float32)
    edges = np.asarray(bin_edges, dtype=float).ravel()

    if logits_arr.ndim != 3:
        raise ValueError(f"distogram logits must be 3D, got shape {logits_arr.shape}")
    if edges.size != logits_arr.shape[-1] - 1:
        raise ValueError(
            "distogram bin_edges length must be one less than logits bins "
            f"(got {edges.size} edges for {logits_arr.shape[-1]} bins)"
        )

    clipped_cutoff = float(np.clip(float(contact_cutoff), 3.0, 20.0))
    upper_bounds = np.concatenate([edges, np.array([np.inf], dtype=float)])
    contact_bins = upper_bounds < clipped_cutoff
    if not np.any(contact_bins):
        contact_bins[0] = True

    shifted = logits_arr - np.nanmax(logits_arr, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted, out=shifted)
    numerator = np.sum(exp_logits[..., contact_bins], axis=-1)
    denominator = np.sum(exp_logits, axis=-1)
    return numerator / denominator


def symmetrize_contact_probs(matrix) -> tuple[np.ndarray, float]:
    """
    Return a symmetrized contact-probability matrix and max asymmetry.

    The contact relation is undirected. AF3 native contact_probs and AF2
    distogram logits should already be symmetric, but averaging makes downstream
    interface summaries robust to tiny numerical or exporter differences.
    """
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"contact probability matrix must be square, got {arr.shape}")

    finite = np.isfinite(arr) & np.isfinite(arr.T)
    max_delta = (
        float(np.max(np.abs(arr[finite] - arr.T[finite]))) if np.any(finite) else 0.0
    )

    both = np.isfinite(arr) & np.isfinite(arr.T)
    only_forward = np.isfinite(arr) & ~np.isfinite(arr.T)
    only_reverse = ~np.isfinite(arr) & np.isfinite(arr.T)
    sym = np.full(arr.shape, np.nan, dtype=float)
    sym[both] = 0.5 * (arr[both] + arr.T[both])
    sym[only_forward] = arr[only_forward]
    sym[only_reverse] = arr.T[only_reverse]
    return sym, max_delta


def summarize_contact_prob_block(
    matrix,
    idx1,
    idx2,
    *,
    top_n: int = 10,
) -> tuple[float, float, float]:
    """Return max, top-N mean, and expected contacts for one inter-chain block."""
    if matrix is None:
        nan = float("nan")
        return nan, nan, nan

    idx1_arr = np.asarray(idx1, dtype=int)
    idx2_arr = np.asarray(idx2, dtype=int)
    if idx1_arr.size == 0 or idx2_arr.size == 0:
        nan = float("nan")
        return nan, nan, nan

    arr = np.asarray(matrix, dtype=float)
    try:
        vals = arr[np.ix_(idx1_arr, idx2_arr)].ravel()
    except Exception:
        nan = float("nan")
        return nan, nan, nan

    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        nan = float("nan")
        return nan, nan, nan

    max_prob = float(np.max(vals))
    n = min(int(top_n), vals.size)
    top_vals = np.partition(vals, vals.size - n)[-n:]
    top_mean = float(np.mean(top_vals))
    expected_contacts = float(np.sum(vals))
    return max_prob, top_mean, expected_contacts
