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

    forward, reverse = np.isfinite(arr), np.isfinite(arr.T)
    both = forward & reverse
    max_delta = float(np.max(np.abs(arr[both] - arr.T[both]))) if np.any(both) else 0.0

    sym = np.full(arr.shape, np.nan, dtype=float)
    sym[both] = 0.5 * (arr[both] + arr.T[both])
    sym[forward & ~reverse] = arr[forward & ~reverse]
    sym[reverse & ~forward] = arr.T[reverse & ~forward]
    return sym, max_delta


def summarize_contact_prob_block(
    matrix,
    idx1,
    idx2,
    *,
    top_n: int = 10,
) -> tuple[float, float, float]:
    """Return max, top-N mean, and expected contacts for one inter-chain block."""
    missing = (float("nan"),) * 3
    if matrix is None:
        return missing

    idx1_arr = np.asarray(idx1, dtype=int)
    idx2_arr = np.asarray(idx2, dtype=int)
    if idx1_arr.size == 0 or idx2_arr.size == 0:
        return missing

    try:
        vals = np.asarray(matrix, dtype=float)[np.ix_(idx1_arr, idx2_arr)].ravel()
    except IndexError:
        return missing

    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return missing

    max_prob = float(np.max(vals))
    n = min(int(top_n), vals.size)
    top_vals = np.partition(vals, vals.size - n)[-n:]
    top_mean = float(np.mean(top_vals))
    expected_contacts = float(np.sum(vals))
    return max_prob, top_mean, expected_contacts
