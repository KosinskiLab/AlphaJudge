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

    AF2 distogram ``bin_edges`` are finite upper bin boundaries. For scoring,
    use the lower-bound convention used by AF2 contact-probability analyses:
    append a 0 A lower bound and include bins whose lower bound is below the
    cutoff. This includes the bin that straddles 8 A in the standard AF2
    distogram, instead of dropping it because its upper edge is slightly above
    8 A.
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

    work = (
        logits_arr
        if np.issubdtype(logits_arr.dtype, np.floating)
        else logits_arr.astype(np.float32)
    )

    lower_bounds = np.concatenate([[0.0], edges])
    clipped_cutoff = float(np.clip(float(contact_cutoff), 3.0, 20.0))
    n_contact_bins = int(np.count_nonzero(lower_bounds < clipped_cutoff))
    n_contact_bins = max(1, min(n_contact_bins, work.shape[-1]))

    max_logit = np.max(work, axis=-1)
    dtype = np.result_type(work.dtype, np.float32)
    numerator = np.zeros(max_logit.shape, dtype=dtype)
    denominator = np.zeros(max_logit.shape, dtype=dtype)
    for bin_idx in range(work.shape[-1]):
        delta = (work[..., bin_idx] - max_logit).astype(dtype, copy=False)
        bin_mass = np.exp(delta)
        denominator += bin_mass
        if bin_idx < n_contact_bins:
            numerator += bin_mass
    return numerator / denominator


def summarize_contact_prob_block(
    matrix: np.ndarray | None,
    idx1: np.ndarray,
    idx2: np.ndarray,
    top_n: int = 10,
) -> tuple[float, float]:
    """Summarize one inter-chain block as max and top-N mean."""
    if matrix is None or idx1.size == 0 or idx2.size == 0:
        nan = float("nan")
        return nan, nan

    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2:
        nan = float("nan")
        return nan, nan
    if idx1.max(initial=-1) >= m.shape[0] or idx2.max(initial=-1) >= m.shape[1]:
        nan = float("nan")
        return nan, nan

    block = m[np.ix_(idx1, idx2)].ravel()
    if (
        m.shape[0] == m.shape[1]
        and idx2.max(initial=-1) < m.shape[0]
        and idx1.max(initial=-1) < m.shape[1]
    ):
        reverse = m[np.ix_(idx2, idx1)].T.ravel()
        both = np.isfinite(block) & np.isfinite(reverse)
        only_reverse = ~np.isfinite(block) & np.isfinite(reverse)
        if np.any(both):
            block[both] = 0.5 * (block[both] + reverse[both])
        if np.any(only_reverse):
            block[only_reverse] = reverse[only_reverse]

    finite = block[np.isfinite(block)]
    if finite.size == 0:
        nan = float("nan")
        return nan, nan

    n = max(1, min(int(top_n), finite.size))
    top = np.partition(finite, finite.size - n)[-n:]
    return float(np.max(finite)), float(np.mean(top))
