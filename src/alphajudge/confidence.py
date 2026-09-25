from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Confidence:
    pae_matrix: np.ndarray  # Keep as numpy array for memory efficiency (7-10x reduction)
    max_pae: float
    iptm: float | None
    ptm: float | None
    iptm_ptm: float | None
    confidence_score: float | None
    plddt_residue: list[float]
    # Per-chain-pair ipTM matrix (AF3 records its original chain order below).
    # When present, use this for per-interface iptm instead of global iptm.
    chain_pair_iptm: list[list[float]] | None = None
    contact_prob_matrix: np.ndarray | None = None
    contact_prob_source: str | None = None
    # Original order in the summary matrix, including ligand-only chains.
    chain_pair_iptm_chain_ids: list[str] | None = None
