from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Confidence:
    pae_matrix: list[list[float]]
    max_pae: float
    iptm: float | None
    ptm: float | None
    iptm_ptm: float | None
    confidence_score: float | None
    plddt_residue: list[float]
    # AF3 only: per-chain-pair ipTM matrix (indexed by chain order).
    # When present, use this for per-interface iptm instead of global iptm.
    chain_pair_iptm: list[list[float]] | None = None
