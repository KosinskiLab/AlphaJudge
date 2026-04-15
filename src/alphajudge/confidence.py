from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Confidence:
    pae_matrix: List[List[float]]
    max_pae: float
    iptm: Optional[float]
    ptm: Optional[float]
    iptm_ptm: Optional[float]
    confidence_score: Optional[float]
    plddt_residue: List[float]
    # AF3 only: per-chain-pair ipTM matrix (indexed by chain order).
    # When present, use this for per-interface iptm instead of global iptm.
    chain_pair_iptm: Optional[List[List[float]]] = None
