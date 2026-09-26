from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# Values of Confidence.global_confidence_scope (and the CSV column of that
# name): whether the global scores (ptm, iptm, iptm_ptm, confidence_score)
# cover only the scored residues or also tokens excluded from interface
# scoring, such as ligands, ions and AF3x crosslinkers.
SCOPE_SCORED_RESIDUES = "scored_residues"
SCOPE_INCLUDES_EXCLUDED_TOKENS = "includes_excluded_tokens"
SCOPE_UNKNOWN = "unknown"
# Values of the iptm_scope column: whether a row's iptm is the chain-pair
# value or the global fallback.
IPTM_SCOPE_CHAIN_PAIR = "chain_pair"
IPTM_SCOPE_GLOBAL = "global"


@dataclass(frozen=True)
class Confidence:
    pae_matrix: np.ndarray  # Keep as numpy array for memory efficiency (7-10x reduction)
    max_pae: float
    iptm: float | None
    ptm: float | None
    iptm_ptm: float | None
    confidence_score: float | None
    plddt_residue: list[float]
    # Per-chain-pair ipTM matrix, read through pair_iptm(). When present, use
    # it for per-interface iptm instead of global iptm.
    chain_pair_iptm: list[list[float]] | None = None
    contact_prob_matrix: np.ndarray | None = None
    contact_prob_source: str | None = None
    # Chain order of chain_pair_iptm as the source wrote it, including
    # ligand-only chains (recorded by the AF3 and Boltz-2 parsers).
    chain_pair_iptm_chain_ids: list[str] | None = None
    # AF3 global scores can include tokens excluded from interface scoring.
    global_confidence_scope: str = SCOPE_UNKNOWN

    def pair_iptm(
        self, chain_a: str, chain_b: str, default_chain_ids: Sequence[str]
    ) -> float | None:
        """ipTM of one chain pair from ``chain_pair_iptm``, or None if unavailable.

        The matrix is indexed in ``chain_pair_iptm_chain_ids`` order. A
        Confidence that does not record the order (one built directly) is taken
        to index it by ``default_chain_ids``.
        """
        matrix = self.chain_pair_iptm
        if matrix is None or not matrix:
            return None
        chain_ids = self.chain_pair_iptm_chain_ids
        if chain_ids is None:
            chain_ids = default_chain_ids
        try:
            i, j = chain_ids.index(chain_a), chain_ids.index(chain_b)
        except ValueError:
            return None
        # Read [i][j], or [j][i] when a ragged matrix lacks the former.
        for row, col in ((i, j), (j, i)):
            try:
                entry = matrix[row]
                value = entry[col] if isinstance(entry, (list, tuple)) else math.nan
                break
            except (IndexError, TypeError):
                continue
        else:
            return None
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None
