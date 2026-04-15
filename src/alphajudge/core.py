"""Backward-compatible facade for the split AlphaJudge core modules."""

from __future__ import annotations

from .complex import Complex
from .confidence import Confidence
from .docking_scores import D0, MPDOCKQ, PDOCKQ, PDOCKQ2, DockQParams, _sigmoid
from .geometry import (
    CHARGED_ATOMS,
    CHARGED_RES,
    HYDROPHOBIC_RES,
    NA_RES,
    POLAR_RES,
    _repr_atom,
    is_pae_token_residue,
    representative_atom,
)
from .interface import Interface

__all__ = [
    "CHARGED_ATOMS",
    "CHARGED_RES",
    "Complex",
    "Confidence",
    "D0",
    "DockQParams",
    "HYDROPHOBIC_RES",
    "Interface",
    "MPDOCKQ",
    "NA_RES",
    "PDOCKQ",
    "PDOCKQ2",
    "POLAR_RES",
    "_repr_atom",
    "_sigmoid",
    "is_pae_token_residue",
    "representative_atom",
]
