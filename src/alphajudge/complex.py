from __future__ import annotations

import logging
import math
from functools import cached_property
from typing import Any

import numpy as np
from Bio.PDB import Chain

from .confidence import Confidence
from .docking_scores import MPDOCKQ
from .geometry import is_pae_token_residue
from .interface import Interface

logger = logging.getLogger(__name__)


class Complex:
    """
    contact_thresh (Angstrom):
      Used to define interface residue-residue contacts via representative atoms.
      Affects Interface contact counts/scores and Complex mpDockQ.
    """
    def __init__(
        self,
        structure,
        confidence: Confidence,
        contact_thresh: float,
        pae_filter: float,
        ipsae_pae_cutoff: float | None = None,
    ):
        self.structure = structure
        self.conf = confidence
        self.contact_thresh = float(contact_thresh)
        self.pae_filter = float(pae_filter)
        self.ipsae_pae_cutoff = None if ipsae_pae_cutoff is None else float(ipsae_pae_cutoff)

        self._res_index_map, self._chain_indices_by_id, self._chains = self._build_maps()

        self._contact_ns_cache: dict[tuple[str, str], tuple[list, list, np.ndarray, np.ndarray]] = {}

        self.interfaces: list[Interface] = []
        for i in range(len(self._chains)):
            for j in range(i + 1, len(self._chains)):
                iface = Interface(self._chains[i], self._chains[j], self)
                if iface.num_intf_residues > 0:
                    self.interfaces.append(iface)

    def _build_maps(self) -> tuple[dict[tuple[str, Any], int], dict[str, list[int]], list[Chain.Chain]]:
        model = next(self.structure.get_models())
        chains = list(model.get_chains())

        res_index_map: dict[tuple[str, Any], int] = {}
        chain_indices_by_id: dict[str, list[int]] = {}
        filtered_chains: list[Chain.Chain] = []

        idx = 0
        for chain in chains:
            kept = [res for res in chain if is_pae_token_residue(res)]
            if not kept:
                continue

            new_chain = Chain.Chain(chain.id)
            for residue in kept:
                new_chain.add(residue.copy())
            filtered_chains.append(new_chain)

            idxs: list[int] = []
            for residue in new_chain:
                res_index_map[(new_chain.id, residue.id)] = idx
                idxs.append(idx)
                idx += 1
            chain_indices_by_id[new_chain.id] = idxs

        pae_n = len(self.conf.pae_matrix)
        if idx != pae_n:
            logger.warning(
                f"token residues counted = {idx}, but PAE is {pae_n}x{pae_n}. "
                f"Indexing may be misaligned for this structure."
            )

        return res_index_map, chain_indices_by_id, filtered_chains

    @property
    def num_chains(self) -> int:
        return len(self._chains)

    @cached_property
    def average_interface_pae(self) -> float:
        vals = [
            i.average_interface_pae for i in self.interfaces
            if not math.isnan(i.average_interface_pae) and i.average_interface_pae <= self.pae_filter
        ]
        return sum(vals) / len(vals) if vals else float(0.0)

    @cached_property
    def average_interface_plddt(self) -> float:
        vals = [i.average_interface_plddt for i in self.interfaces]
        return sum(vals) / len(vals) if vals else float(0.0)

    @cached_property
    def contact_pairs_global(self) -> int:
        return sum(iface.contact_pairs for iface in self.interfaces)

    @cached_property
    def mpDockQ(self) -> float:
        if self.num_chains <= 2:
            return float("nan")
        x = self.average_interface_plddt * math.log10(self.contact_pairs_global + 1)
        return MPDOCKQ.score(x)
