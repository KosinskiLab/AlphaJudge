from __future__ import annotations

import logging
import math
from functools import cached_property
from typing import Any, Dict, List, Tuple

import numpy as np
from Bio.PDB import Chain, NeighborSearch

from .confidence import Confidence
from .docking_scores import MPDOCKQ
from .geometry import is_pae_token_residue, representative_atom
from .interface import Interface


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

        self._contact_ns_cache: Dict[Tuple[str, str], Tuple[list, list, np.ndarray, np.ndarray]] = {}
        self._all_atom_ns_cache: Dict[Tuple[str, str], Tuple[list, NeighborSearch]] = {}
        self._token_dist_cache: Dict[Tuple[str, str], np.ndarray] = {}

        self.interfaces: List[Interface] = []
        for i in range(len(self._chains)):
            for j in range(i + 1, len(self._chains)):
                iface = Interface(self._chains[i], self._chains[j], self)
                if iface.num_intf_residues > 0:
                    self.interfaces.append(iface)

    def _build_maps(self) -> tuple[Dict[Tuple[str, Any], int], Dict[str, List[int]], List[Chain.Chain]]:
        model = next(self.structure.get_models())
        chains = list(model.get_chains())

        res_index_map: Dict[Tuple[str, Any], int] = {}
        chain_indices_by_id: Dict[str, List[int]] = {}
        filtered_chains: List[Chain.Chain] = []

        idx = 0
        for chain in chains:
            kept = [res for res in chain if is_pae_token_residue(res)]
            if not kept:
                continue

            new_chain = Chain.Chain(chain.id)
            for residue in kept:
                new_chain.add(residue.copy())
            filtered_chains.append(new_chain)

            idxs: List[int] = []
            for residue in new_chain:
                res_index_map[(new_chain.id, residue.id)] = idx
                idxs.append(idx)
                idx += 1
            chain_indices_by_id[new_chain.id] = idxs

        pae_n = len(self.conf.pae_matrix)
        if idx != pae_n:
            logging.warning(
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
        reps = []
        for chain in self._chains:
            for residue in chain:
                try:
                    reps.append((representative_atom(residue), residue))
                except Exception:
                    continue
        if not reps:
            return 0

        ns = NeighborSearch([atom for atom, _ in reps])
        seen, count = set(), 0
        for atom1, residue1 in reps:
            for atom2 in ns.search(atom1.coord, self.contact_thresh):
                if atom2 is atom1:
                    continue
                residue2 = atom2.get_parent()
                chain1 = residue1.get_parent().id
                chain2 = residue2.get_parent().id
                if chain1 == chain2:
                    continue
                key = tuple(sorted([(chain1, residue1.id), (chain2, residue2.id)]))
                if key not in seen:
                    seen.add(key)
                    count += 1
        return count

    @cached_property
    def mpDockQ(self) -> float:
        if self.num_chains <= 2:
            return float("nan")
        x = self.average_interface_plddt * math.log10(self.contact_pairs_global + 1)
        return MPDOCKQ.score(x)
