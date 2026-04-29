from __future__ import annotations

import math
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
from Bio.PDB import NeighborSearch

from .biophysics import (
    buried_surface_area as _pisa_buried_surface_area,
    disulfide_bonds as _pisa_disulfide_bonds,
    hydrogen_bonds as _pisa_hydrogen_bonds,
    interface_solvation_energy as _pisa_interface_solvation_energy,
    salt_bridges as _pisa_salt_bridges,
    shape_complementarity as _scasa_sc,
    zernike_shape_complementarity as _zernike_sc,
)
from .docking_scores import D0, PDOCKQ, PDOCKQ2
from .geometry import (
    CHARGED_RES,
    HYDROPHOBIC_RES,
    NA_RES,
    POLAR_RES,
    representative_atom,
)

if TYPE_CHECKING:
    from .complex import Complex


class Interface:
    def __init__(self, chain1, chain2, complex_ctx: Complex):
        self.c = complex_ctx
        self.chain1 = list(chain1)
        self.chain2 = list(chain2)

        # Always set shared refs first
        self._pae = self.c.conf.pae_matrix
        self._rim = self.c._res_index_map
        self._cid = self.c._chain_indices_by_id

        if not self.chain1 or not self.chain2:
            self._cid1_id = ""
            self._cid2_id = ""
            self._idx1 = np.array([], dtype=int)
            self._idx2 = np.array([], dtype=int)
            self._has_na = False
            self._res1, self._res2, self._pairs = set(), set(), set()
            self._avg_plddt = 0.0
            self._avg_pae = 0.0
            return

        self._cid1_id = self.chain1[0].get_parent().id
        self._cid2_id = self.chain2[0].get_parent().id
        self._idx1 = np.asarray(self._cid.get(self._cid1_id, []), dtype=int)
        self._idx2 = np.asarray(self._cid.get(self._cid2_id, []), dtype=int)

        self._has_na = any(
            r.get_resname().strip().upper() in NA_RES for r in (self.chain1 + self.chain2)
        )

        self._res1, self._res2, self._pairs = self._get_pairs()
        self._avg_plddt = self._avg_plddt_union()
        self._avg_pae = self._avg_pae_over_pairs()

    @property
    def num_intf_residues(self) -> int:
        return len(self._res1 | self._res2)

    @cached_property
    def average_interface_plddt(self) -> float:
        return self._avg_plddt

    @cached_property
    def average_interface_pae(self) -> float:
        return self._avg_pae

    @cached_property
    def iptm_chainpair(self) -> float | None:
        """
        Per-interface ipTM from AF3 chain_pair_iptm when available.
        Returns None for AF2 (no per-interface ipTM).
        """
        cpi = getattr(self.c.conf, "chain_pair_iptm", None)
        if cpi is None or not cpi:
            return None
        chain_ids = [ch.id for ch in self.c._chains]
        try:
            i = chain_ids.index(self._cid1_id)
            j = chain_ids.index(self._cid2_id)
        except ValueError:
            return None
        try:
            row = cpi[i]
            val = row[j] if isinstance(row, (list, tuple)) else float("nan")
        except (IndexError, TypeError):
            try:
                row = cpi[j]
                val = row[i] if isinstance(row, (list, tuple)) else float("nan")
            except (IndexError, TypeError):
                return None
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    @cached_property
    def contact_pairs(self) -> int:
        return len(self._pairs)

    @cached_property
    def pDockQ(self) -> float:
        if self.contact_pairs <= 0 or math.isnan(self._avg_plddt):
            return 0.0
        return PDOCKQ.score(self._avg_plddt * math.log10(self.contact_pairs))

    def _mean_ptm_dir(self, reverse: bool) -> float:
        vals = []
        for r1, r2 in self._pairs:
            i = self._rim.get((r1.get_parent().id, r1.id))
            j = self._rim.get((r2.get_parent().id, r2.id))
            if i is None or j is None:
                continue
            pae = float(self._pae[j, i] if reverse else self._pae[i, j])
            vals.append(1.0 / (1.0 + (pae / D0) ** 2))
        return float(np.mean(vals)) if vals else float("nan")

    def pDockQ2(self) -> tuple[float, float]:
        """
        Return (score_max, mean_ptm_for_direction_that_won).
        Returns (0.0, 0.0) when interface is not found.
        """
        if self.contact_pairs <= 0 or math.isnan(self._avg_plddt):
            return 0.0, 0.0

        m_ab = self._mean_ptm_dir(reverse=False)
        m_ba = self._mean_ptm_dir(reverse=True)

        s_ab = PDOCKQ2.score(self._avg_plddt * m_ab) if not math.isnan(m_ab) else float("nan")
        s_ba = PDOCKQ2.score(self._avg_plddt * m_ba) if not math.isnan(m_ba) else float("nan")

        if math.isnan(s_ab) and math.isnan(s_ba):
            return 0.0, 0.0
        if math.isnan(s_ba) or (not math.isnan(s_ab) and s_ab >= s_ba):
            return s_ab, (0.0 if math.isnan(m_ab) else m_ab)
        return s_ba, (0.0 if math.isnan(m_ba) else m_ba)

    def ipsae(self, pae_cutoff=10.0) -> float:
        """
        Interface pTM-based Surface Accuracy Estimation (ipSAE).
        """
        if getattr(self.c, "ipsae_pae_cutoff", None) is not None:
            pae_cutoff = float(self.c.ipsae_pae_cutoff)
        return self._ipsae_asym(float(pae_cutoff))

    def lis(self) -> float:
        """Returns 0.0 when interface is not found or no valid PAE pairs."""
        def _lis_dir(idx_src: np.ndarray, idx_dst: np.ndarray) -> float:
            if idx_src.size == 0 or idx_dst.size == 0:
                return 0.0
            sub = self._pae[np.ix_(idx_src, idx_dst)].ravel()
            valid = sub[sub < 12.0]
            if valid.size == 0:
                return 0.0
            return float(np.mean((12.0 - valid) / 12.0))

        a = _lis_dir(self._idx1, self._idx2)
        b = _lis_dir(self._idx2, self._idx1)
        return float(0.5 * (a + b))

    @cached_property
    def polar(self) -> float:
        return self._frac(POLAR_RES)

    @cached_property
    def hydrophobic(self) -> float:
        return self._frac(HYDROPHOBIC_RES)

    @cached_property
    def charged(self) -> float:
        return self._frac(CHARGED_RES)

    @cached_property
    def score_complex(self) -> float:
        if self.contact_pairs <= 0 or math.isnan(self._avg_plddt):
            return float("nan")
        return self._avg_plddt * math.log10(self.contact_pairs)

    @cached_property
    def hb(self) -> int:
        return _pisa_hydrogen_bonds(self.chain1, self.chain2)

    @cached_property
    def sb(self) -> int:
        return _pisa_salt_bridges(self.chain1, self.chain2)

    @cached_property
    def ss(self) -> int:
        return _pisa_disulfide_bonds(self.chain1, self.chain2)

    @cached_property
    def sc(self) -> float:
        return _scasa_sc(self.chain1, self.chain2)

    @cached_property
    def zernike_sc(self) -> float:
        return _zernike_sc(self.chain1, self.chain2, distance=self.c.contact_thresh)

    @cached_property
    def int_area(self) -> float:
        return _pisa_buried_surface_area(self.chain1, self.chain2)

    @cached_property
    def int_solv_en(self) -> float:
        return _pisa_interface_solvation_energy(self.chain1, self.chain2)

    def _get_pairs(self):
        res_pairs: set[tuple[Any, Any]] = set()
        a1, a2, coords1, coords2 = self._contact_atom_data()
        if not len(a1) or not len(a2):
            return set(), set(), res_pairs

        diff = coords1[:, None, :] - coords2[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        mask = dist2 <= (self.c.contact_thresh ** 2)
        idx_i, idx_j = np.where(mask)

        for i, j in zip(idx_i.tolist(), idx_j.tolist()):
            res_pairs.add((a1[i].get_parent(), a2[j].get_parent()))

        r1 = {p[0] for p in res_pairs}
        r2 = {p[1] for p in res_pairs}
        return r1, r2, res_pairs

    def _contact_atom_data(self) -> tuple[list, list, np.ndarray, np.ndarray]:
        key = tuple(sorted((self._cid1_id, self._cid2_id)))
        cached = self.c._contact_ns_cache.get(key)
        if cached is None:
            a1, a2 = [], []
            for residue in self.chain1:
                try:
                    a1.append(representative_atom(residue))
                except Exception:
                    pass
            for residue in self.chain2:
                try:
                    a2.append(representative_atom(residue))
                except Exception:
                    pass

            coords1 = np.array([a.coord for a in a1], dtype=float) if a1 else np.empty((0, 3))
            coords2 = np.array([a.coord for a in a2], dtype=float) if a2 else np.empty((0, 3))
            cached = (a1, a2, coords1, coords2)
            self.c._contact_ns_cache[key] = cached
        return cached

    def _avg_plddt_union(self) -> float:
        res_set = self._res1 | self._res2
        if not res_set:
            return float("nan")

        vals = []
        for residue in res_set:
            try:
                vals.append(float(representative_atom(residue).get_bfactor()))
            except Exception:
                continue
        return float(sum(vals) / len(vals)) if vals else float("nan")

    def _avg_pae_over_pairs(self) -> float:
        vals = []
        for r1, r2 in self._pairs:
            i = self._rim.get((r1.get_parent().id, r1.id))
            j = self._rim.get((r2.get_parent().id, r2.id))
            if i is None or j is None:
                continue
            try:
                vals.append(float(self._pae[i, j]))
                vals.append(float(self._pae[j, i]))
            except Exception:
                continue
        return sum(vals) / len(vals) if vals else float("nan")

    def _ipsae_asym(self, cutoff: float) -> float:
        def calc(idx_src: np.ndarray, idx_dst: np.ndarray) -> float:
            if idx_src.size == 0 or idx_dst.size == 0:
                return 0.0

            min_d0 = 2.0 if self._has_na else 1.0
            sub = self._pae[np.ix_(idx_src, idx_dst)]   # (n_src, n_dst)
            valid = sub < cutoff                          # (n_src, n_dst)
            n_valid = valid.sum(axis=1)                   # (n_src,)
            has_valid = n_valid > 0
            if not has_valid.any():
                return 0.0

            length = np.maximum(27.0, n_valid.astype(float))
            d0 = np.maximum(min_d0, 1.24 * (length - 15.0) ** (1.0 / 3.0) - 1.8)
            ptm_all = 1.0 / (1.0 + (sub / d0[:, None]) ** 2)
            row_scores = np.where(
                has_valid,
                np.where(valid, ptm_all, 0.0).sum(axis=1) / np.maximum(n_valid, 1),
                0.0,
            )
            return float(row_scores.max())

        return max(calc(self._idx1, self._idx2), calc(self._idx2, self._idx1))

    def _frac(self, names: set[str]) -> float:
        residues = self._res1 | self._res2
        if not residues:
            return 0.0
        return sum(1 for r in residues if r.get_resname().strip().upper() in names) / len(residues)
