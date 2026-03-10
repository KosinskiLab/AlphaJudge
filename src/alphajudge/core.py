from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from Bio.PDB import Chain, Model, NeighborSearch, Structure
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.SASA import ShrakeRupley

# ---- residue type constants ----
NA_RES: Set[str] = {
    "A", "C", "G", "U",
    "DA", "DC", "DG", "DT", "DU",
    "RA", "RC", "RG", "RU",
}

POLAR_RES: Set[str] = {"SER", "THR", "ASN", "GLN", "TYR", "CYS"}
HYDROPHOBIC_RES: Set[str] = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP"}
CHARGED_RES: Set[str] = {"ARG", "LYS", "ASP", "GLU", "HIS"}

# Atoms that are part of charged groups for salt bridge calculations
CHARGED_ATOMS: Dict[str, Set[str]] = {
    "ARG": {"NE", "CZ", "NH1", "NH2"},
    "LYS": {"NZ"},
    "ASP": {"CG", "OD1", "OD2"},
    "GLU": {"CD", "OE1", "OE2"},
}

# ---- Shrake-Rupley helpers (speed) ----
_SR_TEMPLATE = ShrakeRupley(probe_radius=1.4, n_points=15)
_SPHERE_15 = np.array(_SR_TEMPLATE._sphere, copy=False)
_RADII_DICT_15 = dict(_SR_TEMPLATE.radii_dict)
_SR_PROBE_RADIUS = float(_SR_TEMPLATE.probe_radius)

# ---- confidence (unified) ----
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

# ---- dockQ constants ----
def _sigmoid(x: float, L: float, x0: float, k: float, b: float) -> float:
    return L / (1 + math.exp(-k * (x - x0))) + b

@dataclass(frozen=True)
class DockQParams:
    L: float
    X0: float
    K: float
    B: float

    def score(self, x: float) -> float:
        return _sigmoid(x, self.L, self.X0, self.K, self.B)

PDOCKQ  = DockQParams(0.724, 152.611, 0.052, 0.018)
PDOCKQ2 = DockQParams(1.31,   84.733,  0.075, 0.005)
MPDOCKQ = DockQParams(0.728, 309.375,  0.098, 0.262)
D0 = 10.0

# -------------------------
# Representative atom helper
# -------------------------
def _repr_atom(res):
    """
    Representative atom for distance / bfactor use.
    - Proteins: CB (else CA)
    - Nucleic acids: C1' / C1* / C1 / P
    - Fallback: first heavy atom
    """
    for name in ("CB", "CA", "C1'", "C1*", "C1", "P"):
        if name in res:
            return res[name]
    for a in res.get_atoms():
        if ((a.element or "").upper() != "H"):
            return a
    raise KeyError("No representative atom found for residue")


# ---- complex and interfaces ----
class Complex:
    """
    contact_thresh (Å):
      Used to define interface residue–residue contacts via representative atoms (CB/CA/C1'/...).
      Affects: Interface.contact_pairs, Interface.score_complex, Interface.pDockQ, Interface.pDockQ2,
              Complex.contact_pairs_global, Complex.mpDockQ.
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

        self._sasa_cache: Dict[str, float] = {}
        self._sasa_complex_cache: Dict[Tuple[str, str], float] = {}
        self._buried_surface_cache: Dict[Tuple[str, str, float, int], List[Tuple[np.ndarray, np.ndarray]]] = {}
        self._contact_ns_cache: Dict[Tuple[str, str], Tuple[list, list, np.ndarray, np.ndarray]] = {}
        self._all_atom_ns_cache: Dict[Tuple[str, str], Tuple[list, NeighborSearch]] = {}
        # NEW: cache representative-atom distances between token residues for ipSAE/LIS masking
        self._token_dist_cache: Dict[Tuple[str, str], np.ndarray] = {}

        self.interfaces: List[Interface] = []
        for i in range(len(self._chains)):
            for j in range(i + 1, len(self._chains)):
                iface = Interface(self._chains[i], self._chains[j], self)
                if iface.num_intf_residues > 0:
                    self.interfaces.append(iface)

    def _is_pae_token_residue(self, res) -> bool:
        """
        Keep only residues that correspond to PAE 'residue tokens' (one per residue).
        - Keep amino acids (standard or modified) if they have CA.
        - Keep nucleic acids (NA_RES) if they have C1' / C1* / C1 (varies by file).
        - Drop waters and typical non-polymer ligands.
        """
        resname = res.get_resname().strip().upper()

        if resname in ("HOH", "WAT", "H2O"):
            return False

        if resname in NA_RES:
            return any(k in res for k in ("C1'", "C1*", "C1"))

        if is_aa(res, standard=False):
            return "CA" in res

        return False

    def _build_maps(self) -> tuple[Dict[Tuple[str, Any], int], Dict[str, List[int]], List[Chain.Chain]]:
        model = next(self.structure.get_models())
        chains = list(model.get_chains())

        res_index_map: Dict[Tuple[str, Any], int] = {}
        chain_indices_by_id: Dict[str, List[int]] = {}
        filtered_chains: List[Chain.Chain] = []

        idx = 0
        for ch in chains:
            kept = [res for res in ch if self._is_pae_token_residue(res)]
            if not kept:
                continue

            new_ch = Chain.Chain(ch.id)
            for res in kept:
                new_ch.add(res.copy())
            filtered_chains.append(new_ch)

            idxs: List[int] = []
            for res in new_ch:  # map must use copied residues
                res_index_map[(new_ch.id, res.id)] = idx
                idxs.append(idx)
                idx += 1
            chain_indices_by_id[new_ch.id] = idxs

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
        for ch in self._chains:
            for res in ch:
                try:
                    reps.append((_repr_atom(res), res))
                except Exception:
                    continue
        if not reps:
            return 0

        ns = NeighborSearch([a for a, _ in reps])
        seen, cnt = set(), 0
        for a1, r1 in reps:
            for a2 in ns.search(a1.coord, self.contact_thresh):
                if a2 is a1:
                    continue
                r2 = a2.get_parent()
                c1, c2 = r1.get_parent().id, r2.get_parent().id
                if c1 == c2:
                    continue
                key = tuple(sorted([(c1, r1.id), (c2, r2.id)]))
                if key not in seen:
                    seen.add(key)
                    cnt += 1
        return cnt

    @cached_property
    def mpDockQ(self) -> float:
        if self.num_chains <= 2:
            return float("nan")
        x = self.average_interface_plddt * math.log10(self.contact_pairs_global + 1)
        return MPDOCKQ.score(x)


class Interface:
    def __init__(self, chain1, chain2, complex_ctx: Complex):
        self.c = complex_ctx
        self.chain1 = list(chain1)
        self.chain2 = list(chain2)

        if not self.chain1 or not self.chain2:
            self._pae = np.asarray(self.c.conf.pae_matrix)
            self._rim = self.c._res_index_map
            self._cid = self.c._chain_indices_by_id
            self._cid1_id = ""
            self._cid2_id = ""
            self._idx1 = np.array([], dtype=int)
            self._idx2 = np.array([], dtype=int)
            self._has_na = False
            self._res1, self._res2, self._pairs = set(), set(), set()
            self._avg_plddt = 0.0
            self._avg_pae = 0.0
            return

        self._pae = np.asarray(self.c.conf.pae_matrix)
        self._rim = self.c._res_index_map
        self._cid = self.c._chain_indices_by_id

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

    # ---------- core measures (public) ----------
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
    def iptm_chainpair(self) -> Optional[float]:
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
        If you want both directions too, return them separately.
        Returns (0.0, 0.0) when interface is not found (no contacts or invalid).
        """
        if self.contact_pairs <= 0 or math.isnan(self._avg_plddt):
            return 0.0, 0.0

        m_ab = self._mean_ptm_dir(reverse=False)  # A->B
        m_ba = self._mean_ptm_dir(reverse=True)   # B->A

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

        Parameters
        ----------
        pae_cutoff:
            Maximum PAE (Å) for a pair to be considered in the ipSAE calculation.
            - If the parent Complex has `ipsae_pae_cutoff` set, that value is used.
            - Otherwise the provided `pae_cutoff` argument (default 10.0 Å) is used.
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

        if math.isnan(a) and math.isnan(b):
            return 0.0
        if math.isnan(a):
            return b
        if math.isnan(b):
            return a
        return float(0.5 * (a + b))

    # composition
    @property
    def polar(self) -> float:
        return self._frac(POLAR_RES)

    @property
    def hydrophobic(self) -> float:
        return self._frac(HYDROPHOBIC_RES)

    @property
    def charged(self) -> float:
        return self._frac(CHARGED_RES)

    @cached_property
    def score_complex(self) -> float:
        if self.contact_pairs <= 0 or math.isnan(self._avg_plddt):
            return float("nan")
        return self._avg_plddt * math.log10(self.contact_pairs)

    # ---------- neighbor search ----------
    @cached_property
    def _ns_all_atoms(self) -> NeighborSearch:
        key = tuple(sorted((self._cid1_id, self._cid2_id)))
        cached = self.c._all_atom_ns_cache.get(key)
        if cached is None:
            atoms = [a for r in (self.chain1 + self.chain2) for a in r]
            cached = (atoms, NeighborSearch(atoms))
            self.c._all_atom_ns_cache[key] = cached
        return cached[1]

    # ---------- NEW: token-pair distances for ipSAE/LIS masking ----------
    def _token_pair_distances(self, idx_src: np.ndarray, idx_dst: np.ndarray) -> np.ndarray:
        """
        Return squared representative-atom distances for all (src,dst) token pairs
        in the same order as self._pae[np.ix_(idx_src, idx_dst)].ravel().

        idx_src/idx_dst are *global* token indices into the full PAE matrix.
        We map them to local indices within each chain's token list.
        """
        if idx_src.size == 0 or idx_dst.size == 0:
            return np.empty((0,), dtype=float)

        # Token indices per chain in *global* numbering (same as used by PAE)
        g1 = np.asarray(self.c._chain_indices_by_id.get(self._cid1_id, []), dtype=int)
        g2 = np.asarray(self.c._chain_indices_by_id.get(self._cid2_id, []), dtype=int)
        if g1.size == 0 or g2.size == 0:
            return np.full((idx_src.size * idx_dst.size,), np.inf, dtype=float)

        # Map global token index -> local index in that chain
        # (fast dict; chains are not huge vs. doing np.where repeatedly)
        map1 = {int(g): i for i, g in enumerate(g1.tolist())}
        map2 = {int(g): j for j, g in enumerate(g2.tolist())}

        # Convert idx_src/idx_dst (global) -> local indices; keep shape, fill unmapped with inf
        src_local = np.array([map1.get(int(g), -1) for g in idx_src.tolist()], dtype=int)
        dst_local = np.array([map2.get(int(g), -1) for g in idx_dst.tolist()], dtype=int)

        # If something is unmapped (shouldn't happen), treat as infinite distance
        if np.any(src_local < 0) or np.any(dst_local < 0):
            # Build full output with inf, then fill mapped sub-block
            out = np.full((idx_src.size, idx_dst.size), np.inf, dtype=float)
            good_i = np.where(src_local >= 0)[0]
            good_j = np.where(dst_local >= 0)[0]
            if good_i.size == 0 or good_j.size == 0:
                return out.ravel()
            src_good = src_local[good_i]
            dst_good = dst_local[good_j]
            dist2_full = self._token_dist2_matrix()
            out[np.ix_(good_i, good_j)] = dist2_full[np.ix_(src_good, dst_good)]
            return out.ravel()

        dist2_full = self._token_dist2_matrix()
        sub = dist2_full[np.ix_(src_local, dst_local)]
        return sub.ravel()


    def _token_dist2_matrix(self) -> np.ndarray:
        """
        Cached squared distance matrix between token residues of chain1 (rows)
        and chain2 (cols), in LOCAL chain token order.

        Handles both directions via transpose.
        """
        # Directional key: store both orientation and allow transpose on lookup
        key_fwd = (self._cid1_id, self._cid2_id)
        key_rev = (self._cid2_id, self._cid1_id)

        cached = self.c._token_dist_cache.get(key_fwd)
        if cached is not None:
            return cached

        cached_rev = self.c._token_dist_cache.get(key_rev)
        if cached_rev is not None:
            return cached_rev.T

        # Build coords in LOCAL token order using self.c._chains (already token-filtered)
        ch_by_id = {ch.id: ch for ch in self.c._chains}
        ch1 = ch_by_id.get(self._cid1_id)
        ch2 = ch_by_id.get(self._cid2_id)
        if ch1 is None or ch2 is None:
            m = np.full((0, 0), np.inf, dtype=float)
            self.c._token_dist_cache[key_fwd] = m
            return m

        coords1 = np.full((len(ch1), 3), np.nan, dtype=float)
        coords2 = np.full((len(ch2), 3), np.nan, dtype=float)

        for i, r in enumerate(ch1):
            try:
                coords1[i] = _repr_atom(r).coord
            except Exception:
                pass
        for j, r in enumerate(ch2):
            try:
                coords2[j] = _repr_atom(r).coord
            except Exception:
                pass

        diff = coords1[:, None, :] - coords2[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        dist2[np.isnan(dist2)] = np.inf

        self.c._token_dist_cache[key_fwd] = dist2
        return dist2


    # ---------- geometry helpers ----------
    @staticmethod
    def _angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        cos_angle = float(np.dot(v1, v2) / (norm1 * norm2))
        cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
        return float(math.degrees(math.acos(cos_angle)))

    @staticmethod
    def _angle_at_point(p1: np.ndarray, vertex: np.ndarray, p2: np.ndarray) -> float:
        return Interface._angle_between_vectors(p1 - vertex, p2 - vertex)

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 0 else v

    @staticmethod
    def _sp2_1h_2dd(donor: np.ndarray, dd1: np.ndarray, dd2: np.ndarray, dist: float = 1.0) -> np.ndarray:
        v1 = Interface._normalize(dd1 - donor)
        v2 = Interface._normalize(dd2 - donor)
        bisector = Interface._normalize(v1 + v2)
        return donor - bisector * dist

    @staticmethod
    def _sp2_2h_1dd(donor: np.ndarray, dd: np.ndarray, ddd: np.ndarray, dist: float = 1.0) -> List[np.ndarray]:
        v_in = Interface._normalize(donor - dd)
        v_ref = Interface._normalize(dd - ddd)
        n = Interface._normalize(np.cross(v_in, v_ref))
        cos60 = 0.5
        sin60 = math.sqrt(3.0) / 2.0
        cross_n_v = np.cross(n, v_in)
        h1 = donor + (v_in * cos60 + cross_n_v * sin60) * dist
        h2 = donor + (v_in * cos60 - cross_n_v * sin60) * dist
        return [h1, h2]

    @staticmethod
    def _sp3_3h_1dd(donor: np.ndarray, dd: np.ndarray, ddd: np.ndarray, dist: float = 1.01) -> List[np.ndarray]:
        v_axis = Interface._normalize(donor - dd)
        v_ref = dd - ddd
        v_proj = v_ref - float(np.dot(v_ref, v_axis)) * v_axis
        u = Interface._normalize(v_proj)
        v = np.cross(v_axis, u)
        cos110 = math.cos(math.radians(110.0))
        sin110 = math.sin(math.radians(110.0))
        axial_comp = -cos110
        trans_comp = sin110
        hydrogens = []
        for angle_deg in [0.0, 120.0, 240.0]:
            rad = math.radians(angle_deg)
            transverse = u * math.cos(rad) + v * math.sin(rad)
            h = donor + (v_axis * axial_comp + transverse * trans_comp) * dist
            hydrogens.append(h)
        return hydrogens

    @staticmethod
    def _sp3_1h_1dd(donor: np.ndarray, dd: np.ndarray, acceptor: np.ndarray, angle_deg: float = 110.0, dist: float = 1.0) -> List[np.ndarray]:
        v_axis = Interface._normalize(donor - dd)
        v_acc = acceptor - donor
        v_proj = v_acc - float(np.dot(v_acc, v_axis)) * v_axis
        norm_proj = float(np.linalg.norm(v_proj))
        if norm_proj < 1e-6:
            arbitrary = np.array([1.0, 0.0, 0.0])
            if abs(v_axis[0]) > 0.9: arbitrary = np.array([0.0, 1.0, 0.0])
            u = Interface._normalize(arbitrary - float(np.dot(arbitrary, v_axis)) * v_axis)
        else:
            u = v_proj / norm_proj
        axial_comp = -math.cos(math.radians(angle_deg))
        trans_comp = math.sin(math.radians(angle_deg))
        h = donor + (v_axis * axial_comp + u * trans_comp) * dist
        return [h]

    @staticmethod
    def _sp2_1h_1dd(donor: np.ndarray, dd: np.ndarray, ddd1: np.ndarray, ddd2: np.ndarray, acceptor: np.ndarray, dist: float = 1.0) -> List[np.ndarray]:
        v1 = Interface._normalize(dd - ddd1)
        v2 = Interface._normalize(dd - ddd2)
        n = Interface._normalize(np.cross(v1, v2))
        v_axis = Interface._normalize(donor - dd)
        cos70 = -math.cos(math.radians(110.0))
        sin70 = math.sin(math.radians(110.0))
        cross_n_v = np.cross(n, v_axis)
        h1 = donor + (v_axis * cos70 + cross_n_v * sin70) * dist
        h2 = donor + (v_axis * cos70 - cross_n_v * sin70) * dist
        d1 = float(np.linalg.norm(h1 - acceptor))
        d2 = float(np.linalg.norm(h2 - acceptor))
        return [h1] if d1 < d2 else [h2]

    @staticmethod
    def _generate_ideal_hydrogens(donor_atom, acceptor_coord: np.ndarray) -> List[np.ndarray]:
        res = donor_atom.get_parent()
        if res is None: return []
        resname = res.get_resname().strip().upper()
        name = donor_atom.id.upper()
        
        def _get_c(n):
            return res[n].coord if n in res else None
            
        if resname == "ARG":
            if name == "NE":
                cd, cz = _get_c("CD"), _get_c("CZ")
                if cd is not None and cz is not None:
                    return [Interface._sp2_1h_2dd(donor_atom.coord, cd, cz)]
            elif name in ("NH1", "NH2"):
                cz, ne = _get_c("CZ"), _get_c("NE")
                if cz is not None and ne is not None:
                    return Interface._sp2_2h_1dd(donor_atom.coord, cz, ne)
        elif resname == "LYS" and name == "NZ":
            ce, cd = _get_c("CE"), _get_c("CD")
            if ce is not None and cd is not None:
                return Interface._sp3_3h_1dd(donor_atom.coord, ce, cd)
        elif resname == "ASN" and name == "ND2":
            cg, cb = _get_c("CG"), _get_c("CB")
            if cg is not None and cb is not None:
                return Interface._sp2_2h_1dd(donor_atom.coord, cg, cb)
        elif resname == "GLN" and name == "NE2":
            cd, cg = _get_c("CD"), _get_c("CG")
            if cd is not None and cg is not None:
                return Interface._sp2_2h_1dd(donor_atom.coord, cd, cg)
        elif resname == "HIS":
            if name == "NE2":
                cd2, ce1 = _get_c("CD2"), _get_c("CE1")
                if cd2 is not None and ce1 is not None:
                    return [Interface._sp2_1h_2dd(donor_atom.coord, cd2, ce1)]
            elif name == "ND1":
                cg, ce1 = _get_c("CG"), _get_c("CE1")
                if cg is not None and ce1 is not None:
                    return [Interface._sp2_1h_2dd(donor_atom.coord, cg, ce1)]
        elif resname == "TRP" and name == "NE1":
            cd1, ce2 = _get_c("CD1"), _get_c("CE2")
            if cd1 is not None and ce2 is not None:
                return [Interface._sp2_1h_2dd(donor_atom.coord, cd1, ce2)]
        elif resname == "SER" and name == "OG":
            cb = _get_c("CB")
            if cb is not None:
                return Interface._sp3_1h_1dd(donor_atom.coord, cb, acceptor_coord, 110.0, 1.0)
        elif resname == "THR" and name == "OG1":
            cb = _get_c("CB")
            if cb is not None:
                return Interface._sp3_1h_1dd(donor_atom.coord, cb, acceptor_coord, 110.0, 1.0)
        elif resname == "TYR" and name == "OH":
            cz, ce1, ce2 = _get_c("CZ"), _get_c("CE1"), _get_c("CE2")
            if cz is not None and ce1 is not None and ce2 is not None:
                return Interface._sp2_1h_1dd(donor_atom.coord, cz, ce1, ce2, acceptor_coord)
        elif resname == "CYS" and name == "SG":
            cb = _get_c("CB")
            if cb is not None:
                return Interface._sp3_1h_1dd(donor_atom.coord, cb, acceptor_coord, 96.0, 1.33)
                
        # Main-chain N (backbone) or N-terminus
        if name == "N":
            ca = _get_c("CA")
            # If standard main chain, we want the previous residue's C
            # But the 'res' object only has atoms for the current residue.
            # We can use biopython's structure navigation.
            c_prev = None
            chain = res.get_parent()
            if chain:
                # get previous residue logically (slow linear search or index)
                idx = -1
                res_list = list(chain)
                try:
                    idx = res_list.index(res)
                except ValueError:
                    pass
                if idx > 0:
                    prev_res = res_list[idx-1]
                    if "C" in prev_res:
                        c_prev = prev_res["C"].coord
            
            if ca is not None and c_prev is not None:
                return [Interface._sp2_1h_2dd(donor_atom.coord, ca, c_prev, 1.0)]
            elif ca is not None:
                # N-terminus fallback: sp3 3H 1DD
                return Interface._sp3_3h_1dd(donor_atom.coord, ca, ca + np.array([1,1,1])) # arbitrary ddd if none

        # N-terminus or unknown
        return []

    @staticmethod
    def _find_hydrogen_atoms(donor_atom) -> List[Any]:
        hydrogens = []
        residue = donor_atom.get_parent()
        if residue is None:
            return hydrogens

        donor_coord = donor_atom.coord
        donor_name = donor_atom.id.upper()

        max_h_dist = 1.33
        for atom in residue:
            if not (atom.element and atom.element.upper() == "H"):
                continue

            dist = float(np.linalg.norm(atom.coord - donor_coord))
            if dist <= max_h_dist:
                hydrogens.append(atom)
                continue

            atom_name = atom.id.upper()
            if donor_name == "N" and atom_name in ("H", "1H", "2H", "3H", "HN"):
                hydrogens.append(atom)
            elif len(donor_name) >= 2:
                if atom_name.startswith("H") and donor_name[1:] in atom_name[1:]:
                    hydrogens.append(atom)
                elif atom_name.startswith("H") and donor_name[-1] in atom_name:
                    if dist <= 1.5:
                        hydrogens.append(atom)

        return hydrogens

    @staticmethod
    def _find_acceptor_antecedent(acceptor_atom) -> Optional[Any]:
        residue = acceptor_atom.get_parent()
        if residue is None:
            return None

        acceptor_coord = acceptor_atom.coord
        closest_atom = None
        min_dist = float("inf")

        for atom in residue:
            if atom is acceptor_atom:
                continue
            if atom.element and atom.element.upper() == "H":
                continue
            dist = float(np.linalg.norm(atom.coord - acceptor_coord))
            if dist < min_dist and dist < 2.0:
                min_dist = dist
                closest_atom = atom
        return closest_atom

    @staticmethod
    def _can_be_donor(atom) -> bool:
        element = (atom.element or "").upper()
        atom_name = atom.id.upper()
        residue = atom.get_parent()
        if residue is None:
            return False
        resname = residue.get_resname().strip().upper()

        if element == "N" and atom_name == "N" and resname != "PRO":
            return True

        if element == "N":
            if resname == "ARG" and atom_name in ("NE", "NH1", "NH2"):
                return True
            if resname == "LYS" and atom_name == "NZ":
                return True
            if resname == "ASN" and atom_name == "ND2":
                return True
            if resname == "GLN" and atom_name == "NE2":
                return True
            if resname == "HIS" and atom_name in ("NE2", "ND1"):
                return True
            if resname == "TRP" and atom_name == "NE1":
                return True

        if element == "O":
            if resname == "SER" and atom_name == "OG":
                return True
            if resname == "THR" and atom_name == "OG1":
                return True
            if resname == "TYR" and atom_name == "OH":
                return True

        if element == "S" and resname == "CYS" and atom_name == "SG":
            return True

        if element == "O" and resname in ("HOH", "WAT", "H2O"):
            return True

        if resname not in {
            "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET",
            "PHE","PRO","SER","THR","TRP","TYR","VAL","HOH","WAT","H2O"
        }:
            if element in ("N", "O"):
                return True

        return False

    @staticmethod
    def _can_be_acceptor(atom) -> bool:
        element = (atom.element or "").upper()
        atom_name = atom.id.upper()
        residue = atom.get_parent()
        if residue is None:
            return False
        resname = residue.get_resname().strip().upper()

        if element == "O" and atom_name in ("O", "OXT"):
            return True

        if element == "O":
            if resname == "ASP" and atom_name in ("OD1", "OD2"):
                return True
            if resname == "GLU" and atom_name in ("OE1", "OE2"):
                return True
            if resname == "ASN" and atom_name == "OD1":
                return True
            if resname == "GLN" and atom_name == "OE1":
                return True
            if resname == "SER" and atom_name == "OG":
                return True
            if resname == "THR" and atom_name == "OG1":
                return True
            if resname == "TYR" and atom_name == "OH":
                return True

        if element == "N" and resname == "HIS" and atom_name in ("ND1", "NE2"):
            return True

        if element == "S" and resname == "CYS" and atom_name == "SG":
            return True

        if element == "O" and resname in ("HOH", "WAT", "H2O"):
            return True

        if resname not in {
            "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET",
            "PHE","PRO","SER","THR","TRP","TYR","VAL","HOH","WAT","H2O"
        }:
            if element in ("O", "N"):
                return True

        return False

    # ---------- HB / SB / SC / areas ----------
    @cached_property
    def hb(self) -> int:
        def _candidate_atoms(chain_residues):
            out = []
            for res in chain_residues:
                for atom in res:
                    elem = (atom.element or "").upper()
                    if elem in {"N", "O", "S"}:
                        out.append(atom)
            return out

        atoms1 = _candidate_atoms(self.chain1)
        atoms2 = _candidate_atoms(self.chain2)
        if not atoms1 or not atoms2:
            return 0

        ids1 = {id(a) for a in atoms1}
        ids2 = {id(a) for a in atoms2}
        ns = self._ns_all_atoms

        max_da_dist = 3.9
        max_ha_dist = 2.5
        min_angle = 90.0

        seen, cnt = set(), 0
        for a1, a2 in ns.search_all(max_da_dist):
            i1, i2 = id(a1), id(a2)
            if not ((i1 in ids1 and i2 in ids2) or (i1 in ids2 and i2 in ids1)):
                continue

            donor = acceptor = None
            match (self._can_be_donor(a1), self._can_be_acceptor(a1),
                   self._can_be_acceptor(a2), self._can_be_donor(a2)):
                case (True, _, True, _):
                    donor, acceptor = a1, a2
                case (_, True, _, True):
                    donor, acceptor = a2, a1

            if donor is None or acceptor is None:
                continue

            da_dist = float(np.linalg.norm(donor.coord - acceptor.coord))
            if da_dist > max_da_dist:
                continue

            hydrogens = self._find_hydrogen_atoms(donor)
            aa = self._find_acceptor_antecedent(acceptor)

            valid_hbond = False
            if hydrogens:
                for h in hydrogens:
                    ha_dist = float(np.linalg.norm(h.coord - acceptor.coord))
                    if ha_dist > max_ha_dist:
                        continue
                    dha = self._angle_at_point(donor.coord, h.coord, acceptor.coord)
                    if dha < min_angle:
                        continue
                    if aa is not None:
                        daaa = self._angle_at_point(donor.coord, acceptor.coord, aa.coord)
                        if daaa < min_angle:
                            continue
                        haaa = self._angle_at_point(h.coord, acceptor.coord, aa.coord)
                        if haaa < min_angle:
                            continue
                    valid_hbond = True
                    break
            else:
                ideal_hydrogens = self._generate_ideal_hydrogens(donor, acceptor.coord)
                if ideal_hydrogens:
                    for h_coord in ideal_hydrogens:
                        ha_dist = float(np.linalg.norm(h_coord - acceptor.coord))
                        if ha_dist <= max_ha_dist:
                            dha = self._angle_at_point(donor.coord, h_coord, acceptor.coord)
                            if dha >= min_angle:
                                if aa is not None:
                                    daaa = self._angle_at_point(donor.coord, acceptor.coord, aa.coord)
                                    if daaa >= min_angle:
                                        haaa = self._angle_at_point(h_coord, acceptor.coord, aa.coord)
                                        if haaa >= min_angle:
                                            valid_hbond = True
                                            break
                                else:
                                    valid_hbond = True
                                    break
                else:
                    da_vec = acceptor.coord - donor.coord
                    da_norm = float(np.linalg.norm(da_vec))
                    if da_norm > 0:
                        inferred_h = donor.coord + (da_vec / da_norm) * 1.0
                        ha_dist = float(np.linalg.norm(inferred_h - acceptor.coord))
                        if ha_dist <= max_ha_dist:
                            dha = self._angle_at_point(donor.coord, inferred_h, acceptor.coord)
                            if dha >= min_angle:
                                if aa is not None:
                                    daaa = self._angle_at_point(donor.coord, acceptor.coord, aa.coord)
                                    if daaa >= min_angle:
                                        haaa = self._angle_at_point(inferred_h, acceptor.coord, aa.coord)
                                        if haaa >= min_angle:
                                            valid_hbond = True
                                else:
                                    valid_hbond = True
                    if (not valid_hbond) and aa is not None:
                        daaa = self._angle_at_point(donor.coord, acceptor.coord, aa.coord)
                        if daaa >= min_angle:
                            valid_hbond = True

            if valid_hbond:
                key = tuple(sorted((i1, i2)))
                if key not in seen:
                    seen.add(key)
                    cnt += 1

        return cnt

    @cached_property
    def sb(self) -> int:
        pos, neg = {"ARG", "LYS"}, {"ASP", "GLU"}
        relevant = pos | neg

        a1 = [
            a for r in self.chain1
            if r.get_resname().strip().upper() in relevant
            for a in r
            if a.id.upper() in CHARGED_ATOMS.get(r.get_resname().strip().upper(), set())
        ]
        a2 = [
            a for r in self.chain2
            if r.get_resname().strip().upper() in relevant
            for a in r
            if a.id.upper() in CHARGED_ATOMS.get(r.get_resname().strip().upper(), set())
        ]
        if not a1 or not a2:
            return 0

        ns = self._ns_all_atoms
        cutoff = 4.0

        ids1 = {id(a): (1 if a.get_parent().get_resname().strip().upper() in pos else -1) for a in a1}
        ids2 = {id(a): (1 if a.get_parent().get_resname().strip().upper() in pos else -1) for a in a2}

        seen, cnt = set(), 0
        for x, y in ns.search_all(cutoff):
            ix, iy = id(x), id(y)
            if ix in ids1 and iy in ids2:
                if ids1[ix] + ids2[iy] == 0:
                    key = tuple(sorted((ix, iy)))
                    if key not in seen:
                        seen.add(key)
                        cnt += 1
            elif ix in ids2 and iy in ids1:
                if ids2[ix] + ids1[iy] == 0:
                    key = tuple(sorted((ix, iy)))
                    if key not in seen:
                        seen.add(key)
                        cnt += 1
        return cnt

    @cached_property
    def sc(self) -> float:
        sA = self._buried_surface(self.chain1, self.chain2, 5.0, 15)
        sB = self._buried_surface(self.chain2, self.chain1, 5.0, 15)
        if not sA or not sB:
            return 0.0
        return self._approx_sc(sA, sB, w=0.5)

    @cached_property
    def int_area(self) -> float:
        return (
            self._sasa_chain(self.chain1)
            + self._sasa_chain(self.chain2)
            - self._sasa_complex(self.chain1, self.chain2)
        )

    @cached_property
    def int_solv_en(self) -> float:
        return -0.0072 * self.int_area

    # ---------- private helpers ----------
    def _get_pairs(self):
        res_pairs: Set[Tuple[Any, Any]] = set()
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

    def _contact_atom_data(self) -> Tuple[list, list, np.ndarray, np.ndarray]:
        key = (self._cid1_id, self._cid2_id)
        cached = self.c._contact_ns_cache.get(key)
        if cached is None:
            a1, a2 = [], []
            for r in self.chain1:
                try:
                    a1.append(_repr_atom(r))
                except Exception:
                    pass
            for r in self.chain2:
                try:
                    a2.append(_repr_atom(r))
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
        for r in res_set:
            try:
                vals.append(float(_repr_atom(r).get_bfactor()))
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

            best, found = 0.0, False
            for i in idx_src:
                row = self._pae[i, idx_dst]
                valid = row < cutoff
                if not np.any(valid):
                    continue
                n = int(np.count_nonzero(valid))
                L = max(27.0, float(n))
                d0 = max(min_d0, 1.24 * (L - 15.0) ** (1.0 / 3.0) - 1.8)
                ptm = 1.0 / (1.0 + (row[valid] / d0) ** 2)
                best = max(best, float(np.mean(ptm)))
                found = True
            return best if found else 0.0

        a = calc(self._idx1, self._idx2)
        b = calc(self._idx2, self._idx1)
        if math.isnan(a) and math.isnan(b):
            return 0.0
        if math.isnan(a):
            return b
        if math.isnan(b):
            return a
        return max(a, b)

    def _frac(self, names: Set[str]) -> float:
        residues = self._res1 | self._res2
        if not residues:
            return 0.0
        return sum(1 for r in residues if r.get_resname().strip().upper() in names) / len(residues)

    # ---------- SASA / SC ----------
    def _sasa_chain(self, residues) -> float:
        chain_id = residues[0].get_parent().id if residues else None
        if chain_id:
            cached = self.c._sasa_cache.get(chain_id)
            if cached is not None:
                return cached
        total = self._compute_sasa_chain(residues)
        if chain_id:
            self.c._sasa_cache[chain_id] = total
        return total

    def _compute_sasa_chain(self, residues) -> float:
        sr = ShrakeRupley()
        s = Structure.Structure("S")
        m = Model.Model(0)
        s.add(m)
        c = Chain.Chain("X")
        m.add(c)

        for r in residues:
            c.add(r.copy())

        try:
            sr.compute(s, level="R")
        except ValueError:
            return 0.0

        total = 0.0
        for r in c:
            if hasattr(r, "sasa"):
                total += float(getattr(r, "sasa", 0.0))
            else:
                total += float(r.xtra.get("EXP_RSASA", 0.0))
        return total

    def _sasa_complex(self, r1, r2) -> float:
        key = tuple(sorted((self._cid1_id, self._cid2_id)))
        cached = self.c._sasa_complex_cache.get(key)
        if cached is not None:
            return cached
        total = self._compute_sasa_complex(r1, r2)
        self.c._sasa_complex_cache[key] = total
        return total

    def _compute_sasa_complex(self, r1, r2) -> float:
        sr = ShrakeRupley()
        s = Structure.Structure("C")
        m = Model.Model(0)
        s.add(m)
        cA = Chain.Chain("A")
        cB = Chain.Chain("B")
        m.add(cA)
        m.add(cB)

        for r in r1:
            cA.add(r.copy())
        for r in r2:
            cB.add(r.copy())

        try:
            sr.compute(s, level="R")
        except ValueError:
            return 0.0

        total = 0.0
        for c in (cA, cB):
            for r in c:
                if hasattr(r, "sasa"):
                    total += float(getattr(r, "sasa", 0.0))
                else:
                    total += float(r.xtra.get("EXP_RSASA", 0.0))
        return total

    def _buried_surface(self, chain_res, other_res, dist=5.0, dots=15):
        chain_id = chain_res[0].get_parent().id if chain_res else None
        other_id = other_res[0].get_parent().id if other_res else None
        cache_key = (chain_id, other_id, float(dist), int(dots))
        if chain_id and other_id:
            cached = self.c._buried_surface_cache.get(cache_key)
            if cached is not None:
                return cached
        result = self._compute_buried_surface(chain_res, other_res, dist, dots)
        if chain_id and other_id:
            self.c._buried_surface_cache[cache_key] = result
        return result

    def _compute_buried_surface(self, chain_res, other_res, dist=5.0, dots=15):
        sr = ShrakeRupley(probe_radius=1.4, n_points=dots)

        # legacy EXP_DOTS path (old Biopython)
        s = Structure.Structure("Z")
        m = Model.Model(0)
        s.add(m)
        cz = Chain.Chain("Z")
        m.add(cz)
        for r in chain_res:
            cz.add(r.copy())
        try:
            sr.compute(s, level="A")
        except ValueError:
            pass

        pts = []
        for r in cz:
            for a in r:
                data = getattr(a, "xtra", {}).get("EXP_DOTS", [])
                for (x, y, z, nx, ny, nz) in data:
                    pts.append((np.array([x, y, z]), np.array([nx, ny, nz])))

        others = [a for rr in other_res for a in rr.get_atoms() if (a.element or "").upper() != "H"]
        if pts and others:
            coords = np.array([a.coord for a in others])
            d2 = float(dist) ** 2
            buried = []
            for xyz, n in pts:
                if np.any(np.sum((coords - xyz) ** 2, axis=1) <= d2):
                    buried.append((xyz, n))
            return buried

        # modern fallback: generate points from SR unit sphere (no self-occlusion modelling)
        atoms = [a for r in chain_res for a in r.get_atoms() if (a.element or "").upper() != "H"]
        if not atoms or not others:
            return []

        other_coords = np.array([a.coord for a in others])
        d2 = float(dist) ** 2

        if dots == 15:
            sphere = _SPHERE_15
            radii_dict = _RADII_DICT_15
            probe_radius = _SR_PROBE_RADIUS
        else:
            sphere = np.array(sr._sphere, copy=False)
            radii_dict = sr.radii_dict
            probe_radius = float(sr.probe_radius)

        buried = []
        atom_coords = np.array([a.coord for a in atoms])
        atom_radii = np.array([radii_dict[a.element] + probe_radius for a in atoms])

        for radius, center in zip(atom_radii, atom_coords):
            points = sphere * radius + center
            diff = points[:, None, :] - other_coords[None, :, :]
            dist2 = np.sum(diff * diff, axis=2)
            mask = np.any(dist2 <= d2, axis=1)
            if np.any(mask):
                for xyz, n in zip(points[mask], sphere[mask]):
                    buried.append((xyz.astype(float), n.astype(float)))

        return buried

    def _approx_sc(self, A, B, w=0.5) -> float:
        cB = np.array([p[0] for p in B], dtype=float)
        nB = np.array([p[1] for p in B], dtype=float)
        cA = np.array([p[0] for p in A], dtype=float)
        nA = np.array([p[1] for p in A], dtype=float)

        sA = []
        for x, nA_vec in A:
            d = cB - x
            d_sq = np.sum(d * d, axis=1)
            j = int(np.argmin(d_sq))
            sA.append(float(np.dot(nA_vec, -nB[j]) * math.exp(-w * float(d_sq[j]))))

        sB = []
        for x, n_vec in B:
            d = cA - x
            d_sq = np.sum(d * d, axis=1)
            j = int(np.argmin(d_sq))
            sB.append(float(np.dot(n_vec, -nA[j]) * math.exp(-w * float(d_sq[j]))))

        return float(0.5 * (np.median(sA) + np.median(sB)))
