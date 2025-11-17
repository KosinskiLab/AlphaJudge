from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional, Dict, Tuple, Any, Set
import math, numpy as np
from Bio.PDB import NeighborSearch, Structure, Model, Chain
from Bio.PDB.SASA import ShrakeRupley

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

# ---- dockQ constants ----
def _sigmoid(x: float, L: float, x0: float, k: float, b: float) -> float:
    return L / (1 + math.exp(-k * (x - x0))) + b

@dataclass(frozen=True)
class _DockQ:
    L: float; X0: float; K: float; B: float
    def score(self, x: float) -> float:
        return _sigmoid(x, self.L, self.X0, self.K, self.B)

PDOCKQ  = _DockQ(0.724, 152.611, 0.052, 0.018)
PDOCKQ2 = _DockQ(1.31,   84.733,  0.075, 0.005)
MPDOCKQ = _DockQ(0.728, 309.375,  0.098, 0.262)
D0 = 10.0

# ---- complex and interfaces ----
class Complex:
    def __init__(self, structure, confidence: Confidence, contact_thresh: float, pae_filter: float):
        self.structure = structure
        self.conf = confidence
        self.contact_thresh = contact_thresh
        self.pae_filter = pae_filter
        self._res_index_map, self._chain_indices_by_id, self._chains = self._build_maps()

        self.interfaces: List[Interface] = []
        for i in range(len(self._chains)):
            for j in range(i+1, len(self._chains)):
                iface = Interface(self._chains[i], self._chains[j], self)
                if iface.num_intf_residues > 0:
                    self.interfaces.append(iface)

    # ---------- maps & residue utilities ----------
    def _build_maps(self):
        model = next(self.structure.get_models())
        chains = list(model.get_chains())
        res_index_map, chain_indices_by_id, idx = {}, {}, 0
        for ch in chains:
            idxs = []
            for res in ch:
                res_index_map[(ch.id, res.id)] = idx
                idxs.append(idx); idx += 1
            chain_indices_by_id[ch.id] = idxs
        return res_index_map, chain_indices_by_id, chains

    @property
    def num_chains(self) -> int:
        return len(self._chains)

    @cached_property
    def average_interface_pae(self) -> float:
        vals = [
            i.average_interface_pae for i in self.interfaces
            if not math.isnan(i.average_interface_pae) and i.average_interface_pae <= self.pae_filter
        ]
        return sum(vals) / len(vals) if vals else float('nan')

    @cached_property
    def average_interface_plddt(self) -> float:
        vals = [i.average_interface_plddt for i in self.interfaces]
        return sum(vals) / len(vals) if vals else float('nan')

    @cached_property
    def contact_pairs_global(self) -> int:
        reps = []
        for ch in self._chains:
            for res in ch:
                try: reps.append((res["CB"] if "CB" in res else res["CA"], res))
                except Exception: continue
        if not reps: return 0
        ns = NeighborSearch([a for a,_ in reps])
        seen, cnt = set(), 0
        for a1, r1 in reps:
            for a2 in ns.search(a1.coord, self.contact_thresh):
                if a2 is a1: continue
                r2 = a2.get_parent()
                c1, c2 = r1.get_parent().id, r2.get_parent().id
                if c1 == c2: continue
                key = tuple(sorted([(c1, r1.id), (c2, r2.id)]))
                if key not in seen:
                    seen.add(key); cnt += 1
        return cnt

    @cached_property
    def mpDockQ(self) -> float:
        if self.num_chains <= 2: return float('nan')
        x = self.average_interface_plddt * math.log10(self.contact_pairs_global + 1)
        return MPDOCKQ.score(x)

class Interface:
    def __init__(self, chain1, chain2, complex_ctx: Complex):
        self.c = complex_ctx
        self.chain1 = list(chain1)
        self.chain2 = list(chain2)

        self._pae = np.asarray(self.c.conf.pae_matrix)
        self._rim = self.c._res_index_map
        self._cid = self.c._chain_indices_by_id

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
    def contact_pairs(self) -> int:
        return len(self._pairs)

    @cached_property
    def pDockQ(self) -> float:
        if self.contact_pairs <= 0 or math.isnan(self._avg_plddt): return float('nan')
        return PDOCKQ.score(self._avg_plddt * math.log10(self.contact_pairs))

    def pDockQ2(self) -> tuple[float, float]:
        vals = self._ptm_values()
        if not vals or math.isnan(self._avg_plddt): return float('nan'), 0.0
        mean_ptm = float(np.mean(vals))
        return PDOCKQ2.score(self._avg_plddt * mean_ptm), mean_ptm

    def ipsae(self, pae_cutoff: float = 10.0) -> float:
        return self._ipsae_asym(pae_cutoff)

    def lis(self) -> float:
        cid1 = self.chain1[0].get_parent().id; cid2 = self.chain2[0].get_parent().id
        idx1 = self._cid.get(cid1, []); idx2 = self._cid.get(cid2, [])
        if not idx1 or not idx2: return float('nan')
        sub = self._pae[np.ix_(idx1, idx2)]
        valid = sub[sub <= 12.0]
        if valid.size == 0: return float('nan')
        return float(np.mean((12.0 - valid) / 12.0))

    # composition
    @property
    def polar(self) -> float:        return self._frac({"SER","THR","ASN","GLN","TYR","CYS"})
    @property
    def hydrophobic(self) -> float:  return self._frac({"ALA","VAL","LEU","ILE","MET","PHE","TRP"})
    @property
    def charged(self) -> float:      return self._frac({"ARG","LYS","ASP","GLU","HIS"})

    # quick “complex” score
    @cached_property
    def score_complex(self) -> float:
        if self.contact_pairs <= 0 or math.isnan(self._avg_plddt): return float('nan')
        return self._avg_plddt * math.log10(self.contact_pairs)

    # HB / SB / SC / areas (self-contained helpers)
    @cached_property
    def hb(self) -> int:
        atoms1 = [a for r in self.chain1 for a in r if a.id.upper().startswith(("N","O"))]
        atoms2 = [a for r in self.chain2 for a in r if a.id.upper().startswith(("N","O"))]
        if not atoms1 or not atoms2: return 0
        ns = NeighborSearch(atoms1 + atoms2); cutoff = 3.5
        seen, cnt = set(), 0
        for a1 in atoms1:
            for a2 in ns.search(a1.coord, cutoff):
                if a2 in atoms2 and a2 is not a1:
                    key = tuple(sorted([id(a1), id(a2)]))
                    if key not in seen:
                        seen.add(key); cnt += 1
        return cnt

    @cached_property
    def sb(self) -> int:
        pos, neg = {"ARG","LYS"}, {"ASP","GLU"}
        a1 = [a for r in self.chain1 if r.get_resname() in pos|neg for a in r if a.id not in ("N","CA","C","O")]
        a2 = [a for r in self.chain2 if r.get_resname() in pos|neg for a in r if a.id not in ("N","CA","C","O")]
        if not a1 or not a2: return 0
        ns = NeighborSearch(a1 + a2); cutoff = 4.0
        seen, cnt = set(), 0
        for x in a1:
            n1 = x.get_parent().get_resname()
            for y in ns.search(x.coord, cutoff):
                if y in a2 and y is not x:
                    n2 = y.get_parent().get_resname()
                    if (n1 in pos and n2 in neg) or (n1 in neg and n2 in pos):
                        key = tuple(sorted([id(x), id(y)]))
                        if key not in seen:
                            seen.add(key); cnt += 1
        return cnt

    @cached_property
    def sc(self) -> float:
        sA = self._buried_surface(self.chain1, self.chain2, 5.0, 15)
        sB = self._buried_surface(self.chain2, self.chain1, 5.0, 15)
        if not sA or not sB: return 0.0
        return self._approx_sc(sA, sB, w=0.5)

    @cached_property
    def int_area(self) -> float:
        return self._sasa_chain(self.chain1) + self._sasa_chain(self.chain2) - self._sasa_complex(self.chain1, self.chain2)

    @cached_property
    def int_solv_en(self) -> float:
        return -0.0072 * self.int_area

    # ---------- private helpers below ----------
    def _get_pairs(self):
        res_pairs: Set[Tuple[Any, Any]] = set()
        a1 = [r["CB"] if "CB" in r else r["CA"] for r in self.chain1]
        a2 = [r["CB"] if "CB" in r else r["CA"] for r in self.chain2]
        ns = NeighborSearch(a1 + a2)
        for x in a1:
            for y in ns.search(x.coord, self.c.contact_thresh):
                if y in a2: res_pairs.add((x.get_parent(), y.get_parent()))
        r1 = {p[0] for p in res_pairs}; r2 = {p[1] for p in res_pairs}
        return r1, r2, res_pairs

    def _avg_plddt_union(self) -> float:
        res_set = self._res1 | self._res2
        vals = []
        for r in res_set:
            try: a = r["CB"]
            except KeyError: a = r["CA"]
            vals.append(a.get_bfactor())
        return sum(vals)/len(vals) if vals else float('nan')

    def _avg_pae_over_pairs(self) -> float:
        vals = []
        for r1, r2 in self._pairs:
            k1 = (r1.get_parent().id, r1.id); i = self._rim.get(k1)
            k2 = (r2.get_parent().id, r2.id); j = self._rim.get(k2)
            if i is None or j is None: continue
            try:
                vals.append(float(self._pae[i, j])); vals.append(float(self._pae[j, i]))
            except Exception: pass
        return sum(vals)/len(vals) if vals else float('nan')

    def _ptm_values(self) -> List[float]:
        out = []
        for r1, r2 in self._pairs:
            i = self._rim.get((r1.get_parent().id, r1.id))
            j = self._rim.get((r2.get_parent().id, r2.id))
            if i is None or j is None: continue
            try:
                pae = float(self._pae[i, j])
                out.append(1.0 / (1.0 + (pae / D0) ** 2))
            except Exception: pass
        return out

    def _ipsae_asym(self, cutoff: float) -> float:
        def calc(src, dst) -> float:
            src_idx = [self._rim.get((r.get_parent().id, r.id)) for r in src]
            dst_idx = [self._rim.get((r.get_parent().id, r.id)) for r in dst]
            src_idx = [i for i in src_idx if i is not None]; dst_idx = [j for j in dst_idx if j is not None]
            if not src_idx or not dst_idx: return float('nan')
            best, found = 0.0, False
            for i in src_idx:
                row = self._pae[i, dst_idx]; valid = row < cutoff
                if not np.any(valid): continue
                n = int(np.count_nonzero(valid)); L = max(27.0, float(n))
                d0 = max(1.0, 1.24 * (L - 15.0) ** (1.0/3.0) - 1.8)
                ptm = 1.0 / (1.0 + (row[valid]/d0) ** 2)
                best, found = max(best, float(np.mean(ptm))), True
            return best if found else float('nan')
        a = calc(self.chain1, self.chain2); b = calc(self.chain2, self.chain1)
        if math.isnan(a) and math.isnan(b): return float('nan')
        if math.isnan(a): return b
        if math.isnan(b): return a
        return max(a, b)

    def _frac(self, names: set[str]) -> float:
        residues = self._res1 | self._res2
        return (sum(1 for r in residues if r.get_resname() in names) / len(residues)) if residues else 0.0

    def _sasa_chain(self, residues) -> float:
        """
        Return solvent-accessible surface area for a set of residues.

        Biopython's ShrakeRupley used to store results in residue.xtra["EXP_RSASA"],
        but newer versions expose them via the .sasa attribute instead.  To remain
        compatible with both behaviours we:

        * run ShrakeRupley on a throwaway structure built from the residues
        * first try to read r.sasa
        * fall back to r.xtra["EXP_RSASA"] if present
        """
        sr = ShrakeRupley()
        s = Structure.Structure("S"); m = Model.Model(0); s.add(m); c = Chain.Chain("X"); m.add(c)
        for r in residues:
            c.add(r.copy())
        try:
            sr.compute(s, level="R")
        except Exception:
            # If SASA computation fails, treat as zero area
            return 0.0

        total = 0.0
        for r in c:
            if hasattr(r, "sasa"):
                total += float(getattr(r, "sasa", 0.0))
            else:
                total += float(r.xtra.get("EXP_RSASA", 0.0))
        return total

    def _sasa_complex(self, r1, r2) -> float:
        """
        Return solvent-accessible surface area for a two-chain complex.

        As in _sasa_chain, support both legacy xtra["EXP_RSASA"] and modern
        .sasa attributes from Biopython's ShrakeRupley implementation.
        """
        sr = ShrakeRupley()
        s = Structure.Structure("C"); m = Model.Model(0); s.add(m)
        cA = Chain.Chain("A"); cB = Chain.Chain("B"); m.add(cA); m.add(cB)
        for r in r1:
            cA.add(r.copy())
        for r in r2:
            cB.add(r.copy())

        try:
            sr.compute(s, level="R")
        except Exception:
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
        """
        Approximate the set of buried surface points on `chain_res` that lie
        within `dist` Å of any atom in `other_res`.

        Older Biopython versions exposed ShrakeRupley points and normals via
        atom.xtra['EXP_DOTS']; newer versions do not.  We therefore:

        1. Try to use EXP_DOTS if present (legacy behaviour).
        2. Otherwise, fall back to generating dots directly from the internal
           ShrakeRupley unit sphere, without modelling intra-chain occlusion.
        """
        sr = ShrakeRupley(probe_radius=1.4, n_points=dots)

        # --- Legacy path: EXP_DOTS available in atom.xtra (old Biopython) ---
        s = Structure.Structure("Z"); m = Model.Model(0); s.add(m); cz = Chain.Chain("Z"); m.add(cz)
        for r in chain_res:
            cz.add(r.copy())
        try:
            sr.compute(s, level="A")
        except Exception:
            # Ignore and fall through to modern code path
            pass

        pts = []
        for r in cz:
            for a in r:
                data = getattr(a, "xtra", {}).get("EXP_DOTS", [])
                for (x, y, z, nx, ny, nz) in data:
                    pts.append((np.array([x, y, z]), np.array([nx, ny, nz])))

        others = [a for rr in other_res for a in rr.get_atoms() if a.element.upper() != "H"]
        if pts and others:
            coords = np.array([a.coord for a in others]); d2 = dist**2
            buried = []
            for xyz, n in pts:
                if np.any(np.sum((coords - xyz)**2, axis=1) <= d2):
                    buried.append((xyz, n))
            return buried

        # --- Modern path: no EXP_DOTS; construct candidate dots from ShrakeRupley sphere ---
        atoms = [a for r in chain_res for a in r.get_atoms() if a.element.upper() != "H"]
        if not atoms or not others:
            return []

        atom_coords = np.array([a.coord for a in atoms])
        other_coords = np.array([a.coord for a in others])
        d2 = dist**2
        sphere = np.array(sr._sphere, copy=False)

        buried = []
        for atom, center in zip(atoms, atom_coords):
            # Approximate surface points on this atom
            radius = sr.radii_dict[atom.element] + sr.probe_radius
            points = sphere * radius + center  # (dots, 3)
            normals = sphere  # unit normals for each point

            # Compute squared distances from each point to all atoms in other_res
            diff = points[:, None, :] - other_coords[None, :, :]
            dist2 = np.sum(diff * diff, axis=2)  # (dots, n_other)
            mask = np.any(dist2 <= d2, axis=1)

            for xyz, n in zip(points[mask], normals[mask]):
                buried.append((xyz.astype(float), n.astype(float)))

        return buried

    def _approx_sc(self, A, B, w=0.5) -> float:
        cB = np.array([p[0] for p in B]); nB = [p[1] for p in B]
        sA = []
        for x, nA in A:
            d = cB - x; j = np.argmin(np.sum(d*d, axis=1))
            sA.append(float(np.dot(nA, -nB[j]) * math.exp(-w * float(np.sum(d[j]*d[j])))))
        cA = np.array([p[0] for p in A]); nA = [p[1] for p in A]
        sB = []
        for x, n in B:
            d = cA - x; j = np.argmin(np.sum(d*d, axis=1))
            sB.append(float(np.dot(n, -nA[j]) * math.exp(-w * float(np.sum(d[j]*d[j])))))
        return float(0.5 * (np.median(sA) + np.median(sB)))
