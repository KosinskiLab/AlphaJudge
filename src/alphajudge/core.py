from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional, Dict, Tuple, Any, Set
import math, numpy as np
from Bio.PDB import NeighborSearch, Structure, Model, Chain
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
    "ARG": {"NE", "CZ", "NH1", "NH2"},  # guanidinium group
    "LYS": {"NZ"},                        # amino group
    "ASP": {"CG", "OD1", "OD2"},         # carboxylate group
    "GLU": {"CD", "OE1", "OE2"},         # carboxylate group
}

# ---- Shrake-Rupley helpers ----
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

# ---- dockQ constants ----
def _sigmoid(x: float, L: float, x0: float, k: float, b: float) -> float:
    return L / (1 + math.exp(-k * (x - x0))) + b

@dataclass(frozen=True)
class DockQParams:
    """Parameters for the DockQ-style logistic scoring function."""

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

# ---- complex and interfaces ----
class Complex:
    def __init__(self, structure, confidence: Confidence, contact_thresh: float, pae_filter: float):
        self.structure = structure
        self.conf = confidence
        self.contact_thresh = contact_thresh
        self.pae_filter = pae_filter
        self._res_index_map, self._chain_indices_by_id, self._chains = self._build_maps()

        self._sasa_cache: Dict[str, float] = {}
        self._sasa_complex_cache: Dict[Tuple[str, str], float] = {}
        self._buried_surface_cache: Dict[Tuple[str, str, float, int], list[tuple[np.ndarray, np.ndarray]]] = {}
        self._contact_ns_cache: Dict[Tuple[str, str], Tuple[list, list, np.ndarray, np.ndarray]] = {}
        self._all_atom_ns_cache: Dict[Tuple[str, str], Tuple[list, NeighborSearch]] = {}

        self.interfaces: List[Interface] = []
        for i in range(len(self._chains)):
            for j in range(i+1, len(self._chains)):
                iface = Interface(self._chains[i], self._chains[j], self)
                if iface.num_intf_residues > 0:
                    self.interfaces.append(iface)

    # ---------- maps & residue utilities ----------
    def _build_maps(self) -> tuple[Dict[Tuple[str, Any], int], Dict[str, List[int]], List[Chain]]:
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

        # Pre-compute chain IDs and their residue index arrays for fast LIS/ipSAE.
        self._cid1_id = self.chain1[0].get_parent().id
        self._cid2_id = self.chain2[0].get_parent().id
        self._idx1 = np.asarray(self._cid.get(self._cid1_id, []), dtype=int)
        self._idx2 = np.asarray(self._cid.get(self._cid2_id, []), dtype=int)

        # Flag whether either chain contains nucleic acids (used in ipSAE).
        self._has_na = any(
            r.get_resname().strip() in NA_RES for r in (self.chain1 + self.chain2)
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
    def contact_pairs(self) -> int:
        return len(self._pairs)

    @cached_property
    def pDockQ(self) -> float:
        """Predicted DockQ score for this interface, using plDDT and contact pairs."""
        if self.contact_pairs <= 0 or math.isnan(self._avg_plddt): return float('nan')
        return PDOCKQ.score(self._avg_plddt * math.log10(self.contact_pairs))

    def pDockQ2(self) -> tuple[float, float]:
        """Alternative DockQ-style score using pairwise PAE-derived ipTM values."""
        vals = self._ptm_values()
        if not vals or math.isnan(self._avg_plddt): return float('nan'), 0.0
        mean_ptm = float(np.mean(vals))
        return PDOCKQ2.score(self._avg_plddt * mean_ptm), mean_ptm

    def ipsae(self, pae_cutoff: float = 10.0) -> float:
        return self._ipsae_asym(pae_cutoff)

    def lis(self) -> float:
        """
        Symmetrised LIS score between the two chains.

        Directional LIS for (A,B) is the mean of (12 - PAE) / 12 over all
        residue pairs (i in A, j in B) with PAE(i,j) <= 12 Å.  ipsae.py reports
        the symmetric variant:

            LIS_score = (LIS[A][B] + LIS[B][A]) / 2
        """

        def _lis_dir(idx_src: np.ndarray, idx_dst: np.ndarray) -> float:
            if idx_src.size == 0 or idx_dst.size == 0:
                return float('nan')
            sub = self._pae[np.ix_(idx_src, idx_dst)]
            valid = sub[sub <= 12.0]
            if valid.size == 0:
                return float('nan')
            return float(np.mean((12.0 - valid) / 12.0))

        a = _lis_dir(self._idx1, self._idx2)
        b = _lis_dir(self._idx2, self._idx1)

        if math.isnan(a) and math.isnan(b):
            return float('nan')
        if math.isnan(a):
            return b
        if math.isnan(b):
            return a
        return float(0.5 * (a + b))

    # composition
    @property
    def polar(self) -> float:
        """Fraction of polar residues at the interface."""
        return self._frac(POLAR_RES)

    @property
    def hydrophobic(self) -> float:
        """Fraction of hydrophobic residues at the interface."""
        return self._frac(HYDROPHOBIC_RES)

    @property
    def charged(self) -> float:
        """Fraction of charged residues at the interface."""
        return self._frac(CHARGED_RES)

    # quick “complex” score
    @cached_property
    def score_complex(self) -> float:
        """Raw plDDT × log10(contact_pairs) score used as a simple complex metric."""
        if self.contact_pairs <= 0 or math.isnan(self._avg_plddt): return float('nan')
        return self._avg_plddt * math.log10(self.contact_pairs)

    # HB / SB / SC / areas (self-contained helpers)
    @cached_property
    def _ns_all_atoms(self) -> NeighborSearch:
        """NeighborSearch over all atoms in both chains (reused by hb/sb)."""
        key = tuple(sorted((self._cid1_id, self._cid2_id)))
        cached = self.c._all_atom_ns_cache.get(key)
        if cached is None:
            atoms = [a for r in (self.chain1 + self.chain2) for a in r]
            cached = (atoms, NeighborSearch(atoms))
            self.c._all_atom_ns_cache[key] = cached
        return cached[1]

    @staticmethod
    def _angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two vectors in degrees."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return math.degrees(math.acos(cos_angle))

    @staticmethod
    def _angle_at_point(p1: np.ndarray, vertex: np.ndarray, p2: np.ndarray) -> float:
        """Calculate angle p1-vertex-p2 in degrees."""
        v1 = p1 - vertex
        v2 = p2 - vertex
        return Interface._angle_between_vectors(v1, v2)

    @staticmethod
    def _find_hydrogen_atoms(donor_atom) -> List[Any]:
        """
        Find hydrogen atoms attached to a donor atom.
        Uses distance-based detection (H-D distance ~1.0-1.1 Å) as primary method,
        with name-based fallback for robustness.
        """
        hydrogens = []
        residue = donor_atom.get_parent()
        if residue is None:
            return hydrogens
        donor_coord = donor_atom.coord
        donor_name = donor_atom.id.upper()
        element = donor_atom.element.upper() if donor_atom.element else ""
        
        # Typical H-D bond lengths: N-H ~1.0 Å, O-H ~1.0 Å
        max_h_dist = 1.3  # Slightly generous to account for variations
        
        for atom in residue:
            if not (atom.element and atom.element.upper() == "H"):
                continue
            
            # Distance-based detection (most reliable)
            dist = np.linalg.norm(atom.coord - donor_coord)
            if dist <= max_h_dist:
                hydrogens.append(atom)
                continue
            
            # Name-based fallback for cases where distance might be off
            atom_name = atom.id.upper()
            # Main chain N -> H, 1H, 2H, 3H
            if donor_name == "N" and atom_name in ("H", "1H", "2H", "3H", "HN"):
                hydrogens.append(atom)
            # Side chain patterns: NE -> HE, ND -> HD, etc.
            elif len(donor_name) >= 2:
                # Pattern: if donor is "NE", look for "HE", "1HE", "2HE", etc.
                if atom_name.startswith("H") and donor_name[1:] in atom_name[1:]:
                    hydrogens.append(atom)
                # Pattern: if donor is "OG", look for "HG", "1HG", etc.
                elif atom_name.startswith("H") and donor_name[-1] in atom_name:
                    # Additional check: distance should be reasonable even if > max_h_dist
                    if dist <= 1.5:  # More lenient for name-based
                        hydrogens.append(atom)
        
        return hydrogens

    @staticmethod
    def _find_acceptor_antecedent(acceptor_atom) -> Optional[Any]:
        """Find the acceptor antecedent (heavy atom attached to acceptor)."""
        residue = acceptor_atom.get_parent()
        if residue is None:
            return None
        acceptor_name = acceptor_atom.id.upper()
        acceptor_coord = acceptor_atom.coord
        
        # Find the closest heavy atom (not H) in the same residue
        closest_atom = None
        min_dist = float('inf')
        for atom in residue:
            if atom is acceptor_atom or (atom.element and atom.element.upper() == "H"):
                continue
            dist = np.linalg.norm(atom.coord - acceptor_coord)
            if dist < min_dist and dist < 2.0:  # Covalent bond distance
                min_dist = dist
                closest_atom = atom
        return closest_atom

    @staticmethod
    def _can_be_donor(atom) -> bool:
        """
        Check if an atom can act as a hydrogen bond donor.
        Based on HBPLUS donor list:
        - Main chain N (all amino acids except imino acids like PRO)
        - Side chain N: ARG NE/NH1/NH2, LYS NZ, ASN ND2, GLN NE2, HIS NE2/ND1, TRP NE1
        - Side chain O: SER OG, THR OG1, TYR OH
        - CYH SG (cysteine with H)
        """
        element = atom.element.upper() if atom.element else ""
        atom_name = atom.id.upper()
        residue = atom.get_parent()
        if residue is None:
            return False
        resname = residue.get_resname().upper()
        
        # Main chain N (all amino acids except PRO which is imino)
        if element == "N" and atom_name == "N" and resname != "PRO":
            return True
        
        # Side chain N donors
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
        
        # Side chain O donors (OH groups)
        if element == "O":
            if resname == "SER" and atom_name == "OG":
                return True
            if resname == "THR" and atom_name == "OG1":
                return True
            if resname == "TYR" and atom_name == "OH":
                return True
        
        # CYH SG (cysteine with H) - less common, but included for completeness
        if element == "S" and resname == "CYS" and atom_name == "SG":
            # Note: This would require checking for H, but we'll include it
            # as CYH is a recognized donor type in HBPLUS
            return True
        
        # Water molecules and HETATM: O atoms can be donors if they have H
        # For water, we'll be permissive - O in HOH can be donor
        if element == "O" and resname in ("HOH", "WAT", "H2O"):
            return True
        
        # For unknown/non-standard residues, be permissive for N and O
        # (fallback for HETATM or modified residues not in standard list)
        if resname not in ("ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", 
                          "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
                          "THR", "TRP", "TYR", "VAL", "HOH", "WAT", "H2O"):
            if element == "N":
                return True
            if element == "O":
                return True
        
        return False

    @staticmethod
    def _can_be_acceptor(atom) -> bool:
        """
        Check if an atom can act as a hydrogen bond acceptor.
        Based on HBPLUS acceptor list:
        - Main chain carbonyl O (all amino acids, including PRO)
        - Side chain O: ASP OD1/OD2, GLU OE1/OE2, ASN OD1, GLN OE1, SER OG, THR OG1, TYR OH
        - Side chain N: HIS ND1/NE2 (can be acceptors)
        - CYH SG, CSS SG (cysteine/cystine sulfur)
        """
        element = atom.element.upper() if atom.element else ""
        atom_name = atom.id.upper()
        residue = atom.get_parent()
        if residue is None:
            return False
        resname = residue.get_resname().upper()
        
        # Main chain carbonyl O (all amino acids, including PRO)
        if element == "O" and atom_name == "O":
            return True
        
        # Terminal carboxylate oxygen (OXT)
        if element == "O" and atom_name == "OXT":
            return True

        # Side chain O acceptors
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
        
        # Side chain N acceptors (HIS nitrogens)
        if element == "N":
            if resname == "HIS" and atom_name in ("ND1", "NE2"):
                return True
        
        # CYH SG, CSS SG (cysteine/cystine sulfur) - less common
        if element == "S" and resname == "CYS" and atom_name == "SG":
            return True
        
        # Water molecules and HETATM: O atoms are acceptors
        if element == "O" and resname in ("HOH", "WAT", "H2O"):
            return True
        
        # For unknown/non-standard residues, be permissive for O and N
        # (fallback for HETATM or modified residues not in standard list)
        if resname not in ("ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", 
                          "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
                          "THR", "TRP", "TYR", "VAL", "HOH", "WAT", "H2O"):
            if element == "O":
                return True
            if element == "N":
                return True
        
        return False

    @cached_property
    def hb(self) -> int:
        """
        Count hydrogen bonds between chain1 and chain2.
        Uses HBPLUS-style criteria:
        - D-A distance <= 3.9 Å
        - H-A distance <= 2.5 Å (if H present)
        - D-H-A angle >= 90° (if H present)
        - D-A-AA angle >= 90°
        - H-A-AA angle >= 90° (if H and AA present)
        """
        # Candidate atoms include N/O/S to cover typical donors/acceptors (e.g. SG)
        def _candidate_atoms(chain):
            out = []
            for res in chain:
                for atom in res:
                    elem = atom.element.upper() if atom.element else ""
                    if elem in {"N", "O", "S"}:
                        out.append(atom)
            return out

        atoms1 = _candidate_atoms(self.chain1)
        atoms2 = _candidate_atoms(self.chain2)
        if not atoms1 or not atoms2: return 0
        ids1 = {id(a) for a in atoms1}
        ids2 = {id(a) for a in atoms2}
        ns = self._ns_all_atoms
        
        # HBPLUS default distances
        max_da_dist = 3.9  # Donor-Acceptor
        max_ha_dist = 2.5  # Hydrogen-Acceptor
        min_angle = 90.0   # Minimum angle in degrees
        
        seen, cnt = set(), 0
        for a1, a2 in ns.search_all(max_da_dist):
            i1, i2 = id(a1), id(a2)
            if not ((i1 in ids1 and i2 in ids2) or (i1 in ids2 and i2 in ids1)):
                continue
            
            # Determine donor and acceptor
            donor, acceptor = None, None
            if self._can_be_donor(a1) and self._can_be_acceptor(a2):
                donor, acceptor = a1, a2
            elif self._can_be_donor(a2) and self._can_be_acceptor(a1):
                donor, acceptor = a2, a1
            elif self._can_be_donor(a1) and self._can_be_donor(a2):
                # Both can be donors - check if either can be acceptor
                if self._can_be_acceptor(a1):
                    donor, acceptor = a2, a1
                elif self._can_be_acceptor(a2):
                    donor, acceptor = a1, a2
            elif self._can_be_acceptor(a1) and self._can_be_acceptor(a2):
                # Both can be acceptors - check if either can be donor
                if self._can_be_donor(a1):
                    donor, acceptor = a1, a2
                elif self._can_be_donor(a2):
                    donor, acceptor = a2, a1
            
            if donor is None or acceptor is None:
                continue
            
            # Check D-A distance (already filtered by search_all, but verify)
            da_dist = np.linalg.norm(donor.coord - acceptor.coord)
            if da_dist > max_da_dist:
                continue
            
            # Find hydrogen atoms attached to donor
            hydrogens = self._find_hydrogen_atoms(donor)
            
            # Find acceptor antecedent
            acceptor_antecedent = self._find_acceptor_antecedent(acceptor)
            
            # Check angle criteria
            valid_hbond = False
            
            if hydrogens:
                # If hydrogen is present, check D-H-A and H-A-AA angles
                for h in hydrogens:
                    ha_dist = np.linalg.norm(h.coord - acceptor.coord)
                    if ha_dist > max_ha_dist:
                        continue
                    
                    # D-H-A angle
                    dha_angle = self._angle_at_point(donor.coord, h.coord, acceptor.coord)
                    if dha_angle < min_angle:
                        continue
                    
                    # D-A-AA angle (if AA exists)
                    if acceptor_antecedent is not None:
                        daaa_angle = self._angle_at_point(donor.coord, acceptor.coord, acceptor_antecedent.coord)
                        if daaa_angle < min_angle:
                            continue
                        
                        # H-A-AA angle
                        haaa_angle = self._angle_at_point(h.coord, acceptor.coord, acceptor_antecedent.coord)
                        if haaa_angle < min_angle:
                            continue
                    else:
                        # No acceptor antecedent (e.g. water); rely on D-H-A + H-A distance only
                        pass
                    
                    valid_hbond = True
                    break
            else:
                # No hydrogen present - infer H position along D-A vector (HBPLUS approach)
                # Place H at typical bond distance (~1.0 Å) from donor along D->A direction
                da_vec = acceptor.coord - donor.coord
                da_vec_norm = np.linalg.norm(da_vec)
                if da_vec_norm > 0:
                    h_bond_length = 1.0  # Typical N-H or O-H bond length
                    inferred_h_coord = donor.coord + (da_vec / da_vec_norm) * h_bond_length
                    
                    # Check H-A distance
                    ha_dist = np.linalg.norm(inferred_h_coord - acceptor.coord)
                    if ha_dist <= max_ha_dist:
                        # D-H-A angle (using inferred H)
                        dha_angle = self._angle_at_point(donor.coord, inferred_h_coord, acceptor.coord)
                        if dha_angle >= min_angle:
                            # D-A-AA angle (if AA exists)
                            if acceptor_antecedent is not None:
                                daaa_angle = self._angle_at_point(donor.coord, acceptor.coord, acceptor_antecedent.coord)
                                if daaa_angle >= min_angle:
                                    # H-A-AA angle
                                    haaa_angle = self._angle_at_point(inferred_h_coord, acceptor.coord, acceptor_antecedent.coord)
                                    if haaa_angle >= min_angle:
                                        valid_hbond = True
                            else:
                                # No AA, but D-H-A angle and H-A distance are satisfied
                                valid_hbond = True
                
                # Fallback: if no H inferred or checks failed, rely on D-A-AA angle
                # (distance constraint already enforced via max_da_dist)
                if not valid_hbond and acceptor_antecedent is not None:
                    daaa_angle = self._angle_at_point(donor.coord, acceptor.coord, acceptor_antecedent.coord)
                    if daaa_angle >= min_angle:
                        valid_hbond = True
            
            if valid_hbond:
                key = tuple(sorted((i1, i2)))
                if key not in seen:
                    seen.add(key)
                    cnt += 1
        
        return cnt

    @cached_property
    def sb(self) -> int:
        pos, neg = {"ARG","LYS"}, {"ASP","GLU"}
        relevant = pos | neg
        a1 = [a for r in self.chain1 if r.get_resname() in relevant 
              for a in r if a.id in CHARGED_ATOMS.get(r.get_resname(), set())]
        a2 = [a for r in self.chain2 if r.get_resname() in relevant 
              for a in r if a.id in CHARGED_ATOMS.get(r.get_resname(), set())]
        if not a1 or not a2: return 0
        ns = self._ns_all_atoms; cutoff = 4.0
        ids1 = {id(a): 1 if a.get_parent().get_resname() in pos else -1 for a in a1}
        ids2 = {id(a): 1 if a.get_parent().get_resname() in pos else -1 for a in a2}
        seen, cnt = set(), 0
        for x, y in ns.search_all(cutoff):
            ix, iy = id(x), id(y)
            if ix in ids1 and iy in ids2:
                if ids1[ix] + ids2[iy] == 0:
                    key = tuple(sorted((ix, iy)))
                    if key not in seen:
                        seen.add(key); cnt += 1
            elif ix in ids2 and iy in ids1:
                if ids2[ix] + ids1[iy] == 0:
                    key = tuple(sorted((ix, iy)))
                    if key not in seen:
                        seen.add(key); cnt += 1
        return cnt

    @cached_property
    def sc(self) -> float:
        """Shape complementarity score based on buried surface points (Lawrence & Colman-style)."""
        sA = self._buried_surface(self.chain1, self.chain2, 5.0, 15)
        sB = self._buried_surface(self.chain2, self.chain1, 5.0, 15)
        if not sA or not sB: return 0.0
        return self._approx_sc(sA, sB, w=0.5)

    @cached_property
    def int_area(self) -> float:
        """Buried solvent-accessible surface area at the interface (Å^2)."""
        return self._sasa_chain(self.chain1) + self._sasa_chain(self.chain2) - self._sasa_complex(self.chain1, self.chain2)

    @cached_property
    def int_solv_en(self) -> float:
        """Crude solvation free-energy term proportional to buried area (negative is stabilising)."""
        return -0.0072 * self.int_area

    # ---------- private helpers below ----------
    def _get_pairs(self):
        res_pairs: Set[Tuple[Any, Any]] = set()
        a1, a2, coords1, coords2 = self._contact_atom_data()
        if not len(a1) or not len(a2):
            return set(), set(), res_pairs
        diff = coords1[:, None, :] - coords2[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        mask = dist2 <= self.c.contact_thresh ** 2
        idx_i, idx_j = np.where(mask)
        for i, j in zip(idx_i.tolist(), idx_j.tolist()):
            res_pairs.add((a1[i].get_parent(), a2[j].get_parent()))
        r1 = {p[0] for p in res_pairs}; r2 = {p[1] for p in res_pairs}
        return r1, r2, res_pairs

    def _contact_atom_data(self) -> Tuple[list, list, np.ndarray, np.ndarray]:
        key = (self._cid1_id, self._cid2_id)
        cached = self.c._contact_ns_cache.get(key)
        if cached is None:
            a1 = [r["CB"] if "CB" in r else r["CA"] for r in self.chain1]
            a2 = [r["CB"] if "CB" in r else r["CA"] for r in self.chain2]
            coords1 = np.array([a.coord for a in a1], dtype=float) if a1 else np.empty((0, 3))
            coords2 = np.array([a.coord for a in a2], dtype=float) if a2 else np.empty((0, 3))
            cached = (a1, a2, coords1, coords2)
            self.c._contact_ns_cache[key] = cached
        return cached

    def _avg_plddt_union(self) -> float:
        res_set = self._res1 | self._res2
        if not res_set:
            return float('nan')
        vals = [
            (r["CB"] if "CB" in r else r["CA"]).get_bfactor()
            for r in res_set
        ]
        return float(sum(vals) / len(vals))

    def _avg_pae_over_pairs(self) -> float:
        vals = []
        for r1, r2 in self._pairs:
            k1 = (r1.get_parent().id, r1.id); i = self._rim.get(k1)
            k2 = (r2.get_parent().id, r2.id); j = self._rim.get(k2)
            if i is None or j is None: continue
            try:
                vals.append(float(self._pae[i, j]))
                vals.append(float(self._pae[j, i]))
            except (IndexError, TypeError, ValueError):
                continue
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
            except (IndexError, TypeError, ValueError):
                continue
        return out

    def _ipsae_asym(self, cutoff: float) -> float:
        """
        Asymmetric ipSAE score, matching ipsae_d0res_asym from ipsae.py.

        For each direction (src -> dst) we:
        - for every residue i in src, collect all j in dst with PAE(i,j) < cutoff
        - let n = number of such pairs (i,j); this is n0res_byres[i]
        - compute a residue-specific d0(i) with a minimum of:
            * 1.0 Å for pure protein–protein interfaces
            * 2.0 Å if either chain contains nucleic acids
        - compute ptm(i) = mean_j 1 / (1 + (PAE(i,j) / d0(i))**2)
        - take the max over residues i in src.

        The public ipsae() then returns max over the two directions, matching
        ipsae_d0res_max for protein-only systems.
        """

        def calc(idx_src: np.ndarray, idx_dst: np.ndarray) -> float:
            if idx_src.size == 0 or idx_dst.size == 0:
                return float('nan')

            # Any nucleic acids in this interface?
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
                best, found = max(best, float(np.mean(ptm))), True
            return best if found else float('nan')

        a = calc(self._idx1, self._idx2)
        b = calc(self._idx2, self._idx1)
        if math.isnan(a) and math.isnan(b): return float('nan')
        if math.isnan(a): return b
        if math.isnan(b): return a
        return max(a, b)

    def _frac(self, names: set[str]) -> float:
        residues = self._res1 | self._res2
        return (sum(1 for r in residues if r.get_resname() in names) / len(residues)) if residues else 0.0

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
        except ValueError:
            # If SASA computation fails (e.g. no atoms), treat as zero area
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
        except ValueError:
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

        # Reuse precomputed Shrake-Rupley sphere/radii when compatible; fall
        # back to the instance-specific values otherwise.
        if dots == 15:
            sphere = _SPHERE_15
            radii_dict = _RADII_DICT_15
            probe_radius = _SR_PROBE_RADIUS
        else:
            sphere = np.array(sr._sphere, copy=False)
            radii_dict = sr.radii_dict
            probe_radius = float(sr.probe_radius)

        buried = []
        atom_radii = np.array([radii_dict[atom.element] + probe_radius for atom in atoms])
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
            d_j_sq = float(d_sq[j])
            sA.append(float(np.dot(nA_vec, -nB[j]) * math.exp(-w * d_j_sq)))
        sB = []
        for x, n_vec in B:
            d = cA - x
            d_sq = np.sum(d * d, axis=1)
            j = int(np.argmin(d_sq))
            d_j_sq = float(d_sq[j])
            sB.append(float(np.dot(n_vec, -nA[j]) * math.exp(-w * d_j_sq)))
        return float(0.5 * (np.median(sA) + np.median(sB)))
