from pathlib import Path
import math
import json
import csv
import argparse
import logging
from typing import Any, Dict, List, Tuple, Set, Optional
from functools import cached_property
import enum
import numpy as np
from dataclasses import dataclass

from Bio.PDB import (
    PDBParser,
    MMCIFParser,
    NeighborSearch,
    Structure,
    Model,
    Chain,
)
from Bio.PDB.Residue import Residue  # type: ignore
from Bio.PDB.SASA import ShrakeRupley  # For SASA computations
from Bio.PDB.Atom import Atom

##################################################
# DockQ constants & data structures
##################################################

def _sigmoid(value: float, L: float, x0: float, k: float, b: float) -> float:
    """Return the sigmoid of 'value' using the given parameters."""
    return L / (1 + math.exp(-k * (value - x0))) + b

@dataclass(frozen=True)
class DockQConstants:
    """
    Dataclass to hold the L, X0, K, B constants for
    each DockQ-type score (pDockQ, pDockQ2, mpDockQ).
    """
    L: float
    X0: float
    K: float
    B: float

    def score(self, x: float) -> float:
        """Applies the stored constants to compute the sigmoid for x."""
        return _sigmoid(x, self.L, self.X0, self.K, self.B)

# ------------------------------------------------------------------------------
# Constants for interface-level pDockQ
# https://www.nature.com/articles/s41467-022-28865-w
PDOCKQ_CONSTANTS = DockQConstants(L=0.724, X0=152.611, K=0.052, B=0.018)

# ------------------------------------------------------------------------------
# Constants for interface-level pDockQ2
# https://academic.oup.com/bioinformatics/article/39/7/btad424/7219714
# Also uses D0 for the PAE transformation
D0 = 10.0
PDOCKQ2_CONSTANTS = DockQConstants(L=1.31, X0=84.733, K=0.075, B=0.005)

# ------------------------------------------------------------------------------
# Constants for mpDockQ (global complex scores)
# https://www.nature.com/articles/s41467-022-33729-4
MPDOCKQ_CONSTANTS = DockQConstants(L=0.728, X0=309.375, K=0.098, B=0.262)
# ------------------------------------------------------------------------------


@enum.unique
class ModelsToAnalyse(enum.Enum):
    BEST = 0
    ALL = 1

DEFAULT_CONTACT_THRESH = 8.0
DEFAULT_PAE_FILTER = 100.0

##################################################
# Simple reading / parsing helpers
##################################################

def extract_job_name(path_to_dir: str) -> str:
    """Use the basename of the directory as the job name."""
    return Path(path_to_dir).resolve().name

def read_json(filepath: str) -> Any:
    p = Path(filepath)
    with p.open() as f:
        data = json.load(f)
    logging.info("Loaded JSON file: %s", str(p))
    return data

def parse_ranking_debug_json_all(directory: str) -> Dict[str, Any]:
    path = Path(directory) / "ranking_debug.json"
    data = read_json(str(path))
    if "order" not in data or not isinstance(data["order"], list):
        raise ValueError("Invalid ranking_debug.json: missing or invalid 'order' key")
    return data

def get_ranking_metric_for_model(data: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Return ranking metrics for a model."""
    has_multimer = ("iptm+ptm" in data) and ("iptm" in data)
    has_monomer = ("plddts" in data) and ("ptm" in data)

    if has_multimer:
        if model not in data["iptm+ptm"] or model not in data["iptm"]:
            raise ValueError(f"Model '{model}' not found in multimer metrics")
        return {
            "model": model,
            "iptm+ptm": data["iptm+ptm"][model],
            "iptm": data["iptm"][model],
            "multimer": True,
        }
    elif has_monomer:
        if model not in data["plddts"] or model not in data["ptm"]:
            raise ValueError(f"Model '{model}' not found in monomer metrics")
        return {
            "model": model,
            "plddts": data["plddts"][model],
            "ptm": data["ptm"][model],
            "multimer": False,
        }
    else:
        raise ValueError("Invalid ranking_debug.json: expected multimer or monomer keys not found")

def load_pae_file(directory: str, model: str) -> Dict[str, Any]:
    pae_filename = f"pae_{model}.json"
    path = Path(directory) / pae_filename
    if not path.exists():
        raise FileNotFoundError(f"PAE file '{pae_filename}' not found in directory '{directory}'.")
    return read_json(str(path))

def find_structure_file(directory: str, model: str) -> str:
    d = Path(directory)
    cif_matches = list(d.glob(f"*{model}*.cif"))
    if cif_matches:
        return str(cif_matches[0])
    pdb_matches = list(d.glob(f"*{model}*.pdb"))
    if pdb_matches:
        return str(pdb_matches[0])
    raise ValueError(f"No structure file (CIF or PDB) found for model '{model}' in directory.")

##################################################
# Precomputation helpers
##################################################

def compute_interface_avg_plddt(
    precomputed: Tuple[Set[Any], Set[Any], Set[Tuple[Any, Any]]]
) -> float:
    """Compute the average interface pLDDT (proxy via CA B-factor) over the union of interface residues."""
    res_set = precomputed[0].union(precomputed[1])
    bvals = []
    for res in res_set:
        try:
            atom = res["CB"]
        except KeyError:
            atom = res["CA"]
        bvals.append(atom.get_bfactor())
    return sum(bvals) / len(bvals) if bvals else float('nan')

def compute_interface_avg_pae(
    precomputed: Tuple[Set[Any], Set[Any], Set[Tuple[Any, Any]]],
    pae_matrix: List[List[float]],
    res_index_map: Dict[Tuple[str, Any], int]
) -> float:
    """Compute the average PAE over all contact pairs (both directions)."""
    pae_values = []
    for res1, res2 in precomputed[2]:
        key1 = (res1.get_parent().id, res1.id)
        key2 = (res2.get_parent().id, res2.id)
        i = res_index_map.get(key1)
        j = res_index_map.get(key2)
        if i is None or j is None:
            continue
        try:
            val1 = float(pae_matrix[i][j])
            val2 = float(pae_matrix[j][i])
            pae_values.extend([val1, val2])
        except Exception:
            continue
    return sum(pae_values) / len(pae_values) if pae_values else float('nan')

##################################################
# Simple shape complementarity from Lawrence & Colman (1993)
# Approximation (toy).
##################################################

def approximate_sc_lc93(surfaceA, surfaceB, w=0.5):
    """
    Partial re-implementation of shape complementarity from:
      Lawrence, M. C. & Colman, P. M. (1993).
      Shape complementarity at protein/protein interfaces.
      J. Mol. Biol. 234, 946–950.

    We compute for each surface point xA in A:
       S_{A->B}(xA) = (nA . -nB') * exp[-w*(dist^2)]
    where xB' is the nearest surface point in B, nB' is that point's normal.
    Then we take the median over xA, do the same for B->A, average them => Sc.

    We skip the 1.5 Å "peripheral band" step for brevity and rely on
    approximate "buried-surface" identification.
    """
    if not surfaceA or not surfaceB:
        return 0.0

    coordsB = np.array([pt[0] for pt in surfaceB])
    normalsB = [pt[1] for pt in surfaceB]

    scoresA = []
    for (xyzA, normA) in surfaceA:
        diff = coordsB - xyzA
        dist_sq = np.sum(diff * diff, axis=1)
        idx_min = np.argmin(dist_sq)
        best_dist = math.sqrt(dist_sq[idx_min])
        best_normB = normalsB[idx_min]
        dotval = np.dot(normA, -best_normB)
        local_score = dotval * math.exp(-w * (best_dist ** 2))
        scoresA.append(local_score)

    coordsA = np.array([pt[0] for pt in surfaceA])
    normalsA = [pt[1] for pt in surfaceA]
    scoresB = []
    for (xyzB, normB) in surfaceB:
        diff = coordsA - xyzB
        dist_sq = np.sum(diff * diff, axis=1)
        idx_min = np.argmin(dist_sq)
        best_dist = math.sqrt(dist_sq[idx_min])
        best_normA = normalsA[idx_min]
        dotval = np.dot(normB, -best_normA)
        local_score = dotval * math.exp(-w * (best_dist ** 2))
        scoresB.append(local_score)

    medA = np.median(scoresA)
    medB = np.median(scoresB)
    return 0.5 * (medA + medB)

def gather_buried_surface_points(chain_residues, other_chain_residues, distance_cutoff=5.0, dot_density=15):
    """
    Approximate "buried portion" of a chain's molecular surface
    by:
      1) generating surface points & normals via ShrakeRupley
      2) marking any points within 'distance_cutoff' of the other chain => "buried"

    We skip the formal "exclude a 1.5 A band" from the LC93 approach for brevity.
    Returns a list of (xyz, normal).
    """
    sr = ShrakeRupley(probe_radius=1.4, n_points=dot_density)
    struct = Structure.Structure("tempA")
    model = Model.Model(0)
    struct.add(model)
    cZ = Chain.Chain("Z")
    model.add(cZ)
    for r in chain_residues:
        cZ.add(r.copy())

    sr.compute(struct, level="A")  # per atom => "EXP_DOTS"

    surface_points = []
    for r in cZ:
        for a in r:
            dots = a.xtra.get("EXP_DOTS", [])
            for (x, y, z, nx, ny, nz) in dots:
                surface_points.append((np.array([x, y, z]), np.array([nx, ny, nz])))

    # Gather heavy atoms from the other chain
    other_atoms = []
    for rr in other_chain_residues:
        for atm in rr.get_atoms():
            if atm.element.upper() != "H":
                other_atoms.append(atm)

    if not surface_points or not other_atoms:
        return []

    coords_other = np.array([atm.coord for atm in other_atoms])
    cutoff_sq = distance_cutoff ** 2

    buried = []
    for (xyz, norm) in surface_points:
        diff = coords_other - xyz
        dist_sq = np.sum(diff * diff, axis=1)
        if np.any(dist_sq <= cutoff_sq):
            buried.append((xyz, norm))

    return buried

##################################################
# InterfaceAnalysis
##################################################

class InterfaceAnalysis:
    """
    Represents a single interface between two chains.
    Calculates:
      - average PAE, pLDDT
      - pDockQ, pDockQ2, ipSAE, LIS
      - shape complementarity (approx)
      - interface area & solvation energy (via ShrakeRupley)
      - hydrogen bonds (hb)
      - salt bridges (sb)
    """
    def __init__(
        self,
        chain1: List[Any],
        chain2: List[Any],
        contact_thresh: float,
        pae_matrix: List[List[float]],
        res_index_map: Dict[Tuple[str, Any], int],
        chain_indices_by_id: Optional[Dict[str, List[int]]] = None,
    ) -> None:
        self.chain1 = chain1
        self.chain2 = chain2
        self.contact_thresh = contact_thresh
        self._pae_matrix = pae_matrix
        self._res_index_map = res_index_map
        self._chain_indices_by_id = chain_indices_by_id

        # Identify interacting residues and contact pairs once
        self.precomputed = self._get_interface_residues()
        self.interface_residues_chain1, self.interface_residues_chain2, self.pairs = self.precomputed

        # Summaries
        self._average_interface_plddt = compute_interface_avg_plddt(self.precomputed)
        self._average_interface_pae = compute_interface_avg_pae(self.precomputed, pae_matrix, res_index_map)

    def _get_interface_residues(self) -> Tuple[Set[Any], Set[Any], Set[Tuple[Any, Any]]]:
        """Identify interacting residues and contact pairs using CB/CA and neighbor search."""
        res_pairs: Set[Tuple[Any, Any]] = set()
        atoms1 = [res["CB"] if "CB" in res else res["CA"] for res in self.chain1]
        atoms2 = [res["CB"] if "CB" in res else res["CA"] for res in self.chain2]
        ns = NeighborSearch(atoms1 + atoms2)
        for atom1 in atoms1:
            neighbors = ns.search(atom1.coord, self.contact_thresh)
            for atom2 in neighbors:
                if atom2 in atoms2:
                    res_pairs.add((atom1.get_parent(), atom2.get_parent()))
        res1_set = {pair[0] for pair in res_pairs}
        res2_set = {pair[1] for pair in res_pairs}
        return (res1_set, res2_set, res_pairs)

    @cached_property
    def average_interface_plddt(self) -> float:
        return self._average_interface_plddt

    @cached_property
    def average_interface_pae(self) -> float:
        return self._average_interface_pae

    @cached_property
    def contact_pairs(self) -> int:
        return len(self.pairs)

    @cached_property
    def score_complex(self) -> float:
        """= average_interface_plddt * log10(contact_pairs)."""
        cp = self.contact_pairs
        if cp <= 0 or math.isnan(self.average_interface_plddt):
            return float('nan')
        return self.average_interface_plddt * math.log10(cp)

    @property
    def num_intf_residues(self) -> int:
        return len(self.interface_residues_chain1.union(self.interface_residues_chain2))

    @property
    def polar(self) -> float:
        polar_res = {'SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS'}
        residues = self.interface_residues_chain1.union(self.interface_residues_chain2)
        count = sum(1 for res in residues if res.get_resname() in polar_res)
        return count / len(residues) if residues else 0.0

    @property
    def hydrophobic(self) -> float:
        hydro_res = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP'}
        residues = self.interface_residues_chain1.union(self.interface_residues_chain2)
        count = sum(1 for res in residues if res.get_resname() in hydro_res)
        return count / len(residues) if residues else 0.0

    @property
    def charged(self) -> float:
        charged_res = {'ARG', 'LYS', 'ASP', 'GLU', 'HIS'}
        residues = self.interface_residues_chain1.union(self.interface_residues_chain2)
        count = sum(1 for res in residues if res.get_resname() in charged_res)
        return count / len(residues) if residues else 0.0

    # -------------------------------------------------------
    # pDockQ, pDockQ2, ipSAE, LIS
    # -------------------------------------------------------
    @cached_property
    def pDockQ(self) -> float:
        n_contacts = self.contact_pairs
        if n_contacts <= 0:
            return float('nan')
        x = self.average_interface_plddt * math.log10(n_contacts)
        return PDOCKQ_CONSTANTS.score(x)

    @cached_property
    def _ptm_values(self) -> List[float]:
        ptm_values = []
        for res1, res2 in self.pairs:
            key1 = (res1.get_parent().id, res1.id)
            key2 = (res2.get_parent().id, res2.id)
            i = self._res_index_map.get(key1)
            j = self._res_index_map.get(key2)
            if i is None or j is None:
                continue
            try:
                pae_val = float(self._pae_matrix[i][j])
                ptm_values.append(1.0 / (1 + (pae_val / D0) ** 2))
            except Exception:
                continue
        return ptm_values

    def pDockQ2(self) -> Tuple[float, float]:
        ptm_values = self._ptm_values
        if not ptm_values or math.isnan(self.average_interface_plddt):
            return float('nan'), 0.0
        mean_ptm = sum(ptm_values) / len(ptm_values)
        x = self.average_interface_plddt * mean_ptm
        pDockQ2_val = PDOCKQ2_CONSTANTS.score(x)
        return pDockQ2_val, mean_ptm

    # -------------------------------------------------------
    # ipSAE (Residue-specific D0 with PAE cutoff)
    # Adopted from https://github.com/DunbrackLab/IPSAE/blob/main/ipsae.py
    # -------------------------------------------------------
    def _pair_type(self) -> str:
        """Return 'nucleic_acid' if either side contains NA residues, else 'protein'."""
        nuc = {"DA", "DC", "DT", "DG", "A", "C", "U", "G"}
        resnames = {r.get_resname().strip() for r in self.interface_residues_chain1.union(self.interface_residues_chain2)}
        return "nucleic_acid" if (resnames & nuc) else "protein"

    @staticmethod
    def _calc_d0(n: int, pair_type: str) -> float:
        """Yang & Skolnick d0 with minimum 1.0 (protein) or 2.0 (NA)."""
        L = max(27.0, float(n))
        base = 1.24 * (L - 15.0) ** (1.0 / 3.0) - 1.8
        minv = 2.0 if pair_type == "nucleic_acid" else 1.0
        return max(minv, base)

    def _chain_indices(self, residues: List[Residue]) -> List[int]:
        idxs = [self._res_index_map.get((r.get_parent().id, r.id)) for r in residues]
        return [i for i in idxs if i is not None]

    def ipsae_d0res_asym(self, pae_cutoff: float = 10.0) -> float:
        """
        A->B asymmetric ipSAE:
        For each residue i in chain1, average ptm over j in chain2 with PAE<cutoff,
        using residue-specific D0 = d0(n_valid_j, pair_type). Return max over i.
        """
        if self._pae_matrix is None:
            return float('nan')
        idxs1 = self._chain_indices(self.chain1)
        idxs2 = self._chain_indices(self.chain2)
        if not idxs1 or not idxs2:
            return float('nan')

        arr = np.asarray(self._pae_matrix)
        pair_type = self._pair_type()
        best = 0.0
        found_any = False
        for i in idxs1:
            row = arr[i, idxs2]
            valid = row < pae_cutoff
            if not np.any(valid):
                continue
            n_valid = int(np.count_nonzero(valid))
            d0_i = self._calc_d0(n_valid, pair_type)
            ptm_vals = 1.0 / (1.0 + (row[valid] / d0_i) ** 2)
            best = max(best, float(np.mean(ptm_vals)))
            found_any = True
        return best if found_any else float('nan')

    def ipsae_d0res_max(self, pae_cutoff: float = 10.0) -> float:
        """Max over directions: max( A->B, B->A ) without re-instantiation."""
        if self._pae_matrix is None:
            return float('nan')
        arr = np.asarray(self._pae_matrix)
        pair_type = self._pair_type()

        def compute_asym(idxs_src: List[int], idxs_dst: List[int]) -> float:
            best_local = 0.0
            found = False
            for i in idxs_src:
                row = arr[i, idxs_dst]
                valid = row < pae_cutoff
                if not np.any(valid):
                    continue
                n_valid = int(np.count_nonzero(valid))
                d0_i = self._calc_d0(n_valid, pair_type)
                ptm_vals = 1.0 / (1.0 + (row[valid] / d0_i) ** 2)
                best_local = max(best_local, float(np.mean(ptm_vals)))
                found = True
            return best_local if found else float('nan')

        idxs1 = self._chain_indices(self.chain1)
        idxs2 = self._chain_indices(self.chain2)
        if not idxs1 or not idxs2:
            return float('nan')

        a_to_b = compute_asym(idxs1, idxs2)
        b_to_a = compute_asym(idxs2, idxs1)

        if math.isnan(a_to_b) and math.isnan(b_to_a):
            return float('nan')
        if math.isnan(a_to_b):
            return b_to_a
        if math.isnan(b_to_a):
            return a_to_b
        return max(a_to_b, b_to_a)

    def ipsae(self, pae_cutoff: float = 10.0) -> float:
        """
        Returns asymmetric ipSAE (A->B)
        with default PAE cutoff 10 Å.
        """
        return self.ipsae_d0res_asym(pae_cutoff)

    # -------------------------------------------------------
    # LIS (Local Interaction Score based on transform of PAEs)
    # Kim, Hu, Comjean, Rodiger, Mohr, Perrimon. https://www.biorxiv.org/content/10.1101/2024.02.19.580970v1
    # Adopted from https://github.com/DunbrackLab/IPSAE/blob/main/ipsae.py
    # -------------------------------------------------------
    def lis(self) -> float:
        """
        LIS: average of (12 - PAE)/12 over all i-j with PAE <= 12 (A->B direction).
        """
        if self._pae_matrix is None:
            return float('nan')
        cid1 = self.chain1[0].get_parent().id
        cid2 = self.chain2[0].get_parent().id
        if self._chain_indices_by_id is not None:
            indices_chain1 = self._chain_indices_by_id.get(cid1, [])
            indices_chain2 = self._chain_indices_by_id.get(cid2, [])
        else:
            indices_chain1 = self._chain_indices(self.chain1)
            indices_chain2 = self._chain_indices(self.chain2)
        if not indices_chain1 or not indices_chain2:
            return float('nan')

        arr = np.asarray(self._pae_matrix)
        submatrix = arr[np.ix_(indices_chain1, indices_chain2)]
        valid = submatrix[submatrix <= 12.0]
        if valid.size == 0:
            return float('nan')
        scores = (12.0 - valid) / 12.0
        return float(np.mean(scores))

    # -------------------------------------------------------
    # Hydrogen bonds (hb)
    # -------------------------------------------------------
    @cached_property
    def hb(self) -> int:
        """
        Naive hydrogen-bond count: any chain1 N/O atom within 3.5 Å
        of chain2 N/O atom (ignores angles, etc.).
        """
        chain1_atoms = []
        for residue in self.chain1:
            for atom in residue.get_atoms():
                aname = atom.get_id().upper()
                if aname.startswith("N") or aname.startswith("O"):
                    chain1_atoms.append(atom)

        chain2_atoms = []
        for residue in self.chain2:
            for atom in residue.get_atoms():
                aname = atom.get_id().upper()
                if aname.startswith("N") or aname.startswith("O"):
                    chain2_atoms.append(atom)

        if not chain1_atoms or not chain2_atoms:
            return 0

        cutoff = 3.5
        ns = NeighborSearch(chain1_atoms + chain2_atoms)
        visited_pairs = set()
        hbond_count = 0

        for a1 in chain1_atoms:
            neighbors = ns.search(a1.coord, cutoff)
            for a2 in neighbors:
                if a2 in chain2_atoms and a2 is not a1:
                    pair = tuple(sorted([id(a1), id(a2)]))
                    if pair not in visited_pairs:
                        visited_pairs.add(pair)
                        hbond_count += 1

        return hbond_count

    # -------------------------------------------------------
    # Salt bridges (sb)
    # -------------------------------------------------------
    @cached_property
    def sb(self) -> int:
        """
        Naive salt-bridge count: side-chain atoms of ARG/LYS within 4.0 Å
        of side-chain atoms of ASP/GLU. Ignores orientation.
        """
        pos_res = {"ARG", "LYS"}
        neg_res = {"ASP", "GLU"}

        chain1_atoms = []
        for residue in self.chain1:
            rname = residue.get_resname()
            if rname in pos_res or rname in neg_res:
                for atom in residue.get_atoms():
                    if atom.get_id() not in ("N", "CA", "C", "O"):
                        chain1_atoms.append(atom)

        chain2_atoms = []
        for residue in self.chain2:
            rname = residue.get_resname()
            if rname in pos_res or rname in neg_res:
                for atom in residue.get_atoms():
                    if atom.get_id() not in ("N", "CA", "C", "O"):
                        chain2_atoms.append(atom)

        if not chain1_atoms or not chain2_atoms:
            return 0

        cutoff = 4.0
        ns = NeighborSearch(chain1_atoms + chain2_atoms)
        visited_pairs = set()
        sb_count = 0

        for a1 in chain1_atoms:
            r1_name = a1.get_parent().get_resname()
            neighbors = ns.search(a1.coord, cutoff)
            for a2 in neighbors:
                if a2 in chain2_atoms and a2 is not a1:
                    r2_name = a2.get_parent().get_resname()
                    if ((r1_name in pos_res and r2_name in neg_res) or
                        (r1_name in neg_res and r2_name in pos_res)):
                        pair = tuple(sorted([id(a1), id(a2)]))
                        if pair not in visited_pairs:
                            visited_pairs.add(pair)
                            sb_count += 1

        return sb_count

    # -------------------------------------------------------
    # Shape complementarity
    # -------------------------------------------------------
    @cached_property
    def sc(self) -> float:
        """
        Approximate shape complementarity following
        Lawrence & Colman (1993) (CCP4 sc code).
        """
        surfaceA = gather_buried_surface_points(
            self.interface_residues_chain1, self.interface_residues_chain2,
            distance_cutoff=5.0, dot_density=15
        )
        surfaceB = gather_buried_surface_points(
            self.interface_residues_chain2, self.interface_residues_chain1,
            distance_cutoff=5.0, dot_density=15
        )
        return approximate_sc_lc93(surfaceA, surfaceB, w=0.5)

    # -------------------------------------------------------
    # Interface area & solvation energy
    # -------------------------------------------------------
    @cached_property
    def int_area(self) -> float:
        """
        Interface area:
          (SASA(chain1) + SASA(chain2)) - SASA(chain1+chain2)
        """
        sasa_c1 = self._compute_sasa_for_chain(self.chain1)
        sasa_c2 = self._compute_sasa_for_chain(self.chain2)
        sasa_complex = self._compute_sasa_for_complex(self.chain1, self.chain2)
        return (sasa_c1 + sasa_c2) - sasa_complex

    @cached_property
    def int_solv_en(self) -> float:
        """
        Rough solvation energy = -gamma * (buried_area).
        gamma ~ 0.0072 (kcal/mol/Å^2).
        """
        gamma = 0.0072
        return -gamma * self.int_area

    # -----------
    # SASA Helpers
    # -----------
    def _compute_sasa_for_chain(self, chain_res_list: List[Residue]) -> float:
        sr = ShrakeRupley()
        tmp_struct = Structure.Structure("chain_s")
        tmp_model = Model.Model(0)
        tmp_struct.add(tmp_model)
        c = Chain.Chain("X")
        tmp_model.add(c)
        for r in chain_res_list:
            c.add(r.copy())

        sr.compute(tmp_struct, level="R")
        total_sasa = 0.0
        for r in c:
            rsasa = r.xtra.get("EXP_RSASA", 0.0)
            total_sasa += rsasa
        return total_sasa

    def _compute_sasa_for_complex(self, chain1: List[Residue], chain2: List[Residue]) -> float:
        sr = ShrakeRupley()
        tmp_struct = Structure.Structure("complex_s")
        tmp_model = Model.Model(0)
        tmp_struct.add(tmp_model)
        cA = Chain.Chain("A")
        cB = Chain.Chain("B")
        tmp_model.add(cA)
        tmp_model.add(cB)
        for r in chain1:
            cA.add(r.copy())
        for r in chain2:
            cB.add(r.copy())

        sr.compute(tmp_struct, level="R")
        total_sasa = 0.0
        for c in (cA, cB):
            for r in c:
                rsasa = r.xtra.get("EXP_RSASA", 0.0)
                total_sasa += rsasa
        return total_sasa

##################################################
# ComplexAnalysis
##################################################

class ComplexAnalysis:
    """
    Represents a predicted complex.
    Loads structure, ranking, PAE data; creates per-interface analyses;
    computes global metrics (mpDockQ, etc.).
    """
    def __init__(
        self,
        structure_file: str,
        pae_file: str,
        ranking_metric: Dict[str, Any],
        contact_thresh: float,
        pae_filter: float
    ) -> None:
        self.structure_file = structure_file
        self.contact_thresh = contact_thresh
        self.pae_filter = pae_filter

        ext = Path(structure_file).suffix.lower()
        if ext == ".cif":
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)
        self.structure = parser.get_structure("complex", structure_file)

        self.ranking_metric = ranking_metric
        self.pae_data = read_json(pae_file)
        try:
            self.max_predicted_aligned_error = self.pae_data[0]["max_predicted_aligned_error"]
        except Exception as e:
            logging.error("Error extracting max_predicted_aligned_error: %s", e)
            self.max_predicted_aligned_error = float('nan')

        try:
            self._predicted_aligned_error = self.pae_data[0]["predicted_aligned_error"]
        except Exception as e:
            logging.error("Error extracting predicted_aligned_error: %s", e)
            self._predicted_aligned_error = None

        if ranking_metric.get("multimer"):
            self._iptm_ptm = ranking_metric["iptm+ptm"]
            self._iptm = ranking_metric["iptm"]
        else:
            self._iptm_ptm = None
            self._iptm = None

        # Build the residue index map & create interfaces
        model = next(self.structure.get_models())
        chains = list(model.get_chains())
        self.res_index_map: Dict[Tuple[str, Any], int] = {}
        self.chain_indices_by_id: Dict[str, List[int]] = {}
        idx = 0
        for chn in chains:
            indices_list: List[int] = []
            for res in chn:
                self.res_index_map[(chn.id, res.id)] = idx
                indices_list.append(idx)
                idx += 1
            self.chain_indices_by_id[chn.id] = indices_list

        self.pae_matrix = self._predicted_aligned_error

        # Create interface analyses
        self.interfaces = []
        for i in range(len(chains)):
            for j in range(i + 1, len(chains)):
                iface = InterfaceAnalysis(
                    list(chains[i]),
                    list(chains[j]),
                    self.contact_thresh,
                    self.pae_matrix,
                    self.res_index_map,
                    self.chain_indices_by_id,
                )
                if iface.num_intf_residues > 0:
                    self.interfaces.append(iface)

    @property
    def num_chains(self) -> int:
        model = next(self.structure.get_models())
        return len(list(model.get_chains()))

    @property
    def iptm_ptm(self) -> float:
        return self._iptm_ptm if self._iptm_ptm is not None else float('nan')

    @property
    def iptm(self) -> float:
        return self._iptm if self._iptm is not None else float('nan')

    @property
    def average_interface_pae(self) -> float:
        if not self.interfaces:
            return float('nan')
        valid_pae = [
            iface.average_interface_pae
            for iface in self.interfaces
            if not math.isnan(iface.average_interface_pae)
               and iface.average_interface_pae <= self.pae_filter
        ]
        return sum(valid_pae) / len(valid_pae) if valid_pae else float('nan')

    @cached_property
    def average_interface_plddt(self) -> float:
        if not self.interfaces:
            return float('nan')
        vals = [iface.average_interface_plddt for iface in self.interfaces]
        return sum(vals) / len(vals)

    @cached_property
    def contact_pairs_global(self) -> int:
        """Residue-level contacts using CB (or CA fallback) across different chains."""
        model = next(self.structure.get_models())
        chains = list(model.get_chains())

        # Select one representative atom per residue (CB if present, else CA)
        rep_atoms = []  # list of (atom, residue)
        for chn in chains:
            for res in chn:
                try:
                    atom = res["CB"] if "CB" in res else res["CA"]
                except Exception:
                    continue
                rep_atoms.append((atom, res))

        if not rep_atoms:
            return 0

        ns = NeighborSearch([a for (a, _) in rep_atoms])
        visited_pairs = set()
        count = 0
        for atom1, res1 in rep_atoms:
            neighbors = ns.search(atom1.coord, self.contact_thresh)
            for atom2 in neighbors:
                if atom2 is atom1:
                    continue
                res2 = atom2.get_parent()
                ch1 = res1.get_parent().id
                ch2 = res2.get_parent().id
                if ch1 == ch2:
                    continue
                # Unique residue-residue pair key
                key = tuple(sorted([(ch1, res1.id), (ch2, res2.id)]))
                if key not in visited_pairs:
                    visited_pairs.add(key)
                    count += 1
        return count

    @cached_property
    def compute_complex_score(self) -> float:
        """= average_interface_plddt * log10(contact_pairs_global + 1)."""
        return self.average_interface_plddt * math.log10(self.contact_pairs_global + 1)

    @cached_property
    def mpDockQ(self) -> float:
        """
        mpDockQ: only meaningful if num_chains > 2
        """
        if self.num_chains <= 2:
            return float('nan')
        return MPDOCKQ_CONSTANTS.score(self.compute_complex_score)

##################################################
# Processing all models
##################################################

def process_all_models(
    directory: str,
    contact_thresh: float,
    pae_filter: float,
    models_to_analyse: ModelsToAnalyse,
) -> None:
    job_name = extract_job_name(directory)
    ranking_data = parse_ranking_debug_json_all(directory)
    ranked_order: List[str] = ranking_data["order"]

    if models_to_analyse == ModelsToAnalyse.BEST:
        models = [ranked_order[0]]
    else:
        models = ranked_order

    output_data = []
    for model in models:
        try:
            r_metric = get_ranking_metric_for_model(ranking_data, model)
            struct_file = find_structure_file(directory, model)
            pae_file = str(Path(directory) / f"pae_{model}.json")
            comp = ComplexAnalysis(struct_file, pae_file, r_metric, contact_thresh, pae_filter)

            # If single chain, fallback to pDockQ from the first interface; else mpDockQ
            if comp.num_chains > 1:
                global_score = comp.mpDockQ
            else:
                global_score = comp.interfaces[0].pDockQ if comp.interfaces else float('nan')

            binding_energy = -1.3 * comp.contact_pairs_global  # placeholder or real
            model_used = model

            if comp.interfaces:
                for iface in comp.interfaces:
                    if iface.num_intf_residues == 0:
                        continue
                    if iface.average_interface_pae > pae_filter:
                        continue

                    pDockQ2_val, _ = iface.pDockQ2()
                    ipSAE_val = iface.ipsae()
                    lis_val = iface.lis()
                    iface_label = f"{iface.chain1[0].get_parent().id}_{iface.chain2[0].get_parent().id}"

                    record = {
                        "jobs": job_name,
                        "model_used": model_used,
                        "interface": iface_label,
                        "iptm_ptm": comp.iptm_ptm,
                        "iptm": comp.iptm,
                        "pDockQ/mpDockQ": global_score,
                        "average_interface_pae": iface.average_interface_pae,
                        "interface_average_plddt": iface.average_interface_plddt,
                        "interface_num_intf_residues": iface.num_intf_residues,
                        "interface_polar": iface.polar,
                        "interface_hydrophobic": iface.hydrophobic,
                        "interface_charged": iface.charged,
                        "interface_contact_pairs": iface.contact_pairs,
                        "interface_score": iface.score_complex,
                        "interface_pDockQ2": pDockQ2_val,
                        "interface_ipSAE": ipSAE_val,
                        "interface_LIS": lis_val,
                        # approximated from CCP4
                        "interface_hb": iface.hb,
                        "interface_sb": iface.sb,
                        "interface_sc": iface.sc,
                        "interface_area": iface.int_area,
                        "interface_solv_en": iface.int_solv_en,
                    }
                    output_data.append(record)

            logging.info("Processed model: %s", model)
        except Exception as e:
            logging.error("Error processing model %s: %s", model, e)

    if not output_data:
        logging.warning("No interfaces passed the filter; writing an empty output.")
        output_data = []

    output_file = Path(directory) / "interfaces.csv"
    with output_file.open("w", newline='') as f:
        if output_data:
            writer = csv.DictWriter(f, fieldnames=output_data[0].keys())
            writer.writeheader()
            writer.writerows(output_data)
        else:
            f.write("")
    logging.info("Unified interface scores written to %s", str(output_file))

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AlphaJudge: Interface scoring for AF outputs")
    parser.add_argument(
        "--path_to_dir",
        required=True,
        help="Path to the directory with predicted models and ranking_debug.json",
    )
    parser.add_argument(
        "--contact_thresh",
        type=float,
        default=DEFAULT_CONTACT_THRESH,
        help="Distance threshold (Å) for defining contacts (default: %(default)s)",
    )
    parser.add_argument(
        "--pae_filter",
        type=float,
        default=DEFAULT_PAE_FILTER,
        help="Max acceptable average interface PAE; interfaces above are skipped (default: %(default)s)",
    )
    parser.add_argument(
        "--models_to_analyse",
        choices=["best", "all"],
        default="best",
        help="If 'all', analyze all models; if 'best', only the top-ranked model",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    mta = ModelsToAnalyse.BEST if args.models_to_analyse == "best" else ModelsToAnalyse.ALL
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    process_all_models(
        args.path_to_dir,
        args.contact_thresh,
        args.pae_filter,
        mta,
    )

if __name__ == '__main__':
    main()