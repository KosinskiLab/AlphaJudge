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

# =========================
# unified metrics container
# =========================

@dataclass(frozen=True)
class ConfidenceMetrics:
    # residue × residue PAE
    pae_matrix: List[List[float]]
    max_pae: float

    # scalar metrics (may be None if not provided by files)
    iptm: Optional[float]
    ptm: Optional[float]
    iptm_ptm: Optional[float]          # “ranking_score”/“iptm+ptm”
    confidence_score: Optional[float]   # parsed if available; else computed fallback

    # per-residue pLDDT (same ordering as residue index map)
    plddt_residue: List[float]


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

# =========================
# simple utils and helpers
# =========================

def _read_json_silent(p: Path) -> Any | None:
    try:
        with p.open() as fh:
            return json.load(fh)
    except Exception:
        return None

def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def extract_job_name(path_to_dir: str) -> str:
    """Use the basename of the directory as the job name."""
    return Path(path_to_dir).resolve().name

def read_json(filepath: str) -> Any:
    p = Path(filepath)
    with p.open() as f:
        data = json.load(f)
    logging.info("Loaded JSON file: %s", str(p))
    return data

def read_csv(filepath: str) -> Dict[str, Any]:
    """Parse AF3 ranking_scores.csv into an order of folder names.

    Input CSV has columns: seed, sample, ranking_score. We sort by
    descending ranking_score and produce folder names of the form:
    seed-{seed}_sample-{sample}
    """
    p = Path(filepath)
    with p.open(newline='') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row]
    if not rows:
        logging.warning(f"CSV {p} is empty.")
        return {"order": []}

    def parse_float(val: Any) -> float:
        try:
            return float(val)
        except Exception:
            return float("nan")

    rows_sorted = sorted(rows, key=lambda r: parse_float(r.get("ranking_score")), reverse=True)
    order = [f"seed-{r['seed']}_sample-{r['sample']}" for r in rows_sorted if 'seed' in r and 'sample' in r]
    logging.info("Loaded CSV file: %s", str(p))
    return {"order": order}

def parse_ranking_debug_json_af2(directory: str) -> Dict[str, Any]:
    path = Path(directory) / "ranking_debug.json"
    data = read_json(str(path))
    if "order" not in data or not isinstance(data["order"], list):
        raise ValueError("Invalid ranking_debug.json: missing or invalid 'order' key")
    data_tagged = dict(data)
    data_tagged["source"] = "af2"
    return data_tagged

def parse_ranking_scores_csv_af3(directory: str) -> Dict[str, Any]:
    path = Path(directory) / "ranking_scores.csv"
    data = read_csv(str(path))
    if "order" not in data or not isinstance(data["order"], list):
        raise ValueError("Invalid ranking_scores.csv: missing or invalid 'order' key")
    data_tagged = dict(data)
    data_tagged["source"] = "af3"
    return data_tagged

def get_ranking_metric_for_model(data: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Return ranking metrics for a model."""
    source = data.get("source")
    has_multimer = ("iptm+ptm" in data) and ("iptm" in data)
    has_monomer = ("plddts" in data) and ("ptm" in data)

    if source == "af2" and has_multimer:
        if model not in data["iptm+ptm"] or model not in data["iptm"]:
            raise ValueError(f"Model '{model}' not found in multimer metrics")
        # iptm may be absent in af2 multimer ranking_debug; keep None if missing
        return {
            "model": model,
            "iptm+ptm": data["iptm+ptm"][model],
            "iptm": data["iptm"][model],
            "ptm": data.get("ptm", {}).get(model, None),
            "multimer": True,
        }
    elif source == "af2" and has_monomer:
        if model not in data["plddts"] or model not in data["ptm"]:
            raise ValueError(f"Model '{model}' not found in monomer metrics")
        return {
            "model": model,
            "plddts": data["plddts"][model],
            "ptm": data["ptm"][model],
            "multimer": False,
        }
    elif source == "af3":
        # AF3 CSV doesn't carry per-model metric dicts; we only have ordering.
        return {"model": model, "multimer": False}
    else:
        raise ValueError("Unknown structure of the directory; expected 'af2' or 'af3' like structure")

def find_pae_file(directory: str, model: str) -> str:
    """Resolve path to PAE/confidence JSON for a given model identifier.

    - AF3: directory/model/confidences.json, else summary_confidences.json
    - AF2: 'pae_{model}.json' in the directory
    """
    model_dir = Path(directory) / model
    if model_dir.is_dir():
        conf_full = model_dir / "confidences.json"
        if conf_full.exists():
            return str(conf_full)
        af3_summary = model_dir / "summary_confidences.json"
        if af3_summary.exists():
            return str(af3_summary)
    # AF2 fallback
    path = Path(directory) / f"pae_{model}.json"
    if path.exists():
        return str(path)
    raise FileNotFoundError(f"Could not find PAE/confidence file for model '{model}'. Checked confidences.json/summary_confidences.json and {path}")

def find_structure_file(directory: str, model: str) -> str:
    """Resolve structure file for a model.

    - AF3: directory/model/model.cif
    - Fallback: glob by model token
    """
    model_dir = Path(directory) / model
    direct_cif = model_dir / "model.cif"
    if direct_cif.exists():
        return str(direct_cif)

    d = Path(directory)
    cif_matches = list(d.glob(f"*{model}*.cif"))
    if cif_matches:
        return str(cif_matches[0])
    pdb_matches = list(d.glob(f"*{model}*.pdb"))
    if pdb_matches:
        return str(pdb_matches[0])
    raise ValueError(f"No structure file (CIF or PDB) found for model '{model}' in directory.")

# ============================================
# residue maps and per-residue pLDDT extraction
# ============================================

def _build_residue_maps(structure) -> tuple[Dict[Tuple[str, Any], int], Dict[str, List[int]], List[Chain]]:
    model = next(structure.get_models())
    chains = list(model.get_chains())
    res_index_map: Dict[Tuple[str, Any], int] = {}
    chain_indices_by_id: Dict[str, List[int]] = {}
    idx = 0
    for ch in chains:
        idxs: List[int] = []
        for res in ch:
            res_index_map[(ch.id, res.id)] = idx
            idxs.append(idx)
            idx += 1
        chain_indices_by_id[ch.id] = idxs
    return res_index_map, chain_indices_by_id, chains

def _plddt_from_structure(
    chains: List[Chain],
    res_index_map: Dict[Tuple[str, Any], int],
) -> List[float]:
    n_res = len(res_index_map)
    plddt = [float('nan')] * n_res
    for ch in chains:
        for res in ch:
            idx = res_index_map.get((ch.id, res.id))
            if idx is None:
                continue
            try:
                plddt[idx] = float(res["CA"].get_bfactor())
                continue
            except Exception:
                pass
            vals = [float(a.get_bfactor()) for a in res.get_atoms()
                    if a.element and a.element.upper() != "H"]
            plddt[idx] = float(np.mean(vals)) if vals else float('nan')
    return plddt

# =========================
# AF2 / AF3 confidence I/O
# =========================

def parse_confidences_from_af2(
    *,
    pae_payload: Any,                      # content of pae_{model}.json
    ranking_debug_metric: Dict[str, Any],  # get_ranking_metric_for_model(...) result
    chains: List[Chain],
    res_index_map: Dict[Tuple[str, Any], int],
    chain_indices_by_id: Dict[str, List[int]],
) -> ConfidenceMetrics:
    """AF2 scheme: scalars in ranking_debug.json; PAE in pae_{model}.json."""
    if not (isinstance(pae_payload, list) and pae_payload and isinstance(pae_payload[0], dict) and
            "predicted_aligned_error" in pae_payload[0]):
        raise ValueError("AF2 PAE payload malformed")

    pae = np.array(pae_payload[0]["predicted_aligned_error"], dtype=float)
    max_pae = float(np.nanmax(pae)) if pae.size else float('nan')

    is_multimer = bool(ranking_debug_metric.get("multimer", False))

    if is_multimer:
        conf = _safe_float(ranking_debug_metric.get("iptm+ptm"))
        iptm = _safe_float(ranking_debug_metric.get("iptm"))
        # If confidence and iptm are present, derive ptm = (conf - 0.8*iptm)/0.2
        if conf is not None and iptm is not None:
            ptm = (conf - 0.8 * iptm) / 0.2
        else:
            ptm = _safe_float(ranking_debug_metric.get("ptm"))
        # Prefer provided conf; else compute from available iptm/ptm
        if conf is not None:
            iptm_ptm = conf
        elif iptm is not None and ptm is not None:
            iptm_ptm = 0.8 * iptm + 0.2 * ptm
        else:
            iptm_ptm = None
        conf_score = iptm_ptm
    else:
        # Monomer: no iptm; ptm provided directly
        iptm = 0.0
        ptm = _safe_float(ranking_debug_metric.get("ptm"))
        iptm_ptm = ptm
        conf_score = ptm

    plddt_res = _plddt_from_structure(chains, res_index_map)
    return ConfidenceMetrics(
        pae_matrix=pae.tolist(),
        max_pae=max_pae,
        iptm=iptm,
        ptm=ptm,
        iptm_ptm=iptm_ptm,
        confidence_score=conf_score,
        plddt_residue=plddt_res,
    )

def parse_confidences_from_af3(
    *,
    summary_payload: Dict[str, Any],   # summary_confidences.json (scalars)
    matrix_payload: Dict[str, Any],    # confidences.json (big PAE matrix)
    chains: List[Chain],
    res_index_map: Dict[Tuple[str, Any], int],
    chain_indices_by_id: Dict[str, List[int]],
) -> ConfidenceMetrics:
    """AF3 scheme: scalars in summary_confidences.json; matrix in confidences.json."""
    iptm       = _safe_float(summary_payload.get("iptm"))
    ptm        = _safe_float(summary_payload.get("ptm"))
    iptm_ptm   = _safe_float(summary_payload.get("ranking_score")) or _safe_float(summary_payload.get("iptm+ptm"))
    conf_score = _safe_float(summary_payload.get("confidence_score"))
    # Derive missing values using AF3 convention: conf ~ 0.8*iptm + 0.2*ptm
    if iptm_ptm is not None and iptm is not None and (ptm is None):
        ptm = (iptm_ptm - 0.8 * iptm) / 0.2
    if conf_score is None and iptm is not None and ptm is not None:
        conf_score = 0.8 * iptm + 0.2 * ptm

    total_res = sum(len(chain_indices_by_id[c.id]) for c in chains)
    pae = np.full((total_res, total_res), 100.0, dtype=float)

    if "predicted_aligned_error" in matrix_payload:
        m = np.array(matrix_payload["predicted_aligned_error"], dtype=float)
        if m.size:
            pae[:, :] = m
        max_pae = float(matrix_payload.get("max_predicted_aligned_error", np.nan))
        if not np.isfinite(max_pae):
            max_pae = float(np.nanmax(m)) if m.size else float('nan')

    elif "pae" in matrix_payload and "token_chain_ids" in matrix_payload:
        tokens = np.array(matrix_payload["pae"], dtype=float)
        t_ids  = matrix_payload["token_chain_ids"]
        max_pae = float(np.nanmax(tokens)) if tokens.size else float('nan')

        seen = []
        for c in t_ids:
            if c not in seen:
                seen.append(c)
        group_to_idx = {g: i for i, g in enumerate(seen)}

        for i, chi in enumerate(chains):
            t_i = [k for k, c in enumerate(t_ids) if group_to_idx.get(c, -1) == i]
            r_i = chain_indices_by_id.get(chi.id, [])
            for j, chj in enumerate(chains):
                t_j = [k for k, c in enumerate(t_ids) if group_to_idx.get(c, -1) == j]
                r_j = chain_indices_by_id.get(chj.id, [])
                if t_i and t_j and r_i and r_j:
                    block = tokens[np.ix_(t_i, t_j)]
                    val = float(np.nanmin(block)) if block.size else 100.0
                    pae[np.ix_(r_i, r_j)] = val

    elif "chain_pair_pae_min" in matrix_payload:
        chain_min = np.array(matrix_payload["chain_pair_pae_min"], dtype=float)
        max_pae = float(np.nanmax(chain_min)) if chain_min.size else float('nan')
        for i, chi in enumerate(chains):
            r_i = chain_indices_by_id.get(chi.id, [])
            for j, chj in enumerate(chains):
                r_j = chain_indices_by_id.get(chj.id, [])
                val = None
                try:
                    val = float(chain_min[i][j])
                except Exception:
                    pass
                if r_i and r_j:
                    pae[np.ix_(r_i, r_j)] = val if val is not None else 100.0
    else:
        raise ValueError("AF3 confidences.json schema not recognized")

    plddt_res = _plddt_from_structure(chains, res_index_map)
    return ConfidenceMetrics(
        pae_matrix=pae.tolist(),
        max_pae=max_pae,
        iptm=iptm,
        ptm=ptm,
        iptm_ptm=iptm_ptm,
        confidence_score=conf_score,
        plddt_residue=plddt_res,
    )

# =========================
# precomputation helpers
# =========================

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
# simple shape complementarity (toy LC93)
##################################################

def approximate_sc_lc93(surfaceA, surfaceB, w=0.5):
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

    # ipSAE
    def _pair_type(self) -> str:
        nuc = {"DA", "DC", "DT", "DG", "A", "C", "U", "G"}
        resnames = {r.get_resname().strip() for r in self.interface_residues_chain1.union(self.interface_residues_chain2)}
        return "nucleic_acid" if (resnames & nuc) else "protein"

    @staticmethod
    def _calc_d0(n: int, pair_type: str) -> float:
        L = max(27.0, float(n))
        base = 1.24 * (L - 15.0) ** (1.0 / 3.0) - 1.8
        minv = 2.0 if pair_type == "nucleic_acid" else 1.0
        return max(minv, base)

    def _chain_indices(self, residues: List[Residue]) -> List[int]:
        idxs = [self._res_index_map.get((r.get_parent().id, r.id)) for r in residues]
        return [i for i in idxs if i is not None]

    def ipsae_d0res_asym(self, pae_cutoff: float = 10.0) -> float:
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
        return self.ipsae_d0res_asym(pae_cutoff)

    # LIS
    def lis(self) -> float:
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

    # hydrogen bonds
    @cached_property
    def hb(self) -> int:
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

    # salt bridges
    @cached_property
    def sb(self) -> int:
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

    # shape complementarity
    @cached_property
    def sc(self) -> float:
        surfaceA = gather_buried_surface_points(
            self.interface_residues_chain1, self.interface_residues_chain2,
            distance_cutoff=5.0, dot_density=15
        )
        surfaceB = gather_buried_surface_points(
            self.interface_residues_chain2, self.interface_residues_chain1,
            distance_cutoff=5.0, dot_density=15
        )
        return approximate_sc_lc93(surfaceA, surfaceB, w=0.5)

    # interface area & solvation energy
    @cached_property
    def int_area(self) -> float:
        sasa_c1 = self._compute_sasa_for_chain(self.chain1)
        sasa_c2 = self._compute_sasa_for_chain(self.chain2)
        sasa_complex = self._compute_sasa_for_complex(self.chain1, self.chain2)
        return (sasa_c1 + sasa_c2) - sasa_complex

    @cached_property
    def int_solv_en(self) -> float:
        gamma = 0.0072
        return -gamma * self.int_area

    # SASA helpers
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
# ComplexAnalysis (file-agnostic)
##################################################

class ComplexAnalysis:
    def __init__(
        self,
        structure,                  # Bio.PDB structure object (already parsed)
        metrics: ConfidenceMetrics, # unified metrics (already parsed)
        contact_thresh: float,
        pae_filter: float,
    ) -> None:
        self.structure = structure
        self.contact_thresh = contact_thresh
        self.pae_filter = pae_filter

        # local maps (analysis-only)
        self.res_index_map, self.chain_indices_by_id, self._chains = _build_residue_maps(self.structure)

        # confidences
        self.pae_matrix = metrics.pae_matrix
        self.max_predicted_aligned_error = metrics.max_pae
        self._iptm = metrics.iptm
        self._ptm = metrics.ptm
        self._iptm_ptm = metrics.iptm_ptm
        self._confidence_score = metrics.confidence_score
        self._plddt_residue = metrics.plddt_residue

        # interfaces
        self.interfaces: List[InterfaceAnalysis] = []
        for i in range(len(self._chains)):
            for j in range(i + 1, len(self._chains)):
                iface = InterfaceAnalysis(
                    list(self._chains[i]),
                    list(self._chains[j]),
                    self.contact_thresh,
                    self.pae_matrix,
                    self.res_index_map,
                    self.chain_indices_by_id,
                )
                if iface.num_intf_residues > 0:
                    self.interfaces.append(iface)

    @property
    def num_chains(self) -> int:
        return len(self._chains)

    @property
    def iptm_ptm(self) -> float:
        return float(self._iptm_ptm) if self._iptm_ptm is not None else float('nan')

    @property
    def iptm(self) -> float:
        return float(self._iptm) if self._iptm is not None else float('nan')

    @property
    def ptm(self) -> float:
        return float(self._ptm) if self._ptm is not None else float('nan')

    @property
    def confidence_score(self) -> float:
        return float(self._confidence_score) if self._confidence_score is not None else float('nan')

    @property
    def plddt_residue(self) -> List[float]:
        return self._plddt_residue

    @property
    def average_interface_pae(self) -> float:
        if not self.interfaces:
            return float('nan')
        vals = [
            i.average_interface_pae
            for i in self.interfaces
            if not math.isnan(i.average_interface_pae) and i.average_interface_pae <= self.pae_filter
        ]
        return sum(vals) / len(vals) if vals else float('nan')

    @cached_property
    def average_interface_plddt(self) -> float:
        if not self.interfaces:
            return float('nan')
        vals = [i.average_interface_plddt for i in self.interfaces]
        return sum(vals) / len(vals)

    @cached_property
    def contact_pairs_global(self) -> int:
        rep_atoms = []
        for ch in self._chains:
            for res in ch:
                try:
                    rep_atoms.append((res["CB"] if "CB" in res else res["CA"], res))
                except Exception:
                    continue
        if not rep_atoms:
            return 0

        ns = NeighborSearch([a for (a, _) in rep_atoms])
        visited_pairs = set()
        count = 0
        for atom1, res1 in rep_atoms:
            for atom2 in ns.search(atom1.coord, self.contact_thresh):
                if atom2 is atom1:
                    continue
                res2 = atom2.get_parent()
                ch1, ch2 = res1.get_parent().id, res2.get_parent().id
                if ch1 == ch2:
                    continue
                key = tuple(sorted([(ch1, res1.id), (ch2, res2.id)]))
                if key not in visited_pairs:
                    visited_pairs.add(key)
                    count += 1
        return count

    @cached_property
    def compute_complex_score(self) -> float:
        return self.average_interface_plddt * math.log10(self.contact_pairs_global + 1)

    @cached_property
    def mpDockQ(self) -> float:
        return MPDOCKQ_CONSTANTS.score(self.compute_complex_score) if self.num_chains > 2 else float('nan')

##################################################
# Processing all models (single-pass I/O, two schemes)
##################################################

def process_all_models(
    directory: str,
    contact_thresh: float,
    pae_filter: float,
    models_to_analyse: ModelsToAnalyse,
) -> None:
    job_name = extract_job_name(directory)
    # If there is ranking_debug.json, this is output from AlphaFold2.
    if (Path(directory) / "ranking_debug.json").exists():
        logging.info("Found ranking_debug.json, this is output from AlphaFold2.")
        ranking_data = parse_ranking_debug_json_af2(directory)
        ranked_order: List[str] = ranking_data["order"]
    elif (Path(directory) / "ranking_scores.csv").exists():
        logging.info("Found ranking_scores.csv, this is output from AlphaFold3.")
        ranking_data = parse_ranking_scores_csv_af3(directory)
        ranked_order: List[str] = ranking_data["order"]
    else:
        raise ValueError("No ranking_debug.json or ranking_scores.csv found in directory.")

    models = [ranked_order[0]] if models_to_analyse == ModelsToAnalyse.BEST else ranked_order

    output_data = []
    for model in models:
        try:
            r_metric = get_ranking_metric_for_model(ranking_data, model)

            # parse structure once
            struct_path = find_structure_file(directory, model)
            parser = MMCIFParser(QUIET=True) if Path(struct_path).suffix.lower() == ".cif" else PDBParser(QUIET=True)
            structure = parser.get_structure("complex", struct_path)

            # residue maps (for pLDDT extraction)
            res_index_map, chain_indices_by_id, chains = _build_residue_maps(structure)

            # route by scheme
            if ranking_data.get("source") == "af2":
                pae_path = Path(directory) / f"pae_{model}.json"
                pae_payload = read_json(str(pae_path))
                metrics = parse_confidences_from_af2(
                    pae_payload=pae_payload,
                    ranking_debug_metric=r_metric,
                    chains=chains,
                    res_index_map=res_index_map,
                    chain_indices_by_id=chain_indices_by_id,
                )
            else:  # af3
                model_dir = Path(directory) / model
                conf_matrix_path = model_dir / "confidences.json"
                summary_path     = model_dir / "summary_confidences.json"

                # load payloads (support case where single file may carry both fields)
                matrix_payload = read_json(str(conf_matrix_path)) if conf_matrix_path.exists() else {}
                summary_payload = read_json(str(summary_path)) if summary_path.exists() else {}

                if not isinstance(summary_payload, dict) or not summary_payload:
                    raise ValueError(f"Missing AF3 summary_confidences.json for {model}")
                if not isinstance(matrix_payload, dict) or not matrix_payload:
                    # some AF3 drops everything into confidences.json; try reusing summary_payload
                    matrix_payload = summary_payload

                metrics = parse_confidences_from_af3(
                    summary_payload=summary_payload,
                    matrix_payload=matrix_payload,
                    chains=chains,
                    res_index_map=res_index_map,
                    chain_indices_by_id=chain_indices_by_id,
                )

            # analysis layer (no file I/O)
            comp = ComplexAnalysis(
                structure=structure,
                metrics=metrics,
                contact_thresh=contact_thresh,
                pae_filter=pae_filter,
            )

            # If single chain, fallback to pDockQ from the first interface; else mpDockQ
            if comp.num_chains > 1:
                global_score = comp.mpDockQ
            else:
                global_score = comp.interfaces[0].pDockQ if comp.interfaces else float('nan')

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
                        "ptm": comp.ptm,
                        "confidence_score": comp.confidence_score,
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
    parser = argparse.ArgumentParser(description="AlphaJudge: interface scoring for AF2/AF3 outputs")
    parser.add_argument(
        "--path_to_dir",
        required=True,
        help="Path to the directory with models predicted by AlphaFold2 or AlphaFold3",
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
