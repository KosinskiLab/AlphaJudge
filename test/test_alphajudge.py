import math
import numpy as np
import pytest
from typing import List
from run_alphajudge import (
    InterfaceAnalysis,
    ComplexAnalysis,
    D0
)

# --- Dummy Classes to Simulate Bio.PDB Objects ---

class DummyAtom:
    def __init__(self, coord, bfactor, parent):
        self.coord = np.array(coord)
        self._bfactor = bfactor
        self.parent = parent

    def get_coord(self):
        return self.coord

    def get_bfactor(self):
        return self._bfactor

    def get_parent(self):
        return self.parent

    def __sub__(self, other):
        # Euclidean distance between atom coordinates.
        return np.linalg.norm(self.coord - other.coord)

class DummyParent:
    def __init__(self, chain_id):
        self.id = chain_id

class DummyResidue:
    def __init__(self, residue_id, resname, chain_id,
                 use_cb=True, coord=(0, 0, 0), bfactor=80.0):
        self.id = residue_id
        self.resname = resname
        self.parent = DummyParent(chain_id)
        if use_cb:
            self.CB = DummyAtom(coord, bfactor, self)
        else:
            self.CA = DummyAtom(coord, bfactor, self)

    def get_parent(self):
        return self.parent

    def get_resname(self):
        return self.resname

    def __getitem__(self, key):
        if key == "CB":
            if hasattr(self, "CB"):
                return self.CB
            else:
                raise KeyError("No CB")
        elif key == "CA":
            if hasattr(self, "CA"):
                return self.CA
            else:
                raise KeyError("No CA")
        raise KeyError(key)

    def __contains__(self, key):
        if key == "CB":
            return hasattr(self, "CB")
        elif key == "CA":
            return hasattr(self, "CA")
        return False

    # Yield each atom in this residue
    def __iter__(self):
        if hasattr(self, "CB"):
            yield self.CB
        if hasattr(self, "CA"):
            yield self.CA

@pytest.fixture
def chains():
    # Two chains, each with two residues:
    # A1 near B1; A2 far from B1 & B2.
    chain_A = [
        DummyResidue("1", "ALA", "A", use_cb=True,  coord=(0, 0, 0),   bfactor=80),
        DummyResidue("2", "GLY", "A", use_cb=False, coord=(10,10,10), bfactor=90),
    ]
    chain_B = [
        DummyResidue("1", "LYS", "B", use_cb=True,  coord=(1,0,0),    bfactor=70),
        DummyResidue("2", "SER", "B", use_cb=True,  coord=(50,50,50), bfactor=60),
    ]
    return chain_A, chain_B


@pytest.fixture
def res_index_map(chains):
    chain_A, chain_B = chains
    index_map = {}
    index = 0
    for res in chain_A:
        index_map[(res.get_parent().id, res.id)] = index
        index += 1
    for res in chain_B:
        index_map[(res.get_parent().id, res.id)] = index
        index += 1
    return index_map


@pytest.fixture
def pae_matrix():
    # 4x4 matrix for 4 residues: A1=0, A2=1, B1=2, B2=3
    # A1-B1 is close => PAE=8 in (0,2) & (2,0).
    return [
        [0,   5,   8,  20],
        [5,   0,   25, 25],
        [8,   10,  0,   6],
        [20, 25,   6,   0]
    ]


@pytest.fixture
def contact_thresh():
    return 12.0

# --- Test Suite for InterfaceAnalysis ---
def test_interface_contact_pairs(chains, res_index_map, pae_matrix, contact_thresh):
    chain_A, chain_B = chains
    iface = InterfaceAnalysis(
        chain_A, chain_B,
        contact_thresh, pae_matrix,
        res_index_map
    )
    # Only residue A1 and B1 are within 12 Å.
    assert iface.contact_pairs == 1


def test_average_interface_plddt(chains, res_index_map, pae_matrix, contact_thresh):
    chain_A, chain_B = chains
    iface = InterfaceAnalysis(
        chain_A, chain_B,
        contact_thresh, pae_matrix,
        res_index_map
    )
    # A1 (bfactor=80) and B1 (bfactor=70) => average=75
    expected = 75.0
    assert iface.average_interface_plddt == pytest.approx(expected, abs=1e-5)


def test_average_interface_pae(chains, res_index_map, pae_matrix, contact_thresh):
    chain_A, chain_B = chains
    iface = InterfaceAnalysis(
        chain_A, chain_B,
        contact_thresh, pae_matrix,
        res_index_map
    )
    # Single contact pair => (A1,B1)=8, (B1,A1)=8 => average=8
    expected = 8.0
    assert iface.average_interface_pae == pytest.approx(expected, abs=1e-5)


def test_pDockQ(chains, res_index_map, pae_matrix, contact_thresh):
    chain_A, chain_B = chains
    iface = InterfaceAnalysis(
        chain_A, chain_B,
        contact_thresh, pae_matrix,
        res_index_map
    )
    # One contact => log10(1)=0 => pDockQ = 0.724/(1+exp(0.052*152.611)) + 0.018
    expected = 0.724 / (1 + math.exp(0.052 * 152.611)) + 0.018
    assert iface.pDockQ == pytest.approx(expected, abs=1e-6)


def test_pDockQ2_and_ipsae(chains, res_index_map, pae_matrix, contact_thresh):
    chain_A, chain_B = chains
    iface = InterfaceAnalysis(
        chain_A, chain_B,
        contact_thresh, pae_matrix,
        res_index_map
    )
    # PTM = 1 / [1 + (8/D0)^2]
    expected_ptm = 1.0 / (1 + (8 / D0) ** 2)
    pDockQ2_val, mean_ptm = iface.pDockQ2()
    assert mean_ptm == pytest.approx(expected_ptm, abs=1e-5)
    x = iface.average_interface_plddt * expected_ptm  # 75 * expected_ptm
    expected_pDockQ2 = 1.31 / (1 + math.exp(-0.075 * (x - 84.733))) + 0.005
    assert pDockQ2_val == pytest.approx(expected_pDockQ2, abs=1e-5)
    assert iface.ipsae() == pytest.approx(expected_ptm, abs=1e-5)


def test_lis(chains, res_index_map, pae_matrix, contact_thresh):
    chain_A, chain_B = chains
    iface = InterfaceAnalysis(
        chain_A, chain_B,
        contact_thresh, pae_matrix,
        res_index_map
    )
    # Only A1-B1 => PAE=8 => LIS=(12-8)/12=0.3333
    assert iface.lis() == pytest.approx(0.33333, abs=1e-5)


def test_interface_composition_fractions(chains, res_index_map, pae_matrix, contact_thresh):
    chain_A, chain_B = chains
    iface = InterfaceAnalysis(
        chain_A, chain_B,
        contact_thresh, pae_matrix,
        res_index_map
    )
    # Interface residues: A1=ALA (hydrophobic), B1=LYS (charged)
    assert iface.num_intf_residues == 2
    assert iface.hydrophobic == pytest.approx(0.5, abs=1e-9)
    assert iface.charged == pytest.approx(0.5, abs=1e-9)
    assert iface.polar == pytest.approx(0.0, abs=1e-9)


def test_zero_contacts_edge_cases(chains, res_index_map, pae_matrix):
    chain_A, chain_B = chains
    # Make threshold too small so A1-B1 is not a contact
    no_contact_thresh = 0.5
    iface = InterfaceAnalysis(
        chain_A, chain_B,
        no_contact_thresh, pae_matrix,
        res_index_map
    )
    assert iface.contact_pairs == 0
    # Average interface pLDDT over empty interface => NaN
    assert math.isnan(iface.average_interface_plddt)
    # Average interface PAE over empty set => NaN
    assert math.isnan(iface.average_interface_pae)
    # pDockQ must be 0 with no contacts
    assert iface.pDockQ == 0.0
    # pDockQ2 returns (nan, 0.0) when there are no pairs
    p2, mean_ptm = iface.pDockQ2()
    assert math.isnan(p2)
    assert mean_ptm == 0.0
    # ipsae averages empty set => 0.0
    assert iface.ipsae() == 0.0

# --- Dummy Model/Structure for ComplexAnalysis ---
class DummyModel:
    """A dummy model that returns lists of dummy chains and atoms."""
    def __init__(self, chains: List[List[DummyResidue]]):
        self._chains = chains

    def get_chains(self):
        return self._chains

    def get_atoms(self):
        """Flatten all residues -> atoms."""
        all_atoms = []
        for chain in self._chains:
            for residue in chain:
                for atom in residue:
                    all_atoms.append(atom)
        return all_atoms

class DummyStructure:
    """A dummy structure that yields a single dummy model."""
    def __init__(self, model: DummyModel):
        self._model = model

    def get_models(self):
        yield self._model

    def get_residues(self):
        residues = []
        for chain in self._model.get_chains():
            residues.extend(chain)
        return residues

# --- Test Suite for ComplexAnalysis ---
@pytest.fixture
def dummy_structure(chains):
    chain_A, chain_B = chains
    return DummyStructure(DummyModel([chain_A, chain_B]))


@pytest.fixture
def ranking_metric_multimer():
    return {
        "multimer": True,
        "iptm+ptm": 0.09161740784114437,
        "iptm": 0.08377959013960515,
    }


@pytest.fixture
def dummy_comp(chains, res_index_map, pae_matrix, dummy_structure, ranking_metric_multimer):
    # Create a dummy ComplexAnalysis instance by subclassing to bypass file I/O
    class DummyComplexAnalysis(ComplexAnalysis):
        def __init__(self, structure_file, pae_file, ranking_metric, contact_thresh):
            # Bypass real file parsing; inject dummy data.
            self.chain_A, self.chain_B = chains
            self.res_index_map = res_index_map
            self.pae_matrix = pae_matrix
            self.ranking_metric = ranking_metric
            self.contact_thresh = contact_thresh
            self.structure = dummy_structure
            # Create the interface analyses
            self.interfaces = []
            model = next(self.structure.get_models())
            _chains = list(model.get_chains())
            for i in range(len(_chains)):
                for j in range(i + 1, len(_chains)):
                    iface = InterfaceAnalysis(
                        _chains[i],
                        _chains[j],
                        self.contact_thresh,
                        self.pae_matrix,
                        self.res_index_map,
                    )
                    if iface.num_intf_residues > 0:
                        self.interfaces.append(iface)

        @property
        def num_chains(self):
            model = next(self.structure.get_models())
            return len(list(model.get_chains()))

    comp = DummyComplexAnalysis("dummy.pdb", "dummy.pae", ranking_metric_multimer, 12.0)
    return comp


def test_interface_creation(dummy_comp):
    # We expect exactly one interface: A1 with B1
    assert len(dummy_comp.interfaces) == 1
    iface = dummy_comp.interfaces[0]
    assert iface.average_interface_plddt == pytest.approx(75.0, abs=1e-5)
    assert iface.average_interface_pae == pytest.approx(8.0, abs=1e-5)


def test_global_metrics(dummy_comp):
    # Should have a nonzero global contact count
    assert dummy_comp.contact_pairs_global > 0
    # For a dimer (2 chains), mpDockQ is not defined => NaN
    assert math.isnan(dummy_comp.mpDockQ)
