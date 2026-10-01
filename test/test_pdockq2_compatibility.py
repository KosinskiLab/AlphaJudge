"""Preserve the pairwise pDockQ2 variant used by AlphaJudge and IPSAE."""
import numpy as np
import pytest
from structure_helpers import make_structure

from alphajudge.complex import Complex
from alphajudge.confidence import Confidence
from alphajudge.parsers import BaseParser


def test_pdockq2_matches_pinned_ipsae_asymmetric_trimer():
    # Reference: DunbrackLab/IPSAE ipsae.py at
    # 6174cf9e71cb1bd660cc805856a18c4871a6dec3, pDockQ2 block.
    # Expected values were captured by running that script on this structure
    # and PAE, with shared pLDDT inputs. They are the maximum of its two
    # directional scores for each pair, not a chain-versus-rest score.
    structure = make_structure(chains="ABC", residues=3, plddt={"A": 95., "B": 30., "C": 80.})
    pae = np.full((9, 9), 2.)
    pae[3:6, :3] = 18.
    pae[6:, 3:6] = 12.
    chains, residue_map, _ = BaseParser._maps(structure)
    confidence = Confidence(pae, 31., .7, .6, .68, .68, BaseParser._plddt(chains, residue_map))
    comp = Complex(structure, confidence, 8., 100., 10.)
    scores = {iface.label: iface.pDockQ2()[0] for iface in comp.interfaces}
    assert scores == pytest.approx({"A_B": .18333777506612065, "B_C": .11509754028543784}, abs=1e-12)
