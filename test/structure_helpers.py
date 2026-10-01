"""Small coordinate fixtures shared by scoring regressions."""
from collections.abc import Mapping

import numpy as np
from Bio.PDB import Atom, Chain, Model, Residue, Structure


def make_structure(*, chains="AB", residues=2, separation=5., plddt=90.):
    structure = Structure.Structure("test")
    model = Model.Model(0)
    structure.add(model)
    serial = 1
    for chain_no, chain_id in enumerate(chains):
        chain = Chain.Chain(chain_id)
        model.add(chain)
        bfactor = plddt[chain_id] if isinstance(plddt, Mapping) else plddt
        for res_no in range(1, residues + 1):
            residue = Residue.Residue((" ", res_no, " "), "ALA", " ")
            chain.add(residue)
            for name, dx in (("CA", 0.), ("CB", .5)):
                residue.add(Atom.Atom(
                    name, np.array([res_no * 3.8 + dx, chain_no * separation, 0.]),
                    bfactor, 1., " ", name, serial, element="C",
                ))
                serial += 1
    return structure
