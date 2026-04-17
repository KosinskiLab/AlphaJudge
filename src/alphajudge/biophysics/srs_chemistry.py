"""Minimal CCP4SRS monomer chemistry used by PISA hydrogen-bond scoring.

The tables mirror the compiled CCP4SRS database distributed with CCP4 7.1 for
standard amino-acid components.  CCP4SRS marks atoms with ``hb_type``:

``D`` donor, ``A`` acceptor, ``B`` donor and acceptor, ``H`` hydrogen,
``N`` neither.  ``Chem::makeBonds`` then adds the listed monomer bonds to the
matching atoms present in the structure before ``Chem::CalcHBonds`` applies
the geometric filters.
"""

from __future__ import annotations

SRS_ATOM_HB_TYPES = {
    "ALA": {"N": "D", "O": "A", "OXT": "A", "H": "H", "H2": "H", "HXT": "H"},
    "ARG": {
        "N": "D",
        "O": "A",
        "NE": "D",
        "NH1": "D",
        "NH2": "D",
        "OXT": "A",
        "H": "H",
        "H2": "H",
        "HE": "H",
        "HH11": "H",
        "HH12": "H",
        "HH21": "H",
        "HH22": "H",
        "HXT": "H",
    },
    "ASN": {
        "N": "D",
        "O": "A",
        "OD1": "A",
        "ND2": "D",
        "OXT": "A",
        "H": "H",
        "H2": "H",
        "HD21": "H",
        "HD22": "H",
        "HXT": "H",
    },
    "ASP": {
        "N": "D",
        "O": "A",
        "OD1": "A",
        "OD2": "A",
        "OXT": "A",
        "H": "H",
        "H2": "H",
        "HD2": "H",
        "HXT": "H",
    },
    "CYS": {"N": "D", "O": "A", "SG": "A", "OXT": "A", "H": "H", "H2": "H", "HXT": "H"},
    "GLN": {
        "N": "D",
        "O": "A",
        "OE1": "A",
        "NE2": "D",
        "OXT": "A",
        "H": "H",
        "H2": "H",
        "HE21": "H",
        "HE22": "H",
        "HXT": "H",
    },
    "GLU": {
        "N": "D",
        "O": "A",
        "OE1": "A",
        "OE2": "A",
        "OXT": "A",
        "H": "H",
        "H2": "H",
        "HE2": "H",
        "HXT": "H",
    },
    "GLY": {"N": "D", "O": "A", "OXT": "A", "H": "H", "H2": "H", "HXT": "H"},
    "HIS": {
        "N": "D",
        "O": "A",
        "ND1": "D",
        "NE2": "D",
        "OXT": "A",
        "H": "H",
        "H2": "H",
        "HD1": "H",
        "HE2": "H",
        "HXT": "H",
    },
    "ILE": {"N": "D", "O": "A", "OXT": "A", "H": "H", "H2": "H", "HXT": "H"},
    "LEU": {"N": "D", "O": "A", "OXT": "A", "H": "H", "H2": "H", "HXT": "H"},
    "LYS": {
        "N": "D",
        "O": "A",
        "NZ": "D",
        "OXT": "A",
        "H": "H",
        "H2": "H",
        "HZ1": "H",
        "HZ2": "H",
        "HZ3": "H",
        "HXT": "H",
    },
    "MET": {"N": "D", "O": "A", "SD": "A", "OXT": "A", "H": "H", "H2": "H", "HXT": "H"},
    "PHE": {"N": "D", "O": "A", "OXT": "A", "H": "H", "H2": "H", "HXT": "H"},
    "PRO": {"O": "A", "OXT": "A", "H": "H", "HXT": "H"},
    "SER": {
        "N": "D",
        "O": "A",
        "OG": "B",
        "OXT": "A",
        "H": "H",
        "H2": "H",
        "HG": "H",
        "HXT": "H",
    },
    "THR": {
        "N": "D",
        "O": "A",
        "OG1": "B",
        "OXT": "A",
        "H": "H",
        "H2": "H",
        "HG1": "H",
        "HXT": "H",
    },
    "TRP": {},
    "TYR": {"N": "D", "O": "A", "OH": "B", "OXT": "A", "H": "H", "H2": "H", "HH": "H", "HXT": "H"},
    "VAL": {"N": "D", "O": "A", "OXT": "A", "H": "H", "H2": "H", "HXT": "H"},
}

SRS_BONDS = {
    "ALA": (("N", "CA"), ("CA", "C"), ("CA", "CB"), ("C", "O"), ("C", "OXT"), ("N", "H"), ("N", "H2"), ("N", "H3"), ("CA", "HA"), ("CB", "HB3"), ("CB", "HB2"), ("CB", "HB1")),
    "ARG": (("N", "CA"), ("CA", "C"), ("CA", "CB"), ("C", "O"), ("C", "OXT"), ("CB", "CG"), ("CG", "CD"), ("CD", "NE"), ("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2"), ("N", "H"), ("N", "H2"), ("N", "H3"), ("CA", "HA"), ("CB", "HB3"), ("CB", "HB2"), ("CG", "HG3"), ("CG", "HG2"), ("CD", "HD3"), ("CD", "HD2"), ("NE", "HE"), ("NH1", "HH11"), ("NH1", "HH12"), ("NH2", "HH21"), ("NH2", "HH22")),
    "ASN": (("N", "CA"), ("CA", "C"), ("CA", "CB"), ("C", "O"), ("C", "OXT"), ("CB", "CG"), ("CG", "OD1"), ("CG", "ND2"), ("N", "H"), ("N", "H2"), ("N", "H3"), ("CA", "HA"), ("CB", "HB3"), ("CB", "HB2"), ("ND2", "HD21"), ("ND2", "HD22")),
    "ASP": (("N", "CA"), ("CA", "C"), ("CA", "CB"), ("C", "O"), ("C", "OXT"), ("CB", "CG"), ("CG", "OD1"), ("CG", "OD2"), ("N", "H"), ("N", "H2"), ("N", "H3"), ("CA", "HA"), ("CB", "HB3"), ("CB", "HB2")),
    "CYS": (("N", "CA"), ("CA", "C"), ("CA", "CB"), ("C", "O"), ("C", "OXT"), ("CB", "SG"), ("N", "H"), ("N", "H2"), ("N", "H3"), ("CA", "HA"), ("CB", "HB3"), ("CB", "HB2"), ("SG", "HG")),
    "GLN": (("N", "CA"), ("CA", "C"), ("CA", "CB"), ("C", "O"), ("C", "OXT"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2"), ("N", "H"), ("N", "H2"), ("N", "H3"), ("CA", "HA"), ("CB", "HB3"), ("CB", "HB2"), ("CG", "HG3"), ("CG", "HG2"), ("NE2", "HE21"), ("NE2", "HE22")),
    "GLU": (("N", "CA"), ("CA", "C"), ("CA", "CB"), ("C", "O"), ("C", "OXT"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2"), ("N", "H"), ("N", "H2"), ("N", "H3"), ("CA", "HA"), ("CB", "HB3"), ("CB", "HB2"), ("CG", "HG3"), ("CG", "HG2")),
    "GLY": (("N", "CA"), ("CA", "C"), ("C", "O"), ("C", "OXT"), ("N", "H"), ("N", "H2"), ("N", "H3"), ("CA", "HA3"), ("CA", "HA2")),
    "HIS": (("N", "CA"), ("CA", "C"), ("CA", "CB"), ("C", "O"), ("C", "OXT"), ("CB", "CG"), ("CG", "ND1"), ("CG", "CD2"), ("ND1", "CE1"), ("CD2", "NE2"), ("CE1", "NE2"), ("N", "H"), ("N", "H2"), ("N", "H3"), ("CA", "HA"), ("CB", "HB3"), ("CB", "HB2"), ("ND1", "HD1"), ("CD2", "HD2"), ("CE1", "HE1"), ("NE2", "HE2")),
    "ILE": (("N", "CA"), ("CA", "C"), ("CA", "CB"), ("C", "O"), ("C", "OXT"), ("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1"), ("N", "H"), ("N", "H2"), ("N", "H3"), ("CA", "HA"), ("CB", "HB"), ("CG1", "HG12"), ("CG1", "HG13"), ("CG2", "HG21"), ("CG2", "HG22"), ("CG2", "HG23"), ("CD1", "HD11"), ("CD1", "HD12"), ("CD1", "HD13")),
    "LEU": (("N", "CA"), ("CA", "C"), ("CA", "CB"), ("C", "O"), ("C", "OXT"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("N", "H"), ("N", "H2"), ("N", "H3"), ("CA", "HA"), ("CB", "HB3"), ("CB", "HB2"), ("CG", "HG"), ("CD1", "HD11"), ("CD1", "HD12"), ("CD1", "HD13"), ("CD2", "HD21"), ("CD2", "HD22"), ("CD2", "HD23")),
    "LYS": (("N", "CA"), ("CA", "C"), ("CA", "CB"), ("C", "O"), ("C", "OXT"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ"), ("N", "H"), ("N", "H2"), ("N", "H3"), ("CA", "HA"), ("CB", "HB3"), ("CB", "HB2"), ("CG", "HG3"), ("CG", "HG2"), ("CD", "HD3"), ("CD", "HD2"), ("CE", "HE3"), ("CE", "HE2"), ("NZ", "HZ1"), ("NZ", "HZ2"), ("NZ", "HZ3")),
    "MET": (("N", "CA"), ("CA", "C"), ("CA", "CB"), ("C", "O"), ("C", "OXT"), ("CB", "CG"), ("CG", "SD"), ("SD", "CE"), ("N", "H"), ("N", "H2"), ("N", "H3"), ("CA", "HA"), ("CB", "HB3"), ("CB", "HB2"), ("CG", "HG3"), ("CG", "HG2"), ("CE", "HE3"), ("CE", "HE2"), ("CE", "HE1")),
    "PHE": (("N", "CA"), ("CA", "C"), ("CA", "CB"), ("C", "O"), ("C", "OXT"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ"), ("N", "H"), ("N", "H2"), ("N", "H3"), ("CA", "HA"), ("CB", "HB3"), ("CB", "HB2"), ("CD1", "HD1"), ("CD2", "HD2"), ("CE1", "HE1"), ("CE2", "HE2"), ("CZ", "HZ")),
    "PRO": (("N", "CA"), ("N", "CD"), ("CA", "C"), ("CA", "CB"), ("C", "O"), ("C", "OXT"), ("CB", "CG"), ("CG", "CD"), ("N", "H"), ("N", "H2"), ("CA", "HA"), ("CB", "HB3"), ("CB", "HB2"), ("CG", "HG3"), ("CG", "HG2"), ("CD", "HD3"), ("CD", "HD2")),
    "SER": (("N", "CA"), ("CA", "C"), ("CA", "CB"), ("C", "O"), ("C", "OXT"), ("CB", "OG"), ("N", "H"), ("N", "H2"), ("N", "H3"), ("CA", "HA"), ("CB", "HB3"), ("CB", "HB2"), ("OG", "HG")),
    "THR": (("N", "CA"), ("CA", "C"), ("CA", "CB"), ("C", "O"), ("C", "OXT"), ("CB", "OG1"), ("CB", "CG2"), ("N", "H"), ("N", "H2"), ("N", "H3"), ("CA", "HA"), ("CB", "HB"), ("OG1", "HG1"), ("CG2", "HG21"), ("CG2", "HG22"), ("CG2", "HG23")),
    "TRP": (("N", "CA"), ("CA", "C"), ("CA", "CB"), ("C", "O"), ("C", "OXT"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "NE1"), ("CD2", "CE2"), ("CD2", "CE3"), ("NE1", "CE2"), ("CE2", "CZ2"), ("CE3", "CZ3"), ("CZ2", "CH2"), ("CZ3", "CH2"), ("N", "H"), ("N", "H2"), ("N", "H3"), ("CA", "HA"), ("CB", "HB3"), ("CB", "HB2"), ("CD1", "HD1"), ("NE1", "HE1"), ("CE3", "HE3"), ("CZ2", "HZ2"), ("CZ3", "HZ3"), ("CH2", "HH2")),
    "TYR": (("N", "CA"), ("CA", "C"), ("CA", "CB"), ("C", "O"), ("C", "OXT"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ"), ("CZ", "OH"), ("N", "H"), ("N", "H2"), ("N", "H3"), ("CA", "HA"), ("CB", "HB3"), ("CB", "HB2"), ("CD1", "HD1"), ("CD2", "HD2"), ("CE1", "HE1"), ("CE2", "HE2"), ("OH", "HH")),
    "VAL": (("N", "CA"), ("CA", "C"), ("CA", "CB"), ("C", "O"), ("C", "OXT"), ("CB", "CG1"), ("CB", "CG2"), ("N", "H"), ("N", "H2"), ("N", "H3"), ("CA", "HA"), ("CB", "HB"), ("CG1", "HG11"), ("CG1", "HG12"), ("CG1", "HG13"), ("CG2", "HG21"), ("CG2", "HG22"), ("CG2", "HG23")),
}


def _build_neighbours() -> dict[str, dict[str, tuple[str, ...]]]:
    neighbours: dict[str, dict[str, set[str]]] = {}
    for resname, bonds in SRS_BONDS.items():
        residue_neighbours: dict[str, set[str]] = {}
        for atom1, atom2 in bonds:
            residue_neighbours.setdefault(atom1, set()).add(atom2)
            residue_neighbours.setdefault(atom2, set()).add(atom1)
        neighbours[resname] = residue_neighbours
    return {
        resname: {atom: tuple(sorted(names)) for atom, names in atom_neighbours.items()}
        for resname, atom_neighbours in neighbours.items()
    }


SRS_NEIGHBOURS = _build_neighbours()
SRS_STANDARD_AA = frozenset(SRS_ATOM_HB_TYPES)
