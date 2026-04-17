from __future__ import annotations

from Bio.PDB.Polypeptide import is_aa


NA_RES: set[str] = {
    "A", "C", "G", "U",
    "DA", "DC", "DG", "DT", "DU",
    "RA", "RC", "RG", "RU",
}

POLAR_RES: set[str] = {"SER", "THR", "ASN", "GLN", "TYR", "CYS"}
HYDROPHOBIC_RES: set[str] = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP"}
CHARGED_RES: set[str] = {"ARG", "LYS", "ASP", "GLU", "HIS"}

CHARGED_ATOMS: dict[str, set[str]] = {
    "ARG": {"NE", "CZ", "NH1", "NH2"},
    "LYS": {"NZ"},
    "ASP": {"CG", "OD1", "OD2"},
    "GLU": {"CD", "OE1", "OE2"},
}


def representative_atom(res):
    """
    Representative atom for distance / bfactor use.
    - Proteins: CB (else CA)
    - Nucleic acids: C1' / C1* / C1 / P
    - Fallback: first heavy atom
    """
    for name in ("CB", "CA", "C1'", "C1*", "C1", "P"):
        if name in res:
            return res[name]
    for atom in res.get_atoms():
        if ((atom.element or "").upper() != "H"):
            return atom
    raise KeyError("No representative atom found for residue")


def is_pae_token_residue(res) -> bool:
    """
    Keep only residues that correspond to PAE residue tokens.
    - Keep amino acids (standard or modified) if they have CA.
    - Keep nucleic acids if they have C1' / C1* / C1.
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


_repr_atom = representative_atom
