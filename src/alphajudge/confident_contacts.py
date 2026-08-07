"""Confident contact count (CCC).

CCC counts inter-chain residue pairs that are simultaneously (i) in physical
contact and (ii) confidently placed relative to one another, the latter judged
by the predicted aligned error.  It was introduced by Lambourne et al. as a
screening statistic for AlphaFold interaction predictions and reported there to
separate real interactions from random pairs better than the confidence scores
in common use at low false-positive rates.

Two conventions have to be fixed to make the count reproducible, and both are
exposed here rather than hard-coded.

**Contact geometry.**  ``INTERACTOME3D`` is the default and follows Mosca et
al.'s Interactome3D definition, which Lambourne et al. adopt.  That definition
lists four rules -- disulfides (Cys S--S <= 2.56 A), hydrogen bonds (N--O <=
3.5 A), salt bridges (N--O <= 5.5 A) and van der Waals contacts (C--C <= 5.0 A)
-- but the hydrogen-bond rule is strictly subsumed by the salt-bridge rule,
being the same atom pair at a shorter distance.  Three tests therefore decide
membership:

* a Cys--Cys sulfur pair within ``2.56`` A,
* a nitrogen--oxygen pair within ``5.5`` A, or
* a carbon--carbon pair within ``5.0`` A

for any heavy-atom pair of the two residues, and each residue pair is counted
once however many atom pairs qualify.  ``REPRESENTATIVE_ATOM`` instead
reuses the contact definition the rest of AlphaJudge uses -- representative
atom, CB else CA, within the complex's ``contact_thresh`` -- so that CCC,
``contact_pairs`` and ``cLIS`` are computed over one identical pair set.  The
two are not interchangeable and the choice is recorded in the score name.

**PAE direction.**  The predicted aligned error is asymmetric.  ``ab`` scores
each pair once, in the order (chain1 residue, chain2 residue), which is the
published convention.  ``ba`` is the reverse, and ``min``/``mean``/``max``
symmetrise.  The non-default directions exist to audit how much of the count
depends on the asymmetry.

.. note::
   The publication is internally inconsistent about the PAE boundary: its main
   text twice says ``PAE <= 4`` A, while the Methods section heading says ``PAE
   < 4`` A.  The authors' released analysis code settles it -- the column
   carrying the count is named ``n_contacts_PAE_lt_4A`` -- so the strict
   comparison is the operational definition and is the default here.  Pass
   ``inclusive=True`` for the main text's reading; the two differ only by the
   contacts sitting exactly on the cutoff.

   The one convention the published material does not fix is the PAE direction.
   The predicted aligned error is asymmetric and the paper refers to "the"
   PAE of a contact without saying which entry.  ``ab`` is the default here.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from .geometry import representative_atom


#: Published PAE cutoff, in Angstroms.  A pair counts when its PAE is strictly
#: below this value (see the module note on the ``<`` / ``<=`` discrepancy).
DEFAULT_PAE_CUTOFF = 4.0

#: Interactome3D-style atom-pair rules: (element pair, maximum distance).
DISULFIDE_MAX = 2.56
NITROGEN_OXYGEN_MAX = 5.5
CARBON_CARBON_MAX = 5.0

#: Widest of the atom-pair rules; used as a neighbour-search prefilter so that
#: no qualifying pair can be missed before the per-rule test is applied.
_PREFILTER_MAX = max(DISULFIDE_MAX, NITROGEN_OXYGEN_MAX, CARBON_CARBON_MAX)


class ContactGeometry(str, Enum):
    """How an inter-chain residue pair is judged to be in contact."""

    INTERACTOME3D = "interactome3d"
    REPRESENTATIVE_ATOM = "representative_atom"


class PaeDirection(str, Enum):
    """Which entry of the asymmetric PAE matrix gates a contact."""

    AB = "ab"
    BA = "ba"
    MIN = "min"
    MEAN = "mean"
    MAX = "max"


def _element(atom) -> str:
    return str(getattr(atom, "element", "") or "").strip().upper()


def _interactome3d_contact(atom1, atom2) -> bool:
    """Whether one heavy-atom pair satisfies any Interactome3D-style rule."""
    e1, e2 = _element(atom1), _element(atom2)
    if "H" in (e1, e2):
        return False
    distance = float(atom1 - atom2)
    if e1 == e2 == "C":
        return distance <= CARBON_CARBON_MAX
    if {e1, e2} == {"N", "O"}:
        return distance <= NITROGEN_OXYGEN_MAX
    if e1 == e2 == "S":
        return (
            distance <= DISULFIDE_MAX
            and atom1.get_parent().get_resname().strip().upper() == "CYS"
            and atom2.get_parent().get_resname().strip().upper() == "CYS"
        )
    return False


def interactome3d_contact_pairs(chain1, chain2) -> set[tuple[object, object]]:
    """Unique (chain1 residue, chain2 residue) pairs in Interactome3D-style contact.

    Ordering is always (chain1, chain2) so that the caller can apply a
    directional PAE convention without re-deriving which residue came from
    which chain.
    """
    from Bio.PDB import NeighborSearch

    residues1, residues2 = list(chain1), list(chain2)
    if not residues1 or not residues2:
        return set()

    from_chain1 = {id(res) for res in residues1}
    atoms = [
        atom
        for res in residues1 + residues2
        for atom in res.get_atoms()
        if _element(atom) != "H"
    ]
    if not atoms:
        return set()

    pairs: set[tuple[object, object]] = set()
    for atom1, atom2 in NeighborSearch(atoms).search_all(_PREFILTER_MAX, level="A"):
        res1, res2 = atom1.get_parent(), atom2.get_parent()
        in1, in2 = id(res1) in from_chain1, id(res2) in from_chain1
        if in1 == in2:  # same chain, or neither recognised
            continue
        if not in1:  # orient as (chain1, chain2)
            atom1, atom2 = atom2, atom1
            res1, res2 = res2, res1
        if (res1, res2) in pairs:
            continue
        if _interactome3d_contact(atom1, atom2):
            pairs.add((res1, res2))
    return pairs


def representative_atom_contact_pairs(chain1, chain2, contact_thresh: float):
    """Unique (chain1, chain2) pairs within ``contact_thresh`` on CB (CA for Gly)."""
    reps1 = [(res, representative_atom(res)) for res in chain1]
    reps2 = [(res, representative_atom(res)) for res in chain2]
    pairs: set[tuple[object, object]] = set()
    for res1, atom1 in reps1:
        if atom1 is None:
            continue
        for res2, atom2 in reps2:
            if atom2 is None:
                continue
            if float(atom1 - atom2) <= float(contact_thresh):
                pairs.add((res1, res2))
    return pairs


def _gated_pae(pae, i: int, j: int, direction: PaeDirection) -> float:
    ab, ba = float(pae[i, j]), float(pae[j, i])
    if direction is PaeDirection.AB:
        return ab
    if direction is PaeDirection.BA:
        return ba
    if direction is PaeDirection.MIN:
        return min(ab, ba)
    if direction is PaeDirection.MAX:
        return max(ab, ba)
    return 0.5 * (ab + ba)


def confident_contact_count(
    pairs,
    pae,
    res_index_map,
    pae_cutoff: float = DEFAULT_PAE_CUTOFF,
    direction: PaeDirection = PaeDirection.AB,
    inclusive: bool = False,
) -> int:
    """Count contacting residue pairs whose PAE is at or below ``pae_cutoff``.

    ``pairs`` must already be oriented (chain1 residue, chain2 residue).  Pairs
    whose residues carry no PAE token -- ligands and other non-polymer entities
    -- are skipped rather than counted as confident.

    The comparison is ``PAE < cutoff``, matching the authors' released code.
    ``inclusive=True`` switches to ``PAE <= cutoff``, which is what the
    publication's main text says; the two differ only by the contacts sitting
    exactly on the cutoff.
    """
    if pae is None or not len(pairs):
        return 0
    pae_arr = np.asarray(pae)
    cutoff = float(pae_cutoff)
    count = 0
    for res1, res2 in pairs:
        i = res_index_map.get((res1.get_parent().id, res1.id))
        j = res_index_map.get((res2.get_parent().id, res2.id))
        if i is None or j is None:
            continue
        gated = _gated_pae(pae_arr, i, j, direction)
        if (gated <= cutoff) if inclusive else (gated < cutoff):
            count += 1
    return count
