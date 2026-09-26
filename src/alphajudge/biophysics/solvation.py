"""PISA MolRef/ASP-style interface solvation energy."""

from __future__ import annotations

import numpy as np

from .pisa_molref import PISA_ASP_PARAMETERS, PISA_STANDARD_AA_MOLREF
from .prosurf import PISA_CODE_NO, PISA_PROBE_RADIUS, _atom_element, _pisa_interface_result

ASP_SPECIAL = -2
ASP_CHARGED_N = 4
ASP_CHARGED_O = 5
ASP_NEUTRAL_NO = 3
ASP_OTHER = 0


def _molref_for_atom(atom) -> tuple[float, float, float, int] | None:
    residue = atom.get_parent()
    resname = residue.get_resname().strip().upper()
    atom_name = atom.id.strip().upper()
    return PISA_STANDARD_AA_MOLREF.get(resname, {}).get(atom_name)


def _state_sas_lookup(atoms, state_sas: np.ndarray) -> dict[tuple[int, str], float]:
    return {
        (id(atom.get_parent()), atom.id.strip().upper()): float(sas)
        for atom, sas in zip(atoms, state_sas)
    }


# Charged nitrogens whose charge PISA assigns to the most exposed of the group.
_CHARGED_N_PARTNERS = {
    "NE2": ("ND1",),
    "ND1": ("NE2",),
    "NE": ("NH1", "NH2"),
    "NH1": ("NE", "NH2"),
    "NH2": ("NE", "NH1"),
}


def _asp_type_pair(
    atom,
    paired_names: tuple[str, ...],
    charged: int,
    state_sas: dict[tuple[int, str], float],
) -> int:
    residue_id = id(atom.get_parent())
    current = state_sas.get((residue_id, atom.id.strip().upper()), 0.0)
    paired = [state_sas[(residue_id, name)] for name in paired_names if (residue_id, name) in state_sas]
    if not paired:
        return charged
    if len(paired) == 1:
        return charged if current > paired[0] else ASP_NEUTRAL_NO
    return charged if current == max([current] + paired) else ASP_NEUTRAL_NO


def _special_asp_type(atom, state_sas: dict[tuple[int, str], float]) -> int:
    element = _atom_element(atom)
    atom_name = atom.id.strip().upper()

    if element == "N":
        if atom_name == "NE2" and atom.get_parent().get_resname().strip().upper() != "HIS":
            return ASP_NEUTRAL_NO
        partners = _CHARGED_N_PARTNERS.get(atom_name)
        return _asp_type_pair(atom, partners, ASP_CHARGED_N, state_sas) if partners else ASP_CHARGED_N

    if element == "O":
        if atom_name.endswith("1"):
            partner = atom_name[:-1] + "2"
        elif atom_name.endswith("2"):
            partner = atom_name[:-1] + "1"
        else:
            return ASP_NEUTRAL_NO
        return _asp_type_pair(atom, (partner,), ASP_CHARGED_O, state_sas)

    return ASP_OTHER


def _pisa_solvation_energy_for_state(atoms, state_sas: np.ndarray) -> float:
    state_lookup = _state_sas_lookup(atoms, state_sas)
    asp_area = [0.0] * len(PISA_ASP_PARAMETERS)
    energy = 0.0

    for atom, sas in zip(atoms, state_sas):
        molref = _molref_for_atom(atom)
        if molref is None:
            continue
        _, ref_asa, atom_asp, asp_id = molref
        if ref_asa < -0.5:
            continue
        asp_type = asp_id
        if asp_type == ASP_SPECIAL:
            asp_type = _special_asp_type(atom, state_lookup)
        if asp_type < 0:
            asp_type = ASP_OTHER

        ref_area = float(sas) - ref_asa
        if asp_type < len(asp_area):
            asp_area[asp_type] += ref_area
        if asp_type <= ASP_OTHER:
            energy += atom_asp * float(sas)

    for asp_type, area in enumerate(asp_area):
        energy += area * PISA_ASP_PARAMETERS[asp_type]
    return float(energy)


def _side_interface_solvation_energy(atoms, atom_sas, atom_int_sas) -> float:
    atom_sas_arr = np.asarray(atom_sas, dtype=float)
    atom_int_sas_arr = np.asarray(atom_int_sas, dtype=float)
    if len(atom_sas_arr) == 0:
        return 0.0
    free_energy = _pisa_solvation_energy_for_state(atoms, atom_sas_arr)
    bound_sas = atom_sas_arr - atom_int_sas_arr
    bound_sas[bound_sas < 1.0e-8] = 0.0
    bound_energy = _pisa_solvation_energy_for_state(atoms, bound_sas)
    return float(bound_energy - free_energy)


def interface_solvation_energy(
    residues1,
    residues2,
    probe_radius: float = PISA_PROBE_RADIUS,
    code_no: int = PISA_CODE_NO,
) -> float:
    """PISA-style interface solvation energy gain in kcal/mol."""
    result = _pisa_interface_result(residues1, residues2, probe_radius, code_no)
    return (
        _side_interface_solvation_energy(result.atoms1, result.atom_sas1, result.atom_int_sas1)
        + _side_interface_solvation_energy(result.atoms2, result.atom_sas2, result.atom_int_sas2)
    )


__all__ = ["interface_solvation_energy"]
