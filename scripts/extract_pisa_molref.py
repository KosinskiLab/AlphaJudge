#!/usr/bin/env python3
"""Extract compact MolRef atom data from a local CCP4 PISA checkout.

The embedded runtime table in ``alphajudge.biophysics.pisa_molref`` was
generated from PISA's ``molref/molref.idx`` and ``molref/molref.rdt`` files.
The binary layout below follows ``pisalib/pisa_molref.cpp``:

* ``molref.idx`` starts with ``nEntries`` followed by entries containing
  ``name[4], classId, symNumber, offset``.
* ``molref.rdt`` stores each ``MolRefData`` as
  ``name[4], refName[4], classId, symNumber, nAtoms`` followed by atom records.
* Atom records are ``name[5], old_name[5], radius, refASA, asp, aspId``.
  MMDB writes these floating values as five-byte 16.16 fixed-point numbers.

This script prints a Python literal for the 20 standard amino acids. It is
developer-only provenance tooling; AlphaJudge does not read PISA files at
runtime.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

STANDARD_AA = (
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
)


def _clean_name(raw: bytes) -> str:
    return raw.replace(b"\0", b"").decode("latin1").strip()


def _read_i32(buf: bytes, offset: int) -> int:
    return struct.unpack_from("<i", buf, offset)[0]


def _read_mmdb_float5(buf: bytes, offset: int) -> float:
    """Read MMDB's five-byte fixed-point float used in PISA MolRef files."""
    if buf[offset : offset + 5] == b"\0\0\0\0\0":
        return 0.0
    raw = int.from_bytes(buf[offset + 1 : offset + 5], byteorder="big", signed=True)
    return raw / 65536.0


def _index_entries(index_path: Path) -> dict[str, int]:
    buf = index_path.read_bytes()
    n_entries = _read_i32(buf, 0)
    entries: dict[str, int] = {}
    pos = 4
    for _ in range(n_entries):
        name = _clean_name(buf[pos : pos + 4])
        offset = _read_i32(buf, pos + 12)
        entries[name] = offset
        pos += 16
    return entries


def _molref_record(data: bytes, offset: int) -> dict[str, tuple[float, float, float, int]]:
    n_atoms = _read_i32(data, offset + 16)
    pos = offset + 20
    atoms = {}
    for _ in range(n_atoms):
        atom_name = _clean_name(data[pos : pos + 5])
        radius = round(_read_mmdb_float5(data, pos + 10), 6)
        ref_asa = round(_read_mmdb_float5(data, pos + 15), 6)
        asp = round(_read_mmdb_float5(data, pos + 20), 8)
        asp_id = _read_i32(data, pos + 25)
        atoms[atom_name] = (radius, ref_asa, asp, asp_id)
        pos += 29
    return atoms


def extract_molref(molref_dir: Path) -> dict[str, dict[str, tuple[float, float, float, int]]]:
    entries = _index_entries(molref_dir / "molref.idx")
    data = (molref_dir / "molref.rdt").read_bytes()
    return {residue: _molref_record(data, entries[residue]) for residue in STANDARD_AA}


def _print_python_table(table: dict[str, dict[str, tuple[float, float, float, int]]]) -> None:
    print("# residue -> atom -> (radius, unfolded reference ASA, atom ASP, aspId)")
    print("PISA_STANDARD_AA_MOLREF = {")
    for residue, atoms in table.items():
        print(f"    {residue!r}: {{")
        for atom, data in atoms.items():
            print(f"        {atom!r}: {data!r},")
        print("    },")
    print("}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "molref_dir",
        nargs="?",
        type=Path,
        default=Path("/g/kosinski/dima/PycharmProjects/pisa/molref"),
        help="Directory containing PISA molref.idx and molref.rdt",
    )
    args = parser.parse_args()

    table = extract_molref(args.molref_dir)
    _print_python_table(table)


if __name__ == "__main__":
    main()
