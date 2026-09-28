#!/usr/bin/env python3
"""
Generate CCP4 PISA/SC reference values for AlphaJudge biophysics tests.

This script is intentionally dev-only. AlphaJudge itself must not import or
shell out to CCP4 at runtime, but these references give us a numerical target
for the pure-Python implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from Bio.PDB import MMCIFParser, PDBIO, PDBParser


REPO_ROOT = Path(__file__).resolve().parents[1]
CCP4_MODULE = "ccp4/7.1.000-shelx-arpwarp-linux64"
DEFAULT_FIXTURES = (
    "test_data/af2/pos_dimers/Q13148+Q92900/ranked_0.pdb",
    "test_data/af2/pos_dimers/Q9BUL8+Q13033/ranked_0.pdb",
    "test_data/af2/neg_dimers/Q14974+Q13033/ranked_0.pdb",
    "test_data/af3/pos_dimers/Q13148+Q92900/ranked_0_model.cif",
    "test_data/af3/pos_dimers/Q9BUL8+Q13033/ranked_0_model.cif",
    "test_data/af3/neg_dimers/Q14974+Q13033/ranked_0_model.cif",
)


@dataclass(frozen=True)
class Fixture:
    path: Path
    chains: tuple[str, str]

    @property
    def id(self) -> str:
        rel = self.path.relative_to(REPO_ROOT)
        return str(rel).replace("/", "__").replace("+", "_plus_").replace(".", "_")

    @property
    def relpath(self) -> str:
        return str(self.path.relative_to(REPO_ROOT))


def _run_ccp4(command: str, *, cwd: Path, timeout: int = 900) -> subprocess.CompletedProcess[str]:
    scratch = cwd / "ccp4_scr"
    wrapped = (
        f"module load {CCP4_MODULE}; "
        f"mkdir -p {shlex.quote(str(scratch))}; "
        f"export CCP4_SCR={shlex.quote(str(scratch))}; "
        f"{command}"
    )
    return subprocess.run(
        ["bash", "-lc", wrapped],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=True,
    )


def _parser_for(path: Path):
    return MMCIFParser(QUIET=True) if path.suffix.lower() == ".cif" else PDBParser(QUIET=True)


def _infer_chains(path: Path) -> tuple[str, str]:
    structure = _parser_for(path).get_structure("fixture", str(path))
    chains = [chain.id for chain in structure[0]]
    if len(chains) != 2:
        raise ValueError(f"{path} must contain exactly two chains; found {chains}")
    return chains[0], chains[1]


def _as_pdb(path: Path, out_dir: Path) -> Path:
    if path.suffix.lower() == ".pdb":
        return path
    structure = MMCIFParser(QUIET=True).get_structure(path.stem, str(path))
    out = out_dir / f"{path.stem}.pdb"
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out))
    return out


def _text(node: ET.Element, name: str, default: str = "") -> str:
    found = node.find(name)
    if found is None or found.text is None:
        return default
    return found.text.strip()


_BOND_FIELDS = {
    f"{field}_{side}": f"{tag}-{side}"
    for side in (1, 2)
    for field, tag in (("chain", "chain"), ("res", "res"), ("seqnum", "seqnum"), ("atom", "atname"))
}


def _parse_bonds(interface: ET.Element, tag: str) -> tuple[int, list[dict[str, str | float]]]:
    """Reported bond count and per-bond records of one PISA bond section."""
    section = interface.find(tag)
    if section is None:
        return 0, []
    bonds = [
        {**{key: _text(bond, xml_tag) for key, xml_tag in _BOND_FIELDS.items()},
         "dist": float(_text(bond, "dist", "nan"))}
        for bond in section.findall("bond")
    ]
    return int(_text(section, "n_bonds", "0")), bonds


def _parse_pisa_xml(xml_text: str) -> dict:
    start = xml_text.find("<pdb_entry>")
    if start > 0:
        xml_text = xml_text[start:]
    root = ET.fromstring(xml_text)
    interface = root.find("interface")
    if interface is None:
        return {
            "area": 0.0,
            "hb": 0,
            "sb": 0,
            "ss": 0,
            "int_solv_en": 0.0,
            "bonds": {"hb": [], "sb": [], "ss": []},
        }

    n_hb, hb = _parse_bonds(interface, "h-bonds")
    n_sb, sb = _parse_bonds(interface, "salt-bridges")
    n_ss, ss = _parse_bonds(interface, "ss-bonds")
    return {
        "area": float(_text(interface, "int_area", "0")),
        "hb": n_hb,
        "sb": n_sb,
        "ss": n_ss,
        "int_solv_en": float(_text(interface, "int_solv_en", "0")),
        "bonds": {"hb": hb, "sb": sb, "ss": ss},
    }


def _run_pisa(fixture: Fixture, pdb_path: Path, out_dir: Path) -> dict:
    digest = hashlib.sha1(fixture.relpath.encode()).hexdigest()[:12]
    session = f"aj_{digest}"
    work = out_dir / fixture.id
    work.mkdir(parents=True, exist_ok=True)
    try:
        analyse = _run_ccp4(f"pisa {session} -analyse {pdb_path}", cwd=work)
        (work / "pisa_analyse.log").write_text(analyse.stdout + analyse.stderr)
        xml = _run_ccp4(f"pisa {session} -xml interfaces", cwd=work).stdout
        (work / "pisa_interfaces.xml").write_text(xml)
        listing = _run_ccp4(f"pisa {session} -list interfaces", cwd=work).stdout
        (work / "pisa_interfaces.txt").write_text(listing)
    finally:
        _run_ccp4(f"pisa {session} -erase", cwd=work)
    return _parse_pisa_xml(xml)


def _run_sc(fixture: Fixture, pdb_path: Path, out_dir: Path) -> float:
    work = out_dir / fixture.id
    work.mkdir(parents=True, exist_ok=True)
    c1, c2 = fixture.chains
    stdin = f"MOLECULE 1\\nCHAIN {c1}\\nMOLECULE 2\\nCHAIN {c2}\\nEND\\n"
    cmd = f"printf '%b' {json.dumps(stdin)} | sc XYZIN {pdb_path}"
    result = _run_ccp4(cmd, cwd=work, timeout=1800)
    log = result.stdout + result.stderr
    (work / "sc.log").write_text(log)
    match = re.search(r"Shape complementarity statistic Sc =\s*([-+0-9.]+)", log)
    if not match:
        raise RuntimeError(f"could not parse SC output for {fixture.relpath}")
    return float(match.group(1))


def _fixtures(paths: Iterable[str]) -> list[Fixture]:
    fixtures = []
    for raw in paths:
        path = (REPO_ROOT / raw).resolve()
        fixtures.append(Fixture(path=path, chains=_infer_chains(path)))
    return fixtures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "test" / "fixtures" / "ccp4_biophysics_reference.json",
    )
    parser.add_argument(
        "--scratch",
        type=Path,
        default=REPO_ROOT / "tmp" / "ccp4_validation",
    )
    parser.add_argument("--fixture", action="append", dest="fixtures")
    parser.add_argument("--skip-sc", action="store_true", help="only run PISA")
    args = parser.parse_args()

    fixture_paths = args.fixtures or list(DEFAULT_FIXTURES)
    fixtures = _fixtures(fixture_paths)

    args.scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="alphajudge_ccp4_") as td:
        converted = Path(td)
        references = []
        for fixture in fixtures:
            pdb_path = _as_pdb(fixture.path, converted)
            pisa = _run_pisa(fixture, pdb_path, args.scratch)
            sc = None if args.skip_sc else _run_sc(fixture, pdb_path, args.scratch)
            references.append(
                {
                    "id": fixture.id,
                    "path": fixture.relpath,
                    "chains": list(fixture.chains),
                    "pisa": pisa,
                    "sc": sc,
                }
            )

    payload = {
        "ccp4_module": CCP4_MODULE,
        "generated_by": "scripts/validate_ccp4_biophysics.py",
        "references": references,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
