from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import gzip
import json
import lzma
from Bio.PDB import PDBParser, MMCIFParser
from ..confidence import Confidence
from ..geometry import is_pae_token_residue, representative_atom


@dataclass
class Run:
    order: list[str]
    source: str
    load_model: Callable[[str], tuple[Any, Confidence]]


class BaseParser(ABC):
    name: str = "base"

    @staticmethod
    @abstractmethod
    def detect(d: Path) -> bool: ...
    @abstractmethod
    def parse_run(self, d: Path) -> Run: ...

    # Compression magic numbers (file header bytes).
    _MAGIC = ((b"\xfd7zXZ\x00", lzma.open), (b"\x1f\x8b", gzip.open))

    @classmethod
    def _open_maybe_compressed(cls, p: Path, mode: str = "rt"):
        """Open ``p`` for reading, detecting xz/gz by its magic bytes.

        Compression is identified from the file header rather than the
        extension, so a compressed file is handled regardless of how it is
        named.
        """
        with p.open("rb") as fh:
            head = fh.read(6)
        for magic, opener in cls._MAGIC:
            if head.startswith(magic):
                return opener(p, mode)
        return p.open(mode)

    @classmethod
    def _read_json(cls, p: Path) -> dict:
        """Read a JSON file, transparently handling xz/gz compression.

        AlphaPulldown's ``--storage_mode slim/minimal`` may store large JSON
        sidecars (e.g. AF3 per-sample ``confidences.json``) compressed. If the
        plain path does not exist, fall back to a ``.xz`` or ``.gz`` sibling so
        scoring is unaffected by the storage mode. Compression is detected from
        the file's magic bytes, not its extension.
        """
        try:
            # Plain path absent: try a compressed sibling from slim/minimal.
            target = cls._first_existing(
                [p, p.with_name(p.name + ".xz"), p.with_name(p.name + ".gz")]
            )
            if target is None:
                return {}
            with cls._open_maybe_compressed(target) as fh:
                return json.load(fh)
        except Exception:
            return {}

    @staticmethod
    def _first_existing(paths) -> Path | None:
        return next((p for p in paths if p.exists()), None)

    @staticmethod
    def _load_structure(path: str):
        p = Path(path)
        parser = MMCIFParser(QUIET=True) if p.suffix.lower() == ".cif" else PDBParser(QUIET=True)
        return parser.get_structure("complex", str(p))

    @staticmethod
    def _guess_struct(d: Path, model: str) -> str:
        if (d / model / "model.cif").exists():
            return str(d / model / "model.cif")
        for ext in ("cif", "pdb"):
            hits = list(d.glob(f"*{model}*.{ext}"))
            if hits:
                return str(hits[0])
        raise ValueError(f"struct for model {model} not found")

    @staticmethod
    def _maps(struct):
        # Index PAE-token residues only (proteins with CA, nucleics with C1').
        # This keeps `rim` / `cid` aligned with the residue-by-residue PAE matrix
        # and with Complex._build_maps(), so plddt_residue[i] and pae_matrix[i, :]
        # refer to the same residue.
        model = next(struct.get_models())
        chains = list(model.get_chains())
        rim, cid, idx = {}, {}, 0
        for ch in chains:
            idxs: list[int] = []
            for res in ch:
                if not is_pae_token_residue(res):
                    continue
                rim[(ch.id, res.id)] = idx
                idxs.append(idx)
                idx += 1
            # Always register the chain so AF3 PAE normalisation can iterate over
            # `chains` and call cid[c.id] without KeyError; empty list is fine.
            cid[ch.id] = idxs
        return chains, rim, cid

    @staticmethod
    def _plddt(chains, rim) -> list[float]:
        # Use representative_atom (CB->CA for proteins, C1' for nucleics) so this
        # matches Interface._avg_plddt_union(): same residue, same B-factor.
        n = len(rim)
        out = [float("nan")] * n
        for ch in chains:
            for res in ch:
                i = rim.get((ch.id, res.id))
                if i is None:
                    continue
                try:
                    out[i] = float(representative_atom(res).get_bfactor())
                except (KeyError, AttributeError):
                    continue
        return out

    @staticmethod
    def _safe_float(x):
        try:
            return float(x)
        except Exception:
            return None


class ParserManager:
    """Registered parsers, tried in registration order."""

    def __init__(self):
        self._parsers: dict[str, BaseParser] = {}

    def register(self, parser_cls: type[BaseParser]) -> None:
        inst = parser_cls()
        self._parsers[inst.name] = inst

    def pick(self, d: Path) -> BaseParser:
        for p in self._parsers.values():
            if p.detect(d):
                return p
        raise ValueError("no supported parser detected")


manager = ParserManager()

# import concrete parsers and register
from .af2 import AF2Parser
from .af3 import AF3Parser
from .boltz import Boltz2Parser
manager.register(AF2Parser)
manager.register(AF3Parser)
manager.register(Boltz2Parser)

pick_parser = manager.pick
register_parser = manager.register
