from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple, Any, Dict, Type
from abc import ABC, abstractmethod
import json
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser
from ..core import Confidence

@dataclass
class Run:
    order: List[str]
    source: str
    load_model: Callable[[str], Tuple[Any, Confidence]]

class BaseParser(ABC):
    name: str = "base"

    @abstractmethod
    def detect(self, d: Path) -> bool: ...
    @abstractmethod
    def parse_run(self, d: Path) -> Run: ...

    @staticmethod
    def _read_json(p: Path) -> dict:
        try:
            with p.open() as fh: return json.load(fh)
        except Exception: return {}

    @staticmethod
    def _load_structure(path: str):
        p = Path(path)
        parser = MMCIFParser(QUIET=True) if p.suffix.lower()==".cif" else PDBParser(QUIET=True)
        return parser.get_structure("complex", str(p))

    @staticmethod
    def _guess_struct(d: Path, model: str) -> str:
        md = d / model
        if (md / "model.cif").exists(): return str(md / "model.cif")
        for ext in ("*.cif","*.pdb"):
            hits = list(d.glob(f"*{model}*{ext[1:]}"))
            if hits: return str(hits[0])
        raise ValueError(f"struct for model {model} not found")

    @staticmethod
    def _maps(struct):
        model = next(struct.get_models()); chains = list(model.get_chains())
        rim, cid, idx = {}, {}, 0
        for ch in chains:
            idxs = []
            for res in ch:
                rim[(ch.id, res.id)] = idx
                idxs.append(idx); idx += 1
            cid[ch.id] = idxs
        return chains, rim, cid

    @staticmethod
    def _plddt(chains, rim) -> List[float]:
        n = len(rim); out = [float('nan')] * n
        for ch in chains:
            for res in ch:
                i = rim.get((ch.id, res.id))
                if i is None: continue
                try:
                    out[i] = float(res["CA"].get_bfactor()); continue
                except Exception:
                    pass
                vals = [float(a.get_bfactor()) for a in res.get_atoms()
                        if a.element and a.element.upper()!="H"]
                out[i] = float(np.mean(vals)) if vals else float('nan')
        return out

    @staticmethod
    def _safe_float(x):
        try: return float(x)
        except Exception: return None


class ParserManager:
    def __init__(self):
        self._order: List[str] = []
        self._parsers: Dict[str, BaseParser] = {}

    def register(self, parser_cls: Type[BaseParser]) -> None:
        inst = parser_cls()
        self._parsers[inst.name] = inst
        if inst.name not in self._order:
            self._order.append(inst.name)

    def unregister(self, name: str) -> None:
        self._parsers.pop(name, None)
        if name in self._order: self._order.remove(name)

    def enable_only(self, names: List[str]) -> None:
        self._order = [n for n in names if n in self._parsers]

    def set_precedence(self, names: List[str]) -> None:
        remaining = [n for n in self._order if n not in names]
        self._order = [n for n in names if n in self._parsers] + remaining

    def list_parsers(self) -> List[str]:
        return [n for n in self._order if n in self._parsers]

    def pick(self, d: Path) -> BaseParser:
        for n in self._order:
            p = self._parsers.get(n)
            if p and p.detect(d): return p
        raise ValueError("no supported parser detected")

manager = ParserManager()

# import concrete parsers and register
from .af2 import AF2Parser
from .af3 import AF3Parser
manager.register(AF2Parser)
manager.register(AF3Parser)

pick_parser = manager.pick
register_parser = manager.register


