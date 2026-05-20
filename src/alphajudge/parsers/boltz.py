from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any

import numpy as np

from . import BaseParser, Run
from ..confidence import Confidence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _BoltzModel:
    name: str
    rank: int
    structure_file: Path
    confidence_file: Path
    pae_file: Path | None
    plddt_file: Path | None


class Boltz2Parser(BaseParser):
    name = "boltz2"

    @staticmethod
    def detect(d: Path) -> bool:
        return bool(Boltz2Parser._model_entries(d))

    def parse_run(self, d: Path) -> Run:
        entries = self._model_entries(d)
        if not entries:
            raise ValueError(f"Boltz-2 prediction files not found in {d}")
        by_name = {entry.name: entry for entry in entries}
        order = [entry.name for entry in entries]

        def load_model(model: str):
            entry = by_name[model]
            struct = self._load_structure(str(entry.structure_file))
            chains, rim, cid = self._maps(struct)

            summary = self._read_json(entry.confidence_file)
            pae, max_pae = self._load_pae(entry.pae_file, chains, cid)
            plddt = self._load_plddt(entry.plddt_file, rim) or self._plddt(chains, rim)

            iptm = self._safe_float(summary.get("iptm"))
            ptm = self._safe_float(summary.get("ptm"))
            confidence_score = self._safe_float(summary.get("confidence_score"))
            iptm_ptm = 0.8 * iptm + 0.2 * ptm if (iptm is not None and ptm is not None) else None
            chain_pair_iptm = self._chain_pair_matrix(summary.get("pair_chains_iptm"))

            return struct, Confidence(
                pae_matrix=pae,
                max_pae=max_pae,
                iptm=iptm,
                ptm=ptm,
                iptm_ptm=iptm_ptm,
                confidence_score=confidence_score,
                plddt_residue=plddt,
                chain_pair_iptm=chain_pair_iptm,
            )

        return Run(order=order, source="boltz2", load_model=load_model)

    @staticmethod
    def _model_entries(d: Path) -> list[_BoltzModel]:
        if not d.is_dir():
            return []

        entries: list[_BoltzModel] = []
        for confidence_file in sorted(d.glob("confidence_*_model_*.json")):
            model_name = confidence_file.name.removeprefix("confidence_").removesuffix(".json")
            structure_file = Boltz2Parser._first_existing(
                [d / f"{model_name}.cif", d / f"{model_name}.pdb"]
            )
            if structure_file is None:
                continue

            entries.append(
                _BoltzModel(
                    name=model_name,
                    rank=Boltz2Parser._rank_from_model_name(model_name),
                    structure_file=structure_file,
                    confidence_file=confidence_file,
                    pae_file=Boltz2Parser._first_existing([d / f"pae_{model_name}.npz"]),
                    plddt_file=Boltz2Parser._first_existing([d / f"plddt_{model_name}.npz"]),
                )
            )

        return sorted(entries, key=lambda entry: (entry.rank, entry.name))

    @staticmethod
    def _first_existing(paths: list[Path]) -> Path | None:
        for path in paths:
            if path.exists():
                return path
        return None

    @staticmethod
    def _rank_from_model_name(model_name: str) -> int:
        match = re.search(r"_model_(\d+)$", model_name)
        if match is None:
            return 10**9
        return int(match.group(1))

    @staticmethod
    def _load_npz_array(path: Path, preferred_key: str) -> np.ndarray | None:
        try:
            with np.load(path) as payload:
                key = preferred_key if preferred_key in payload else payload.files[0]
                return np.array(payload[key], dtype=float)
        except Exception as exc:
            logger.warning(f"could not read {path}: {exc}")
            return None

    @classmethod
    def _load_pae(cls, pae_file: Path | None, chains, cid) -> tuple[np.ndarray, float]:
        total = sum(len(cid[c.id]) for c in chains)
        pae = np.full((total, total), 100.0, dtype=float)
        if pae_file is None:
            return pae, 100.0

        matrix = cls._load_npz_array(pae_file, "pae")
        if matrix is None or not matrix.size:
            return pae, 100.0

        max_pae = float(np.nanmax(matrix))
        if matrix.shape == pae.shape:
            return matrix, max_pae

        if matrix.ndim == 2 and matrix.shape[0] >= total and matrix.shape[1] >= total:
            trimmed = matrix[:total, :total]
            return trimmed, float(np.nanmax(trimmed)) if trimmed.size else 100.0

        logger.warning(
            f"Boltz-2 PAE shape {matrix.shape} != expected {pae.shape}; "
            "using default PAE=100 for all residue pairs."
        )
        return pae, max_pae

    @classmethod
    def _load_plddt(cls, plddt_file: Path | None, rim: dict[tuple[str, Any], int]) -> list[float] | None:
        if plddt_file is None:
            return None
        values = cls._load_npz_array(plddt_file, "plddt")
        if values is None:
            return None
        flat = values.ravel()
        if flat.size != len(rim):
            if flat.size > len(rim):
                return [float(v) for v in flat[:len(rim)]]
            logger.warning(
                f"Boltz-2 pLDDT length {flat.size} != expected {len(rim)}; "
                "using structure B-factors instead."
            )
            return None
        return [float(v) for v in flat]

    @staticmethod
    def _chain_pair_matrix(raw: Any) -> list[list[float]] | None:
        if isinstance(raw, (list, tuple)):
            return [list(row) if isinstance(row, (list, tuple)) else [] for row in raw]
        if not isinstance(raw, dict):
            return None

        indices: set[int] = set()
        for key, row in raw.items():
            try:
                indices.add(int(key))
            except (TypeError, ValueError):
                continue
            if isinstance(row, dict):
                for nested_key in row:
                    try:
                        indices.add(int(nested_key))
                    except (TypeError, ValueError):
                        continue
        if not indices:
            return None

        size = max(indices) + 1
        matrix = [[float("nan") for _ in range(size)] for _ in range(size)]
        for key, row in raw.items():
            if not isinstance(row, dict):
                continue
            try:
                i = int(key)
            except (TypeError, ValueError):
                continue
            for nested_key, value in row.items():
                try:
                    j = int(nested_key)
                    matrix[i][j] = float(value)
                except (TypeError, ValueError):
                    continue
        return matrix
