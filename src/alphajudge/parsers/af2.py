from __future__ import annotations
import gzip
import logging
import lzma
import pickle
from pathlib import Path
import numpy as np
from . import BaseParser, Run
from ..confidence import Confidence
from ..contact_probs import (
    AF2_DISTOGRAM_CONTACT_CUTOFF,
    contact_probs_from_distogram,
)

logger = logging.getLogger(__name__)

class AF2Parser(BaseParser):
    name = "af2"
    _warned_missing_distogram = False

    def detect(self, d: Path) -> bool:
        return (d / "ranking_debug.json").exists()

    def parse_run(self, d: Path) -> Run:
        rj = self._read_json(d / "ranking_debug.json")
        order = rj["order"]

        def load_model(model: str):
            struct = self._load_structure(self._guess_struct(d, model))
            chains, rim, _ = self._maps(struct)

            # AF2: full residue×residue matrix in pae_{model}.json
            pae_payload = self._read_json(d / f"pae_{model}.json")
            pae = np.array(pae_payload[0]["predicted_aligned_error"], dtype=float)
            max_pae = float(np.nanmax(pae) if pae.size else np.nan)
            contact_probs = self._load_contact_probs_from_result_pkl(d, model, pae.shape)

            # AF2 rankings
            is_multimer = ("iptm+ptm" in rj) and ("iptm" in rj)
            if is_multimer:
                iptm = self._safe_float(rj["iptm"].get(model))
                ptm  = self._safe_float(rj.get("ptm", {}).get(model))
                iptm_ptm = self._safe_float(rj["iptm+ptm"].get(model))
                # Backfill when PTM is not provided in AF2 multimer JSON
                if ptm is None and (iptm_ptm is not None) and (iptm is not None):
                    try:
                        ptm = (iptm_ptm - 0.8 * iptm) / 0.2
                    except Exception:
                        ptm = None
                # If iptm+ptm itself is missing but both iptm and ptm exist, derive it
                if iptm_ptm is None and (iptm is not None) and (ptm is not None):
                    iptm_ptm = 0.8 * iptm + 0.2 * ptm
                conf = iptm_ptm
            else:
                iptm, ptm = 0.0, self._safe_float(rj["ptm"][model])
                iptm_ptm = ptm; conf = ptm

            plddt = self._plddt(chains, rim)
            return struct, Confidence(
                pae_matrix=pae, max_pae=max_pae,
                iptm=iptm, ptm=ptm, iptm_ptm=iptm_ptm, confidence_score=conf,
                plddt_residue=plddt,
                contact_prob_matrix=contact_probs,
                contact_prob_source=(
                    f"af2_distogram_le_{AF2_DISTOGRAM_CONTACT_CUTOFF:g}A"
                    if contact_probs is not None
                    else None
                ),
            )
        return Run(order=order, source="af2", load_model=load_model)

    @classmethod
    def _load_contact_probs_from_result_pkl(
        cls, d: Path, model: str, expected_shape: tuple[int, int]
    ) -> np.ndarray | None:
        result_pkl = cls._find_result_pkl(d, model)
        if result_pkl is None:
            return None

        try:
            payload = cls._read_pickle(result_pkl)
        except Exception as e:
            logger.warning(f"could not read AF2 result pickle {result_pkl}: {e}")
            return None

        if not isinstance(payload, dict):
            return None
        distogram = payload.get("distogram")
        if not isinstance(distogram, dict):
            if not cls._warned_missing_distogram:
                logger.warning(
                    "AF2 result pickle %s has no distogram; AF2 contact-probability "
                    "columns will be empty/NaN. Full AlphaPulldown result pickles are "
                    "required; disable --remove_keys_from_pickles to retain distograms.",
                    result_pkl,
                )
                cls._warned_missing_distogram = True
            else:
                logger.debug(
                    "AF2 result pickle %s has no distogram; contact scores unavailable.",
                    result_pkl,
                )
            return None
        logits = distogram.get("logits")
        bin_edges = distogram.get("bin_edges")
        if logits is None or bin_edges is None:
            return None

        try:
            contact_probs = contact_probs_from_distogram(
                np.asarray(logits),
                np.asarray(bin_edges),
                AF2_DISTOGRAM_CONTACT_CUTOFF,
            )
        except Exception as e:
            logger.warning(f"could not derive AF2 contact probabilities from {result_pkl}: {e}")
            return None

        if contact_probs.shape != expected_shape:
            logger.warning(
                f"AF2 contact probability shape {contact_probs.shape} != expected "
                f"{expected_shape}; skipping contact probabilities."
            )
            return None

        return contact_probs

    @staticmethod
    def _find_result_pkl(d: Path, model: str) -> Path | None:
        suffixes = ("", ".gz", ".xz")
        stems = [
            d / f"result_{model}.pkl",
            d / model / "result.pkl",
            d / model / f"result_{model}.pkl",
        ]
        candidates: list[Path] = []
        for stem in stems:
            candidates.extend(stem.with_name(stem.name + suffix) for suffix in suffixes)
        candidates.extend(sorted(d.glob(f"result*{model}*.pkl*")))
        candidates.extend(sorted((d / model).glob("result*.pkl*")) if (d / model).is_dir() else [])

        seen: set[Path] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _read_pickle(path: Path):
        if path.suffix == ".gz":
            with gzip.open(path, "rb") as f:
                return pickle.load(f)
        if path.suffix == ".xz":
            with lzma.open(path, "rb") as f:
                return pickle.load(f)
        with path.open("rb") as f:
            return pickle.load(f)
