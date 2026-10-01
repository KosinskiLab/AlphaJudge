from __future__ import annotations
import logging
import pickle
import re
from pathlib import Path
import numpy as np
from . import BaseParser, Run
from ..confidence import Confidence, validate_pae
from ..contact_probs import (
    AF2_DISTOGRAM_CONTACT_CUTOFF,
    contact_probs_from_distogram,
)

logger = logging.getLogger(__name__)

class AF2Parser(BaseParser):
    name = "af2"
    _warned_missing_distogram = False

    def detect(self, d: Path) -> bool:
        return any((d / f"ranking_debug.json{suffix}").exists() for suffix in ("", ".xz", ".gz"))

    def parse_run(self, d: Path) -> Run:
        rj = self._read_json(d / "ranking_debug.json", required=True)
        order = rj["order"]
        structure_files = {}

        def load_model(model: str):
            structure_path = Path(self._guess_struct(d, model, order.index(model)))
            struct = self._load_structure(structure_path)
            structure_files[model] = str(structure_path.relative_to(d))
            chains, rim, _ = self._maps(struct)

            result_path, result = self._load_result(d, model)
            pae_path = d / f"pae_{model}.json"
            json_path = self._first_existing([pae_path.with_name(pae_path.name + s) for s in ("", ".xz", ".gz")])
            if json_path is not None:
                payload = self._read_json(json_path, required=True)
                try:
                    raw_pae = (payload[0] if isinstance(payload, list) else payload)["predicted_aligned_error"]
                except (KeyError, IndexError, TypeError) as exc:
                    raise ValueError(f"{json_path}: missing predicted_aligned_error matrix") from exc
                pae_source = str(json_path)
            elif "predicted_aligned_error" in result:
                raw_pae = result["predicted_aligned_error"]
                pae_source = str(result_path)
            else:
                raise ValueError(f"No full PAE matrix for {model}: expected {pae_path} or predicted_aligned_error in result pickle")
            try:
                pae = np.asarray(raw_pae, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{pae_source}: invalid PAE matrix: {exc}") from exc
            validate_pae(pae, len(rim), source=pae_source)
            max_pae = float(np.max(pae)) if pae.size else float("nan")
            contact_probs = self._contact_probs_from_result(result, result_path, pae.shape)

            # AF2 rankings
            is_multimer = "iptm+ptm" in rj or "iptm" in rj or "iptm" in result
            def score(name):
                value = self._safe_float(rj.get(name, {}).get(model))
                return value if value is not None else self._safe_float(result.get(name))

            if is_multimer:
                iptm, ptm, iptm_ptm = score("iptm"), score("ptm"), score("iptm+ptm")
                # Backfill when PTM is not provided in AF2 multimer JSON
                if ptm is None and (iptm_ptm is not None) and (iptm is not None):
                    ptm = (iptm_ptm - 0.8 * iptm) / 0.2
                # If iptm+ptm itself is missing but both iptm and ptm exist, derive it
                if iptm_ptm is None and (iptm is not None) and (ptm is not None):
                    iptm_ptm = 0.8 * iptm + 0.2 * ptm
                conf = iptm_ptm
            else:
                iptm, ptm = 0.0, score("ptm")
                iptm_ptm = conf = ptm

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
        return Run(order=order, source="af2", load_model=load_model, structure_files=structure_files)

    @classmethod
    def _load_contact_probs_from_result_pkl(
        cls, d: Path, model: str, expected_shape: tuple[int, int]
    ) -> np.ndarray | None:
        result_pkl, payload = cls._load_result(d, model)
        return cls._contact_probs_from_result(payload, result_pkl, expected_shape)

    @classmethod
    def _load_result(cls, d: Path, model: str) -> tuple[Path | None, dict]:
        result_pkl = cls._find_result_pkl(d, model)
        if result_pkl is None:
            return None, {}

        try:
            with cls._open_maybe_compressed(result_pkl, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:
            logger.warning(f"could not read AF2 result pickle {result_pkl}: {e}")
            return result_pkl, {}

        if not isinstance(payload, dict):
            logger.warning("AF2 result pickle %s does not contain a mapping", result_pkl)
            return result_pkl, {}
        return result_pkl, payload

    @classmethod
    def _contact_probs_from_result(
        cls, payload: dict, result_pkl: Path | None, expected_shape: tuple[int, int]
    ) -> np.ndarray | None:
        if result_pkl is None or not payload:
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

    @classmethod
    def _find_result_pkl(cls, d: Path, model: str) -> Path | None:
        stems = [
            d / f"result_{model}.pkl",
            d / model / "result.pkl",
            d / model / f"result_{model}.pkl",
        ]
        candidates = [
            stem.with_name(stem.name + suffix) for stem in stems for suffix in ("", ".gz", ".xz")
        ]
        pattern = re.compile(r"(?:^|[_\-.])" + re.escape(model) + r"(?:$|[_\-.])")
        candidates.extend(p for p in sorted(d.glob("result*.pkl*")) if pattern.search(p.name))
        if (d / model).is_dir():
            candidates.extend(sorted((d / model).glob("result*.pkl*")))
        return cls._first_existing(candidates)
