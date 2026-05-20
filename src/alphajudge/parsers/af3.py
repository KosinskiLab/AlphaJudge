from __future__ import annotations
from pathlib import Path
from typing import Any
import csv
import logging
import numpy as np
from . import BaseParser, Run
from ..confidence import Confidence

logger = logging.getLogger(__name__)

class AF3Parser(BaseParser):
    name = "af3"

    @staticmethod
    def detect(d: Path) -> bool:
        return AF3Parser._ranking_scores_file(d) is not None

    def parse_run(self, d: Path) -> Run:
        ranking_file = self._ranking_scores_file(d)
        if ranking_file is None:
            raise ValueError(f"AF3 ranking scores file not found in {d}")
        order, ranking_scores = self._read_csv_order(ranking_file)
        job_prefix = self._job_prefix_from_ranking_file(ranking_file)

        def load_model(model: str):
            model_dir = d / model
            is_best_model = bool(order and model == order[0])
            struct = self._load_structure(
                self._guess_af3_struct(d, model, job_prefix, is_best_model)
            )
            chains, rim, cid = self._maps(struct)

            summary = self._read_json(
                self._find_af3_json(
                    d, model, "summary_confidences", job_prefix, is_best_model
                )
            )
            matrix = self._read_json(
                self._find_af3_json(d, model, "confidences", job_prefix, is_best_model)
            ) or summary

            iptm = self._safe_float(summary.get("iptm"))
            ptm  = self._safe_float(summary.get("ptm"))
            ranking_score = self._safe_float(summary.get("ranking_score"))
            if ranking_score is None:
                ranking_score = ranking_scores.get(model)
            iptm_ptm = 0.2 * ptm + 0.8 * iptm if (iptm is not None and ptm is not None) else None

            chain_pair_iptm_raw = summary.get("chain_pair_iptm")
            chain_pair_iptm = None
            if isinstance(chain_pair_iptm_raw, (list, tuple)):
                chain_pair_iptm = [list(r) if isinstance(r, (list, tuple)) else [] for r in chain_pair_iptm_raw]

            pae, max_pae = self._normalize_pae_af3(matrix, chains, cid)
            plddt = self._plddt(chains, rim)

            return struct, Confidence(
                pae_matrix=pae, max_pae=max_pae,
                iptm=iptm, ptm=ptm, iptm_ptm=iptm_ptm, confidence_score=ranking_score,
                plddt_residue=plddt, chain_pair_iptm=chain_pair_iptm,
            )
        return Run(order=order, source="af3", load_model=load_model)

    # ---- AF3-specific helpers ----
    @staticmethod
    def _ranking_scores_file(d: Path) -> Path | None:
        plain = d / "ranking_scores.csv"
        if plain.exists():
            return plain
        hits = sorted(d.glob("*_ranking_scores.csv"))
        return hits[0] if hits else None

    @staticmethod
    def _job_prefix_from_ranking_file(p: Path) -> str | None:
        if p.name == "ranking_scores.csv":
            return None
        suffix = "_ranking_scores"
        if p.stem.endswith(suffix):
            return p.stem[: -len(suffix)]
        return None

    @staticmethod
    def _read_csv_order(p: Path) -> tuple[list[str], dict[str, float]]:
        with p.open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if r]
        def pf(x: str | None) -> float:
            try: return float(x)  # type: ignore[arg-type]
            except Exception: return float("nan")
        rows.sort(key=lambda r: pf(r.get("ranking_score")), reverse=True)
        order: list[str] = []
        scores: dict[str, float] = {}
        for r in rows:
            if "seed" not in r or "sample" not in r:
                continue
            model = f"seed-{r['seed']}_sample-{r['sample']}"
            order.append(model)
            score = pf(r.get("ranking_score"))
            if np.isfinite(score):
                scores[model] = score
        return order, scores

    @staticmethod
    def _find_existing(paths: list[Path]) -> Path | None:
        for p in paths:
            if p.exists():
                return p
        return None

    @classmethod
    def _find_af3_json(
        cls,
        d: Path,
        model: str,
        kind: str,
        job_prefix: str | None,
        is_best_model: bool,
    ) -> Path:
        model_dir = d / model
        candidates = [model_dir / f"{kind}.json"]
        if job_prefix:
            candidates.append(model_dir / f"{job_prefix}_{model}_{kind}.json")
            if is_best_model:
                candidates.append(d / f"{job_prefix}_{kind}.json")
        if model_dir.is_dir():
            candidates.extend(sorted(model_dir.glob(f"*_{kind}.json")))
        if is_best_model:
            candidates.append(d / f"ranked_0_{kind}.json")
        return cls._find_existing(candidates) or candidates[0]

    @classmethod
    def _guess_af3_struct(
        cls,
        d: Path,
        model: str,
        job_prefix: str | None,
        is_best_model: bool,
    ) -> str:
        model_dir = d / model
        candidates: list[Path] = []
        for ext in ("cif", "pdb"):
            candidates.append(model_dir / f"model.{ext}")
            if job_prefix:
                candidates.append(model_dir / f"{job_prefix}_{model}_model.{ext}")
        if model_dir.is_dir():
            for ext in ("cif", "pdb"):
                candidates.extend(sorted(model_dir.glob(f"*{model}*_model.{ext}")))
                candidates.extend(sorted(model_dir.glob(f"*.{ext}")))
        if job_prefix and is_best_model:
            for ext in ("cif", "pdb"):
                candidates.append(d / f"{job_prefix}_model.{ext}")
        for ext in ("cif", "pdb"):
            candidates.extend(sorted(d.glob(f"*{model}*.{ext}")))
        found = cls._find_existing(candidates)
        if found is not None:
            return str(found)
        raise ValueError(f"struct for model {model} not found")

    @staticmethod
    def _normalize_pae_af3(matrix: dict, chains, cid) -> tuple[np.ndarray, float]:
        total = sum(len(cid[c.id]) for c in chains)
        pae = np.full((total, total), 100.0, dtype=float)
        max_pae = float('nan')

        if "predicted_aligned_error" in matrix:
            # some AF3 builds still store a full matrix in confidences.json
            m = np.array(matrix["predicted_aligned_error"], dtype=float)
            if m.size:
                if m.shape == pae.shape:
                    pae[:, :] = m
                else:
                    logger.warning(
                        f"predicted_aligned_error shape {m.shape} != expected {pae.shape}; "
                        "skipping PAE assignment."
                    )
            max_pae = float(matrix.get("max_predicted_aligned_error", np.nan))
            if not np.isfinite(max_pae):
                max_pae = float(np.nanmax(m)) if m.size else float('nan')

        elif "pae" in matrix and "token_chain_ids" in matrix:
            # Prefer to use the full token×token PAE matrix directly when it
            # matches the residue×residue layout, to retain per-residue detail.
            tokens = np.array(matrix["pae"], dtype=float)
            max_pae = float(np.nanmax(tokens)) if tokens.size else float('nan')

            if tokens.shape == pae.shape:
                # 1:1 correspondence between tokens and residues; assume that
                # token order matches the global residue order used for cid.
                pae[:, :] = tokens
            else:
                # Fallback to coarse chain-pair mapping using token_chain_ids.
                ids = matrix["token_chain_ids"]
                # map token groups → chain indices
                seen: list[Any] = []
                for c in ids:
                    if c not in seen:
                        seen.append(c)
                group = {v: i for i, v in enumerate(seen)}
                for i, chi in enumerate(chains):
                    ti = [k for k, c in enumerate(ids) if group.get(c, -1) == i]; ri = cid.get(chi.id, [])
                    for j, chj in enumerate(chains):
                        tj = [k for k, c in enumerate(ids) if group.get(c, -1) == j]; rj = cid.get(chj.id, [])
                        if ti and tj and ri and rj:
                            block = tokens[np.ix_(ti, tj)]
                            val = float(np.nanmin(block)) if block.size else 100.0
                            pae[np.ix_(ri, rj)] = val

        elif "chain_pair_pae_min" in matrix:
            cp = np.array(matrix["chain_pair_pae_min"], dtype=float)
            max_pae = float(np.nanmax(cp)) if cp.size else float('nan')
            for i, chi in enumerate(chains):
                ri = cid.get(chi.id, [])
                for j, chj in enumerate(chains):
                    rj = cid.get(chj.id, [])
                    try: val = float(cp[i][j])
                    except Exception: val = None
                    if ri and rj:
                        pae[np.ix_(ri, rj)] = 100.0 if val is None else val
        else:
            logger.warning(
                "No recognised PAE keys in confidences.json/summary_confidences.json; "
                "using default PAE=100 for all residue pairs."
            )

        return pae, float(max_pae)
