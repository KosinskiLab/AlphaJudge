from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Any, Tuple
import csv, numpy as np
from . import BaseParser, Run
from ..core import Confidence

class AF3Parser(BaseParser):
    name = "af3"

    def detect(self, d: Path) -> bool:
        return (d / "ranking_scores.csv").exists()

    def parse_run(self, d: Path) -> Run:
        order = self._read_csv_order(d / "ranking_scores.csv")

        def load_model(model: str):
            model_dir = d / model
            struct = self._load_structure(self._guess_struct(d, model))
            chains, rim, cid = self._maps(struct)

            summary = self._read_json(model_dir / "summary_confidences.json")
            matrix  = self._read_json(model_dir / "confidences.json") or summary

            iptm = self._safe_float(summary.get("iptm"))
            ptm  = self._safe_float(summary.get("ptm"))
            iptm_ptm = self._safe_float(summary.get("ranking_score")) or self._safe_float(summary.get("iptm+ptm"))
            conf = self._safe_float(summary.get("confidence_score"))
            if iptm_ptm is not None and iptm is not None and ptm is None:
                ptm = (iptm_ptm - 0.8 * iptm) / 0.2
            if conf is None and iptm is not None and ptm is not None:
                conf = 0.8 * iptm + 0.2 * ptm

            pae, max_pae = self._normalize_pae_af3(matrix, chains, cid)
            plddt = self._plddt(chains, rim)

            return struct, Confidence(
                pae_matrix=pae, max_pae=max_pae,
                iptm=iptm, ptm=ptm, iptm_ptm=iptm_ptm, confidence_score=conf,
                plddt_residue=plddt,
            )
        return Run(order=order, source="af3", load_model=load_model)

    # ---- AF3-specific helpers ----
    @staticmethod
    def _read_csv_order(p: Path) -> List[str]:
        with p.open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if r]
        def pf(x: Optional[str]) -> float:
            try: return float(x)  # type: ignore[arg-type]
            except Exception: return float("nan")
        rows.sort(key=lambda r: pf(r.get("ranking_score")), reverse=True)
        return [f"seed-{r['seed']}_sample-{r['sample']}" for r in rows if 'seed' in r and 'sample' in r]

    @staticmethod
    def _normalize_pae_af3(matrix: dict, chains, cid) -> Tuple[List[List[float]], float]:
        total = sum(len(cid[c.id]) for c in chains)
        pae = np.full((total, total), 100.0, dtype=float)
        max_pae = float('nan')

        if "predicted_aligned_error" in matrix:
            # some AF3 builds still store a full matrix in confidences.json
            m = np.array(matrix["predicted_aligned_error"], dtype=float)
            if m.size: pae[:, :] = m
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
                seen: List[Any] = []
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
            raise ValueError("unknown AF3 confidences schema")

        return pae.tolist(), float(max_pae)

