from __future__ import annotations
from pathlib import Path
import numpy as np
from . import BaseParser, Run
from ..core import Confidence

class AF2Parser(BaseParser):
    name = "af2"

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
                pae_matrix=pae.tolist(), max_pae=max_pae,
                iptm=iptm, ptm=ptm, iptm_ptm=iptm_ptm, confidence_score=conf,
                plddt_residue=plddt,
            )
        return Run(order=order, source="af2", load_model=load_model)

