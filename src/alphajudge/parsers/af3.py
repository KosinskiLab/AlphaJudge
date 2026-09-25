from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv
import logging
import numpy as np
from . import BaseParser, Run
from ..confidence import Confidence
from ..geometry import is_pae_token_residue

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ResidueTokens:
    """Where each scored residue sits in AF3's token-by-token matrices."""

    residue_count: int
    # Size the token matrices must have; None when token metadata is unusable.
    token_count: int | None
    # Token of each scored residue in cid order; None when unmappable.
    index: np.ndarray | None

    @property
    def excludes_tokens(self) -> bool:
        """Whether the token matrices also carry tokens that are not scored."""
        return self.token_count is not None and self.token_count > self.residue_count

    def select(self, token_matrix: np.ndarray) -> np.ndarray | None:
        """The residue-by-residue block of a token matrix, or None."""
        n = self.token_count
        if self.index is None or token_matrix.shape != (n, n):
            return None
        return token_matrix[np.ix_(self.index, self.index)]


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

            token_chain_ids = matrix.get("token_chain_ids")
            summary_chain_ids = (
                list(dict.fromkeys(str(c) for c in token_chain_ids))
                if isinstance(token_chain_ids, (list, tuple))
                else [str(c.id) for c in chains]
            )
            if chain_pair_iptm is not None and (
                len(chain_pair_iptm) != len(summary_chain_ids)
                or any(len(row) != len(summary_chain_ids) for row in chain_pair_iptm)
            ):
                logger.warning("chain_pair_iptm dimensions do not match the source chains; skipping pair ipTM.")
                chain_pair_iptm = None

            pae, max_pae, residue_tokens = self._normalize_pae_af3(matrix, chains, cid)
            contact_probs = self._normalize_contact_probs_af3(matrix, residue_tokens)
            plddt = self._plddt(chains, rim)
            global_scope = (
                "includes_excluded_tokens" if residue_tokens.excludes_tokens else "scored_residues"
            )

            return struct, Confidence(
                pae_matrix=pae, max_pae=max_pae,
                iptm=iptm, ptm=ptm, iptm_ptm=iptm_ptm, confidence_score=ranking_score,
                plddt_residue=plddt, chain_pair_iptm=chain_pair_iptm,
                contact_prob_matrix=contact_probs,
                contact_prob_source="af3_contact_probs" if contact_probs is not None else None,
                chain_pair_iptm_chain_ids=summary_chain_ids,
                global_confidence_scope=global_scope,
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

        # Exact-name candidates, each also tried with .xz/.gz so that
        # AlphaPulldown's slim/minimal storage modes (which compress per-sample
        # confidences.json) are read transparently.
        exact = [model_dir / f"{kind}.json"]
        if job_prefix:
            exact.append(model_dir / f"{job_prefix}_{model}_{kind}.json")
            if is_best_model:
                exact.append(d / f"{job_prefix}_{kind}.json")
        if is_best_model:
            exact.append(d / f"ranked_0_{kind}.json")

        candidates: list[Path] = []
        for base in exact:
            candidates.append(base)
            candidates.append(base.with_name(base.name + ".xz"))
            candidates.append(base.with_name(base.name + ".gz"))

        # Wildcard fallback within the model dir. Exclude summary_confidences.json
        # from a "confidences" search: it ends in "_confidences.json" but only
        # carries coarse per-chain-pair PAE, not the full token x token matrix.
        # Match the summary file by suffix so both "summary_confidences.json" and
        # the job-prefixed "<job>_summary_confidences.json" (official AF3 layout)
        # are excluded.
        def _is_summary(name: str) -> bool:
            base = name
            for ext in (".xz", ".gz"):
                if base.endswith(ext):
                    base = base[: -len(ext)]
            return base == "summary_confidences.json" or base.endswith("_summary_confidences.json")

        if model_dir.is_dir():
            for pattern in (f"*_{kind}.json", f"*_{kind}.json.xz", f"*_{kind}.json.gz"):
                for hit in sorted(model_dir.glob(pattern)):
                    if kind == "confidences" and _is_summary(hit.name):
                        continue
                    candidates.append(hit)

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
    def _normalize_pae_af3(matrix: dict, chains, cid) -> tuple[np.ndarray, float, _ResidueTokens]:
        """Residue-level PAE, its maximum, and the residue-to-token map used.

        The map is returned so the other token matrices reuse it.
        """
        raw = matrix.get("predicted_aligned_error", matrix.get("pae"))
        if raw is None:
            if "chain_pair_pae_min" in matrix:
                raise ValueError(
                    "AF3 confidences contain only chain-pair minima, not residue-level PAE; "
                    "provide the full confidences.json to compute interface scores."
                )
            raise ValueError(
                "unknown AF3 confidences schema: expected residue-level PAE "
                "in predicted_aligned_error or pae"
            )

        tokens = np.asarray(raw, dtype=float)
        if tokens.ndim != 2 or tokens.shape[0] != tokens.shape[1] or not tokens.size:
            raise ValueError(f"AF3 PAE must be a nonempty square matrix, got {tokens.shape}.")
        residue_tokens = AF3Parser._residue_tokens_af3(matrix, chains, cid)
        total = residue_tokens.residue_count
        expected_shape = (total, total)
        pae = residue_tokens.select(tokens)
        if pae is None:
            raise ValueError(
                f"Cannot align AF3 PAE shape {tokens.shape} to {expected_shape}: "
                "token chain/residue identifiers must map each scored residue unambiguously. "
                "Chain-pair minima cannot substitute for residue-level PAE."
            )
        if not np.all(np.isfinite(pae)) or np.any(pae < 0):
            raise ValueError("Aligned AF3 PAE contains non-finite or negative values.")
        if tokens.shape != expected_shape:
            logger.info(
                "Aligned %d AF3 PAE tokens to %d scored residues; excluded %d tokens.",
                len(tokens), total, len(tokens) - total,
            )
        max_pae = float(matrix.get("max_predicted_aligned_error", np.nan))
        if not np.isfinite(max_pae):
            max_pae = float(np.max(pae))
        return pae, float(max_pae), residue_tokens

    @staticmethod
    def _normalize_contact_probs_af3(matrix: dict, residue_tokens: _ResidueTokens) -> np.ndarray | None:
        raw = matrix.get("contact_probs")
        if raw is None:
            return None

        expected_shape = (residue_tokens.residue_count,) * 2
        contact_probs = np.asarray(raw, dtype=float)
        if contact_probs.ndim != 2:
            logger.warning(
                f"contact_probs must be a 2D matrix, got shape {contact_probs.shape}; "
                "skipping contact probabilities."
            )
            return None

        aligned = residue_tokens.select(contact_probs)
        if aligned is None:
            logger.warning(
                f"Cannot align contact_probs shape {contact_probs.shape} to {expected_shape}; "
                "skipping contact probabilities."
            )
        return aligned

    @staticmethod
    def _residue_tokens_af3(matrix: dict, chains, cid) -> _ResidueTokens:
        """Find the token of each scored residue in AF3's token matrices.

        Supplied residue identifiers are authoritative: a failed match never
        falls back to positional assignment. Without them each retained chain
        must match by chain ID with one token per residue, and without any
        token metadata the matrices must already be in residue order.
        """
        total = sum(len(cid[c.id]) for c in chains)
        if "token_chain_ids" not in matrix:
            # Legacy residue matrices without token metadata are already ordered.
            return _ResidueTokens(total, total, np.arange(total))
        token_chain_ids = matrix["token_chain_ids"]
        if not isinstance(token_chain_ids, (list, tuple)):
            return _ResidueTokens(total, None, None)
        ids = [str(x) for x in token_chain_ids]
        unmappable = _ResidueTokens(total, len(ids), None)
        token_res_ids = matrix.get("token_res_ids", matrix.get("token_residue_ids"))
        if token_res_ids is not None and (
            not isinstance(token_res_ids, (list, tuple)) or len(token_res_ids) != len(ids)
        ):
            return unmappable

        index = np.full(total, -1, dtype=int)
        if token_res_ids is None:
            for chain in chains:
                residues = cid.get(chain.id, [])
                if not residues:
                    continue
                tokens = [k for k, chain_id in enumerate(ids) if chain_id == str(chain.id)]
                if len(tokens) != len(residues):
                    return unmappable
                index[residues] = tokens
        else:
            # Several tokens share an ID on ligands (discarded) and on modified
            # residues, which AF3 tokenizes per atom (resolved by _residue_token).
            tokens_by_residue: dict[tuple[str, int], list[int]] = {}
            for token, (chain_id, raw_res_id) in enumerate(zip(ids, token_res_ids)):
                try:
                    res_id = int(raw_res_id)
                except (TypeError, ValueError):
                    continue
                tokens_by_residue.setdefault((chain_id, res_id), []).append(token)
            for chain in chains:
                residues = cid.get(chain.id, [])
                if not residues:
                    continue
                kept = [res for res in chain if is_pae_token_residue(res)]
                if len(kept) != len(residues):
                    return unmappable
                tokens = [
                    AF3Parser._residue_token(
                        res, tokens_by_residue.get((str(chain.id), int(res.id[1])), [])
                    )
                    for res in kept
                ]
                if None in tokens or len(set(tokens)) != len(tokens):
                    return unmappable
                index[residues] = tokens
        if (index < 0).any():
            return unmappable
        return _ResidueTokens(total, len(ids), index)

    @staticmethod
    def _residue_token(residue, token_indices: list[int]) -> int | None:
        """The token carrying a scored residue's PAE row, or None if ambiguous.

        AF3 gives a standard residue one token centred on CA (protein) or C1'
        (nucleic acid), but a modified residue one token per atom, in the
        residue's atom order. Take the per-atom token of that same centre atom,
        and only when the tokens and the residue's atoms correspond one to one.
        """
        if len(token_indices) == 1:
            return token_indices[0]
        atom_names = [atom.get_id() for atom in residue]
        if len(token_indices) != len(atom_names):
            return None
        for centre in ("CA", "C1'"):
            if centre in atom_names:
                return token_indices[atom_names.index(centre)]
        return None
