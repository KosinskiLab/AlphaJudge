"""AlphaJudge threshold calculator: pick prevalence + tolerable error + score +
AlphaFold version, get the score threshold.

This is a deterministic LOOKUP over the frozen benchmark ROC curves
(``threshold_rocs.json``, produced by ``scripts/freeze_threshold_rocs.py``), not
a model. For a chosen score and version we invert the stored ROC at a requested
false-positive rate (or precision / recall) and project precision to the user's
prevalence with the manuscript formula::

    precision(pi) = pi * TPR / [ pi * TPR + (1 - pi) * FPR ]

The thresholds are calibrated on the 4-organism database-supported benchmark
(positives vs database-negative re-pairings). They are approximately right for
other settings (the paper's generalisation result) but are NOT exact for other
interface classes such as antibody-antigen or peptide complexes.

Guards that the manuscript's caveats require, enforced here:
  * tail extrapolation: a requested ``max_fpr`` below the curve's
    ``fpr_resolution`` (~1/n_neg) returns the most stringent SUPPORTED point with
    ``resolved=False`` and a note, rather than a fabricated tail threshold;
  * uncertainty: every result carries Wilson 95% CIs on recall and precision;
  * prevalence is the user's input: ``precision_vs_prevalence`` exposes the full
    precision-vs-prevalence curve so the dominant role of prevalence is visible.

numpy + stdlib only. CLI: ``python -m alphajudge.thresholds --help``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

_DEFAULT_ARTIFACT = Path(__file__).with_name("threshold_rocs.json")

_VERSIONS = ("af2", "af3", "both")
# friendly aliases so a user can pass "ipSAE" instead of "interface_ipSAE"
_SCORE_ALIASES = {
    "ipsae": "interface_ipSAE",
    "lis": "interface_LIS",
    "pdockq2": "interface_pDockQ2",
    "iptm": "iptm",
    "sc": "interface_sc",
}

TRANSFERABILITY_NOTE = (
    "Thresholds are calibrated on the 4-organism database-supported benchmark; "
    "treat them as a starting point for other interface types."
)


@dataclass(frozen=True)
class OperatingPoint:
    """One resolved operating point. ``threshold`` is a single cutoff for the
    af2/af3 curves, or an ``(t_af2, t_af3)`` pair for the ``both`` AND-rule."""

    score: str
    version: str
    threshold: float | tuple[float, float]
    fpr: float
    recall: float
    precision: float
    precision_ci: tuple[float, float]
    recall_ci: tuple[float, float]
    prevalence: float
    resolved: bool
    note: str = ""

    def describe(self) -> str:
        thr = (
            f"af2>={self.threshold[0]:.3f} AND af3>={self.threshold[1]:.3f}"
            if isinstance(self.threshold, tuple)
            else f">={self.threshold:.3f}"
        )
        flag = "" if self.resolved else "  [UNRESOLVED: request below benchmark resolution]"
        line = (
            f"{self.score} ({self.version}) {thr}\n"
            f"  recall   = {self.recall:.2f} "
            f"[{self.recall_ci[0]:.2f}, {self.recall_ci[1]:.2f}]\n"
            f"  FPR      = {self.fpr:.4f}\n"
            f"  precision @ prevalence {self.prevalence:.4g} = {self.precision:.2f} "
            f"[{self.precision_ci[0]:.2f}, {self.precision_ci[1]:.2f}]{flag}"
        )
        if self.note:
            line += f"\n  note: {self.note}"
        return line


# --------------------------------------------------------------------------- IO


@lru_cache(maxsize=4)
def load_rocs(path: str | None = None) -> dict:
    """Load and cache the frozen ROC artifact (package default if ``path`` is None)."""
    p = Path(path) if path else _DEFAULT_ARTIFACT
    if not p.exists():
        raise FileNotFoundError(
            f"frozen ROC artifact not found: {p}\n"
            "Build it with: python scripts/freeze_threshold_rocs.py"
        )
    return json.loads(p.read_text())


def _resolve_score(score: str, rocs: dict) -> str:
    if score in rocs["curves"]:
        return score
    alias = _SCORE_ALIASES.get(score.lower())
    if alias and alias in rocs["curves"]:
        return alias
    raise KeyError(
        f"unknown score {score!r}; available: {sorted(rocs['curves'])} "
        f"(aliases: {sorted(_SCORE_ALIASES)})"
    )


def _curve(score: str, version: str, path: str | None) -> tuple[str, dict]:
    if version not in _VERSIONS:
        raise ValueError(f"version must be one of {_VERSIONS}, got {version!r}")
    rocs = load_rocs(path)
    score = _resolve_score(score, rocs)
    return score, rocs["curves"][score][version]


def _threshold_value(curve: dict, idx: int) -> float | tuple[float, float]:
    if "thresholds" in curve:
        return float(curve["thresholds"][idx])
    return float(curve["thresholds_af2"][idx]), float(curve["thresholds_af3"][idx])


# ----------------------------------------------------------------------- maths


def project_precision(prevalence: float, tpr: float, fpr: float) -> float:
    """precision(pi) = pi*TPR / (pi*TPR + (1-pi)*FPR). The Methods formula.

    Returns NaN only when both TPR and FPR are 0 (no calls at this threshold).
    """
    num = prevalence * tpr
    den = num + (1.0 - prevalence) * fpr
    if den <= 0.0:
        return float("nan")
    return float(num / den)


def _precision_ci(prevalence, tpr, fpr, tpr_hw, fpr_hw):
    """Propagate the Wilson half-widths on TPR/FPR through the precision formula.

    Precision rises with TPR and falls with FPR, so the bounds use the
    favourable / unfavourable corners.
    """
    hi = project_precision(prevalence, min(tpr + tpr_hw, 1.0), max(fpr - fpr_hw, 0.0))
    lo = project_precision(prevalence, max(tpr - tpr_hw, 0.0), min(fpr + fpr_hw, 1.0))
    return (lo, hi)


def _make_point(score, version, curve, idx, prevalence, resolved, note):
    fpr = float(curve["fpr"][idx])
    tpr = float(curve["tpr"][idx])
    fpr_hw = float(curve["fpr_ci_hw"][idx])
    tpr_hw = float(curve["tpr_ci_hw"][idx])
    prec = project_precision(prevalence, tpr, fpr)
    notes = [n for n in (note, TRANSFERABILITY_NOTE) if n]
    return OperatingPoint(
        score=score,
        version=version,
        threshold=_threshold_value(curve, idx),
        fpr=fpr,
        recall=tpr,
        precision=prec,
        precision_ci=_precision_ci(prevalence, tpr, fpr, tpr_hw, fpr_hw),
        recall_ci=(max(tpr - tpr_hw, 0.0), min(tpr + tpr_hw, 1.0)),
        prevalence=prevalence,
        resolved=resolved,
        note="; ".join(notes),
    )


# --------------------------------------------------------------- lookups (API)


def threshold_for_fpr(score, version, prevalence, max_fpr, path=None) -> OperatingPoint:
    """Most permissive threshold whose FPR <= ``max_fpr`` (i.e. highest recall
    within the tolerated false-positive rate); precision projected to ``prevalence``.

    If ``max_fpr`` is below the curve's ``fpr_resolution`` the request cannot be
    estimated from the benchmark negatives; we return the most stringent
    supported point with ``resolved=False``.
    """
    score, curve = _curve(score, version, path)
    fpr = np.asarray(curve["fpr"], float)
    res = curve.get("fpr_resolution")
    resolved = True
    note = ""
    if res is not None and 0 < max_fpr < res:
        resolved = False
        note = (
            f"requested FPR {max_fpr:.2g} is below the benchmark resolution "
            f"{res:.2g} (~1/n_neg); returning the most stringent supported point."
        )
        idx = int(np.argmin(fpr))  # smallest achievable FPR
        return _make_point(score, version, curve, idx, prevalence, resolved, note)
    # thresholds are sorted ascending => fpr is non-increasing; pick the most
    # permissive (lowest) threshold whose fpr still <= max_fpr
    ok = np.flatnonzero(fpr <= max_fpr)
    idx = int(ok.min()) if ok.size else int(np.argmin(fpr))
    if ok.size == 0:
        resolved = False
        note = "no threshold reaches the requested FPR; returning the lowest-FPR point."
    return _make_point(score, version, curve, idx, prevalence, resolved, note)


def threshold_for_precision(score, version, prevalence, min_precision, path=None) -> OperatingPoint:
    """Least stringent threshold reaching >= ``min_precision`` at ``prevalence``
    (maximising recall subject to the precision floor)."""
    score, curve = _curve(score, version, path)
    fpr = np.asarray(curve["fpr"], float)
    tpr = np.asarray(curve["tpr"], float)
    prec = np.array([project_precision(prevalence, t, f) for t, f in zip(tpr, fpr)])
    ok = np.flatnonzero(prec >= min_precision)
    if ok.size == 0:
        idx = int(np.nanargmax(prec))
        return _make_point(score, version, curve, idx, prevalence, False,
                           "precision floor unreachable at this prevalence; "
                           "returning the highest-precision point.")
    # among points meeting the floor, take the one with greatest recall
    idx = int(ok[np.argmax(tpr[ok])])
    return _make_point(score, version, curve, idx, prevalence, True, "")


def threshold_for_recall(score, version, prevalence, min_recall, path=None) -> OperatingPoint:
    """Most precise threshold (lowest FPR) still retaining >= ``min_recall``."""
    score, curve = _curve(score, version, path)
    fpr = np.asarray(curve["fpr"], float)
    tpr = np.asarray(curve["tpr"], float)
    ok = np.flatnonzero(tpr >= min_recall)
    if ok.size == 0:
        idx = int(np.nanargmax(tpr))
        return _make_point(score, version, curve, idx, prevalence, False,
                           "recall floor unreachable; returning the highest-recall point.")
    idx = int(ok[np.argmin(fpr[ok])])
    return _make_point(score, version, curve, idx, prevalence, True, "")


def precision_vs_prevalence(score, version, threshold, prevalences, path=None):
    """(prevalence, precision) at a FIXED threshold across a list of prevalences.

    Backs the "enter a range of prevalence" view. ``threshold`` is matched to the
    nearest grid point of the chosen curve; for ``both`` pass the af2 cutoff and
    the matching af3 cutoff is taken from the same grid index.
    """
    score, curve = _curve(score, version, path)
    if "thresholds" in curve:
        grid = np.asarray(curve["thresholds"], float)
        idx = int(np.argmin(np.abs(grid - threshold)))
    else:
        grid = np.asarray(curve["thresholds_af2"], float)
        idx = int(np.argmin(np.abs(grid - threshold)))
    tpr = float(curve["tpr"][idx])
    fpr = float(curve["fpr"][idx])
    return [(float(pi), project_precision(pi, tpr, fpr)) for pi in prevalences]


# ------------------------------------------------------------------------- CLI


def _build_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="alphajudge.thresholds",
        description="AlphaJudge threshold calculator (lookup over benchmark ROCs).",
    )
    p.add_argument("--score", default="ipSAE",
                   help="score name or alias (ipSAE, LIS, pDockQ2, iptm, sc)")
    p.add_argument("--version", choices=_VERSIONS, default="af2")
    p.add_argument("--prevalence", type=float, default=0.05,
                   help="expected true-partner fraction in your screen")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--max-fpr", type=float, help="tolerable false-positive rate")
    g.add_argument("--min-precision", type=float, help="required precision at --prevalence")
    g.add_argument("--min-recall", type=float, help="required recall (sensitivity)")
    p.add_argument("--artifact", default=None, help="path to threshold_rocs.json")
    p.add_argument("--sweep-prevalence", nargs="*", type=float, default=None,
                   help="also print precision at these prevalences for the chosen threshold")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.max_fpr is not None:
        op = threshold_for_fpr(args.score, args.version, args.prevalence, args.max_fpr, args.artifact)
    elif args.min_precision is not None:
        op = threshold_for_precision(args.score, args.version, args.prevalence, args.min_precision, args.artifact)
    else:
        op = threshold_for_recall(args.score, args.version, args.prevalence, args.min_recall, args.artifact)
    print(op.describe())
    if args.sweep_prevalence:
        thr = op.threshold[0] if isinstance(op.threshold, tuple) else op.threshold
        print("\nprecision vs prevalence at this threshold:")
        for pi, prec in precision_vs_prevalence(args.score, args.version, thr, args.sweep_prevalence, args.artifact):
            print(f"  prevalence {pi:<8.4g} -> precision {prec:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
