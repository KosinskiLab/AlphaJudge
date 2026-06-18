# AlphaJudge threshold calculator — design (paper only, no implementation)

**Status:** design sketch, not built. Created 2026-06-18.
**Author of request:** DM ("a calculator where you choose prevalence, tolerable
false rate, score, and AF2/AF3/both, and it gives you the score threshold").

## 1. What it is

A deterministic **lookup over the benchmark ROC curves**, not a model. The user
supplies an operating point; the tool inverts the stored empirical ROC for the
chosen score/version, then projects precision to the user's prevalence with the
same formula already in the manuscript Methods:

```
precision(pi) = pi * TPR / [ pi * TPR + (1 - pi) * FPR ]
```

No training, no raw data at runtime — it reads one frozen artifact (Section 3),
exactly like `BENCHMARK_QUANTILES` in `meta_score.py` is a frozen artifact read
by the report sliders. This makes the calculator the operational embodiment of
the paper's recommendation ("do not transfer thresholds; calibrate per method,
version and prevalence").

## 2. Inputs and outputs

### Inputs
| input        | type                              | notes |
|--------------|-----------------------------------|-------|
| `score`      | `"interface_ipSAE" \| "interface_LIS" \| ...` | any score with a frozen ROC |
| `version`    | `"af2" \| "af3" \| "both"`        | `both` = AF2-and-AF3 agreement rule (Section 5) |
| `prevalence` | float in (0, 1)                   | user's expected true-partner fraction; may be a list/range |
| **one** target, choose exactly one: | | the operating constraint |
| `max_fpr`    | float in (0, 1)                   | "false positives I tolerate" → solve for threshold at this FPR |
| `min_precision` | float in (0, 1)                | "precision I need at my prevalence" → solve for threshold |
| `min_recall` | float in (0, 1)                   | "sensitivity I need" → solve for threshold |

### Output (a small dataclass / dict)
| field        | meaning |
|--------------|---------|
| `threshold`  | score cutoff to apply (on the chosen score's native scale) |
| `fpr`        | false-positive rate at that cutoff (prevalence-independent) |
| `recall`     | = TPR at that cutoff (prevalence-independent) |
| `precision`  | precision **projected to the supplied prevalence** |
| `precision_ci` | Wilson/bootstrap 95% CI on precision |
| `recall_ci`  | bootstrap 95% CI on recall |
| `resolved`   | bool — False if the requested FPR is below the benchmark's resolution (Section 6); then `threshold` is the most stringent supported point and a warning is set |
| `note`       | human-readable caveat string (empty if clean) |

Returning **CIs and a `resolved` flag is mandatory, not optional** — see the
caveats in Section 6. A bare point threshold would invite the exact misuse the
paper warns against.

## 3. Frozen artifact schema (`threshold_rocs.json`)

One file, produced once by a freeze script (Section 4), shipped in the package
next to the code (mirrors how `BENCHMARK_QUANTILES` lives in `meta_score.py`;
JSON is preferred here because the curves are large). Calibrated on **positives
vs database-negatives** of the main 4-organism benchmark — the same population
the manuscript ROC/AUROC numbers come from.

```jsonc
{
  "schema_version": 1,
  "source_csv": "benchmark_best.balanced.csv",          // provenance
  "source_sha16": "….",                                  // reproducibility, matches freeze convention
  "n_pos": 1937,
  "n_neg": 1937,
  "orientation": "higher_is_stronger",                   // values pre-oriented via FEATURE_DIRECTIONS
  "curves": {
    "interface_ipSAE": {
      "af2": {
        "auroc": 0.880,
        "n_pos": 1937, "n_neg": 1937,
        // The ROC as a monotone table sampled densely enough to interpolate.
        // Each row is one threshold's operating point. Stored sorted by threshold.
        "thresholds": [0.00, 0.01, …, 1.00],
        "fpr":        [1.00, 0.97, …, 0.00],
        "tpr":        [1.00, 0.99, …, 0.00],
        // pre-computed Wilson half-widths so runtime needs no benchmark rows
        "tpr_ci_lo":  [...], "tpr_ci_hi":  [...],
        "fpr_ci_lo":  [...], "fpr_ci_hi":  [...],
        // smallest FPR distinguishable from 0 given n_neg (Section 6)
        "fpr_resolution": 0.00052
      },
      "af3": { … },
      "both": { … }                                       // joint-rule curve, Section 5
    },
    "interface_LIS": { "af2": {…}, "af3": {…}, "both": {…} }
    // … one block per supported score
  }
}
```

Notes on the schema:
- **Pre-oriented.** Values already follow `FEATURE_DIRECTIONS` (PAE, solvation
  energy sign-flipped), so the runtime never re-orients — same rule the rest of
  the package uses.
- **Dense threshold grid, not raw points.** Storing a fixed grid (e.g. 1,001
  thresholds 0..1 for bounded scores, or quantile-spaced for unbounded ones)
  keeps lookup to a vectorised `searchsorted`/`interp` and the file small and
  stable across rebuilds.
- **CIs baked in.** Wilson half-widths for TPR/FPR are computed at freeze time
  from `n_pos`/`n_neg`, so the runtime carries no benchmark data and stays
  numpy-only.
- **`fpr_resolution`** = 1 / n_neg-ish (the FPR step of one negative); requests
  below it are flagged `resolved=False`.
- **Per-organism is out of scope for v1.** The frozen curves are pooled (or
  macro, matching the paper's headline). A `by_organism` sub-key is a clean v2
  extension if anyone wants organism-specific thresholds.

## 4. Freeze script — `scripts/freeze_threshold_rocs.py`

Mirrors `freeze_metascore_quantiles.py` exactly in spirit:
- reads the benchmark CSV (default = the synced `benchmark_best.balanced.csv`),
- for each `score x version`: orient via `bench_common.oriented_values`, compute
  the ROC (`bench_common` already has the AUROC machinery; the curve is the same
  sweep), sample onto the fixed threshold grid, attach Wilson CIs,
- writes `src/alphajudge/threshold_rocs.json` and prints a one-line summary
  (AUROCs per score/version) for the changelog.
- numpy + stdlib only; no pandas, no sklearn → runs in a stock install.
- records `source_sha16` so a rebuild is verifiably the same data.

CLI shape (consistent with the other `freeze_*`/`evaluate_*` scripts):
```
python scripts/freeze_threshold_rocs.py \
    --input-csv .../benchmark_best.balanced.csv \
    --scores interface_ipSAE interface_LIS \
    --out src/alphajudge/threshold_rocs.json
```

## 5. The `both` (AF2-and-AF3 agreement) curve

`af2` and `af3` are each that version's own ROC — trivial. `both` is the only
entry needing a modelling decision, because "confident on both" is a *joint*
rule over the paired predictions (we have AF2 and AF3 scores for every pair).

**v1 choice — single shared operating quantile.** Sweep one parameter `q`; at
each `q` set each version's threshold to its own `q`-quantile of that version's
score, label a pair positive iff *both* versions clear their own threshold, and
record the joint (FPR, TPR). This yields a proper monotone ROC for the AND-rule
expressed as a single knob, and the manuscript's `n = 40 / 93%` point is one
sample on it. The returned `threshold` for `version="both"` is then a **pair**
`(t_af2, t_af3)` (schema: `both.thresholds` holds 2-tuples), and the `note`
explains that both must be met.

Rejected alternatives (documented so the choice is auditable): independent
2-D threshold grid (richer but no longer a 1-knob "calculator", and the joint
cell counts get sparse — exactly the n=40 small-count problem, amplified);
logistic meta-model over the two scores (that is a *model*, not a lookup, and
changes the tool's character and validation burden).

## 6. Caveats the tool must enforce (not optional)

These are the failure modes a naive calculator invites; the design above
neutralises each:

1. **Tail extrapolation.** A very small `max_fpr` reads the far tail of the ROC,
   estimated from a handful of negatives. → `fpr_resolution` gate sets
   `resolved=False` and returns the most stringent *supported* point plus a
   note; CIs widen honestly there rather than hiding the uncertainty.
2. **Prevalence is the user's guess.** Precision is only as good as `pi`. →
   accept a **range** of `pi` and return the precision curve over it, making the
   paper's central "precision is prevalence-dominated" point visible instead of
   giving one false-precise number.
3. **Transferability.** Thresholds are calibrated on the 4-organism physical-
   intersection benchmark. Approximately right elsewhere (that is the paper's
   generalisation result) but not exact for other interface types
   (antibody-antigen, peptide). → fixed `note` on every result; stated in docs.

## 7. Public API (signatures only — no bodies)

Lives in a new module `src/alphajudge/thresholds.py`, loaded lazily like the
report calibration.

```python
# src/alphajudge/thresholds.py

from dataclasses import dataclass

@dataclass(frozen=True)
class OperatingPoint:
    threshold: float | tuple[float, float]   # tuple when version == "both"
    fpr: float
    recall: float
    precision: float
    precision_ci: tuple[float, float]
    recall_ci: tuple[float, float]
    resolved: bool
    note: str = ""


def load_rocs(path: str | None = None) -> dict:
    """Load the frozen threshold_rocs.json (package default if path is None).
    Cached on first call (module-level), mirroring meta_score's frozen dicts."""


def project_precision(prevalence: float, tpr: float, fpr: float) -> float:
    """precision(pi) = pi*TPR / (pi*TPR + (1-pi)*FPR). The exact Methods formula."""


def threshold_for_fpr(
    score: str,
    version: str,                 # "af2" | "af3" | "both"
    prevalence: float,
    max_fpr: float,
) -> OperatingPoint:
    """Most permissive threshold whose FPR <= max_fpr; precision projected to
    `prevalence`. Sets resolved=False if max_fpr < that curve's fpr_resolution."""


def threshold_for_precision(
    score: str, version: str, prevalence: float, min_precision: float,
) -> OperatingPoint:
    """Least stringent threshold reaching >= min_precision at this prevalence."""


def threshold_for_recall(
    score: str, version: str, prevalence: float, min_recall: float,
) -> OperatingPoint:
    """Most precise threshold still retaining >= min_recall (TPR)."""


def precision_vs_prevalence(
    score: str, version: str, threshold: float, prevalences,
) -> list[tuple[float, float]]:
    """(pi, precision) over a list/array of prevalences at a fixed threshold —
    backs the 'enter a range of pi' UX in caveat #2."""
```

CLI surface (one subcommand under the existing `alphajudge` CLI, or a tiny
standalone), e.g.:
```
alphajudge threshold --score ipSAE --version both --prevalence 0.001 --max-fpr 0.01
# -> threshold (t_af2=…, t_af3=…), recall=…, precision=… [CI …], resolved=…
```

## 8. Build tiers (when/if implemented)

- **Tier 1 (minimal, paper-citable):** `thresholds.py` + `freeze_threshold_rocs.py`
  + `threshold_rocs.json` + the CLI subcommand. Pure numpy, in-repo, reproducible.
- **Tier 2 (nice-to-have):** a thin Streamlit/HTML page with sliders over the
  same `thresholds.py` — a supplementary "AlphaJudge threshold calculator"
  resource. No new logic, just a front-end.

## 9. How this connects to the manuscript

The stringent/balanced/permissive rows already in the Discussion
("ipSAE >~ 0.5 stringent ~1% FPR / ~55% recall; 0.2 balanced; 0.05 permissive;
AF3 thresholds lower ~0.36/0.05/0.01") are **three hand-picked samples of exactly
this lookup**. The calculator generalises them to the whole curve and any
prevalence. If built, it would be cited as a Data/Software-availability resource;
no change to the results or claims is needed — it is a usability layer over
numbers the paper already reports.
```
