# Zernike Prototype Handoff

Last updated: 2026-05-03

## Working State

- Branch: `zernike`
- Latest pushed implementation commit before this update: `8ea886e Cache normal-gap Zernike benchmark coefficients`
- New in this update:
  - per-row normal-gap field diagnostics in benchmark score CSVs
  - adjustable diagnostic subset sizes via `--diagnostic-af3-sample-size` and `--diagnostic-hard-per-class`
  - fast lower-density normal-gap candidates
  - contact-signal normal-gap candidates that keep absolute low-pass contact amount instead of using only a normalized quality ratio
- Benchmark root: `/g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26/predictions`
- Important local caveat: `src/alphajudge/parsers/__init__.py` has unrelated local edits and should not be reverted or staged unless intentionally working on that file.

## Core Hypothesis

AlphaFold2/3 often predict the global complex arrangement better than local side-chain geometry. SC can fail on AF3 positives because it is sensitive to local clashes, rotamers, and exact Connolly dot pairing. Zernike should therefore act as a geometry-only low-pass interface score: preserve global contact/gap structure while reducing local noise.

## What We Tried

- Initial per-side Zernike descriptor cosine:
  - Implemented atom, residue-bead, surface, and joint-volume variants.
  - Real subset showed many scores saturated near `1.0`, including negatives.
  - Conclusion: broad rotation-invariant shape similarity is not interaction quality.

- Shared-grid atom gap scores:
  - Moved from per-side descriptor cosine to shared-box overlap/gap fields.
  - Better dynamic range and some AF3 signal, but still failed full promotion gates.

- Atom gap penalty / bandpass:
  - Best prior candidate: `atom_gaussian__g32__o4__s1.5__mgapband__f12`
  - All-data AUROC around `0.570`, AF3 AUROC around `0.664`, AF3 rescue around `21.9%`.
  - Conclusion: useful AF3 rescue direction, but not enough global/AF2 discrimination.

- Normal-aware smoothed Connolly gap Zernike:
  - Added representation `surface_normal_gap`.
  - Added score mode `normal_gap_field`.
  - Uses buried Connolly surface dots from both sides, nearest opposite dot pairing, opposing-normal complementarity, and three smoothed midpoint fields:
    - good contact
    - clash/tangential contact
    - far gap
  - Scores Zernike-smoothed good-contact signal against Zernike-smoothed clash and far-gap penalties.
  - Default candidate IDs:
    - `surface_normal_gap__g32__o4__s1.5__d3__tr3__pr2.3__mnormalgap__f12`
    - `surface_normal_gap__g32__o6__s1.5__d3__tr3__pr2.3__mnormalgap__f12`
    - `surface_normal_gap__g32__o8__s1.5__d3__tr3__pr2.3__mnormalgap__f12`

## Current Benchmark Evidence

- Tiny real human smoke run on 8 interfaces succeeded for the new normal-gap candidate.
- Normal-gap score was not saturated:
  - score range: about `0.34-0.72`
  - positive median: about `0.64`
  - negative median: about `0.49`
- This is too small to claim accuracy, but it confirms the new score avoids the previous near-`1.0` saturation failure mode.
- Runtime is currently expensive because Connolly dot generation dominates; normal-gap candidates should be benchmarked through Slurm or with persistent caches.

## 2026-05-03 Human AF3 Quick Diagnostic

Artifacts committed with this branch:

- `docs/zernike_human_normal_gap_quick_candidate_summary.csv`
- `docs/zernike_human_normal_gap_quick_candidate_diagnostic_summary.csv`
- `docs/zernike_human_normal_gap_quick_field_summary.csv`
- `docs/zernike_human_normal_gap_key_findings.svg`
- `docs/zernike_human_normal_gap_key_findings.png`

Command shape:

```bash
python scripts/benchmark_zernike_rescue.py \
  --bench-root /g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26/predictions \
  --out-dir /tmp/alphajudge_human_normal_gap_quick \
  --mode diagnostic \
  --organism human \
  --diagnostic-hard-per-class 3 \
  --diagnostic-af3-sample-size 24 \
  --runtime-sample-size 0 \
  --robustness-sample-size 0 \
  --jobs 12 \
  --candidate-id atom_gaussian__g32__o4__s1.5__mgapband__f12 \
  --candidate-id surface_normal_gap__g32__o4__s1.5__d3__tr3__pr2.3__mnormalgap__f12 \
  --candidate-id surface_normal_gap__g32__o6__s1.5__d3__tr3__pr2.3__mnormalgap__f12 \
  --candidate-id surface_normal_gap__g32__o8__s1.5__d3__tr3__pr2.3__mnormalgap__f12 \
  --candidate-id surface_normal_gap__g24__o4__s1.5__d1.5__tr3__pr2.3__mnormalgap__f12 \
  --candidate-id surface_normal_gap__g24__o6__s1.5__d1.5__tr3__pr2.3__mnormalgap__f12 \
  --candidate-id surface_normal_gap__g24__o8__s1.5__d1.5__tr3__pr2.3__mnormalgap__f12 \
  --candidate-id surface_normal_gap__g32__o4__s1.5__d3__tr3__pr2.3__mnormalcontact__f12 \
  --candidate-id surface_normal_gap__g32__o6__s1.5__d3__tr3__pr2.3__mnormalcontact__f12 \
  --candidate-id surface_normal_gap__g24__o4__s1.5__d1.5__tr3__pr2.3__mnormalcontact__f12 \
  --candidate-id surface_normal_gap__g24__o6__s1.5__d1.5__tr3__pr2.3__mnormalcontact__f12
```

Dataset:

- 30 human AF3 diagnostic rows:
  - 3 lowest-SC positives and 3 highest-SC negatives
  - 24 mixed AF3 rows enriched for low-SC positives and high-SC negatives
- SC is intentionally weak on this subset:
  - pooled AF3 AUROC `0.102`
  - positive median SC `0.059`
  - negative median SC `0.106`

Key result:

- Prior atom gap-band remains best on this quick subset:
  - `atom_gaussian__g32__o4__s1.5__mgapband__f12`
  - pooled AF3 AUROC `0.858`
  - AP `0.875`
  - AF3 failure rescue rate `0.400`
  - hard-slice AUROC `1.000`
- Pure normal-gap quality ratio is not good enough:
  - best quality-ratio normal-gap AUROC only `0.444`
  - positive/negative medians are inverted or nearly tied
  - reason from field diagnostics: positives have larger good-contact mass/signal, but the normalized quality ratio cancels this and lets some hard negatives score higher
- Contact-signal normal-gap is the promising normal-aware direction:
  - `surface_normal_gap__g24__o4__s1.5__d1.5__tr3__pr2.3__mnormalcontact__f12`
  - pooled AF3 AUROC `0.684`
  - AP `0.669`
  - AF3 failure rescue rate `0.333`
  - positive median `0.170`
  - negative median `0.112`
  - no saturation rejection on pooled quick diagnostic
- The full-density contact variant is similar but not clearly better:
  - `surface_normal_gap__g32__o4__s1.5__d3__tr3__pr2.3__mnormalcontact__f12`
  - pooled AF3 AUROC `0.667`
  - AP `0.643`
  - positive median `0.315`
  - negative median `0.216`
- Lower-density fast normal-gap is therefore worth keeping:
  - it is not worse in this quick diagnostic
  - it has a smaller surface-point budget
  - it should be preferred for larger exploratory sweeps unless a full benchmark contradicts this

Runtime/cache evidence:

- First fresh 30-row run with two normal-gap geometries took several minutes at about `0.07 rows/s` with 12 workers.
- Rerun in the same output/cache directory with contact candidates hit all cached normal-gap coefficients:
  - `normal_gap_coefficient_cache_hits: 60`
  - `normal_gap_coefficient_cache_misses: 0`
  - progress reported about `240 rows/s`
- Conclusion: normal-gap is currently cache-friendly but too expensive for casual uncached full sweeps. Use Slurm or warm caches for larger runs.

Interpretation:

- The normal-aware fields are informative, but the original score was the wrong functional form.
- Absolute low-pass good-contact amount matters. A pure quality ratio over-penalizes larger true interfaces because they naturally carry more clash/far-gap signal.
- The best next normal-gap family should be contact-gated:
  - preserve low-order opposing-normal good-contact signal
  - apply clash/far penalties
  - avoid normalizing away interface size completely
- Production `interface_zernike_sc` should remain unchanged.

## Useful Commands

Run focused tests:

```bash
python -m py_compile src/alphajudge/biophysics/zernike.py scripts/benchmark_zernike_rescue.py
pytest -q test/test_biophysics.py test/test_zernike_benchmark.py
```

Run tiny real smoke:

```bash
python scripts/benchmark_zernike_rescue.py \
  --bench-root /g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26/predictions \
  --out-dir /tmp/alphajudge_normal_gap_smoke \
  --mode smoke \
  --organism human \
  --smoke-sample-size 8 \
  --runtime-sample-size 0 \
  --robustness-sample-size 0 \
  --jobs 2 \
  --progress-every 4 \
  --candidate-id surface_normal_gap__g32__o4__s1.5__d3__tr3__pr2.3__mnormalgap__f12
```

## Cache Notes

- Normal-gap benchmark coefficients are cached under `normal_gap_coefficients/` inside the benchmark cache directory.
- Cache key includes model file, interface label, surface geometry parameters, normal-gap distance scales, grid size, sigma, padding, and fit order.
- A tiny 4-row real human smoke run showed the expected behavior:
  - first pass: Connolly work dominated at about `0.04 rows/s`
  - second pass: cache hits for all 4 normal-gap rows at about `100 rows/s`

## Next Implementation Ideas

- Run a larger human AF3 diagnostic for the contact-signal normal-gap candidates only, preferably through Slurm:
  - `surface_normal_gap__g24__o4__s1.5__d1.5__tr3__pr2.3__mnormalcontact__f12`
  - `surface_normal_gap__g24__o6__s1.5__d1.5__tr3__pr2.3__mnormalcontact__f12`
  - `surface_normal_gap__g32__o4__s1.5__d3__tr3__pr2.3__mnormalcontact__f12`
  - atom gap-band comparator
- Explore contact-scale sensitivity for `normal_gap_contact_field`:
  - current scale is `500`
  - try `250`, `750`, and `1000`
  - keep it analytic; do not learn weights from labels
- Consider an even cleaner normal-gap score:
  - `contact_amount * good_signal / (good_signal + clash + far)`
  - or separate contact amount from quality in plots instead of collapsing immediately
- Use lower `surface_density=1.5`, `grid_size=24` as the default exploratory normal-gap geometry unless larger benchmarks show it loses real accuracy.
- Do not promote `interface_zernike_sc` until full 16-cell benchmark passes SC guardrails.
