# Zernike Prototype Handoff

Last updated: 2026-05-03

## Working State

- Branch: `zernike`
- Latest pushed commits:
  - `33264b3 Align parser residue maps with PAE tokens`
  - `e52030f Benchmark SC-gated atom gap rescue`
- Current Zernike update:
  - SC-gated atom-gap rescue candidates that add a small Zernike correction only when `interface_sc` is low
  - full 16-cell all-organism benchmark for the tuned SC-gated atom-gap candidate
  - comparative plot against SC and the previous pure atom-gap best candidate
  - important interpretation update: pure Zernike did not beat SC globally, but SC-anchored Zernike rescue improves full-benchmark AUROC
- Benchmark root: `/g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26/predictions`
- Expected workspace state after the latest push: clean on `zernike`.

## Quick Start For Future Agents

- Do not assume pure Zernike beat SC. It did not.
- The strongest result so far is a hybrid rescue score, not a replacement for SC:
  `atom_gaussian__g32__o0__s1.5__mscoverlap__rs0.4__rf0.01__f12`.
- The tuned SC-gated candidate improves AUROC in all 8 organism/backend cells and passes AUROC guardrails, but production integration is intentionally paused because runtime, robustness, and threshold behavior still need checking.
- If asked to continue productively, the next best step is not another broad Zernike sweep. Run runtime plus side-chain jitter robustness for the tuned SC-gated candidate, then decide whether to expose it as a new score such as `interface_sc_zernike_rescue`.
- If asked for a paper/presentation artifact, use `docs/zernike_sc_gated_atom_gap_full_comparison.svg` and cite the tuned full benchmark CSVs below.

## Continuation Rules

- Every meaningful code, benchmark, or interpretation change should be reflected in both places:
  - a Git commit on branch `zernike`
  - this handoff file, with enough context for a future human, Codex, or Claude Code session to continue without chat history
- Keep commits small and named by outcome, for example `Benchmark SC-gated atom gap rescue` rather than `update files`.
- Commit benchmark artifacts needed to reproduce the story:
  - summary CSVs
  - per-cell metric CSVs
  - run metadata JSON
  - final presentation/paper plots
- Do not commit bulky per-interface score tables unless they are specifically needed for downstream inspection; prefer `/tmp` or a documented scratch output directory for rerunnable intermediates.
- Always report the benchmark root, command shape, candidate IDs, dataset slice, and whether runtime/robustness were skipped.
- Keep `interface_sc` as the explicit baseline in every benchmark summary and report deltas versus SC, not only absolute Zernike metrics.
- Do not promote or overwrite production `interface_zernike_sc` unless the candidate passes the full 16-cell guardrails and the handoff records the decision.
- If a score is hybrid SC+Zernike, label it as hybrid. Do not describe it as pure Zernike or geometry-only.
- Preserve unrelated user/local changes. If cleanup is needed, move ambiguous scratch files to a timestamped `/tmp` backup rather than deleting them.
- Before committing, run at least the focused compile/test commands in the Useful Commands section unless the change is documentation-only.

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

## 2026-05-03 Atom Gap-Band N Sweep

Artifacts committed with this branch:

- `docs/zernike_atom_gap_n_sweep_candidate_summary.csv`
- `docs/zernike_atom_gap_n_sweep_candidate_metrics.csv`
- `docs/zernike_atom_gap_n_sweep_candidate_diagnostic_summary.csv`
- `docs/zernike_atom_gap_n_sweep_order_curve.svg`
- `docs/zernike_atom_gap_n_sweep_order_curve.png`
- `docs/zernike_atom_gap_n_sweep_order_curve_selected_candidates.csv`
- `docs/zernike_atom_gap_n_sweep_human_candidate_summary.csv`
- `docs/zernike_atom_gap_n_sweep_human_candidate_metrics.csv`

Implementation additions:

- New benchmark-only score modes:
  - `gap_zernike_excess_bandpass`: subtracts expected random density overlap before multiplying by Zernike band energy.
  - `gap_zernike_soft_bandpass`: keeps `n >= 2`, keeps orders up to `N`, and smoothly tapers higher orders.
  - `gap_zernike_excess_contact_bandpass`: uses excess overlap as absolute contact amount, then multiplies by band quality.
- New sweep parameters:
  - `gap_band_soft_width`
  - `gap_contact_scale`
- Cached gap coefficients still use `fit_order=12`; lower `N` candidates reuse the same order-12 fit.

Stage 1 command shape:

```bash
python scripts/benchmark_zernike_rescue.py \
  --bench-root /g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26/predictions \
  --out-dir /tmp/alphajudge_zernike_atom_gap_n_sweep_diagnostic \
  --mode diagnostic \
  --backend af3 \
  --diagnostic-hard-per-class 6 \
  --diagnostic-af3-sample-size 200 \
  --runtime-sample-size 0 \
  --robustness-sample-size 0 \
  --jobs 16 \
  --progress-every 25 \
  --candidate-id ...atom-gap-only candidates...
```

Stage 1 dataset:

- 212 real AF3 rows:
  - human AF3 hard slice: 6 lowest-SC positives + 6 highest-SC negatives
  - fixed 200-row AF3 mixed sample across organisms
- SC is intentionally inverted on this hard diagnostic:
  - pooled AF3 AUROC `0.143`
  - positive median SC `0.068`
  - negative median SC `0.100`

Stage 1 result:

- Best rescue rate was contact-gated excess:
  - `atom_gaussian__g32__o6__s1.5__mgapcontact__cs0.07__f12`
  - rescue `0.311`
  - AF3 AUROC `0.638`
  - rejected because AF3 AUROC is more than `0.01` below the current atom gap-band reference.
- Best non-saturated diagnostic-pass rows were old overlap/nonuniform and soft-band variants:
  - `atom_gaussian__g32__o0__s1.5__mgapnonuniform__f12`: AF3 AUROC `0.687`, rescue `0.272`
  - `atom_gaussian__g32__o0__s1.5__moverlap__f12`: AF3 AUROC `0.687`, rescue `0.272`
  - `atom_gaussian__g32__o6__s1__mgapsoft__f12`: AF3 AUROC `0.670`, rescue `0.252`
  - `atom_gaussian__g32__o6__s1__mgapsoft__w2__f12`: AF3 AUROC `0.671`, rescue `0.233`
- Diagnostic interpretation:
  - Increasing hard bandpass `N` helped this deliberately hard AF3 slice, but this contradicts the earlier full all-organism result where higher `N` hurt.
  - Excess-overlap normalization alone did not create a clear win.
  - Soft bandpass is the only new family worth checking beyond Stage 1.

Stage 2 full-human survivor command shape:

```bash
python scripts/benchmark_zernike_rescue.py \
  --bench-root /g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26/predictions \
  --out-dir /tmp/alphajudge_zernike_atom_gap_n_sweep_human \
  --mode full \
  --organism human \
  --survivors-from docs/zernike_atom_gap_n_sweep_candidate_summary.csv \
  --survivor-limit 10 \
  --runtime-sample-size 0 \
  --robustness-sample-size 0 \
  --jobs 16 \
  --progress-every 200
```

Stage 2 result on full human AF2+AF3:

- SC baseline:
  - AF3 AUROC `0.620`
  - AF2 AUROC `0.735`
  - all-human AUROC `0.673`
- Best soft-band survivor:
  - `atom_gaussian__g24__o6__s1.5__mgapsoft__w2__f12`
  - AF3 AUROC `0.622`, only `+0.002` vs SC
  - AF2 AUROC `0.524`
  - all-human AUROC `0.557`
  - rescue `0.197`
- Old overlap still has slightly higher AF3 AUROC but bad score direction and very bad AF2:
  - `atom_gaussian__g32__o0__s1.5__moverlap__f12`
  - AF3 AUROC `0.628`
  - AF2 AUROC `0.470`
  - all-human AUROC `0.524`
- Decision:
  - Do not run Stage 3 full 16-cell for these atom-gap survivors yet.
  - None of the calibrated atom gap modes solves the main problem: AF3 rescue exists, but AF2/global discrimination collapses.

## 2026-05-03 SC-Gated Atom-Gap Full Benchmark

Artifacts from this update:

- `docs/zernike_sc_gated_atom_gap_full_candidate_summary.csv`
- `docs/zernike_sc_gated_atom_gap_full_candidate_metrics.csv`
- `docs/zernike_sc_gated_atom_gap_full_run_metadata.json`
- `docs/zernike_sc_gated_atom_gap_tuned_full_candidate_summary.csv`
- `docs/zernike_sc_gated_atom_gap_tuned_full_candidate_metrics.csv`
- `docs/zernike_sc_gated_atom_gap_tuned_full_run_metadata.json`
- `docs/zernike_sc_gated_atom_gap_full_comparison.svg`
- `docs/zernike_sc_gated_atom_gap_full_comparison.png`

Formula tested:

```text
score = interface_sc
        + weight * exp(-(max(interface_sc, 0) / scale)^2)
        * max(0, atom_gap_overlap - floor)
```

Interpretation:

- This is no longer a pure Zernike score. It is an SC-anchored rescue score: SC remains the baseline, and the low-resolution atom-gap Zernike/overlap signal is allowed to add evidence mainly for low-SC cases.
- This directly addresses the failure mode of the pure atom-gap candidates, which rescued AF3 positives but also inflated many negatives and damaged AF2/global discrimination.
- The tuned full-benchmark candidate is `atom_gaussian__g32__o0__s1.5__mscoverlap__rs0.4__rf0.01__f12`.
- Parameters:
  - atom Gaussian shared-box overlap
  - grid `32`
  - sigma `1.5`
  - Zernike fit order `12` for cached shared representation, but final rescue uses the raw shared-grid overlap base signal
  - rescue weight `0.2`
  - rescue scale `0.4`
  - rescue floor `0.01`

Full benchmark command shape:

```bash
python scripts/benchmark_zernike_rescue.py \
  --bench-root /g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26/predictions \
  --out-dir /tmp/alphajudge_sc_gated_atom_gap_tuned_full \
  --mode full \
  --runtime-sample-size 0 \
  --robustness-sample-size 0 \
  --jobs 32 \
  --progress-every 1000 \
  --candidate-id atom_gaussian__g32__o0__s1.5__mscoverlap__rs0.4__rf0.01__f12
```

Headline result versus `interface_sc`:

- Pooled all-data AUROC: `0.720` vs SC `0.681`, delta `+0.039`.
- Pooled AF3 AUROC: `0.665` vs SC `0.628`, delta `+0.037`.
- Pooled AF2 AUROC: `0.783` vs SC `0.758`, delta `+0.026`.
- AF3 low-SC failure-slice rescue rate: `0.204` vs SC `0.000` by construction of the failure slice.
- Positive-minus-negative median score separation is still small: `0.0135`, close to SC `0.0142`, so the win is mainly ranking/AUROC rather than a dramatic threshold-separated score scale.
- Guardrail pass is `1`; accepted-for-production remains `0` only because this family is intentionally marked benchmark-only and runtime/robustness were skipped for the full run.

Per-cell AUROC deltas versus SC:

| Organism | Backend | Delta AUROC | Delta AP |
| --- | --- | ---: | ---: |
| Arabidopsis | AF2 | `+0.030` | `-0.019` |
| Arabidopsis | AF3 | `+0.049` | `+0.038` |
| E. coli | AF2 | `+0.006` | `-0.001` |
| E. coli | AF3 | `+0.054` | `+0.058` |
| Human | AF2 | `+0.023` | `+0.013` |
| Human | AF3 | `+0.032` | `+0.036` |
| Yeast | AF2 | `+0.032` | `+0.009` |
| Yeast | AF3 | `+0.034` | `+0.050` |

Answer to the practical question:

- It is better than SC for full-benchmark ranking/discrimination: every organism/backend AUROC improves, so the candidate gives more true-positive-vs-false-positive separation across thresholds.
- It is not proven to have fewer false positives at every single operating threshold. AP drops slightly in Arabidopsis AF2 and E. coli AF2, so fixed-threshold deployment needs threshold-specific validation.
- At the AF3 low-SC rescue threshold definition, false-positive rate is controlled by each candidate's negative 90th percentile; under that controlled-FP comparison, the tuned SC-gated candidate rescues about `20%` of SC-failure positives.

Decision:

- This is the first Zernike-derived candidate that is clearly worth a production-design discussion.
- Do not replace `interface_zernike_sc` yet without an explicit decision, because the candidate is hybrid SC+Zernike rather than pure geometry-only Zernike.
- The new SC-gated score modes are benchmark-only until runtime, robustness, and threshold behavior are checked.
- Next if we want production integration:
  - run runtime sample and side-chain jitter robustness for this tuned candidate
  - choose whether to expose it as a new score name, for example `interface_sc_zernike_rescue`, rather than silently replacing the existing Zernike score
  - validate operating thresholds if the downstream use is threshold-based rather than rank-based

## Useful Commands

Run focused tests:

```bash
python -m py_compile \
  src/alphajudge/biophysics/zernike.py \
  src/alphajudge/parsers/__init__.py \
  scripts/benchmark_zernike_rescue.py \
  scripts/plot_zernike_sc_gated_atom_gap_full.py
pytest -q test/test_biophysics.py test/test_zernike_benchmark.py
```

Regenerate the latest comparison plot from committed CSVs:

```bash
python scripts/plot_zernike_sc_gated_atom_gap_full.py
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

- Atom gap-band has likely hit its current geometry-only ceiling as a standalone pure score.
- If continuing atom gap, use it as an auxiliary rescue feature rather than a standalone production replacement:
  - gate it to low-SC AF3-like failures, or
  - combine it with SC later, if hybrid scores become allowed.
- The more promising pure-Zernike direction is no longer simple atom density overlap; it is normal/contact-aware fields or a new explicit gap/clash volume with an absolute contact term.
- For the next pure geometry pass, prefer:
  - contact-gated normal-gap with lower-density cached Connolly dots
  - or atom/residue good-contact, clash, and far-gap fields instead of a single overlap field
- For the current best hybrid, do not silently overwrite `interface_zernike_sc`. If accepted, add it as a separately named score after runtime, robustness, and threshold checks.
