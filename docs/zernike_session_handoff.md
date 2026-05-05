# Zernike Prototype Handoff

Last updated: 2026-05-05

## 2026-05-05 Paper Draft Snapshot

A LaTeX manuscript draft and supporting figures were added locally and are
intentionally NOT committed to the public repo. They are gitignored:

- `docs/paper_alphajudge.tex` — manuscript draft (single source of truth for
  the current paper outline, figure references, captions, and `\todo{}` notes)
- `docs/figures/` — paper figures copied from
  `/g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26/`:
  - `flowchart.png`             — `data/exports/flowchart.png`
  - `scores_histos.png`         — `data/exports/scores_histos.png`
  - `dimers_venn_human.png`     — `data/exports/dimers_venn_human_canon.png`
  - `ipsae_cutoff_roc_<organism>.png` — `ipsae_scan/roc/roc__<organism>__all_cutoffs.png`
  - `classifier_roc_models.png` — `classifier/clf_out/roc_models.png`
  - `classifier_perm_importance.png` — `classifier/clf_out/perm_importance_plot.png`

Editorial decisions for this draft:

- **Zernike is intentionally NOT mentioned in the paper.** The decision is to
  leave Zernike out of the manuscript story entirely; the SC-gated atom-gap
  rescue candidate documented below is not in the paper either. Keep this
  branch and these notes alive only for internal continuity.
- **Shape complementarity (SC, `interface_sc`) is described as the strongest
  biophysical single score** (pooled AUROC `0.681`, the top of the biophysical
  tier). SC is the chosen biophysical reference in the paper, and the AF3
  degradation of SC is highlighted as a finding.
- The leading single-score predictors in the paper are PAE-derived:
  `interface_LIS` (`0.866`), `interface_ipSAE` (`0.863`),
  `interface_pDockQ2` (`0.848`).
- Tuned ipSAE PAE cutoff result: optimal cutoffs are `20`–`30 Å`, well above
  the original ipSAE defaults; per-organism tuned AUROC up to `0.93`.
- Multivariate classifier panel (`classifier/clf_out_repeats_all/`) reaches
  mean AUROC `~0.87` over `20` random grouped splits; ipSAE-only baseline is
  `0.856` over the same splits — i.e. the multivariate gain is ~`0.02` AUROC.
- Permutation importance: `interface_ipSAE` dominates;
  `average_interface_pae`, `pDockQ/mpDockQ`, `interface_LIS` follow;
  biophysical features carry near-zero marginal importance once PAE features
  are present.

Canonical numbers used in the paper (recomputed `2026-05-05` from
`/g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26/merged_best_interfaces_all_models.csv`,
`n=7345` rows, `3833` AF2 / `3512` AF3, `3823` pos / `3522` neg):

```
Pooled AUROC:
  interface_LIS        0.866
  interface_ipSAE      0.863
  interface_pDockQ2    0.848
  iptm                 0.841
  average_interface_pae 0.839 (sign-flipped)
  iptm_ptm             0.828
  interface_score      0.810
  pDockQ/mpDockQ       0.810
  interface_area       0.692
  interface_sc         0.681
  interface_sb         0.664
  interface_hb         0.653

Per (organism, backend) interface_sc AUROC (the AF3 degradation finding):
  arabidopsis af2 0.794 -> af3 0.619
  ecoli       af2 0.812 -> af3 0.611
  human       af2 0.735 -> af3 0.620
  yeast       af2 0.788 -> af3 0.651
```

Tuned ipSAE PAE-cutoff results from
`ipsae_scan/roc/best_cutoff_per_organism.csv`:

```
arabidopsis  PAE 20  AUROC 0.897   n_pos 182  n_neg 180
ecoli        PAE 30  AUROC 0.931   n_pos 96   n_neg 93
human        PAE 25  AUROC 0.859   n_pos 1100 n_neg 852
yeast        PAE 25  AUROC 0.876   n_pos 514  n_neg 495
```

Multivariate classifier validation AUROC (`20` random grouped splits) from
`classifier/clf_out_repeats_all/auc_summary_all_repeats.csv`:

```
LogReg       0.877 ± 0.012
SGD-log      0.874 ± 0.011
HistGB       0.871 ± 0.010
RF           0.871 ± 0.010
MLP          0.870 ± 0.011
ipSAE-only   0.856 ± 0.009
ipTM-only    0.853 ± 0.014
Chance       0.500
```

Continuation rules for the paper draft:

- The `.tex` and `docs/figures/` are gitignored. Do NOT commit them.
- Anything new for the paper goes through the same gitignored paths.
- If figures are regenerated in `benchmark_26/`, re-copy them with
  `cp` to `docs/figures/` so the local `.tex` keeps building.
- If the paper claims change, update both the `.tex` and this section so
  future sessions see consistent numbers.

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

## Whole-Project Research Takeaway

- AlphaJudge has a future, but the best score is unlikely to be a pure Zernike descriptor.
- The Zernike line itself has a future only as a hybrid rescue/sanity-check feature. As a pure replacement for SC, it should be considered exhausted for now.
- The reason is probably structural, not just poor tuning: AF2/AF3 false positives often already look interface-like because the model has optimized a plausible complex. Scores that mainly ask whether surfaces fit, including SC, atom-gap overlap, and low-pass Zernike descriptors, are partly scoring the same thing AF already made plausible.
- On the full benchmark, the strongest existing single features are already confidence/PAE-interface scores:
  - `interface_LIS`: all-data AUROC about `0.866`
  - `interface_ipSAE`: all-data AUROC about `0.863`
  - `interface_pDockQ2`: all-data AUROC about `0.848`
  - `interface_sc`: all-data AUROC about `0.681`
  - tuned SC-gated atom-gap rescue: all-data AUROC about `0.720`
- This means Zernike/SC-style geometry should be treated as complementary evidence and rescue/sanity-check signal, not as the main interaction classifier.
- A simple untrained rank ensemble of `interface_LIS`, `interface_ipSAE`, `interface_pDockQ2`, and tuned SC-gated atom-gap rescue gave a quick exploratory AUROC around `0.902` on the full benchmark:
  - Arabidopsis AF2/AF3: about `0.932` / `0.916`
  - E. coli AF2/AF3: about `0.946` / `0.946`
  - Human AF2/AF3: about `0.891` / `0.886`
  - Yeast AF2/AF3: about `0.927` / `0.897`
- This is not yet a production result because the combination was inspected after the Zernike work and needs proper cross-validation/calibration. It is still the clearest direction: combine confidence-block evidence with a low-resolution geometry rescue, rather than betting on one analytic geometry score.
- Freeze broad Zernike tuning after the current SC-gated rescue candidate. Do runtime/robustness checks, expose it as a separate score if accepted, and then pivot effort to better orthogonal evidence.
- Recommended next baseline: train a small cross-validated classifier on the existing AlphaJudge features, with AF2/AF3-aware calibration and organism/backend holdouts. This is the honest baseline any new hand-built score must beat.
- Best next out-of-the-box signals:
  - **Inverse-folding interface sequence likelihood.** Score each interface residue by the log-likelihood a structure-conditioned sequence model assigns to the actual amino acid, conditioning the model on its own chain backbone alone and then on both chains together; the gap is an interface-specificity term. Rationale: a real interface accommodates the residues it carries, so a designer model rates them as likely; AF false-positive interfaces often look geometrically plausible but place residues no inverse-folding model would pick at that site. Tools:
    - ESM-IF1 / GVPTransformer — Hsu et al. 2022, [paper](https://www.biorxiv.org/content/10.1101/2022.04.10.487779v1), [code](https://github.com/facebookresearch/esm/tree/main/examples/inverse_folding)
    - ProteinMPNN — Dauparas et al. 2022, [paper](https://www.science.org/doi/10.1126/science.add2187), [code](https://github.com/dauparas/ProteinMPNN)
  - **MSA conservation/coevolution residuals at predicted interface residues.** Build a paired MSA over the two chains, compute per-residue conservation (e.g. Shannon entropy / ConSurf-style) and inter-chain coupling strength (DCA/EVcomplex), then report the residual: interface-residue score minus the matched background of exposed *non-interface* surface residues from the same protein. Rationale: real binding patches are evolutionarily constrained and carry excess inter-chain couplings versus random surface; subtracting the same-protein surface background cancels protein-wide conservation bias and isolates interface-specific signal. Tools:
    - MSA building — HHblits ([hh-suite](https://github.com/soedinglab/hh-suite)), JackHMMER ([HMMER](http://hmmer.org/))
    - Conservation — [ConSurf](https://consurf.tau.ac.il/)
    - Inter-chain coevolution — EVcomplex / EVcouplings, Hopf et al. 2014, [paper](https://elifesciences.org/articles/03430), [code](https://github.com/debbiemarkslab/EVcouplings); MSA Transformer (Rao et al. 2021), [code](https://github.com/facebookresearch/esm)
  - **Conformational consistency across AF model/seed ensembles.** For each pair, generate an ensemble (the 5 AF2 model weights, multiple AF3 seeds, optionally dropout-on aggressive sampling) and quantify interface stability across runs: Jaccard overlap of predicted interface-residue sets, per-residue interface-RMSD spread after partner-aligned superposition, and a cluster pass on interface-CA RMSD reporting the largest cluster fraction and the gap to the second cluster. Rationale: true PPIs converge to one consistent interface across stochastic re-runs; spurious ones drift or split across incompatible binding modes, so consensus geometry is an orthogonal confidence axis the per-model PAE/pLDDT scores already in the benchmark do not capture. Cheap path: reuse the 5 AF2 models we already have for an `interface_consensus` score. Expensive path: AFsample-style aggressive sampling. Tools:
    - AlphaFold-Multimer — Evans et al. 2022, [biorxiv](https://www.biorxiv.org/content/10.1101/2021.10.04.463034)
    - AFsample — Wallner 2023, [paper](https://academic.oup.com/bioinformatics/article/39/9/btad573/7274860), code at [wallnerlab](https://github.com/wallnerlab)
    - AFsample2 (induced conformational diversity) — [code](https://github.com/wallnerlab/AFsample2)
- Research framing: AlphaJudge should probably expose several named evidence axes:
  - `interface_confidence_evidence`: LIS/ipSAE/pDockQ2/PAE-block strength
  - `interface_geometry_plausibility`: SC, area, solvation, clash/gap/contact-field evidence
  - `interface_zernike_rescue`: low-resolution geometry evidence for low-SC AF3 cases
  - `interface_consensus`: agreement across AF models/seeds/backends when available
  - `interface_specificity`: per-bait/per-target background or promiscuity correction when many candidates are scored together

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

## 2026-05-05 Metascore-First Decision

- Implemented the branch wrap-up direction as a transparent production metascore rather than another broad Zernike/orthogonal-signal sweep.
- New production column: `interface_meta_score`.
- Inputs:
  - higher is better: `interface_LIS`, `interface_ipSAE`, `interface_pDockQ2`, `iptm`, `confidence_score`, `pDockQ/mpDockQ`, `interface_sc`, `interface_area`
  - inverted before calibration: `average_interface_pae`, `interface_solv_en`
- Calibration:
  - frozen benchmark deciles from the full all-organism benchmark-26 best-interface run
  - each input maps to a `0-1` percentile and missing/non-finite values are ignored
  - final score is the mean of available calibrated percentiles
- The SC-gated atom-gap/Zernike rescue remains benchmark-only and is not part of `interface_meta_score`.
- Reproducible diagnostic script:

```bash
python scripts/analyze_interface_meta_score.py \
  --input-csv /g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26/benchmark_best.20260422_111918_904fef5.csv
```

Investigation evidence:

- Best single current score on the full tuned-run CSV was `interface_LIS` at AUROC about `0.866`.
- A selected current-score rank metascore reached about `0.873` AUROC / `0.910` AP before frozen-decile approximation.
- The production frozen-decile `interface_meta_score` is intentionally simpler and should be treated as a ranking/prioritization score, not a universal binary threshold.
- PCA remains diagnostic only: PC1 is mostly the confidence/PAE block, while geometry/Zernike-like axes live in later PCs and are weaker alone.

## Useful Commands

Run focused tests:

```bash
python -m py_compile \
  src/alphajudge/biophysics/zernike.py \
  src/alphajudge/meta_score.py \
  src/alphajudge/parsers/__init__.py \
  scripts/benchmark_zernike_rescue.py \
  scripts/analyze_interface_meta_score.py \
  scripts/plot_zernike_sc_gated_atom_gap_full.py
pytest -q test/test_meta_score.py test/test_biophysics.py test/test_zernike_benchmark.py
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
