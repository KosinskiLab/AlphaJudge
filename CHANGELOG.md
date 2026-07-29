# Changelog

## 1.3.0 - 2026-07-29

### Added
- **Interface contact-probability scores.** AlphaFold's own estimate that two residues are in contact, taken from the predicted distance distribution — AlphaFold3's native `contact_probs`, and for AlphaFold2 by softmaxing the distogram logits and summing the mass below a contact threshold. Emits `interface_contact_prob_max`, `interface_contact_prob_top10_mean`, `interface_expected_contacts` and `interface_contact_prob_source` per interface. Requires the run to retain its distance distributions (AF3 `*_distogram.npz`, or AF2 result pickles written without pruning); when they are absent the columns are simply empty.
- On a 12,315-pair four-organism benchmark this is **the strongest single score on AlphaFold3** (AUROC 0.842 versus 0.816 for ipSAE and 0.833 for the ten-feature meta-score), and the relationship is one-directional: contact probability adds +0.024 AUROC on top of ipSAE while ipSAE adds nothing back. On AlphaFold2 — where the quantity is reconstructed from the distogram rather than read from a trained head — the gain is small and organism-inconsistent, so ipSAE/LIS remain the AF2 recommendation.

### Changed
- **`AF2_DISTOGRAM_CONTACT_CUTOFF` is now 8 Å (was 12 Å), with an upper-bound bin convention.** The contact threshold was swept from 4 to 20 Å in 2 Å steps on the benchmark: discrimination peaks at 6–8 Å for both AlphaFold versions and decays only gently out to 20 Å. The previous 12 Å lower-bound setting scores ~0.005 AUROC lower on AF2. `interface_contact_prob_source` accordingly reports `af2_distogram_le_8A` rather than `af2_distogram_lb_lt_12A`.
- **`interface_meta_score` now averages eleven features, adding `interface_contact_prob_top10_mean`** (deciles frozen on the benchmark's positives, as for the other features). Only one of the two contact-probability variants is included — `max` and `top10_mean` correlate at ρ = 0.997, so including both would merely double-weight one signal. **This shifts absolute meta-score values** (mean +0.010 on AF2 rows, −0.005 on AF3, max 0.053), so meta-score readings and decile labels are not directly comparable with 1.2.0; per-feature scores and within-run rankings are unaffected. Runs without contact probabilities fall back to the previous ten-feature mean via the existing missing-feature rule.

## 1.2.0 - 2026-06-15

### Added
- iLIS interface score (AFM-LIS; Kim et al., github.com/flyark/AFM-LIS): the geometric mean `iLIS = sqrt(LIS * cLIS)`, where `cLIS` is the LIS PAE transform averaged only over residue pairs in direct contact (CB-else-CA within `contact_thresh`, default 8 Å). Adds `Interface.clis()`/`ilis()` and emits `interface_cLIS` / `interface_iLIS` next to `interface_LIS` in the per-run CSV. Validated against the official AFM-LIS `lis.py` (agreement to 4 decimal places).

### Changed
- Percentile sliders and meta-score are now calibrated on **positive (interacting) benchmark pairs only** (3,878 AF2/AF3 positive rows) instead of the full decoy-padded population, so predictions are ranked against the distribution of real interfaces. Per-feature AUROC is unchanged (monotonic transform); production `interface_meta_score` AUROC on the balanced benchmark is 0.878. Use `scripts/freeze_metascore_quantiles.py --label-filter positive|negative|all` to reproduce the deciles (`all` reproduces the prior scale bit-for-bit).
- Removed the black poly-line connecting per-group slider markers in the report; each metric still shows its percentile marker, and the Meta marker is recomputed from the current calibration so it stays consistent with the recalibrated sliders.

## 1.1.0 - 2026-06-03

### Added
- RCSB/wwPDB-style PDF validation report. The `--report` flag writes a per-run `report.pdf` next to `interfaces.csv`, with per-interface slider pages, a per-interface table for multimers, an AFDB-style PAE panel, and a complex-level evidence section.
- `--aggregate_report PDF`: a multi-page cohort validation PDF built from the `--summary` CSV — meta-score histogram, top-N interfaces table, one slider page per interface ranked by meta-score, and a "Per-complex evidence" section (top-N capped by distinct complex).
- Interface meta-score module used to rank and summarise interfaces in the reports.

### Changed
- The interface meta-score (and report) now use `interface_hb` (interface hydrogen bonds) instead of `interface_area`.
- Refreshed `BENCHMARK_QUANTILES` to the canonical `final_sync_20260523` benchmark table; percentile/meta-score outputs change accordingly.
- Report visuals now follow wwPDB/RCSB conventions more closely (AlphaJudge branding, AFDB-style PAE PNG, grouped slider polylines, cleaner headers/footers).

### Fixed
- `--aggregate_report` now guards against a stale summary CSV instead of emitting an inconsistent report.

## 1.0.2 - 2026-05-20

- Added official DeepMind AF3 layout and Boltz-2 parser support; custom parser subclasses should implement `BaseParser.detect` as `detect(d: Path) -> bool` because it is now a static method.
- Unknown or missing AF3 confidence schemas now raise an error instead of fabricating default PAE values.
