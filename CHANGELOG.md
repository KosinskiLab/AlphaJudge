# Changelog

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
