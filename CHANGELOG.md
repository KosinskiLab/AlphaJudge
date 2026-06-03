# Changelog

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
