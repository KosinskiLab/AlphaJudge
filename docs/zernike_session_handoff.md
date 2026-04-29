# Zernike Prototype Handoff

Last updated: 2026-04-29

## Working State

- Branch: `zernike`
- Latest pushed implementation commit before this note: `8d52479 Add normal-aware Connolly gap Zernike fields`
- New in this handoff commit: persistent benchmark caching for normal-gap order-12 coefficient bundles.
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

- Run a human diagnostic subset across `o4/o6/o8` normal-gap candidates and compare against SC plus atom gap-band.
- If normal-gap is promising but slow, test lower `surface_density`, smaller `grid_size`, and/or cached Connolly dot reuse.
- Consider field-level diagnostics in benchmark CSVs:
  - good mass
  - clash mass
  - far mass
  - good/clash/far structured ratios
- Do not promote `interface_zernike_sc` until full 16-cell benchmark passes SC guardrails.
