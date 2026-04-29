# All-Organism Atom Gap Penalty Benchmark

Source benchmark:
`/g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26/predictions`

Benchmark output:
`/g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26/zernike_all_organisms_atom_gap_penalties_20260429_160322`

Run scope:
- 7,345 rows from 4 organisms, AF2/AF3, positive and negative pairs.
- Compared `interface_sc`, old per-side atom Zernike cosine, plain Atom Gap overlap, and Atom Gap penalties derived from the same order-12 gap coefficients.
- Runtime and robustness samples were skipped for this score-comparison pass.

Best new candidate:
`atom_gaussian__g32__o4__s1.5__mgapband__f12`

Main result:
- SC remains the best global discriminator: pooled all-data AUROC `0.681`.
- Plain Atom Gap pooled AF3 AUROC: `0.644`; all-data AUROC: `0.535`.
- Best band-pass Atom Gap pooled AF3 AUROC: `0.664`; all-data AUROC: `0.570`.
- Best band-pass Atom Gap rescues `21.9%` of AF3 low-SC positives.
- The band-pass penalty improves over plain overlap but still fails the production guardrail because AF2 AUROC remains far below SC.

Best band-pass per-cell AUROC delta versus SC:
- Arabidopsis AF3: `+0.041`
- E. coli AF3: `+0.101`
- Human AF3: `+0.034`
- Yeast AF3: `+0.021`
- Arabidopsis AF2: `-0.301`
- E. coli AF2: `-0.352`
- Human AF2: `-0.237`
- Yeast AF2: `-0.256`

Interpretation:
The low/mid-order band-pass term does what we wanted mechanistically: it improves the AF3 signal and rescues more low-SC positives than plain overlap. However, the core failure remains: geometry-only overlap-style Zernike scores rank many AF2 negatives too highly. This supports using Zernike as an AF3 rescue/diagnostic feature for now, not as a pure replacement for SC.
