# All-Organism Atom Gap Benchmark

Source benchmark:
`/g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26/predictions`

Benchmark output:
`/g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26/zernike_all_organisms_atom_gap_scores_53086604`

Run scope:
- 7,345 rows from 4 organisms, AF2/AF3, positive and negative pairs.
- Compared `interface_sc`, old per-side atom Zernike cosine, and new shared-grid Atom Gap overlap.
- Runtime and robustness samples were skipped for this plotting pass.

Main result:
- `interface_sc` remains the best global discriminator: pooled all-data AUROC `0.681`.
- Atom Gap improves pooled AF3 AUROC versus SC: `0.644` vs `0.628`.
- Atom Gap rescues `20.5%` of AF3 low-SC positives by the negative-p90 rule.
- Atom Gap fails the current production gate because pooled all-data AUROC drops to `0.535`, mainly from very poor AF2 separation.
- Old per-side atom Zernike cosine remains saturated near `1.0` and is not a good production candidate.

Per-cell Atom Gap AUROC delta versus SC:
- Arabidopsis AF3: `+0.037`
- E. coli AF3: `+0.108`
- Human AF3: `+0.008`
- Yeast AF3: `+0.004`
- Arabidopsis AF2: `-0.360`
- E. coli AF2: `-0.422`
- Human AF2: `-0.266`
- Yeast AF2: `-0.315`

Interpretation:
Atom Gap is useful as an AF3-specific diagnostic/rescue signal, but the simple overlap score is not a safe SC replacement. The next iteration should keep the AF3 signal while adding a geometry term that penalizes broad nonspecific overlap in AF2-like confident models.
