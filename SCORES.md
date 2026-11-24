# AlphaJudge scores

AlphaJudge writes one row per interface (chain pair) and model to `interfaces.csv`. Its scores
fall into two families. **Confidence scores** are read from the predictor output (ipTM, pTM,
ranking score) or computed from it (PAE, pLDDT, contact probabilities); they describe how sure
AlphaFold is of the interface. **Biophysical descriptors** are computed from the coordinates
alone, without AlphaFold's uncertainty estimates; they were developed for experimental structures
and were checked against CCP4 7.1 and PISA reference values on AF2 and AF3 models. A composite
**metascore** combines eleven of them.

The table below gives the definition AlphaJudge uses for every score, the CSV column that holds
it, and whether higher or lower values support an interaction. 

## Conventions

- **Contacts.** Two residues of different chains are in contact when their representative atoms
  (Cβ, or Cα when there is no Cβ; C1′ for nucleic acids) are within `--contact_thresh` (default 8 Å). A chain pair with
  at least one contact is an interface.
- **PAE directions.** PAE is not symmetric. Unless stated otherwise, both directions of each
  residue pair are used.
- **Cutoffs.** `--ipsae_pae_cutoff` (default 10 Å) applies to ipSAE only. Metascore percentiles
  are calibrated at the defaults (`--contact_thresh 8`, `--pae_filter 100`,
  `--ipsae_pae_cutoff 10`); other settings are flagged in `metascore_calibration_status`.
- **Better.** "Higher" or "Lower" states which direction indicates stronger support for an
  interaction. "—" marks columns that are reported but were not benchmarked as scores.

## Score definitions

| Score | CSV column | Computed from | Definition | Better |
|---|---|---|---|---|
| **Confidence scores** | | | | |
| Contact probability (top-ten mean) [1, 2] | `interface_contact_prob_top10_mean` | AF2: 64-bin distogram; AF3: native `contact_probs` matrix | Mean of the ten largest predicted interchain probabilities of a distance below 8 Å (all values if fewer than ten). AF2: softmax-normalized distogram, summing the bins whose upper boundary is below 8 Å; requires the `distogram` key in the result pickles. AF3: native matrix, each entry averaged with its transpose | Higher |
| Contact probability (maximum) | `interface_contact_prob_max` | As above | Largest single interchain probability of a distance below 8 Å | Higher |
| Expected contacts | `interface_expected_contacts` | As above | Sum of all interchain contact probabilities | — |
| LIS [3] | `interface_LIS` | PAE | Mean of (12 − PAE)/12 over all interchain residue pairs with PAE below 12 Å, computed for each chain direction and averaged | Higher |
| cLIS [4] | `interface_cLIS` | PAE and contacts | LIS restricted to interchain contacts | Higher |
| iLIS [4] | `interface_iLIS` | PAE and contacts | Geometric mean of LIS and cLIS, √(LIS × cLIS) | Higher |
| ipSAE [5, 6] | `interface_ipSAE` | PAE | Follows the reference software. Interchain residue pairs with PAE below `--ipsae_pae_cutoff` enter a residue-specific *d*₀ normalization; the score is maximized over each residue's qualifying partners in each chain direction, and the larger of the two directional values is reported. No distance cutoff enters the score | Higher |
| pDockQ2 [7] | `interface_pDockQ2` | PAE and pLDDT | *L*/[1 + exp(−*k*(*x* − *x*₀))] + *b*, where *x* is the mean interface pLDDT multiplied by the mean of 1/[1 + (PAE/10 Å)²] over the contacts, with *L* = 1.31, *x*₀ = 84.733, *k* = 0.075, *b* = 0.005; the larger of the two PAE directions is reported | Higher |
| ipTM [8, 9] | `iptm` | Predicted aligned-error distribution | Predicted fold accuracy of the interface, as output by AlphaFold. AF3: per chain pair from `chain_pair_iptm`; AF2: the model's global ipTM (`iptm_scope` records which) | Higher |
| Model confidence [8, 9] | `confidence_score` | Each predictor's own ranking score | AF2-Multimer: 0.8 ipTM + 0.2 pTM; AF3: `ranking_score`; Boltz-2: `confidence_score` | Higher |
| 0.8 ipTM + 0.2 pTM | `iptm_ptm` | ipTM and pTM | The AlphaFold-Multimer ranking combination, computed for every backend | Higher |
| pTM | `ptm` | Predicted aligned-error distribution | Predicted TM-score of the whole model, as output by AlphaFold | — |
| Average interface PAE [8, 10] | `average_interface_pae` | PAE | Mean PAE over the contacts, both directions | Lower |
| Average interface pLDDT | `interface_average_plddt` | pLDDT | Mean pLDDT of the interface residues of both chains | — |
| pDockQ / mpDockQ [11, 12] | `pDockQ/mpDockQ` | Interface pLDDT and contacts | One value per model. Two chains: pDockQ, *L*/[1 + exp(−*k*(*x* − *x*₀))] + *b*, where *x* is the mean interface pLDDT multiplied by log₁₀ of the number of contacts, with the corrected constants *L* = 0.724, *x*₀ = 152.611, *k* = 0.052, *b* = 0.018. More than two chains: mpDockQ, with *x* the mean interface pLDDT over all interfaces multiplied by log₁₀(total contacts + 1), and *L* = 0.728, *x*₀ = 309.375, *k* = 0.098, *b* = 0.262 | Higher |
| Interface score | `interface_score` | Interface pLDDT and contacts | Mean interface pLDDT × log₁₀(number of contacts). This is the *x* of pDockQ, and pDockQ increases monotonically with it | Higher |
| Confident-contact count (CCC) [13, 14] | `interface_ccc` | PAE and atom contacts | Number of interchain residue pairs in atomic contact under any of the Interactome3D rules (Cys S–S ≤ 2.56 Å, N–O ≤ 3.5 Å for a hydrogen bond, N–O ≤ 5.5 Å for a salt bridge, C–C ≤ 5.0 Å) whose chain-1-to-chain-2 PAE entry is strictly below 4 Å; each residue pair is counted once. Other contact, direction and boundary conventions are available through `Interface.confident_contacts(...)` | Higher |
| **Biophysical descriptors** | | | | |
| Shape complementarity [15] | `interface_sc` | Molecular (Connolly) surface | Complementarity of the two molecular surfaces (Lawrence–Colman definition) | Higher |
| Buried area [16, 17] | `interface_area` | Solvent-accessible surface area | Interface size (Å²) | Higher |
| Solvation energy [16, 17] | `interface_solv_en` | Solvent-accessible surface area × atomic solvation parameters | Solvation energy gain on forming the interface (kcal/mol), a measure of hydrophobic burial | Lower |
| Hydrogen bonds [16, 17] | `interface_hb` | Donor and acceptor geometry | Interchain hydrogen bonds, PISA-style: donor–acceptor distance 2.0–3.9 Å with CCP4 donor and acceptor types and a 90° angle limit; pairs counted as salt bridges are excluded | Higher |
| Salt bridges [16, 17] | `interface_sb` | Charged-atom geometry | Interchain ionic contacts, PISA-style: a Lys, Arg or His side-chain nitrogen (or N-terminal N) 2.0–4.0 Å from an Asp or Glu side-chain oxygen (or C-terminal O) | Higher |
| Disulfide bonds [16, 17] | `interface_ss` | Cysteine SG–SG distance | Putative interchain disulfides, counted at an SG–SG distance ≤ 2.3 Å | — |
| Number of contacts | `interface_contact_pairs` | Representative-atom distance | Unique interchain residue pairs in contact | Higher |
| Number of interface residues | `interface_num_intf_residues` | Representative-atom distance | Residues in at least one interchain contact | — |
| Fractions of polar, hydrophobic and charged interface residues | `interface_polar`, `interface_hydrophobic`, `interface_charged` | Residue types | Fraction of the interface residues in each class | — |
| **Composite** | | | | |
| Metascore | `interface_meta_score` | Eleven AlphaJudge scores | Components: contact probability (top-ten mean), LIS, ipSAE, pDockQ2, ipTM, model confidence, average interface PAE, pDockQ/mpDockQ, shape complementarity, hydrogen bonds and solvation energy. Each component is oriented so that higher means stronger support, converted to a percentile against frozen reference deciles computed from the benchmark positives of each AF version, and averaged; missing components are skipped, and `metascore_features` lists those used. For AF3 predictions whose global confidences also cover ligands or other unscored tokens, model confidence and a global ipTM are left out. Contact probability is a component, so the metascore is not independent of it | Higher |

## References

1. Humphreys, I. R. *et al.* Computed structures of core eukaryotic protein complexes.
   *Science* **374**, eabm4805 (2021). https://doi.org/10.1126/science.abm4805
2. Google DeepMind. AlphaFold 3 output documentation, release v3.0.2 (2024).
   https://github.com/google-deepmind/alphafold3/blob/v3.0.2/docs/output.md
3. Kim, A.-R. *et al.* Enhanced protein–protein interaction discovery via AlphaFold-Multimer.
   *bioRxiv* (2024). https://doi.org/10.1101/2024.02.19.580970
4. Kim, A.-R. *et al.* FlyPredictome: a structural atlas of predicted protein–protein interactions
   in *Drosophila*. *bioRxiv* (2026). https://doi.org/10.64898/2026.04.14.718529
5. Dunbrack, R. L. Rēs ipSAE loquuntur: what's wrong with AlphaFold's ipTM score and how to fix
   it. *bioRxiv* (2025). https://doi.org/10.1101/2025.02.10.637595
6. Dunbrack, R. L. IPSAE: software for scoring interprotein interactions in AlphaFold2 and
   AlphaFold3 predicted structures (2025). https://github.com/DunbrackLab/IPSAE
7. Zhu, W., Shenoy, A., Kundrotas, P. & Elofsson, A. Evaluation of AlphaFold-Multimer prediction
   on multi-chain protein complexes. *Bioinformatics* **39**, btad424 (2023).
   https://doi.org/10.1093/bioinformatics/btad424
8. Evans, R. *et al.* Protein complex prediction with AlphaFold-Multimer. *bioRxiv* (2022).
   https://doi.org/10.1101/2021.10.04.463034
9. Abramson, J. *et al.* Accurate structure prediction of biomolecular interactions with
   AlphaFold 3. *Nature* **630**, 493–500 (2024). https://doi.org/10.1038/s41586-024-07487-w
10. Jumper, J. *et al.* Highly accurate protein structure prediction with AlphaFold. *Nature*
    **596**, 583–589 (2021). https://doi.org/10.1038/s41586-021-03819-2
11. Bryant, P., Pozzati, G. & Elofsson, A. Improved prediction of protein–protein interactions
    using AlphaFold2. *Nat. Commun.* **13**, 1265 (2022); Author Correction, *Nat. Commun.*
    **13**, 1694 (2022). https://doi.org/10.1038/s41467-022-28865-w,
    https://doi.org/10.1038/s41467-022-29480-5
12. Bryant, P. *et al.* Predicting the structure of large protein complexes using AlphaFold and
    Monte Carlo tree search. *Nat. Commun.* **13**, 6028 (2022).
    https://doi.org/10.1038/s41467-022-33729-4
13. Lambourne, L. *et al.* Experimental assessment of AI-based interactome mapping.
    *Nat. Commun.* **17**, 4894 (2026). https://doi.org/10.1038/s41467-026-70942-x
14. Mosca, R., Céol, A. & Aloy, P. Interactome3D: adding structural details to protein networks.
    *Nat. Methods* **10**, 47–53 (2013). https://doi.org/10.1038/nmeth.2289
15. Lawrence, M. C. & Colman, P. M. Shape complementarity at protein/protein interfaces.
    *J. Mol. Biol.* **234**, 946–950 (1993). https://doi.org/10.1006/jmbi.1993.1648
16. Krissinel, E. & Henrick, K. Inference of macromolecular assemblies from crystalline state.
    *J. Mol. Biol.* **372**, 774–797 (2007). https://doi.org/10.1016/j.jmb.2007.05.022
17. Agirre, J. *et al.* The CCP4 suite: integrative software for macromolecular crystallography.
    *Acta Cryst. D* **79**, 449–461 (2023). https://doi.org/10.1107/S2059798323003595
