# AlphaJudge 🧠⚖️

*a cross-examiner for AlphaFold2 models*

[![license: MIT](https://img.shields.io/badge/license-MIT-informational.svg)](#-citation-and-license)
[![python](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![platform](https://img.shields.io/badge/platform-Linux%20%7C%20HPC-lightgrey.svg)]()

AlphaJudge evaluates **AlphaFold2-predicted complexes** and emits a tidy **CSV** that fuses **AI-derived confidence** (ipTM, pDockQ, PAE, pLDDT) with **physics/geometry** (area, contacts, H-bonds, salt bridges, electrostatics, etc.). It’s a fast sanity check for interface plausibility and a convenient feature generator for ML.

---

## 🔍 what it does

AlphaJudge parses AF2 multimer outputs and summarizes **per-model / per-interface** metrics:

| category                  | metrics (examples)                                                                  | notes                |
| ------------------------- | ----------------------------------------------------------------------------------- | -------------------- |
| **AlphaFold internal**    | ipTM, pTM, pDockQ/mpDockQ, avg interface PAE, avg interface pLDDT                   | from AF2 confidences |
| **physical & geometric**  | buried area, contact pairs, H-bonds, salt bridges, hydrophobic/polar/charged counts | CCP4-style reimpls   |
| **energetic (opt-in)**    | binding energy (PyRosetta), interface solvation, electrostatic complementarity      | extra deps           |
| **external / ML add-ons** | PI-score, VoroIF-GNN, ConSurf, ActifpTM, ipSAE, Pesto                               | optional             |
| **experimental / future** | FoldSeek-Multimer, conservation-weighted contacts, distogram-derived contacts       | planned              |

Use cases: rank poses, benchmark AF2 confidences, or feed ML on PPI datasets.

---

## 🧩 pipeline overview

```
AlphaFold2 models  →  AlphaJudge  →  scored_interfaces.csv
```

**under the hood**

1. parse AF PDB/JSON/PKL • 2) detect interfaces • 3) compute descriptors • 4) export a tidy table

---

## 🧰 installation

```bash
git clone https://github.com/KosinskiLab/AlphaJudge.git
cd AlphaJudge
mamba env create -f environment.yaml
conda activate alphajudge
```

**requirements:** Python ≥3.10, biopython, numpy, pandas, freesasa
**optional:** PyRosetta, VoroIF-GNN, Pesto

---

## 🚀 usage

```bash
alphajudge run \
  --models /path/to/alphafold_outputs \
  --output scores.csv
```

**useful flags**

```
--include-pyrosetta         # add binding energy
--include-voroif             # add VoroIF-GNN
--contact-threshold 5.0      # interface cutoff (Å)
```

---

## 📊 example output

| model_id | ipTM | mpDockQ | avg_intf_pae | int_area (Å²) | h_bonds |   sc | binding_energy (a.u.) | electrostatic_complementarity | PI_score |
| :------- | ---: | ------: | -----------: | ------------: | ------: | ---: | ------------------------: | ----------------------------: | -------: |
| A+B      | 0.86 |    0.72 |         2.10 |          1830 |       9 | 0.71 |                    -12345 |                          0.43 |     0.85 |
| A+C      | 0.64 |    0.39 |         5.42 |           780 |       3 | 0.48 |                     -2345 |                          0.12 |     0.41 |

---

## 🧠 motivation

AF2 scores ≠ guaranteed biophysics. **AlphaJudge** adds interpretable, physics-aware checks and export-ready features for downstream modeling and ML.

---

## 📚 citation and license

Please cite:

> Kosinski Lab, EMBL Hamburg. *AlphaJudge: an evaluation pipeline for AlphaFold-predicted complexes.* (2025).
> [https://github.com/KosinskiLab/AlphaJudge](https://github.com/KosinskiLab/AlphaJudge)

License: **MIT**. AlphaFold, PyRosetta, and other tools remain under their respective licenses.

---

## 💡 roadmap

* ActifpTM, ipSAE, FoldSeek-Multimer integration
* small MLP to combine features on PPI gold-standard datasets
* optional HTML report with plots and per-interface summaries

---

*“because even AlphaFold deserves a fair trial.”* ⚖️

---
