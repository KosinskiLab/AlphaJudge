# AlphaJudge: I am the score!

AlphaJudge evaluates AlphaFold-predicted protein complexes by merging AI-derived confidences (ipTM, pTM, iptm+ptm/confidence_score, pLDDT, PAE) with fast, self-contained interface biophysics (contacts, H-bonds, salt bridges, buried area, solvation proxy, shape complementarity) into a tidy CSV for downstream analysis.

![AlphaJudge icon](images/icon.png)

[![license: MIT](https://img.shields.io/badge/license-MIT-informational.svg)](#-citation-and-license)
[![python](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![platform](https://img.shields.io/badge/platform-Linux%20%7C%20HPC-lightgrey.svg)]()

---

## What it does

AlphaJudge parses AF2 and AF3 outputs and summarizes per-model / per-interface metrics:

| category | metrics (examples) | notes |
| --- | --- | --- |
| **AlphaFold internal** | ipTM, pTM, iptm+ptm/confidence_score, avg interface PAE, avg interface pLDDT | unified for AF2/AF3 |
| **physical & geometric** | buried area, contact pairs, H-bonds, salt bridges, interface composition | self-contained |
| **derived scores** | pDockQ, pDockQ2, mpDockQ, ipSAE, LIS, interface score | implemented here |

Use cases: rank poses, sanity-check AF confidences, or export features for ML.

---

## Pipeline overview

```
AlphaFold models (AF2 or AF3)  →  AlphaJudge  →  interfaces.csv
```

- Detects AF2 vs AF3 automatically from the run directory
- Loads structure and confidences, computes interface descriptors
- Writes `interfaces.csv` into the same directory

---

## Installation

Option A: conda/mamba env (recommended)

```bash
git clone https://github.com/KosinskiLab/AlphaJudge.git
cd AlphaJudge
mamba env create -f environment.yaml
conda activate alphajudge
```

Option B: pip install in existing environment

```bash
pip install -e .
```

Requirements: Python ≥3.10; runtime deps are `biopython`, `numpy`.

---

## CLI usage

The package exposes an `alphajudge` entry point.

```bash
alphajudge --path_to_dir /path/to/alphafold_run \
           --contact_thresh 8.0 \
           --pae_filter 100.0 \
           --models_to_analyse best
```

- **--path_to_dir**: Run directory containing AF2 or AF3 outputs
- **--contact_thresh**: Contact cutoff in Å (default: 8.0)
- **--pae_filter**: Skip interfaces with avg interface PAE above this (default: 100.0)
- **--models_to_analyse**: `best` or `all` (default: best)

Output: writes `interfaces.csv` next to the input directory.

Examples

```bash
# AF2 example (directory contains ranking_debug.json, pae_*.json, and model files)
alphajudge --path_to_dir test_data/af2/pos_dimers/Q13148+Q92900

# AF3 example (directory contains ranking_scores.csv, per-model summary/confidence files, and model files)
alphajudge --path_to_dir test_data/af3/pos_dimers/Q13148+Q92900 --models_to_analyse all
```

---

## Programmatic use

Minimal example:

```python
from pathlib import Path
from alphajudge.parsers import pick_parser
from alphajudge.runner import process

run_dir = Path("test_data/af2/pos_dimers/Q13148+Q92900")
parser = pick_parser(run_dir)
print("Detected parser:", parser.name)  # "af2" or "af3"
process(str(run_dir), contact_thresh=8.0, pae_filter=100.0, models_to_analyse="best")
print("Wrote:", run_dir / "interfaces.csv")
```

Key outputs per interface include: `average_interface_pae`, `interface_average_plddt`, `interface_contact_pairs`, `interface_area`, `interface_hb`, `interface_sb`, `interface_sc`, `interface_solv_en`, `interface_ipSAE`, `interface_LIS`, `interface_pDockQ2`, and per-run `pDockQ/mpDockQ`.

---

## Expected input layout

AlphaJudge expects standard AlphaFold run outputs.

- AF2: directory with `ranking_debug.json`, `pae_<model>.json`, and model structure files (`model.cif` or `*.pdb/*.cif`)
- AF3: directory with `ranking_scores.csv`, per-model `summary_confidences.json` and `confidences.json` (or top-level `ranked_0_summary_confidences.json`), and structure files

The tool searches for `model.cif` inside each model subdirectory first; otherwise it tries to match `*<model>*.cif` or `*<model>*.pdb` at the run root.

---

## Output schema (CSV)

AlphaJudge writes `interfaces.csv` with one row per interface (and includes the selected model). Core fields include:

- **jobs**: run directory name
- **model_used**: selected model identifier
- **interface**: chain-pair label (e.g., `A_B`)
- **iptm_ptm, iptm, ptm, confidence_score**: unified AF confidences
- **pDockQ/mpDockQ**: global dockQ-like score (mpDockQ if multimer; pDockQ if dimer)
- **average_interface_pae, interface_average_plddt, interface_num_intf_residues**
- **interface_contact_pairs, interface_score, interface_pDockQ2, interface_ipSAE, interface_LIS**
- **interface_hb, interface_sb, interface_sc, interface_area, interface_solv_en**

Exact header is asserted in tests to be consistent across AF2 and AF3 runs.

---

## Testing

```bash
pytest -q
```

Tests exercise both AF2 and AF3 parsers and validate the CSV fields against bundled fixtures in `test_data/`.

---

## Citation and license

Please cite:

> AlphaJudge: an evaluation pipeline for AlphaFold-predicted complexes. (2025).
> `https://github.com/KosinskiLab/AlphaJudge`

License: MIT for this repository. AlphaFold/AF3, PyRosetta, and other tools remain under their own licenses.

---

## Roadmap

- Additional metrics: ActifpTM, FoldSeek-Multimer integration
- Optional HTML report with plots and per-interface summaries
- Lightweight learned combiner for PPI benchmarking
