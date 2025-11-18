# AlphaJudge: I am the score!

AlphaJudge evaluates AlphaFold-predicted protein complexes by merging AI-derived confidences (ipTM, pTM, iptm+ptm/confidence_score, pLDDT, PAE) with fast, self-contained interface biophysics (contacts, H-bonds, salt bridges, buried area, solvation proxy, shape complementarity) into a tidy CSV for downstream analysis.

![AlphaJudge icon](images/icon.png)

[![license: MIT](https://img.shields.io/badge/license-MIT-informational.svg)](#citation-and-license)
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

Create conda/mamba env

```bash
git clone https://github.com/KosinskiLab/AlphaJudge.git
cd AlphaJudge
mamba env create -f environment.yaml
mamba activate alphajudge
```

Then, pip install in the existing environment

```bash
pip install .
```

or pip editable install in existing environment

```bash
pip install -e .
```
Requirements: Python ≥3.10; runtime deps are `biopython`, `numpy`, `matplotlib` (installed automatically with `pip install .`).

---

## CLI usage

The package exposes an `alphajudge` entry point.

```bash
# Basic synopsis
alphajudge PATH [PATH ...] \
  --models_to_analyse {best,all} \
  --contact_thresh 8.0 \
  --pae_filter 100.0 \
  [-r|--recursive] \
  [-o|--summary SUMMARY.csv]
```

- **PATH**: One or more run directories or roots to search
- **--contact_thresh**: Contact cutoff in Å (default: 8.0)
- **--pae_filter**: Skip interfaces with avg interface PAE above this (default: 100.0)
- **--models_to_analyse**: `best` or `all` (default: best)
- **-r / --recursive**: Recursively discover runs under each PATH
- **-o / --summary**: Write an aggregated CSV across all processed runs

Outputs:
- Always writes `interfaces.csv` inside each processed run directory.
- For each processed model, also writes a PAE heatmap PNG `pae_<model>.png` next to `interfaces.csv`.
- If `--summary` is provided, also writes a union-header CSV at the given path containing rows from all runs.

Examples

```bash
# Single AF2 run (directory contains ranking_debug.json, pae_*.json, and model files)
alphajudge test_data/af2/pos_dimers/Q13148+Q92900

# Single AF3 run (directory contains ranking_scores.csv, per-model summary/confidence files, and model files)
alphajudge test_data/af3/pos_dimers/Q13148+Q92900 --models_to_analyse all

# Aggregate multiple runs into one summary
alphajudge test_data/af2/pos_dimers/Q13148+Q92900 \
           test_data/af3/pos_dimers/Q13148+Q92900 \
           -o interfaces_summary.csv

# Recursively discover runs under roots and write a combined summary
alphajudge test_data/af2/pos_dimers test_data/af3/pos_dimers -r -o interfaces_summary.csv
```

---

## Programmatic use

Minimal example:

```python
from pathlib import Path
from alphajudge.parsers import pick_parser
from alphajudge.runner import process, process_many

run_dir = Path("test_data/af2/pos_dimers/Q13148+Q92900")
parser = pick_parser(run_dir)
print("Detected parser:", parser.name)  # "af2" or "af3"
process(str(run_dir), contact_thresh=8.0, pae_filter=100.0, models_to_analyse="best")
print("Wrote:", run_dir / "interfaces.csv")

# Multiple runs + optional recursion and summary
process_many(
    [str(run_dir), "test_data/af3/pos_dimers/Q13148+Q92900"],
    contact_thresh=8.0,
    pae_filter=100.0,
    models_to_analyse="best",
    recursive=False,
    summary_csv="interfaces_summary.csv",
)
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

## Docker

A minimal multi-stage Dockerfile is provided under `docker/`:

```bash
# Build image (runs tests in the build stage)
docker build -t alphajudge -f docker/Dockerfile .

# Inspect CLI inside the runtime image
docker run --rm alphajudge alphajudge --help
```

---

## Citation and license

Please cite:

> AlphaJudge: we will come up with a better name. (xxxx).
> `https://github.com/KosinskiLab/AlphaJudge`

License: MIT for this repository. AlphaFold2/AlphaFold3, and other tools remain under their own licenses.

---
