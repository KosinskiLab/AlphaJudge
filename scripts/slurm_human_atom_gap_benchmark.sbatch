#!/usr/bin/env bash
#SBATCH --job-name=zernike_human_atom_gap
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=900G
#SBATCH --time=2-00:00:00
#SBATCH --output=/g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26/slurm_logs/%x_%j.out
#SBATCH --error=/g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26/slurm_logs/%x_%j.err

set -euo pipefail

export BLIS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export VECLIB_MAXIMUM_THREADS=1

REPO_ROOT=/g/kosinski/dima/PycharmProjects/AlphaJudge
BENCH_ROOT=/g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26/predictions
OUT_ROOT=/g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26
PYTHON=/home/dmolodenskiy/.conda/envs/AlphaPulldown_alphafold3/bin/python
CANDIDATE_ID=atom_gaussian__g32__o0__s1.5__moverlap__f12

cd "$REPO_ROOT"

OUT_DIR="${OUT_ROOT}/zernike_human_atom_gap_full_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

echo "repo: $REPO_ROOT"
echo "commit: $(git rev-parse HEAD)"
echo "candidate: $CANDIDATE_ID"
echo "out_dir: $OUT_DIR"
echo "cpus: ${SLURM_CPUS_PER_TASK:-256}"

"$PYTHON" scripts/benchmark_zernike_rescue.py \
  --bench-root "$BENCH_ROOT" \
  --out-dir "$OUT_DIR" \
  --mode full \
  --organism human \
  --runtime-sample-size 200 \
  --robustness-sample-size 50 \
  --jobs "${SLURM_CPUS_PER_TASK:-256}" \
  --progress-every 100 \
  --candidate-id "$CANDIDATE_ID"
