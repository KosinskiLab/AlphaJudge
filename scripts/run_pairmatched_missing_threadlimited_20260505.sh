#!/usr/bin/env bash
set -euo pipefail

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

AP_ROOT="/g/transform/kosinski/dima/IntAct_BioGRID_STRING/AlphaPulldownSnakemake"
CONFIG_DIR="${AP_ROOT}/config/paper_pairmatched_20260505_scratch_missing"
LOG_DIR="/scratch/dima/benchmark_26/threadlimited_missing_20260505_logs"
SNAKEMAKE_BIN="/home/dmolodenskiy/.conda/envs/snake/bin/snakemake"
JOBS_PER_CELL="${JOBS_PER_CELL:-16}"
MAX_USER_JOBS="${MAX_USER_JOBS:-280}"
POLL_SECONDS="${POLL_SECONDS:-60}"
START_DELAY_SECONDS="${START_DELAY_SECONDS:-8}"

mkdir -p "${LOG_DIR}"
cd "${AP_ROOT}"

echo "[START threadlimited] $(date -Is) host=$(hostname)"
echo "[CONFIG] JOBS_PER_CELL=${JOBS_PER_CELL} MAX_USER_JOBS=${MAX_USER_JOBS}"
echo "[THREADS] OPENBLAS=${OPENBLAS_NUM_THREADS} OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS} NUMEXPR=${NUMEXPR_NUM_THREADS}"
"${SNAKEMAKE_BIN}" --version

# Excludes cells already controlled by still-running Snakemake masters:
# arabidopsis_af2_pos_pairs, arabidopsis_af3_pos_pairs,
# arabidopsis_af3_neg_pairs, and human_af2_pos_pairs.
cells=(
  ecoli_af2_pos_pairs
  ecoli_af2_neg_pairs
  ecoli_af3_pos_pairs
  ecoli_af3_neg_pairs
  human_af2_neg_pairs
  human_af3_pos_pairs
  human_af3_neg_pairs
  yeast_af2_pos_pairs
  yeast_af2_neg_pairs
  yeast_af3_pos_pairs
  yeast_af3_neg_pairs
)

slurm_job_count() {
  squeue -u "${USER}" -h | wc -l
}

wait_for_slurm_room() {
  local cell="$1"
  local count
  while true; do
    count="$(slurm_job_count)"
    if (( count + JOBS_PER_CELL <= MAX_USER_JOBS )); then
      return 0
    fi
    echo "[WAIT] ${cell}: ${count} Slurm jobs; waiting for room under ${MAX_USER_JOBS}"
    sleep "${POLL_SECONDS}"
  done
}

run_cell() {
  local cell="$1"
  local config="${CONFIG_DIR}/config_${cell}.yaml"
  local log="${LOG_DIR}/${cell}.log"
  echo "[RUN ${cell}] $(date -Is) jobs=${JOBS_PER_CELL} log=${log}"
  "${SNAKEMAKE_BIN}" \
    --nolock \
    --configfile "${config}" \
    --executor slurm \
    --profile config/profiles/slurm \
    --jobs "${JOBS_PER_CELL}" \
    --restart-times 5 \
    --rerun-triggers code params software-env \
    >"${log}" 2>&1
  echo "[DONE ${cell}] $(date -Is)"
}

pids=()
for cell in "${cells[@]}"; do
  wait_for_slurm_room "${cell}"
  run_cell "${cell}" &
  pids+=("$!")
  sleep "${START_DELAY_SECONDS}"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

echo "[DONE threadlimited] $(date -Is) status=${status}"
exit "${status}"
