#!/usr/bin/env bash
set -euo pipefail

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

AP_ROOT="/g/transform/kosinski/dima/IntAct_BioGRID_STRING/AlphaPulldownSnakemake"
CONFIG_DIR="${AP_ROOT}/config/paper_pairmatched_20260505_scratch_missing"
LOG_DIR="${LOG_DIR:-/scratch/dima/benchmark_26/af2_cached_missing_20260505_logs}"
SNAKEMAKE_BIN="/home/dmolodenskiy/.conda/envs/snake/bin/snakemake"
JOBS_PER_CELL="${JOBS_PER_CELL:-20}"
MAX_USER_JOBS="${MAX_USER_JOBS:-260}"
POLL_SECONDS="${POLL_SECONDS:-60}"
START_DELAY_SECONDS="${START_DELAY_SECONDS:-8}"
KEEP_GOING="${KEEP_GOING:-1}"
CACHED_AF2_DB="/scratch_cached/AlphaFold_DBs/2.3.2"

mkdir -p "${LOG_DIR}"
cd "${AP_ROOT}"

echo "[START af2_cached] $(date -Is) host=$(hostname)"
echo "[CONFIG] JOBS_PER_CELL=${JOBS_PER_CELL} MAX_USER_JOBS=${MAX_USER_JOBS} KEEP_GOING=${KEEP_GOING}"
echo "[THREADS] OPENBLAS=${OPENBLAS_NUM_THREADS} OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS} NUMEXPR=${NUMEXPR_NUM_THREADS}"
"${SNAKEMAKE_BIN}" --version

cells=(
  arabidopsis_af2_pos_pairs
  arabidopsis_af2_neg_pairs
  ecoli_af2_pos_pairs
  ecoli_af2_neg_pairs
  human_af2_pos_pairs
  human_af2_neg_pairs
  yeast_af2_pos_pairs
  yeast_af2_neg_pairs
)

assert_cached_config() {
  local config="$1"
  if ! grep -q "databases_directory: ${CACHED_AF2_DB}" "${config}"; then
    echo "[ERROR] ${config} does not use databases_directory: ${CACHED_AF2_DB}" >&2
    return 1
  fi
  if ! grep -q "backend_weights_directory: ${CACHED_AF2_DB}" "${config}"; then
    echo "[ERROR] ${config} does not use backend_weights_directory: ${CACHED_AF2_DB}" >&2
    return 1
  fi
}

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
  local snakemake_args=(
    --nolock
    --rerun-incomplete
    --configfile "${config}"
    --executor slurm
    --profile config/profiles/slurm
    --jobs "${JOBS_PER_CELL}"
    --restart-times "${RESTART_TIMES:-10}"
    --rerun-triggers code software-env
  )
  assert_cached_config "${config}"
  echo "[RUN ${cell}] $(date -Is) jobs=${JOBS_PER_CELL} log=${log}"
  if [[ "${KEEP_GOING}" == "1" || "${KEEP_GOING}" == "true" || "${KEEP_GOING}" == "yes" ]]; then
    snakemake_args+=(--keep-going)
  fi
  "${SNAKEMAKE_BIN}" "${snakemake_args[@]}" >"${log}" 2>&1
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

echo "[DONE af2_cached] $(date -Is) status=${status}"
exit "${status}"
