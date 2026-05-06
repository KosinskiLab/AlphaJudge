#!/usr/bin/env bash
set -euo pipefail

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

AP_ROOT="${AP_ROOT:-/scratch/dima/AlphaPulldownSnakemake_af3_20260505}"
CONFIG_DIR="${AP_ROOT}/config/paper_pairmatched_20260505_af3_clean"
LOG_DIR="${LOG_DIR:-/scratch/dima/benchmark_26/af3_clean_20260505_logs}"
SNAKEMAKE_BIN="${SNAKEMAKE_BIN:-/home/dmolodenskiy/.conda/envs/snake/bin/snakemake}"
FEATURE_JOBS_PER_CELL="${FEATURE_JOBS_PER_CELL:-12}"
INFERENCE_JOBS_PER_CELL="${INFERENCE_JOBS_PER_CELL:-10}"
MAX_USER_JOBS="${MAX_USER_JOBS:-240}"
POLL_SECONDS="${POLL_SECONDS:-60}"
START_DELAY_SECONDS="${START_DELAY_SECONDS:-8}"
CACHED_AF3_DB="/scratch_cached/AlphaFold_DBs/3.0.0"
AF3_WEIGHTS="/g/alphafold/af3/"

mkdir -p "${LOG_DIR}"
cd "${AP_ROOT}"

echo "[START af3_clean] $(date -Is) host=$(hostname)"
echo "[CONFIG] AP_ROOT=${AP_ROOT}"
echo "[CONFIG] FEATURE_JOBS_PER_CELL=${FEATURE_JOBS_PER_CELL} INFERENCE_JOBS_PER_CELL=${INFERENCE_JOBS_PER_CELL} MAX_USER_JOBS=${MAX_USER_JOBS}"
echo "[THREADS] OPENBLAS=${OPENBLAS_NUM_THREADS} OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS} NUMEXPR=${NUMEXPR_NUM_THREADS}"
"${SNAKEMAKE_BIN}" --version

feature_cells=(
  arabidopsis_af3_features
  ecoli_af3_features
  human_af3_features
  yeast_af3_features
)

inference_cells=(
  arabidopsis_af3_pos_pairs
  arabidopsis_af3_neg_pairs
  ecoli_af3_pos_pairs
  ecoli_af3_neg_pairs
  human_af3_pos_pairs
  human_af3_neg_pairs
  yeast_af3_pos_pairs
  yeast_af3_neg_pairs
)

assert_af3_config() {
  local config="$1"
  if ! grep -q "databases_directory: ${CACHED_AF3_DB}" "${config}"; then
    echo "[ERROR] ${config} does not use databases_directory: ${CACHED_AF3_DB}" >&2
    return 1
  fi
  if ! grep -q "backend_weights_directory: ${AF3_WEIGHTS}" "${config}"; then
    echo "[ERROR] ${config} does not use backend_weights_directory: ${AF3_WEIGHTS}" >&2
    return 1
  fi
  if ! grep -q -- "--data_pipeline: alphafold3" "${config}"; then
    echo "[ERROR] ${config} does not use AF3 data pipeline" >&2
    return 1
  fi
  if ! grep -q -- "--fold_backend: alphafold3" "${config}"; then
    echo "[ERROR] ${config} does not use AF3 fold backend" >&2
    return 1
  fi
}

assert_no_cross_pairset_features() {
  local config="$1"
  if grep -qE '/af3/(pos_pairs|neg_pairs)/features' "${config}"; then
    echo "[ERROR] ${config} points at pairset-local AF3 features; use the stable cache instead" >&2
    return 1
  fi
}

slurm_job_count() {
  squeue -u "${USER}" -h | wc -l
}

wait_for_slurm_room() {
  local cell="$1"
  local jobs="$2"
  local count
  while true; do
    count="$(slurm_job_count)"
    if (( count + jobs <= MAX_USER_JOBS )); then
      return 0
    fi
    echo "[WAIT] ${cell}: ${count} Slurm jobs; waiting for room under ${MAX_USER_JOBS}"
    sleep "${POLL_SECONDS}"
  done
}

run_cell() {
  local cell="$1"
  local jobs="$2"
  local phase="$3"
  local config="${CONFIG_DIR}/config_${cell}.yaml"
  local log="${LOG_DIR}/${cell}.log"
  assert_af3_config "${config}"
  if [[ "${phase}" == "inference" ]]; then
    assert_no_cross_pairset_features "${config}"
  fi
  echo "[RUN ${phase} ${cell}] $(date -Is) jobs=${jobs} log=${log}"
  "${SNAKEMAKE_BIN}" \
    --nolock \
    --rerun-incomplete \
    --configfile "${config}" \
    --executor slurm \
    --profile config/profiles/slurm \
    --jobs "${jobs}" \
    --restart-times 5 \
    --rerun-triggers code software-env \
    >"${log}" 2>&1
  echo "[DONE ${phase} ${cell}] $(date -Is)"
}

run_phase() {
  local phase="$1"
  local jobs="$2"
  shift 2
  local cells=("$@")
  local pids=()
  local status=0

  for cell in "${cells[@]}"; do
    wait_for_slurm_room "${cell}" "${jobs}"
    run_cell "${cell}" "${jobs}" "${phase}" &
    pids+=("$!")
    sleep "${START_DELAY_SECONDS}"
  done

  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  return "${status}"
}

status=0
if ! run_phase "features" "${FEATURE_JOBS_PER_CELL}" "${feature_cells[@]}"; then
  status=1
fi

if [[ "${status}" -eq 0 ]]; then
  if ! run_phase "inference" "${INFERENCE_JOBS_PER_CELL}" "${inference_cells[@]}"; then
    status=1
  fi
else
  echo "[SKIP inference] feature-cache phase failed"
fi

echo "[DONE af3_clean] $(date -Is) status=${status}"
exit "${status}"
