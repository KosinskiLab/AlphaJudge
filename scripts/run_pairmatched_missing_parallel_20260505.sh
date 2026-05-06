#!/usr/bin/env bash
set -euo pipefail

AP_ROOT="/g/transform/kosinski/dima/IntAct_BioGRID_STRING/AlphaPulldownSnakemake"
CONFIG_DIR="${AP_ROOT}/config/paper_pairmatched_20260505_scratch_missing"
LOG_DIR="/scratch/dima/benchmark_26/parallel_missing_20260505_logs"
JOBS_PER_CELL="${JOBS_PER_CELL:-16}"
MAX_USER_JOBS="${MAX_USER_JOBS:-280}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "${LOG_DIR}"
cd "${AP_ROOT}"

echo "[START parallel] $(date -Is) host=$(hostname)"
echo "[CONFIG] JOBS_PER_CELL=${JOBS_PER_CELL} MAX_USER_JOBS=${MAX_USER_JOBS}"
conda run -n snake snakemake --version

cells=(
  arabidopsis_af2_neg_pairs
  arabidopsis_af3_pos_pairs
  arabidopsis_af3_neg_pairs
  ecoli_af2_pos_pairs
  ecoli_af2_neg_pairs
  ecoli_af3_pos_pairs
  ecoli_af3_neg_pairs
  human_af2_pos_pairs
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
  echo "[RUN ${cell}] $(date -Is) config=${config} log=${log}"
  conda run -n snake snakemake \
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
  sleep 5
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

echo "[DONE parallel] $(date -Is) status=${status}"
exit "${status}"
