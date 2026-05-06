#!/usr/bin/env bash
# Unattended watchdog for the pairmatched 20260505 missing-only refill.
# Keeps one Snakemake controller alive per cell, in its own detached screen.
# Restarts any cell whose snakemake exited (--keep-going still exits non-zero
# on persistent failures). Refuses to launch new cells if doing so would push
# the user's Slurm queue over MAX_USER_JOBS.
#
# Approved 2026-05-06 to run during a 2-week vacation. Stop with:
#   screen -S watchdog_unattended_20260506 -X quit
set -uo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

AP_ROOT="${AP_ROOT:-/g/transform/kosinski/dima/IntAct_BioGRID_STRING/AlphaPulldownSnakemake}"
CONFIG_DIR="${CONFIG_DIR:-${AP_ROOT}/config/paper_pairmatched_20260505_scratch_missing}"
SNAKEMAKE_BIN="${SNAKEMAKE_BIN:-/home/dmolodenskiy/.conda/envs/snake/bin/snakemake}"
LOG_ROOT="${LOG_ROOT:-/scratch/dima/benchmark_26/watchdog_logs_20260506}"
WATCHDOG_LOG="${WATCHDOG_LOG:-/scratch/dima/benchmark_26/watchdog_20260506.log}"
MAX_USER_JOBS="${MAX_USER_JOBS:-290}"
POLL_SECONDS="${POLL_SECONDS:-300}"
LAUNCH_ESTIMATE_DIVISOR="${LAUNCH_ESTIMATE_DIVISOR:-1}"

# Per-cell Snakemake --jobs caps (DAG concurrency, not Slurm cap). Sized so a
# fully-loaded set of cells stays close to MAX_USER_JOBS in steady state.
declare -A CELL_JOBS=(
  [arabidopsis_af2_pos_pairs]=40
  [arabidopsis_af2_neg_pairs]=40
  [ecoli_af2_pos_pairs]=40
  [ecoli_af2_neg_pairs]=40
  [human_af2_pos_pairs]=40
  [human_af2_neg_pairs]=60
  [yeast_af2_pos_pairs]=40
  [yeast_af2_neg_pairs]=40
  [arabidopsis_af3_pos_pairs]=20
  [arabidopsis_af3_neg_pairs]=30
  [ecoli_af3_pos_pairs]=20
  [ecoli_af3_neg_pairs]=20
  [human_af3_pos_pairs]=40
  [human_af3_neg_pairs]=60
  [yeast_af3_pos_pairs]=30
  [yeast_af3_neg_pairs]=40
)

# 8*40 (AF2) + 4*20 + 2*30 + 2*60 (AF3) = 580 max DAG-active. Real running
# count is GPU-quota / HTC-quota throttled. The watchdog only gates new cell
# launches, so active Snakemake controllers can still submit after a launch. Use
# LAUNCH_ESTIMATE_DIVISOR=1 for conservative full-jobs accounting; use 2 only if
# the queue is badly underfilled and the user explicitly accepts soft-cap overshoot.

mkdir -p "${LOG_ROOT}"
touch "${WATCHDOG_LOG}"

log() {
  echo "$(date -Is) $*" | tee -a "${WATCHDOG_LOG}"
}

slurm_job_count() {
  squeue -u "${USER}" -h 2>/dev/null | wc -l
}

cell_alive() {
  local cell="$1"
  pgrep -f "config_${cell}\.yaml" >/dev/null 2>&1
}

launch_cell() {
  local cell="$1"
  local jobs="${CELL_JOBS[$cell]:-30}"
  local config="${CONFIG_DIR}/config_${cell}.yaml"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local cell_log="${LOG_ROOT}/${cell}.${stamp}.log"
  local screen_name="snake_${cell}"

  if [[ ! -f "${config}" ]]; then
    log "[ERROR] missing config: ${config}; skipping ${cell}"
    return 1
  fi
  if screen -ls 2>/dev/null | grep -qE "\.${screen_name}\b"; then
    screen -S "${screen_name}" -X quit 2>/dev/null || true
    sleep 2
  fi

  log "[LAUNCH] ${cell} jobs=${jobs} log=${cell_log}"
  screen -dmS "${screen_name}" bash -lc "
    cd '${AP_ROOT}' && \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    '${SNAKEMAKE_BIN}' \
      --nolock \
      --rerun-incomplete \
      --configfile '${config}' \
      --executor slurm \
      --profile config/profiles/slurm \
      --jobs ${jobs} \
      --restart-times 10 \
      --rerun-triggers code software-env \
      --keep-going \
      >>'${cell_log}' 2>&1
  "
}

# Adopt cells already running (don't relaunch them). The 13:49 yeast restart
# uses screen 'snake_af2_yeast_neg_20260506' but its Snakemake's command-line
# match against config_yeast_af2_neg_pairs.yaml so cell_alive() will see it.
log "[START] watchdog_unattended_20260506 host=$(hostname) MAX_USER_JOBS=${MAX_USER_JOBS} POLL=${POLL_SECONDS}s LAUNCH_ESTIMATE_DIVISOR=${LAUNCH_ESTIMATE_DIVISOR}"
log "[START] tracked cells: ${!CELL_JOBS[*]}"

while true; do
  total="$(slurm_job_count)"
  for cell in "${!CELL_JOBS[@]}"; do
    if cell_alive "${cell}"; then
      continue
    fi
    jobs="${CELL_JOBS[$cell]}"
    # Don't launch if doing so would push us over MAX_USER_JOBS once ramped up.
    estimate=$(( (jobs + LAUNCH_ESTIMATE_DIVISOR - 1) / LAUNCH_ESTIMATE_DIVISOR ))
    if (( total + estimate > MAX_USER_JOBS )); then
      log "[HOLD] ${cell}: total=${total} +${jobs}/${LAUNCH_ESTIMATE_DIVISOR} would exceed cap ${MAX_USER_JOBS}"
      continue
    fi
    launch_cell "${cell}" || log "[FAIL] launch_cell ${cell}"
    total=$((total + estimate))
    sleep 5
  done
  sleep "${POLL_SECONDS}"
done
