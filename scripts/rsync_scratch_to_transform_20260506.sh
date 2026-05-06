#!/usr/bin/env bash
# Daily rsync of completed AlphaPulldown predictions from scratch (volatile,
# auto-cleaned) to /g/transform (durable). Approved 2026-05-06 ahead of a
# 2-week vacation; run from cron once per day.
#
# - --update      : never overwrite a newer dest file
# - --partial     : keep partial transfers if interrupted
# - no --delete   : we never want to remove dest files
# - exclude .snakemake / slurm-logs to keep transfer small and meaningful
set -uo pipefail

SRC="${SRC:-/scratch/dima/benchmark_26/predictions/}"
DEST="${DEST:-/g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26/predictions/}"
LOG_DIR="${LOG_DIR:-/scratch/dima/benchmark_26/sync_logs}"
DURABLE_LOG_DIR="${DURABLE_LOG_DIR:-/g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26/sync_logs_20260506}"
mkdir -p "${LOG_DIR}"
mkdir -p "${DURABLE_LOG_DIR}"

stamp="$(date +%Y%m%d_%H%M%S)"
log="${LOG_DIR}/sync_scratch_to_transform_${stamp}.log"

{
  echo "[START] $(date -Is) host=$(hostname)"
  echo "[SRC]  ${SRC}"
  echo "[DEST] ${DEST}"
  rsync -a --update --partial --stats --human-readable \
    --exclude='.snakemake/' \
    --exclude='slurm-logs/' \
    --exclude='*.tmp' \
    "${SRC}" "${DEST}"
  rc=$?
  echo "[DONE]  $(date -Is) status=${rc}"
} >>"${log}" 2>&1

cp -p "${log}" "${DURABLE_LOG_DIR}/" 2>/dev/null || true
exit "${rc}"
