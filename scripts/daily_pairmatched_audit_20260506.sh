#!/usr/bin/env bash
# Daily lightweight audit of pairmatched prediction completion.
# Writes audit/missing-list snapshots to scratch and mirrors them to /g/transform
# so progress can be inspected even if scratch is cleaned.
set -uo pipefail

REPO="${REPO:-/g/kosinski/dima/PycharmProjects/AlphaJudge}"
PYTHON_BIN="${PYTHON_BIN:-/home/dmolodenskiy/.conda/envs/snake/bin/python3.12}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/dima/benchmark_26}"
DURABLE_ROOT="${DURABLE_ROOT:-/g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26/audits_20260506}"
LOG_DIR="${LOG_DIR:-${SCRATCH_ROOT}/audit_logs}"

mkdir -p "${LOG_DIR}" "${SCRATCH_ROOT}" "${DURABLE_ROOT}"

stamp="$(date +%Y%m%d_%H%M%S)"
out_csv="${SCRATCH_ROOT}/pairmatched_audit_${stamp}.tsv"
missing_dir="${SCRATCH_ROOT}/missing_pairmatched_${stamp}"
log="${LOG_DIR}/daily_pairmatched_audit_${stamp}.log"

{
  echo "[START] $(date -Is) host=$(hostname)"
  echo "[OUT] ${out_csv}"
  echo "[MISSING] ${missing_dir}"
  "${PYTHON_BIN}" "${REPO}/scripts/audit_pairmatched_predictions.py" \
    --out-csv "${out_csv}" \
    --write-missing-dir "${missing_dir}"
  rc=$?
  echo "[DONE] $(date -Is) status=${rc}"
} >>"${log}" 2>&1

cp -p "${out_csv}" "${DURABLE_ROOT}/" 2>/dev/null || true
rsync -a "${missing_dir}/" "${DURABLE_ROOT}/$(basename "${missing_dir}")/" 2>/dev/null || true
cp -p "${log}" "${DURABLE_ROOT}/" 2>/dev/null || true

exit "${rc}"
