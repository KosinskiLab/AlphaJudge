#!/usr/bin/env bash
# Cron guard for the unattended pairmatched watchdog. If the watchdog process
# dies, restart it in a detached screen. To intentionally disable auto-restart:
#   touch /scratch/dima/benchmark_26/DISABLE_WATCHDOG_20260506
set -uo pipefail

REPO="${REPO:-/g/kosinski/dima/PycharmProjects/AlphaJudge}"
WATCHDOG="${WATCHDOG:-${REPO}/scripts/watchdog_unattended_20260506.sh}"
SCREEN_NAME="${SCREEN_NAME:-watchdog_unattended_20260506}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/dima/benchmark_26}"
DURABLE_LOG_DIR="${DURABLE_LOG_DIR:-/g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26/watchdog_logs_20260506}"
DISABLE_FILE="${DISABLE_FILE:-${SCRATCH_ROOT}/DISABLE_WATCHDOG_20260506}"
LOG="${LOG:-${SCRATCH_ROOT}/watchdog_ensure_20260506.log}"

mkdir -p "${SCRATCH_ROOT}" "${DURABLE_LOG_DIR}"

log() {
  local message
  message="$(date -Is) $*"
  echo "${message}" >>"${LOG}"
  echo "${message}" >>"${DURABLE_LOG_DIR}/watchdog_ensure_20260506.log" 2>/dev/null || true
}

if [[ -e "${DISABLE_FILE}" ]]; then
  log "[DISABLED] ${DISABLE_FILE} exists; not restarting watchdog"
  exit 0
fi

if pgrep -f "${WATCHDOG}" >/dev/null 2>&1; then
  log "[OK] watchdog already running"
  exit 0
fi

screen -S "${SCREEN_NAME}" -X quit >/dev/null 2>&1 || true
sleep 2
screen -dmS "${SCREEN_NAME}" bash -lc "${WATCHDOG}"
log "[RESTART] launched ${SCREEN_NAME} with ${WATCHDOG}"
