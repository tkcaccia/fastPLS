#!/usr/bin/env bash

set -euo pipefail

RUN_ROOT="${FASTPLS_WAIT_RUN_ROOT:?Set FASTPLS_WAIT_RUN_ROOT to the full pipeline run root}"
REPO_ROOT="${FASTPLS_WAIT_REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
INTERVAL_SEC="${FASTPLS_WAIT_INTERVAL_SEC:-1800}"
RESULTS_DIR="${FASTPLS_PKG_COMPARE_RESULTS_DIR:-${RUN_ROOT}/pls_package_comparison}"
LOG_DIR="${RUN_ROOT}/logs"
LOG_FILE="${LOG_DIR}/package_comparison_waiter.log"
PID_FILE="${RUN_ROOT}.package_comparison_waiter.pid"

mkdir -p "${LOG_DIR}"
echo "$$" >"${PID_FILE}"

log_msg() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >>"${LOG_FILE}"
}

is_complete() {
  grep -q "Full variable benchmark completed" "${RUN_ROOT}/launcher.log" 2>/dev/null ||
    grep -q "Full variable benchmark completed" "${RUN_ROOT}/logs/"*.log 2>/dev/null
}

log_msg "waiter started run_root=${RUN_ROOT} repo_root=${REPO_ROOT} interval=${INTERVAL_SEC}s"

while :; do
  if is_complete; then
    log_msg "pipeline complete; starting package comparison"
    export FASTPLS_BENCH_LIB="${RUN_ROOT}/Rlib"
    export FASTPLS_PKG_COMPARE_RESULTS_DIR="${RESULTS_DIR}"
    export FASTPLS_PKG_COMPARE_DATASETS="${FASTPLS_PKG_COMPARE_DATASETS:-singlecell,nmr}"
    export FASTPLS_PKG_COMPARE_REPS="${FASTPLS_PKG_COMPARE_REPS:-3}"
    export FASTPLS_PKG_COMPARE_TIMEOUT_SEC="${FASTPLS_PKG_COMPARE_TIMEOUT_SEC:-1200}"
    bash "${REPO_ROOT}/scripts/remote_run_pls_package_comparison.sh" \
      >"${LOG_DIR}/pls_package_comparison.log" 2>&1
    status=$?
    log_msg "package comparison finished status=${status} results=${RESULTS_DIR}"
    rm -f "${PID_FILE}"
    exit "${status}"
  fi

  rows=0
  raw="${RUN_ROOT}/real_datasets/dataset_memory_compare_raw.csv"
  if [ -f "${raw}" ]; then
    rows="$(python3 - "${raw}" <<'PY'
import csv, sys
try:
    print(sum(1 for _ in csv.DictReader(open(sys.argv[1], newline=""))))
except Exception:
    print(0)
PY
)"
  fi
  log_msg "pipeline not complete yet; rows=${rows}; sleeping ${INTERVAL_SEC}s"
  sleep "${INTERVAL_SEC}"
done
