#!/usr/bin/env bash

set -eu

RUN_ROOT="${1:-$HOME/fastPLS_publication_benchmarks_20260722_score_reuse}"
REPO_ROOT="${FASTPLS_REPO_ROOT:-$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)}"
POLL_SEC="${FASTPLS_WATCH_INTERVAL_SEC:-300}"
PROGRESS_FILE="${RUN_ROOT}/progress.log"
TASK_FILE="${RUN_ROOT}/pipeline1/real_datasets/nmr_task.rds"
OUTPUT_DIR="${RUN_ROOT}/publication_figures/nmr"
LIB_LOC="${RUN_ROOT}/pipeline1/Rlib"

while ! grep -q "PUBLICATION BENCHMARK SUITE COMPLETE" "${PROGRESS_FILE}" 2>/dev/null; do
  sleep "${POLL_SEC}"
done

if [ ! -f "${TASK_FILE}" ]; then
  printf '[%s] NMR task missing: %s\n' "$(date '+%F %T')" "${TASK_FILE}" >&2
  exit 1
fi

printf '[%s] Starting NMR publication spectrum figure\n' "$(date '+%F %T')"
Rscript "${REPO_ROOT}/benchmark/plot_nmr_spectrum_prediction.R" \
  "${TASK_FILE}" "${OUTPUT_DIR}" 100 cuda "${LIB_LOC}"
printf '[%s] Finished NMR publication spectrum figure\n' "$(date '+%F %T')"
