#!/usr/bin/env bash

set -eu

RUN_ROOT="${1:-$HOME/fastPLS_publication_benchmarks_20260722_score_reuse}"
REPO_ROOT="${FASTPLS_REPO_ROOT:-$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)}"
POLL_SEC="${FASTPLS_WATCH_INTERVAL_SEC:-300}"
PROGRESS_FILE="${RUN_ROOT}/progress.log"
RAW_FILE="${RUN_ROOT}/pipeline1/real_datasets/dataset_memory_compare_raw.csv"
OUTPUT_DIR="${RUN_ROOT}/publication_figures/backend_overview"

while ! grep -q "PUBLICATION BENCHMARK SUITE COMPLETE" "${PROGRESS_FILE}" 2>/dev/null; do
  sleep "${POLL_SEC}"
done

if [ ! -f "${RAW_FILE}" ]; then
  printf '[%s] Pipeline 1 raw table missing: %s\n' "$(date '+%F %T')" "${RAW_FILE}" >&2
  exit 1
fi

printf '[%s] Starting publication backend overview\n' "$(date '+%F %T')"
Rscript "${REPO_ROOT}/benchmark/plot_publication_backend_overview.R" \
  "${RAW_FILE}" "${OUTPUT_DIR}"
printf '[%s] Finished publication backend overview\n' "$(date '+%F %T')"
