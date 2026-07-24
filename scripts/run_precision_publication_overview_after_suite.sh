#!/usr/bin/env bash

set -eu

RUN_ROOT="${1:-$HOME/fastPLS_publication_benchmarks_20260722_score_reuse}"
REPO_ROOT="${FASTPLS_REPO_ROOT:-$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)}"
POLL_SEC="${FASTPLS_WATCH_INTERVAL_SEC:-300}"
PROGRESS_FILE="${RUN_ROOT}/progress.log"
PAIRED_FILE="${RUN_ROOT}/precision_comparison/precision_memory_paired_comparison.csv"
OUTPUT_DIR="${RUN_ROOT}/publication_figures/precision_overview"

while ! grep -q "PUBLICATION BENCHMARK SUITE COMPLETE" "${PROGRESS_FILE}" 2>/dev/null; do
  sleep "${POLL_SEC}"
done

if [ ! -f "${PAIRED_FILE}" ]; then
  printf '[%s] Matched precision table missing: %s\n' "$(date '+%F %T')" "${PAIRED_FILE}" >&2
  exit 1
fi

printf '[%s] Starting publication precision overview\n' "$(date '+%F %T')"
Rscript "${REPO_ROOT}/benchmark/plot_publication_precision_overview.R" \
  "${PAIRED_FILE}" "${OUTPUT_DIR}"
printf '[%s] Finished publication precision overview\n' "$(date '+%F %T')"
