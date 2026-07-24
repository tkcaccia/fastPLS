#!/usr/bin/env bash

set -eu

RUN_ROOT="${1:-$HOME/fastPLS_publication_benchmarks_20260722_score_reuse}"
REPO_ROOT="${FASTPLS_REPO_ROOT:-$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)}"
POLL_SEC="${FASTPLS_WATCH_INTERVAL_SEC:-300}"
PROGRESS_FILE="${RUN_ROOT}/progress.log"
PIPELINE1_RAW="${RUN_ROOT}/pipeline1/real_datasets/dataset_memory_compare_raw.csv"
PIPELINE4_SUMMARY="${RUN_ROOT}/pipeline4/results/pipeline4_imagenet_summary.csv"
OUTPUT_DIR="${RUN_ROOT}/publication_figures/cknn_case_study"

while ! grep -q "PUBLICATION BENCHMARK SUITE COMPLETE" "${PROGRESS_FILE}" 2>/dev/null; do
  sleep "${POLL_SEC}"
done

for file in "${PIPELINE1_RAW}" "${PIPELINE4_SUMMARY}"; do
  if [ ! -f "${file}" ]; then
    printf '[%s] Required cKNN input missing: %s\n' "$(date '+%F %T')" "${file}" >&2
    exit 1
  fi
done

printf '[%s] Starting publication cKNN case-study figure\n' "$(date '+%F %T')"
Rscript "${REPO_ROOT}/benchmark/plot_publication_cknn_case_study.R" \
  "${PIPELINE1_RAW}" "${PIPELINE4_SUMMARY}" "${OUTPUT_DIR}"
printf '[%s] Finished publication cKNN case-study figure\n' "$(date '+%F %T')"
