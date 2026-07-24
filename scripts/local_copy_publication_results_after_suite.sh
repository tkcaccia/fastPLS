#!/usr/bin/env bash

set -eu

REMOTE_HOST="${1:-chiamaka@137.158.224.178}"
REMOTE_RUN_ROOT="${2:-/home/chiamaka/fastPLS_publication_benchmarks_20260722_score_reuse}"
LOCAL_DEST="${3:-$HOME/Documents/GPUPLS/fastPLS_publication_from_chiamaka_20260722_score_reuse/final}"
POLL_SEC="${FASTPLS_WATCH_INTERVAL_SEC:-300}"

while ! ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "grep -q 'PUBLICATION BENCHMARK SUITE COMPLETE' '${REMOTE_RUN_ROOT}/progress.log'" \
  2>/dev/null; do
  sleep "${POLL_SEC}"
done

# The post-suite figure watchers start only after the completion marker.
wait_count=0
while ! ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "test -f '${REMOTE_RUN_ROOT}/publication_figures/nmr/nmr_observed_predicted_spectrum.png' && test -f '${REMOTE_RUN_ROOT}/publication_figures/backend_overview/publication_backend_overview.png' && test -f '${REMOTE_RUN_ROOT}/publication_figures/cknn_case_study/publication_cknn_case_study.png' && test -f '${REMOTE_RUN_ROOT}/publication_figures/precision_overview/publication_precision_overview.png' && test -f '${REMOTE_RUN_ROOT}/supplementary_kernel_sensitivity/plots/kernel_sensitivity_classification.png' && test -f '${REMOTE_RUN_ROOT}/supplementary_kernel_sensitivity/plots/kernel_sensitivity_regression.png'" \
  2>/dev/null; do
  wait_count=$((wait_count + 1))
  if [ "${wait_count}" -ge 120 ]; then
    printf 'Timed out waiting for post-suite publication figures. Copying completed artifacts.\n' >&2
    break
  fi
  sleep "${POLL_SEC}"
done

mkdir -p "${LOCAL_DEST}"
rsync -az --prune-empty-dirs \
  --exclude='**/Rlib/***' \
  --exclude='**/run_rows/***' \
  --exclude='**/gpu_logs/***' \
  --exclude='**/logs/***' \
  --exclude='*.rds' \
  --include='*/' \
  --include='*.csv' \
  --include='*.png' \
  --include='*.pdf' \
  --include='*.md' \
  --include='*.txt' \
  --include='*.json' \
  --include='progress.log' \
  --exclude='*' \
  "${REMOTE_HOST}:${REMOTE_RUN_ROOT}/" "${LOCAL_DEST}/"

find "${LOCAL_DEST}" -type f -print | sort > "${LOCAL_DEST}/copied_files.txt"
printf 'Publication artifacts copied to %s\n' "${LOCAL_DEST}"
