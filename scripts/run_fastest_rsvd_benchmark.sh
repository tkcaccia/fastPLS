#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OPENBLAS_ROOT="${OPENBLAS_ROOT:-/opt/homebrew/opt/openblas}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

N="${N:-3000}"
P="${P:-800}"
K="${K:-10}"
SIGNAL_RANK="${SIGNAL_RANK:-10}"
REPS="${REPS:-3}"

cd "${REPO_ROOT}"

echo "Running large RSVD benchmark"
echo "  repo: ${REPO_ROOT}"
echo "  openblas: ${OPENBLAS_ROOT}"
echo "  OPENBLAS_NUM_THREADS: ${OPENBLAS_NUM_THREADS}"
echo "  n=${N} p=${P} k=${K} signal_rank=${SIGNAL_RANK} reps=${REPS}"

OPENBLAS_ROOT="${OPENBLAS_ROOT}" \
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
Rscript benchmark/benchmark_large_rsvd.R \
  --n="${N}" \
  --p="${P}" \
  --k="${K}" \
  --signal-rank="${SIGNAL_RANK}" \
  --reps="${REPS}" \
  --use-openblas=1 \
  --openblas-threads="${OPENBLAS_NUM_THREADS}"

LATEST_CSV="$(ls -1t benchmark/results/large-rsvd-*.csv | head -n 1)"
echo
echo "Latest result:"
echo "  ${REPO_ROOT}/${LATEST_CSV}"
