#!/usr/bin/env bash

set -euo pipefail

SCRIPT_ROOT="${SCRIPT_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_ROOT}/.." && pwd)}"
OUTROOT="${OUTROOT:-${REPO_ROOT}/benchmark_results_synthetic}"

if ! command -v Rscript >/dev/null 2>&1; then
  echo "ERROR: Rscript not found in PATH"
  exit 1
fi

if [ ! -f "${SCRIPT_ROOT}/hpc_full_multianalysis_benchmark.R" ]; then
  echo "ERROR: missing ${SCRIPT_ROOT}/hpc_full_multianalysis_benchmark.R"
  exit 1
fi

if [ ! -f "${SCRIPT_ROOT}/hpc_full_multianalysis_plots.R" ]; then
  echo "ERROR: missing ${SCRIPT_ROOT}/hpc_full_multianalysis_plots.R"
  exit 1
fi

mkdir -p "${OUTROOT}"
cd "${SCRIPT_ROOT}"

export FASTPLS_DATASETS="${FASTPLS_DATASETS:-synthetic_regression_base,synthetic_regression_n_small,synthetic_regression_n_large,synthetic_regression_p_wide,synthetic_regression_q_wide,synthetic_regression_rank_low,synthetic_regression_rank_high,synthetic_regression_decay_steep,synthetic_regression_decay_clustered,synthetic_regression_decay_flat,synthetic_regression_snr_high,synthetic_regression_snr_low,synthetic_regression_dropout,synthetic_classification_balanced,synthetic_classification_imbalanced,synthetic_classification_dropout}"
export FASTPLS_ONLY_NCOMP="${FASTPLS_ONLY_NCOMP:-true}"
export FASTPLS_SYNTHETIC_FIXED_NCOMP="${FASTPLS_SYNTHETIC_FIXED_NCOMP:-20}"
export FASTPLS_NCOMP_LIST="${FASTPLS_NCOMP_LIST:-${FASTPLS_SYNTHETIC_FIXED_NCOMP}}"
export FASTPLS_DEFAULT_NCOMP="${FASTPLS_DEFAULT_NCOMP:-${FASTPLS_SYNTHETIC_FIXED_NCOMP}}"
export FASTPLS_REPS="${FASTPLS_REPS:-1}"
export FASTPLS_INCLUDE_CUDA="${FASTPLS_INCLUDE_CUDA:-true}"
export FASTPLS_INCLUDE_R_IMPL="${FASTPLS_INCLUDE_R_IMPL:-false}"
export FASTPLS_INCLUDE_PLS_PKG="${FASTPLS_INCLUDE_PLS_PKG:-false}"
export FASTPLS_MULTI_APPEND=false
export FASTPLS_MULTI_OUTDIR="${OUTROOT}"

echo "[INFO] Synthetic multianalysis benchmark"
echo "[INFO] OUTROOT=${OUTROOT}"
echo "[INFO] DATASETS=${FASTPLS_DATASETS}"
echo "[INFO] FIXED_NCOMP=${FASTPLS_NCOMP_LIST}"

Rscript "${SCRIPT_ROOT}/hpc_full_multianalysis_benchmark.R"
Rscript "${SCRIPT_ROOT}/hpc_full_multianalysis_plots.R"

echo "[INFO] Completed synthetic benchmark in ${OUTROOT}"
