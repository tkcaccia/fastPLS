#!/usr/bin/env bash

set -euo pipefail

SCRIPT_ROOT="${SCRIPT_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_ROOT}/.." && pwd)}"
OUTROOT="${OUTROOT:-${REPO_ROOT}/benchmark_results_synthetic_variable_sweeps}"

if ! command -v Rscript >/dev/null 2>&1; then
  echo "ERROR: Rscript not found in PATH"
  exit 1
fi

if [ ! -f "${SCRIPT_ROOT}/benchmark_synthetic_variable_sweeps.R" ]; then
  echo "ERROR: missing ${SCRIPT_ROOT}/benchmark_synthetic_variable_sweeps.R"
  exit 1
fi

if [ ! -f "${SCRIPT_ROOT}/plot_synthetic_variable_sweeps.R" ]; then
  echo "ERROR: missing ${SCRIPT_ROOT}/plot_synthetic_variable_sweeps.R"
  exit 1
fi

mkdir -p "${OUTROOT}"
cd "${REPO_ROOT}"

export FASTPLS_SYNTH_VAR_OUTDIR="${OUTROOT}"
export FASTPLS_SYNTH_VAR_REPS="${FASTPLS_SYNTH_VAR_REPS:-3}"
export FASTPLS_SYNTH_VAR_NCOMP="${FASTPLS_SYNTH_VAR_NCOMP:-20}"
export FASTPLS_SYNTH_VAR_TIMEOUT_SEC="${FASTPLS_SYNTH_VAR_TIMEOUT_SEC:-1200}"
export FASTPLS_SYNTH_VAR_MAX_HOST_RSS_MB="${FASTPLS_SYNTH_VAR_MAX_HOST_RSS_MB:-10240}"
export FASTPLS_SYNTH_VAR_INCLUDE_GPU="${FASTPLS_SYNTH_VAR_INCLUDE_GPU:-true}"
export FASTPLS_SYNTH_VAR_INCLUDE_R="${FASTPLS_SYNTH_VAR_INCLUDE_R:-false}"
export FASTPLS_SYNTH_VAR_INCLUDE_PLS_PKG="${FASTPLS_SYNTH_VAR_INCLUDE_PLS_PKG:-true}"
export FASTPLS_SYNTH_VAR_INCLUDE_CLASSIFICATION="${FASTPLS_SYNTH_VAR_INCLUDE_CLASSIFICATION:-true}"
export FASTPLS_SYNTH_VAR_FAMILIES="${FASTPLS_SYNTH_VAR_FAMILIES:-reg_n,reg_p,reg_q,class_n,class_p}"
export FASTPLS_MEASURE_MEMORY="${FASTPLS_MEASURE_MEMORY:-true}"
export FASTPLS_MEMORY_SAMPLE_SEC="${FASTPLS_MEMORY_SAMPLE_SEC:-0.2}"

echo "[INFO] Synthetic variable-sweep benchmark"
echo "[INFO] OUTROOT=${OUTROOT}"
echo "[INFO] FAMILIES=${FASTPLS_SYNTH_VAR_FAMILIES}"
echo "[INFO] REPS=${FASTPLS_SYNTH_VAR_REPS}"
echo "[INFO] NCOMP=${FASTPLS_SYNTH_VAR_NCOMP}"
echo "[INFO] TIMEOUT_SEC=${FASTPLS_SYNTH_VAR_TIMEOUT_SEC}"
echo "[INFO] MAX_HOST_RSS_MB=${FASTPLS_SYNTH_VAR_MAX_HOST_RSS_MB}"

Rscript "${SCRIPT_ROOT}/benchmark_synthetic_variable_sweeps.R"
Rscript "${SCRIPT_ROOT}/plot_synthetic_variable_sweeps.R" "${OUTROOT}"

echo "[INFO] Completed synthetic variable-sweep benchmark in ${OUTROOT}"
