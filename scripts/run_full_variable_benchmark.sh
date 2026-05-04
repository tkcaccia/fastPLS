#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${FASTPLS_FULL_VARIABLE_RUN_ROOT:-${REPO_ROOT}/benchmark_results_full_variable_${STAMP}}"

mkdir -p "${RUN_ROOT}/logs"

LIB_LOC="${FASTPLS_BENCH_LIB:-${RUN_ROOT}/Rlib}"
mkdir -p "${LIB_LOC}"

echo "[INFO] repo=${REPO_ROOT}"
echo "[INFO] run_root=${RUN_ROOT}"
echo "[INFO] lib=${LIB_LOC}"

echo "[INFO] Installing fastPLS into isolated benchmark library"
R CMD INSTALL --preclean --library="${LIB_LOC}" "${REPO_ROOT}" >"${RUN_ROOT}/logs/install.log" 2>&1

export R_LIBS_USER="${LIB_LOC}${R_LIBS_USER:+:${R_LIBS_USER}}"
export FASTPLS_BENCH_LIB="${LIB_LOC}"

# Real datasets: use dataset-specific component grids from
# remote_run_dataset_memory_compare.sh unless the caller overrides them.
export FASTPLS_RESULTS_DIR="${FASTPLS_REAL_RESULTS_DIR:-${RUN_ROOT}/real_datasets}"
export FASTPLS_DATASETS="${FASTPLS_DATASETS:-metref,ccle,cifar100,prism,gtex_v8,tcga_pan_cancer,singlecell,tcga_brca,tcga_hnsc_methylation,nmr,cbmc_citeseq,imagenet}"
export FASTPLS_COMPARE_REPS="${FASTPLS_COMPARE_REPS:-3}"
export FASTPLS_RUN_TIMEOUT_SEC="${FASTPLS_RUN_TIMEOUT_SEC:-1200}"
export FASTPLS_SAVE_PREDICTIONS="${FASTPLS_SAVE_PREDICTIONS:-false}"
export FASTPLS_SKIP_PLOT="${FASTPLS_SKIP_PLOT:-false}"
export FASTPLS_SKIP_HEAVY_R="${FASTPLS_SKIP_HEAVY_R:-false}"

echo "[INFO] Starting real-dataset benchmark with dataset-specific ncomp grids"
bash "${REPO_ROOT}/scripts/remote_run_dataset_memory_compare.sh" >"${RUN_ROOT}/logs/real_datasets.log" 2>&1

# Simulated datasets: sweep n, p, and q.  By default we run both 5 and 50
# components in separate result folders, matching the previous reporting style.
export FASTPLS_SYNTH_VAR_REPS="${FASTPLS_SYNTH_VAR_REPS:-3}"
export FASTPLS_SYNTH_VAR_TIMEOUT_SEC="${FASTPLS_SYNTH_VAR_TIMEOUT_SEC:-1200}"
export FASTPLS_SYNTH_VAR_MAX_HOST_RSS_MB="${FASTPLS_SYNTH_VAR_MAX_HOST_RSS_MB:-10240}"
export FASTPLS_SYNTH_VAR_INCLUDE_GPU="${FASTPLS_SYNTH_VAR_INCLUDE_GPU:-true}"
export FASTPLS_SYNTH_VAR_INCLUDE_R="${FASTPLS_SYNTH_VAR_INCLUDE_R:-false}"
export FASTPLS_SYNTH_VAR_INCLUDE_PLS_PKG="${FASTPLS_SYNTH_VAR_INCLUDE_PLS_PKG:-true}"
export FASTPLS_SYNTH_VAR_FAMILIES="${FASTPLS_SYNTH_VAR_FAMILIES:-reg_n,reg_p,reg_q,class_n,class_p,class_q}"

SYNTH_NCOMP_LIST="${FASTPLS_SYNTH_NCOMP_LIST:-5,50}"
for synth_ncomp in $(printf '%s' "${SYNTH_NCOMP_LIST}" | tr ',' ' '); do
  export OUTROOT="${FASTPLS_SYNTH_RESULTS_DIR:-${RUN_ROOT}/simulated_datasets}/ncomp_${synth_ncomp}"
  export FASTPLS_SYNTH_VAR_NCOMP="${synth_ncomp}"
  echo "[INFO] Starting simulated benchmark ncomp=${synth_ncomp}"
  bash "${REPO_ROOT}/benchmark/workflow_synthetic_variable_sweeps.sh" >"${RUN_ROOT}/logs/simulated_ncomp_${synth_ncomp}.log" 2>&1
done

echo "[INFO] Full variable benchmark completed"
echo "[INFO] Results: ${RUN_ROOT}"
