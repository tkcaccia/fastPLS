#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${FASTPLS_FULL_STANDARD_RUN_ROOT:-${REPO_ROOT}/benchmark_results_full_standard_${STAMP}}"
BENCH_PRECISION="${FASTPLS_BENCH_PRECISION:-float32}"
SOURCE_COMMIT="${FASTPLS_SOURCE_COMMIT:-$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unavailable)}"

mkdir -p "${RUN_ROOT}/logs"

LIB_LOC="${FASTPLS_BENCH_LIB:-${RUN_ROOT}/Rlib}"
mkdir -p "${LIB_LOC}"

echo "[INFO] repo=${REPO_ROOT}"
echo "[INFO] run_root=${RUN_ROOT}"
echo "[INFO] lib=${LIB_LOC}"
echo "[INFO] precision=${BENCH_PRECISION}"
echo "[INFO] real benchmark: dataset-specific component grids"
echo "[INFO] simulated benchmark: n, p, q sweeps at ncomp=${FASTPLS_SYNTH_VAR_NCOMP:-5}"

echo "[INFO] Installing fastPLS into isolated benchmark library"
if command -v nvcc >/dev/null 2>&1 && command -v nvidia-smi >/dev/null 2>&1; then
  export FASTPLS_USE_CUDA="${FASTPLS_USE_CUDA:-1}"
fi
R CMD INSTALL --preclean --library="${LIB_LOC}" "${REPO_ROOT}" >"${RUN_ROOT}/logs/install.log" 2>&1

EXISTING_R_LIBS="$(Rscript -e 'cat(paste(.libPaths(), collapse = .Platform$path.sep))')"
export R_LIBS_USER="${LIB_LOC}${EXISTING_R_LIBS:+:${EXISTING_R_LIBS}}"
export FASTPLS_BENCH_LIB="${LIB_LOC}"
export FASTPLS_BENCH_PRECISION="${BENCH_PRECISION}"

export FASTPLS_RESULTS_DIR="${FASTPLS_REAL_RESULTS_DIR:-${RUN_ROOT}/real_datasets}"
export FASTPLS_DATASETS="${FASTPLS_DATASETS:-metref,ccle,cifar100,prism,gtex_v8,tcga_pan_cancer,retina,tabula,tcga_brca,tcga_hnsc_methylation,nmr,cbmc_citeseq}"
export FASTPLS_COMPARE_REPS="${FASTPLS_COMPARE_REPS:-3}"
export FASTPLS_COMPARE_LARGE_REPS="${FASTPLS_COMPARE_LARGE_REPS:-3}"
export FASTPLS_RUN_TIMEOUT_SEC="${FASTPLS_RUN_TIMEOUT_SEC:-1200}"
export FASTPLS_SAVE_PREDICTIONS="${FASTPLS_SAVE_PREDICTIONS:-false}"
export FASTPLS_SKIP_PLOT="${FASTPLS_SKIP_PLOT:-false}"
echo "[INFO] Starting real-dataset component benchmark"
bash "${REPO_ROOT}/scripts/remote_run_dataset_memory_compare.sh" >"${RUN_ROOT}/logs/real_datasets.log" 2>&1

export OUTROOT="${FASTPLS_SYNTH_RESULTS_DIR:-${RUN_ROOT}/simulated_datasets}"
export FASTPLS_SYNTH_VAR_NCOMP="${FASTPLS_SYNTH_VAR_NCOMP:-5}"
export FASTPLS_SYNTH_VAR_REPS="${FASTPLS_SYNTH_VAR_REPS:-3}"
export FASTPLS_SYNTH_VAR_TIMEOUT_SEC="${FASTPLS_SYNTH_VAR_TIMEOUT_SEC:-1200}"
export FASTPLS_SYNTH_VAR_MAX_HOST_RSS_MB="${FASTPLS_SYNTH_VAR_MAX_HOST_RSS_MB:-10240}"
export FASTPLS_SYNTH_VAR_INCLUDE_GPU="${FASTPLS_SYNTH_VAR_INCLUDE_GPU:-true}"
export FASTPLS_SYNTH_VAR_INCLUDE_R="${FASTPLS_SYNTH_VAR_INCLUDE_R:-false}"
export FASTPLS_SYNTH_VAR_INCLUDE_PLS_PKG="${FASTPLS_SYNTH_VAR_INCLUDE_PLS_PKG:-true}"
export FASTPLS_SYNTH_VAR_FAMILIES="${FASTPLS_SYNTH_VAR_FAMILIES:-reg_n,reg_p,reg_q,class_n,class_p,class_q}"

{
  echo "timestamp=${STAMP}"
  echo "source_commit=${SOURCE_COMMIT}"
  echo "precision=${BENCH_PRECISION}"
  echo "datasets=${FASTPLS_DATASETS}"
  echo "real_reps=${FASTPLS_COMPARE_REPS}"
  echo "large_reps=${FASTPLS_COMPARE_LARGE_REPS}"
  Rscript -e 'library(fastPLS); cat("fastPLS_version=", as.character(packageVersion("fastPLS")), "\n", sep=""); cat("cuda=", has_cuda(), "\n", sep=""); cat("metal=", has_metal(), "\n", sep="")'
} >"${RUN_ROOT}/manifest.txt" 2>&1

echo "[INFO] Starting simulated n/p/q benchmark"
bash "${REPO_ROOT}/benchmark/workflow_synthetic_variable_sweeps.sh" >"${RUN_ROOT}/logs/simulated_datasets.log" 2>&1

echo "[INFO] Full standard benchmark completed"
echo "[INFO] Results: ${RUN_ROOT}"
