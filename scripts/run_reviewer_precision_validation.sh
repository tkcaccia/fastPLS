#!/usr/bin/env bash

# Reproducible reviewer validation: matched float64/float32 runs on one
# classification and one multivariate-regression task. The standard isolated
# runner records timing, RSS, GPU samples, requested/executed methods, and
# failures without dropping unsupported component counts.

set -u

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${FASTPLS_REVIEW_PRECISION_RUN_ROOT:-${REPO_ROOT}/benchmark_results_reviewer_precision_${STAMP}}"
BENCH_LIB="${FASTPLS_BENCH_LIB:-${RUN_ROOT}/Rlib}"
SOURCE_COMMIT="${FASTPLS_SOURCE_COMMIT:-$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unavailable)}"
SOURCE_TREE="$(git -C "${REPO_ROOT}" rev-parse HEAD^{tree} 2>/dev/null || echo unavailable)"
SOURCE_TAG="$(git -C "${REPO_ROOT}" describe --tags --exact-match HEAD 2>/dev/null || echo untagged)"
WORKTREE_STATE="clean"
if ! git -C "${REPO_ROOT}" diff --quiet || ! git -C "${REPO_ROOT}" diff --cached --quiet; then
  WORKTREE_STATE="modified"
fi
DATASETS="${FASTPLS_REVIEW_DATASETS:-cifar100,nmr}"
VARIANTS="${FASTPLS_REVIEW_VARIANTS:-cpp_plssvd_cpu_rsvd,gpu_plssvd_rsvd,cpp_simpls_cpu_rsvd,gpu_simpls_rsvd,cpp_opls_cpu_rsvd,gpu_opls_rsvd,cpp_kernelpls_cpu_rsvd,gpu_kernelpls_rsvd}"
REPS="${FASTPLS_REVIEW_REPS:-3}"
TIMEOUT_SEC="${FASTPLS_REVIEW_TIMEOUT_SEC:-1200}"

mkdir -p "${RUN_ROOT}" "${BENCH_LIB}"

R CMD INSTALL --preclean --library="${BENCH_LIB}" "${REPO_ROOT}"
EXISTING_R_LIBS="$(Rscript -e 'cat(paste(.libPaths(), collapse=.Platform$path.sep))')"
export R_LIBS_USER="${BENCH_LIB}${EXISTING_R_LIBS:+:${EXISTING_R_LIBS}}"

{
  echo "benchmark=matched_float32_float64_reviewer_validation"
  echo "started=$(date --iso-8601=seconds 2>/dev/null || date)"
  echo "source_commit=${SOURCE_COMMIT}"
  echo "source_tree=${SOURCE_TREE}"
  echo "source_tag=${SOURCE_TAG}"
  echo "worktree_state=${WORKTREE_STATE}"
  echo "datasets=${DATASETS}"
  echo "variants=${VARIANTS}"
  echo "replicates=${REPS}"
  echo "timeout_sec=${TIMEOUT_SEC}"
  hostname
  uname -a
  free -h 2>/dev/null || vm_stat 2>/dev/null || true
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
  Rscript -e 'library(fastPLS); cat("fastPLS=", as.character(packageVersion("fastPLS")), " cuda=", has_cuda(), " metal=", has_metal(), "\\n", sep="")'
} > "${RUN_ROOT}/manifest.txt" 2>&1

for precision in float64 float32; do
  out_dir="${RUN_ROOT}/${precision}"
  FASTPLS_RESULTS_DIR="${out_dir}" \
  FASTPLS_BENCH_LIB="${BENCH_LIB}" \
  FASTPLS_DATASETS="${DATASETS}" \
  FASTPLS_CIFAR100_NCOMP_LIST="${FASTPLS_REVIEW_CIFAR100_NCOMP:-100}" \
  FASTPLS_NMR_NCOMP_LIST="${FASTPLS_REVIEW_NMR_NCOMP:-100}" \
  FASTPLS_COMPARE_REPS="${REPS}" \
  FASTPLS_COMPARE_LARGE_REPS="${REPS}" \
  FASTPLS_BENCH_PRECISION="${precision}" \
  FASTPLS_VARIANTS="${VARIANTS}" \
  FASTPLS_RUN_TIMEOUT_SEC="${TIMEOUT_SEC}" \
  bash "${REPO_ROOT}/scripts/remote_run_dataset_memory_compare.sh" \
    > "${RUN_ROOT}/${precision}.log" 2>&1
done

Rscript "${REPO_ROOT}/benchmark/summarize_precision_memory_comparison.R" \
  "${RUN_ROOT}/float32/dataset_memory_compare_raw.csv" \
  "${RUN_ROOT}/float64/dataset_memory_compare_raw.csv" \
  "${RUN_ROOT}/precision_summary"

echo "finished=$(date --iso-8601=seconds 2>/dev/null || date)" >> "${RUN_ROOT}/manifest.txt"
echo "${RUN_ROOT}"
