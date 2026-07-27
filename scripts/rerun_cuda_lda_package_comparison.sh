#!/usr/bin/env bash

# Re-run CUDA SIMPLS-LDA rows after a benchmark-adapter correction.  This is
# intentionally narrow: it preserves the completed external-package rows and
# overwrites only the previously invalid fastPLS prediction rows.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RESULTS_DIR="${FASTPLS_PKG_COMPARE_RESULTS_DIR:?Set FASTPLS_PKG_COMPARE_RESULTS_DIR}"
DATASETS="${FASTPLS_CUDA_LDA_REPAIR_DATASETS:-metref,ccle,tcga_brca,tcga_hnsc_methylation,gtex_v8,tcga_pan_cancer,retina,tabula}"

for dataset in $(printf '%s' "${DATASETS}" | tr ',' ' '); do
  FASTPLS_PKG_COMPARE_RESULTS_DIR="${RESULTS_DIR}" \
  FASTPLS_PKG_COMPARE_DATASETS="${dataset}" \
  FASTPLS_PKG_COMPARE_METHODS="fastPLS_simpls_cuda_rsvd_lda" \
  FASTPLS_PKG_COMPARE_REPS="${FASTPLS_PKG_COMPARE_REPS:-3}" \
  FASTPLS_PKG_COMPARE_TIMEOUT_SEC="${FASTPLS_PKG_COMPARE_TIMEOUT_SEC:-3600}" \
  FASTPLS_BENCH_PRECISION="${FASTPLS_BENCH_PRECISION:-float64}" \
  bash "${REPO_ROOT}/scripts/remote_run_pls_package_comparison.sh"
done
