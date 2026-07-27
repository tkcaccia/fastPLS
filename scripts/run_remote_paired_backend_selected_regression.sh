#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/fastPLS_review_cycle_20260724/src}"
LIB_LOC="${LIB_LOC:-$HOME/R/x86_64-pc-linux-gnu-library/4.5}"
OUT_ROOT="${OUT_ROOT:-$HOME/fastPLS_review_paired_backend_20260725}"
TASK_ROOT="${TASK_ROOT:-$HOME/fastPLS_cycle14_allreal_replicated_20260723b}"
RUNNER="${REPO_ROOT}/benchmark/benchmark_dataset_memory_compare.R"

mkdir -p "${OUT_ROOT}/rows" "${OUT_ROOT}/predictions"

run_one() {
  local dataset="$1"
  local task="$2"
  local variant="$3"
  local ncomp="$4"
  local stem="${dataset}__${variant}__k${ncomp}"

  Rscript "${RUNNER}" \
    --mode=run_one \
    --task_rds="${task}" \
    --row_out="${OUT_ROOT}/rows/${stem}.csv" \
    --pred_out="${OUT_ROOT}/predictions/${stem}.rds" \
    --variant_name="${variant}" \
    --lib_loc="${LIB_LOC}" \
    --requested_ncomp="${ncomp}" \
    --replicate=1
}

CBMC_TASK="${TASK_ROOT}/cbmc_citeseq/rep3/results/cbmc_citeseq_task.rds"
PRISM_TASK="${TASK_ROOT}/prism/rep3/results/prism_task.rds"

for variant in cpp_plssvd_cpu_rsvd gpu_plssvd_rsvd; do
  run_one cbmc_citeseq "${CBMC_TASK}" "${variant}" 10
done
for variant in cpp_simpls_cpu_rsvd gpu_simpls_rsvd; do
  run_one cbmc_citeseq "${CBMC_TASK}" "${variant}" 50
done
for variant in cpp_opls_cpu_rsvd gpu_opls_rsvd; do
  run_one cbmc_citeseq "${CBMC_TASK}" "${variant}" 50
done
for variant in cpp_kernelpls_cpu_rsvd gpu_kernelpls_rsvd; do
  run_one cbmc_citeseq "${CBMC_TASK}" "${variant}" 50
done

for variant in cpp_plssvd_cpu_rsvd gpu_plssvd_rsvd; do
  run_one prism "${PRISM_TASK}" "${variant}" 10
done
for variant in cpp_simpls_cpu_rsvd gpu_simpls_rsvd; do
  run_one prism "${PRISM_TASK}" "${variant}" 5
done
for variant in cpp_opls_cpu_rsvd gpu_opls_rsvd; do
  run_one prism "${PRISM_TASK}" "${variant}" 2
done
for variant in cpp_kernelpls_cpu_rsvd gpu_kernelpls_rsvd; do
  run_one prism "${PRISM_TASK}" "${variant}" 5
done

echo "${OUT_ROOT}"
