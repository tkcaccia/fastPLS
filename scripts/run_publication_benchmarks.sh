#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${FASTPLS_PUBLICATION_RUN_ROOT:-$HOME/fastPLS_publication_benchmarks_${STAMP}}"
LOG_DIR="${RUN_ROOT}/logs"
PIPELINE1_ROOT="${RUN_ROOT}/pipeline1"
BENCH_LIB="${PIPELINE1_ROOT}/Rlib"
IMAGENET_SOURCE="${FASTPLS_IMAGENET_FLOAT32_RDATA:-$HOME/Documents/fastpls/data/imagenet_float32.RData}"
IMAGENET_TASK="${FASTPLS_IMAGENET_FLOAT32_TASK:-$HOME/Documents/fastpls/data/imagenet_float32_seed123_train1000000_task.rds}"

mkdir -p "${RUN_ROOT}" "${LOG_DIR}"

run_stage() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START ${name}" | tee -a "${RUN_ROOT}/progress.log"
  "$@" >"${LOG_DIR}/${name}.log" 2>&1
  local status=$?
  echo "[$(date '+%F %T')] END ${name} status=${status}" | tee -a "${RUN_ROOT}/progress.log"
  printf '%s,%s,%s\n' "${name}" "${status}" "$(date --iso-8601=seconds 2>/dev/null || date)" >> "${RUN_ROOT}/stage_status.csv"
  return 0
}

{
  echo "stage,status,finished_at"
} > "${RUN_ROOT}/stage_status.csv"

{
  echo "benchmark_suite=fastPLS publication Pipelines 1-4"
  echo "started=$(date --iso-8601=seconds 2>/dev/null || date)"
  echo "repo_root=${REPO_ROOT}"
  echo "run_root=${RUN_ROOT}"
  echo "commit=$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unavailable)"
  echo "working_tree_changes=$(git -C "${REPO_ROOT}" status --porcelain | wc -l | tr -d ' ')"
  echo "hostname=$(hostname)"
  uname -a
  free -h 2>/dev/null || vm_stat 2>/dev/null || true
  lscpu 2>/dev/null || sysctl -n machdep.cpu.brand_string 2>/dev/null || true
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
  Rscript -e 'cat(R.version.string, "\n"); cat("BLAS=", extSoftVersion()[["BLAS"]], "\n", sep="")' 2>/dev/null || true
} > "${RUN_ROOT}/manifest.txt" 2>&1

run_stage pipeline1 env \
  FASTPLS_FULL_STANDARD_RUN_ROOT="${PIPELINE1_ROOT}" \
  FASTPLS_BENCH_LIB="${BENCH_LIB}" \
  FASTPLS_BENCH_PRECISION=float32 \
  FASTPLS_COMPARE_REPS=3 \
  FASTPLS_COMPARE_LARGE_REPS=3 \
  FASTPLS_SYNTH_VAR_REPS=3 \
  FASTPLS_SYNTH_VAR_NCOMP=5 \
  FASTPLS_RUN_TIMEOUT_SEC=1200 \
  FASTPLS_SYNTH_VAR_TIMEOUT_SEC=1200 \
  FASTPLS_USE_CUDA=1 \
  bash "${REPO_ROOT}/benchmark/run_pipeline1_real_and_simulated.sh"

if [ ! -d "${BENCH_LIB}/fastPLS" ]; then
  mkdir -p "${BENCH_LIB}"
  run_stage install_fallback env FASTPLS_USE_CUDA=1 \
    R CMD INSTALL --preclean --library="${BENCH_LIB}" "${REPO_ROOT}"
fi
EXISTING_R_LIBS="$(Rscript -e 'cat(paste(.libPaths(), collapse=.Platform$path.sep))')"
export R_LIBS_USER="${BENCH_LIB}${EXISTING_R_LIBS:+:${EXISTING_R_LIBS}}"
export FASTPLS_BENCH_LIB="${BENCH_LIB}"

PRECISION64_DIR="${RUN_ROOT}/precision_float64"
PRECISION_VARIANTS="cpp_plssvd_cpu_rsvd,cpp_plssvd_irlba,gpu_plssvd_rsvd"
PRECISION_VARIANTS="${PRECISION_VARIANTS},cpp_simpls_cpu_rsvd,cpp_simpls_irlba,gpu_simpls_rsvd"
PRECISION_VARIANTS="${PRECISION_VARIANTS},cpp_opls_cpu_rsvd,cpp_opls_irlba,gpu_opls_rsvd"
PRECISION_VARIANTS="${PRECISION_VARIANTS},cpp_kernelpls_cpu_rsvd,cpp_kernelpls_irlba,gpu_kernelpls_rsvd"
PRECISION_VARIANTS="${PRECISION_VARIANTS},cpp_plssvd_cpu_rsvd_lda,cpp_plssvd_irlba_lda,gpu_plssvd_rsvd_lda"
PRECISION_VARIANTS="${PRECISION_VARIANTS},cpp_simpls_cpu_rsvd_lda,cpp_simpls_irlba_lda,gpu_simpls_rsvd_lda"
PRECISION_VARIANTS="${PRECISION_VARIANTS},cpp_opls_cpu_rsvd_lda,cpp_opls_irlba_lda,gpu_opls_rsvd_lda"
PRECISION_VARIANTS="${PRECISION_VARIANTS},cpp_kernelpls_cpu_rsvd_lda,cpp_kernelpls_irlba_lda,gpu_kernelpls_rsvd_lda"
run_stage precision_float64 env \
  FASTPLS_RESULTS_DIR="${PRECISION64_DIR}" \
  FASTPLS_BENCH_LIB="${BENCH_LIB}" \
  FASTPLS_BENCH_PRECISION=float64 \
  FASTPLS_DATASETS="metref,cifar100,retina,tabula,nmr" \
  FASTPLS_METREF_NCOMP_LIST=22 \
  FASTPLS_CIFAR100_NCOMP_LIST=100 \
  FASTPLS_SMALL_MULTI_NCOMP_LIST=50 \
  FASTPLS_NMR_NCOMP_LIST=100 \
  FASTPLS_COMPARE_REPS=3 \
  FASTPLS_COMPARE_LARGE_REPS=3 \
  FASTPLS_RUN_TIMEOUT_SEC=3600 \
  FASTPLS_VARIANTS="${PRECISION_VARIANTS}" \
  bash "${REPO_ROOT}/scripts/remote_run_dataset_memory_compare.sh"

FLOAT32_RAW="${PIPELINE1_ROOT}/real_datasets/dataset_memory_compare_raw.csv"
FLOAT64_RAW="${PRECISION64_DIR}/dataset_memory_compare_raw.csv"
if [ -f "${FLOAT32_RAW}" ] && [ -f "${FLOAT64_RAW}" ]; then
  run_stage precision_summary Rscript \
    "${REPO_ROOT}/benchmark/summarize_precision_memory_comparison.R" \
    "${FLOAT32_RAW}" "${FLOAT64_RAW}" "${RUN_ROOT}/precision_comparison"
fi

run_stage pipeline2 env \
  FASTPLS_PKG_COMPARE_RESULTS_DIR="${RUN_ROOT}/pipeline2" \
  FASTPLS_BENCH_LIB="${BENCH_LIB}" \
  FASTPLS_BENCH_PRECISION=float32 \
  FASTPLS_PKG_COMPARE_REPS=3 \
  FASTPLS_PKG_COMPARE_TIMEOUT_SEC=3600 \
  bash "${REPO_ROOT}/benchmark/run_pipeline2_package_comparison.sh"

# Pipeline 3 uses the compiled native CV engines. The paired precision benchmark
# above provides the strict float32 memory comparison without replacing native
# CV timing with an R-level fold loop.
run_stage pipeline3 env \
  FASTPLS_PIPELINE3_RESULTS_DIR="${RUN_ROOT}/pipeline3" \
  FASTPLS_BENCH_LIB="${BENCH_LIB}" \
  FASTPLS_BENCH_PRECISION=float64 \
  FASTPLS_PIPELINE3_REPS=3 \
  FASTPLS_PIPELINE3_KFOLD=10 \
  FASTPLS_PIPELINE3_TIMEOUT_SEC=3600 \
  bash "${REPO_ROOT}/benchmark/run_pipeline3_cv_vs_fit.sh"

if [ ! -f "${IMAGENET_TASK}" ]; then
  if [ -f "${IMAGENET_SOURCE}" ]; then
    run_stage prepare_imagenet_float32 Rscript \
      "${REPO_ROOT}/benchmark/prepare_imagenet_float32_task.R" \
      "${IMAGENET_SOURCE}" "${IMAGENET_TASK}" 1000000 123
  else
    echo "[$(date '+%F %T')] SKIP Pipeline 4: missing ${IMAGENET_SOURCE}" | tee -a "${RUN_ROOT}/progress.log"
  fi
fi

if [ -f "${IMAGENET_TASK}" ]; then
  run_stage pipeline4 env \
    RUN_ROOT="${RUN_ROOT}/pipeline4" \
    FASTPLS_REPO_ROOT="${REPO_ROOT}" \
    FASTPLS_LIB="${BENCH_LIB}" \
    TASK_RDS="${IMAGENET_TASK}" \
    NCOMP_GRID="100 200 300 400 500 600 700 800 900 1000" \
    BACKENDS="cpu cuda" \
    CLASSIFIERS="argmax lda cknn" \
    REPS=1 \
    TIMEOUT_SEC=10000 \
    bash "${REPO_ROOT}/benchmark/run_pipeline4_imagenet_simpls_rsvd.sh"
fi

echo "finished=$(date --iso-8601=seconds 2>/dev/null || date)" >> "${RUN_ROOT}/manifest.txt"
echo "[$(date '+%F %T')] PUBLICATION BENCHMARK SUITE COMPLETE" | tee -a "${RUN_ROOT}/progress.log"
echo "${RUN_ROOT}"
