#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"

WAIT_RESULTS_DIR="${WAIT_RESULTS_DIR:-/home/chiamaka/fastPLS_benchmark_clean/results_dataset_memory_compare_all11_custom_v10}"
POLL_SEC="${POLL_SEC:-60}"

SYNTH_OUT_DIR="${SYNTH_OUT_DIR:-/home/chiamaka/fastPLS_benchmark_clean/results_synthetic_smoke_after_v10}"
IMAGENET_OUT_DIR="${IMAGENET_OUT_DIR:-/home/chiamaka/fastPLS_benchmark_clean/results_dataset_memory_compare_imagenet_after_v10}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

wait_for_v10() {
  while pgrep -f "${WAIT_RESULTS_DIR}" >/dev/null 2>&1; do
    log "Waiting for real-data run to finish: ${WAIT_RESULTS_DIR}"
    sleep "${POLL_SEC}"
  done
}

run_synthetic_smoke() {
  log "Starting synthetic smoke benchmark"
  rm -rf "${SYNTH_OUT_DIR}"
  mkdir -p "${SYNTH_OUT_DIR}"

  FASTPLS_SYNTH_SMOKE_OUTDIR="${SYNTH_OUT_DIR}" \
  FASTPLS_SYNTH_SMOKE_REPS="${FASTPLS_SYNTH_SMOKE_REPS:-3}" \
  FASTPLS_SYNTH_SMOKE_INCLUDE_GPU="${FASTPLS_SYNTH_SMOKE_INCLUDE_GPU:-true}" \
  FASTPLS_SYNTH_SMOKE_INCLUDE_R="${FASTPLS_SYNTH_SMOKE_INCLUDE_R:-true}" \
  FASTPLS_SYNTH_SMOKE_INCLUDE_PLS_PKG="${FASTPLS_SYNTH_SMOKE_INCLUDE_PLS_PKG:-true}" \
  Rscript "${REPO_ROOT}/benchmark/benchmark_synthetic_smoke_chiamaka.R"

  FASTPLS_SYNTH_SMOKE_OUTDIR="${SYNTH_OUT_DIR}" \
  Rscript "${REPO_ROOT}/benchmark/write_synthetic_smoke_summary.R"

  FASTPLS_SYNTH_SMOKE_OUTDIR="${SYNTH_OUT_DIR}" \
  Rscript "${REPO_ROOT}/benchmark/plot_synthetic_smoke_results.R"

  log "Synthetic smoke benchmark completed: ${SYNTH_OUT_DIR}"
}

run_imagenet_timed() {
  log "Starting imagenet follow-up with timeout"
  rm -rf "${IMAGENET_OUT_DIR}"

  FASTPLS_RESULTS_DIR="${IMAGENET_OUT_DIR}" \
  FASTPLS_DATASETS="imagenet" \
  FASTPLS_COMPARE_REPS="${FASTPLS_IMAGENET_REPS:-1}" \
  FASTPLS_IMAGENET_NCOMP_LIST="${FASTPLS_IMAGENET_NCOMP_LIST:-2,5,10,20,50,100}" \
  FASTPLS_RUN_TIMEOUT_SEC="${FASTPLS_RUN_TIMEOUT_SEC:-600}" \
  FASTPLS_SAVE_PREDICTIONS="false" \
  sh "${REPO_ROOT}/scripts/remote_run_dataset_memory_compare.sh"

  log "Imagenet follow-up completed: ${IMAGENET_OUT_DIR}"
}

log "Queued follow-up started"
wait_for_v10
run_synthetic_smoke
run_imagenet_timed
log "Queued follow-up finished"
