#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-${HOME}/fastPLS_xprod_irlba_20260429}"
SRC="${SRC:-${ROOT}/src}"
RLIB="${RLIB:-${ROOT}/Rlib}"
DATA_DIR="${DATA_DIR:-${ROOT}/data_combined}"
RUN_ID="${RUN_ID:-modified_xprod_irlba_$(date +%Y%m%d_%H%M%S)}"
OUTROOT="${OUTROOT:-${ROOT}/results/${RUN_ID}}"
MIN_AVAILABLE_RAM_MB="${FASTPLS_PIPELINE_MIN_AVAILABLE_RAM_MB:-12000}"
RAM_WAIT_SEC="${FASTPLS_PIPELINE_RAM_WAIT_SEC:-120}"

export R_LIBS="${RLIB}${R_LIBS:+:${R_LIBS}}"
export R_LIBS_USER="${RLIB}"
export FASTPLS_LIB_LOC="${RLIB}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

available_ram_mb() {
  awk '/MemAvailable:/ { printf "%d\n", $2 / 1024 }' /proc/meminfo
}

wait_for_ram() {
  label="$1"
  while :; do
    avail="$(available_ram_mb)"
    if [ "${avail}" -ge "${MIN_AVAILABLE_RAM_MB}" ]; then
      log "RAM gate passed for ${label}: MemAvailable=${avail} MB"
      return 0
    fi
    log "Waiting before ${label}: MemAvailable=${avail} MB < ${MIN_AVAILABLE_RAM_MB} MB"
    sleep "${RAM_WAIT_SEC}"
  done
}

prepare_data_dir() {
  mkdir -p "${DATA_DIR}"
  for f in "${HOME}/Documents/Rdatasets"/*.RData "${HOME}/Documents/Rdatasets"/*.Rdata; do
    [ -f "${f}" ] || continue
    ln -sfn "${f}" "${DATA_DIR}/$(basename "${f}")"
  done
  for f in "${HOME}/Documents/fastpls/data"/*.RData "${HOME}/Documents/fastpls/data"/*.Rdata; do
    [ -f "${f}" ] || continue
    ln -sfn "${f}" "${DATA_DIR}/$(basename "${f}")"
  done
}

write_manifest() {
  {
    printf 'run_id=%s\n' "${RUN_ID}"
    printf 'root=%s\n' "${ROOT}"
    printf 'src=%s\n' "${SRC}"
    printf 'rlib=%s\n' "${RLIB}"
    printf 'data_dir=%s\n' "${DATA_DIR}"
    printf 'outroot=%s\n' "${OUTROOT}"
    printf 'min_available_ram_mb=%s\n' "${MIN_AVAILABLE_RAM_MB}"
    printf 'hostname='
    hostname
    printf 'date='
    date -Is
    printf 'git_head='
    git -C "${SRC}" rev-parse HEAD 2>/dev/null || true
    printf 'fastPLS_lib='
    Rscript --vanilla -e "cat(system.file(package='fastPLS'))" || true
    printf '\n'
    nvidia-smi || true
  } > "${OUTROOT}/manifest.txt"
}

run_real_datasets() {
  wait_for_ram "real dataset pipeline"
  log "Starting real dataset pipeline"
  (
    cd "${SRC}"
    export R_LIBS="${RLIB}${R_LIBS:+:${R_LIBS}}"
    export R_LIBS_USER="${RLIB}"
    export FASTPLS_LIB_LOC="${RLIB}"
    export SCRIPT_ROOT="${SRC}/benchmark"
    export REPO_ROOT="${SRC}"
    export FASTPLS_DATA_ROOT="${DATA_DIR}"
    export FASTPLS_RESULTS_DIR="${OUTROOT}/real"
    export FASTPLS_BENCH_LIB="${RLIB}"
    export FASTPLS_DATASETS="${FASTPLS_DATASETS:-metref,ccle,cifar100,prism,gtex_v8,tcga_pan_cancer,singlecell,tcga_brca,tcga_hnsc_methylation,nmr,cbmc_citeseq}"
    export FASTPLS_COMPARE_REPS="${FASTPLS_COMPARE_REPS:-3}"
    export FASTPLS_RUN_TIMEOUT_SEC="${FASTPLS_RUN_TIMEOUT_SEC:-1200}"
    export FASTPLS_SAVE_PREDICTIONS="${FASTPLS_SAVE_PREDICTIONS:-false}"
    export FASTPLS_THREADS="${FASTPLS_THREADS:-1}"
    export OMP_NUM_THREADS="${FASTPLS_THREADS}"
    export OPENBLAS_NUM_THREADS="${FASTPLS_THREADS}"
    export MKL_NUM_THREADS="${FASTPLS_THREADS}"
    export VECLIB_MAXIMUM_THREADS="${FASTPLS_THREADS}"
    sh scripts/remote_run_dataset_memory_compare.sh
  ) > "${OUTROOT}/real_pipeline.log" 2>&1
  log "Completed real dataset pipeline"
}

run_synthetic_variable_sweeps() {
  ncomp="$1"
  wait_for_ram "synthetic variable sweeps ncomp=${ncomp}"
  log "Starting synthetic variable sweeps ncomp=${ncomp}"
  (
    cd "${SRC}"
    export R_LIBS="${RLIB}${R_LIBS:+:${R_LIBS}}"
    export R_LIBS_USER="${RLIB}"
    export FASTPLS_LIB_LOC="${RLIB}"
    export SCRIPT_ROOT="${SRC}/benchmark"
    export REPO_ROOT="${SRC}"
    export OUTROOT="${OUTROOT}/synthetic_variable_sweeps_ncomp_${ncomp}"
    export FASTPLS_SYNTH_VAR_OUTDIR="${OUTROOT}/synthetic_variable_sweeps_ncomp_${ncomp}"
    export FASTPLS_SYNTH_VAR_REPS="${FASTPLS_SYNTH_VAR_REPS:-1}"
    export FASTPLS_SYNTH_VAR_NCOMP="${ncomp}"
    export FASTPLS_SYNTH_VAR_TIMEOUT_SEC="${FASTPLS_SYNTH_VAR_TIMEOUT_SEC:-1200}"
    export FASTPLS_SYNTH_VAR_MAX_HOST_RSS_MB="${FASTPLS_SYNTH_VAR_MAX_HOST_RSS_MB:-10240}"
    export FASTPLS_SYNTH_VAR_INCLUDE_GPU="${FASTPLS_SYNTH_VAR_INCLUDE_GPU:-true}"
    export FASTPLS_SYNTH_VAR_INCLUDE_R="${FASTPLS_SYNTH_VAR_INCLUDE_R:-false}"
    export FASTPLS_SYNTH_VAR_INCLUDE_PLS_PKG="${FASTPLS_SYNTH_VAR_INCLUDE_PLS_PKG:-true}"
    bash benchmark/workflow_synthetic_variable_sweeps.sh
  ) > "${OUTROOT}/synthetic_variable_sweeps_ncomp_${ncomp}.log" 2>&1
  log "Completed synthetic variable sweeps ncomp=${ncomp}"
}

mkdir -p "${OUTROOT}"
prepare_data_dir
write_manifest

log "Modified xprod/IRLBA pipeline starting"
log "Results: ${OUTROOT}"
run_real_datasets
run_synthetic_variable_sweeps 5
run_synthetic_variable_sweeps 50
log "Modified xprod/IRLBA pipeline finished"
