#!/usr/bin/env bash
set -euo pipefail

TASK_RDS="${TASK_RDS:-$HOME/Documents/fastpls/data/imagenet_float32_seed123_train1000000_task.rds}"
RUNNER="${RUNNER:-$HOME/fastPLS_cmpb_cycle79/benchmark_imagenet_qualified_top5_path.R}"
OUT_ROOT="${OUT_ROOT:-/mnt/sata_ssd/fastPLS_cmpb_cycle79_imagenet_top5}"
NCOMP="${NCOMP:-100,200,300,400,500,600,700,800,900,1000}"
OVERSAMPLE="${OVERSAMPLE:-20}"
POWER="${POWER:-1}"
SEED="${SEED:-123}"
TIMEOUT_SEC="${TIMEOUT_SEC:-7200}"

mkdir -p "${OUT_ROOT}"

run_one() {
  local classifier="$1"
  local output="${OUT_ROOT}/imagenet_simpls_cuda_${classifier}_qualified_top5.csv"
  local log="${OUT_ROOT}/imagenet_simpls_cuda_${classifier}_qualified_top5.log"
  local time_log="${OUT_ROOT}/imagenet_simpls_cuda_${classifier}_qualified_top5.time"
  local gpu_log="${OUT_ROOT}/imagenet_simpls_cuda_${classifier}_gpu_trace.csv"

  printf 'timestamp,memory_used_mb,utilization_gpu_pct\n' > "${gpu_log}"
  (
    while true; do
      values="$(nvidia-smi \
        --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null || true)"
      printf '%s,%s\n' "$(date +%s.%N)" "${values// /}" >> "${gpu_log}"
      sleep 0.5
    done
  ) &
  local monitor_pid=$!

  set +e
  TASK_RDS="${TASK_RDS}" \
  OUTPUT_CSV="${output}" \
  BACKEND=cuda \
  CLASSIFIER="${classifier}" \
  NCOMP="${NCOMP}" \
  OVERSAMPLE="${OVERSAMPLE}" \
  POWER="${POWER}" \
  SEED="${SEED}" \
  /usr/bin/time -v timeout "${TIMEOUT_SEC}" Rscript "${RUNNER}" \
    > "${log}" 2> "${time_log}"
  local status=$?
  set -e

  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
  if [[ "${status}" -ne 0 ]]; then
    printf '%s failed with status %s\n' "${classifier}" "${status}" >&2
  fi
  return "${status}"
}

run_one argmax
run_one lda

printf '%s\n' "${OUT_ROOT}"
