#!/usr/bin/env bash
set -u

REPO_ROOT="${REPO_ROOT:-$HOME/fastPLS_cmpb_cycle79}"
FASTPLS_LIB="${FASTPLS_LIB:-$HOME/fastPLS_cmpb_cycle80/Rlib}"
TASK_RDS="${TASK_RDS:-$HOME/Documents/fastpls/data/imagenet_float32_seed123_train1000000_task.rds}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/sata_ssd/fastPLS_cmpb_cycle80_imagenet_float32_simpls_lda_path}"
NCOMP_GRID="${NCOMP_GRID:-100,200,300,400,500,600,700,800,900,1000}"
OVERSAMPLE="${OVERSAMPLE:-20}"
POWER="${POWER:-2}"
SEED="${SEED:-123}"
BLOCK_SIZE="${BLOCK_SIZE:-10000}"
TIMEOUT_SEC="${TIMEOUT_SEC:-10000}"
SOURCE_ARCHIVE_SHA256="${SOURCE_ARCHIVE_SHA256:-}"

mkdir -p "${OUTPUT_DIR}"
output_csv="${OUTPUT_DIR}/imagenet_float32_simpls_lda_path.csv"
time_log="${OUTPUT_DIR}/imagenet_float32_simpls_lda_path.time"
gpu_log="${OUTPUT_DIR}/imagenet_float32_simpls_lda_path_gpu_trace.csv"

(
  printf '%s\n' "timestamp,memory_used_mb,utilization_gpu_pct"
  while :; do
    now="$(date --iso-8601=seconds)"
    sample="$(nvidia-smi \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null | head -1)"
    [ -n "${sample}" ] && printf '%s,%s\n' "${now}" "${sample}"
    sleep 2
  done
) >"${gpu_log}" &
monitor_pid=$!

set +e
FASTPLS_LIB="${FASTPLS_LIB}" \
TASK_RDS="${TASK_RDS}" \
OUTPUT_CSV="${output_csv}" \
NCOMP_GRID="${NCOMP_GRID}" \
OVERSAMPLE="${OVERSAMPLE}" \
POWER="${POWER}" \
SEED="${SEED}" \
BLOCK_SIZE="${BLOCK_SIZE}" \
SOURCE_ARCHIVE_SHA256="${SOURCE_ARCHIVE_SHA256}" \
  /usr/bin/time -v timeout --signal=TERM --kill-after=30s "${TIMEOUT_SEC}" \
    Rscript "${REPO_ROOT}/benchmark_imagenet_float32_simpls_lda_path.R" \
    >"${time_log}" 2>&1
exit_status=$?
set -e

kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true
printf '%s\n' "${exit_status}" >"${OUTPUT_DIR}/exit_status.txt"
exit "${exit_status}"
