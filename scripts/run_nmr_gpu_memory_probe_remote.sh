#!/usr/bin/env bash
set -euo pipefail

RUNNER="${RUNNER:-$HOME/fastPLS_cmpb_cycle79/benchmark_nmr_qualified_solver.R}"
INPUT="${INPUT:-$HOME/Documents/fastpls/data/nmr.RData}"
OUT_ROOT="${OUT_ROOT:-/mnt/sata_ssd/fastPLS_cmpb_cycle80_nmr_gpu_memory}"
FASTPLS_LIB="${FASTPLS_LIB:-$HOME/fastPLS_cmpb_cycle80/Rlib}"
OVERSAMPLE="${OVERSAMPLE:-20}"
POWER="${POWER:-2}"
SEED="${SEED:-123}"
REPLICATES="${REPLICATES:-3}"
TIMEOUT_SEC="${TIMEOUT_SEC:-7200}"
SOURCE_ARCHIVE_SHA256="${SOURCE_ARCHIVE_SHA256:-}"

mkdir -p "${OUT_ROOT}"
printf '%s\n' \
  "family,ncomp,baseline_gpu_mb,peak_gpu_mb,incremental_gpu_mb,after_gpu_mb,source_archive_sha256" \
  >"${OUT_ROOT}/nmr_cuda_gpu_memory_summary.csv"

gpu_used_mb() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
    head -1 | tr -d ' '
}

run_route() {
  local family="$1"
  local ncomp="$2"
  local stem="${family}_cuda_rsvd_k${ncomp}"
  local trace="${OUT_ROOT}/${stem}_gpu_trace.csv"
  local baseline
  local peak
  local after
  baseline="$(gpu_used_mb)"

  (
    printf '%s\n' "timestamp,memory_used_mb"
    while :; do
      printf '%s,%s\n' "$(date +%s.%N)" "$(gpu_used_mb)"
      sleep 0.05
    done
  ) >"${trace}" &
  local monitor_pid=$!

  set +e
  FASTPLS_LIB="${FASTPLS_LIB}" \
    /usr/bin/time -v timeout --signal=TERM --kill-after=30s "${TIMEOUT_SEC}" \
    Rscript "${RUNNER}" \
      --input="${INPUT}" \
      --output="${OUT_ROOT}/${stem}.csv" \
      --prediction_output="${OUT_ROOT}/${stem}_prediction.rds" \
      --family="${family}" \
      --backend=cuda \
      --solver=rsvd \
      --ncomp="${ncomp}" \
      --oversample="${OVERSAMPLE}" \
      --power="${POWER}" \
      --seed="${SEED}" \
      --replicates="${REPLICATES}" \
      >"${OUT_ROOT}/${stem}.log" \
      2>"${OUT_ROOT}/${stem}.time"
  local exit_status=$?
  set -e

  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
  after="$(gpu_used_mb)"
  peak="$(awk -F, 'NR > 1 && $2 + 0 > peak { peak = $2 + 0 } END { print peak + 0 }' "${trace}")"
  printf '%s,%s,%s,%s,%s,%s,%s\n' \
    "${family}" "${ncomp}" "${baseline}" "${peak}" \
    "$((peak - baseline))" "${after}" "${SOURCE_ARCHIVE_SHA256}" \
    >>"${OUT_ROOT}/nmr_cuda_gpu_memory_summary.csv"
  printf '%s\n' "${exit_status}" >"${OUT_ROOT}/${stem}_exit_status.txt"
  if [ "${exit_status}" -ne 0 ]; then
    return "${exit_status}"
  fi
}

run_route plssvd 5
run_route simpls 50
