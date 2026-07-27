#!/usr/bin/env bash
set -euo pipefail

TASK_RDS="${TASK_RDS:-$HOME/Documents/fastpls/data/imagenet_float32_seed123_train1000000_task.rds}"
RUNNER="${RUNNER:-$HOME/fastPLS_cmpb_cycle79/benchmark_imagenet_qualified_top5_path.R}"
OUT_ROOT="${OUT_ROOT:-/mnt/sata_ssd/fastPLS_cmpb_cycle79_imagenet_top5_scalar}"
NCOMP_LIST="${NCOMP_LIST:-100,200,300,400,500,600,700,800,900,1000}"
CLASSIFIERS="${CLASSIFIERS:-argmax,lda}"
OVERSAMPLE="${OVERSAMPLE:-20}"
POWER="${POWER:-1}"
SEED="${SEED:-123}"
TIMEOUT_SEC="${TIMEOUT_SEC:-10000}"

mkdir -p "${OUT_ROOT}/points"
status_file="${OUT_ROOT}/run_status.csv"
if [[ ! -f "${status_file}" ]]; then
  printf '%s\n' \
    'classifier,ncomp,exit_status,status,output_csv,time_log,gpu_log' \
    > "${status_file}"
fi

IFS=',' read -r -a classifiers <<< "${CLASSIFIERS}"
IFS=',' read -r -a component_counts <<< "${NCOMP_LIST}"

for classifier in "${classifiers[@]}"; do
  for ncomp in "${component_counts[@]}"; do
    stem="imagenet_simpls_cuda_${classifier}_n${ncomp}_qualified_top5"
    output="${OUT_ROOT}/points/${stem}.csv"
    log="${OUT_ROOT}/points/${stem}.log"
    time_log="${OUT_ROOT}/points/${stem}.time"
    gpu_log="${OUT_ROOT}/points/${stem}_gpu_trace.csv"

    if [[ -s "${output}" ]] && grep -q '"success"' "${output}"; then
      printf '[SKIP] %s ncomp=%s already succeeded\n' "${classifier}" "${ncomp}"
      continue
    fi

    printf '[RUN] %s ncomp=%s at %s\n' \
      "${classifier}" "${ncomp}" "$(date --iso-8601=seconds)"
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
    monitor_pid=$!

    set +e
    TASK_RDS="${TASK_RDS}" \
    OUTPUT_CSV="${output}" \
    BACKEND=cuda \
    CLASSIFIER="${classifier}" \
    NCOMP="${ncomp}" \
    OVERSAMPLE="${OVERSAMPLE}" \
    POWER="${POWER}" \
    SEED="${SEED}" \
    /usr/bin/time -v timeout --signal=TERM --kill-after=30s \
      "${TIMEOUT_SEC}" Rscript "${RUNNER}" > "${log}" 2> "${time_log}"
    exit_status=$?
    set -e

    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true

    case "${exit_status}" in
      0) status="success" ;;
      124|137) status="killed_timeout_or_memory" ;;
      *) status="failed" ;;
    esac
    printf '%s,%s,%s,%s,%s,%s,%s\n' \
      "${classifier}" "${ncomp}" "${exit_status}" "${status}" \
      "${output}" "${time_log}" "${gpu_log}" >> "${status_file}"
  done
done

python3 - "${OUT_ROOT}" <<'PY'
import csv
import glob
import os
import sys

root = sys.argv[1]
rows = []
fieldnames = None
for path in sorted(glob.glob(os.path.join(root, "points", "*.csv"))):
    if path.endswith("_gpu_trace.csv"):
        continue
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames and fieldnames is None:
                fieldnames = reader.fieldnames
            rows.extend(reader)
    except OSError:
        pass

if fieldnames:
    with open(
        os.path.join(root, "imagenet_qualified_top5_scalar_results.csv"),
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
PY

printf '%s\n' "${OUT_ROOT}"
