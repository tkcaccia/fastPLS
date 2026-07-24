#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="${FASTPLS_REPO_ROOT:-$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$HOME/fastPLS_pipeline4_float32_$(date +%Y%m%d_%H%M%S)}"
TASK_RDS="${TASK_RDS:-$HOME/Documents/fastpls/data/imagenet_float32_seed123_train1000000_task.rds}"
NCOMP_GRID="${NCOMP_GRID:-100 200 300 400 500 600 700 800 900 1000}"
BACKENDS="${BACKENDS:-cpu cuda}"
CLASSIFIERS="${CLASSIFIERS:-argmax lda cknn}"
REPS="${REPS:-1}"
SCALING="${SCALING:-centering}"
CKNN_MEMORY="${CKNN_MEMORY:-streaming}"
TIMEOUT_SEC="${TIMEOUT_SEC:-10000}"
FASTPLS_LIB="${FASTPLS_LIB:-}"
OUT_DIR="${RUN_ROOT}/results"
RAW_CSV="${OUT_DIR}/pipeline4_imagenet_raw.csv"
SUMMARY_CSV="${OUT_DIR}/pipeline4_imagenet_summary.csv"

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/rows" "${OUT_DIR}/gpu_samples"
rm -f "${RAW_CSV}" "${SUMMARY_CSV}"

gpu_sampler() {
  local r_pid="$1"
  local log_file="$2"
  while kill -0 "${r_pid}" 2>/dev/null; do
    nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null |
      awk -F',' -v pid="${r_pid}" '($1 + 0) == pid {gsub(/ /, "", $2); print $2}' >> "${log_file}"
    sleep 0.2
  done
}

peak_gpu_mb() {
  local file="$1"
  if [ -s "${file}" ]; then
    awk 'BEGIN{m=-1} /^[0-9.]+$/ {if (($1 + 0) > m) m=$1+0} END{if(m<0) print ""; else print m}' "${file}"
  fi
}

append_annotated_row() {
  local row_file="$1" time_file="$2" gpu_file="$3" process_status="$4"
  python3 - "${row_file}" "${time_file}" "${gpu_file}" "${process_status}" "${RAW_CSV}" <<'PY'
import csv, os, re, sys

row_file, time_file, gpu_file, process_status, raw_file = sys.argv[1:]

def time_value(lines, label):
    for line in lines:
        if label in line:
            return line.split(":", 1)[1].strip()
    return ""

def elapsed_seconds(value):
    if not value:
        return ""
    try:
        parts = [float(x) for x in value.split(":")]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return parts[0]
    except Exception:
        return ""

lines = open(time_file, errors="replace").read().splitlines() if os.path.exists(time_file) else []
rss_kb = time_value(lines, "Maximum resident set size")
user_sec = time_value(lines, "User time (seconds)")
system_sec = time_value(lines, "System time (seconds)")
elapsed = time_value(lines, "Elapsed (wall clock) time")
gpu_values = []
if os.path.exists(gpu_file):
    for value in open(gpu_file):
        try:
            gpu_values.append(float(value.strip()))
        except Exception:
            pass

if os.path.exists(row_file) and os.path.getsize(row_file):
    with open(row_file, newline="") as fh:
        row = next(csv.DictReader(fh))
else:
    row = {
        "dataset": "imagenet", "status": process_status,
        "error_message": "Benchmark process ended without writing a result row"
    }

if row.get("status") in ("", "started") or process_status != "ok":
    row["status"] = process_status
row["process_status"] = process_status
row["peak_host_rss_mb"] = round(float(rss_kb) / 1024, 3) if rss_kb else ""
row["peak_gpu_mem_mb"] = max(gpu_values) if gpu_values else ""
row["user_cpu_sec"] = user_sec
row["system_cpu_sec"] = system_sec
row["wall_elapsed_sec"] = elapsed_seconds(elapsed)

need_header = not os.path.exists(raw_file) or os.path.getsize(raw_file) == 0
existing_rows = []
existing_fields = []
if not need_header:
    with open(raw_file, newline="") as fh:
        reader = csv.DictReader(fh)
        existing_fields = list(reader.fieldnames or [])
        existing_rows = list(reader)
fields = existing_fields + [field for field in row if field not in existing_fields]
if not fields:
    fields = list(row)
if existing_rows and fields != existing_fields:
    with open(raw_file, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing_rows)
    need_header = False
for field in fields:
    row.setdefault(field, "")

with open(raw_file, "a", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    if need_header:
        writer.writeheader()
    writer.writerow(row)
PY
}

{
  echo "pipeline=4"
  echo "run_root=${RUN_ROOT}"
  echo "repo_root=${REPO_ROOT}"
  echo "task_rds=${TASK_RDS}"
  echo "ncomp_grid=${NCOMP_GRID}"
  echo "backends=${BACKENDS}"
  echo "classifiers=${CLASSIFIERS}"
  echo "reps=${REPS}"
  echo "precision=float32"
  echo "seed=123"
  echo "started=$(date --iso-8601=seconds 2>/dev/null || date)"
  echo "hostname=$(hostname)"
  git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null | sed 's/^/commit=/' || true
  Rscript -e 'cat("R=", R.version.string, "\n", sep=""); suppressPackageStartupMessages(library(fastPLS)); cat("fastPLS=", as.character(packageVersion("fastPLS")), "\n", sep=""); cat("cuda=", has_cuda(), "\n", sep="")' 2>&1
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
} > "${OUT_DIR}/manifest.txt"

for backend in ${BACKENDS}; do
  for classifier in ${CLASSIFIERS}; do
    for ncomp in ${NCOMP_GRID}; do
      rep=1
      while [ "${rep}" -le "${REPS}" ]; do
        id="${backend}_${classifier}_ncomp${ncomp}_rep${rep}"
        row_file="${OUT_DIR}/rows/${id}.csv"
        stdout_file="${OUT_DIR}/logs/${id}.stdout.log"
        time_file="${OUT_DIR}/logs/${id}.time.log"
        gpu_file="${OUT_DIR}/gpu_samples/${id}.csv"
        pid_file="${OUT_DIR}/logs/${id}.pid"
        rm -f "${row_file}" "${pid_file}" "${gpu_file}"
        echo "[$(date '+%F %T')] RUN ${id}" | tee -a "${OUT_DIR}/run.log"

        env \
          FASTPLS_LIB="${FASTPLS_LIB}" TASK_RDS="${TASK_RDS}" ROW_CSV="${row_file}" \
          BACKEND="${backend}" CLASSIFIER="${classifier}" NCOMP="${ncomp}" \
          REPLICATE="${rep}" SCALING="${SCALING}" CKNN_MEMORY="${CKNN_MEMORY}" \
          SEED=123 PID_FILE="${pid_file}" \
          /usr/bin/time -v timeout --signal=TERM --kill-after=30s "${TIMEOUT_SEC}" \
          Rscript "${REPO_ROOT}/benchmark/benchmark_imagenet_simpls_rsvd_classifiers.R" \
          >"${stdout_file}" 2>"${time_file}" &
        command_pid=$!

        r_pid=""
        for _ in $(seq 1 300); do
          if [ -s "${pid_file}" ]; then
            r_pid="$(cat "${pid_file}")"
            break
          fi
          kill -0 "${command_pid}" 2>/dev/null || break
          sleep 0.1
        done
        sampler_pid=""
        if [ -n "${r_pid}" ]; then
          gpu_sampler "${r_pid}" "${gpu_file}" &
          sampler_pid=$!
        fi

        wait "${command_pid}"
        exit_code=$?
        [ -n "${sampler_pid}" ] && wait "${sampler_pid}" 2>/dev/null || true
        process_status="ok"
        if [ "${exit_code}" -eq 124 ] || [ "${exit_code}" -eq 137 ]; then
          process_status="killed_timeout"
        elif [ "${exit_code}" -ne 0 ]; then
          if grep -Eqi 'out of memory|cannot allocate|killed.*signal 9|CUDA.*memory' "${time_file}"; then
            process_status="killed_memory"
          else
            process_status="error_${exit_code}"
          fi
        fi
        append_annotated_row "${row_file}" "${time_file}" "${gpu_file}" "${process_status}"
        echo "[$(date '+%F %T')] DONE ${id} status=${process_status}" | tee -a "${OUT_DIR}/run.log"
        rep=$((rep + 1))
      done
    done
  done
done

Rscript "${REPO_ROOT}/benchmark/plot_imagenet_simpls_rsvd_classifiers.R" "${OUT_DIR}" \
  >"${OUT_DIR}/plot.log" 2>&1 || true
echo "finished=$(date --iso-8601=seconds 2>/dev/null || date)" >> "${OUT_DIR}/manifest.txt"
echo "results=${OUT_DIR}"
