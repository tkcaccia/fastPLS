#!/usr/bin/env bash

set -u

RUN_ROOT="${1:-$HOME/fastPLS_publication_benchmarks_20260722_score_reuse}"
REPO_ROOT="${FASTPLS_REPO_ROOT:-$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)}"
POLL_SEC="${FASTPLS_WATCH_INTERVAL_SEC:-300}"
PROGRESS_FILE="${RUN_ROOT}/progress.log"
TASK_DIR="${RUN_ROOT}/pipeline1/real_datasets"
LIB_LOC="${RUN_ROOT}/pipeline1/Rlib"
OUTPUT_DIR="${RUN_ROOT}/supplementary_kernel_sensitivity"
TUNING_DIR="${OUTPUT_DIR}/tuning"
RUN_ROWS_DIR="${OUTPUT_DIR}/run_rows"
LOG_DIR="${OUTPUT_DIR}/logs"
GPU_DIR="${OUTPUT_DIR}/gpu_samples"
FINAL_REPS="${FASTPLS_KERNEL_FINAL_REPS:-3}"
KFOLD="${FASTPLS_KERNEL_KFOLD:-5}"
TUNE_TIMEOUT="${FASTPLS_KERNEL_TUNE_TIMEOUT_SEC:-3600}"
RUN_TIMEOUT="${FASTPLS_KERNEL_RUN_TIMEOUT_SEC:-1200}"

mkdir -p "${TUNING_DIR}" "${RUN_ROWS_DIR}" "${LOG_DIR}" "${GPU_DIR}"

while ! grep -q "PUBLICATION BENCHMARK SUITE COMPLETE" "${PROGRESS_FILE}" 2>/dev/null; do
  sleep "${POLL_SEC}"
done

EXISTING_R_LIBS="$(Rscript -e 'cat(paste(.libPaths(), collapse=.Platform$path.sep))')"
export R_LIBS_USER="${LIB_LOC}${EXISTING_R_LIBS:+:${EXISTING_R_LIBS}}"

log_msg() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${OUTPUT_DIR}/progress.log"
}

peak_rss_from_log() {
  local path="$1"
  local kb
  kb="$(awk -F: '/Maximum resident set size/ {gsub(/^[ \t]+/, "", $2); print $2; exit}' "${path}" 2>/dev/null || true)"
  if [ -n "${kb}" ]; then
    awk -v kb="${kb}" 'BEGIN { printf "%.3f\n", kb / 1024.0 }'
  else
    printf 'NA\n'
  fi
}

gpu_sampler() {
  local pid="$1"
  local path="$2"
  while kill -0 "${pid}" 2>/dev/null; do
    nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null |
      awk -F',' -v target="${pid}" '{gsub(/ /, "", $1); gsub(/ /, "", $2); if ($1 == target) print $2}' >> "${path}" || true
    sleep 0.2
  done
}

peak_gpu_from_log() {
  local path="$1"
  if [ -s "${path}" ]; then
    awk 'BEGIN {m=-1} /^[0-9.]+$/ {if ($1 > m) m=$1} END {if (m < 0) print "NA"; else print m}' "${path}"
  else
    printf 'NA\n'
  fi
}

declare -A NCOMP_GRIDS=(
  [metref]="2,5,10,22,50,100"
  [ccle]="2,5,10,18,50,100"
  [prism]="2,5,10,20,50,100"
  [nmr]="2,5,10,20,50,100"
)

log_msg "Starting supplementary kernel tuning"
for dataset in metref ccle prism nmr; do
  task_rds="${TASK_DIR}/${dataset}_task.rds"
  if [ ! -f "${task_rds}" ]; then
    log_msg "SKIP dataset=${dataset}: task file missing (${task_rds})"
    continue
  fi
  log_msg "TUNE dataset=${dataset} ncomp=${NCOMP_GRIDS[${dataset}]}"
  timeout --signal=TERM --kill-after=30s "${TUNE_TIMEOUT}" \
    Rscript "${REPO_ROOT}/benchmark/benchmark_kernel_sensitivity.R" \
      --mode=tune_one \
      --task-rds="${task_rds}" \
      --out-dir="${TUNING_DIR}" \
      --ncomp-grid="${NCOMP_GRIDS[${dataset}]}" \
      --kfold="${KFOLD}" \
      --seed=123 \
      --lib-loc="${LIB_LOC}" \
      >"${LOG_DIR}/${dataset}_tuning.stdout.log" \
      2>"${LOG_DIR}/${dataset}_tuning.stderr.log" || \
    log_msg "TUNING FAILED dataset=${dataset}; see ${LOG_DIR}/${dataset}_tuning.stderr.log"
done

cuda_available="$(Rscript -e 'suppressPackageStartupMessages(library(fastPLS)); cat(isTRUE(has_cuda()))' 2>/dev/null || printf 'FALSE')"

for dataset in metref ccle prism nmr; do
  task_rds="${TASK_DIR}/${dataset}_task.rds"
  selected_csv="${TUNING_DIR}/${dataset}_kernel_selected.csv"
  if [ ! -s "${selected_csv}" ]; then
    log_msg "SKIP final dataset=${dataset}: no selected kernel configurations"
    continue
  fi
  while IFS=$'\t' read -r kernel gamma degree coef0 ncomp; do
    [ -n "${kernel}" ] || continue
    for backend in cpu cuda; do
      if [ "${backend}" = "cuda" ] && [ "${cuda_available}" != "TRUE" ]; then
        log_msg "SKIP dataset=${dataset} kernel=${kernel} backend=cuda: CUDA unavailable"
        continue
      fi
      rep=1
      while [ "${rep}" -le "${FINAL_REPS}" ]; do
        run_id="${dataset}__${kernel}__${backend}__n${ncomp}__rep${rep}"
        row_out="${RUN_ROWS_DIR}/${run_id}.csv"
        pid_file="${RUN_ROWS_DIR}/${run_id}.pid"
        stdout_log="${LOG_DIR}/${run_id}.stdout.log"
        time_log="${LOG_DIR}/${run_id}.time.log"
        gpu_log="${GPU_DIR}/${run_id}.txt"
        rm -f "${row_out}" "${pid_file}" "${stdout_log}" "${time_log}" "${gpu_log}"
        log_msg "RUN dataset=${dataset} kernel=${kernel} backend=${backend} ncomp=${ncomp} rep=${rep}"
        /usr/bin/time -v timeout --signal=TERM --kill-after=30s "${RUN_TIMEOUT}" \
          Rscript "${REPO_ROOT}/benchmark/benchmark_kernel_sensitivity.R" \
            --mode=run_one \
            --task-rds="${task_rds}" \
            --row-out="${row_out}" \
            --pid-file="${pid_file}" \
            --kernel="${kernel}" \
            --gamma="${gamma}" \
            --degree="${degree}" \
            --coef0="${coef0}" \
            --ncomp="${ncomp}" \
            --backend="${backend}" \
            --replicate="${rep}" \
            --lib-loc="${LIB_LOC}" \
            >"${stdout_log}" 2>"${time_log}" &
        cmd_pid=$!
        r_pid=""
        for _ in $(seq 1 200); do
          if [ -s "${pid_file}" ]; then
            r_pid="$(cat "${pid_file}")"
            break
          fi
          kill -0 "${cmd_pid}" 2>/dev/null || break
          sleep 0.1
        done
        sampler_pid=""
        if [ "${backend}" = "cuda" ] && [ -n "${r_pid}" ]; then
          gpu_sampler "${r_pid}" "${gpu_log}" &
          sampler_pid=$!
        fi
        run_status=0
        wait "${cmd_pid}" || run_status=$?
        if [ -n "${sampler_pid}" ]; then wait "${sampler_pid}" || true; fi
        host_peak="$(peak_rss_from_log "${time_log}")"
        gpu_peak="$(peak_gpu_from_log "${gpu_log}")"
        if [ ! -s "${row_out}" ]; then
          failure_status="error"
          failure_msg="Benchmark process exited before writing a row"
          if [ "${run_status}" -eq 124 ] || grep -q 'status 124' "${time_log}" 2>/dev/null; then
            failure_status="killed_timeout"
            failure_msg="Run exceeded ${RUN_TIMEOUT} seconds"
          elif grep -qi 'out of memory\|terminated by signal 9' "${time_log}" 2>/dev/null; then
            failure_status="killed_memory"
            failure_msg="Run was killed under memory pressure"
          fi
          Rscript "${REPO_ROOT}/benchmark/benchmark_kernel_sensitivity.R" \
            --mode=write_failure --task-rds="${task_rds}" --row-out="${row_out}" \
            --kernel="${kernel}" --gamma="${gamma}" --degree="${degree}" --coef0="${coef0}" \
            --ncomp="${ncomp}" --backend="${backend}" --replicate="${rep}" \
            --status="${failure_status}" --msg="${failure_msg}" --lib-loc="${LIB_LOC}"
        fi
        Rscript "${REPO_ROOT}/benchmark/benchmark_kernel_sensitivity.R" \
          --mode=annotate_row --row-out="${row_out}" \
          --peak-host-rss-mb="${host_peak}" --peak-gpu-mem-mb="${gpu_peak}" \
          --lib-loc="${LIB_LOC}"
        row_status="$(Rscript -e 'd <- read.csv(commandArgs(TRUE)[1], stringsAsFactors=FALSE); cat(d$status[[1]])' "${row_out}" 2>/dev/null || printf 'error')"
        if [ "${run_status}" -eq 124 ] || [ "${row_status}" != "ok" ]; then
          log_msg "STOP REPEATS dataset=${dataset} kernel=${kernel} backend=${backend} status=${row_status} exit=${run_status}"
          break
        fi
        rep=$((rep + 1))
      done
    done
  done < <(Rscript -e 'd <- read.csv(commandArgs(TRUE)[1], stringsAsFactors=FALSE); write.table(d[,c("kernel","gamma","degree","coef0","ncomp")], row.names=FALSE, col.names=FALSE, quote=FALSE, sep="\t", na="NA")' "${selected_csv}")
done

Rscript "${REPO_ROOT}/benchmark/benchmark_kernel_sensitivity.R" \
  --mode=summarize --out-dir="${OUTPUT_DIR}" --lib-loc="${LIB_LOC}"
Rscript "${REPO_ROOT}/benchmark/plot_kernel_sensitivity.R" \
  "${OUTPUT_DIR}" "${OUTPUT_DIR}/plots"
log_msg "Supplementary kernel sensitivity benchmark complete"
