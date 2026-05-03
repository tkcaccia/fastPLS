#!/usr/bin/env bash

set -u

RUN_ROOT="${FASTPLS_WATCH_RUN_ROOT:?Set FASTPLS_WATCH_RUN_ROOT to the benchmark results directory}"
REPO_ROOT="${FASTPLS_WATCH_REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
BASE_DIR="${FASTPLS_WATCH_BASE_DIR:-$(cd "${REPO_ROOT}/.." && pwd)}"
INTERVAL_SEC="${FASTPLS_WATCH_INTERVAL_SEC:-1800}"
MAX_IDLE_SEC="${FASTPLS_WATCH_MAX_IDLE_SEC:-5400}"
MAX_CYCLES="${FASTPLS_WATCH_MAX_CYCLES:-0}"

LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"
WATCH_LOG="${LOG_DIR}/watchdog.log"
PID_FILE="${RUN_ROOT}.watchdog.pid"

log_msg() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >>"${WATCH_LOG}"
}

raw_file() {
  printf '%s/real_datasets/dataset_memory_compare_raw.csv' "${RUN_ROOT}"
}

raw_rows() {
  local raw
  raw="$(raw_file)"
  if [ -f "${raw}" ]; then
    python3 - "${raw}" <<'PY'
import csv, sys
try:
    with open(sys.argv[1], newline="") as fh:
        print(sum(1 for _ in csv.DictReader(fh)))
except Exception:
    print(0)
PY
  else
    printf '0\n'
  fi
}

raw_mtime() {
  local raw
  raw="$(raw_file)"
  if [ -f "${raw}" ]; then
    stat -c '%Y' "${raw}" 2>/dev/null || printf '0\n'
  else
    printf '0\n'
  fi
}

active_processes() {
  ps -eo pid=,etime=,stat=,cmd= |
    grep -E "${RUN_ROOT}|${REPO_ROOT}/scripts/(run_full_variable_benchmark|remote_run_dataset_memory_compare)|benchmark_(dataset_memory_compare|synthetic_variable_sweeps)\\.R" |
    grep -v grep |
    grep -v remote_watch_full_variable_benchmark || true
}

is_completed() {
  grep -q "Full variable benchmark completed" "${RUN_ROOT}.launch.log" 2>/dev/null ||
    grep -q "Full variable benchmark completed" "${RUN_ROOT}/logs/"*.log 2>/dev/null
}

plot_real_if_possible() {
  local raw
  raw="$(raw_file)"
  if [ -s "${raw}" ] && [ -f "${REPO_ROOT}/benchmark/plot_dataset_memory_compare.R" ]; then
    Rscript "${REPO_ROOT}/benchmark/plot_dataset_memory_compare.R" "${RUN_ROOT}/real_datasets" \
      >>"${LOG_DIR}/watchdog_plot_real.log" 2>&1 || log_msg "WARN real plot regeneration failed"
  fi
}

plot_simulated_if_possible() {
  local d
  for d in "${RUN_ROOT}/simulated_datasets"/ncomp_*; do
    [ -d "${d}" ] || continue
    [ -s "${d}/synthetic_variable_sweeps_raw.csv" ] || continue
    [ -f "${REPO_ROOT}/benchmark/plot_synthetic_variable_sweeps.R" ] || continue
    Rscript "${REPO_ROOT}/benchmark/plot_synthetic_variable_sweeps.R" "${d}" \
      >>"${LOG_DIR}/watchdog_plot_simulated.log" 2>&1 || log_msg "WARN simulated plot regeneration failed for ${d}"
  done
}

summarize_csv() {
  local raw
  raw="$(raw_file)"
  if [ -f "${raw}" ]; then
    python3 - "${raw}" <<'PY'
import csv, sys, collections
try:
    rows = list(csv.DictReader(open(sys.argv[1], newline="")))
except Exception as exc:
    print("csv_error=%s" % exc)
    raise SystemExit(0)
print("rows=%d datasets=%s classifiers=%s status=%s" % (
    len(rows),
    dict(collections.Counter(r.get("dataset", "") for r in rows)),
    dict(collections.Counter(r.get("classifier", "") for r in rows)),
    dict(collections.Counter(r.get("status", "") for r in rows)),
))
if rows:
    r = rows[-1]
    keys = ["dataset", "variant_name", "classifier", "requested_ncomp", "replicate", "status", "metric_value"]
    print("last=%s" % {k: r.get(k, "") for k in keys})
PY
  else
    printf 'rows=0 datasets={} classifiers={} status={}\n'
  fi
}

echo "$$" >"${PID_FILE}"
log_msg "watchdog started run_root=${RUN_ROOT} repo_root=${REPO_ROOT} base_dir=${BASE_DIR} interval=${INTERVAL_SEC}s max_idle=${MAX_IDLE_SEC}s"

last_rows="$(raw_rows)"
last_mtime="$(raw_mtime)"
last_progress_epoch="$(date +%s)"
cycle=0

while :; do
  cycle=$((cycle + 1))
  now="$(date +%s)"
  rows="$(raw_rows)"
  mtime="$(raw_mtime)"
  active="$(active_processes)"
  completed=0
  if is_completed; then completed=1; fi

  if [ "${rows}" != "${last_rows}" ] || [ "${mtime}" != "${last_mtime}" ]; then
    last_progress_epoch="${now}"
    last_rows="${rows}"
    last_mtime="${mtime}"
  fi

  idle=$((now - last_progress_epoch))
  log_msg "cycle=${cycle} active=$([ -n "${active}" ] && printf yes || printf no) completed=${completed} rows=${rows} idle_sec=${idle}"
  summarize_csv >>"${WATCH_LOG}" 2>&1

  plot_real_if_possible
  plot_simulated_if_possible

  if [ "${completed}" -eq 1 ]; then
    log_msg "pipeline completed; watchdog exiting"
    rm -f "${PID_FILE}"
    exit 0
  fi

  if [ -z "${active}" ]; then
    log_msg "ERROR no active benchmark process before completion; manual inspection required"
  elif [ "${idle}" -gt "${MAX_IDLE_SEC}" ]; then
    log_msg "WARN no CSV progress for ${idle}s; keeping pipeline alive for timeout supervisor"
    active_processes >>"${WATCH_LOG}" 2>&1
  fi

  if [ "${MAX_CYCLES}" -gt 0 ] 2>/dev/null && [ "${cycle}" -ge "${MAX_CYCLES}" ]; then
    log_msg "max cycles reached; watchdog exiting"
    rm -f "${PID_FILE}"
    exit 0
  fi

  sleep "${INTERVAL_SEC}"
done
