#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"

REMOTE_ROOT="${REMOTE_ROOT:-/home/chiamaka/fastPLS_remote_cifar100_compare}"
BASELINE_LIB="${FASTPLS_BASELINE_LIB:-/home/chiamaka/R/fastpls_baseline_lib}"
TEST_LIB="${FASTPLS_TEST_LIB:-/home/chiamaka/R/fastpls_test_lib}"
RESULTS_DIR="${FASTPLS_RESULTS_DIR:-${REMOTE_ROOT}/results}"
LOG_DIR="${RESULTS_DIR}/logs"
RUN_ROWS_DIR="${RESULTS_DIR}/run_rows"
GPU_LOG_DIR="${RESULTS_DIR}/gpu_samples"
PRED_DIR="${RESULTS_DIR}/predictions"
RAW_CSV="${RESULTS_DIR}/cifar100_remote_compare_raw.csv"
TASK_RDS="${RESULTS_DIR}/cifar100_task.rds"
TASK_META_RDS="${RESULTS_DIR}/cifar100_task_meta.rds"
COMPARE_REPS="${FASTPLS_COMPARE_REPS:-3}"
NCOMP_LIST="${FASTPLS_NCOMP_LIST:-2,5,10,20,50,100,200,500}"
INCLUDE_OPTIONAL_CPU="${FASTPLS_INCLUDE_OPTIONAL_CPU:-true}"
INCLUDE_TEST_CPU="${FASTPLS_INCLUDE_TEST_CPU:-true}"
VARIANT_OVERRIDE="${FASTPLS_VARIANTS:-}"
RUN_TIMEOUT_SEC="${FASTPLS_RUN_TIMEOUT_SEC:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TIME_BIN="${TIME_BIN:-/usr/bin/time}"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}" "${RUN_ROWS_DIR}" "${GPU_LOG_DIR}" "${PRED_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found"
  exit 1
fi

if ! command -v "${TIME_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${TIME_BIN} not found"
  exit 1
fi

Rscript "${REPO_ROOT}/benchmark/benchmark_cifar100_remote_compare.R" \
  --mode=prepare_task \
  --task-rds="${TASK_RDS}" \
  --meta-rds="${TASK_META_RDS}" \
  --split-seed=123

if [ -n "${VARIANT_OVERRIDE}" ]; then
  variants="${VARIANT_OVERRIDE}"
else
  variants="baseline_gpu_plssvd baseline_gpu_simpls_fast baseline_cpu_plssvd_cpu_rsvd baseline_cpu_simpls_fast_cpu_rsvd test_gpu_plssvd test_gpu_simpls_fast"
  if [ "${INCLUDE_OPTIONAL_CPU}" = "true" ]; then
    variants="${variants} baseline_cpu_plssvd_irlba baseline_cpu_simpls_fast_irlba"
  fi
  if [ "${INCLUDE_TEST_CPU}" = "true" ]; then
    variants="${variants} test_cpu_plssvd_cpu_rsvd test_cpu_simpls_fast_cpu_rsvd"
    if [ "${INCLUDE_OPTIONAL_CPU}" = "true" ]; then
      variants="${variants} test_cpu_plssvd_irlba test_cpu_simpls_fast_irlba"
    fi
  fi
fi

gpu_sampler() {
  r_pid="$1"
  log_file="$2"
  seen_pid_sample=0
  while kill -0 "${r_pid}" 2>/dev/null; do
    samples="$(nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | \
      awk -F',' -v pid="${r_pid}" '($1 + 0) == pid {gsub(/ /, "", $2); print $2}')"
    if [ -n "${samples}" ]; then
      seen_pid_sample=1
      printf '%s\n' "${samples}" | while IFS= read -r mb; do
        [ -n "${mb}" ] && printf 'pid,%s\n' "${mb}" >> "${log_file}"
      done
    elif [ "${seen_pid_sample}" -eq 0 ]; then
      total_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk 'NR == 1 {gsub(/ /, "", $1); print $1; exit}')"
      if [ -n "${total_used}" ]; then
        printf 'fallback,%s\n' "${total_used}" >> "${log_file}"
      fi
    fi
    sleep 0.2
  done
}

peak_gpu_from_log() {
  log_file="$1"
  if [ ! -s "${log_file}" ]; then
    echo "NA"
    return
  fi
  if grep -q '^pid,' "${log_file}"; then
    awk -F',' 'BEGIN{m=-1} /^pid,[0-9.]+$/ { if (($2 + 0) > m) m = $2 + 0 } END { if (m < 0) print "NA"; else print m }' "${log_file}"
  else
    awk -F',' 'BEGIN{m=-1}
      /^[0-9.]+$/ { if (($1 + 0) > m) m = $1 + 0 }
      /^fallback,[0-9.]+$/ { if (($2 + 0) > m) m = $2 + 0 }
      END { if (m < 0) print "NA"; else print m }' "${log_file}"
  fi
}

peak_rss_from_time_log() {
  time_log="$1"
  rss_kb="$(awk -F: '/Maximum resident set size/ {gsub(/^[ \t]+/, "", $2); print $2; exit}' "${time_log}" 2>/dev/null || true)"
  if [ -z "${rss_kb}" ]; then
    echo "NA"
  else
    "${PYTHON_BIN}" - <<PY
rss_kb = float("${rss_kb}")
print(round(rss_kb / 1024.0, 3))
PY
  fi
}

append_row() {
  raw_csv="$1"
  row_csv="$2"
  variant_name="$3"
  host_rss="$4"
  gpu_peak="$5"
  status_hint="$6"
  msg_hint="$7"

  "${PYTHON_BIN}" - "$raw_csv" "$row_csv" "$variant_name" "$host_rss" "$gpu_peak" "$status_hint" "$msg_hint" <<'PY'
import csv, os, sys

raw_csv, row_csv, variant_name, host_rss, gpu_peak, status_hint, msg_hint = sys.argv[1:]

def parse_variant(name):
    if name.startswith("baseline_"):
        code_tree = "baseline"
    elif name.startswith("test_"):
        code_tree = "test"
    else:
        code_tree = ""
    if "plssvd" in name:
        method_family = "plssvd"
    else:
        method_family = "simpls_fast"
    engine = "GPU" if "_gpu_" in name else "CPU"
    backend = "gpu_native" if engine == "GPU" else ("irlba" if name.endswith("_irlba") else "cpu_rsvd")
    return code_tree, method_family, engine, backend

fieldnames = [
    "variant_name","code_tree","method_family","engine","backend","precision_mode","label_mode",
    "replicate","requested_ncomp","effective_ncomp","n_train","n_test","p","n_classes",
    "fit_time_ms","predict_time_ms","total_time_ms","accuracy","prediction_file","reference_variant_name",
    "prediction_agreement","peak_host_rss_mb","peak_gpu_mem_mb","status","msg","dataset_path","split_seed"
]

row = None
if os.path.exists(row_csv) and os.path.getsize(row_csv) > 0:
    with open(row_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        row = next(reader, None)

if row is None:
    code_tree, method_family, engine, backend = parse_variant(variant_name)
    row = {k: "" for k in fieldnames}
    row.update({
        "variant_name": variant_name,
        "code_tree": code_tree,
        "method_family": method_family,
        "engine": engine,
        "backend": backend,
        "precision_mode": "default",
        "label_mode": "default",
        "status": status_hint or "error",
        "msg": msg_hint,
    })

if host_rss not in ("", "NA"):
    row["peak_host_rss_mb"] = host_rss
if gpu_peak not in ("", "NA"):
    row["peak_gpu_mem_mb"] = gpu_peak

existing_status = row.get("status", "")
if status_hint:
    if existing_status in ("", "ok", "capped"):
      row["status"] = status_hint
    elif existing_status.startswith("skipped"):
      row["status"] = existing_status
if msg_hint and not row.get("msg"):
    row["msg"] = msg_hint

need_header = not os.path.exists(raw_csv) or os.path.getsize(raw_csv) == 0
with open(raw_csv, "a", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if need_header:
        writer.writeheader()
    writer.writerow({k: row.get(k, "") for k in fieldnames})
PY
}

run_one() {
  variant_name="$1"
  requested_ncomp="$2"
  rep_id="$3"

  case "${variant_name}" in
    baseline_*) lib_loc="${BASELINE_LIB}" ;;
    test_*) lib_loc="${TEST_LIB}" ;;
    *) echo "ERROR: cannot resolve lib for ${variant_name}"; exit 1 ;;
  esac

  run_id="$(printf '%s__n%s__rep%s' "${variant_name}" "${requested_ncomp}" "${rep_id}")"
  row_csv="${RUN_ROWS_DIR}/${run_id}.csv"
  pid_file="${RUN_ROWS_DIR}/${run_id}.pid"
  pred_file="${PRED_DIR}/${run_id}.rds"
  stdout_log="${LOG_DIR}/${run_id}.stdout.log"
  time_log="${LOG_DIR}/${run_id}.time.log"
  gpu_log="${GPU_LOG_DIR}/${run_id}.txt"
  run_script="${RUN_ROWS_DIR}/${run_id}.run.sh"

  rm -f "${row_csv}" "${pid_file}" "${pred_file}" "${stdout_log}" "${time_log}" "${gpu_log}" "${run_script}"
  cat > "${run_script}" <<EOF
#!/bin/sh
exec Rscript "${REPO_ROOT}/benchmark/benchmark_cifar100_remote_compare.R" \
  --mode=run_one \
  --task-rds="${TASK_RDS}" \
  --row-out="${row_csv}" \
  --pid-file="${pid_file}" \
  --pred-out="${pred_file}" \
  --variant-name="${variant_name}" \
  --lib-loc="${lib_loc}" \
  --requested-ncomp="${requested_ncomp}" \
  --replicate="${rep_id}" \
  --include-optional-cpu="${INCLUDE_OPTIONAL_CPU}" \
  --include-test-cpu="${INCLUDE_TEST_CPU}"
EOF
  chmod +x "${run_script}"

  if [ "${RUN_TIMEOUT_SEC}" -gt 0 ] && command -v timeout >/dev/null 2>&1; then
    timeout -k 10 "${RUN_TIMEOUT_SEC}s" "${TIME_BIN}" -v "${run_script}" >"${stdout_log}" 2>"${time_log}" &
  else
    "${TIME_BIN}" -v "${run_script}" >"${stdout_log}" 2>"${time_log}" &
  fi
  cmd_pid=$!

  r_pid=""
  sampler_pid=""
  i=0
  while [ "${i}" -lt 200 ]; do
    if [ -s "${pid_file}" ]; then
      r_pid="$(cat "${pid_file}")"
      break
    fi
    if ! kill -0 "${cmd_pid}" 2>/dev/null; then
      break
    fi
    i=$((i + 1))
    sleep 0.1
  done

  if [ -n "${r_pid}" ]; then
    gpu_sampler "${r_pid}" "${gpu_log}" &
    sampler_pid=$!
  fi

  rc=0
  if ! wait "${cmd_pid}"; then
    rc=$?
  fi
  if [ -n "${sampler_pid}" ]; then
    wait "${sampler_pid}" || true
  fi

  host_rss="$(peak_rss_from_time_log "${time_log}")"
  gpu_peak="$(peak_gpu_from_log "${gpu_log}")"
  status_hint=""
  case "${rc}" in
    0) status_hint="" ;;
    124) status_hint="timed_out" ;;
    137|143) status_hint="killed" ;;
    *) status_hint="error" ;;
  esac
  msg_hint="$( (tail -n 20 "${stdout_log}" 2>/dev/null; tail -n 20 "${time_log}" 2>/dev/null) | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g' | cut -c1-800 )"

  append_row "${RAW_CSV}" "${row_csv}" "${variant_name}" "${host_rss}" "${gpu_peak}" "${status_hint}" "${msg_hint}"

  echo "[DONE] variant=${variant_name} ncomp=${requested_ncomp} rep=${rep_id} rc=${rc} host_rss_mb=${host_rss} gpu_peak_mb=${gpu_peak}"
}

echo "[INFO] Remote CIFAR100 compare starting"
echo "[INFO] RESULTS_DIR=${RESULTS_DIR}"
echo "[INFO] NCOMP_LIST=${NCOMP_LIST}"
echo "[INFO] COMPARE_REPS=${COMPARE_REPS}"
echo "[INFO] VARIANTS=${variants}"

for variant_name in ${variants}; do
  for requested_ncomp in $(printf '%s' "${NCOMP_LIST}" | tr ',' ' '); do
    for rep_id in $(seq 1 "${COMPARE_REPS}"); do
      run_one "${variant_name}" "${requested_ncomp}" "${rep_id}"
    done
  done
done

export FASTPLS_RESULTS_DIR="${RESULTS_DIR}"
export FASTPLS_BASELINE_COMMIT="$(cat "${REMOTE_ROOT}/meta/baseline_commit.txt" 2>/dev/null || true)"
export FASTPLS_TEST_COMMIT="$(cat "${REMOTE_ROOT}/meta/test_commit.txt" 2>/dev/null || true)"
export FASTPLS_TEST_COMMIT_NOTE="$(cat "${REMOTE_ROOT}/meta/test_commit_note.txt" 2>/dev/null || true)"
export FASTPLS_BASELINE_LIB="${BASELINE_LIB}"
export FASTPLS_TEST_LIB="${TEST_LIB}"
export FASTPLS_COMPARE_REPS="${COMPARE_REPS}"

Rscript "${REPO_ROOT}/benchmark/write_cifar100_remote_compare_report.R" --results-dir="${RESULTS_DIR}" --task-meta-rds="${TASK_META_RDS}"
Rscript "${REPO_ROOT}/benchmark/plot_cifar100_remote_compare.R" --results-dir="${RESULTS_DIR}"

RESULTS_PARENT="$(dirname "${RESULTS_DIR}")"
RESULTS_BASENAME="$(basename "${RESULTS_DIR}")"
tar -czf "${REMOTE_ROOT}/cifar100_remote_compare_results.tar.gz" -C "${RESULTS_PARENT}" "${RESULTS_BASENAME}"

echo "[INFO] Remote CIFAR100 compare completed"
echo "[INFO] Tarball: ${REMOTE_ROOT}/cifar100_remote_compare_results.tar.gz"
