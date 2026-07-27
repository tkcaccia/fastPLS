#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"

RESULTS_DIR="${FASTPLS_RESULTS_DIR:-${REPO_ROOT}/benchmark_results_dataset_memory_compare}"
LIB_LOC="${FASTPLS_BENCH_LIB:-${HOME}/R/fastpls_bench_fresh}"
DATASETS="${FASTPLS_DATASETS:-metref,ccle,cifar100,prism,gtex_v8,tcga_pan_cancer,retina,tabula,tcga_brca,tcga_hnsc_methylation,nmr,cbmc_citeseq}"
NCOMP_LIST="${FASTPLS_NCOMP_LIST:-2,5,10,18,20,50}"
METREF_NCOMP_LIST="${FASTPLS_METREF_NCOMP_LIST:-2,5,10,22,50,100}"
CCLE_NCOMP_LIST="${FASTPLS_CCLE_NCOMP_LIST:-2,5,10,18,50,100}"
CIFAR100_NCOMP_LIST="${FASTPLS_CIFAR100_NCOMP_LIST:-2,5,10,20,50,100,200}"
IMAGENET_NCOMP_LIST="${FASTPLS_IMAGENET_NCOMP_LIST:-2,5,10,20,50,100}"
NMR_NCOMP_LIST="${FASTPLS_NMR_NCOMP_LIST:-2,5,10,20,50,100,200,500}"
SMALL_MULTI_NCOMP_LIST="${FASTPLS_SMALL_MULTI_NCOMP_LIST:-2,5,10,20,50}"
MID_MULTI_NCOMP_LIST="${FASTPLS_MID_MULTI_NCOMP_LIST:-2,5,10,20,50,100}"
GTEX_V8_NCOMP_LIST="${FASTPLS_GTEX_V8_NCOMP_LIST:-2,5,10,20,32,50,100}"
TCGA_PAN_CANCER_NCOMP_LIST="${FASTPLS_TCGA_PAN_CANCER_NCOMP_LIST:-2,5,10,20,32,50,100}"
REPS="${FASTPLS_COMPARE_REPS:-3}"
LARGE_REPS="${FASTPLS_COMPARE_LARGE_REPS:-${REPS}}"
SPLIT_SEED="${FASTPLS_SPLIT_SEED:-123}"
PRECISION="${FASTPLS_BENCH_PRECISION:-float32}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TIME_BIN="${TIME_BIN:-/usr/bin/time}"
TIME_ARGS="${TIME_ARGS:--v}"
if ! "${TIME_BIN}" ${TIME_ARGS} true >/dev/null 2>/tmp/fastpls_time_probe.$$; then
  if "${TIME_BIN}" -l true >/dev/null 2>/tmp/fastpls_time_probe.$$; then
    TIME_ARGS="-l"
  fi
fi
rm -f /tmp/fastpls_time_probe.$$
TIMEOUT_BIN="${TIMEOUT_BIN:-timeout}"
RUN_TIMEOUT_SEC="${FASTPLS_RUN_TIMEOUT_SEC:-0}"
VARIANTS_FILTER="${FASTPLS_VARIANTS:-}"
SKIP_PLOT="${FASTPLS_SKIP_PLOT:-false}"

RAW_CSV="${RESULTS_DIR}/dataset_memory_compare_raw.csv"
RUN_ROWS_DIR="${RESULTS_DIR}/run_rows"
GPU_LOG_DIR="${RESULTS_DIR}/gpu_samples"
PRED_DIR="${RESULTS_DIR}/predictions"
LOG_DIR="${RESULTS_DIR}/logs"
SAVE_PREDICTIONS="${FASTPLS_SAVE_PREDICTIONS:-false}"

mkdir -p "${RESULTS_DIR}" "${RUN_ROWS_DIR}" "${GPU_LOG_DIR}" "${LOG_DIR}"
if [ "${SAVE_PREDICTIONS}" = "true" ]; then
  mkdir -p "${PRED_DIR}"
fi
rm -f "${RAW_CSV}"

memory_sampler() {
  r_pid="$1"
  log_file="$2"
  ready_file="$3"
  first_sample=1
  while kill -0 "${r_pid}" 2>/dev/null; do
    rss_kb="$(awk '/^VmRSS:/ {print $2; exit}' "/proc/${r_pid}/status" 2>/dev/null || true)"
    if [ -z "${rss_kb}" ]; then
      rss_kb="$(ps -o rss= -p "${r_pid}" 2>/dev/null | awk 'NR == 1 {print $1}' || true)"
    fi
    if [ -n "${rss_kb}" ]; then
      "${PYTHON_BIN}" - "${rss_kb}" >> "${log_file}" <<'PY'
import sys
print("rss,{:.6f}".format(float(sys.argv[1]) / 1024.0))
PY
    fi
    samples="$(nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | \
      awk -F',' -v pid="${r_pid}" '($1 + 0) == pid {gsub(/ /, "", $2); print $2}')"
    if [ -n "${samples}" ]; then
      printf '%s\n' "${samples}" | while IFS= read -r mb; do
        [ -n "${mb}" ] && printf 'pid,%s\n' "${mb}" >> "${log_file}"
      done
    fi
    if [ "${first_sample}" -eq 1 ]; then
      : > "${ready_file}"
      first_sample=0
    fi
    sleep 0.02
  done
  if [ "${first_sample}" -eq 1 ]; then
    : > "${ready_file}"
  fi
}

peak_host_window_from_log() {
  log_file="$1"
  if [ ! -s "${log_file}" ]; then
    echo "NA"
    return
  fi
  awk -F',' 'BEGIN{m=-1} /^rss,[0-9.]+$/ { if (($2 + 0) > m) m = $2 + 0 } END { if (m < 0) print "NA"; else print m }' "${log_file}"
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
    echo "NA"
  fi
}

peak_rss_from_time_log() {
  time_log="$1"
  rss_kb="$(awk -F: '/Maximum resident set size/ {gsub(/^[ \t]+/, "", $2); print $2; exit}' "${time_log}" 2>/dev/null || true)"
  if [ -n "${rss_kb}" ]; then
    "${PYTHON_BIN}" - <<PY
rss_kb = float("${rss_kb}")
print(round(rss_kb / 1024.0, 3))
PY
    return
  fi
  rss_bytes="$(awk '/maximum resident set size/ {print $1; exit}' "${time_log}" 2>/dev/null || true)"
  if [ -n "${rss_bytes}" ]; then
    "${PYTHON_BIN}" - <<PY
rss_bytes = float("${rss_bytes}")
print(round(rss_bytes / (1024.0 * 1024.0), 3))
PY
    return
  fi
  echo "NA"
}

row_status_from_csv() {
  row_csv="$1"
  "${PYTHON_BIN}" - "${row_csv}" <<'PY'
import csv, sys
path = sys.argv[1]
try:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    print(rows[-1].get("status", "") if rows else "")
except Exception:
    print("")
PY
}

append_row() {
  raw_csv="$1"
  row_csv="$2"
  process_peak_host="$3"
  fit_window_peak_host="$4"
  gpu_peak="$5"

  "${PYTHON_BIN}" - "$raw_csv" "$row_csv" "$process_peak_host" "$fit_window_peak_host" "$gpu_peak" <<'PY'
import csv, os, sys

raw_csv, row_csv, process_peak_host, fit_window_peak_host, gpu_peak = sys.argv[1:]

with open(row_csv, newline="") as fh:
    reader = csv.DictReader(fh)
    row = next(reader)

if process_peak_host not in ("", "NA"):
    row["peak_host_rss_mb"] = process_peak_host
if fit_window_peak_host not in ("", "NA"):
    row["fit_window_peak_host_rss_mb"] = fit_window_peak_host
if gpu_peak not in ("", "NA"):
    row["peak_gpu_mem_mb"] = gpu_peak
try:
    baseline = float(row.get("rss_before_fit_mb", ""))
    candidates = [
        float(row.get(key, ""))
        for key in (
            "fit_window_peak_host_rss_mb",
            "rss_after_fit_mb",
            "rss_after_predict_mb",
        )
        if row.get(key, "") not in ("", "NA")
    ]
    peak = max(candidates)
    row["fit_window_peak_host_rss_mb"] = peak
    row["incremental_host_rss_mb"] = max(0.0, peak - baseline)
except (TypeError, ValueError):
    pass
try:
    baseline = float(row.get("gpu_before_fit_mb", ""))
    candidates = [
        float(row.get(key, ""))
        for key in (
            "peak_gpu_mem_mb",
            "gpu_after_fit_mb",
            "gpu_after_predict_mb",
        )
        if row.get(key, "") not in ("", "NA")
    ]
    peak = max(candidates)
    row["peak_gpu_mem_mb"] = peak
    row["incremental_gpu_mem_mb"] = max(0.0, peak - baseline)
except (TypeError, ValueError):
    pass

need_header = not os.path.exists(raw_csv) or os.path.getsize(raw_csv) == 0
with open(raw_csv, "a", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
    if need_header:
        writer.writeheader()
    writer.writerow(row)
PY
}

write_missing_row() {
  row_csv="$1"
  task_rds="$2"
  variant_name="$3"
  requested_ncomp="$4"
  rep_id="$5"
  host_rss="$6"
  gpu_peak="$7"
  time_log="$8"
  pred_file="$9"

  status="missing_row"
  msg="Benchmark process exited without producing a row CSV"
  if [ -s "${time_log}" ]; then
    if grep -q 'Command terminated by signal 9' "${time_log}"; then
      status="killed_sig9"
      msg="Benchmark process terminated by signal 9"
    elif grep -q 'Command exited with non-zero status 124' "${time_log}"; then
      status="killed_timeout"
      msg="Benchmark process exceeded timeout"
    elif grep -qi 'out of memory' "${time_log}"; then
      status="killed_oom"
      msg="Benchmark process appears to have been killed due to memory pressure"
    fi
  fi

  Rscript - "${REPO_ROOT}" "${task_rds}" "${variant_name}" "${requested_ncomp}" "${rep_id}" "${host_rss}" "${gpu_peak}" "${status}" "${msg}" "${pred_file}" "${row_csv}" <<'RS'
args <- commandArgs(trailingOnly = TRUE)
repo_root <- args[[1L]]
task_rds <- args[[2L]]
variant_name <- args[[3L]]
requested_ncomp <- as.integer(args[[4L]])
rep_id <- as.integer(args[[5L]])
host_rss <- suppressWarnings(as.numeric(args[[6L]]))
gpu_peak <- suppressWarnings(as.numeric(args[[7L]]))
status <- args[[8L]]
msg <- args[[9L]]
pred_file <- args[[10L]]
row_csv <- args[[11L]]

source(file.path(repo_root, "benchmark", "helpers_dataset_memory_compare.R"))
task <- readRDS(task_rds)
spec <- variant_spec(variant_name)

effective_ncomp <- tryCatch(
  safe_effective_ncomp(task, requested_ncomp, method_family = spec$method_family),
  error = function(e) NA_integer_
)
metric_name <- if (identical(task$task_type, "classification")) {
  "accuracy"
} else if (isTRUE(task$n_classes == 1L)) {
  "q2"
} else {
  "rmsd"
}

if (!is.finite(host_rss)) host_rss <- NA_real_
if (!is.finite(gpu_peak)) gpu_peak <- NA_real_
if (!nzchar(pred_file) || !file.exists(pred_file)) pred_file <- NA_character_

row <- data.frame(
  dataset = task$dataset,
  task_type = task$task_type,
  variant_name = variant_name,
  requested_method = spec$method_family,
  executed_method = spec$method_family,
  method_family = spec$method_family,
  method_panel = method_panel_label(spec$method_family),
  engine = spec$engine,
  backend = spec$backend,
  implementation_label = spec$implementation_label,
  classifier = spec$classifier,
  replicate = as.integer(rep_id),
  requested_ncomp = as.integer(requested_ncomp),
  effective_ncomp = as.integer(effective_ncomp),
  n_train = as.integer(task$n_train),
  n_test = as.integer(task$n_test),
  p = as.integer(task$p),
  n_classes = as.integer(task$n_classes),
  precision = if (!is.null(task$precision)) task$precision else benchmark_matrix_precision(task$Xtrain),
  execution_precision = NA_character_,
  classifier_backend = NA_character_,
  classifier_numeric_path = NA_character_,
  input_storage_mb = if (!is.null(task$input_storage_mb)) as.numeric(task$input_storage_mb) else NA_real_,
  fit_time_ms = NA_real_,
  predict_time_ms = NA_real_,
  total_time_ms = NA_real_,
  metric_name = metric_name,
  metric_value = NA_real_,
  accuracy = NA_real_,
  top5_accuracy = NA_real_,
  balanced_accuracy = NA_real_,
  macro_f1 = NA_real_,
  prediction_file = pred_file,
  peak_host_rss_mb = host_rss,
  peak_gpu_mem_mb = gpu_peak,
  fit_window_peak_host_rss_mb = NA_real_,
  incremental_host_rss_mb = NA_real_,
  rss_before_fit_mb = NA_real_,
  rss_after_fit_mb = NA_real_,
  rss_after_predict_mb = NA_real_,
  gpu_before_fit_mb = NA_real_,
  gpu_after_fit_mb = NA_real_,
  gpu_after_predict_mb = NA_real_,
  incremental_gpu_mem_mb = NA_real_,
  status = status,
  msg = msg,
  dataset_path = task$dataset_path,
  split_seed = as.integer(task$split_seed),
  stringsAsFactors = FALSE
)

utils::write.csv(row, row_csv, row.names = FALSE, quote = TRUE, na = "")
RS
}

for dataset_id in $(printf '%s' "${DATASETS}" | tr ',' ' '); do
  dataset_ncomp_list="${NCOMP_LIST}"
  case "${dataset_id}" in
    metref)
      dataset_ncomp_list="${METREF_NCOMP_LIST}"
      ;;
    ccle)
      dataset_ncomp_list="${CCLE_NCOMP_LIST}"
      ;;
    cifar100)
      dataset_ncomp_list="${CIFAR100_NCOMP_LIST}"
      ;;
    imagenet)
      dataset_ncomp_list="${IMAGENET_NCOMP_LIST}"
      ;;
    nmr)
      dataset_ncomp_list="${NMR_NCOMP_LIST}"
      ;;
    retina|singlecell|tabula|tcga_brca|tcga_hnsc_methylation)
      dataset_ncomp_list="${SMALL_MULTI_NCOMP_LIST}"
      ;;
    gtex_v8)
      dataset_ncomp_list="${GTEX_V8_NCOMP_LIST}"
      ;;
    tcga_pan_cancer)
      dataset_ncomp_list="${TCGA_PAN_CANCER_NCOMP_LIST}"
      ;;
    prism)
      dataset_ncomp_list="${MID_MULTI_NCOMP_LIST}"
      ;;
  esac

  task_rds="${RESULTS_DIR}/${dataset_id}_task.rds"
  meta_rds="${RESULTS_DIR}/${dataset_id}_task_meta.rds"
  Rscript "${REPO_ROOT}/benchmark/benchmark_dataset_memory_compare.R" \
    --mode=prepare_task \
    --dataset-id="${dataset_id}" \
    --task-rds="${task_rds}" \
    --meta-rds="${meta_rds}" \
    --split-seed="${SPLIT_SEED}" \
    --precision="${PRECISION}"

  variants="$(TASK_META_RDS="${meta_rds}" Rscript -e "source('${REPO_ROOT}/benchmark/helpers_dataset_memory_compare.R'); specs <- variant_specs(); meta <- readRDS(Sys.getenv('TASK_META_RDS')); if (!identical(meta\$task_type, 'classification')) specs <- specs[specs\$classifier == 'argmax', , drop = FALSE]; keep <- trimws(Sys.getenv('FASTPLS_VARIANTS', '')); if (nzchar(keep)) { keep_vec <- trimws(strsplit(keep, ',', fixed = TRUE)[[1L]]); specs <- specs[specs\$variant_name %in% keep_vec, , drop = FALSE] }; cat(paste(specs\$variant_name, collapse=' '))")"

  dataset_reps="${REPS}"
  case "${dataset_id}" in
    cifar100|imagenet|nmr|prism)
      dataset_reps="${LARGE_REPS}"
      ;;
  esac

  for variant_name in ${variants}; do
    variant_timed_out=0
    for requested_ncomp in $(printf '%s' "${dataset_ncomp_list}" | tr ',' ' '); do
      if [ "${variant_timed_out}" -eq 1 ]; then
        echo "[INFO] Skipping ${dataset_id}/${variant_name} ncomp=${requested_ncomp} after earlier timeout for this variant"
        continue
      fi
      rep_id=1
      while [ "${rep_id}" -le "${dataset_reps}" ]; do
        run_id="$(printf '%s__%s__%s__n%s__rep%s' "${dataset_id}" "${PRECISION}" "${variant_name}" "${requested_ncomp}" "${rep_id}")"
        row_csv="${RUN_ROWS_DIR}/${run_id}.csv"
        pid_file="${RUN_ROWS_DIR}/${run_id}.pid"
        fit_ready_file="${RUN_ROWS_DIR}/${run_id}.fit_ready"
        sampler_ready_file="${RUN_ROWS_DIR}/${run_id}.sampler_ready"
        if [ "${SAVE_PREDICTIONS}" = "true" ]; then
          pred_file="${PRED_DIR}/${run_id}.rds"
        else
          pred_file=""
        fi
        stdout_log="${LOG_DIR}/${run_id}.stdout.log"
        time_log="${LOG_DIR}/${run_id}.time.log"
        gpu_log="${GPU_LOG_DIR}/${run_id}.txt"
        run_script="${RUN_ROWS_DIR}/${run_id}.run.sh"

        rm -f "${row_csv}" "${pid_file}" "${fit_ready_file}" "${sampler_ready_file}" "${pred_file}" "${stdout_log}" "${time_log}" "${gpu_log}" "${run_script}"
        cat > "${run_script}" <<EOF
#!/bin/sh
exec Rscript "${REPO_ROOT}/benchmark/benchmark_dataset_memory_compare.R" \
  --mode=run_one \
  --task-rds="${task_rds}" \
  --row-out="${row_csv}" \
  --pid-file="${pid_file}" \
  --fit-ready-file="${fit_ready_file}" \
  --sampler-ready-file="${sampler_ready_file}" \
  --pred-out="${pred_file}" \
  --variant-name="${variant_name}" \
  --lib-loc="${LIB_LOC}" \
  --requested-ncomp="${requested_ncomp}" \
  --replicate="${rep_id}"
EOF
        chmod +x "${run_script}"

        if [ "${RUN_TIMEOUT_SEC}" -gt 0 ] 2>/dev/null; then
          "${TIME_BIN}" ${TIME_ARGS} "${TIMEOUT_BIN}" --signal=TERM --kill-after=30s "${RUN_TIMEOUT_SEC}" "${run_script}" >"${stdout_log}" 2>"${time_log}" &
        else
          "${TIME_BIN}" ${TIME_ARGS} "${run_script}" >"${stdout_log}" 2>"${time_log}" &
        fi
        cmd_pid=$!

        r_pid=""
        sampler_pid=""
        i=0
        while [ "${i}" -lt 600 ]; do
          if [ -s "${pid_file}" ] && [ -s "${fit_ready_file}" ]; then
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
          memory_sampler "${r_pid}" "${gpu_log}" "${sampler_ready_file}" &
          sampler_pid=$!
        fi

        wait "${cmd_pid}" || true
        if [ -n "${sampler_pid}" ]; then
          wait "${sampler_pid}" || true
        fi

        host_rss="$(peak_rss_from_time_log "${time_log}")"
        fit_window_peak_host="$(peak_host_window_from_log "${gpu_log}")"
        gpu_peak="$(peak_gpu_from_log "${gpu_log}")"
        if [ ! -s "${row_csv}" ]; then
          write_missing_row "${row_csv}" "${task_rds}" "${variant_name}" "${requested_ncomp}" "${rep_id}" "${host_rss}" "${gpu_peak}" "${time_log}" "${pred_file}"
        fi
        append_row "${RAW_CSV}" "${row_csv}" "${host_rss}" "${fit_window_peak_host}" "${gpu_peak}"
        row_status="$(row_status_from_csv "${row_csv}")"
        if [ "${row_status}" = "killed_timeout" ]; then
          echo "[INFO] Timeout for ${dataset_id}/${variant_name} ncomp=${requested_ncomp} rep=${rep_id}; skipping remaining reps and higher ncomp for this variant"
          variant_timed_out=1
          break
        fi

        rep_id=$((rep_id + 1))
      done
    done
  done
  if [ "${SKIP_PLOT}" = "true" ]; then
    echo "[INFO] Skipping plot generation after ${dataset_id}"
  else
    Rscript "${REPO_ROOT}/benchmark/plot_dataset_memory_compare.R" "${RESULTS_DIR}"
  fi
done

if [ "${SKIP_PLOT}" = "true" ]; then
  echo "[INFO] Skipping plot generation for ${RESULTS_DIR}"
else
  Rscript "${REPO_ROOT}/benchmark/plot_dataset_memory_compare.R" "${RESULTS_DIR}"
fi
echo "[INFO] Results written to ${RESULTS_DIR}"
