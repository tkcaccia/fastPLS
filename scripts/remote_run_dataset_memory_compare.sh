#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"

RESULTS_DIR="${FASTPLS_RESULTS_DIR:-${REPO_ROOT}/benchmark_results_dataset_memory_compare}"
LIB_LOC="${FASTPLS_BENCH_LIB:-${HOME}/R/fastpls_bench_fresh}"
DATASETS="${FASTPLS_DATASETS:-cifar100,ccle}"
NCOMP_LIST="${FASTPLS_NCOMP_LIST:-2,5,10,18,20,50}"
METREF_NCOMP_LIST="${FASTPLS_METREF_NCOMP_LIST:-2,5,10,22,50,100}"
CCLE_NCOMP_LIST="${FASTPLS_CCLE_NCOMP_LIST:-2,5,10,18,50,100}"
CIFAR100_NCOMP_LIST="${FASTPLS_CIFAR100_NCOMP_LIST:-2,5,10,20,50,100,200}"
NMR_NCOMP_LIST="${FASTPLS_NMR_NCOMP_LIST:-2,5,10,20,50,100,200,500}"
SMALL_MULTI_NCOMP_LIST="${FASTPLS_SMALL_MULTI_NCOMP_LIST:-2,5,10,20,50}"
MID_MULTI_NCOMP_LIST="${FASTPLS_MID_MULTI_NCOMP_LIST:-2,5,10,20,50,100}"
REPS="${FASTPLS_COMPARE_REPS:-3}"
SPLIT_SEED="${FASTPLS_SPLIT_SEED:-123}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TIME_BIN="${TIME_BIN:-/usr/bin/time}"
VARIANTS_FILTER="${FASTPLS_VARIANTS:-}"
SKIP_PLOT="${FASTPLS_SKIP_PLOT:-false}"

RAW_CSV="${RESULTS_DIR}/dataset_memory_compare_raw.csv"
RUN_ROWS_DIR="${RESULTS_DIR}/run_rows"
GPU_LOG_DIR="${RESULTS_DIR}/gpu_samples"
PRED_DIR="${RESULTS_DIR}/predictions"
LOG_DIR="${RESULTS_DIR}/logs"

mkdir -p "${RESULTS_DIR}" "${RUN_ROWS_DIR}" "${GPU_LOG_DIR}" "${PRED_DIR}" "${LOG_DIR}"
rm -f "${RAW_CSV}"

gpu_sampler() {
  r_pid="$1"
  log_file="$2"
  while kill -0 "${r_pid}" 2>/dev/null; do
    samples="$(nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | \
      awk -F',' -v pid="${r_pid}" '($1 + 0) == pid {gsub(/ /, "", $2); print $2}')"
    if [ -n "${samples}" ]; then
      printf '%s\n' "${samples}" | while IFS= read -r mb; do
        [ -n "${mb}" ] && printf 'pid,%s\n' "${mb}" >> "${log_file}"
      done
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
    echo "NA"
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
  host_rss="$3"
  gpu_peak="$4"

  "${PYTHON_BIN}" - "$raw_csv" "$row_csv" "$host_rss" "$gpu_peak" <<'PY'
import csv, os, sys

raw_csv, row_csv, host_rss, gpu_peak = sys.argv[1:]

with open(row_csv, newline="") as fh:
    reader = csv.DictReader(fh)
    row = next(reader)

if host_rss not in ("", "NA"):
    row["peak_host_rss_mb"] = host_rss
if gpu_peak not in ("", "NA"):
    row["peak_gpu_mem_mb"] = gpu_peak

need_header = not os.path.exists(raw_csv) or os.path.getsize(raw_csv) == 0
with open(raw_csv, "a", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
    if need_header:
        writer.writeheader()
    writer.writerow(row)
PY
}

variants="$(Rscript -e "source('${REPO_ROOT}/benchmark/helpers_dataset_memory_compare.R'); specs <- variant_specs(); keep <- trimws(Sys.getenv('FASTPLS_VARIANTS', '')); if (nzchar(keep)) { keep_vec <- trimws(strsplit(keep, ',', fixed = TRUE)[[1L]]); specs <- specs[specs\$variant_name %in% keep_vec, , drop = FALSE] }; cat(paste(specs\$variant_name, collapse=' '))")"

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
    nmr)
      dataset_ncomp_list="${NMR_NCOMP_LIST}"
      ;;
    singlecell|tcga_brca|tcga_hnsc_methylation)
      dataset_ncomp_list="${SMALL_MULTI_NCOMP_LIST}"
      ;;
    tcga_pan_cancer|gtex_v8|prism)
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
    --split-seed="${SPLIT_SEED}"

  for variant_name in ${variants}; do
    for requested_ncomp in $(printf '%s' "${dataset_ncomp_list}" | tr ',' ' '); do
      rep_id=1
      while [ "${rep_id}" -le "${REPS}" ]; do
        run_id="$(printf '%s__%s__n%s__rep%s' "${dataset_id}" "${variant_name}" "${requested_ncomp}" "${rep_id}")"
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
exec Rscript "${REPO_ROOT}/benchmark/benchmark_dataset_memory_compare.R" \
  --mode=run_one \
  --task-rds="${task_rds}" \
  --row-out="${row_csv}" \
  --pid-file="${pid_file}" \
  --pred-out="${pred_file}" \
  --variant-name="${variant_name}" \
  --lib-loc="${LIB_LOC}" \
  --requested-ncomp="${requested_ncomp}" \
  --replicate="${rep_id}"
EOF
        chmod +x "${run_script}"

        "${TIME_BIN}" -v "${run_script}" >"${stdout_log}" 2>"${time_log}" &
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

        wait "${cmd_pid}" || true
        if [ -n "${sampler_pid}" ]; then
          wait "${sampler_pid}" || true
        fi

        host_rss="$(peak_rss_from_time_log "${time_log}")"
        gpu_peak="$(peak_gpu_from_log "${gpu_log}")"
        append_row "${RAW_CSV}" "${row_csv}" "${host_rss}" "${gpu_peak}"

        rep_id=$((rep_id + 1))
      done
    done
  done
done

if [ "${SKIP_PLOT}" = "true" ]; then
  echo "[INFO] Skipping plot generation for ${RESULTS_DIR}"
else
  Rscript "${REPO_ROOT}/benchmark/plot_dataset_memory_compare.R" "${RESULTS_DIR}"
fi
echo "[INFO] Results written to ${RESULTS_DIR}"
