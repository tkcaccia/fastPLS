#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"

RESULTS_DIR="${FASTPLS_RESULTS_DIR:-${REPO_ROOT}/benchmark_results_singlecell_ecoc_simpls}"
LIB_LOC="${FASTPLS_BENCH_LIB:-${HOME}/R/fastpls_bench_flash_standard}"
NCOMP="${FASTPLS_ECOC_NCOMP:-50}"
CODE_DIM="${FASTPLS_ECOC_DIM:-50}"
REPS="${FASTPLS_ECOC_REPS:-1}"
SPLIT_SEED="${FASTPLS_SPLIT_SEED:-123}"
VARIANTS="${FASTPLS_ECOC_VARIANTS:-cpp_rsvd,cpp_irlba,r_rsvd,r_irlba,cuda}"
RESPONSE_MODES="${FASTPLS_ECOC_RESPONSE_MODES:-onehot,random_ecoc50,balanced_ecoc50,hadamard_ecoc50,gaussian50,orthogonal_random50,centroid_pca50}"
RUN_TIMEOUT_SEC="${FASTPLS_RUN_TIMEOUT_SEC:-1200}"
TIME_BIN="${TIME_BIN:-/usr/bin/time}"
TIMEOUT_BIN="${TIMEOUT_BIN:-timeout}"

RAW_CSV="${RESULTS_DIR}/singlecell_ecoc_simpls_raw.csv"
SUMMARY_CSV="${RESULTS_DIR}/singlecell_ecoc_simpls_summary.csv"
LOG_DIR="${RESULTS_DIR}/logs"
RUN_DIR="${RESULTS_DIR}/runs"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}" "${RUN_DIR}"
rm -f "${RAW_CSV}" "${SUMMARY_CSV}"

peak_rss_from_time_log() {
  time_log="$1"
  rss_kb="$(awk -F: '/Maximum resident set size/ {gsub(/^[ \t]+/, "", $2); print $2; exit}' "${time_log}" 2>/dev/null || true)"
  if [ -z "${rss_kb}" ]; then
    echo "NA"
  else
    python3 - <<PY
rss_kb = float("${rss_kb}")
print(round(rss_kb / 1024.0, 3))
PY
  fi
}

append_run_row() {
  row_csv="$1"
  raw_csv="$2"
  host_rss="$3"
  stdout_log="$4"
  time_log="$5"
  python3 - "$row_csv" "$raw_csv" "$host_rss" "$stdout_log" "$time_log" <<'PY'
import csv
import os
import sys

row_csv, raw_csv, host_rss, stdout_log, time_log = sys.argv[1:]
rows = []
if os.path.exists(row_csv) and os.path.getsize(row_csv):
    with open(row_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
if not rows:
    rows = [{
        "dataset": "singlecell",
        "variant": "",
        "response_mode": "",
        "method": "simpls",
        "implementation": "",
        "engine": "",
        "backend": "",
        "replicate": "",
        "ncomp": "",
        "code_dim": "",
        "n_train": "",
        "n_test": "",
        "p": "",
        "n_classes": "",
        "fit_time_ms": "",
        "predict_time_ms": "",
        "total_time_ms": "",
        "accuracy": "",
        "macro_f1": "",
        "status": "error",
        "msg": "Rscript did not produce a row"
    }]
for row in rows:
    row["peak_host_rss_mb"] = "" if host_rss == "NA" else host_rss
    row["stdout_log"] = stdout_log
    row["time_log"] = time_log

fieldnames = list(rows[0].keys())
need_header = not os.path.exists(raw_csv) or os.path.getsize(raw_csv) == 0
with open(raw_csv, "a", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if need_header:
        writer.writeheader()
    for row in rows:
        writer.writerow(row)
PY
}

for variant in $(printf '%s' "${VARIANTS}" | tr ',' ' '); do
  for response_mode in $(printf '%s' "${RESPONSE_MODES}" | tr ',' ' '); do
    rep_id=1
    while [ "${rep_id}" -le "${REPS}" ]; do
      run_id="$(printf 'singlecell__%s__%s__rep%s' "${variant}" "${response_mode}" "${rep_id}")"
      out_subdir="${RUN_DIR}/${run_id}"
      stdout_log="${LOG_DIR}/${run_id}.stdout.log"
      time_log="${LOG_DIR}/${run_id}.time.log"
      rm -rf "${out_subdir}"
      mkdir -p "${out_subdir}"
      echo "[RUN] ${run_id}"
      if [ "${RUN_TIMEOUT_SEC}" -gt 0 ] 2>/dev/null; then
        "${TIME_BIN}" -v "${TIMEOUT_BIN}" --signal=TERM --kill-after=30s "${RUN_TIMEOUT_SEC}" \
          Rscript "${REPO_ROOT}/benchmark/benchmark_singlecell_ecoc_simpls.R" \
            --out-dir="${out_subdir}" \
            --lib-loc="${LIB_LOC}" \
            --ncomp="${NCOMP}" \
            --code-dim="${CODE_DIM}" \
            --reps=1 \
            --split-seed="${SPLIT_SEED}" \
            --variants="${variant}" \
            --response-modes="${response_mode}" \
            --timeout-note="${RUN_TIMEOUT_SEC}s" >"${stdout_log}" 2>"${time_log}" || true
      else
        "${TIME_BIN}" -v \
          Rscript "${REPO_ROOT}/benchmark/benchmark_singlecell_ecoc_simpls.R" \
            --out-dir="${out_subdir}" \
            --lib-loc="${LIB_LOC}" \
            --ncomp="${NCOMP}" \
            --code-dim="${CODE_DIM}" \
            --reps=1 \
            --split-seed="${SPLIT_SEED}" \
            --variants="${variant}" \
            --response-modes="${response_mode}" >"${stdout_log}" 2>"${time_log}" || true
      fi
      host_rss="$(peak_rss_from_time_log "${time_log}")"
      append_run_row "${out_subdir}/singlecell_ecoc_simpls_raw.csv" "${RAW_CSV}" "${host_rss}" "${stdout_log}" "${time_log}"
      rep_id=$((rep_id + 1))
    done
  done
done

Rscript - "${RAW_CSV}" "${SUMMARY_CSV}" "${RESULTS_DIR}" <<'RS'
args <- commandArgs(trailingOnly = TRUE)
raw_csv <- args[[1L]]
summary_csv <- args[[2L]]
results_dir <- args[[3L]]
raw <- utils::read.csv(raw_csv, check.names = FALSE)
ok <- raw[raw$status == "ok", , drop = FALSE]
if (nrow(ok)) {
  summary <- aggregate(
    cbind(fit_time_ms, predict_time_ms, total_time_ms, accuracy, macro_f1, peak_host_rss_mb) ~
      variant + response_mode + implementation + engine + backend + status,
    ok,
    function(x) median(x, na.rm = TRUE)
  )
} else {
  summary <- raw[0, , drop = FALSE]
}
utils::write.csv(summary, summary_csv, row.names = FALSE)
print(summary)

if (nrow(ok) && requireNamespace("ggplot2", quietly = TRUE)) {
  plot_dir <- file.path(results_dir, "plots")
  dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)
  ok$label <- paste(ok$implementation, ok$backend, ok$response_mode, sep = " / ")
  p1 <- ggplot2::ggplot(ok, ggplot2::aes(x = label, y = total_time_ms, fill = response_mode)) +
    ggplot2::geom_col(position = "dodge") +
    ggplot2::coord_flip() +
    ggplot2::labs(title = "singlecell SIMPLS ncomp=50: one-hot vs ECOC-50", x = NULL, y = "Total time (ms)") +
    ggplot2::theme_bw(base_size = 13)
  p2 <- ggplot2::ggplot(ok, ggplot2::aes(x = label, y = accuracy, fill = response_mode)) +
    ggplot2::geom_col(position = "dodge") +
    ggplot2::coord_flip() +
    ggplot2::labs(title = "singlecell SIMPLS ncomp=50: accuracy", x = NULL, y = "Accuracy") +
    ggplot2::theme_bw(base_size = 13)
  p3 <- ggplot2::ggplot(ok, ggplot2::aes(x = label, y = peak_host_rss_mb, fill = response_mode)) +
    ggplot2::geom_col(position = "dodge") +
    ggplot2::coord_flip() +
    ggplot2::labs(title = "singlecell SIMPLS ncomp=50: peak RSS", x = NULL, y = "Peak host RSS (MB)") +
    ggplot2::theme_bw(base_size = 13)
  ggplot2::ggsave(file.path(plot_dir, "singlecell_ecoc_time.png"), p1, width = 9, height = 5, dpi = 160)
  ggplot2::ggsave(file.path(plot_dir, "singlecell_ecoc_accuracy.png"), p2, width = 9, height = 5, dpi = 160)
  ggplot2::ggsave(file.path(plot_dir, "singlecell_ecoc_peak_rss.png"), p3, width = 9, height = 5, dpi = 160)
}
RS

echo "[INFO] Results written to ${RESULTS_DIR}"
