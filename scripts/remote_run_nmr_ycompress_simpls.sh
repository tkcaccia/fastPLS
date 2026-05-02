#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"

RESULTS_DIR="${FASTPLS_RESULTS_DIR:-${REPO_ROOT}/benchmark_results_nmr_ycompress_simpls}"
LIB_LOC="${FASTPLS_BENCH_LIB:-${HOME}/R/fastpls_bench_flash_standard}"
NCOMP="${FASTPLS_YCOMPRESS_NCOMP:-50}"
CODE_DIM="${FASTPLS_YCOMPRESS_DIM:-50}"
VARIANTS="${FASTPLS_YCOMPRESS_VARIANTS:-cpp_rsvd,cpp_irlba,cuda}"
RESPONSE_MODES="${FASTPLS_YCOMPRESS_MODES:-full_y,y_pca50,gaussian50,orthogonal_random50}"
SPLIT_SEED="${FASTPLS_SPLIT_SEED:-123}"
RUN_TIMEOUT_SEC="${FASTPLS_RUN_TIMEOUT_SEC:-1200}"
TIME_BIN="${TIME_BIN:-/usr/bin/time}"
TIMEOUT_BIN="${TIMEOUT_BIN:-timeout}"

RAW_CSV="${RESULTS_DIR}/nmr_ycompress_simpls_raw.csv"
SUMMARY_CSV="${RESULTS_DIR}/nmr_ycompress_simpls_summary.csv"
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

append_row() {
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
    rows = [{"dataset": "nmr", "method": "simpls", "status": "error", "msg": "Rscript did not produce a row"}]
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
    writer.writerows(rows)
PY
}

for variant in $(printf '%s' "${VARIANTS}" | tr ',' ' '); do
  for response_mode in $(printf '%s' "${RESPONSE_MODES}" | tr ',' ' '); do
    run_id="$(printf 'nmr__%s__%s' "${variant}" "${response_mode}")"
    out_subdir="${RUN_DIR}/${run_id}"
    stdout_log="${LOG_DIR}/${run_id}.stdout.log"
    time_log="${LOG_DIR}/${run_id}.time.log"
    rm -rf "${out_subdir}"
    mkdir -p "${out_subdir}"
    echo "[RUN] ${run_id}"
    if [ "${RUN_TIMEOUT_SEC}" -gt 0 ] 2>/dev/null; then
      "${TIME_BIN}" -v "${TIMEOUT_BIN}" --signal=TERM --kill-after=30s "${RUN_TIMEOUT_SEC}" \
        Rscript "${REPO_ROOT}/benchmark/benchmark_nmr_ycompress_simpls.R" \
          --out-dir="${out_subdir}" \
          --lib-loc="${LIB_LOC}" \
          --ncomp="${NCOMP}" \
          --code-dim="${CODE_DIM}" \
          --variant="${variant}" \
          --response-mode="${response_mode}" \
          --split-seed="${SPLIT_SEED}" >"${stdout_log}" 2>"${time_log}" || true
    else
      "${TIME_BIN}" -v \
        Rscript "${REPO_ROOT}/benchmark/benchmark_nmr_ycompress_simpls.R" \
          --out-dir="${out_subdir}" \
          --lib-loc="${LIB_LOC}" \
          --ncomp="${NCOMP}" \
          --code-dim="${CODE_DIM}" \
          --variant="${variant}" \
          --response-mode="${response_mode}" \
          --split-seed="${SPLIT_SEED}" >"${stdout_log}" 2>"${time_log}" || true
    fi
    host_rss="$(peak_rss_from_time_log "${time_log}")"
    append_row "${out_subdir}/nmr_ycompress_row.csv" "${RAW_CSV}" "${host_rss}" "${stdout_log}" "${time_log}"
  done
done

Rscript - "${RAW_CSV}" "${SUMMARY_CSV}" "${RESULTS_DIR}" <<'RS'
args <- commandArgs(trailingOnly = TRUE)
raw_csv <- args[[1L]]
summary_csv <- args[[2L]]
results_dir <- args[[3L]]
raw <- utils::read.csv(raw_csv, check.names = FALSE)
utils::write.csv(raw, summary_csv, row.names = FALSE)
print(raw[, intersect(c("variant", "response_mode", "prep_time_ms", "fit_time_ms", "predict_time_ms", "reconstruct_time_ms", "total_time_ms", "rmsd", "q2_global", "mae", "peak_host_rss_mb", "status", "msg"), names(raw)), drop = FALSE])

if (requireNamespace("ggplot2", quietly = TRUE)) {
  ok <- raw[raw$status == "ok", , drop = FALSE]
  if (nrow(ok)) {
    plot_dir <- file.path(results_dir, "plots")
    dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)
    ok$response_mode <- factor(ok$response_mode, levels = unique(ok$response_mode))
    p_time <- ggplot2::ggplot(ok, ggplot2::aes(x = response_mode, y = total_time_ms, fill = variant)) +
      ggplot2::geom_col(position = "dodge") +
      ggplot2::coord_flip() +
      ggplot2::labs(title = "NMR SIMPLS ncomp=50: Y compression total time", x = NULL, y = "Total time (ms)") +
      ggplot2::theme_bw(base_size = 13)
    p_rmsd <- ggplot2::ggplot(ok, ggplot2::aes(x = response_mode, y = rmsd, fill = variant)) +
      ggplot2::geom_col(position = "dodge") +
      ggplot2::coord_flip() +
      ggplot2::labs(title = "NMR SIMPLS ncomp=50: reconstructed-Y RMSD", x = NULL, y = "RMSD") +
      ggplot2::theme_bw(base_size = 13)
    p_q2 <- ggplot2::ggplot(ok, ggplot2::aes(x = response_mode, y = q2_global, fill = variant)) +
      ggplot2::geom_col(position = "dodge") +
      ggplot2::coord_flip() +
      ggplot2::labs(title = "NMR SIMPLS ncomp=50: reconstructed-Y Q2", x = NULL, y = "Q2 global") +
      ggplot2::theme_bw(base_size = 13)
    p_mem <- ggplot2::ggplot(ok, ggplot2::aes(x = response_mode, y = peak_host_rss_mb, fill = variant)) +
      ggplot2::geom_col(position = "dodge") +
      ggplot2::coord_flip() +
      ggplot2::labs(title = "NMR SIMPLS ncomp=50: peak RSS", x = NULL, y = "Peak host RSS (MB)") +
      ggplot2::theme_bw(base_size = 13)
    ggplot2::ggsave(file.path(plot_dir, "nmr_ycompress_time.png"), p_time, width = 9, height = 5, dpi = 160)
    ggplot2::ggsave(file.path(plot_dir, "nmr_ycompress_rmsd.png"), p_rmsd, width = 9, height = 5, dpi = 160)
    ggplot2::ggsave(file.path(plot_dir, "nmr_ycompress_q2.png"), p_q2, width = 9, height = 5, dpi = 160)
    ggplot2::ggsave(file.path(plot_dir, "nmr_ycompress_peak_rss.png"), p_mem, width = 9, height = 5, dpi = 160)
  }
}
RS

echo "[INFO] Results written to ${RESULTS_DIR}"
