#!/bin/sh

set -eu

REPO_ROOT="${FASTPLS_REPO_ROOT:-$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)}"
SELECTION_CSV="${FASTPLS_SELECTION_CSV:-${REPO_ROOT}/paired_backend_selected_summary.csv}"
RESULTS_ROOT="${FASTPLS_RESULTS_ROOT:-${REPO_ROOT}/selected_memory_baseline}"
WORK_ROOT="${FASTPLS_MEMORY_WORK_ROOT:-${RESULTS_ROOT}/work}"
LIB_LOC="${FASTPLS_BENCH_LIB:-${HOME}/R/x86_64-pc-linux-gnu-library/4.5}"
REPS="${FASTPLS_MEMORY_REPS:-3}"
PRECISION="${FASTPLS_BENCH_PRECISION:-float64}"
RUN_TIMEOUT_SEC="${FASTPLS_RUN_TIMEOUT_SEC:-1200}"

mkdir -p "${RESULTS_ROOT}" "${WORK_ROOT}"
grid_file="${RESULTS_ROOT}/selected_grid.tsv"

SELECTION_CSV="${SELECTION_CSV}" Rscript - "${grid_file}" <<'RS'
input <- Sys.getenv("SELECTION_CSV")
output <- commandArgs(trailingOnly = TRUE)[[1L]]
x <- utils::read.csv(input, stringsAsFactors = FALSE)
x <- x[
  x$status == "ok" &
    nzchar(x$variant_name) &
    is.finite(x$effective_ncomp),
  c("dataset", "method_panel", "engine", "variant_name", "effective_ncomp"),
  drop = FALSE
]
x$variant_name <- sub(
  "^fastpls_plssvd_cpu_rsvd$", "cpp_plssvd_cpu_rsvd", x$variant_name
)
x$variant_name <- sub(
  "^fastpls_plssvd_cuda_rsvd$", "gpu_plssvd_rsvd", x$variant_name
)
x$variant_name <- sub(
  "^fastpls_simpls_cpu_rsvd$", "cpp_simpls_cpu_rsvd", x$variant_name
)
x$variant_name <- sub(
  "^fastpls_simpls_cuda_rsvd$", "gpu_simpls_rsvd", x$variant_name
)
x <- unique(x)
x <- x[order(x$dataset, x$method_panel, x$engine), , drop = FALSE]
utils::write.table(
  x, output, sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE
)
RS

while IFS="$(printf '\t')" read -r dataset method engine variant ncomp; do
  final_dir="${RESULTS_ROOT}/${dataset}__${method}__${engine}"
  run_dir="${WORK_ROOT}/${dataset}__${method}__${engine}"
  if [ -s "${final_dir}/dataset_memory_compare_raw.csv" ]; then
    completed="$(Rscript -e "x<-read.csv('${final_dir}/dataset_memory_compare_raw.csv'); cat(sum(x\$status == 'ok'))")"
    if [ "${completed}" -ge "${REPS}" ]; then
      echo "[SELECTED MEMORY] already complete: dataset=${dataset} method=${method} engine=${engine}"
      continue
    fi
  fi
  echo "[SELECTED MEMORY] dataset=${dataset} method=${method} engine=${engine} ncomp=${ncomp}"
  rm -rf "${run_dir}"
  FASTPLS_RESULTS_DIR="${run_dir}" \
  FASTPLS_BENCH_LIB="${LIB_LOC}" \
  FASTPLS_DATASETS="${dataset}" \
  FASTPLS_NCOMP_LIST="${ncomp}" \
  FASTPLS_METREF_NCOMP_LIST="${ncomp}" \
  FASTPLS_CCLE_NCOMP_LIST="${ncomp}" \
  FASTPLS_CIFAR100_NCOMP_LIST="${ncomp}" \
  FASTPLS_IMAGENET_NCOMP_LIST="${ncomp}" \
  FASTPLS_NMR_NCOMP_LIST="${ncomp}" \
  FASTPLS_SMALL_MULTI_NCOMP_LIST="${ncomp}" \
  FASTPLS_MID_MULTI_NCOMP_LIST="${ncomp}" \
  FASTPLS_GTEX_V8_NCOMP_LIST="${ncomp}" \
  FASTPLS_TCGA_PAN_CANCER_NCOMP_LIST="${ncomp}" \
  FASTPLS_VARIANTS="${variant}" \
  FASTPLS_COMPARE_REPS="${REPS}" \
  FASTPLS_COMPARE_LARGE_REPS="${REPS}" \
  FASTPLS_BENCH_PRECISION="${PRECISION}" \
  FASTPLS_RUN_TIMEOUT_SEC="${RUN_TIMEOUT_SEC}" \
  FASTPLS_SKIP_PLOT=true \
  sh "${REPO_ROOT}/scripts/remote_run_dataset_memory_compare.sh"
  mkdir -p "${final_dir}"
  cp "${run_dir}/dataset_memory_compare_raw.csv" "${final_dir}/"
  rm -rf "${run_dir}"
done < "${grid_file}"

find "${RESULTS_ROOT}" -mindepth 2 -maxdepth 2 -name dataset_memory_compare_raw.csv \
  -print | sort > "${RESULTS_ROOT}/raw_files.txt"

RAW_FILES="${RESULTS_ROOT}/raw_files.txt" Rscript - "${RESULTS_ROOT}/selected_memory_raw.csv" <<'RS'
paths <- readLines(Sys.getenv("RAW_FILES"), warn = FALSE)
rows <- lapply(paths, function(path) {
  x <- utils::read.csv(path, stringsAsFactors = FALSE)
  x$source_file <- path
  x
})
utils::write.csv(do.call(rbind, rows), commandArgs(trailingOnly = TRUE)[[1L]], row.names = FALSE)
RS

echo "[SELECTED MEMORY] Results written to ${RESULTS_ROOT}"
