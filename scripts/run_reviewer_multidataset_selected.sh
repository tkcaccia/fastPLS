#!/bin/sh

set -eu

REPO_ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
SELECTION_CSV="${1:?selection CSV is required}"
RESULTS_ROOT="${2:?results directory is required}"
OPLS_SELECTION_CSV="${3:-}"

mkdir -p "${RESULTS_ROOT}"

selection_tsv="${RESULTS_ROOT}/selected_points.tsv"
Rscript - "${SELECTION_CSV}" "${OPLS_SELECTION_CSV}" "${selection_tsv}" <<'RS'
args <- commandArgs(trailingOnly = TRUE)
base <- read.csv(args[[1L]], stringsAsFactors = FALSE)
base <- base[base$status == "ok" & base$method != "opls", , drop = FALSE]
if (nzchar(args[[2L]]) && file.exists(args[[2L]])) {
  opls <- read.csv(args[[2L]], stringsAsFactors = FALSE)
  opls <- opls[opls$status == "ok", , drop = FALSE]
  base <- rbind(base, opls)
}
base <- base[!base$dataset %in% c("nmr", "imagenet"), , drop = FALSE]
base <- base[order(base$dataset, base$method), , drop = FALSE]
write.table(
  base[c("dataset", "method", "best_ncomp")],
  args[[3L]],
  sep = "\t",
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)
RS

while IFS="$(printf '\t')" read -r dataset method ncomp; do
  case "${method}" in
    plssvd)
      cpu_variant="cpp_plssvd_cpu_rsvd"
      gpu_variant="gpu_plssvd_rsvd"
      ;;
    simpls)
      cpu_variant="cpp_simpls_cpu_rsvd"
      gpu_variant="gpu_simpls_rsvd"
      ;;
    opls)
      cpu_variant="cpp_opls_cpu_rsvd"
      gpu_variant="gpu_opls_rsvd"
      ;;
    kernelpls)
      cpu_variant="cpp_kernelpls_cpu_rsvd"
      gpu_variant="gpu_kernelpls_rsvd"
      ;;
    *)
      echo "Unsupported method: ${method}" >&2
      exit 1
      ;;
  esac
  case "${FASTPLS_SELECTED_ENGINE:-both}" in
    cpu) variants="${cpu_variant}" ;;
    gpu) variants="${gpu_variant}" ;;
    both) variants="${cpu_variant},${gpu_variant}" ;;
    *)
      echo "FASTPLS_SELECTED_ENGINE must be cpu, gpu, or both" >&2
      exit 1
      ;;
  esac

  run_dir="${RESULTS_ROOT}/runs/${dataset}_${method}"
  echo "[SELECTED] dataset=${dataset} method=${method} ncomp=${ncomp}"
  FASTPLS_RESULTS_DIR="${run_dir}" \
  FASTPLS_BENCH_LIB="${FASTPLS_BENCH_LIB:-${HOME}/R/x86_64-pc-linux-gnu-library/4.5}" \
  FASTPLS_DATASETS="${dataset}" \
  FASTPLS_NCOMP_LIST="${ncomp}" \
  FASTPLS_METREF_NCOMP_LIST="${ncomp}" \
  FASTPLS_CCLE_NCOMP_LIST="${ncomp}" \
  FASTPLS_CIFAR100_NCOMP_LIST="${ncomp}" \
  FASTPLS_NMR_NCOMP_LIST="${ncomp}" \
  FASTPLS_SMALL_MULTI_NCOMP_LIST="${ncomp}" \
  FASTPLS_MID_MULTI_NCOMP_LIST="${ncomp}" \
  FASTPLS_GTEX_V8_NCOMP_LIST="${ncomp}" \
  FASTPLS_TCGA_PAN_CANCER_NCOMP_LIST="${ncomp}" \
  FASTPLS_VARIANTS="${variants}" \
  FASTPLS_BENCH_PRECISION="float64" \
  FASTPLS_COMPARE_REPS="3" \
  FASTPLS_COMPARE_LARGE_REPS="1" \
  FASTPLS_RUN_TIMEOUT_SEC="1200" \
  FASTPLS_SKIP_PLOT="true" \
  sh "${REPO_ROOT}/scripts/remote_run_dataset_memory_compare.sh"
done < "${selection_tsv}"

Rscript - "${RESULTS_ROOT}" <<'RS'
root <- commandArgs(trailingOnly = TRUE)[[1L]]
files <- list.files(file.path(root, "runs"), "dataset_memory_compare_raw[.]csv$",
                    recursive = TRUE, full.names = TRUE)
rows <- do.call(rbind, lapply(files, function(path) {
  x <- read.csv(path, stringsAsFactors = FALSE)
  x$source_file <- path
  x
}))
write.csv(rows, file.path(root, "multidataset_selected_raw.csv"), row.names = FALSE)
RS
