#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="${FASTPLS_EXTERNAL_TIMING_RESULTS_DIR:-${REPO_ROOT}/benchmark_results/external_simpls_timing}"
DATASETS="${FASTPLS_EXTERNAL_TIMING_DATASETS:-metref,ccle,tcga_brca,tcga_hnsc_methylation,gtex_v8,tcga_pan_cancer,retina,tabula,cifar100}"
PROFILES="${FASTPLS_EXTERNAL_TIMING_PROFILES:-estimator_kernel,complete_workflow}"
IMPLEMENTATIONS="${FASTPLS_EXTERNAL_TIMING_IMPLEMENTATIONS:-fastpls,pls}"
REPS="${FASTPLS_EXTERNAL_TIMING_REPS:-3}"
TIMEOUT_SEC="${FASTPLS_EXTERNAL_TIMING_TIMEOUT_SEC:-10000}"
TIMEOUT_BIN="${TIMEOUT_BIN:-timeout}"
TIME_BIN="${TIME_BIN:-/usr/bin/time}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p "${RESULTS_DIR}/rows" "${RESULTS_DIR}/logs"
cat >"${RESULTS_DIR}/benchmark_parameters.txt" <<EOF
datasets=${DATASETS}
profiles=${PROFILES}
implementations=${IMPLEMENTATIONS}
repetitions=${REPS}
timeout_sec=${TIMEOUT_SEC}
warmup=none
package_and_data_loading_timed=false
rss_baseline=immediately_before_fit_after_gc
rss_peak=maximum_process_rss_over_worker_lifetime
rss_increment=maximum_process_rss_minus_prefit_process_rss
rss_increment_scope=baseline_corrected_process_increment_not_algorithmic_workspace
precision=float64
blas_threads=1
split_seed=123
EOF

annotate_resources() {
  local row="$1"
  local resources="$2"
  [ -s "${row}" ] || return 0
  Rscript - "${row}" "${resources}" <<'RS'
args <- commandArgs(TRUE)
d <- read.csv(args[[1L]], check.names = FALSE)
text <- if (file.exists(args[[2L]])) readLines(args[[2L]], warn = FALSE) else character()
value <- function(label) {
  hit <- grep(paste0("^[[:space:]]*", label, ":[[:space:]]*"), text, value = TRUE)
  if (!length(hit)) return(NA_real_)
  suppressWarnings(as.numeric(sub(".*:[[:space:]]*", "", hit[[length(hit)]])))
}
d$process_peak_rss_mb <- value("Maximum resident set size \\(kbytes\\)") / 1024
d$user_cpu_sec <- value("User time \\(seconds\\)")
d$system_cpu_sec <- value("System time \\(seconds\\)")
write.csv(d, args[[1L]], row.names = FALSE, quote = TRUE, na = "")
RS
}

ncomp_for_dataset() {
  case "$1" in
    metref) echo 22 ;;
    ccle) echo 50 ;;
    tcga_brca) echo 5 ;;
    tcga_hnsc_methylation) echo 2 ;;
    gtex_v8) echo 32 ;;
    tcga_pan_cancer) echo 50 ;;
    retina) echo 50 ;;
    tabula) echo 50 ;;
    cifar100) echo 100 ;;
    *) echo 50 ;;
  esac
}

for dataset in $(printf '%s' "${DATASETS}" | tr ',' ' '); do
  ncomp="$(ncomp_for_dataset "${dataset}")"
  for profile in $(printf '%s' "${PROFILES}" | tr ',' ' '); do
    for implementation in $(printf '%s' "${IMPLEMENTATIONS}" | tr ',' ' '); do
      for replicate in $(seq 1 "${REPS}"); do
        id="${dataset}__${profile}__${implementation}__rep${replicate}"
        row="${RESULTS_DIR}/rows/${id}.csv"
        log="${RESULTS_DIR}/logs/${id}.log"
        resources="${RESULTS_DIR}/logs/${id}.resources.txt"
        echo "[RUN] ${id}"
        set +e
        "${TIME_BIN}" -v "${TIMEOUT_BIN}" --signal=TERM --kill-after=30s "${TIMEOUT_SEC}" \
          Rscript "${REPO_ROOT}/benchmark/external_simpls_timing/worker.R" \
            --dataset="${dataset}" --profile="${profile}" \
            --implementation="${implementation}" --ncomp="${ncomp}" \
            --replicate="${replicate}" --seed=123 --timeout-sec="${TIMEOUT_SEC}" \
            --row-out="${row}" >"${log}" 2>"${resources}"
        code=$?
        set -e
        if [ "${code}" -ne 0 ] && [ ! -s "${row}" ]; then
          status="failed_process_${code}"
          if [ "${code}" -eq 124 ]; then status="timeout"; fi
          Rscript - "${row}" "${dataset}" "${profile}" "${implementation}" "${replicate}" "${status}" <<'RS'
args <- commandArgs(TRUE)
dir.create(dirname(args[[1L]]), recursive = TRUE, showWarnings = FALSE)
write.csv(data.frame(dataset=args[[2L]],comparison_profile=args[[3L]],implementation=args[[4L]],replicate=as.integer(args[[5L]]),status=args[[6L]],error_message=args[[6L]]),args[[1L]],row.names=FALSE)
RS
        fi
        annotate_resources "${row}" "${resources}"
      done
    done
  done
done

Rscript "${REPO_ROOT}/benchmark/external_simpls_timing/summarize.R" "${RESULTS_DIR}"
echo "[DONE] ${RESULTS_DIR}"
