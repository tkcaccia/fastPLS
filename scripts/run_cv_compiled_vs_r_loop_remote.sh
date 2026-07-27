#!/usr/bin/env bash

set -euo pipefail

RESULTS_DIR="${1:-$HOME/fastPLS_cv_compiled_vs_r_loop}"
REPO_ROOT="${FASTPLS_REPO_ROOT:-$HOME/fastPLS}"
REPS="${FASTPLS_CV_COMPARE_REPS:-3}"
TIMEOUT_SEC="${FASTPLS_CV_COMPARE_TIMEOUT_SEC:-7200}"

mkdir -p "${RESULTS_DIR}/rows" "${RESULTS_DIR}/logs"

# task|dataset|ncomp|method|backend|svd|classifier
CASES=(
  "$HOME/fastPLS_cycle9_plssvd_vs_simpls_20260723/results/metref_task.rds|metref|22|simpls|cpu|irlba|argmax"
  "$HOME/fastPLS_cycle9_plssvd_vs_simpls_20260723/results/metref_task.rds|metref|22|simpls|cpu|rsvd|argmax"
  "$HOME/fastPLS_cycle9_plssvd_vs_simpls_20260723/results/metref_task.rds|metref|22|simpls|cuda|rsvd|argmax"
  "$HOME/fastPLS_revision_cycle13_20260725/retina_tabula_selected_outer/runs/retina_simpls/retina_task.rds|retina|20|simpls|cpu|irlba|argmax"
  "$HOME/fastPLS_revision_cycle13_20260725/retina_tabula_selected_outer/runs/retina_simpls/retina_task.rds|retina|20|simpls|cpu|rsvd|argmax"
  "$HOME/fastPLS_cycle9_plssvd_vs_simpls_20260723/results/prism_task.rds|prism|5|simpls|cpu|irlba|argmax"
  "$HOME/fastPLS_cycle9_plssvd_vs_simpls_20260723/results/prism_task.rds|prism|5|simpls|cpu|rsvd|argmax"
)

if [ "${FASTPLS_CV_COMPARE_INCLUDE_STRESS:-false}" = "true" ]; then
  CASES+=(
    "$HOME/fastPLS_cycle9_plssvd_vs_simpls_20260723/results/cifar100_task.rds|cifar100|50|simpls|cpu|rsvd|argmax"
    "$HOME/fastPLS_cycle9_plssvd_vs_simpls_20260723/results/nmr_task.rds|nmr|50|simpls|cpu|rsvd|argmax"
  )
fi

for spec in "${CASES[@]}"; do
  IFS='|' read -r task dataset ncomp method backend svd classifier <<<"${spec}"
  run_id="${dataset}_${method}_${backend}_${svd}_${classifier}_n${ncomp}"
  for replicate_id in $(seq 1 "${REPS}"); do
    row="${RESULTS_DIR}/rows/${run_id}_rep${replicate_id}.csv"
    log="${RESULTS_DIR}/logs/${run_id}_rep${replicate_id}.log"
    echo "[$(date -Iseconds)] ${run_id} rep=${replicate_id}" | tee "${log}"
    set +e
    timeout --signal=TERM --kill-after=30s "${TIMEOUT_SEC}" \
      Rscript "${REPO_ROOT}/benchmark/benchmark_cv_compiled_vs_r_loop.R" \
        --task="${task}" \
        --output="${row}" \
        --dataset="${dataset}" \
        --ncomp="${ncomp}" \
        --method="${method}" \
        --backend="${backend}" \
        --svd-method="${svd}" \
        --classifier="${classifier}" \
        --kfold=10 \
        --reps=1 \
        --replicate="${replicate_id}" \
        --seed=123 >>"${log}" 2>&1
    status=$?
    set -e
    if [ "${status}" -ne 0 ]; then
      echo "status=${status}" >>"${log}"
    fi
  done
done

Rscript - "${RESULTS_DIR}" <<'RS'
args <- commandArgs(trailingOnly = TRUE)
results_dir <- args[[1L]]
files <- list.files(file.path(results_dir, "rows"), pattern = "\\.csv$", full.names = TRUE)
raw <- do.call(rbind, lapply(files, read.csv, check.names = FALSE))
write.csv(raw, file.path(results_dir, "cv_compiled_vs_r_loop_raw.csv"), row.names = FALSE)

ok <- raw[raw$status == "success", , drop = FALSE]
keys <- c("dataset", "task_type", "n", "p", "q", "method", "backend",
          "svd_method", "classifier", "ncomp", "kfold", "metric_name")
split_rows <- split(ok, interaction(ok[keys], drop = TRUE, lex.order = TRUE))
summary_rows <- lapply(split_rows, function(x) {
  out <- x[1L, keys, drop = FALSE]
  out$replicates <- nrow(x)
  out$compiled_median_sec <- median(x$compiled_sec, na.rm = TRUE)
  out$compiled_iqr_sec <- IQR(x$compiled_sec, na.rm = TRUE)
  out$r_loop_median_sec <- median(x$r_loop_sec, na.rm = TRUE)
  out$r_loop_iqr_sec <- IQR(x$r_loop_sec, na.rm = TRUE)
  out$speedup_median <- median(x$speedup_r_loop_over_compiled, na.rm = TRUE)
  out$compiled_metric <- median(x$compiled_metric, na.rm = TRUE)
  out$r_loop_metric <- median(x$r_loop_metric, na.rm = TRUE)
  out$metric_abs_diff_max <- max(x$metric_abs_diff, na.rm = TRUE)
  out$prediction_agreement_min <- if (all(is.na(x$prediction_agreement))) NA_real_ else min(x$prediction_agreement, na.rm = TRUE)
  out$prediction_correlation_min <- if (all(is.na(x$prediction_correlation))) NA_real_ else min(x$prediction_correlation, na.rm = TRUE)
  out$max_abs_prediction_diff <- if (all(is.na(x$max_abs_prediction_diff))) NA_real_ else max(x$max_abs_prediction_diff, na.rm = TRUE)
  out$identical_fold_partition <- all(x$identical_fold_partition)
  out
})
summary <- do.call(rbind, summary_rows)
write.csv(summary, file.path(results_dir, "cv_compiled_vs_r_loop_summary.csv"), row.names = FALSE)
print(summary)
RS
