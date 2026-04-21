#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_file <- if (length(script_arg)) sub("^--file=", "", script_arg[[1L]]) else file.path(getwd(), "write_cifar100_remote_compare_report.R")
script_dir <- dirname(normalizePath(script_file, winslash = "/", mustWork = FALSE))
source(file.path(script_dir, "helpers_cifar100_remote_compare.R"))

args <- parse_kv_args()
results_dir <- normalizePath(arg_value(args, "results_dir", default = Sys.getenv("FASTPLS_RESULTS_DIR", getwd())), winslash = "/", mustWork = TRUE)
raw_csv <- file.path(results_dir, "cifar100_remote_compare_raw.csv")
summary_csv <- file.path(results_dir, "cifar100_remote_compare_summary.csv")
manifest_txt <- file.path(results_dir, "cifar100_remote_compare_manifest.txt")
report_md <- file.path(results_dir, "cifar100_remote_compare_report.md")
task_meta_rds <- normalizePath(arg_value(args, "task_meta_rds", default = file.path(results_dir, "cifar100_task_meta.rds")), winslash = "/", mustWork = TRUE)

if (!file.exists(raw_csv)) stop("Raw CSV not found: ", raw_csv)
raw <- fread(raw_csv)
task_meta <- readRDS(task_meta_rds)
ok_statuses <- c("ok", "capped")

raw[engine != "GPU", peak_gpu_mem_mb := NA_real_]
raw[status %in% ok_statuses, msg := ""]

prediction_agreement <- function(row_idx, dt) {
  row <- dt[row_idx]
  ref_variant <- row$reference_variant_name[[1L]]
  if (!nzchar(ref_variant) || is.na(ref_variant)) return(NA_real_)
  if (!(row$status[[1L]] %in% c("ok", "capped"))) return(NA_real_)
  pred_path <- row$prediction_file[[1L]]
  if (!nzchar(pred_path) || !file.exists(pred_path)) return(NA_real_)
  ref_row <- dt[
    variant_name == ref_variant &
      replicate == row$replicate[[1L]] &
      requested_ncomp == row$requested_ncomp[[1L]] &
      status %in% c("ok", "capped")
  ]
  if (!nrow(ref_row)) return(NA_real_)
  ref_path <- ref_row$prediction_file[[1L]]
  if (!nzchar(ref_path) || !file.exists(ref_path)) return(NA_real_)
  pred_obj <- read_prediction_file(pred_path)
  ref_obj <- read_prediction_file(ref_path)
  if (is.null(pred_obj) || is.null(ref_obj)) return(NA_real_)
  if (length(pred_obj$pred) != length(ref_obj$pred)) return(NA_real_)
  mean(as.character(pred_obj$pred) == as.character(ref_obj$pred), na.rm = TRUE)
}

if (!("prediction_agreement" %in% names(raw))) raw[, prediction_agreement := NA_real_]
for (i in seq_len(nrow(raw))) {
  raw[i, prediction_agreement := prediction_agreement(i, raw)]
}
fwrite(raw, raw_csv)

summary_dt <- raw[
  status %in% ok_statuses,
  .(
    effective_ncomp_median = median(effective_ncomp, na.rm = TRUE),
    effective_ncomp_iqr = IQR(effective_ncomp, na.rm = TRUE),
    fit_time_ms_median = median(fit_time_ms, na.rm = TRUE),
    fit_time_ms_iqr = IQR(fit_time_ms, na.rm = TRUE),
    predict_time_ms_median = median(predict_time_ms, na.rm = TRUE),
    predict_time_ms_iqr = IQR(predict_time_ms, na.rm = TRUE),
    total_time_ms_median = median(total_time_ms, na.rm = TRUE),
    total_time_ms_iqr = IQR(total_time_ms, na.rm = TRUE),
    accuracy_median = median(accuracy, na.rm = TRUE),
    accuracy_iqr = IQR(accuracy, na.rm = TRUE),
    peak_host_rss_mb_median = median(peak_host_rss_mb, na.rm = TRUE),
    peak_host_rss_mb_iqr = IQR(peak_host_rss_mb, na.rm = TRUE),
    peak_gpu_mem_mb_median = median(peak_gpu_mem_mb, na.rm = TRUE),
    peak_gpu_mem_mb_iqr = IQR(peak_gpu_mem_mb, na.rm = TRUE),
    prediction_agreement_median = median(prediction_agreement, na.rm = TRUE),
    prediction_agreement_iqr = IQR(prediction_agreement, na.rm = TRUE),
    reps_ok = .N
  ),
  by = .(
    variant_name, code_tree, method_family, engine, backend,
    precision_mode, label_mode, requested_ncomp, n_train, n_test, p, n_classes
  )
]

status_dt <- raw[, .N, by = .(variant_name, status)]
fwrite(summary_dt, summary_csv)

summary_n50 <- summary_dt[requested_ncomp == 50]
fastest_gpu <- summary_n50[engine == "GPU"][order(total_time_ms_median)][1L]
best_acc <- summary_n50[order(-accuracy_median, total_time_ms_median)][1L]
least_host <- summary_n50[order(peak_host_rss_mb_median)][1L]
least_gpu <- summary_n50[engine == "GPU"][order(peak_gpu_mem_mb_median)][1L]

baseline_gpu_ref <- summary_dt[grepl("^baseline_gpu_", variant_name)]
test_gpu_ref <- summary_dt[grepl("^test_gpu_", variant_name)]
gpu_compare <- merge(
  baseline_gpu_ref[, .(method_family, requested_ncomp, baseline_total_time_ms = total_time_ms_median, baseline_accuracy = accuracy_median)],
  test_gpu_ref[, .(method_family, requested_ncomp, test_variant = variant_name, test_total_time_ms = total_time_ms_median, test_accuracy = accuracy_median)],
  by = c("method_family", "requested_ncomp"),
  allow.cartesian = TRUE
)
if (nrow(gpu_compare)) {
  gpu_compare[, `:=`(
    speedup_test_vs_baseline = baseline_total_time_ms / test_total_time_ms,
    accuracy_diff_test_minus_baseline = test_accuracy - baseline_accuracy
  )]
}

cpu_ref <- summary_dt[grepl("^baseline_cpu_.*_cpu_rsvd$", variant_name)]
cpu_compare <- merge(
  cpu_ref[, .(method_family, requested_ncomp, cpu_ref_total_time_ms = total_time_ms_median, cpu_ref_accuracy = accuracy_median)],
  summary_dt[engine == "GPU", .(method_family, requested_ncomp, variant_name, total_time_ms_median, accuracy_median)],
  by = c("method_family", "requested_ncomp"),
  allow.cartesian = TRUE
)
if (nrow(cpu_compare)) {
  cpu_compare[, `:=`(
    speedup_gpu_vs_cpu_ref = cpu_ref_total_time_ms / total_time_ms_median,
    accuracy_diff_gpu_minus_cpu_ref = accuracy_median - cpu_ref_accuracy
  )]
}

successful_runs <- raw[status %in% ok_statuses, .N]
skipped_counts <- raw[status %in% c("error", "killed", "timed_out", "skipped_no_cuda", "skipped_ncomp_above_cap"), .N, by = .(variant_name, status, msg)]

manifest_lines <- c(
  sprintf("machine_name=%s", safe_system_output("hostname")),
  sprintf("gpu_model=%s", safe_system_output("nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ';' -")),
  sprintf("driver_version=%s", safe_system_output("nvidia-smi --query-gpu=driver_version --format=csv,noheader | paste -sd ';' -")),
  sprintf("cuda_version=%s", safe_system_output("nvidia-smi | awk -F'CUDA Version: ' '/CUDA Version/ {print $2}' | awk '{print $1}'")),
  sprintf("r_version=%s", safe_system_output("R --version | head -n 1")),
  sprintf("dataset_path=%s", task_meta$dataset_path),
  sprintf("n_train=%d", task_meta$n_train),
  sprintf("n_test=%d", task_meta$n_test),
  sprintf("p=%d", task_meta$p),
  sprintf("n_classes=%d", task_meta$n_classes),
  sprintf("split_seed=%d", task_meta$split_seed),
  sprintf("baseline_commit=%s", Sys.getenv("FASTPLS_BASELINE_COMMIT", "")),
  sprintf("test_commit=%s", Sys.getenv("FASTPLS_TEST_COMMIT", "")),
  sprintf("test_commit_note=%s", Sys.getenv("FASTPLS_TEST_COMMIT_NOTE", "")),
  sprintf("baseline_lib=%s", Sys.getenv("FASTPLS_BASELINE_LIB", "")),
  sprintf("test_lib=%s", Sys.getenv("FASTPLS_TEST_LIB", "")),
  sprintf("timing_reps=%s", Sys.getenv("FASTPLS_COMPARE_REPS", "")),
  sprintf("successful_runs=%d", successful_runs),
  sprintf("variants_run=%s", paste(sort(unique(raw$variant_name)), collapse = ",")),
  sprintf(
    "skipped_variants=%s",
    if (nrow(skipped_counts)) paste(sprintf("%s[%s]=%d", skipped_counts$variant_name, skipped_counts$status, skipped_counts$N), collapse = "; ") else "<none>"
  )
)
writeLines(manifest_lines, manifest_txt)

best_practical <- if (nrow(summary_n50[engine == "GPU"])) {
  summary_n50[engine == "GPU"][order(-accuracy_median, total_time_ms_median)][1L, variant_name]
} else {
  summary_n50[order(-accuracy_median, total_time_ms_median)][1L, variant_name]
}

report_lines <- c(
  "# CIFAR100 Remote Compare",
  "",
  sprintf("- Machine: `%s`", safe_system_output("hostname")),
  sprintf("- Dataset path: `%s`", task_meta$dataset_path),
  sprintf("- Dimensions: `n_train=%d`, `n_test=%d`, `p=%d`, `classes=%d`", task_meta$n_train, task_meta$n_test, task_meta$p, task_meta$n_classes),
  sprintf("- Split seed: `%d`", task_meta$split_seed),
  sprintf("- Baseline commit: `%s`", Sys.getenv("FASTPLS_BASELINE_COMMIT", "")),
  sprintf("- Test commit: `%s`", Sys.getenv("FASTPLS_TEST_COMMIT", "")),
  "",
  "## Answers",
  "",
  sprintf("1. Fastest GPU variant at `ncomp=50`: `%s`.", if (nrow(fastest_gpu)) fastest_gpu$variant_name else "n/a"),
  sprintf("2. Best accuracy at `ncomp=50`: `%s`.", if (nrow(best_acc)) best_acc$variant_name else "n/a"),
  sprintf("3. Lowest host memory at `ncomp=50`: `%s`.", if (nrow(least_host)) least_host$variant_name else "n/a"),
  sprintf("4. Lowest GPU memory at `ncomp=50`: `%s`.", if (nrow(least_gpu)) least_gpu$variant_name else "n/a"),
  sprintf(
    "5. Experimental GPU vs baseline GPU: `%s`.",
    if (nrow(gpu_compare)) {
      paste(
        gpu_compare[requested_ncomp == 50, sprintf(
          "%s speedup %.2fx, accuracy diff %.4f",
          test_variant, speedup_test_vs_baseline, accuracy_diff_test_minus_baseline
        )],
        collapse = "; "
      )
    } else "no paired GPU comparison available"
  ),
  sprintf(
    "6. GPU vs CPU reference at `ncomp=50`: `%s`.",
    if (nrow(cpu_compare)) {
      paste(
        cpu_compare[requested_ncomp == 50, sprintf(
          "%s speedup %.2fx vs CPU reference, accuracy diff %.4f",
          variant_name, speedup_gpu_vs_cpu_ref, accuracy_diff_gpu_minus_cpu_ref
        )],
        collapse = "; "
      )
    } else "no paired GPU/CPU comparison available"
  ),
  sprintf(
    "7. Accuracy divergence across `ncomp`: `%s`.",
    if (nrow(summary_dt)) {
      top <- summary_dt[, .SD[which.max(accuracy_median)], by = variant_name]
      paste(sprintf("%s best accuracy %.4f at ncomp=%d", top$variant_name, top$accuracy_median, top$requested_ncomp), collapse = "; ")
    } else "no successful runs"
  ),
  sprintf(
    "8. Memory failures or OOM events: `%s`.",
    if (nrow(skipped_counts)) paste(sprintf("%s[%s]: %d", skipped_counts$variant_name, skipped_counts$status, skipped_counts$N), collapse = "; ") else "none recorded"
  ),
  sprintf("9. Best practical recommendation on this machine: `%s`.", best_practical),
  "",
  "## Main Summary (ncomp = 50)",
  ""
)

if (nrow(summary_n50)) {
  pretty_n50 <- summary_n50[, .(
    variant_name,
    total_time_ms_median = round(total_time_ms_median, 2),
    accuracy_median = round(accuracy_median, 4),
    peak_host_rss_mb_median = round(peak_host_rss_mb_median, 2),
    peak_gpu_mem_mb_median = round(peak_gpu_mem_mb_median, 2),
    effective_ncomp_median = round(effective_ncomp_median, 2),
    reps_ok
  )]
  report_lines <- c(report_lines, capture.output(print(pretty_n50)))
} else {
  report_lines <- c(report_lines, "No successful ncomp=50 rows.")
}

writeLines(report_lines, report_md)

cat("Summary written to:", summary_csv, "\n")
cat("Manifest written to:", manifest_txt, "\n")
cat("Report written to:", report_md, "\n")
