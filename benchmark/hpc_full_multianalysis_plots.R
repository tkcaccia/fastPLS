#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_file <- if (length(script_arg)) sub("^--file=", "", script_arg[[1]]) else file.path(getwd(), "benchmark", "hpc_full_multianalysis_plots.R")
script_dir <- dirname(normalizePath(script_file, winslash = "/", mustWork = FALSE))
repo_root <- normalizePath(file.path(script_dir, ".."), winslash = "/", mustWork = FALSE)
default_data_root <- if (dir.exists("/Users/stefano/HPC-firenze/image_analysis/dinoV2/Rdatasets")) {
  "/Users/stefano/HPC-firenze/image_analysis/dinoV2/Rdatasets"
} else if (dir.exists(file.path(repo_root, "Rdataset"))) {
  file.path(repo_root, "Rdataset")
} else if (dir.exists(file.path(repo_root, "Data"))) {
  file.path(repo_root, "Data")
} else {
  "/scratch/firenze/image_analysis/dinoV2/Rdatasets"
}
default_out_root <- if (dir.exists(repo_root)) {
  file.path(repo_root, "benchmark_results_local")
} else {
  file.path(default_data_root, "benchmark_results")
}

base_dir <- path.expand(Sys.getenv("FASTPLS_DATA_ROOT", default_data_root))
out_dir <- path.expand(Sys.getenv("FASTPLS_MULTI_OUTDIR", file.path(default_out_root, "multianalysis")))
raw_csv <- file.path(out_dir, "multianalysis_raw.csv")
sum_csv <- file.path(out_dir, "multianalysis_summary.csv")
plot_dataset <- tolower(trimws(Sys.getenv("FASTPLS_PLOT_DATASET", "")))
plot_scope <- if (nzchar(plot_dataset)) plot_dataset else "all"
plot_dir <- file.path(out_dir, "plots", plot_scope)
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(raw_csv)) stop("Raw CSV not found: ", raw_csv)
if (!file.exists(sum_csv)) stop("Summary CSV not found: ", sum_csv)

raw <- fread(raw_csv)
sumtab <- fread(sum_csv)

if (nzchar(plot_dataset)) {
  raw <- raw[tolower(dataset) == plot_dataset]
  sumtab <- sumtab[tolower(dataset) == plot_dataset]
}

if (!nrow(raw)) {
  cat("No rows found for plotting scope:", plot_scope, "- skipping plots.\n")
  quit(save = "no", status = 0)
}

ok <- raw[status == "ok"]
if (!nrow(ok)) {
  status_counts <- raw[, .N, by = status][order(-N)]
  cat("No status==ok rows for plotting scope:", plot_scope, "- skipping plots.\n")
  cat("Status counts:\n")
  print(status_counts)
  quit(save = "no", status = 0)
}

ok_sum <- ok[, .(
  train_ms_median = median(train_ms, na.rm = TRUE),
  train_ms_mean = mean(train_ms, na.rm = TRUE),
  metric_median = median(metric_value, na.rm = TRUE),
  metric_mean = mean(metric_value, na.rm = TRUE),
  model_size_mb_median = median(model_size_mb, na.rm = TRUE),
  n_ok = .N
), by = .(dataset, analysis, analysis_value, ncomp, metric_name, method_id, engine, algorithm, svd_method, param_set)]

ok_sum[, analysis_value_num := suppressWarnings(as.numeric(analysis_value))]

analysis_levels <- unique(ok_sum$analysis)

plot_time <- function(dt, analysis_name, use_ncomp_x = FALSE) {
  xvar <- if (use_ncomp_x) "ncomp" else if (all(is.finite(dt$analysis_value_num))) "analysis_value_num" else "analysis_value"
  p <- ggplot(dt, aes_string(x = xvar, y = "train_ms_median", color = "method_id", group = "method_id")) +
    geom_line(linewidth = 0.8, alpha = 0.9) +
    geom_point(size = 1.8) +
    facet_wrap(~dataset, scales = "free") +
    theme_minimal(base_size = 11) +
    labs(
      title = paste0("Train Time - ", analysis_name),
      x = if (use_ncomp_x) "ncomp" else "analysis value",
      y = "Median train time (ms)",
      color = "method"
    )
  ggsave(file.path(plot_dir, paste0("time_", analysis_name, "_", plot_scope, ".png")), p, width = 13, height = 8, dpi = 150)
}

plot_metric <- function(dt, analysis_name, use_ncomp_x = FALSE) {
  xvar <- if (use_ncomp_x) "ncomp" else if (all(is.finite(dt$analysis_value_num))) "analysis_value_num" else "analysis_value"
  p <- ggplot(dt, aes_string(x = xvar, y = "metric_median", color = "method_id", group = "method_id")) +
    geom_line(linewidth = 0.8, alpha = 0.9) +
    geom_point(size = 1.8) +
    facet_grid(metric_name ~ dataset, scales = "free") +
    theme_minimal(base_size = 11) +
    labs(
      title = paste0("Performance Metric - ", analysis_name),
      x = if (use_ncomp_x) "ncomp" else "analysis value",
      y = "Median metric (accuracy or rmsd)",
      color = "method"
    )
  ggsave(file.path(plot_dir, paste0("metric_", analysis_name, "_", plot_scope, ".png")), p, width = 14, height = 9, dpi = 150)
}

plot_subanalysis <- function(dt, analysis_name, use_ncomp_x = FALSE) {
  xvar <- if (use_ncomp_x) "ncomp" else if (all(is.finite(dt$analysis_value_num))) "analysis_value_num" else "analysis_value"

  p1 <- ggplot(dt, aes_string(x = xvar, y = "train_ms_median", color = "svd_method", group = "svd_method")) +
    geom_line(linewidth = 0.7) +
    geom_point(size = 1.4) +
    facet_grid(dataset ~ algorithm, scales = "free") +
    theme_minimal(base_size = 10) +
    labs(
      title = paste0("Subanalysis Train Time - ", analysis_name),
      x = if (use_ncomp_x) "ncomp" else "analysis value",
      y = "Median train time (ms)",
      color = "svd"
    )
  ggsave(file.path(plot_dir, paste0("sub_time_", analysis_name, "_dataset_algorithm_", plot_scope, ".png")), p1, width = 15, height = 10, dpi = 150)

  p2 <- ggplot(dt, aes_string(x = xvar, y = "metric_median", color = "svd_method", group = "svd_method")) +
    geom_line(linewidth = 0.7) +
    geom_point(size = 1.4) +
    facet_grid(metric_name + dataset ~ algorithm, scales = "free") +
    theme_minimal(base_size = 10) +
    labs(
      title = paste0("Subanalysis Metric - ", analysis_name),
      x = if (use_ncomp_x) "ncomp" else "analysis value",
      y = "Median metric (accuracy/rmsd)",
      color = "svd"
    )
  ggsave(file.path(plot_dir, paste0("sub_metric_", analysis_name, "_dataset_algorithm_", plot_scope, ".png")), p2, width = 15, height = 11, dpi = 150)
}

for (an in analysis_levels) {
  dt <- ok_sum[analysis == an]
  if (!nrow(dt)) next
  use_ncomp <- identical(an, "ncomp")

  plot_time(dt, an, use_ncomp_x = use_ncomp)
  plot_metric(dt, an, use_ncomp_x = use_ncomp)
  plot_subanalysis(dt, an, use_ncomp_x = use_ncomp)
}

# Save compact table used for plotting
fwrite(ok_sum, file.path(out_dir, paste0("multianalysis_plot_summary_", plot_scope, ".csv")))

cat("Plots generated in:", plot_dir, "\n")
