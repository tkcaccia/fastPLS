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
if (!"predict_ms" %in% names(ok)) ok[, predict_ms := NA_real_]
if (!"total_ms" %in% names(ok)) ok[, total_ms := train_ms + fifelse(is.na(predict_ms), 0, predict_ms)]
for (mem_col in c("peak_host_rss_mb", "host_rss_delta_mb", "peak_gpu_mem_mb", "gpu_mem_delta_mb")) {
  if (!mem_col %in% names(ok)) ok[, (mem_col) := NA_real_]
}

ok_sum <- ok[, .(
  train_ms_median = median(train_ms, na.rm = TRUE),
  train_ms_mean = mean(train_ms, na.rm = TRUE),
  predict_ms_median = median(predict_ms, na.rm = TRUE),
  predict_ms_mean = mean(predict_ms, na.rm = TRUE),
  total_ms_median = median(total_ms, na.rm = TRUE),
  total_ms_mean = mean(total_ms, na.rm = TRUE),
  metric_median = median(metric_value, na.rm = TRUE),
  metric_mean = mean(metric_value, na.rm = TRUE),
  model_size_mb_median = median(model_size_mb, na.rm = TRUE),
  peak_host_rss_mb_median = median(peak_host_rss_mb, na.rm = TRUE),
  host_rss_delta_mb_median = median(host_rss_delta_mb, na.rm = TRUE),
  peak_gpu_mem_mb_median = median(peak_gpu_mem_mb, na.rm = TRUE),
  gpu_mem_delta_mb_median = median(gpu_mem_delta_mb, na.rm = TRUE),
  n_ok = .N
), by = .(dataset, analysis, analysis_value, ncomp, metric_name, method_id, engine, algorithm, svd_method, fast_profile, param_set)]

ok_sum[, analysis_value_num := suppressWarnings(as.numeric(analysis_value))]
ok_sum[, algorithm_panel := fifelse(
  algorithm == "simpls_fast" & fast_profile %in% c("incdefl", "gpu_native"),
  "simpls",
  algorithm
)]
ok_sum[, algorithm_panel := factor(
  algorithm_panel,
  levels = c("plssvd", "simpls", "opls", "kernelpls")
)]
ok_sum[, svd_family := fifelse(
  grepl("rsvd", svd_method, ignore.case = TRUE),
  "rsvd",
  fifelse(grepl("gpu", svd_method, ignore.case = TRUE),
          "gpu",
          fifelse(grepl("irlba", svd_method, ignore.case = TRUE),
                  "irlba",
                  fifelse(grepl("exact", svd_method, ignore.case = TRUE),
                          "exact",
                          tolower(svd_method))))
)]
ok_sum[, engine_shape := fifelse(
  engine == "pls_pkg",
  "pls_pkg",
  fifelse(engine == "R", "R", fifelse(engine == "Rcpp", "Rcpp", fifelse(engine == "GPU", "GPU", engine)))
)]

svd_levels <- c("irlba", "rsvd", "gpu", "exact", sort(setdiff(unique(ok_sum$svd_family), c("irlba", "rsvd", "gpu", "exact"))))
svd_levels <- svd_levels[svd_levels %in% unique(ok_sum$svd_family)]
svd_palette <- c(
  irlba = "#1b9e77",
  rsvd = "#d95f02",
  gpu = "#e7298a",
  exact = "#7570b3"
)
if (length(setdiff(svd_levels, names(svd_palette)))) {
  extra_levels <- setdiff(svd_levels, names(svd_palette))
  extra_cols <- grDevices::hcl.colors(length(extra_levels), palette = "Dark 3")
  names(extra_cols) <- extra_levels
  svd_palette <- c(svd_palette, extra_cols)
}
svd_palette <- svd_palette[svd_levels]

shape_levels <- c("pls_pkg", "R", "Rcpp", "GPU", sort(setdiff(unique(ok_sum$engine_shape), c("pls_pkg", "R", "Rcpp", "GPU"))))
shape_levels <- shape_levels[shape_levels %in% unique(ok_sum$engine_shape)]
shape_values <- c(
  pls_pkg = 15,
  R = 17,
  Rcpp = 16,
  GPU = 18
)
if (length(setdiff(shape_levels, names(shape_values)))) {
  extra_shapes <- c(0, 1, 2, 5, 6, 7, 8)
  names(extra_shapes) <- setdiff(shape_levels, names(shape_values))
  shape_values <- c(shape_values, extra_shapes)
}
shape_values <- shape_values[shape_levels]

analysis_levels <- unique(ok_sum$analysis)

plot_time <- function(dt, analysis_name, use_ncomp_x = FALSE) {
  xvar <- if (use_ncomp_x) "ncomp" else if (all(is.finite(dt$analysis_value_num))) "analysis_value_num" else "analysis_value"
  p <- ggplot(dt, aes_string(x = xvar, y = "total_ms_median", color = "svd_family", shape = "engine_shape", group = "method_id")) +
    geom_line(linewidth = 0.8, alpha = 0.9) +
    geom_point(size = 1.8) +
    scale_color_manual(values = svd_palette, breaks = svd_levels, drop = FALSE) +
    scale_shape_manual(values = shape_values, breaks = shape_levels, drop = FALSE) +
    facet_wrap(~dataset, scales = "free_y") +
    theme_minimal(base_size = 11) +
    labs(
      title = paste0("Total Fit + Prediction Time - ", analysis_name),
      x = if (use_ncomp_x) "ncomp" else "analysis value",
      y = "Median total time (fit + prediction, ms)",
      color = "svd",
      shape = "implementation"
    )
  ggsave(file.path(plot_dir, paste0("time_", analysis_name, "_", plot_scope, ".png")), p, width = 13, height = 8, dpi = 150)
}

plot_metric <- function(dt, analysis_name, use_ncomp_x = FALSE) {
  xvar <- if (use_ncomp_x) "ncomp" else if (all(is.finite(dt$analysis_value_num))) "analysis_value_num" else "analysis_value"
  p <- ggplot(dt, aes_string(x = xvar, y = "metric_median", color = "svd_family", shape = "engine_shape", group = "method_id")) +
    geom_line(linewidth = 0.8, alpha = 0.9) +
    geom_point(size = 1.8) +
    scale_color_manual(values = svd_palette, breaks = svd_levels, drop = FALSE) +
    scale_shape_manual(values = shape_values, breaks = shape_levels, drop = FALSE) +
    facet_grid(metric_name ~ dataset, scales = "free_y") +
    theme_minimal(base_size = 11) +
    labs(
      title = paste0("Performance Metric - ", analysis_name),
      x = if (use_ncomp_x) "ncomp" else "analysis value",
      y = "Median metric (accuracy or rmsd)",
      color = "svd",
      shape = "implementation"
    )
  ggsave(file.path(plot_dir, paste0("metric_", analysis_name, "_", plot_scope, ".png")), p, width = 14, height = 9, dpi = 150)
}

plot_subanalysis <- function(dt, analysis_name, use_ncomp_x = FALSE) {
  xvar <- if (use_ncomp_x) "ncomp" else if (all(is.finite(dt$analysis_value_num))) "analysis_value_num" else "analysis_value"

  p1 <- ggplot(dt, aes_string(x = xvar, y = "total_ms_median", color = "svd_family", shape = "engine_shape", group = "method_id")) +
    geom_line(linewidth = 0.7) +
    geom_point(size = 1.4) +
    scale_color_manual(values = svd_palette, breaks = svd_levels, drop = FALSE) +
    scale_shape_manual(values = shape_values, breaks = shape_levels, drop = FALSE) +
    facet_grid(dataset ~ algorithm_panel, scales = "free_y") +
    theme_minimal(base_size = 10) +
    labs(
      title = paste0("Subanalysis Total Fit + Prediction Time - ", analysis_name),
      x = if (use_ncomp_x) "ncomp" else "analysis value",
      y = "Median total time (fit + prediction, ms)",
      color = "svd",
      shape = "implementation"
    )
  ggsave(file.path(plot_dir, paste0("sub_time_", analysis_name, "_dataset_algorithm_", plot_scope, ".png")), p1, width = 15, height = 10, dpi = 150)

  p2 <- ggplot(dt, aes_string(x = xvar, y = "metric_median", color = "svd_family", shape = "engine_shape", group = "method_id")) +
    geom_line(linewidth = 0.7) +
    geom_point(size = 1.4) +
    scale_color_manual(values = svd_palette, breaks = svd_levels, drop = FALSE) +
    scale_shape_manual(values = shape_values, breaks = shape_levels, drop = FALSE) +
    facet_grid(metric_name + dataset ~ algorithm_panel, scales = "free_y") +
    theme_minimal(base_size = 10) +
    labs(
      title = paste0("Subanalysis Metric - ", analysis_name),
      x = if (use_ncomp_x) "ncomp" else "analysis value",
      y = "Median metric (accuracy/rmsd)",
      color = "svd",
      shape = "implementation"
    )
  ggsave(file.path(plot_dir, paste0("sub_metric_", analysis_name, "_dataset_algorithm_", plot_scope, ".png")), p2, width = 15, height = 11, dpi = 150)
}

plot_three_by_four <- function(dt, analysis_name, use_ncomp_x = FALSE) {
  if (!identical(analysis_name, "ncomp")) return(invisible(NULL))
  xvar <- "ncomp"
  metric_label <- unique(dt$metric_name[!is.na(dt$metric_name)])[1]
  metric_label <- switch(
    metric_label,
    accuracy = "Accuracy",
    q2 = "Q2",
    rmsd = "RMSD",
    toupper(metric_label)
  )

  host_col <- if (any(is.finite(dt$host_rss_delta_mb_median))) "host_rss_delta_mb_median" else "peak_host_rss_mb_median"
  gpu_col <- if (any(is.finite(dt$gpu_mem_delta_mb_median))) "gpu_mem_delta_mb_median" else "peak_gpu_mem_mb_median"
  long <- rbindlist(
    list(
      dt[, .(dataset, ncomp, algorithm_panel, svd_family, engine_shape, method_id, panel_row = "Total time (fit + prediction, ms)", value = total_ms_median)],
      dt[, .(dataset, ncomp, algorithm_panel, svd_family, engine_shape, method_id, panel_row = metric_label, value = metric_median)],
      dt[, .(dataset, ncomp, algorithm_panel, svd_family, engine_shape, method_id, panel_row = "RAM allocation (peak RSS delta, MB)", value = get(host_col))],
      dt[, .(dataset, ncomp, algorithm_panel, svd_family, engine_shape, method_id, panel_row = "GPU memory allocation (peak delta, MB)", value = get(gpu_col))]
    ),
    fill = TRUE
  )
  long[, panel_row := factor(
    panel_row,
    levels = c(
      "Total time (fit + prediction, ms)",
      metric_label,
      "RAM allocation (peak RSS delta, MB)",
      "GPU memory allocation (peak delta, MB)"
    )
  )]
  finite_long <- long[is.finite(value)]
  if (!nrow(finite_long)) return(invisible(NULL))

  missing_panels <- long[, .(has_data = any(is.finite(value))), by = .(panel_row, algorithm_panel)][has_data == FALSE]
  if (nrow(missing_panels)) {
    missing_panels[, `:=`(
      ncomp = median(long$ncomp, na.rm = TRUE),
      value = 0,
      label = "not recorded"
    )]
  }

  p <- ggplot(long, aes_string(x = xvar, y = "value", color = "svd_family", shape = "engine_shape", group = "method_id")) +
    geom_line(linewidth = 0.7, alpha = 0.9, na.rm = TRUE) +
    geom_point(size = 1.5, na.rm = TRUE)
  if (nrow(missing_panels)) {
    p <- p + geom_text(
      data = missing_panels,
      aes(x = ncomp, y = value, label = label),
      inherit.aes = FALSE,
      size = 3.5,
      color = "grey35"
    )
  }
  p <- p +
    scale_color_manual(values = svd_palette, breaks = svd_levels, drop = FALSE) +
    scale_shape_manual(values = shape_values, breaks = shape_levels, drop = FALSE) +
    scale_x_continuous(breaks = sort(unique(long$ncomp))) +
    facet_grid(panel_row ~ algorithm_panel, scales = "free_y", drop = FALSE) +
    theme_minimal(base_size = 11) +
    theme(
      plot.background = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA),
      legend.background = element_rect(fill = "white", colour = NA),
      legend.key = element_rect(fill = "white", colour = NA)
    ) +
    labs(
      title = paste0("3x4 ncomp benchmark - ", plot_scope),
      x = "ncomp",
      y = "Median value (row-specific shared scale)",
      color = "svd",
      shape = "implementation"
    )
  ggsave(file.path(plot_dir, paste0("three_by_four_", analysis_name, "_", plot_scope, ".png")), p, width = 16, height = 14, dpi = 150, bg = "white")
  invisible(p)
}

for (an in analysis_levels) {
  dt <- ok_sum[analysis == an]
  if (!nrow(dt)) next
  use_ncomp <- identical(an, "ncomp")

  plot_time(dt, an, use_ncomp_x = use_ncomp)
  plot_metric(dt, an, use_ncomp_x = use_ncomp)
  plot_subanalysis(dt, an, use_ncomp_x = use_ncomp)
  plot_three_by_four(dt, an, use_ncomp_x = use_ncomp)
}

fwrite(ok_sum, file.path(out_dir, paste0("multianalysis_plot_summary_", plot_scope, ".csv")))

cat("Plots generated in:", plot_dir, "\n")
