#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_file <- if (length(script_arg)) sub("^--file=", "", script_arg[[1]]) else file.path(getwd(), "benchmark", "plot_simulated_fastpls_results.R")
script_dir <- dirname(normalizePath(script_file, winslash = "/", mustWork = FALSE))
repo_root <- normalizePath(file.path(script_dir, ".."), winslash = "/", mustWork = FALSE)

source(file.path(script_dir, "helpers_simulated_fastpls.R"), local = TRUE)

out_dir <- path.expand(Sys.getenv("FASTPLS_SIM_OUTDIR", file.path(repo_root, "benchmark_results_simulated_fastpls")))
summary_path <- file.path(out_dir, "simulated_fastpls_summary.csv")
if (!file.exists(summary_path)) stop("Missing simulated summary CSV: ", summary_path)

plot_dir <- file.path(out_dir, "plots")
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

dt <- fread(summary_path)
if (!nrow(dt)) stop("Empty simulated summary CSV: ", summary_path)

dt[, predictive_metric := fifelse(task_type == "classification", accuracy_median, Q2_median)]
dt[, metric_label := fifelse(task_type == "classification", "Accuracy", "Q2")]

color_vals <- c(
  "Rcpp / plssvd / arpack" = "#7570b3",
  "Rcpp / plssvd / cpu_rsvd" = "#d95f02",
  "Rcpp / plssvd / irlba" = "#1b9e77",
  "Rcpp / simpls / arpack" = "#7570b3",
  "Rcpp / simpls / cpu_rsvd" = "#d95f02",
  "Rcpp / simpls / irlba" = "#1b9e77",
  "Rcpp / simpls_fast / arpack" = "#7570b3",
  "Rcpp / simpls_fast / cpu_rsvd" = "#d95f02",
  "Rcpp / simpls_fast / irlba" = "#1b9e77",
  "GPU / plssvd" = "#e7298a",
  "GPU / simpls_fast" = "#66a61e",
  "pls_pkg / simpls" = "#666666",
  "R / plssvd / arpack" = "#8da0cb",
  "R / plssvd / cpu_rsvd" = "#fc8d62",
  "R / plssvd / irlba" = "#66c2a5",
  "R / simpls / arpack" = "#8da0cb",
  "R / simpls / cpu_rsvd" = "#fc8d62",
  "R / simpls / irlba" = "#66c2a5",
  "R / simpls_fast / arpack" = "#8da0cb",
  "R / simpls_fast / cpu_rsvd" = "#fc8d62",
  "R / simpls_fast / irlba" = "#66c2a5"
)
shape_vals <- c(
  Rcpp = 16,
  R = 17,
  GPU = 18,
  pls_pkg = 15
)

plot_perf_ncomp <- function(task_type) {
  task_name <- task_type
  sub <- dt[analysis_type == "ncomp" & task_type == task_name & is.finite(predictive_metric)]
  if (!nrow(sub)) return(invisible(NULL))
  p <- ggplot(sub, aes(x = requested_ncomp, y = predictive_metric, color = method_label, shape = engine, group = method_id)) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 2) +
    facet_wrap(~sim_family, scales = "free_y") +
    scale_color_manual(values = color_vals, guide = guide_legend(ncol = 2)) +
    scale_shape_manual(values = shape_vals) +
    labs(
      title = sprintf("Predictive performance vs requested ncomp (%s)", task_type),
      x = "Requested ncomp",
      y = unique(sub$metric_label),
      color = "Method/backend",
      shape = "Engine"
    ) +
    theme_bw(base_size = 11) +
    theme(legend.position = "bottom")
  ggsave(file.path(plot_dir, sprintf("performance_vs_ncomp_%s.png", task_type)), p, width = 14, height = 9, dpi = 160)
}

plot_runtime_ncomp <- function(task_type) {
  task_name <- task_type
  sub <- dt[analysis_type == "ncomp" & task_type == task_name & is.finite(elapsed_ms_median)]
  if (!nrow(sub)) return(invisible(NULL))
  p <- ggplot(sub, aes(x = requested_ncomp, y = elapsed_ms_median, color = method_label, shape = engine, group = method_id)) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 2) +
    scale_y_log10() +
    facet_wrap(~sim_family, scales = "free_y") +
    scale_color_manual(values = color_vals, guide = guide_legend(ncol = 2)) +
    scale_shape_manual(values = shape_vals) +
    labs(
      title = sprintf("Runtime vs requested ncomp (%s)", task_type),
      x = "Requested ncomp",
      y = "Median elapsed time (ms, log scale)",
      color = "Method/backend",
      shape = "Engine"
    ) +
    theme_bw(base_size = 11) +
    theme(legend.position = "bottom")
  ggsave(file.path(plot_dir, sprintf("runtime_vs_ncomp_%s.png", task_type)), p, width = 14, height = 9, dpi = 160)
}

plot_speedup <- function() {
  sub <- dt[analysis_type == "ncomp" & engine %in% c("Rcpp", "R")]
  if (!nrow(sub)) return(invisible(NULL))
  base <- sub[method == "simpls", .(
    sim_family, task_type, requested_ncomp, engine, svd_method,
    elapsed_simpls = elapsed_ms_median
  )]
  comp <- sub[method %in% c("simpls_fast", "plssvd"), .(
    sim_family, task_type, requested_ncomp, engine, svd_method, method,
    elapsed_other = elapsed_ms_median
  )]
  sp <- merge(comp, base, by = c("sim_family", "task_type", "requested_ncomp", "engine", "svd_method"), all = FALSE)
  if (!nrow(sp)) return(invisible(NULL))
  sp[, speedup_vs_simpls := elapsed_simpls / elapsed_other]
  sp[, backend_label := paste(engine, svd_method, sep = " / ")]
  p <- ggplot(sp, aes(x = requested_ncomp, y = speedup_vs_simpls, color = backend_label, group = backend_label)) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "grey50") +
    geom_line(linewidth = 0.8) +
    geom_point(size = 2) +
    facet_grid(task_type + method ~ sim_family, scales = "free_y") +
    labs(
      title = "Speedup vs simpls baseline",
      x = "Requested ncomp",
      y = "simpls elapsed / method elapsed",
      color = "Backend"
    ) +
    theme_bw(base_size = 10) +
    theme(legend.position = "bottom")
  ggsave(file.path(plot_dir, "speedup_vs_simpls.png"), p, width = 15, height = 10, dpi = 160)
}

plot_scaling <- function(analysis_name, x_col, filename) {
  sub <- dt[analysis_type == analysis_name]
  if (!nrow(sub)) return(invisible(NULL))
  p <- ggplot(sub, aes(x = .data[[x_col]], y = elapsed_ms_median, color = method_label, shape = engine, group = method_id)) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 2) +
    facet_wrap(~sim_family, scales = "free_y") +
    scale_y_log10() +
    scale_color_manual(values = color_vals, guide = guide_legend(ncol = 2)) +
    scale_shape_manual(values = shape_vals) +
    labs(
      title = sprintf("Elapsed time vs %s", x_col),
      x = x_col,
      y = "Median elapsed time (ms, log scale)",
      color = "Method/backend",
      shape = "Engine"
    ) +
    theme_bw(base_size = 11) +
    theme(legend.position = "bottom")
  ggsave(file.path(plot_dir, filename), p, width = 14, height = 9, dpi = 160)
}

plot_spectrum_noise <- function(value_col, title_text, file_name, fill_label) {
  sub <- dt[analysis_type == "spectrum_and_noise" & is.finite(get(value_col))]
  if (!nrow(sub)) return(invisible(NULL))
  p <- ggplot(sub, aes(x = noise_regime, y = spectrum_regime, fill = .data[[value_col]])) +
    geom_tile(color = "white") +
    facet_grid(method_label ~ sim_family, scales = "free") +
    scale_fill_viridis_c(option = "C") +
    labs(
      title = title_text,
      x = "Noise regime",
      y = "Spectrum regime",
      fill = fill_label
    ) +
    theme_bw(base_size = 10) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      strip.text.y = element_text(size = 8)
    )
  ggsave(file.path(plot_dir, file_name), p, width = 16, height = 12, dpi = 160)
}

plot_perf_ncomp("regression")
plot_perf_ncomp("classification")
plot_runtime_ncomp("regression")
plot_runtime_ncomp("classification")
plot_speedup()
plot_scaling("sample_fraction", "sample_fraction", "elapsed_vs_sample_fraction.png")
plot_scaling("xvar_fraction", "xvar_fraction", "elapsed_vs_xvar_fraction.png")
plot_scaling("yvar_fraction", "yvar_fraction", "elapsed_vs_yvar_fraction.png")
plot_spectrum_noise("elapsed_ms_median", "Spectrum/noise runtime", "spectrum_noise_runtime.png", "Elapsed ms")
plot_spectrum_noise("predictive_median", "Spectrum/noise predictive score", "spectrum_noise_metric.png", "Predictive score")

cat("Plots written to:", plot_dir, "\n")
