#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
results_dir <- if (length(args)) args[[1L]] else Sys.getenv("FASTPLS_RESULTS_DIR", getwd())
results_dir <- normalizePath(results_dir, winslash = "/", mustWork = TRUE)

if (!requireNamespace("data.table", quietly = TRUE)) stop("data.table must be installed")
if (!requireNamespace("ggplot2", quietly = TRUE)) stop("ggplot2 must be installed")

library(data.table)
library(ggplot2)

raw_path <- file.path(results_dir, "dataset_memory_compare_raw.csv")
if (!file.exists(raw_path)) stop("Missing raw CSV: ", raw_path)

plot_dir <- file.path(results_dir, "plots")
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

raw_dt <- fread(raw_path)
raw_dt <- raw_dt[status %in% c("ok", "capped")]
if (!nrow(raw_dt)) stop("No successful rows available for plotting")

method_levels <- c("plssvd", "simpls", "simpls-fast")
impl_levels <- c("Cpp", "R", "CUDA 64-bit", "CUDA 32-bit", "pls_pkg")
metric_levels <- c("Time", "Accuracy", "CPU memory", "GPU memory")

summarize_metric <- function(dt, metric_col, metric_label) {
  dt[, .(
    metric_value = median(get(metric_col), na.rm = TRUE)
  ), by = .(dataset, method_panel, implementation_label, requested_ncomp)][
    , metric := metric_label]
}

summary_dt <- rbindlist(list(
  summarize_metric(raw_dt, "total_time_ms", "Time"),
  summarize_metric(raw_dt, "accuracy", "Accuracy"),
  summarize_metric(raw_dt, "peak_host_rss_mb", "CPU memory"),
  summarize_metric(raw_dt, "peak_gpu_mem_mb", "GPU memory")
), use.names = TRUE, fill = TRUE)

summary_dt <- summary_dt[is.finite(metric_value)]
summary_dt[, method_panel := factor(method_panel, levels = method_levels)]
summary_dt[, implementation_label := factor(implementation_label, levels = impl_levels)]
summary_dt[, metric := factor(metric, levels = metric_levels)]

fwrite(summary_dt, file.path(results_dir, "dataset_memory_compare_summary.csv"))

panel_title <- function(dataset_id) {
  switch(
    dataset_id,
    cifar100 = "CIFAR100",
    ccle = "CCLE",
    dataset_id
  )
}

for (ds in unique(summary_dt$dataset)) {
  sub <- summary_dt[dataset == ds]
  if (!nrow(sub)) next

  p <- ggplot(sub, aes(x = requested_ncomp, y = metric_value, color = implementation_label, group = implementation_label)) +
    geom_line(linewidth = 0.8, na.rm = TRUE) +
    geom_point(size = 1.8, na.rm = TRUE) +
    facet_grid(metric ~ method_panel, scales = "free_y", drop = FALSE) +
    labs(
      title = sprintf("%s method comparison", panel_title(ds)),
      x = "Number of components",
      y = NULL,
      color = NULL
    ) +
    theme_bw(base_size = 11) +
    theme(
      legend.position = "bottom",
      panel.grid.minor = element_blank(),
      strip.background = element_rect(fill = "grey95")
    )

  ggsave(
    filename = file.path(plot_dir, sprintf("%s_3x4_methods_memory.png", ds)),
    plot = p,
    width = 16,
    height = 13,
    dpi = 180
  )
}

cat("Plots written to:", plot_dir, "\n")
