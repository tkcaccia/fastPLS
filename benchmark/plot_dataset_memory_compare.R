#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
results_dir <- if (length(args)) args[[1L]] else Sys.getenv("FASTPLS_RESULTS_DIR", getwd())
results_dir <- normalizePath(results_dir, winslash = "/", mustWork = TRUE)

if (!requireNamespace("data.table", quietly = TRUE)) stop("data.table must be installed")
if (!requireNamespace("ggplot2", quietly = TRUE)) stop("ggplot2 must be installed")
if (!requireNamespace("patchwork", quietly = TRUE)) stop("patchwork must be installed")

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(patchwork)
})

raw_path <- file.path(results_dir, "dataset_memory_compare_raw.csv")
if (!file.exists(raw_path)) stop("Missing raw CSV: ", raw_path)

plot_dir <- file.path(results_dir, "plots")
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

raw_dt <- fread(raw_path)
raw_dt <- raw_dt[status %in% c("ok", "capped")]
if (!nrow(raw_dt)) stop("No successful rows available for plotting")

first_non_missing <- function(x, default = NA_character_) {
  x <- x[!is.na(x)]
  if (!length(x)) return(default)
  x[[1L]]
}

raw_dt[, dataset := trimws(as.character(dataset))]
raw_dt[, method_panel := trimws(as.character(method_panel))]
raw_dt[, implementation_label := trimws(as.character(implementation_label))]
raw_dt[, engine := trimws(as.character(engine))]
raw_dt[, backend := trimws(as.character(backend))]
raw_dt[, variant_name := trimws(as.character(variant_name))]

raw_dt[, backend_family := fifelse(
  backend == "pls_pkg",
  "pls_pkg",
  fifelse(
    engine == "GPU" & grepl("fp32", variant_name, ignore.case = TRUE),
    "gpu_fp32",
    fifelse(
      engine == "GPU" & grepl("fp64", variant_name, ignore.case = TRUE),
      "gpu_fp64",
      backend
    )
  )
)]

raw_dt[, line_id := paste(implementation_label, backend_family, sep = " / ")]

summarize_metric <- function(dt, metric_col, metric_label) {
  dt[is.finite(get(metric_col)), .(
    metric_value = as.numeric(median(get(metric_col), na.rm = TRUE)),
    task_type = first_non_missing(task_type, NA_character_),
    metric_name = first_non_missing(metric_name, NA_character_),
    n_train = as.integer(first_non_missing(n_train, NA_integer_)),
    n_test = as.integer(first_non_missing(n_test, NA_integer_)),
    p = as.integer(first_non_missing(p, NA_integer_)),
    n_classes = as.integer(first_non_missing(n_classes, NA_integer_))
  ), by = .(
    dataset,
    method_panel,
    implementation_label,
    backend_family,
    line_id,
    requested_ncomp
  )][, metric := metric_label]
}

summary_dt <- rbindlist(list(
  summarize_metric(raw_dt, "total_time_ms", "Time"),
  summarize_metric(raw_dt, "metric_value", "Performance"),
  summarize_metric(raw_dt, "peak_host_rss_mb", "CPU memory"),
  summarize_metric(raw_dt, "peak_gpu_mem_mb", "GPU memory")
), use.names = TRUE, fill = TRUE)

summary_dt <- summary_dt[is.finite(metric_value)]
fwrite(summary_dt, file.path(results_dir, "dataset_memory_compare_summary.csv"))

method_levels <- c("plssvd", "simpls", "simpls-fast")
backend_palette <- c(
  cpu_rsvd = "#d95f02",
  irlba = "#1b9e77",
  arpack = "#7570b3",
  gpu_fp64 = "#e7298a",
  gpu_fp32 = "#66a61e",
  pls_pkg = "#666666"
)

shape_values <- c(
  Cpp = 16,
  R = 17,
  `CUDA 64-bit` = 18,
  `CUDA 32-bit` = 15,
  pls_pkg = 8
)

linetype_values <- c(
  Cpp = "solid",
  R = "22",
  `CUDA 64-bit` = "longdash",
  `CUDA 32-bit` = "dotdash",
  pls_pkg = "solid"
)

backend_labels <- c(
  cpu_rsvd = "rsvd",
  irlba = "irlba",
  arpack = "arpack",
  gpu_fp64 = "gpu fp64",
  gpu_fp32 = "gpu fp32",
  pls_pkg = "pls_pkg"
)

metric_axis_label <- function(metric_name) {
  if (is.null(metric_name) || is.na(metric_name) || !nzchar(metric_name)) return("Metric")
  switch(
    tolower(metric_name),
    accuracy = "Accuracy",
    q2 = "Q2",
    rmsd = "RMSD",
    metric_name
  )
}

panel_title <- function(method_name) {
  switch(
    method_name,
    plssvd = "plssvd",
    simpls = "simpls",
    `simpls-fast` = "simpls-fast",
    method_name
  )
}

make_scale_spec <- function(metric_kind, perf_metric_name = NULL, values) {
  values <- values[is.finite(values)]
  if (!length(values)) {
    return(list(
      kind = if (metric_kind == "Time") "log10" else "continuous",
      limits = if (metric_kind == "Time") c(1, 10) else c(0, 1),
      breaks = if (metric_kind == "Time") c(1, 10) else c(0, 0.5, 1),
      labels = if (metric_kind == "Time") c("1", "10") else c("0.0", "0.5", "1.0")
    ))
  }

  if (metric_kind == "Time") {
    values <- values[values > 0]
    if (!length(values)) {
      return(list(kind = "log10", limits = c(1, 10), breaks = c(1, 10), labels = c("1", "10")))
    }
    lo <- 10 ^ floor(log10(min(values, na.rm = TRUE)))
    hi <- 10 ^ ceiling(log10(max(values, na.rm = TRUE)))
    br <- 10 ^ seq.int(log10(lo), log10(hi), by = 1)
    return(list(kind = "log10", limits = c(lo, hi), breaks = br, labels = as.character(br)))
  }

  if (metric_kind == "Performance") {
    pm <- tolower(perf_metric_name)
    if (pm == "accuracy") {
      br <- seq(0, 1, by = 0.2)
      return(list(kind = "continuous", limits = c(0, 1), breaks = br, labels = sprintf("%.1f", br)))
    }
    if (pm == "q2") {
      lo <- min(0, min(values, na.rm = TRUE))
      lo <- floor(lo * 5) / 5
      hi <- 1
      br <- pretty(c(lo, hi), n = 6)
      br <- br[br >= lo & br <= hi]
      return(list(kind = "continuous", limits = c(lo, hi), breaks = br, labels = sprintf("%.2f", br)))
    }
    lo <- 0
    hi <- max(values, na.rm = TRUE)
    if (!is.finite(hi) || hi <= lo) hi <- lo + 1
    br <- pretty(c(lo, hi), n = 6)
    br <- br[br >= lo & br <= hi]
    return(list(kind = "continuous", limits = c(lo, hi), breaks = br, labels = as.character(br)))
  }

  lo <- min(values, na.rm = TRUE)
  hi <- max(values, na.rm = TRUE)
  if (!is.finite(hi) || hi <= lo) hi <- lo + 1
  br <- pretty(c(lo, hi), n = 6)
  br <- br[br >= lo & br <= hi]
  list(kind = "continuous", limits = c(lo, hi), breaks = br, labels = as.character(br))
}

empty_panel <- function(title_text, y_label, scale_spec, x_breaks, x_limits) {
  y_mid <- if (scale_spec$kind == "log10") 10 ^ mean(log10(scale_spec$limits)) else mean(scale_spec$limits)
  p <- ggplot() +
    annotate("text", x = mean(x_limits), y = y_mid, label = "Not run", size = 5) +
    labs(title = title_text, x = "Number of components", y = y_label) +
    theme_bw(base_size = 11) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      panel.grid.minor = element_blank()
    ) +
    scale_x_continuous(
      breaks = x_breaks,
      limits = x_limits,
      expand = expansion(mult = c(0.01, 0.01))
    )

  if (scale_spec$kind == "log10") {
    p + scale_y_log10(
      limits = scale_spec$limits,
      breaks = scale_spec$breaks,
      labels = scale_spec$labels,
      expand = expansion(mult = c(0.02, 0.02))
    )
  } else {
    p + scale_y_continuous(
      limits = scale_spec$limits,
      breaks = scale_spec$breaks,
      labels = scale_spec$labels,
      expand = expansion(mult = c(0.02, 0.02))
    )
  }
}

build_panel <- function(dsub, panel_method, panel_metric, y_label, scale_spec, x_breaks, x_limits, show_legend = FALSE) {
  pdat <- dsub[method_panel == panel_method & metric == panel_metric]
  title_text <- panel_title(panel_method)

  if (!nrow(pdat)) {
    return(empty_panel(title_text, y_label, scale_spec, x_breaks, x_limits))
  }

  p <- ggplot(
    pdat,
    aes(
      x = requested_ncomp,
      y = metric_value,
      color = backend_family,
      shape = implementation_label,
      linetype = implementation_label,
      group = line_id
    )
  ) +
    geom_line(linewidth = 0.8, alpha = 0.95, na.rm = TRUE) +
    geom_point(size = 2.0, na.rm = TRUE) +
    labs(
      title = title_text,
      x = "Number of components",
      y = y_label,
      color = "SVD/backend",
      shape = "Implementation"
    ) +
    scale_color_manual(values = backend_palette, labels = backend_labels, drop = FALSE) +
    scale_shape_manual(values = shape_values, drop = FALSE) +
    scale_linetype_manual(values = linetype_values, drop = FALSE) +
    scale_x_continuous(
      breaks = x_breaks,
      limits = x_limits,
      expand = expansion(mult = c(0.01, 0.01))
    ) +
    theme_bw(base_size = 11) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      legend.position = if (show_legend) "bottom" else "none",
      panel.grid.minor = element_blank()
    )

  if (scale_spec$kind == "log10") {
    p + scale_y_log10(
      limits = scale_spec$limits,
      breaks = scale_spec$breaks,
      labels = scale_spec$labels,
      expand = expansion(mult = c(0.02, 0.02))
    )
  } else {
    p + scale_y_continuous(
      limits = scale_spec$limits,
      breaks = scale_spec$breaks,
      labels = scale_spec$labels,
      expand = expansion(mult = c(0.02, 0.02))
    )
  }
}

for (dataset_id in unique(summary_dt$dataset)) {
  dsub <- summary_dt[dataset == dataset_id]
  if (!nrow(dsub)) next

  x_breaks <- sort(unique(dsub$requested_ncomp))
  x_limits <- range(x_breaks)

  perf_metric_name <- first_non_missing(dsub[metric == "Performance", metric_name], "metric")
  time_scale <- make_scale_spec("Time", values = dsub[metric == "Time", metric_value])
  perf_scale <- make_scale_spec("Performance", perf_metric_name = perf_metric_name, values = dsub[metric == "Performance", metric_value])
  cpu_scale <- make_scale_spec("CPU memory", values = dsub[metric == "CPU memory", metric_value])
  gpu_scale <- make_scale_spec("GPU memory", values = dsub[metric == "GPU memory", metric_value])

  title_txt <- sprintf(
    "%s | %s | train_n=%s, test_n=%s, p=%s, classes/y=%s",
    dataset_id,
    first_non_missing(dsub$task_type, "task"),
    first_non_missing(dsub$n_train, NA_integer_),
    first_non_missing(dsub$n_test, NA_integer_),
    first_non_missing(dsub$p, NA_integer_),
    first_non_missing(dsub$n_classes, NA_integer_)
  )

  panels <- list(
    build_panel(dsub, "plssvd", "Time", "Total time (ms)", time_scale, x_breaks, x_limits),
    build_panel(dsub, "simpls", "Time", "Total time (ms)", time_scale, x_breaks, x_limits),
    build_panel(dsub, "simpls-fast", "Time", "Total time (ms)", time_scale, x_breaks, x_limits),

    build_panel(dsub, "plssvd", "Performance", metric_axis_label(perf_metric_name), perf_scale, x_breaks, x_limits),
    build_panel(dsub, "simpls", "Performance", metric_axis_label(perf_metric_name), perf_scale, x_breaks, x_limits),
    build_panel(dsub, "simpls-fast", "Performance", metric_axis_label(perf_metric_name), perf_scale, x_breaks, x_limits),

    build_panel(dsub, "plssvd", "CPU memory", "Peak host RSS (MB)", cpu_scale, x_breaks, x_limits),
    build_panel(dsub, "simpls", "CPU memory", "Peak host RSS (MB)", cpu_scale, x_breaks, x_limits),
    build_panel(dsub, "simpls-fast", "CPU memory", "Peak host RSS (MB)", cpu_scale, x_breaks, x_limits),

    build_panel(dsub, "plssvd", "GPU memory", "Peak GPU memory (MB)", gpu_scale, x_breaks, x_limits),
    build_panel(dsub, "simpls", "GPU memory", "Peak GPU memory (MB)", gpu_scale, x_breaks, x_limits),
    build_panel(dsub, "simpls-fast", "GPU memory", "Peak GPU memory (MB)", gpu_scale, x_breaks, x_limits, show_legend = TRUE)
  )

  combo <- ((panels[[1]] | panels[[2]] | panels[[3]]) /
              (panels[[4]] | panels[[5]] | panels[[6]]) /
              (panels[[7]] | panels[[8]] | panels[[9]]) /
              (panels[[10]] | panels[[11]] | panels[[12]])) +
    plot_layout(guides = "collect") +
    plot_annotation(title = title_txt) &
    theme(
      plot.title = element_text(face = "bold", size = 13),
      legend.position = "bottom",
      legend.box = "horizontal"
    )

  ggsave(
    file.path(plot_dir, sprintf("%s_3x4_methods_memory.png", dataset_id)),
    combo,
    width = 16,
    height = 18,
    dpi = 180
  )
}

cat("Plots written to:", plot_dir, "\n")
