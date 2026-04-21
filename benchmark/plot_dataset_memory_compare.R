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

method_levels <- c("plssvd", "simpls", "simpls-fast")
metric_levels <- c("Time", "Performance", "CPU memory", "GPU memory")

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
    metric_value = median(get(metric_col), na.rm = TRUE),
    metric_iqr = IQR(get(metric_col), na.rm = TRUE),
    task_type = first_non_missing(task_type, NA_character_),
    metric_name = first_non_missing(metric_name, NA_character_),
    n_train = first_non_missing(n_train, NA_integer_),
    n_test = first_non_missing(n_test, NA_integer_),
    p = first_non_missing(p, NA_integer_),
    n_classes = first_non_missing(n_classes, NA_integer_)
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
summary_dt[, method_panel := factor(method_panel, levels = method_levels)]
summary_dt[, metric := factor(metric, levels = metric_levels)]

backend_levels <- c("cpu_rsvd", "irlba", "arpack", "gpu_fp64", "gpu_fp32", "pls_pkg")
backend_levels <- backend_levels[backend_levels %in% unique(summary_dt$backend_family)]
backend_palette <- c(
  cpu_rsvd = "#d95f02",
  irlba = "#1b9e77",
  arpack = "#7570b3",
  gpu_fp64 = "#e7298a",
  gpu_fp32 = "#66a61e",
  pls_pkg = "#666666"
)
backend_palette <- backend_palette[backend_levels]

shape_levels <- c("Cpp", "R", "CUDA 64-bit", "CUDA 32-bit", "pls_pkg")
shape_levels <- shape_levels[shape_levels %in% unique(summary_dt$implementation_label)]
shape_values <- c(
  Cpp = 16,
  R = 17,
  `CUDA 64-bit` = 18,
  `CUDA 32-bit` = 15,
  pls_pkg = 8
)
shape_values <- shape_values[shape_levels]

linetype_values <- c(
  Cpp = "solid",
  R = "22",
  `CUDA 64-bit` = "longdash",
  `CUDA 32-bit` = "dotdash",
  pls_pkg = "solid"
)
linetype_values <- linetype_values[shape_levels]

backend_labels <- c(
  cpu_rsvd = "rsvd",
  irlba = "irlba",
  arpack = "arpack",
  gpu_fp64 = "gpu fp64",
  gpu_fp32 = "gpu fp32",
  pls_pkg = "pls_pkg"
)
backend_labels <- backend_labels[backend_levels]

fwrite(summary_dt, file.path(results_dir, "dataset_memory_compare_summary.csv"))

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

expand_equal_limits <- function(lims, log_y = FALSE) {
  if (is.null(lims) || length(lims) != 2L || any(!is.finite(lims))) return(NULL)
  lo <- lims[[1L]]
  hi <- lims[[2L]]
  if (isTRUE(log_y)) {
    lo <- max(lo, 1e-12)
    hi <- max(hi, 1e-12)
    if (lo == hi) {
      lo <- lo / 1.25
      hi <- hi * 1.25
    }
    return(c(lo, hi))
  }
  if (lo == hi) {
    pad <- if (lo == 0) 1 else abs(lo) * 0.05
    lo <- lo - pad
    hi <- hi + pad
  }
  c(lo, hi)
}

compute_y_breaks <- function(lims, log_y = FALSE) {
  lims <- expand_equal_limits(lims, log_y = log_y)
  if (is.null(lims)) return(NULL)
  if (isTRUE(log_y)) {
    lo_exp <- floor(log10(lims[[1L]]))
    hi_exp <- ceiling(log10(lims[[2L]]))
    br <- 10 ^ seq(lo_exp, hi_exp, by = 1)
    br <- br[br >= lims[[1L]] & br <= lims[[2L]]]
    if (!length(br)) br <- lims
    return(br)
  }
  br <- pretty(lims, n = 5)
  br <- br[br >= lims[[1L]] & br <= lims[[2L]]]
  if (!length(br)) br <- lims
  br
}

panel_title <- function(dataset_id) {
  switch(
    tolower(dataset_id),
    cifar100 = "CIFAR100",
    ccle = "CCLE",
    dataset_id
  )
}

method_title <- function(method_name) {
  switch(
    method_name,
    plssvd = "plssvd",
    simpls = "simpls",
    `simpls-fast` = "simpls-fast",
    method_name
  )
}

empty_panel_plot <- function(title_text, y_label = NULL, y_limits = NULL, y_breaks = NULL, log_y = FALSE, x_breaks = NULL, x_limits = NULL) {
  if (is.null(x_breaks) || !length(x_breaks)) {
    x_breaks <- c(0, 1)
  }
  if (is.null(x_limits) || length(x_limits) != 2L || any(!is.finite(x_limits))) {
    x_limits <- range(x_breaks, na.rm = TRUE)
  }

  if (is.null(y_limits) || length(y_limits) != 2L || any(!is.finite(y_limits))) y_limits <- c(0, 1)
  y_limits <- expand_equal_limits(y_limits, log_y = log_y)
  if (is.null(y_breaks)) y_breaks <- compute_y_breaks(y_limits, log_y = log_y)

  if (isTRUE(log_y)) {
    y_limits <- pmax(y_limits, 1e-12)
    y_mid <- 10 ^ mean(log10(y_limits))
  } else {
    y_mid <- mean(y_limits)
  }

  p <- ggplot() +
    annotate("text", x = mean(x_limits), y = y_mid, label = "Not run", size = 5) +
    labs(title = title_text, x = "Number of components", y = y_label) +
    theme_bw(base_size = 12) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      panel.grid.minor = element_blank(),
      axis.title.y = if (is.null(y_label)) element_blank() else element_text()
    ) +
    scale_x_continuous(breaks = x_breaks, limits = x_limits, expand = expansion(mult = c(0.01, 0.01)))

  if (isTRUE(log_y)) {
    p <- p + scale_y_log10(limits = y_limits, breaks = y_breaks, expand = expansion(mult = c(0.01, 0.03)))
  } else {
    p <- p + scale_y_continuous(limits = y_limits, breaks = y_breaks, expand = expansion(mult = c(0.01, 0.03)))
  }

  p
}

build_metric_panel <- function(dsub, panel_method, panel_metric, y_label, show_legend = FALSE, y_limits = NULL, y_breaks = NULL, log_y = FALSE, x_breaks = NULL, x_limits = NULL) {
  pdat <- dsub[method_panel == panel_method & metric == panel_metric]
  title_text <- method_title(panel_method)

  if (!nrow(pdat)) {
    return(empty_panel_plot(
      title_text,
      y_label,
      y_limits = y_limits,
      y_breaks = y_breaks,
      log_y = log_y,
      x_breaks = x_breaks,
      x_limits = x_limits
    ))
  }

  if (is.null(x_breaks)) {
    x_breaks <- sort(unique(dsub$requested_ncomp))
  }
  if (is.null(x_limits) && length(x_breaks)) {
    x_limits <- range(x_breaks, na.rm = TRUE)
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
    scale_color_manual(
      values = backend_palette,
      breaks = backend_levels,
      labels = backend_labels,
      drop = FALSE
    ) +
    scale_shape_manual(values = shape_values, breaks = shape_levels, drop = FALSE) +
    scale_linetype_manual(values = linetype_values, breaks = shape_levels, drop = FALSE) +
    scale_x_continuous(
      breaks = x_breaks,
      limits = x_limits,
      expand = expansion(mult = c(0.01, 0.01))
    ) +
    guides(
      color = guide_legend(order = 1, nrow = 2, byrow = TRUE),
      shape = guide_legend(order = 2, nrow = 2, byrow = TRUE),
      linetype = "none"
    ) +
    labs(
      title = title_text,
      x = "Number of components",
      y = y_label,
      color = "SVD/backend",
      shape = "Implementation"
    ) +
    theme_bw(base_size = 11) +
    theme(
      legend.position = if (show_legend) "bottom" else "none",
      legend.box = "horizontal",
      legend.justification = "center",
      plot.title = element_text(face = "bold", hjust = 0.5),
      panel.grid.minor = element_blank()
    )

  y_limits <- expand_equal_limits(y_limits, log_y = log_y)
  if (is.null(y_breaks)) y_breaks <- compute_y_breaks(y_limits, log_y = log_y)

  if (!is.null(y_limits) && all(is.finite(y_limits)) && length(y_limits) == 2L) {
    if (isTRUE(log_y)) {
      p <- p + scale_y_log10(limits = y_limits, breaks = y_breaks, expand = expansion(mult = c(0.01, 0.03)))
    } else {
      p <- p + scale_y_continuous(limits = y_limits, breaks = y_breaks, expand = expansion(mult = c(0.01, 0.03)))
    }
  } else if (isTRUE(log_y)) {
    p <- p + scale_y_log10()
  }

  p
}

for (ds in unique(summary_dt$dataset)) {
  dsub <- summary_dt[dataset == ds]
  if (!nrow(dsub)) next

  x_breaks <- sort(unique(as.numeric(as.character(dsub$requested_ncomp))))
  x_breaks <- x_breaks[is.finite(x_breaks)]
  x_limits <- if (length(x_breaks)) range(x_breaks, na.rm = TRUE) else NULL

  dims_row <- dsub[1L]
  perf_metric_name <- first_non_missing(dsub[metric == "Performance", metric_name], "metric")
  perf_axis_label <- metric_axis_label(perf_metric_name)
  title_txt <- sprintf(
    "%s | %s | train_n=%s, test_n=%s, p=%s, classes/y=%s",
    panel_title(ds),
    first_non_missing(dsub$task_type, "task"),
    dims_row$n_train,
    dims_row$n_test,
    dims_row$p,
    dims_row$n_classes
  )

  row_limits <- list()
  row_breaks <- list()
  for (metric_name in metric_levels) {
    vals <- dsub[metric == metric_name, metric_value]
    vals <- vals[is.finite(vals)]
    is_log <- identical(metric_name, "Time")
    if (is_log) vals <- vals[vals > 0]
    lims <- if (length(vals)) range(vals, na.rm = TRUE) else NULL
    row_limits[[metric_name]] <- expand_equal_limits(lims, log_y = is_log)
    row_breaks[[metric_name]] <- compute_y_breaks(row_limits[[metric_name]], log_y = is_log)
  }

  panels <- list(
    build_metric_panel(dsub, "plssvd", "Time", "Total time (ms)", y_limits = row_limits[["Time"]], y_breaks = row_breaks[["Time"]], log_y = TRUE, x_breaks = x_breaks, x_limits = x_limits),
    build_metric_panel(dsub, "simpls", "Time", "Total time (ms)", y_limits = row_limits[["Time"]], y_breaks = row_breaks[["Time"]], log_y = TRUE, x_breaks = x_breaks, x_limits = x_limits),
    build_metric_panel(dsub, "simpls-fast", "Time", "Total time (ms)", y_limits = row_limits[["Time"]], y_breaks = row_breaks[["Time"]], log_y = TRUE, x_breaks = x_breaks, x_limits = x_limits),
    build_metric_panel(dsub, "plssvd", "Performance", perf_axis_label, y_limits = row_limits[["Performance"]], y_breaks = row_breaks[["Performance"]], x_breaks = x_breaks, x_limits = x_limits),
    build_metric_panel(dsub, "simpls", "Performance", perf_axis_label, y_limits = row_limits[["Performance"]], y_breaks = row_breaks[["Performance"]], x_breaks = x_breaks, x_limits = x_limits),
    build_metric_panel(dsub, "simpls-fast", "Performance", perf_axis_label, y_limits = row_limits[["Performance"]], y_breaks = row_breaks[["Performance"]], x_breaks = x_breaks, x_limits = x_limits),
    build_metric_panel(dsub, "plssvd", "CPU memory", "Peak host RSS (MB)", y_limits = row_limits[["CPU memory"]], y_breaks = row_breaks[["CPU memory"]], x_breaks = x_breaks, x_limits = x_limits),
    build_metric_panel(dsub, "simpls", "CPU memory", "Peak host RSS (MB)", y_limits = row_limits[["CPU memory"]], y_breaks = row_breaks[["CPU memory"]], x_breaks = x_breaks, x_limits = x_limits),
    build_metric_panel(dsub, "simpls-fast", "CPU memory", "Peak host RSS (MB)", y_limits = row_limits[["CPU memory"]], y_breaks = row_breaks[["CPU memory"]], x_breaks = x_breaks, x_limits = x_limits),
    build_metric_panel(dsub, "plssvd", "GPU memory", "Peak GPU memory (MB)", y_limits = row_limits[["GPU memory"]], y_breaks = row_breaks[["GPU memory"]], x_breaks = x_breaks, x_limits = x_limits),
    build_metric_panel(dsub, "simpls", "GPU memory", "Peak GPU memory (MB)", y_limits = row_limits[["GPU memory"]], y_breaks = row_breaks[["GPU memory"]], x_breaks = x_breaks, x_limits = x_limits),
    build_metric_panel(dsub, "simpls-fast", "GPU memory", "Peak GPU memory (MB)", show_legend = TRUE, y_limits = row_limits[["GPU memory"]], y_breaks = row_breaks[["GPU memory"]], x_breaks = x_breaks, x_limits = x_limits)
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
      legend.box = "horizontal",
      legend.justification = "center",
      legend.direction = "horizontal"
    )

  ggsave(
    filename = file.path(plot_dir, sprintf("%s_3x4_methods_memory.png", ds)),
    plot = combo,
    width = 16,
    height = 18,
    dpi = 180
  )
}

cat("Plots written to:", plot_dir, "\n")
