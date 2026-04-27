#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_file <- if (length(script_arg)) sub("^--file=", "", script_arg[[1L]]) else file.path(getwd(), "plot_cifar100_remote_compare.R")
script_dir <- dirname(normalizePath(script_file, winslash = "/", mustWork = FALSE))

args <- commandArgs(trailingOnly = TRUE)
parse_args <- function(x) {
  out <- list()
  for (arg in x) {
    if (!startsWith(arg, "--")) next
    bits <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
    out[[gsub("-", "_", bits[[1L]], fixed = TRUE)]] <- paste(bits[-1L], collapse = "=")
  }
  out
}
arg_map <- parse_args(args)
results_dir <- normalizePath(if (!is.null(arg_map$results_dir)) arg_map$results_dir else Sys.getenv("FASTPLS_RESULTS_DIR", getwd()), winslash = "/", mustWork = TRUE)

raw_csv <- file.path(results_dir, "cifar100_remote_compare_raw.csv")
summary_csv <- file.path(results_dir, "cifar100_remote_compare_summary.csv")
if (!file.exists(raw_csv)) stop("Missing raw CSV: ", raw_csv)
if (!file.exists(summary_csv)) stop("Missing summary CSV: ", summary_csv)

raw <- fread(raw_csv)
sumtab <- fread(summary_csv)
plot_dir <- file.path(results_dir, "plots")
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

ok <- sumtab[reps_ok > 0]
if (!nrow(ok)) stop("No successful rows available for plotting")

variant_colors <- c(
  baseline_gpu_plssvd = "#1b9e77",
  baseline_gpu_simpls_fast = "#d95f02",
  baseline_cpu_plssvd_cpu_rsvd = "#7570b3",
  baseline_cpu_simpls_fast_cpu_rsvd = "#e7298a",
  baseline_cpu_plssvd_irlba = "#66a61e",
  baseline_cpu_simpls_fast_irlba = "#e6ab02",
  test_gpu_plssvd = "#a6761d",
  test_gpu_simpls_fast = "#666666",
  test_gpu_plssvd_host_qr = "#1f78b4",
  test_gpu_simpls_fast_host_qr = "#33a02c",
  test_gpu_plssvd_qless = "#6a3d9a",
  test_gpu_simpls_fast_qless = "#b15928",
  test_cpu_plssvd_cpu_rsvd = "#1f78b4",
  test_cpu_simpls_fast_cpu_rsvd = "#b2df8a",
  test_cpu_plssvd_irlba = "#fb9a99",
  test_cpu_simpls_fast_irlba = "#cab2d6"
)
known_variants <- unique(ok$variant_name)
variant_colors <- variant_colors[names(variant_colors) %in% known_variants]
missing_variants <- setdiff(known_variants, names(variant_colors))
if (length(missing_variants)) {
  fallback_cols <- grDevices::hcl.colors(length(missing_variants), palette = "Dark 3")
  names(fallback_cols) <- missing_variants
  variant_colors <- c(variant_colors, fallback_cols)
}

common_theme <- theme_bw(base_size = 11) +
  theme(
    legend.position = "bottom",
    legend.box = "vertical",
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold")
  )

line_plot <- function(dt, y, y_iqr, title, ylab, file_name, facet = FALSE, filter_gpu_only = FALSE) {
  pdat <- copy(dt)
  if (filter_gpu_only) pdat <- pdat[engine == "GPU"]
  if (!nrow(pdat)) return(invisible(NULL))
  p <- ggplot(pdat, aes(x = requested_ncomp, y = .data[[y]], color = variant_name, group = variant_name)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2) +
    geom_errorbar(aes(ymin = pmax(.data[[y]] - .data[[y_iqr]]/2, 0), ymax = .data[[y]] + .data[[y_iqr]]/2), width = 4, alpha = 0.5) +
    scale_color_manual(values = variant_colors, drop = FALSE) +
    labs(title = title, x = "ncomp", y = ylab, color = "Variant") +
    common_theme
  if (isTRUE(facet)) {
    p <- p + facet_wrap(~method_family, scales = "free_y")
  }
  ggsave(file.path(plot_dir, file_name), p, width = 11, height = if (facet) 6.5 else 5.5, dpi = 170)
}

line_plot(ok[method_family == "plssvd"], "total_time_ms_median", "total_time_ms_iqr", "PLSSVD time vs ncomp", "Median total time (ms)", "time_vs_ncomp_plssvd.png")
line_plot(ok[method_family == "simpls_fast"], "total_time_ms_median", "total_time_ms_iqr", "SIMPLS-fast time vs ncomp", "Median total time (ms)", "time_vs_ncomp_simpls_fast.png")
line_plot(ok[method_family == "plssvd"], "accuracy_median", "accuracy_iqr", "PLSSVD accuracy vs ncomp", "Median top-1 accuracy", "accuracy_vs_ncomp_plssvd.png")
line_plot(ok[method_family == "simpls_fast"], "accuracy_median", "accuracy_iqr", "SIMPLS-fast accuracy vs ncomp", "Median top-1 accuracy", "accuracy_vs_ncomp_simpls_fast.png")
line_plot(ok, "peak_host_rss_mb_median", "peak_host_rss_mb_iqr", "Host RSS vs ncomp", "Peak host RSS (MB)", "host_rss_vs_ncomp.png", facet = TRUE)
line_plot(ok[engine == "GPU"], "peak_gpu_mem_mb_median", "peak_gpu_mem_mb_iqr", "GPU peak memory vs ncomp", "Peak GPU memory (MB)", "gpu_peak_mem_vs_ncomp.png", facet = TRUE, filter_gpu_only = TRUE)

baseline_gpu <- ok[grepl("^baseline_gpu_", variant_name), .(method_family, requested_ncomp, base_total = total_time_ms_median)]
speed_gpu <- merge(ok, baseline_gpu, by = c("method_family", "requested_ncomp"), all.x = TRUE)
speed_gpu[, speedup_vs_baseline_gpu := base_total / total_time_ms_median]
line_plot(speed_gpu[is.finite(speedup_vs_baseline_gpu)], "speedup_vs_baseline_gpu", "total_time_ms_iqr", "Speedup vs baseline GPU", "Speedup (baseline GPU / variant)", "speedup_vs_baseline_gpu.png", facet = TRUE)

baseline_cpu <- ok[grepl("^baseline_cpu_.*_cpu_rsvd$", variant_name), .(method_family, requested_ncomp, base_cpu_total = total_time_ms_median)]
speed_cpu <- merge(ok, baseline_cpu, by = c("method_family", "requested_ncomp"), all.x = TRUE)
speed_cpu[, speedup_vs_cpu_reference := base_cpu_total / total_time_ms_median]
line_plot(speed_cpu[is.finite(speedup_vs_cpu_reference)], "speedup_vs_cpu_reference", "total_time_ms_iqr", "Speedup vs CPU reference", "Speedup (baseline CPU rsvd / variant)", "speedup_vs_cpu_reference.png", facet = TRUE)

trade <- ok[requested_ncomp == 50]
if (nrow(trade)) {
  p_trade <- ggplot(trade, aes(
    x = total_time_ms_median,
    y = accuracy_median,
    color = variant_name,
    shape = engine,
    size = peak_host_rss_mb_median
  )) +
    geom_point(alpha = 0.9) +
    geom_text(aes(label = variant_name), hjust = 0, nudge_x = 0.01 * max(trade$total_time_ms_median, na.rm = TRUE), size = 3, show.legend = FALSE) +
    scale_color_manual(values = variant_colors, drop = FALSE) +
    facet_wrap(~method_family, scales = "free") +
    labs(
      title = "Speed / accuracy / memory tradeoff at ncomp = 50",
      x = "Median total time (ms)",
      y = "Median top-1 accuracy",
      color = "Variant",
      shape = "Engine",
      size = "Peak host RSS (MB)"
    ) +
    common_theme
  ggsave(file.path(plot_dir, "summary_tradeoff_accuracy_speed_memory.png"), p_trade, width = 12, height = 6, dpi = 170)
}

cat("Plots written to:", plot_dir, "\n")
