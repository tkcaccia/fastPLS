#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(ggplot2))

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args)) args[[1L]] else
  file.path(getwd(), "benchmark_results", "metal_validation_20260726")
root <- normalizePath(root, winslash = "/", mustWork = TRUE)
out_dir <- file.path(root, "summary")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

files <- list.files(root, pattern = "metal_validation_raw\\.csv$",
                    recursive = TRUE, full.names = TRUE)
files <- files[!grepl("/smoke[^/]*/|_probe/", files)]
raw <- do.call(rbind, lapply(files, function(path) {
  x <- read.csv(path, stringsAsFactors = FALSE)
  for (column in c("kernel", "north")) {
    if (!column %in% names(x)) x[[column]] <- NA
  }
  x$source_file <- path
  x
}))

group_cols <- c(
  "experiment", "dataset", "task_type", "method", "backend_requested",
  "precision", "classifier", "svd_method", "oversample", "power",
  "kernel", "north", "ncomp", "n_train", "n_test", "p", "q"
)
key_data <- raw[group_cols]
for (column in names(key_data)) {
  if (is.numeric(key_data[[column]])) {
    key_data[[column]][is.na(key_data[[column]])] <- -999
  } else {
    key_data[[column]][is.na(key_data[[column]]) | !nzchar(key_data[[column]])] <-
      "<none>"
  }
}
key <- interaction(key_data, drop = TRUE, lex.order = TRUE)
safe_median <- function(x) {
  x <- x[is.finite(x)]
  if (length(x)) median(x) else NA_real_
}
safe_min <- function(x) {
  x <- x[is.finite(x)]
  if (length(x)) min(x) else NA_real_
}
safe_max <- function(x) {
  x <- x[is.finite(x)]
  if (length(x)) max(x) else NA_real_
}
summary_rows <- lapply(split(raw, key), function(x) {
  ok <- x$status == "success"
  data.frame(
    x[1L, group_cols, drop = FALSE],
    successes = sum(ok),
    failures = sum(!ok),
    median_total_sec = safe_median(x$total_sec[ok]),
    min_total_sec = safe_min(x$total_sec[ok]),
    max_total_sec = safe_max(x$total_sec[ok]),
    median_fit_sec = safe_median(x$fit_sec[ok]),
    median_prediction_sec = safe_median(x$prediction_sec[ok]),
    median_metric = safe_median(x$metric_value[ok]),
    min_metric = safe_min(x$metric_value[ok]),
    max_metric = safe_max(x$metric_value[ok]),
    median_peak_rss_mb = safe_median(x$peak_rss_mb[ok]),
    median_incremental_peak_rss_mb =
      safe_median(x$incremental_peak_rss_mb[ok]),
    stringsAsFactors = FALSE
  )
})
summary <- do.call(rbind, summary_rows)
write.csv(raw, file.path(out_dir, "metal_validation_all_raw.csv"), row.names = FALSE)
write.csv(summary, file.path(out_dir, "metal_validation_summary.csv"),
          row.names = FALSE)

real <- subset(
  summary,
  experiment == "real_dataset" & svd_method == "rsvd" &
    oversample == 10 & power == 1
)
real$implementation <- paste(real$backend_requested, real$precision, sep = " / ")
real$method_head <- paste(real$method, real$classifier, sep = " / ")

theme_pub <- theme_bw(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    strip.background = element_rect(fill = "grey95", colour = "grey60")
  )

if (nrow(real)) {
  p_time <- ggplot(real, aes(implementation, median_total_sec,
                             fill = backend_requested)) +
    geom_col(width = 0.75, colour = "black", linewidth = 0.25) +
    facet_grid(dataset ~ method_head, scales = "free_y") +
    scale_y_log10() +
    labs(x = NULL, y = "Median fitting + prediction time (s)",
         title = "Apple M3 CPU and Metal runtime (rSVD)") +
    theme_pub +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  ggsave(file.path(out_dir, "real_runtime.pdf"), p_time,
         width = 13, height = 8)

  p_metric <- ggplot(real, aes(implementation, median_metric,
                               fill = precision)) +
    geom_col(width = 0.75, colour = "black", linewidth = 0.25) +
    facet_grid(dataset ~ method_head) +
    scale_y_continuous(limits = c(0, 1), expand = expansion(mult = c(0, 0.05))) +
    labs(x = NULL, y = "Predictive metric",
         title = "Apple M3 predictive agreement (rSVD)") +
    theme_pub +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  ggsave(file.path(out_dir, "real_predictive_metric.pdf"), p_metric,
         width = 13, height = 8)

  p_memory <- ggplot(real, aes(implementation,
                               median_incremental_peak_rss_mb,
                               fill = precision)) +
    geom_col(width = 0.75, colour = "black", linewidth = 0.25) +
    facet_grid(dataset ~ method_head, scales = "free_y") +
    labs(x = NULL, y = "Incremental peak unified-process RSS (MB)",
         title = "Apple M3 memory use") +
    theme_pub +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  ggsave(file.path(out_dir, "real_memory.pdf"), p_memory,
         width = 13, height = 8)
}

scaling <- subset(summary, experiment == "synthetic_scaling")
if (nrow(scaling)) {
  scaling$implementation <- paste(
    toupper(scaling$backend_requested), toupper(scaling$method), sep = " / "
  )
  p_scaling <- ggplot(
    scaling,
    aes(dataset, median_total_sec, colour = backend_requested,
        shape = method, group = interaction(backend_requested, method))
  ) +
    geom_point(size = 2.8) +
    geom_line() +
    scale_y_log10() +
    labs(x = "Synthetic matrix regime", y = "Median total time (s)",
         title = "PLS-SVD and SIMPLS scaling on Apple M3") +
    theme_pub +
    theme(axis.text.x = element_text(angle = 25, hjust = 1))
  ggsave(file.path(out_dir, "synthetic_scaling.pdf"), p_scaling,
         width = 9, height = 5.5)
}

cat("Wrote summary to", out_dir, "\n")
