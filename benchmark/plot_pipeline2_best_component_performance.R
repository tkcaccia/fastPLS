#!/usr/bin/env Rscript

# Select each method's best component count from replicate-level benchmark data
# and create the cross-dataset performance figure used in the manuscript.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) {
  stop("Usage: plot_pipeline2_best_component_performance.R RAW_CSV OUTPUT_DIR", call. = FALSE)
}

raw_path <- normalizePath(args[[1L]], mustWork = TRUE)
output_dir <- args[[2L]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop("ggplot2 is required to create the pipeline 2 best-component figure.", call. = FALSE)
}

raw <- utils::read.csv(raw_path, stringsAsFactors = FALSE, check.names = FALSE)
needed <- c("dataset", "method_id", "ncomp_requested", "status", "metric_name",
            "metric_value", "total_runtime_ms")
missing <- setdiff(needed, names(raw))
if (length(missing)) {
  stop("Benchmark CSV is missing: ", paste(missing, collapse = ", "), call. = FALSE)
}

ok <- raw[tolower(raw$status) == "ok" & is.finite(raw$metric_value), , drop = FALSE]
if (!nrow(ok)) stop("No successful rows with finite predictive metrics.", call. = FALSE)

median_or_na <- function(x) {
  x <- x[is.finite(x)]
  if (!length(x)) NA_real_ else stats::median(x)
}

group_key <- interaction(ok$dataset, ok$method_id, ok$ncomp_requested, drop = TRUE)
component_summary <- do.call(rbind, lapply(split(ok, group_key), function(d) {
  data.frame(
    dataset = d$dataset[[1L]],
    method_id = d$method_id[[1L]],
    package = if ("package" %in% names(d)) d$package[[1L]] else NA_character_,
    algorithm = if ("algorithm" %in% names(d)) d$algorithm[[1L]] else NA_character_,
    classifier = if ("classifier" %in% names(d)) d$classifier[[1L]] else NA_character_,
    metric_name = tolower(d$metric_name[[1L]]),
    ncomp = as.integer(d$ncomp_requested[[1L]]),
    metric_value = median_or_na(d$metric_value),
    total_runtime_ms = median_or_na(d$total_runtime_ms),
    replicate_count = nrow(d),
    stringsAsFactors = FALSE
  )
}))

is_loss <- function(metric) metric %in% c("rmsd", "rmse", "mae", "mse")
best_rows <- do.call(rbind, lapply(split(component_summary,
                                         interaction(component_summary$dataset,
                                                     component_summary$method_id,
                                                     drop = TRUE)), function(d) {
  direction <- if (is_loss(d$metric_name[[1L]])) "min" else "max"
  target <- if (identical(direction, "min")) min(d$metric_value, na.rm = TRUE) else max(d$metric_value, na.rm = TRUE)
  tied <- d[is.finite(d$metric_value) & abs(d$metric_value - target) <= .Machine$double.eps^0.5, , drop = FALSE]
  tied[order(tied$ncomp, tied$total_runtime_ms), , drop = FALSE][1L, , drop = FALSE]
}))
row.names(best_rows) <- NULL
best_rows$method_label <- gsub("_", " ", best_rows$method_id, fixed = TRUE)
best_rows$dataset <- factor(best_rows$dataset, levels = unique(best_rows$dataset))
best_rows$method_label <- factor(best_rows$method_label,
                                 levels = rev(unique(best_rows$method_label)))
utils::write.csv(best_rows,
                 file.path(output_dir, "pipeline2_best_component_performance.csv"),
                 row.names = FALSE, quote = TRUE, na = "")

base_theme <- ggplot2::theme_bw(base_size = 10) +
  ggplot2::theme(
    strip.text = ggplot2::element_text(face = "bold"),
    axis.text.y = ggplot2::element_text(size = 7),
    panel.grid.minor = ggplot2::element_blank(),
    legend.position = "bottom",
    legend.title = ggplot2::element_blank()
  )

p_metric <- ggplot2::ggplot(best_rows, ggplot2::aes(method_label, metric_value, colour = package)) +
  ggplot2::geom_point(size = 2.1, alpha = 0.9, na.rm = TRUE) +
  ggplot2::coord_flip() +
  ggplot2::facet_wrap(~dataset, scales = "free_y", ncol = 4) +
  ggplot2::labs(
    title = "Predictive performance at each method's selected component count",
    subtitle = "Accuracy and Q2 are maximized; RMSD is minimized. Point labels give selected components.",
    x = NULL, y = "Dataset-specific predictive metric", colour = "Implementation"
  ) +
  ggplot2::geom_text(ggplot2::aes(label = paste0("k=", ncomp)),
                     hjust = -0.12, size = 2.2, show.legend = FALSE, na.rm = TRUE) +
  ggplot2::expand_limits(y = 0) + base_theme

p_time <- ggplot2::ggplot(best_rows,
                           ggplot2::aes(method_label, total_runtime_ms / 1000, colour = package)) +
  ggplot2::geom_point(size = 2.1, alpha = 0.9, na.rm = TRUE) +
  ggplot2::scale_y_log10() +
  ggplot2::coord_flip() +
  ggplot2::facet_wrap(~dataset, scales = "free_y", ncol = 4) +
  ggplot2::labs(
    title = "Total time at the selected component count",
    x = NULL, y = "Fit plus prediction time (s, log scale)", colour = "Implementation"
  ) + base_theme

save_plot <- function(plot, stem) {
  ggplot2::ggsave(file.path(output_dir, paste0(stem, ".png")), plot,
                  width = 15, height = 10, dpi = 320, bg = "white")
  ggplot2::ggsave(file.path(output_dir, paste0(stem, ".pdf")), plot,
                  width = 15, height = 10, device = grDevices::cairo_pdf)
}
save_plot(p_metric, "pipeline2_best_component_prediction")
save_plot(p_time, "pipeline2_best_component_time")

message("Wrote best-component table and figures to: ", output_dir)
