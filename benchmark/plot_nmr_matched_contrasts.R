#!/usr/bin/env Rscript

# Re-express the existing fixed-100-component NMR results as matched contrasts.
# No model is refitted: this script reads archived predictions and summaries.

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1L) args[[1L]] else "."
out_dir <- if (length(args) >= 2L) args[[2L]] else
  file.path(root, "benchmark_results", "nmr_matched_contrasts_20260726")

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop("ggplot2 is required.", call. = FALSE)
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

reference_dir <- file.path(
  root, "benchmark_results", "review_nmr_reference_20260725"
)
simpls_dir <- file.path(root, "benchmark_results", "review_nmr_20260724")
summary_file <- file.path(
  simpls_dir, "plots", "nmr_final_method_summary.csv"
)

prediction_files <- c(
  "PLS-SVD CPU" = file.path(
    reference_dir, "predictions", "fastpls_plssvd_cpu_rsvd__rep1.rds"
  ),
  "PLS-SVD CUDA" = file.path(
    reference_dir, "predictions", "fastpls_plssvd_cuda_rsvd__rep1.rds"
  ),
  "SIMPLS CPU" = file.path(simpls_dir, "nmr_final_cpu.rds"),
  "SIMPLS CUDA" = file.path(simpls_dir, "nmr_final_cuda.rds")
)
stopifnot(all(file.exists(prediction_files)), file.exists(summary_file))

per_sample <- lapply(names(prediction_files), function(method) {
  object <- readRDS(prediction_files[[method]])
  data.frame(
    method = method,
    RMSD = sqrt(rowMeans((object$observed - object$predicted)^2)),
    stringsAsFactors = FALSE
  )
})
names(per_sample) <- names(prediction_files)

contrasts <- list(
  "Backend within PLS-SVD\n(family and solver fixed)" =
    c("PLS-SVD CPU", "PLS-SVD CUDA"),
  "Backend within SIMPLS\n(family and solver fixed)" =
    c("SIMPLS CPU", "SIMPLS CUDA"),
  "Family on CPU\n(backend and solver fixed)" =
    c("PLS-SVD CPU", "SIMPLS CPU"),
  "Family on CUDA\n(backend and solver fixed)" =
    c("PLS-SVD CUDA", "SIMPLS CUDA")
)

prediction_rows <- do.call(rbind, lapply(names(contrasts), function(contrast) {
  methods <- contrasts[[contrast]]
  rows <- do.call(rbind, per_sample[methods])
  rows$contrast <- contrast
  rows
}))
prediction_rows$contrast <- factor(
  prediction_rows$contrast, levels = names(contrasts)
)
prediction_rows$method <- factor(
  prediction_rows$method, levels = names(prediction_files)
)

colours <- c(
  "PLS-SVD CPU" = "#4E79A7",
  "PLS-SVD CUDA" = "#E15759",
  "SIMPLS CPU" = "#76B7B2",
  "SIMPLS CUDA" = "#F28E2B"
)
theme_publication <- ggplot2::theme_classic(base_size = 11) +
  ggplot2::theme(
    plot.title = ggplot2::element_text(face = "bold", size = 12),
    strip.text = ggplot2::element_text(face = "bold", size = 8),
    axis.title = ggplot2::element_text(face = "bold"),
    axis.text.x = ggplot2::element_text(angle = 25, hjust = 1, size = 7),
    legend.position = "none"
  )

p_prediction <- ggplot2::ggplot(
  prediction_rows, ggplot2::aes(method, RMSD, fill = method)
) +
  ggplot2::geom_boxplot(
    width = 0.62, outlier.alpha = 0.25, linewidth = 0.35
  ) +
  ggplot2::facet_wrap(~contrast, ncol = 2, scales = "free_x") +
  ggplot2::scale_fill_manual(values = colours, drop = FALSE) +
  ggplot2::labs(
    title = "Matched NMR prediction contrasts",
    x = NULL, y = "Per-spectrum RMSD"
  ) +
  theme_publication

ggplot2::ggsave(
  file.path(out_dir, "nmr_matched_prediction_contrasts.png"),
  p_prediction, width = 7.2, height = 5.0, dpi = 320
)
ggplot2::ggsave(
  file.path(out_dir, "nmr_matched_prediction_contrasts.pdf"),
  p_prediction, width = 7.2, height = 5.0
)

summary <- utils::read.csv(summary_file, check.names = FALSE)
summary$method <- as.character(summary$method)
value <- function(method, column) {
  summary[summary$method == method, column][[1L]]
}

resource_contrasts <- data.frame(
  contrast = c(
    "hardware_within_plssvd",
    "hardware_within_simpls",
    "family_on_cpu",
    "family_on_cuda"
  ),
  plot_label = c(
    "PLS-SVD\nbackend",
    "SIMPLS\nbackend",
    "CPU\nfamily",
    "CUDA\nfamily"
  ),
  numerator = c("CPU", "CPU", "SIMPLS", "SIMPLS"),
  denominator = c("CUDA", "CUDA", "PLS-SVD", "PLS-SVD"),
  time_ratio = c(
    value("PLS-SVD CPU", "total_time_sec") /
      value("PLS-SVD CUDA", "total_time_sec"),
    value("SIMPLS CPU", "total_time_sec") /
      value("SIMPLS CUDA", "total_time_sec"),
    value("SIMPLS CPU", "total_time_sec") /
      value("PLS-SVD CPU", "total_time_sec"),
    value("SIMPLS CUDA", "total_time_sec") /
      value("PLS-SVD CUDA", "total_time_sec")
  ),
  host_rss_ratio = c(
    value("PLS-SVD CUDA", "host_rss_mb") /
      value("PLS-SVD CPU", "host_rss_mb"),
    value("SIMPLS CUDA", "host_rss_mb") /
      value("SIMPLS CPU", "host_rss_mb"),
    value("SIMPLS CPU", "host_rss_mb") /
      value("PLS-SVD CPU", "host_rss_mb"),
    value("SIMPLS CUDA", "host_rss_mb") /
      value("PLS-SVD CUDA", "host_rss_mb")
  ),
  stringsAsFactors = FALSE
)
resource_long <- rbind(
  data.frame(
    contrast = resource_contrasts$plot_label,
    metric = "Runtime ratio",
    value = resource_contrasts$time_ratio
  ),
  data.frame(
    contrast = resource_contrasts$plot_label,
    metric = "Host RSS ratio",
    value = resource_contrasts$host_rss_ratio
  )
)
resource_long$contrast <- factor(
  resource_long$contrast, levels = resource_contrasts$plot_label
)
resource_long$metric <- factor(
  resource_long$metric, levels = c("Runtime ratio", "Host RSS ratio")
)

p_resources <- ggplot2::ggplot(
  resource_long, ggplot2::aes(contrast, value, fill = metric)
) +
  ggplot2::geom_hline(yintercept = 1, colour = "#555555", linetype = "dashed") +
  ggplot2::geom_col(width = 0.64) +
  ggplot2::geom_text(
    ggplot2::aes(label = sprintf("%.2f", value)),
    vjust = -0.25, size = 3
  ) +
  ggplot2::facet_wrap(~metric, scales = "free_y", ncol = 2) +
  ggplot2::scale_fill_manual(
    values = c("Runtime ratio" = "#4E79A7", "Host RSS ratio" = "#E15759")
  ) +
  ggplot2::scale_y_continuous(
    expand = ggplot2::expansion(mult = c(0, 0.12))
  ) +
  ggplot2::labs(
    title = "Matched NMR computational contrasts",
    subtitle = paste(
      "Backend: CPU/CUDA runtime and CUDA/CPU RSS;",
      "family: SIMPLS/PLS-SVD for both measures"
    ),
    x = NULL, y = "Ratio"
  ) +
  theme_publication +
  ggplot2::theme(
    axis.text.x = ggplot2::element_text(angle = 0, hjust = 0.5, size = 8),
    plot.subtitle = ggplot2::element_text(size = 8)
  )

ggplot2::ggsave(
  file.path(out_dir, "nmr_matched_resource_contrasts.png"),
  p_resources, width = 7.2, height = 5.0, dpi = 320
)
ggplot2::ggsave(
  file.path(out_dir, "nmr_matched_resource_contrasts.pdf"),
  p_resources, width = 7.2, height = 5.0
)

utils::write.csv(
  resource_contrasts[
    ,
    c(
      "contrast",
      "numerator",
      "denominator",
      "time_ratio",
      "host_rss_ratio"
    )
  ],
  file.path(out_dir, "nmr_matched_contrast_summary.csv"),
  row.names = FALSE
)
