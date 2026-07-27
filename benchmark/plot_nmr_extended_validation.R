#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
selection_dir <- if (length(args) >= 1L) args[[1L]] else
  "benchmark_results/review_nmr_extended_selection_20260725"
final_dir <- if (length(args) >= 2L) args[[2L]] else
  "benchmark_results/review_nmr_20260724"
reference_dir <- if (length(args) >= 3L) args[[3L]] else
  "benchmark_results/review_nmr_reference_20260725"
output_dir <- if (length(args) >= 4L) args[[4L]] else
  file.path(selection_dir, "plots")

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop("ggplot2 is required.", call. = FALSE)
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
`%||%` <- function(x, y) if (is.null(x)) y else x

raw <- utils::read.csv(file.path(selection_dir, "nmr_component_selection_raw.csv"))
summary <- utils::read.csv(file.path(selection_dir, "nmr_component_selection_summary.csv"))
decision <- utils::read.csv(file.path(selection_dir, "nmr_component_selection_decision.csv"))
raw <- raw[raw$status == "ok", , drop = FALSE]

theme_nmr <- ggplot2::theme_classic(base_size = 14) +
  ggplot2::theme(
    plot.title = ggplot2::element_text(face = "bold"),
    plot.subtitle = ggplot2::element_text(size = 11),
    axis.title = ggplot2::element_text(face = "bold"),
    legend.position = "bottom"
  )

p_selection <- ggplot2::ggplot() +
  ggplot2::geom_line(
    data = raw,
    ggplot2::aes(ncomp, RMSD, group = factor(split_seed)),
    colour = "#7F7F7F", linewidth = 0.45, alpha = 0.5
  ) +
  ggplot2::geom_point(
    data = raw,
    ggplot2::aes(ncomp, RMSD, group = factor(split_seed)),
    colour = "#7F7F7F", size = 1.4, alpha = 0.55
  ) +
  ggplot2::geom_ribbon(
    data = summary,
    ggplot2::aes(ncomp, ymin = RMSD_q25, ymax = RMSD_q75),
    fill = "#4E79A7", alpha = 0.22
  ) +
  ggplot2::geom_line(
    data = summary, ggplot2::aes(ncomp, RMSD_median),
    colour = "#1F4E79", linewidth = 1.15
  ) +
  ggplot2::geom_point(
    data = summary, ggplot2::aes(ncomp, RMSD_median),
    colour = "#1F4E79", fill = "white", shape = 21, stroke = 0.8, size = 2.6
  ) +
  ggplot2::geom_vline(
    xintercept = decision$selected_ncomp, colour = "#D55E00",
    linetype = "dashed", linewidth = 0.75
  ) +
  ggplot2::scale_x_continuous(breaks = summary$ncomp) +
  ggplot2::labs(
    title = "Training-only selection of NMR components",
    subtitle = sprintf(
      "Five repeated 80/20 inner splits; median and interquartile range; selected = %d",
      decision$selected_ncomp
    ),
    x = "Number of components", y = "Validation RMSD"
  ) +
  theme_nmr +
  ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))

ggplot2::ggsave(
  file.path(output_dir, "nmr_component_selection_repeated.png"),
  p_selection, width = 9.5, height = 5.6, dpi = 320
)
ggplot2::ggsave(
  file.path(output_dir, "nmr_component_selection_repeated.pdf"),
  p_selection, width = 9.5, height = 5.6
)

method_files <- c(
  "Deposited reference" = file.path(
    reference_dir, "predictions", "deposited_fastsimpls_irlba__rep1.rds"
  ),
  "PLS-SVD CPU" = file.path(
    reference_dir, "predictions", "fastpls_plssvd_cpu_rsvd__rep1.rds"
  ),
  "PLS-SVD CUDA" = file.path(
    reference_dir, "predictions", "fastpls_plssvd_cuda_rsvd__rep1.rds"
  ),
  "SIMPLS CPU" = file.path(final_dir, "nmr_final_cpu.rds"),
  "SIMPLS CUDA" = file.path(final_dir, "nmr_final_cuda.rds")
)
missing <- method_files[!file.exists(method_files)]
if (length(missing)) {
  stop("Missing final prediction files: ", paste(missing, collapse = ", "), call. = FALSE)
}

response_rows <- list()
response_summary <- list()
for (method in names(method_files)) {
  object <- readRDS(method_files[[method]])
  observed <- object$observed
  predicted <- object$predicted
  response_rmsd <- sqrt(colMeans((observed - predicted)^2))
  response_mae <- colMeans(abs(observed - predicted))
  response_rows[[method]] <- data.frame(
    method = method,
    response = colnames(observed) %||% seq_len(ncol(observed)),
    RMSD = response_rmsd,
    MAE = response_mae,
    stringsAsFactors = FALSE
  )
  response_summary[[method]] <- data.frame(
    method = method,
    n_response = length(response_rmsd),
    RMSD_mean = mean(response_rmsd),
    RMSD_median = stats::median(response_rmsd),
    RMSD_q25 = unname(stats::quantile(response_rmsd, 0.25)),
    RMSD_q75 = unname(stats::quantile(response_rmsd, 0.75)),
    RMSD_q90 = unname(stats::quantile(response_rmsd, 0.90)),
    RMSD_q95 = unname(stats::quantile(response_rmsd, 0.95)),
    RMSD_q99 = unname(stats::quantile(response_rmsd, 0.99)),
    RMSD_max = max(response_rmsd),
    MAE_median = stats::median(response_mae),
    stringsAsFactors = FALSE
  )
  rm(object, observed, predicted)
  gc(full = TRUE)
}
response <- do.call(rbind, response_rows)
response_summary <- do.call(rbind, response_summary)
method_levels <- names(method_files)
response$method <- factor(response$method, levels = method_levels)

colours <- c(
  "Deposited reference" = "#7F7F7F",
  "PLS-SVD CPU" = "#4E79A7",
  "PLS-SVD CUDA" = "#E15759",
  "SIMPLS CPU" = "#76B7B2",
  "SIMPLS CUDA" = "#F28E2B"
)
p_response <- ggplot2::ggplot(
  response, ggplot2::aes(method, RMSD, fill = method)
) +
  ggplot2::geom_boxplot(
    width = 0.62, outlier.shape = NA, linewidth = 0.45
  ) +
  ggplot2::coord_cartesian(
    ylim = c(0, unname(stats::quantile(response$RMSD, 0.99)))
  ) +
  ggplot2::scale_fill_manual(values = colours, drop = FALSE) +
  ggplot2::labs(
    title = "Response-wise error across the predicted NMR spectrum",
    subtitle = "Boxes show 28,355 response-specific RMSD values; display truncated at the 99th percentile",
    x = NULL, y = "Response-wise RMSD"
  ) +
  theme_nmr +
  ggplot2::theme(
    legend.position = "none",
    axis.text.x = ggplot2::element_text(angle = 25, hjust = 1, size = 10)
  )

ggplot2::ggsave(
  file.path(output_dir, "nmr_responsewise_rmsd.png"),
  p_response, width = 9.5, height = 5.6, dpi = 320
)
ggplot2::ggsave(
  file.path(output_dir, "nmr_responsewise_rmsd.pdf"),
  p_response, width = 9.5, height = 5.6
)

utils::write.csv(
  response_summary,
  file.path(output_dir, "nmr_responsewise_error_summary.csv"),
  row.names = FALSE
)
utils::write.csv(
  response,
  file.path(output_dir, "nmr_responsewise_error_raw.csv"),
  row.names = FALSE
)
