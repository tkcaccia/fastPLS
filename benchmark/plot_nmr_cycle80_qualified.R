#!/usr/bin/env Rscript

# Build the qualified NMR main figure and machine-readable comparison tables.
# The figure distinguishes family-selected predictive workflows from matched
# solver/backend comparisons and from the historical deposited workflow.

options(stringsAsFactors = FALSE)
`%||%` <- function(x, y) {
  if (is.null(x) || !length(x)) y else x
}

args <- commandArgs(trailingOnly = TRUE)
qualified_dir <- if (length(args) >= 1L) args[[1L]] else
  "benchmark_results/manuscript_revision_cycle80_20260727/nmr_qualified"
historical_dir <- if (length(args) >= 2L) args[[2L]] else
  "/Users/stefano/Documents/GPUPLS/manuscript_work_20260722/evidence/nmr_reference_cycle4"
historical_summary_file <- if (length(args) >= 3L) args[[3L]] else
  "benchmark_results/manuscript_revision_cycle64_20260726/nmr_historical_reference_165_summary.csv"
output_dir <- if (length(args) >= 4L) args[[4L]] else qualified_dir

if (!requireNamespace("ggplot2", quietly = TRUE) ||
    !requireNamespace("patchwork", quietly = TRUE)) {
  stop("ggplot2 and patchwork are required.", call. = FALSE)
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

theme_nmr <- ggplot2::theme_classic(base_size = 11) +
  ggplot2::theme(
    plot.title = ggplot2::element_text(face = "bold", size = 11),
    plot.subtitle = ggplot2::element_text(size = 8.5),
    axis.title = ggplot2::element_text(face = "bold"),
    axis.text = ggplot2::element_text(colour = "black"),
    strip.background = ggplot2::element_rect(
      fill = "#F2F2F2", colour = "#B3B3B3"
    ),
    strip.text = ggplot2::element_text(face = "bold", size = 9),
    legend.position = "bottom",
    legend.title = ggplot2::element_blank()
  )

colours <- c(
  "Deposited PLS-SVD/IRLBA (165)" = "#6B6B6B",
  "PLS-SVD CUDA rSVD (5)" = "#0072B2",
  "SIMPLS CUDA rSVD (50)" = "#D55E00",
  "Observed" = "#111111"
)

qualified <- read.csv(file.path(qualified_dir, "nmr_qualified_summary.csv"))
qualified$gpu_incremental_mb <- NA_real_
gpu_memory_file <- file.path(
  qualified_dir,
  "nmr_cuda_gpu_memory_summary.csv"
)
if (file.exists(gpu_memory_file)) {
  gpu_memory <- read.csv(gpu_memory_file)
  for (i in seq_len(nrow(gpu_memory))) {
    match_row <- qualified$family == gpu_memory$family[[i]] &
      qualified$backend == "cuda" &
      qualified$solver == "rsvd" &
      qualified$ncomp == gpu_memory$ncomp[[i]]
    qualified$gpu_incremental_mb[match_row] <-
      gpu_memory$incremental_gpu_mb[[i]]
  }
}
agreement <- read.csv(file.path(qualified_dir, "nmr_qualified_agreement.csv"))
per_sample <- read.csv(file.path(qualified_dir, "nmr_qualified_per_sample.csv"))
per_response <- read.csv(file.path(
  qualified_dir, "nmr_qualified_per_response.csv"
))
curve <- read.csv(file.path(
  qualified_dir, "nmr_representative_spectrum.csv"
))
selection <- read.csv(file.path(
  qualified_dir, "nmr_representative_spectrum_selection.csv"
))
historical_summary <- read.csv(historical_summary_file)

historical_prediction <- readRDS(file.path(
  historical_dir, "legacy_fastsimpls_plssvd_irlba_prediction.rds"
))
observed <- readRDS(file.path(historical_dir, "nmr_observed_test_only.rds"))
historical_sample_rmsd <- sqrt(rowMeans(
  (observed - historical_prediction)^2
))
historical_response_rmsd <- sqrt(colMeans(
  (observed - historical_prediction)^2
))
historical_center <- attr(historical_prediction, "scaled:center")
if (is.null(historical_center) ||
    length(historical_center) != ncol(observed)) {
  historical_q2 <- NA_real_
} else {
  historical_q2 <- 1 - sum(
    (observed - historical_prediction)^2
  ) / sum(sweep(observed, 2L, historical_center, "-")^2)
}
historical_rmsd <- sqrt(mean((observed - historical_prediction)^2))

selected_labels <- c(
  "PLS-SVD CUDA rSVD" = "PLS-SVD CUDA rSVD (5)",
  "SIMPLS CUDA rSVD" = "SIMPLS CUDA rSVD (50)"
)
selected <- qualified[
  qualified$label %in% names(selected_labels), , drop = FALSE
]
selected$workflow <- unname(selected_labels[selected$label])

metric_long <- rbind(
  data.frame(
    workflow = "Deposited PLS-SVD/IRLBA (165)",
    metric = c("RMSD", "Q\u00b2"),
    value = c(historical_rmsd, historical_q2)
  ),
  do.call(rbind, lapply(seq_len(nrow(selected)), function(i) {
    data.frame(
      workflow = selected$workflow[[i]],
      metric = c("RMSD", "Q\u00b2"),
      value = c(selected$RMSD[[i]], selected$Q2[[i]])
    )
  }))
)
metric_long$workflow <- factor(
  metric_long$workflow,
  levels = names(colours)[seq_len(3L)]
)
p_metric <- ggplot2::ggplot(
  metric_long, ggplot2::aes(workflow, value, fill = workflow)
) +
  ggplot2::geom_col(width = 0.66, colour = "black", linewidth = 0.3) +
  ggplot2::facet_wrap(~metric, scales = "free_y", nrow = 1L) +
  ggplot2::scale_fill_manual(values = colours) +
  ggplot2::scale_x_discrete(labels = c(
    "Deposited PLS-SVD/IRLBA (165)" = "Reference\nPLS-SVD/IRLBA\n(165)",
    "PLS-SVD CUDA rSVD (5)" = "PLS-SVD\nCUDA rSVD\n(5)",
    "SIMPLS CUDA rSVD (50)" = "SIMPLS\nCUDA rSVD\n(50)"
  )) +
  ggplot2::scale_y_continuous(labels = function(x) format(
    x, digits = 4, scientific = FALSE
  )) +
  ggplot2::labs(
    title = "A  Predictive performance",
    subtitle = "Family-selected workflows; the deposited model is historical context",
    x = NULL, y = NULL
  ) +
  theme_nmr +
  ggplot2::guides(fill = "none") +
  ggplot2::theme(
    legend.position = "none",
    axis.text.x = ggplot2::element_text(size = 7.5)
  )

sample_plot <- subset(
  per_sample,
  label %in% names(selected_labels)
)
sample_plot$workflow <- unname(selected_labels[sample_plot$label])
sample_plot <- rbind(
  sample_plot[, c("workflow", "RMSD")],
  data.frame(
    workflow = "Deposited PLS-SVD/IRLBA (165)",
    RMSD = historical_sample_rmsd
  )
)
sample_plot$workflow <- factor(
  sample_plot$workflow,
  levels = names(colours)[seq_len(3L)]
)
p_sample <- ggplot2::ggplot(
  sample_plot, ggplot2::aes(workflow, RMSD, fill = workflow)
) +
  ggplot2::geom_boxplot(
    width = 0.62, outlier.shape = NA, colour = "black", linewidth = 0.35
  ) +
  ggplot2::coord_cartesian(
    ylim = c(0, unname(stats::quantile(sample_plot$RMSD, 0.99)))
  ) +
  ggplot2::scale_fill_manual(values = colours) +
  ggplot2::scale_x_discrete(labels = c(
    "Deposited PLS-SVD/IRLBA (165)" = "Reference\nPLS-SVD/IRLBA\n(165)",
    "PLS-SVD CUDA rSVD (5)" = "PLS-SVD\nCUDA rSVD\n(5)",
    "SIMPLS CUDA rSVD (50)" = "SIMPLS\nCUDA rSVD\n(50)"
  )) +
  ggplot2::labs(
    title = "B  Held-out spectrum errors",
    subtitle = "321 per-spectrum RMSD values; display limited at the 99th percentile",
    x = NULL, y = "Per-spectrum RMSD"
  ) +
  theme_nmr +
  ggplot2::guides(fill = "none") +
  ggplot2::theme(
    legend.position = "none",
    axis.text.x = ggplot2::element_text(size = 7.5)
  )

response_plot <- subset(
  per_response,
  label %in% names(selected_labels)
)
response_plot$workflow <- unname(selected_labels[response_plot$label])
response_plot <- rbind(
  response_plot[, c("workflow", "RMSD")],
  data.frame(
    workflow = "Deposited PLS-SVD/IRLBA (165)",
    RMSD = historical_response_rmsd
  )
)
response_plot$workflow <- factor(
  response_plot$workflow,
  levels = names(colours)[seq_len(3L)]
)
p_response <- ggplot2::ggplot(
  response_plot, ggplot2::aes(workflow, RMSD, fill = workflow)
) +
  ggplot2::geom_boxplot(
    width = 0.62, outlier.shape = NA, colour = "black", linewidth = 0.35
  ) +
  ggplot2::coord_cartesian(
    ylim = c(0, unname(stats::quantile(response_plot$RMSD, 0.99)))
  ) +
  ggplot2::scale_fill_manual(values = colours) +
  ggplot2::scale_x_discrete(labels = c(
    "Deposited PLS-SVD/IRLBA (165)" = "Reference\nPLS-SVD/IRLBA\n(165)",
    "PLS-SVD CUDA rSVD (5)" = "PLS-SVD\nCUDA rSVD\n(5)",
    "SIMPLS CUDA rSVD (50)" = "SIMPLS\nCUDA rSVD\n(50)"
  )) +
  ggplot2::labs(
    title = "C  Response-wise errors",
    subtitle = "28,355 spectral coordinates; display limited at the 99th percentile",
    x = NULL, y = "Response-wise RMSD"
  ) +
  theme_nmr +
  ggplot2::guides(fill = "none") +
  ggplot2::theme(
    legend.position = "none",
    axis.text.x = ggplot2::element_text(size = 7.5)
  )

representative_index <- selection$sample_index[[1L]]
if (!identical(dim(historical_prediction), dim(observed)) ||
    representative_index > nrow(historical_prediction)) {
  stop("Historical prediction does not match the qualified held-out set.",
       call. = FALSE)
}
curve <- subset(
  curve,
  series %in% c("Observed", names(selected_labels))
)
curve$series <- ifelse(
  curve$series %in% names(selected_labels),
  unname(selected_labels[curve$series]),
  curve$series
)
curve <- rbind(
  curve,
  data.frame(
    ppm = suppressWarnings(as.numeric(colnames(observed))),
    sample_index = representative_index,
    sample_id = rownames(observed)[representative_index],
    series = "Deposited PLS-SVD/IRLBA (165)",
    intensity = historical_prediction[representative_index, ],
    stringsAsFactors = FALSE
  )
)
curve$series <- factor(
  curve$series,
  levels = c("Observed", names(colours)[seq_len(3L)])
)

spectrum_plot <- function(data, title, subtitle = NULL) {
  ggplot2::ggplot(
    data,
    ggplot2::aes(ppm, intensity, colour = series, linewidth = series)
  ) +
    ggplot2::geom_line() +
    ggplot2::scale_x_reverse() +
    ggplot2::scale_colour_manual(values = colours) +
    ggplot2::scale_linewidth_manual(values = c(
      "Observed" = 0.55,
      "Deposited PLS-SVD/IRLBA (165)" = 0.34,
      "PLS-SVD CUDA rSVD (5)" = 0.34,
      "SIMPLS CUDA rSVD (50)" = 0.34
    )) +
    ggplot2::labs(
      title = title, subtitle = subtitle,
      x = "Chemical shift (ppm)", y = "Intensity"
    ) +
    theme_nmr +
    ggplot2::theme(
      legend.position = "bottom",
      legend.text = ggplot2::element_text(size = 7)
    )
}
p_full <- spectrum_plot(
  curve,
  "D  Full held-out spectrum",
  paste0(
    "Sample ", selection$sample_id[[1L]],
    "; selected reproducibly as nearest the median SIMPLS held-out RMSD"
  )
)
p_zoom <- spectrum_plot(
  subset(curve, ppm >= 0.5 & ppm <= 1.7),
  "E  Expanded 1.7-0.5 ppm region"
)

historical_row <- historical_summary[
  historical_summary$variant_name == "nature_fastsimpls_plssvd", ,
  drop = FALSE
]
resources <- rbind(
  data.frame(
    workflow = "Deposited PLS-SVD/IRLBA (165)",
    metric = c(
      "Total time (s)", "Host RSS increment (MB)",
      "GPU increment (MB)"
    ),
    value = c(
      historical_row$total_time_sec_median[[1L]],
      historical_row$incremental_peak_host_rss_mb_median[[1L]],
      NA_real_
    )
  ),
  do.call(rbind, lapply(seq_len(nrow(qualified)), function(i) {
    family_label <- if (qualified$family[[i]] == "plssvd") {
      "PLS-SVD"
    } else {
      "SIMPLS"
    }
    solver_label <- if (qualified$solver[[i]] == "irlba") {
      "IRLBA"
    } else {
      "rSVD"
    }
    data.frame(
      workflow = sprintf(
        "%s %s\n%s, %d",
        family_label,
        toupper(qualified$backend[[i]]),
        solver_label,
        qualified$ncomp[[i]]
      ),
      metric = c(
        "Total time (s)", "Host RSS increment (MB)",
        "GPU increment (MB)"
      ),
      value = c(
        qualified$total_time_sec_median[[i]],
        qualified$incremental_process_peak_rss_mb[[i]],
        qualified$gpu_incremental_mb[[i]] %||% NA_real_
      )
    )
  }))
)
resources <- resources[is.finite(resources$value), , drop = FALSE]
resources$workflow[resources$workflow == "Deposited PLS-SVD/IRLBA (165)"] <-
  "Deposited PLS-SVD\nIRLBA, 165"
resources$workflow <- factor(
  resources$workflow,
  levels = rev(unique(resources$workflow))
)
p_resources <- ggplot2::ggplot(
  resources,
  ggplot2::aes(value, workflow, colour = grepl("CUDA", workflow))
) +
  ggplot2::geom_segment(
    ggplot2::aes(x = 0, xend = value, yend = workflow),
    colour = "#B7B7B7", linewidth = 0.45
  ) +
  ggplot2::geom_point(size = 2.2) +
  ggplot2::facet_wrap(~metric, scales = "free_x", nrow = 1L) +
  ggplot2::scale_colour_manual(values = c(
    "FALSE" = "#8C8C8C", "TRUE" = "#56B4E9"
  )) +
  ggplot2::labs(
    title = "F  Computational resources",
    subtitle = "Three fits; baseline-corrected increments; GPU includes context",
    x = NULL, y = NULL
  ) +
  theme_nmr +
  ggplot2::guides(colour = "none") +
  ggplot2::theme(
    legend.position = "none",
    axis.text.y = ggplot2::element_text(size = 6.8),
    strip.text = ggplot2::element_text(size = 6.8)
  )

figure <- (
  p_metric | p_sample
) / (
  p_response | p_resources
) / (
  p_full | p_zoom
) +
  patchwork::plot_layout(guides = "collect", heights = c(1, 1, 1.05)) +
  patchwork::plot_annotation(
    title = "NMR spectral prediction: predictive performance, reconstruction, and computational cost",
    theme = ggplot2::theme(
      plot.title = ggplot2::element_text(face = "bold", size = 14)
    )
  ) &
  ggplot2::theme(
    plot.margin = ggplot2::margin(4, 4, 4, 4),
    legend.position = "bottom"
  )

ggplot2::ggsave(
  file.path(output_dir, "nmr_qualified_main_figure.png"),
  figure, width = 10.2, height = 13.0, dpi = 360, bg = "white"
)
ggplot2::ggsave(
  file.path(output_dir, "nmr_qualified_main_figure.pdf"),
  figure, width = 10.2, height = 13.0, bg = "white"
)

write.csv(metric_long, file.path(
  output_dir, "nmr_family_selected_and_historical_metrics.csv"
), row.names = FALSE)
write.csv(resources, file.path(
  output_dir, "nmr_qualified_and_historical_resources.csv"
), row.names = FALSE)
write.csv(agreement, file.path(
  output_dir, "nmr_qualified_solver_backend_agreement.csv"
), row.names = FALSE)
write.csv(curve, file.path(
  output_dir, "nmr_qualified_representative_spectrum_with_reference.csv"
), row.names = FALSE)

cat(normalizePath(
  file.path(output_dir, "nmr_qualified_main_figure.png"),
  winslash = "/", mustWork = TRUE
), "\n")
