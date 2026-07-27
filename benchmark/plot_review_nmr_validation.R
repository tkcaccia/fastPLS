#!/usr/bin/env Rscript

# Publication figures for the NMR review-validation output. The representative
# spectrum is selected by median held-out RMSD, not by visual inspection.

args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args) >= 1L) args[[1L]] else "benchmark_results/review_nmr_20260724"
output_dir <- if (length(args) >= 2L) args[[2L]] else file.path(input_dir, "plots")
reference_dir <- if (length(args) >= 3L) args[[3L]] else NULL
simpls_replicate_summary <- if (length(args) >= 4L) args[[4L]] else
  "benchmark_results/review_nmr_replicates_100/nmr_replicates_summary.csv"
if (!requireNamespace("ggplot2", quietly = TRUE)) stop("ggplot2 is required.", call. = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cpu <- readRDS(file.path(input_dir, "nmr_final_cpu.rds"))
cuda <- readRDS(file.path(input_dir, "nmr_final_cuda.rds"))
selection <- utils::read.csv(file.path(input_dir, "nmr_inner_component_selection.csv"))

axis <- suppressWarnings(as.numeric(colnames(cpu$observed)))
if (length(axis) != ncol(cpu$observed) || any(!is.finite(axis))) axis <- seq_len(ncol(cpu$observed))
sample_rmsd <- cpu$per_sample$RMSD
sample_index <- which.min(abs(sample_rmsd - stats::median(sample_rmsd)))
spectrum_data <- rbind(
  data.frame(chemical_shift_ppm = axis, intensity = cpu$observed[sample_index, ], series = "Observed"),
  data.frame(chemical_shift_ppm = axis, intensity = cuda$predicted[sample_index, ], series = "SIMPLS-rSVD prediction")
)
base_theme <- ggplot2::theme_classic(base_size = 14) +
  ggplot2::theme(
    plot.title = ggplot2::element_text(face = "bold"),
    plot.subtitle = ggplot2::element_text(size = 11),
    legend.position = "bottom", legend.title = ggplot2::element_blank(),
    axis.title = ggplot2::element_text(face = "bold")
  )

overlay <- function(title, limits = NULL) {
  p <- ggplot2::ggplot(spectrum_data, ggplot2::aes(chemical_shift_ppm, intensity, colour = series)) +
    ggplot2::geom_line(linewidth = 0.35, alpha = 0.9) +
    ggplot2::scale_colour_manual(values = c("Observed" = "#111111", "SIMPLS-rSVD prediction" = "#D55E00")) +
    ggplot2::scale_x_reverse() +
    ggplot2::labs(
      title = title,
      subtitle = sprintf("Representative held-out spectrum (median RMSD); %d components", cuda$summary$ncomp),
      x = "Chemical shift (ppm)", y = "Intensity"
    ) + base_theme
  if (!is.null(limits)) p <- p + ggplot2::coord_cartesian(xlim = rev(limits))
  p
}

p_full <- overlay("Observed and predicted NMR spectrum")
p_zoom <- overlay("Zoomed NMR spectrum (0.5-1.7 ppm)", c(0.5, 1.7))
ggplot2::ggsave(file.path(output_dir, "nmr_spectrum_full.png"), p_full, width = 11, height = 5, dpi = 320)
ggplot2::ggsave(file.path(output_dir, "nmr_spectrum_full.pdf"), p_full, width = 11, height = 5)
ggplot2::ggsave(file.path(output_dir, "nmr_spectrum_zoom.png"), p_zoom, width = 11, height = 5, dpi = 320)
ggplot2::ggsave(file.path(output_dir, "nmr_spectrum_zoom.pdf"), p_zoom, width = 11, height = 5)

comparison_labels <- c(
  deposited_fastsimpls_irlba = "Deposited reference",
  fastpls_plssvd_cpu_rsvd = "PLS-SVD CPU",
  fastpls_plssvd_cuda_rsvd = "PLS-SVD CUDA",
  fastpls_simpls_cpu_rsvd = "SIMPLS CPU",
  fastpls_simpls_cuda_rsvd = "SIMPLS CUDA"
)
comparison_colours <- c(
  "Deposited reference" = "#7F7F7F",
  "PLS-SVD CPU" = "#4E79A7",
  "PLS-SVD CUDA" = "#E15759",
  "SIMPLS CPU" = "#76B7B2",
  "SIMPLS CUDA" = "#F28E2B"
)

if (!is.null(reference_dir) && dir.exists(reference_dir)) {
  prediction_path <- function(id) {
    file.path(reference_dir, "predictions", paste0(id, "__rep1.rds"))
  }
  per_sample_from_prediction <- function(path) {
    object <- readRDS(path)
    sqrt(rowMeans((object$observed - object$predicted)^2))
  }
  rmsd_data <- rbind(
    data.frame(method = comparison_labels[["deposited_fastsimpls_irlba"]],
               RMSD = per_sample_from_prediction(prediction_path("deposited_fastsimpls_irlba"))),
    data.frame(method = comparison_labels[["fastpls_plssvd_cpu_rsvd"]],
               RMSD = per_sample_from_prediction(prediction_path("fastpls_plssvd_cpu_rsvd"))),
    data.frame(method = comparison_labels[["fastpls_plssvd_cuda_rsvd"]],
               RMSD = per_sample_from_prediction(prediction_path("fastpls_plssvd_cuda_rsvd"))),
    data.frame(method = comparison_labels[["fastpls_simpls_cpu_rsvd"]],
               RMSD = cpu$per_sample$RMSD),
    data.frame(method = comparison_labels[["fastpls_simpls_cuda_rsvd"]],
               RMSD = cuda$per_sample$RMSD)
  )
} else {
  rmsd_data <- rbind(
    data.frame(method = "SIMPLS CPU", RMSD = cpu$per_sample$RMSD),
    data.frame(method = "SIMPLS CUDA", RMSD = cuda$per_sample$RMSD)
  )
}
rmsd_data$method <- factor(rmsd_data$method, levels = unname(comparison_labels))
p_rmsd <- ggplot2::ggplot(rmsd_data, ggplot2::aes(method, RMSD, fill = method)) +
  ggplot2::geom_boxplot(width = 0.58, outlier.alpha = 0.35) +
  ggplot2::scale_fill_manual(values = comparison_colours, drop = FALSE) +
  ggplot2::labs(title = "Held-out NMR prediction error", x = NULL, y = "Per-spectrum RMSD") +
  base_theme +
  ggplot2::theme(
    legend.position = "none",
    axis.text.x = ggplot2::element_text(angle = 28, hjust = 1, size = 9)
  )
ggplot2::ggsave(file.path(output_dir, "nmr_per_spectrum_rmsd.png"), p_rmsd, width = 8.5, height = 5, dpi = 320)
ggplot2::ggsave(file.path(output_dir, "nmr_per_spectrum_rmsd.pdf"), p_rmsd, width = 8.5, height = 5)

rss_mb <- function(log_file) {
  line <- grep("Maximum resident set size", readLines(log_file, warn = FALSE), value = TRUE)
  if (!length(line)) return(NA_real_)
  as.numeric(sub(".*: ", "", line[[1L]])) / 1024
}
gpu_peak <- as.numeric(readLines(file.path(input_dir, "nmr_final_cuda_gpu_mem_peak_mb.txt"), warn = FALSE)[[1L]])
summary <- rbind(
  data.frame(method = "SIMPLS CPU", total_time_sec = cpu$summary$total_time_sec,
             host_rss_mb = rss_mb(file.path(input_dir, "nmr_final_cpu.log")),
             gpu_peak_mb = NA_real_, RMSD = cpu$summary$RMSD,
             Q2 = cpu$summary$Q2, prediction_agreement = NA_real_),
  data.frame(method = "SIMPLS CUDA", total_time_sec = cuda$summary$total_time_sec,
             host_rss_mb = rss_mb(file.path(input_dir, "nmr_final_cuda.log")),
             gpu_peak_mb = gpu_peak, RMSD = cuda$summary$RMSD,
             Q2 = cuda$summary$Q2, prediction_agreement = NA_real_)
)
if (file.exists(simpls_replicate_summary)) {
  replicate_summary <- utils::read.csv(simpls_replicate_summary, check.names = FALSE)
  for (backend_name in c("cpu", "cuda")) {
    index <- match(toupper(backend_name), toupper(replicate_summary$backend))
    output_index <- match(
      if (backend_name == "cpu") "SIMPLS CPU" else "SIMPLS CUDA",
      summary$method
    )
    if (!is.na(index) && !is.na(output_index)) {
      summary$total_time_sec[[output_index]] <-
        replicate_summary$total_time_sec_median[[index]]
      summary$host_rss_mb[[output_index]] <-
        replicate_summary$host_rss_mb_median[[index]]
      summary$gpu_peak_mb[[output_index]] <-
        if (backend_name == "cpu") NA_real_ else
          replicate_summary$gpu_peak_mb_median[[index]]
      summary$RMSD[[output_index]] <-
        replicate_summary$RMSD_median[[index]]
      summary$Q2[[output_index]] <-
        replicate_summary$Q2_median[[index]]
    }
  }
}
if (!is.null(reference_dir) && dir.exists(reference_dir)) {
  reference_summary <- utils::read.csv(
    file.path(reference_dir, "nmr_reference_comparison_summary.csv"),
    check.names = FALSE
  )
  reference_summary <- reference_summary[
    reference_summary$variant %in% names(comparison_labels)[1:3], ,
    drop = FALSE
  ]
  summary <- rbind(
    data.frame(
      method = unname(comparison_labels[reference_summary$variant]),
      total_time_sec = reference_summary$total_time_sec_median,
      host_rss_mb = reference_summary$host_rss_mb_median,
      gpu_peak_mb = reference_summary$gpu_peak_mb_median,
      RMSD = reference_summary$RMSD_median,
      Q2 = reference_summary$Q2_median,
      prediction_agreement =
        reference_summary$prediction_correlation_vs_reference
    ),
    summary
  )
  reference_prediction <- readRDS(
    prediction_path("deposited_fastsimpls_irlba")
  )$predicted
  summary$prediction_agreement[summary$method == "SIMPLS CPU"] <-
    stats::cor(as.vector(reference_prediction), as.vector(cpu$predicted))
  summary$prediction_agreement[summary$method == "SIMPLS CUDA"] <-
    stats::cor(as.vector(reference_prediction), as.vector(cuda$predicted))
}
summary$method <- factor(summary$method, levels = unname(comparison_labels))
summary_long <- rbind(
  data.frame(method = summary$method, metric = "Total time (s)", value = summary$total_time_sec),
  data.frame(method = summary$method, metric = "Peak host RSS (MB)", value = summary$host_rss_mb),
  data.frame(method = summary$method, metric = "Peak GPU memory (MB)", value = summary$gpu_peak_mb)
)
summary_long <- summary_long[is.finite(summary_long$value), , drop = FALSE]
summary_long$label <- ifelse(
  summary_long$metric == "Total time (s)",
  sprintf("%.1f", summary_long$value),
  sprintf("%.0f", summary_long$value)
)
p_resources <- ggplot2::ggplot(summary_long, ggplot2::aes(method, value, fill = method)) +
  ggplot2::geom_col(width = 0.62) +
  ggplot2::geom_text(ggplot2::aes(label = label), vjust = -0.35, size = 3) +
  ggplot2::facet_wrap(~metric, scales = "free_y", nrow = 1) +
  ggplot2::scale_fill_manual(values = comparison_colours, drop = FALSE) +
  ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0, 0.08))) +
  ggplot2::labs(title = "NMR computational resources at 100 components", x = NULL, y = NULL) +
  base_theme +
  ggplot2::theme(
    legend.position = "none",
    axis.text.x = ggplot2::element_text(angle = 28, hjust = 1, size = 8)
  )
ggplot2::ggsave(file.path(output_dir, "nmr_speed_memory.png"), p_resources, width = 12, height = 4.8, dpi = 320)
ggplot2::ggsave(file.path(output_dir, "nmr_speed_memory.pdf"), p_resources, width = 12, height = 4.8)

utils::write.csv(selection, file.path(output_dir, "nmr_inner_component_selection.csv"), row.names = FALSE)
utils::write.csv(summary, file.path(output_dir, "nmr_final_method_summary.csv"), row.names = FALSE)
utils::write.csv(data.frame(selected_test_sample = sample_index, selection_rule = "per-spectrum RMSD closest to held-out median", selected_sample_rmsd = sample_rmsd[[sample_index]], selected_sample_correlation = cpu$per_sample$correlation[[sample_index]]), file.path(output_dir, "nmr_figure_metadata.csv"), row.names = FALSE)
