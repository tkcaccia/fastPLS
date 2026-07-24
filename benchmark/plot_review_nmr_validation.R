#!/usr/bin/env Rscript

# Publication figures for the NMR review-validation output. The representative
# spectrum is selected by median held-out RMSD, not by visual inspection.

args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args) >= 1L) args[[1L]] else "benchmark_results/review_nmr_20260724"
output_dir <- if (length(args) >= 2L) args[[2L]] else file.path(input_dir, "plots")
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

rmsd_data <- rbind(data.frame(backend = "CPU", RMSD = cpu$per_sample$RMSD), data.frame(backend = "CUDA", RMSD = cuda$per_sample$RMSD))
p_rmsd <- ggplot2::ggplot(rmsd_data, ggplot2::aes(backend, RMSD, fill = backend)) +
  ggplot2::geom_boxplot(width = 0.58, outlier.alpha = 0.35) +
  ggplot2::scale_fill_manual(values = c(CPU = "#4E79A7", CUDA = "#E15759")) +
  ggplot2::labs(title = "Held-out NMR prediction error", x = NULL, y = "Per-spectrum RMSD") + base_theme + ggplot2::theme(legend.position = "none")
ggplot2::ggsave(file.path(output_dir, "nmr_per_spectrum_rmsd.png"), p_rmsd, width = 7, height = 5, dpi = 320)
ggplot2::ggsave(file.path(output_dir, "nmr_per_spectrum_rmsd.pdf"), p_rmsd, width = 7, height = 5)

rss_mb <- function(log_file) {
  line <- grep("Maximum resident set size", readLines(log_file, warn = FALSE), value = TRUE)
  if (!length(line)) return(NA_real_)
  as.numeric(sub(".*: ", "", line[[1L]])) / 1024
}
gpu_peak <- as.numeric(readLines(file.path(input_dir, "nmr_final_cuda_gpu_mem_peak_mb.txt"), warn = FALSE)[[1L]])
summary <- rbind(
  data.frame(backend = "CPU", total_time_sec = cpu$summary$total_time_sec, host_rss_mb = rss_mb(file.path(input_dir, "nmr_final_cpu.log")), gpu_peak_mb = 0),
  data.frame(backend = "CUDA", total_time_sec = cuda$summary$total_time_sec, host_rss_mb = rss_mb(file.path(input_dir, "nmr_final_cuda.log")), gpu_peak_mb = gpu_peak)
)
summary_long <- rbind(
  data.frame(backend = summary$backend, metric = "Total time (s)", value = summary$total_time_sec),
  data.frame(backend = summary$backend, metric = "Peak host RSS (MB)", value = summary$host_rss_mb),
  data.frame(backend = "CUDA", metric = "Peak GPU memory (MB)", value = gpu_peak)
)
p_resources <- ggplot2::ggplot(summary_long, ggplot2::aes(backend, value, fill = backend)) +
  ggplot2::geom_col(width = 0.62) + ggplot2::facet_wrap(~metric, scales = "free_y", nrow = 1) +
  ggplot2::scale_fill_manual(values = c(CPU = "#4E79A7", CUDA = "#E15759")) +
  ggplot2::labs(title = "NMR computational resources at the selected component count", x = NULL, y = NULL) + base_theme + ggplot2::theme(legend.position = "none")
ggplot2::ggsave(file.path(output_dir, "nmr_speed_memory.png"), p_resources, width = 12, height = 4.8, dpi = 320)
ggplot2::ggsave(file.path(output_dir, "nmr_speed_memory.pdf"), p_resources, width = 12, height = 4.8)

utils::write.csv(selection, file.path(output_dir, "nmr_inner_component_selection.csv"), row.names = FALSE)
utils::write.csv(summary, file.path(output_dir, "nmr_final_cpu_cuda_summary.csv"), row.names = FALSE)
utils::write.csv(data.frame(selected_test_sample = sample_index, selection_rule = "per-spectrum RMSD closest to held-out median", selected_sample_rmsd = sample_rmsd[[sample_index]], selected_sample_correlation = cpu$per_sample$correlation[[sample_index]]), file.path(output_dir, "nmr_figure_metadata.csv"), row.names = FALSE)
