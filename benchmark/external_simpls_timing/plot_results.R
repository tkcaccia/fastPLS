#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: plot_results.R RESULTS_DIR")
out <- normalizePath(args[[1L]], winslash = "/", mustWork = TRUE)
s <- read.csv(file.path(out, "external_simpls_timing_summary.csv"), check.names = FALSE)

if (!requireNamespace("ggplot2", quietly = TRUE)) stop("ggplot2 is required.")
labels <- c(
  ccle = "CCLE", cifar100 = "CIFAR-100", gtex_v8 = "GTEx v8",
  metref = "MetRef", retina = "Retina", tabula = "Tabula Muris",
  tcga_brca = "TCGA-BRCA", tcga_hnsc_methylation = "TCGA-HNSC methyl.",
  tcga_pan_cancer = "TCGA Pan-Cancer"
)
s$Dataset <- factor(labels[s$dataset], levels = rev(unname(labels)))
s$Implementation <- factor(s$implementation, c("fastpls", "pls"), c("fastPLS", "pls::simpls.fit"))
s$Profile <- factor(
  s$comparison_profile,
  c("estimator_kernel", "complete_workflow"),
  c("Minimum common outputs", "Ordinary public workflows")
)
s$lo <- pmax(s$median_total_sec - s$iqr_total_sec / 2, .Machine$double.eps)
s$hi <- s$median_total_sec + s$iqr_total_sec / 2

palette <- c("fastPLS" = "#0072B2", "pls::simpls.fit" = "#D55E00")
p <- ggplot2::ggplot(
  s,
  ggplot2::aes(y = Dataset, x = median_total_sec, color = Implementation, shape = Implementation)
) +
  ggplot2::geom_errorbarh(ggplot2::aes(xmin = lo, xmax = hi), height = 0.16, position = ggplot2::position_dodge(width = 0.52)) +
  ggplot2::geom_point(size = 2.5, stroke = 0.8, position = ggplot2::position_dodge(width = 0.52)) +
  ggplot2::facet_wrap(~Profile, ncol = 2, scales = "free_x") +
  ggplot2::scale_x_log10() +
  ggplot2::scale_color_manual(values = palette) +
  ggplot2::scale_shape_manual(values = c("fastPLS" = 16, "pls::simpls.fit" = 17)) +
  ggplot2::labs(
    x = "Median fit plus prediction time (s; log scale; bars show IQR)",
    y = NULL, color = NULL, shape = NULL
  ) +
  ggplot2::theme_bw(base_size = 10) +
  ggplot2::theme(
    legend.position = "top",
    panel.grid.minor = ggplot2::element_blank(),
    strip.background = ggplot2::element_rect(fill = "#E8EEF2", color = "#333333"),
    strip.text = ggplot2::element_text(face = "bold")
  )

ggplot2::ggsave(file.path(out, "external_simpls_timing_profiles.pdf"), p, width = 8.4, height = 4.8)
ggplot2::ggsave(file.path(out, "external_simpls_timing_profiles.png"), p, width = 8.4, height = 4.8, dpi = 300)

absolute <- transform(
  s,
  Memory_measure = "Absolute process peak RSS",
  median_memory_mb = median_process_peak_rss_mb,
  iqr_memory_mb = iqr_process_peak_rss_mb
)
increment <- transform(
  s,
  Memory_measure = "Peak minus pre-fit process RSS",
  median_memory_mb = median_baseline_corrected_peak_increment_mb,
  iqr_memory_mb = iqr_baseline_corrected_peak_increment_mb
)
memory <- rbind(absolute, increment)
memory$Memory_measure <- factor(
  memory$Memory_measure,
  c("Absolute process peak RSS", "Peak minus pre-fit process RSS")
)
memory$lo_memory <- pmax(memory$median_memory_mb - memory$iqr_memory_mb / 2, .Machine$double.eps)
memory$hi_memory <- memory$median_memory_mb + memory$iqr_memory_mb / 2

pm <- ggplot2::ggplot(
  memory,
  ggplot2::aes(y = Dataset, x = median_memory_mb, color = Implementation, shape = Implementation)
) +
  ggplot2::geom_errorbarh(
    ggplot2::aes(xmin = lo_memory, xmax = hi_memory), height = 0.16,
    position = ggplot2::position_dodge(width = 0.52)
  ) +
  ggplot2::geom_point(size = 2.5, stroke = 0.8, position = ggplot2::position_dodge(width = 0.52)) +
  ggplot2::facet_grid(Memory_measure ~ Profile, scales = "free_x") +
  ggplot2::scale_x_log10() +
  ggplot2::scale_color_manual(values = palette) +
  ggplot2::scale_shape_manual(values = c("fastPLS" = 16, "pls::simpls.fit" = 17)) +
  ggplot2::labs(
    x = "Host memory (MB; log scale; bars show IQR)",
    y = NULL, color = NULL, shape = NULL
  ) +
  ggplot2::theme_bw(base_size = 9) +
  ggplot2::theme(
    legend.position = "top",
    panel.grid.minor = ggplot2::element_blank(),
    strip.background = ggplot2::element_rect(fill = "#E8EEF2", color = "#333333"),
    strip.text = ggplot2::element_text(face = "bold")
  )

ggplot2::ggsave(file.path(out, "external_simpls_memory_profiles.pdf"), pm, width = 8.4, height = 7.2)
ggplot2::ggsave(file.path(out, "external_simpls_memory_profiles.png"), pm, width = 8.4, height = 7.2, dpi = 300)
