#!/usr/bin/env Rscript

# Build the matched 165-component NMR figure used in the CMPB manuscript.

options(stringsAsFactors = FALSE)

if (!requireNamespace("ggplot2", quietly = TRUE) ||
    !requireNamespace("patchwork", quietly = TRUE) ||
    !requireNamespace("scales", quietly = TRUE)) {
  stop("ggplot2, patchwork, and scales are required.", call. = FALSE)
}

args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args) >= 1L) args[[1L]] else
  "benchmark_results/manuscript_revision_cycle120_20260827/nmr_fixed165"
output_dir <- if (length(args) >= 2L) args[[2L]] else input_dir
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

summary <- read.csv(file.path(input_dir, "nmr_165_summary.csv"))
per_sample <- read.csv(file.path(input_dir, "nmr_per_sample_rmsd.csv"))
spectrum <- read.csv(file.path(
  input_dir, "nmr_fixed165_representative_spectrum.csv"
))
selection <- read.csv(file.path(
  input_dir, "nmr_fixed165_representative_selection.csv"
))

labels <- c(
  "nature_fastsimpls_plssvd" = "Deposited PLS-SVD / IRLBA",
  "cpp_plssvd_irlba" = "fastPLS PLS-SVD CPU / IRLBA",
  "cpp_plssvd_cpu_rsvd" = "fastPLS PLS-SVD CPU / rSVD",
  "gpu_plssvd_rsvd" = "fastPLS PLS-SVD CUDA / rSVD",
  "cpp_simpls_irlba" = "fastPLS SIMPLS CPU / IRLBA",
  "cpp_simpls_cpu_rsvd" = "fastPLS SIMPLS CPU / rSVD",
  "gpu_simpls_rsvd" = "fastPLS SIMPLS CUDA / rSVD"
)
levels_workflow <- unname(labels)
colours <- c(
  "Deposited PLS-SVD / IRLBA" = "#D55E00",
  "fastPLS PLS-SVD CPU / IRLBA" = "#56B4E9",
  "fastPLS PLS-SVD CPU / rSVD" = "#0072B2",
  "fastPLS PLS-SVD CUDA / rSVD" = "#004C6D",
  "fastPLS SIMPLS CPU / IRLBA" = "#8FD175",
  "fastPLS SIMPLS CPU / rSVD" = "#009E73",
  "fastPLS SIMPLS CUDA / rSVD" = "#006B4F"
)

summary <- summary[match(names(labels), summary$variant_name), ]
stopifnot(!anyNA(summary$variant_name), all(summary$failed_replicates == 0L))
summary$workflow <- factor(
  unname(labels[summary$variant_name]), levels = rev(levels_workflow)
)
per_sample$workflow <- factor(
  unname(labels[per_sample$variant_name]), levels = rev(levels_workflow)
)
stopifnot(!anyNA(per_sample$workflow))

theme_nmr <- ggplot2::theme_classic(base_size = 10) +
  ggplot2::theme(
    plot.title = ggplot2::element_text(face = "bold", size = 10),
    plot.subtitle = ggplot2::element_text(size = 7.8),
    axis.title = ggplot2::element_text(face = "bold", size = 8.5),
    axis.text = ggplot2::element_text(colour = "black", size = 7.5),
    legend.position = "bottom",
    legend.text = ggplot2::element_text(size = 7),
    plot.margin = ggplot2::margin(4, 5, 4, 4)
  )

point_panel <- function(data, value, title, subtitle, xlab, log10 = FALSE) {
  p <- ggplot2::ggplot(
    data,
    ggplot2::aes(x = .data[[value]], y = workflow, colour = workflow)
  ) +
    ggplot2::geom_point(size = 2.5) +
    ggplot2::scale_colour_manual(values = colours, drop = FALSE) +
    ggplot2::labs(title = title, subtitle = subtitle, x = xlab, y = NULL) +
    theme_nmr +
    ggplot2::theme(legend.position = "none")
  if (log10) {
    p <- p + ggplot2::scale_x_log10(labels = scales::label_number())
  }
  p
}

p_rmsd <- point_panel(
  summary, "global_rmsd", "A  Held-out prediction",
  "Same split, preprocessing, precision, and 165 components",
  "Global RMSD"
) + ggplot2::scale_x_continuous(labels = scales::label_scientific(digits = 3))

p_time <- point_panel(
  summary, "total_time_sec_median", "B  Fitting plus prediction time",
  "Median of three isolated runs", "Total time (s, log scale)", TRUE
)

p_memory <- point_panel(
  summary, "incremental_peak_host_rss_mb_median", "C  Host-memory increment",
  "Baseline-corrected process peak", "Incremental host RSS (MiB, log scale)", TRUE
)

p_distribution <- ggplot2::ggplot(
  per_sample,
  ggplot2::aes(x = rmsd, y = workflow, fill = workflow)
) +
  ggplot2::geom_boxplot(
    width = 0.58, outlier.size = 0.55, linewidth = 0.35
  ) +
  ggplot2::scale_fill_manual(values = colours, drop = FALSE) +
  ggplot2::scale_x_continuous(labels = scales::label_scientific(digits = 2)) +
  ggplot2::labs(
    title = "D  Held-out per-spectrum error",
    subtitle = "All 321 spectra; deposited PLS-SVD included",
    x = "Per-spectrum RMSD", y = NULL
  ) +
  theme_nmr +
  ggplot2::theme(legend.position = "none")

spectrum_long <- rbind(
  data.frame(ppm = spectrum$ppm, intensity = spectrum$observed,
             series = "Observed"),
  data.frame(ppm = spectrum$ppm,
             intensity = spectrum$fastpls_simpls_cuda_rsvd,
             series = "fastPLS SIMPLS/CUDA"),
  data.frame(ppm = spectrum$ppm,
             intensity = spectrum$deposited_plssvd_irlba,
             series = "Deposited PLS-SVD")
)
spectrum_colours <- c(
  "Observed" = "#111111",
  "fastPLS SIMPLS/CUDA" = "#0072B2",
  "Deposited PLS-SVD" = "#D55E00"
)

spectrum_panel <- function(data, title, subtitle = NULL) {
  ggplot2::ggplot(
    data, ggplot2::aes(ppm, intensity, colour = series, linetype = series)
  ) +
    ggplot2::geom_line(linewidth = 0.42, alpha = 0.9) +
    ggplot2::scale_x_reverse() +
    ggplot2::scale_colour_manual(values = spectrum_colours) +
    ggplot2::scale_linetype_manual(values = c(
      "Observed" = "solid",
      "fastPLS SIMPLS/CUDA" = "dashed",
      "Deposited PLS-SVD" = "dotted"
    )) +
    ggplot2::labs(
      title = title, subtitle = subtitle,
      x = "Chemical shift (ppm)", y = "Intensity"
    ) +
    theme_nmr +
    ggplot2::theme(legend.position = "none")
}

sample_label <- as.character(selection$sample_id[[1L]])
p_full <- spectrum_panel(
  spectrum_long,
  "E  Representative held-out spectrum",
  paste0(
    sample_label,
    "; black observed, blue dashed SIMPLS/CUDA, orange dotted deposited PLS-SVD"
  )
)
p_zoom <- spectrum_panel(
  subset(spectrum_long, ppm >= 0.5 & ppm <= 1.7),
  "F  Expanded 1.7-0.5 ppm region"
)

figure <- (
  p_rmsd + p_time + p_memory +
    patchwork::plot_layout(widths = c(1.25, 1, 1))
) / (
  p_distribution + p_full + p_zoom +
    patchwork::plot_layout(widths = c(1.25, 1, 1))
) +
  patchwork::plot_annotation(
    title = "NMR multivariate prediction and computation at 165 components",
    subtitle = paste(
      "float64; 1,200 training and 321 held-out spectra; all workflows use",
      "the same centering-only protocol and prediction target"
    ),
    theme = ggplot2::theme(
      plot.title = ggplot2::element_text(face = "bold", size = 14),
      plot.subtitle = ggplot2::element_text(size = 9)
    )
  )

ggplot2::ggsave(
  file.path(output_dir, "Figure_4_nmr_fixed165.png"), figure,
  width = 12.2, height = 8.1, dpi = 360, bg = "white"
)
ggplot2::ggsave(
  file.path(output_dir, "Figure_4_nmr_fixed165.pdf"), figure,
  width = 12.2, height = 8.1, device = grDevices::cairo_pdf
)

writeLines(
  c(
    "Figure 4 fixed-165 inputs",
    "ncomp=165 for every workflow",
    "precision=float64",
    "successful replicates=3 per workflow",
    paste0("representative sample=", sample_label),
    "rSVD controls: archived 0.99.6 automatic backend-specific controls; seed=123",
    "preprocessing: original centering-only NMR protocol"
  ),
  file.path(output_dir, "Figure_4_nmr_fixed165_manifest.txt")
)
