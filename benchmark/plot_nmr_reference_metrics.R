#!/usr/bin/env Rscript

# Plot the fixed-100-component NMR predictive and computational metrics with
# the deposited Nature Communications PLS-SVD/IRLBA workflow shown explicitly.

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1L) args[[1L]] else "."
out_dir <- if (length(args) >= 2L) args[[2L]] else
  file.path(root, "benchmark_results", "nmr_reference_metrics_20260726")

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop("ggplot2 is required.", call. = FALSE)
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

reference_file <- file.path(
  root,
  "benchmark_results",
  "review_nmr_reference_20260725",
  "nmr_reference_comparison_summary.csv"
)
simpls_file <- file.path(
  root,
  "benchmark_results",
  "review_nmr_replicates_100",
  "nmr_replicates_summary.csv"
)
method_file <- file.path(
  root,
  "benchmark_results",
  "review_nmr_20260724",
  "plots",
  "nmr_final_method_summary.csv"
)
stopifnot(
  file.exists(reference_file),
  file.exists(simpls_file),
  file.exists(method_file)
)

reference <- utils::read.csv(reference_file, check.names = FALSE)
simpls <- utils::read.csv(simpls_file, check.names = FALSE)
method_summary <- utils::read.csv(method_file, check.names = FALSE)

method_order <- c(
  "Deposited reference\nPLS-SVD/IRLBA CPU",
  "fastPLS\nPLS-SVD/rSVD CPU",
  "fastPLS\nPLS-SVD/rSVD CUDA",
  "fastPLS\nSIMPLS/rSVD CPU",
  "fastPLS\nSIMPLS/rSVD CUDA"
)
method_colors <- c(
  "Deposited reference\nPLS-SVD/IRLBA CPU" = "#6B6B6B",
  "fastPLS\nPLS-SVD/rSVD CPU" = "#4E79A7",
  "fastPLS\nPLS-SVD/rSVD CUDA" = "#E15759",
  "fastPLS\nSIMPLS/rSVD CPU" = "#76B7B2",
  "fastPLS\nSIMPLS/rSVD CUDA" = "#F28E2B"
)

reference_id <- c(
  "deposited_fastsimpls_irlba",
  "fastpls_plssvd_cpu_rsvd",
  "fastpls_plssvd_cuda_rsvd"
)
reference_rows <- reference[
  match(reference_id, reference$variant),
]
agreement <- method_summary$prediction_agreement[
  match(c("SIMPLS CPU", "SIMPLS CUDA"), method_summary$method)
]

summary <- data.frame(
  method = method_order,
  family = c("PLS-SVD", "PLS-SVD", "PLS-SVD", "SIMPLS", "SIMPLS"),
  solver = c("IRLBA", "rSVD", "rSVD", "rSVD", "rSVD"),
  backend = c("CPU", "CPU", "CUDA", "CPU", "CUDA"),
  ncomp = 100L,
  repetitions = c(
    reference_rows$n_repetitions,
    simpls$n_repetitions[match(c("CPU", "CUDA"), simpls$backend)]
  ),
  RMSD = c(
    reference_rows$RMSD_median,
    simpls$RMSD_median[match(c("CPU", "CUDA"), simpls$backend)]
  ),
  RMSD_iqr = c(
    reference_rows$RMSD_iqr,
    simpls$RMSD_iqr[match(c("CPU", "CUDA"), simpls$backend)]
  ),
  Q2 = c(
    reference_rows$Q2_median,
    simpls$Q2_median[match(c("CPU", "CUDA"), simpls$backend)]
  ),
  Q2_iqr = c(
    rep(0, nrow(reference_rows)),
    simpls$Q2_iqr[match(c("CPU", "CUDA"), simpls$backend)]
  ),
  prediction_correlation = c(
    reference_rows$prediction_correlation_vs_reference,
    agreement
  ),
  total_time_sec = c(
    reference_rows$total_time_sec_median,
    simpls$total_time_sec_median[match(c("CPU", "CUDA"), simpls$backend)]
  ),
  total_time_sec_iqr = c(
    reference_rows$total_time_sec_iqr,
    simpls$total_time_sec_iqr[match(c("CPU", "CUDA"), simpls$backend)]
  ),
  host_rss_mb = c(
    reference_rows$host_rss_mb_median,
    simpls$host_rss_mb_median[match(c("CPU", "CUDA"), simpls$backend)]
  ),
  host_rss_mb_iqr = c(
    reference_rows$host_rss_mb_iqr,
    simpls$host_rss_mb_iqr[match(c("CPU", "CUDA"), simpls$backend)]
  ),
  gpu_peak_mb = c(
    reference_rows$gpu_peak_mb_median,
    simpls$gpu_peak_mb_median[match(c("CPU", "CUDA"), simpls$backend)]
  ),
  gpu_peak_mb_iqr = c(
    reference_rows$gpu_peak_mb_iqr,
    simpls$gpu_peak_mb_iqr[match(c("CPU", "CUDA"), simpls$backend)]
  ),
  stringsAsFactors = FALSE
)
summary$method <- factor(summary$method, levels = rev(method_order))
utils::write.csv(
  summary,
  file.path(out_dir, "nmr_reference_metric_comparison.csv"),
  row.names = FALSE
)

predictive <- rbind(
  data.frame(
    method = summary$method,
    metric = "RMSD (10^-3)\n(lower is better)",
    value = summary$RMSD * 1000,
    lower = (summary$RMSD - summary$RMSD_iqr / 2) * 1000,
    upper = (summary$RMSD + summary$RMSD_iqr / 2) * 1000
  ),
  data.frame(
    method = summary$method,
    metric = "Q2\n(higher is better)",
    value = summary$Q2,
    lower = summary$Q2 - summary$Q2_iqr / 2,
    upper = summary$Q2 + summary$Q2_iqr / 2
  ),
  data.frame(
    method = summary$method,
    metric = "Prediction correlation\nwith deposited reference",
    value = summary$prediction_correlation,
    lower = summary$prediction_correlation,
    upper = summary$prediction_correlation
  )
)
predictive$metric <- factor(
  predictive$metric,
  levels = c(
    "RMSD (10^-3)\n(lower is better)",
    "Q2\n(higher is better)",
    "Prediction correlation\nwith deposited reference"
  )
)

resources <- rbind(
  data.frame(
    method = summary$method,
    metric = "Total time (s)",
    value = summary$total_time_sec,
    lower = summary$total_time_sec - summary$total_time_sec_iqr / 2,
    upper = summary$total_time_sec + summary$total_time_sec_iqr / 2
  ),
  data.frame(
    method = summary$method,
    metric = "Peak host RSS (MB)",
    value = summary$host_rss_mb,
    lower = summary$host_rss_mb - summary$host_rss_mb_iqr / 2,
    upper = summary$host_rss_mb + summary$host_rss_mb_iqr / 2
  ),
  data.frame(
    method = summary$method,
    metric = "Peak GPU memory (MB)",
    value = summary$gpu_peak_mb,
    lower = summary$gpu_peak_mb - summary$gpu_peak_mb_iqr / 2,
    upper = summary$gpu_peak_mb + summary$gpu_peak_mb_iqr / 2
  )
)
resources$metric <- factor(
  resources$metric,
  levels = c("Total time (s)", "Peak host RSS (MB)", "Peak GPU memory (MB)")
)

theme_nmr <- ggplot2::theme_bw(base_size = 9.5) +
  ggplot2::theme(
    panel.grid.minor = ggplot2::element_blank(),
    panel.grid.major.y = ggplot2::element_line(
      colour = "#E7E7E7",
      linewidth = 0.3
    ),
    strip.text = ggplot2::element_text(face = "bold", size = 7.5),
    axis.title = ggplot2::element_text(face = "bold", size = 8.5),
    axis.text = ggplot2::element_text(size = 7),
    legend.position = "none",
    plot.title = ggplot2::element_text(face = "bold", size = 11),
    plot.margin = ggplot2::margin(5, 6, 4, 5)
  )

p_predictive <- ggplot2::ggplot(
  predictive,
  ggplot2::aes(
    x = value,
    y = method,
    colour = method
  )
) +
  ggplot2::geom_errorbar(
    ggplot2::aes(xmin = lower, xmax = upper),
    orientation = "y",
    width = 0.22,
    linewidth = 0.45,
    na.rm = TRUE
  ) +
  ggplot2::geom_point(size = 2.6, na.rm = TRUE) +
  ggplot2::facet_wrap(~metric, ncol = 1, scales = "free_x") +
  ggplot2::scale_colour_manual(values = method_colors, drop = FALSE) +
  ggplot2::scale_x_continuous(
    labels = scales::label_number(accuracy = 0.0001),
    expand = ggplot2::expansion(mult = c(0.08, 0.10))
  ) +
  ggplot2::labs(
    title = "Predictive metrics at 100 components",
    x = NULL,
    y = NULL
  ) +
  theme_nmr

p_resources <- ggplot2::ggplot(
  resources[is.finite(resources$value) & resources$value > 0,],
  ggplot2::aes(
    x = value,
    y = method,
    colour = method
  )
) +
  ggplot2::geom_errorbar(
    ggplot2::aes(xmin = lower, xmax = upper),
    orientation = "y",
    width = 0.22,
    linewidth = 0.45,
    na.rm = TRUE
  ) +
  ggplot2::geom_point(size = 2.6, na.rm = TRUE) +
  ggplot2::facet_wrap(~metric, ncol = 1, scales = "free_x") +
  ggplot2::scale_colour_manual(values = method_colors, drop = FALSE) +
  ggplot2::scale_x_log10(
    labels = scales::label_number(),
    expand = ggplot2::expansion(mult = c(0.10, 0.10))
  ) +
  ggplot2::labs(
    title = "Computational metrics at 100 components",
    x = "Logarithmic scale",
    y = NULL
  ) +
  theme_nmr

for (name in c("predictive", "resources")) {
  plot <- if (name == "predictive") p_predictive else p_resources
  ggplot2::ggsave(
    file.path(out_dir, paste0("nmr_reference_", name, ".png")),
    plot,
    width = 5.2,
    height = 7.2,
    dpi = 320,
    bg = "white"
  )
  ggplot2::ggsave(
    file.path(out_dir, paste0("nmr_reference_", name, ".pdf")),
    plot,
    width = 5.2,
    height = 7.2,
    bg = "white"
  )
}

cat("Wrote NMR reference metric plots to", out_dir, "\n")
