#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1L) normalizePath(args[[1L]], mustWork = TRUE) else "."
out_dir <- if (length(args) >= 2L) args[[2L]] else
  file.path(root, "benchmark_results", "nmr_simpls_outlier_diagnostic_20260726")

cpu_file <- file.path(root, "benchmark_results", "review_nmr_20260724",
                      "nmr_final_cpu.rds")
cuda_file <- file.path(root, "benchmark_results", "review_nmr_20260724",
                       "nmr_final_cuda.rds")
reference_file <- file.path(
  root, "benchmark_results", "review_nmr_reference_20260725", "predictions",
  "deposited_fastsimpls_irlba__rep1.rds"
)

stopifnot(file.exists(cpu_file), file.exists(cuda_file), file.exists(reference_file))
if (!requireNamespace("ggplot2", quietly = TRUE) ||
    !requireNamespace("patchwork", quietly = TRUE)) {
  stop("ggplot2 and patchwork are required.", call. = FALSE)
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
cpu <- readRDS(cpu_file)
cuda <- readRDS(cuda_file)
reference <- readRDS(reference_file)

sample_rmsd <- sqrt(rowMeans((cpu$observed - cpu$predicted)^2))
sample_index <- which.max(sample_rmsd)
sample_id <- rownames(cpu$observed)[sample_index]
ppm <- suppressWarnings(as.numeric(colnames(cpu$observed)))
if (any(!is.finite(ppm))) stop("The response column names are not numeric ppm values.")

observed <- cpu$observed[sample_index, ]
predictions <- list(
  "Deposited PLS-SVD/IRLBA" = reference$predicted[sample_index, ],
  "fastPLS SIMPLS/rSVD CPU" = cpu$predicted[sample_index, ],
  "fastPLS SIMPLS/rSVD CUDA" = cuda$predicted[sample_index, ]
)
colours <- c(
  "Observed" = "#111111",
  "Deposited PLS-SVD/IRLBA" = "#009E73",
  "fastPLS SIMPLS/rSVD CPU" = "#D55E00",
  "fastPLS SIMPLS/rSVD CUDA" = "#0072B2"
)
line_types <- c(
  "Observed" = "solid",
  "Deposited PLS-SVD/IRLBA" = "dashed",
  "fastPLS SIMPLS/rSVD CPU" = "dotdash",
  "fastPLS SIMPLS/rSVD CUDA" = "longdash"
)

long <- rbind(
  data.frame(ppm = ppm, intensity = observed, series = "Observed"),
  do.call(rbind, lapply(names(predictions), function(name) {
    data.frame(ppm = ppm, intensity = predictions[[name]], series = name)
  }))
)
long$series <- factor(long$series, levels = names(colours))

metric_rows <- do.call(rbind, lapply(names(predictions), function(name) {
  predicted <- predictions[[name]]
  residual <- predicted - observed
  keep <- !(ppm > 4.6 & ppm < 4.8)
  fit <- stats::lm(predicted ~ observed)
  data.frame(
    sample_id = sample_id,
    method = name,
    RMSD_all = sqrt(mean(residual^2)),
    correlation_all = stats::cor(observed, predicted),
    RMSD_without_4.6_4.8_ppm = sqrt(mean(residual[keep]^2)),
    correlation_without_4.6_4.8_ppm =
      stats::cor(observed[keep], predicted[keep]),
    squared_error_share_4.6_4.8_ppm =
      sum(residual[!keep]^2) / sum(residual^2),
    mean_residual = mean(residual),
    calibration_intercept = unname(stats::coef(fit)[[1L]]),
    calibration_slope = unname(stats::coef(fit)[[2L]]),
    stringsAsFactors = FALSE
  )
}))

cpu_residual <- predictions[["fastPLS SIMPLS/rSVD CPU"]] - observed
breaks <- seq(floor(min(ppm) * 10) / 10, ceiling(max(ppm) * 10) / 10, by = 0.1)
bin <- cut(ppm, breaks = breaks, include.lowest = TRUE, right = FALSE)
region_error <- stats::aggregate(
  cpu_residual^2, list(ppm_region = bin), sum, na.rm = TRUE
)
names(region_error)[[2L]] <- "squared_error"
region_error$error_share <- region_error$squared_error /
  sum(region_error$squared_error)
region_error <- region_error[order(region_error$error_share, decreasing = TRUE), ]

pointwise <- data.frame(
  sample_id = sample_id,
  ppm = ppm,
  observed = observed,
  reference_prediction = predictions[["Deposited PLS-SVD/IRLBA"]],
  simpls_cpu_prediction = predictions[["fastPLS SIMPLS/rSVD CPU"]],
  simpls_cuda_prediction = predictions[["fastPLS SIMPLS/rSVD CUDA"]]
)
pointwise$simpls_cpu_residual <- pointwise$simpls_cpu_prediction - observed
pointwise$simpls_cuda_residual <- pointwise$simpls_cuda_prediction - observed

theme_spectrum <- ggplot2::theme_classic(base_size = 12) +
  ggplot2::theme(
    plot.title = ggplot2::element_text(face = "bold"),
    legend.position = "bottom",
    legend.title = ggplot2::element_blank(),
    axis.title = ggplot2::element_text(face = "bold")
  )

overlay_plot <- function(data, title, limits = NULL) {
  plot <- ggplot2::ggplot(
    data,
    ggplot2::aes(ppm, intensity, colour = series, linetype = series)
  ) +
    ggplot2::geom_line(alpha = 0.9, linewidth = 0.48) +
    ggplot2::scale_colour_manual(values = colours, drop = FALSE) +
    ggplot2::scale_linetype_manual(values = line_types, drop = FALSE) +
    ggplot2::scale_x_reverse(limits = limits) +
    ggplot2::labs(title = title, x = "Chemical shift (ppm)", y = "Intensity") +
    theme_spectrum
  plot
}

p_full <- overlay_plot(
  long,
  sprintf("Largest held-out SIMPLS error: %s", sample_id),
  c(12, -1)
)
p_water <- overlay_plot(
  long[long$ppm >= 4.55 & long$ppm <= 4.85, ],
  "Residual-water region",
  c(4.85, 4.55)
)
p_nonwater <- overlay_plot(
  long[long$ppm >= 0.5 & long$ppm <= 4.5, ],
  "Non-water metabolite region",
  c(4.5, 0.5)
)

top_regions <- head(region_error, 12L)
top_regions$ppm_region <- factor(
  top_regions$ppm_region,
  levels = rev(top_regions$ppm_region)
)
p_error <- ggplot2::ggplot(
  top_regions,
  ggplot2::aes(ppm_region, 100 * error_share)
) +
  ggplot2::geom_col(fill = "#D55E00", width = 0.72) +
  ggplot2::coord_flip() +
  ggplot2::labs(
    title = "CPU SIMPLS squared-error concentration",
    x = "Chemical-shift interval (ppm)",
    y = "Share of total squared error (%)"
  ) +
  theme_spectrum +
  ggplot2::theme(legend.position = "none")

combined <- (p_full / p_water) / (p_nonwater | p_error) +
  patchwork::plot_layout(heights = c(1.05, 1, 1.15), guides = "collect") &
  ggplot2::theme(legend.position = "bottom")

ggplot2::ggsave(
  file.path(out_dir, "nmr_simpls_outlier_diagnostic.png"),
  combined, width = 13, height = 12, dpi = 320
)
ggplot2::ggsave(
  file.path(out_dir, "nmr_simpls_outlier_diagnostic.pdf"),
  combined, width = 13, height = 12, device = grDevices::cairo_pdf
)
utils::write.csv(
  metric_rows, file.path(out_dir, "nmr_simpls_outlier_metrics.csv"),
  row.names = FALSE
)
utils::write.csv(
  region_error, file.path(out_dir, "nmr_simpls_outlier_region_error.csv"),
  row.names = FALSE
)
utils::write.csv(
  pointwise, file.path(out_dir, "nmr_simpls_outlier_spectrum.csv"),
  row.names = FALSE
)

print(metric_rows)
cat("Diagnostic files written to:", normalizePath(out_dir), "\n")
