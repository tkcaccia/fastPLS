#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args)) args[[1L]] else
  file.path("benchmark_results", "frozen_release_0.99.25")
derived <- file.path(root, "nmr", "derived")
out <- file.path(root, "figures")
dir.create(out, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

theme_pub <- function() {
  theme_classic(base_size = 9.5, base_family = "sans") +
    theme(
      legend.position = "none",
      plot.title = element_text(face = "bold", size = 10.5),
      plot.subtitle = element_text(size = 8.2, color = "#444444"),
      axis.text = element_text(size = 8),
      strip.background = element_rect(fill = "#F2F2F2", color = NA),
      strip.text = element_text(face = "bold", size = 8)
    )
}

route_summary <- read.csv(
  file.path(derived, "nmr_frozen_route_summary.csv"), check.names = FALSE
)
route_summary <- subset(
  route_summary,
  (family == "plssvd" & ncomp == 5) | (family == "simpls" & ncomp == 50)
)
route_summary$label <- with(route_summary, paste0(
  ifelse(family == "plssvd", "PLS-SVD", "SIMPLS"), " ",
  toupper(backend), " / ", ifelse(solver == "irlba", "IRLBA", "rSVD"),
  " (", ncomp, ")"
))
route_summary$group <- "fastPLS 0.99.25"
route_summary$RMSD <- route_summary$RMSD_median
route_summary$total_time <- route_summary$total_time_sec_median
route_summary$host_increment <- pmax(
  route_summary$after_fit_rss_mb_median - route_summary$baseline_rss_mb_median,
  0.01
)

historical_path <- file.path(
  "benchmark_results", "manuscript_revision_cycle64_20260726",
  "nmr_historical_reference_165_summary.csv"
)
historical <- read.csv(historical_path, check.names = FALSE)
historical <- historical[historical$variant_name == "nature_fastsimpls_plssvd", ]
historical_row <- data.frame(
  label = "Deposited PLS-SVD / IRLBA (165)",
  group = "Historical deposited workflow",
  RMSD = historical$global_rmsd,
  total_time = historical$total_time_sec_median,
  host_increment = historical$incremental_peak_host_rss_mb_median
)
comparison <- rbind(
  route_summary[c("label", "group", "RMSD", "total_time", "host_increment")],
  historical_row
)
order <- c(
  "Deposited PLS-SVD / IRLBA (165)",
  "PLS-SVD CPU / IRLBA (5)", "PLS-SVD CPU / rSVD (5)",
  "PLS-SVD CUDA / rSVD (5)", "SIMPLS CPU / IRLBA (50)",
  "SIMPLS CPU / rSVD (50)", "SIMPLS CUDA / rSVD (50)"
)
comparison$label <- factor(comparison$label, levels = rev(order))
colors <- c("fastPLS 0.99.25" = "#0072B2",
            "Historical deposited workflow" = "#D55E00")
shapes <- c("fastPLS 0.99.25" = 16,
            "Historical deposited workflow" = 17)

p_rmsd <- ggplot(comparison, aes(RMSD, label, color = group, shape = group)) +
  geom_point(size = 2.7) + scale_color_manual(values = colors) +
  scale_shape_manual(values = shapes) +
  labs(title = "A  Family-selected prediction",
       subtitle = "Historical workflow shown as context, not a matched implementation",
       x = "Held-out RMSD", y = NULL) + theme_pub() +
  theme(legend.position = "top", legend.title = element_blank())

p_time <- ggplot(comparison, aes(total_time, label, color = group, shape = group)) +
  geom_point(size = 2.5) + scale_x_log10() + scale_color_manual(values = colors) +
  scale_shape_manual(values = shapes) +
  labs(title = "B  Fitting plus prediction time",
       subtitle = "Median of three runs",
       x = "Total time (s, log scale)", y = NULL) + theme_pub()

p_mem <- ggplot(comparison, aes(host_increment, label, color = group, shape = group)) +
  geom_point(size = 2.5) + scale_x_log10() + scale_color_manual(values = colors) +
  scale_shape_manual(values = shapes) +
  labs(title = "C  Host-memory increment",
       subtitle = "fastPLS post-fit; historical peak",
       x = "Host RSS increment (MiB, log scale)", y = NULL) + theme_pub()

per_sample <- read.csv(file.path(derived, "nmr_frozen_per_sample.csv"),
                       check.names = FALSE)
per_sample$route <- factor(per_sample$route, levels = rev(c(
  "PLS-SVD CPU / IRLBA (5)", "PLS-SVD CPU / rSVD (5)",
  "PLS-SVD CUDA / rSVD (5)", "SIMPLS CPU / IRLBA (50)",
  "SIMPLS CPU / rSVD (50)", "SIMPLS CUDA / rSVD (50)"
)))
p_sample <- ggplot(per_sample, aes(RMSD, route)) +
  geom_boxplot(fill = "#9ECAE1", color = "black", outlier.alpha = 0.35,
               linewidth = 0.35) +
  labs(title = "D  Held-out per-spectrum error",
       subtitle = "All 321 held-out spectra",
       x = "Per-spectrum RMSD", y = NULL) + theme_pub()

spectrum <- read.csv(
  file.path(derived, "nmr_frozen_representative_spectrum.csv"),
  check.names = FALSE
)
spectrum <- spectrum[is.finite(spectrum$ppm), ]
spectrum_long <- rbind(
  data.frame(ppm = spectrum$ppm, intensity = spectrum$observed,
             series = "Observed"),
  data.frame(ppm = spectrum$ppm, intensity = spectrum$predicted,
             series = "Predicted")
)
spectrum_colors <- c(Observed = "#111111", Predicted = "#0072B2")
p_full <- ggplot(spectrum_long, aes(ppm, intensity, color = series)) +
  geom_line(aes(linetype = series), linewidth = 0.38, alpha = 0.9) +
  scale_x_reverse() + scale_color_manual(values = spectrum_colors) +
  labs(title = "E  Representative held-out spectrum",
       subtitle = "Closest sample to median SIMPLS CUDA/rSVD per-spectrum RMSD",
       x = "Chemical shift (ppm)", y = "Intensity") + theme_pub() +
  theme(legend.position = "top", legend.title = element_blank())

p_zoom <- ggplot(subset(spectrum_long, ppm >= 0.5 & ppm <= 1.7),
                 aes(ppm, intensity, color = series)) +
  geom_line(aes(linetype = series), linewidth = 0.48, alpha = 0.9) +
  scale_x_reverse() + scale_color_manual(values = spectrum_colors) +
  labs(title = "F  Expanded 1.7-0.5 ppm region",
       x = "Chemical shift (ppm)", y = "Intensity") + theme_pub()

figure <- (p_rmsd | p_time | p_mem) / (p_sample | p_full | p_zoom) +
  plot_layout(widths = c(1.12, 1, 1)) +
  plot_annotation(
    title = "NMR multivariate prediction and computation",
    subtitle = "fastPLS 0.99.25 archived-release analysis; float64; 1,200 training and 321 held-out spectra"
  )
ggsave(file.path(out, "Figure_4_frozen_nmr.png"), figure,
       width = 12.2, height = 7.8, dpi = 400, bg = "white")
ggsave(file.path(out, "Figure_4_frozen_nmr.pdf"), figure,
       width = 12.2, height = 7.8, device = cairo_pdf)

cat("Wrote frozen NMR figure to", normalizePath(out), "\n")
