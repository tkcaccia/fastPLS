#!/usr/bin/env Rscript

root <- file.path("benchmark_results", "frozen_release_0.99.25")
out <- file.path(root, "figures")
dir.create(out, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

d <- read.csv(file.path(root, "controlled_scaling", "controlled_scaling_summary.csv"),
              check.names = FALSE)
d <- subset(d, route %in% c("cpu_irlba_explicit", "cpu_rsvd_auto", "cuda_rsvd_auto"))
d$route_label <- factor(d$route,
  c("cpu_irlba_explicit", "cpu_rsvd_auto", "cuda_rsvd_auto"),
  c("CPU IRLBA", "CPU rSVD", "CUDA rSVD"))
d$qualified <- d$numerical_failures == 0
d$factor_label <- factor(d$factor_name,
  c("n", "p", "q", "ncomp", "prefix_count", "rank", "class_count", "crosscov_mb"),
  c("Samples", "Predictors", "Responses", "Components", "Requested prefixes",
    "Cross-covariance rank", "Classes", "Cross-covariance MiB"))

theme_pub <- theme_classic(base_size = 9) +
  theme(strip.background = element_rect(fill = "#F1F1F1", color = NA),
        strip.text = element_text(face = "bold"), legend.position = "top")

p1 <- ggplot(d, aes(factor_value, median_total_sec, color = route_label,
                    linetype = route_label, group = route_label)) +
  geom_line() + geom_point(aes(shape = qualified), size = 1.5) +
  facet_wrap(~factor_label, scales = "free_x", ncol = 4) +
  scale_y_log10() +
  scale_color_manual(values = c("#0072B2", "#009E73", "#D55E00")) +
  scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 4),
                     labels = c(`TRUE` = "Met tolerance", `FALSE` = "Outside tolerance")) +
  labs(title = "A  Controlled SIMPLS runtime scaling",
       subtitle = "Median of three isolated runs; open points are excluded from speed claims",
       x = "Swept factor", y = "Total time (s, log scale)", color = NULL, shape = NULL) + theme_pub

p2 <- ggplot(d, aes(factor_value, median_incremental_rss_mb, color = route_label,
                    linetype = route_label, shape = qualified, group = route_label)) +
  geom_line() + geom_point(size = 1.6) +
  facet_wrap(~factor_label, scales = "free_x", ncol = 4) +
  scale_y_log10() +
  scale_color_manual(values = c("#0072B2", "#009E73", "#D55E00")) +
  scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 4),
                     labels = c(`TRUE` = "Met tolerance", `FALSE` = "Outside tolerance")) +
  labs(title = "B  Baseline-corrected host-memory scaling",
       subtitle = "Complete-process increment, not isolated algorithm workspace",
       x = "Swept factor", y = "Peak RSS increment (MiB, log scale)", color = NULL, shape = NULL) + theme_pub

fig <- p1 / p2 + plot_annotation(
  title = "Archived-release controlled one-factor scaling",
  subtitle = "fastPLS 0.99.25; rSVD oversampling 20, two power iterations; 486/486 runs completed"
)
ggsave(file.path(out, "Figure_S1_frozen_scaling.png"), fig,
       width = 10.2, height = 8.0, dpi = 360, bg = "white")
ggsave(file.path(out, "Figure_S1_frozen_scaling.pdf"), fig,
       width = 10.2, height = 8.0, device = cairo_pdf)

cat("Wrote frozen controlled-scaling figure to", normalizePath(out), "\n")
