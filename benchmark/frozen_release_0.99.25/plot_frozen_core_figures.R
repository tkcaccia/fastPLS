#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args)) args[[1L]] else
  file.path("benchmark_results", "frozen_release_0.99.25")
out <- file.path(root, "figures")
dir.create(out, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

theme_pub <- function() {
  theme_classic(base_size = 10, base_family = "sans") +
    theme(
      axis.text.x = element_text(angle = 40, hjust = 1),
      legend.position = "top",
      legend.title = element_blank(),
      plot.title = element_text(face = "bold", size = 11),
      plot.subtitle = element_text(size = 8.5, color = "#444444"),
      strip.background = element_rect(fill = "#F2F2F2", color = NA),
      strip.text = element_text(face = "bold")
    )
}

labels <- c(
  ccle = "CCLE", cifar100 = "CIFAR-100", gtex_v8 = "GTEx v8",
  metref = "MetRef", retina = "Retina", tabula = "Tabula Muris",
  tcga_brca = "TCGA-BRCA",
  tcga_hnsc_methylation = "TCGA-HNSC",
  tcga_pan_cancer = "TCGA Pan-Cancer"
)
palette <- c(fastpls = "#0072B2", pls = "#D55E00", cpu = "#0072B2", cuda = "#E69F00")
shapes <- c(fastpls = 16, pls = 17, cpu = 16, cuda = 17)

# Figure 2: repeated ordinary public workflows in isolated one-CPU processes.
external <- read.csv(
  file.path(root, "external_simpls", "external_simpls_timing_summary.csv"),
  check.names = FALSE
)
external <- subset(external, comparison_profile == "complete_workflow")
external$dataset_label <- factor(
  unname(labels[external$dataset]), levels = unname(labels)
)
external$implementation <- factor(external$implementation, c("fastpls", "pls"))
p_time <- ggplot(external, aes(dataset_label, median_total_sec,
                                color = implementation, shape = implementation,
                                group = implementation)) +
  geom_point(position = position_dodge(width = 0.42), size = 2.4) +
  geom_errorbar(
    aes(ymin = pmax(median_total_sec - iqr_total_sec / 2, 1e-5),
        ymax = median_total_sec + iqr_total_sec / 2),
    position = position_dodge(width = 0.42), width = 0.18
  ) +
  scale_y_log10() +
  scale_color_manual(values = palette, breaks = c("fastpls", "pls"),
                     labels = c("fastPLS", "pls")) +
  scale_shape_manual(values = shapes, breaks = c("fastpls", "pls"),
                     labels = c("fastPLS", "pls")) +
  labs(title = "A  Total fitting and prediction time",
       subtitle = "Median and IQR; three fresh processes",
       x = NULL, y = "Total time (s, log scale)") + theme_pub()

p_mem <- ggplot(external, aes(dataset_label,
                               median_baseline_corrected_peak_increment_mb,
                               color = implementation, shape = implementation,
                               group = implementation)) +
  geom_point(position = position_dodge(width = 0.42), size = 2.4) +
  geom_errorbar(
    aes(ymin = pmax(median_baseline_corrected_peak_increment_mb -
                      iqr_baseline_corrected_peak_increment_mb / 2, 0.05),
        ymax = median_baseline_corrected_peak_increment_mb +
          iqr_baseline_corrected_peak_increment_mb / 2),
    position = position_dodge(width = 0.42), width = 0.18
  ) +
  scale_y_log10() +
  scale_color_manual(values = palette, guide = "none") +
  scale_shape_manual(values = shapes, guide = "none") +
  labs(title = "B  Baseline-corrected process memory",
       subtitle = "Complete-process increment, not isolated workspace",
       x = NULL, y = "Peak RSS increment (MiB, log scale)") + theme_pub()

p_accuracy <- ggplot(external, aes(dataset_label, median_accuracy,
                                    color = implementation, shape = implementation,
                                    group = implementation)) +
  geom_point(position = position_dodge(width = 0.42), size = 2.4) +
  scale_color_manual(values = palette, guide = "none") +
  scale_shape_manual(values = shapes, guide = "none") +
  scale_y_continuous(labels = function(x) sprintf("%.0f%%", 100 * x),
                     limits = c(0, 1)) +
  labs(title = "C  Held-out classification accuracy",
       subtitle = "Argmax decoding; identical stored split and component count",
       x = NULL, y = "Accuracy") + theme_pub()

fig2 <- p_time / p_mem / p_accuracy +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = "Repeated single-CPU SIMPLS workflows",
    subtitle = "fastPLS 0.99.25 versus pls 2.9.0; deterministic float64 routes"
  )
ggsave(file.path(out, "Figure_2_frozen_external_simpls.png"), fig2,
       width = 7.2, height = 8.1, dpi = 400, bg = "white")
ggsave(file.path(out, "Figure_2_frozen_external_simpls.pdf"), fig2,
       width = 7.2, height = 8.1, device = cairo_pdf)

# Figure 3: selected paired CPU/CUDA workflows from the same frozen release.
selected <- read.csv(
  file.path(root, "selected_backend", "selected_backend_all_runs.csv"),
  check.names = FALSE
)
selected$dataset_label <- factor(
  unname(c(cifar100 = "CIFAR-100", metref = "MetRef", retina = "Retina")[selected$dataset]),
  c("MetRef", "Retina", "CIFAR-100")
)
selected$backend <- factor(selected$backend, c("cpu", "cuda"))
selected$rss_increment_mb <- pmax(
  selected$rss_after_prediction_mb - selected$rss_before_fit_mb, 0
)
selected$gpu_increment_mb <- pmax(
  selected$gpu_after_prediction_mb - selected$gpu_before_fit_mb, 0
)

median_iqr <- function(data, value) {
  aggregate(data[[value]], data[c("dataset_label", "backend")], function(z) {
    c(median = median(z), q25 = unname(quantile(z, 0.25)),
      q75 = unname(quantile(z, 0.75)))
  })
}
unpack <- function(x, value) {
  ans <- median_iqr(x, value)
  values <- if (is.matrix(ans$x)) ans$x else do.call(rbind, ans$x)
  cbind(ans[1:2], as.data.frame(values))
}
t_summary <- unpack(selected, "total_time_sec")
m_summary <- unpack(selected, "rss_increment_mb")
a_summary <- aggregate(accuracy ~ dataset_label + backend, selected, median)

p3_time <- ggplot(t_summary, aes(dataset_label, median, color = backend,
                                  shape = backend)) +
  geom_point(position = position_dodge(width = 0.34), size = 2.8) +
  geom_errorbar(aes(ymin = q25, ymax = q75),
                position = position_dodge(width = 0.34), width = 0.14) +
  scale_y_log10() +
  scale_color_manual(values = palette, breaks = c("cpu", "cuda"),
                     labels = c("CPU", "CUDA")) +
  scale_shape_manual(values = shapes, breaks = c("cpu", "cuda"),
                     labels = c("CPU", "CUDA")) +
  labs(title = "A  Total runtime", subtitle = "Median and IQR; three runs",
       x = NULL, y = "Seconds (log scale)") + theme_pub()

p3_mem <- ggplot(m_summary, aes(dataset_label, pmax(median, 0.05), color = backend,
                                 shape = backend)) +
  geom_point(position = position_dodge(width = 0.34), size = 2.8) +
  geom_errorbar(aes(ymin = pmax(q25, 0.05), ymax = pmax(q75, 0.05)),
                position = position_dodge(width = 0.34), width = 0.14) +
  scale_y_log10() + scale_color_manual(values = palette, guide = "none") +
  scale_shape_manual(values = shapes, guide = "none") +
  labs(title = "B  Host-memory increment",
       subtitle = "After prediction minus pre-fit RSS snapshot",
       x = NULL, y = "MiB (log scale)") + theme_pub()

p3_acc <- ggplot(a_summary, aes(dataset_label, accuracy, color = backend,
                                 shape = backend, group = backend)) +
  geom_point(position = position_dodge(width = 0.28), size = 2.6) +
  scale_color_manual(values = palette, guide = "none") +
  scale_shape_manual(values = shapes, guide = "none") +
  scale_y_continuous(labels = function(x) sprintf("%.1f%%", 100 * x)) +
  labs(title = "C  Paired accuracy",
       subtitle = "Same split, component count, and qualified rSVD controls",
       x = NULL, y = "Accuracy") + theme_pub()

fig3 <- p3_time | p3_mem | p3_acc
fig3 <- fig3 + plot_layout(guides = "collect") +
  plot_annotation(
    title = "Selected CPU and CUDA SIMPLS-rSVD workflows",
    subtitle = "fastPLS 0.99.25; oversampling 20, two power iterations, seed 123"
  )
ggsave(file.path(out, "Figure_3_frozen_cpu_cuda.png"), fig3,
       width = 10.2, height = 4.0, dpi = 400, bg = "white")
ggsave(file.path(out, "Figure_3_frozen_cpu_cuda.pdf"), fig3,
       width = 10.2, height = 4.0, device = cairo_pdf)

cat("Wrote frozen-release core figures to", normalizePath(out), "\n")
