#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

root <- normalizePath(
  Sys.getenv("FASTPLS_REPO", unset = "."),
  winslash = "/",
  mustWork = TRUE
)
output_dir <- file.path(
  root,
  "artifacts",
  "CMPB_rewrite_20260826_cycle108",
  "figures"
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

external <- read.csv(
  file.path(
    root,
    "benchmark_results",
    "manuscript_revision_cycle57_20260726",
    "external_single_cpu_accuracy_time_memory.csv"
  ),
  stringsAsFactors = FALSE
)
audit <- read.csv(
  file.path(
    root,
    "benchmark_results",
    "manuscript_revision_cycle62_20260726",
    "accelerator_paired_concordance_audit.csv"
  ),
  stringsAsFactors = FALSE
)

method_levels <- c(
  "fastPLS SIMPLS / argmax",
  "fastPLS SIMPLS / LDA",
  "pls / SIMPLS",
  "plsgenomics / PLS-LDA",
  "mdatools / PLS-DA",
  "plsdepot / SIMPLS",
  "pcv / SIMPLS",
  "chemometrics / PLS eigen",
  "mixOmics / PLS-DA",
  "spls / sPLS-DA"
)
dataset_levels <- c(
  "CCLE", "CIFAR-100", "GTEx v8", "MetRef", "Retina",
  "Tabula\nMuris", "TCGA-\nBRCA", "TCGA-HNSC\nmethyl.",
  "TCGA Pan-\nCancer"
)

external$dataset_label[external$dataset_label == "Tabula Muris"] <- "Tabula\nMuris"
external$dataset_label[external$dataset_label == "TCGA-BRCA"] <- "TCGA-\nBRCA"
external$dataset_label[external$dataset_label == "TCGA-HNSC methyl."] <- "TCGA-HNSC\nmethyl."
external$dataset_label[external$dataset_label == "TCGA Pan-Cancer"] <- "TCGA Pan-\nCancer"

grid <- expand.grid(
  method = method_levels,
  dataset_label = dataset_levels,
  stringsAsFactors = FALSE
)
external <- merge(
  grid,
  external,
  by = c("method", "dataset_label"),
  all.x = TRUE,
  sort = FALSE
)
external$method <- factor(external$method, levels = rev(method_levels))
external$dataset_label <- factor(external$dataset_label, levels = dataset_levels)
external$accuracy_label <- ifelse(
  is.finite(external$median_accuracy),
  sprintf("%.3f", external$median_accuracy),
  "NE"
)
external$time_label <- ifelse(
  !is.finite(external$time_sec),
  "NE",
  ifelse(
    external$time_sec < 1,
    sprintf("%.3f", external$time_sec),
    ifelse(external$time_sec < 100, sprintf("%.1f", external$time_sec), sprintf("%.0f", external$time_sec))
  )
)
external$rss_label <- ifelse(
  is.finite(external$median_peak_host_rss_mb),
  sprintf("%.0f", external$median_peak_host_rss_mb),
  "NE"
)

heatmap_theme <- theme_minimal(base_size = 11) +
  theme(
    axis.title = element_blank(),
    axis.text.x = element_text(size = 9.0, face = "bold"),
    axis.text.y = element_text(size = 9.2),
    panel.grid = element_blank(),
    plot.title = element_text(size = 12.5, face = "bold", hjust = 0),
    plot.subtitle = element_text(size = 9.3, hjust = 0),
    plot.margin = margin(4, 7, 4, 7),
    legend.position = "right",
    legend.title = element_text(size = 8.4),
    legend.text = element_text(size = 8)
  )

accuracy_plot <- ggplot(external, aes(dataset_label, method, fill = median_accuracy)) +
  geom_tile(color = "white", linewidth = 0.65) +
  geom_text(aes(label = accuracy_label), size = 2.85) +
  scale_fill_gradientn(
    colours = c("#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"),
    limits = c(0.65, 1), oob = scales::squish, na.value = "#E6E6E6",
    name = "Accuracy"
  ) +
  labs(title = "A  Predictive accuracy", subtitle = "Fixed outer-test split; NE denotes not evaluated") +
  heatmap_theme

runtime_plot <- ggplot(external, aes(dataset_label, method, fill = log10(time_sec))) +
  geom_tile(color = "white", linewidth = 0.65) +
  geom_text(aes(label = time_label), size = 2.85) +
  scale_fill_gradientn(
    colours = c("#FFF7EC", "#FDD49E", "#FC8D59", "#D7301F", "#7F0000"),
    na.value = "#E6E6E6", name = expression(log[10] * " seconds")
  ) +
  labs(title = "B  Total fitting plus prediction time", subtitle = "Cell labels report seconds; three isolated float64 runs") +
  heatmap_theme

memory_plot <- ggplot(external, aes(dataset_label, method, fill = log10(median_peak_host_rss_mb))) +
  geom_tile(color = "white", linewidth = 0.65) +
  geom_text(aes(label = rss_label), size = 2.85) +
  scale_fill_gradientn(
    colours = c("#F7FCF5", "#C7E9C0", "#74C476", "#238B45", "#00441B"),
    na.value = "#E6E6E6", name = expression(log[10] * " MB")
  ) +
  labs(title = "C  Peak host memory", subtitle = "Cell labels report absolute complete-process RSS (MB)") +
  heatmap_theme

external_figure <- accuracy_plot / runtime_plot / memory_plot +
  plot_annotation(
    title = "Single-CPU SIMPLS classification workflows",
    subtitle = paste(
      "fastPLS and independent R implementations; one effective BLAS thread,",
      "matched float64 inputs and fixed outer splits"
    ),
    theme = theme(
      plot.title = element_text(size = 14.5, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5)
    )
  )

ggsave(
  file.path(output_dir, "Figure_2_external_packages_updated.png"),
  external_figure, width = 10.5, height = 13.5, units = "in", dpi = 360, bg = "white"
)
ggsave(
  file.path(output_dir, "Figure_2_external_packages_updated.pdf"),
  external_figure, width = 10.5, height = 13.5, units = "in", device = cairo_pdf
)

family_labels <- c(
  plssvd = "PLS-SVD", simpls = "SIMPLS", opls = "OPLS", kernelpls = "kernel PLS"
)
dataset_labels <- c(
  cbmc_citeseq = "CBMC CITE-seq", ccle = "CCLE", cifar100 = "CIFAR-100",
  gtex_v8 = "GTEx v8", metref = "MetRef", prism = "PRISM", retina = "Retina",
  tabula = "Tabula Muris", tcga_brca = "TCGA-BRCA",
  tcga_hnsc_methylation = "TCGA-HNSC", tcga_pan_cancer = "TCGA Pan-Cancer"
)
audit$family_label <- factor(unname(family_labels[audit$family]), levels = unname(family_labels))
audit$dataset_label <- unname(dataset_labels[audit$dataset])
audit$display_value <- ifelse(audit$speed_eligible, log2(audit$speedup), NA_real_)
audit$cell_label <- ifelse(
  audit$speed_eligible,
  sprintf("%.2fx", audit$speedup),
  ifelse(
    audit$evidence_status == "discordant_metric", "Metric\ndiscordant",
    ifelse(audit$evidence_status == "discordant_prediction", "Prediction\ndiscordant", "Prediction\nnot retained")
  )
)

backend_theme <- theme_minimal(base_size = 10) +
  theme(
    panel.grid = element_blank(),
    plot.title = element_text(face = "bold", size = 11.5),
    plot.subtitle = element_text(size = 8.4),
    axis.title = element_blank(),
    axis.text = element_text(size = 8.2),
    legend.position = "bottom",
    plot.margin = margin(5, 7, 5, 7)
  )

backend_panel <- function(data, accelerator, title) {
  eligible <- data[data$speed_eligible, , drop = FALSE]
  excluded <- data[!data$speed_eligible, , drop = FALSE]
  ggplot(data, aes(family_label, dataset_label)) +
    geom_tile(data = excluded, fill = "#D9D9D9", color = "white", linewidth = 0.65) +
    geom_tile(data = eligible, aes(fill = display_value), color = "white", linewidth = 0.65) +
    geom_text(aes(label = cell_label), size = 2.45) +
    scale_fill_gradient2(
      low = "#B2182B", mid = "#F7F7F7", high = "#2166AC", midpoint = 0,
      breaks = c(-4, -2, 0, 2, 4), labels = sprintf("%.2fx", 2^c(-4, -2, 0, 2, 4)),
      name = paste("CPU /", accelerator)
    ) +
    labs(
      title = title,
      subtitle = "Eligible: metric difference <= 0.005; prediction agreement >= 0.995"
    ) +
    backend_theme +
    theme(axis.text.x = element_text(angle = 25, hjust = 1))
}

cuda_plot <- backend_panel(
  audit[audit$accelerator == "CUDA", , drop = FALSE],
  "CUDA", "A  Numerically concordant CPU/CUDA runtime ratio"
)
metal_plot <- backend_panel(
  audit[audit$accelerator == "Metal", , drop = FALSE],
  "Metal", "B  Numerically concordant CPU/Metal runtime ratio"
)
accelerator_figure <- (cuda_plot | metal_plot) +
  plot_annotation(
    title = "CPU/accelerator runtime ratios for numerically concordant workflows",
    subtitle = "Ratios above one favor the accelerator; gray cells are excluded from speed claims.",
    theme = theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 9.5, hjust = 0.5)
    )
  )

ggsave(
  file.path(output_dir, "Figure_3_accelerator_concordance_updated.png"),
  accelerator_figure, width = 10.5, height = 5.4, units = "in", dpi = 360, bg = "white"
)
ggsave(
  file.path(output_dir, "Figure_3_accelerator_concordance_updated.pdf"),
  accelerator_figure, width = 10.5, height = 5.4, units = "in", device = cairo_pdf
)
