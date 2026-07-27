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
input <- file.path(
  root,
  "benchmark_results",
  "manuscript_multidataset_summary_20260725",
  "source",
  "external_float64_summary.csv"
)
output_dir <- file.path(
  root,
  "benchmark_results",
  "manuscript_revision_cycle51_20260726"
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

results <- read.csv(input, check.names = FALSE, stringsAsFactors = FALSE)

method_labels <- c(
  fastPLS_simpls_cpu_irlba = "fastPLS SIMPLS / argmax",
  fastPLS_simpls_cpu_irlba_lda = "fastPLS SIMPLS / LDA",
  pls_simpls_fit = "pls / SIMPLS",
  plsgenomics_pls_lda = "plsgenomics / PLS-LDA",
  mdatools_plsda_or_pls = "mdatools / PLS-DA",
  plsdepot_simpls = "plsdepot / SIMPLS",
  pcv_simpls = "pcv / SIMPLS",
  chemometrics_pls_eigen = "chemometrics / PLS eigen",
  mixOmics_plsda = "mixOmics / PLS-DA",
  spls_splsda = "spls / sPLS-DA"
)
dataset_labels <- c(
  ccle = "CCLE",
  cifar100 = "CIFAR-100",
  gtex_v8 = "GTEx v8",
  metref = "MetRef",
  retina = "Retina",
  tabula = "Tabula\nMuris",
  tcga_brca = "TCGA-\nBRCA",
  tcga_hnsc_methylation = "TCGA-HNSC\nmethyl.",
  tcga_pan_cancer = "TCGA Pan-\nCancer"
)

selected <- results[
  results$method_id %in% names(method_labels) &
    results$dataset %in% names(dataset_labels),
  c(
    "dataset", "method_id", "package", "algorithm", "classifier",
    "ncomp_requested", "reps_ok", "median_time_ms", "iqr_time_ms",
    "median_accuracy", "iqr_metric"
  )
]
selected$method <- unname(method_labels[selected$method_id])
selected$dataset_label <- unname(dataset_labels[selected$dataset])
selected$time_sec <- selected$median_time_ms / 1000

grid <- expand.grid(
  method = unname(method_labels),
  dataset_label = unname(dataset_labels),
  stringsAsFactors = FALSE
)
plot_data <- merge(
  grid,
  selected,
  by = c("method", "dataset_label"),
  all.x = TRUE,
  sort = FALSE
)
plot_data$method <- factor(
  plot_data$method,
  levels = rev(unname(method_labels))
)
plot_data$dataset_label <- factor(
  plot_data$dataset_label,
  levels = unname(dataset_labels)
)
plot_data$accuracy_label <- ifelse(
  is.finite(plot_data$median_accuracy),
  sprintf("%.3f", plot_data$median_accuracy),
  "NE"
)
plot_data$time_label <- ifelse(
  !is.finite(plot_data$time_sec),
  "NE",
  ifelse(
    plot_data$time_sec < 1,
    sprintf("%.3f", plot_data$time_sec),
    ifelse(
      plot_data$time_sec < 100,
      sprintf("%.1f", plot_data$time_sec),
      sprintf("%.0f", plot_data$time_sec)
    )
  )
)

common_theme <- theme_minimal(base_size = 10.5) +
  theme(
    axis.title = element_blank(),
    axis.text.x = element_text(size = 9, face = "bold"),
    axis.text.y = element_text(size = 9.2),
    panel.grid = element_blank(),
    plot.title = element_text(size = 12, face = "bold", hjust = 0),
    plot.subtitle = element_text(size = 9.5, hjust = 0),
    plot.margin = margin(6, 8, 6, 8),
    legend.position = "right",
    legend.title = element_text(size = 9),
    legend.text = element_text(size = 8.5)
  )

accuracy_plot <- ggplot(
  plot_data,
  aes(x = dataset_label, y = method, fill = median_accuracy)
) +
  geom_tile(color = "white", linewidth = 0.7) +
  geom_text(aes(label = accuracy_label), size = 3.0) +
  scale_fill_gradientn(
    colours = c("#F7FBFF", "#BDD7E7", "#6BAED6", "#2171B5", "#08306B"),
    limits = c(0.65, 1),
    oob = scales::squish,
    na.value = "#E6E6E6",
    name = "Accuracy"
  ) +
  labs(
    title = "A  Predictive accuracy",
    subtitle = "Outer-test accuracy; NE denotes not evaluated"
  ) +
  common_theme

runtime_plot <- ggplot(
  plot_data,
  aes(x = dataset_label, y = method, fill = log10(time_sec))
) +
  geom_tile(color = "white", linewidth = 0.7) +
  geom_text(aes(label = time_label), size = 3.0) +
  scale_fill_gradientn(
    colours = c("#FFF7EC", "#FDD49E", "#FC8D59", "#D7301F", "#7F0000"),
    na.value = "#E6E6E6",
    name = expression(log[10] * " seconds")
  ) +
  labs(
    title = "B  Total fitting plus prediction time",
    subtitle = "Cell labels report seconds; medians from three isolated float64 runs"
  ) +
  common_theme

combined <- accuracy_plot / runtime_plot +
  plot_annotation(
    title = "SIMPLS classification workflows and independent R implementations",
    theme = theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5)
    )
  )

write.csv(
  selected,
  file.path(output_dir, "external_simpls_argmax_lda_main.csv"),
  row.names = FALSE
)
ggsave(
  file.path(output_dir, "external_simpls_argmax_lda_main.png"),
  combined,
  width = 10.5,
  height = 10,
  units = "in",
  dpi = 300,
  bg = "white"
)
ggsave(
  file.path(output_dir, "external_simpls_argmax_lda_main.pdf"),
  combined,
  width = 10.5,
  height = 10,
  units = "in",
  bg = "white"
)
