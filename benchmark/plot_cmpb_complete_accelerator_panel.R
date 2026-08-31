#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

root <- normalizePath(Sys.getenv("FASTPLS_REPO", "."), mustWork = TRUE)
out_dir <- file.path(
  root,
  Sys.getenv(
    "FASTPLS_ACCELERATOR_FIGURE_DIR",
    "benchmark_results/release_0.99.34_accelerator_figure"
  )
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cuda <- read.csv(
  Sys.getenv(
    "FASTPLS_CUDA_PAIRED",
    file.path(
      root, "benchmark_results", "release_0.99.34_cuda_matched",
      "matched_cuda_paired.csv"
    )
  ),
  stringsAsFactors = FALSE
)
cuda$ratio <- cuda$cpu_accelerator_ratio
cuda$metric_ok <- abs(cuda$metric_delta) <= 0.005
cuda$prediction_ok <- cuda$prediction_agreement >= 0.995
cuda$release <- "fastPLS 0.99.34 CUDA panel"

metal <- read.csv(
  Sys.getenv(
    "FASTPLS_METAL_PAIRED",
    file.path(
      root, "benchmark_results", "release_0.99.34_metal_matched",
      "matched_metal_paired.csv"
    )
  ),
  stringsAsFactors = FALSE
)
metal$ratio <- metal$cpu_accelerator_ratio
metal$metric_ok <- abs(metal$metric_delta) <= 0.005
metal$prediction_ok <- metal$prediction_agreement >= 0.995
metal$release <- "fastPLS 0.99.34 Metal panel"

dataset_order <- c(
  "tabula", "tcga_hnsc_methylation", "tcga_brca", "tcga_pan_cancer",
  "retina", "prism", "metref", "gtex_v8", "cifar100", "ccle",
  "cbmc_citeseq"
)
dataset_labels <- c(
  tabula = "Tabula Muris",
  tcga_hnsc_methylation = "TCGA-HNSC",
  tcga_brca = "TCGA-BRCA",
  tcga_pan_cancer = "TCGA Pan-Cancer",
  retina = "Retina",
  prism = "PRISM",
  metref = "MetRef",
  gtex_v8 = "GTEx v8",
  cifar100 = "CIFAR-100",
  ccle = "CCLE",
  cbmc_citeseq = "CBMC CITE-seq"
)
method_order <- c("plssvd", "simpls", "opls", "kernelpls")
method_labels <- c(
  plssvd = "PLS-SVD", simpls = "SIMPLS", opls = "OPLS",
  kernelpls = "kernel PLS"
)

prepare <- function(x) {
  x$dataset <- factor(x$dataset, levels = dataset_order,
                      labels = unname(dataset_labels[dataset_order]))
  x$method <- factor(x$method, levels = method_order,
                     labels = unname(method_labels[method_order]))
  x$concordant <- x$metric_ok & x$prediction_ok
  x$fill_value <- ifelse(is.finite(x$ratio) & x$ratio > 0, log2(x$ratio), NA_real_)
  x$label <- ifelse(
    is.finite(x$ratio) & x$ratio > 0,
    sprintf("%.2fx", x$ratio),
    "NE"
  )
  x
}
cuda <- prepare(cuda)
metal <- prepare(metal)

panel <- function(data, title, legend_title) {
  ggplot(data, aes(method, dataset, fill = fill_value)) +
    geom_tile(color = "white", linewidth = 0.7) +
    geom_text(aes(label = label), size = 2.6, lineheight = 0.9) +
    scale_fill_gradient2(
      low = "#B2182B", mid = "#F7F7F7", high = "#2166AC",
      midpoint = 0, limits = c(-4.5, 4.5), oob = scales::squish,
      na.value = "#D8D8D8", name = legend_title,
      breaks = c(-4, -2, 0, 2, 4),
      labels = sprintf("%.2fx", 2^c(-4, -2, 0, 2, 4))
    ) +
    labs(title = title, x = NULL, y = NULL) +
    theme_minimal(base_size = 10) +
    theme(
      panel.grid = element_blank(),
      axis.text.x = element_text(angle = 28, hjust = 1, size = 9),
      axis.text.y = element_text(size = 8.7),
      plot.title = element_text(face = "bold", size = 11),
      legend.position = "bottom",
      legend.key.width = grid::unit(1.6, "cm"),
      plot.margin = margin(5, 7, 5, 7)
    )
}

figure <- panel(cuda, "A  CPU/CUDA runtime ratio", "CPU / CUDA") |
  panel(metal, "B  CPU/Metal runtime ratio", "CPU / Metal")
figure <- figure + plot_annotation(
  title = "CPU/accelerator runtime ratios across all completed tests",
  subtitle = paste(
    "Every cell reports CPU time divided by accelerator time;",
    "values above one favor the accelerator"
  ),
  theme = theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(size = 9.5, hjust = 0.5)
  )
)

ggsave(
  file.path(out_dir, "Figure_3_complete_cuda_metal_panel.png"), figure,
  width = 11.2, height = 7.7, units = "in", dpi = 320, bg = "white"
)
ggsave(
  file.path(out_dir, "Figure_3_complete_cuda_metal_panel.pdf"), figure,
  width = 11.2, height = 7.7, units = "in", device = cairo_pdf
)

write.csv(cuda, file.path(out_dir, "Figure_3_cuda_plot_data.csv"), row.names = FALSE)
write.csv(metal, file.path(out_dir, "Figure_3_metal_plot_data.csv"), row.names = FALSE)
