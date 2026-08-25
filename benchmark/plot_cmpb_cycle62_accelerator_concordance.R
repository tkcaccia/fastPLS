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
source_dir <- file.path(
  root,
  "benchmark_results",
  "manuscript_revision_cycle62_20260726"
)
solver_dir <- file.path(
  root,
  "benchmark_results",
  "manuscript_revision_cycle57_20260726"
)
audit <- read.csv(
  file.path(source_dir, "accelerator_paired_concordance_audit.csv"),
  stringsAsFactors = FALSE
)
solver <- read.csv(
  file.path(solver_dir, "internal_rsvd_irlba_speedup.csv"),
  stringsAsFactors = FALSE
)

family_labels <- c(
  plssvd = "PLS-SVD",
  simpls = "SIMPLS",
  opls = "OPLS",
  kernelpls = "kernel PLS"
)
dataset_labels <- c(
  cbmc_citeseq = "CBMC CITE-seq",
  ccle = "CCLE",
  cifar100 = "CIFAR-100",
  gtex_v8 = "GTEx v8",
  metref = "MetRef",
  prism = "PRISM",
  retina = "Retina",
  tabula = "Tabula Muris",
  tcga_brca = "TCGA-BRCA",
  tcga_hnsc_methylation = "TCGA-HNSC",
  tcga_pan_cancer = "TCGA Pan-Cancer"
)
audit$family_label <- factor(
  unname(family_labels[audit$family]),
  levels = unname(family_labels)
)
audit$dataset_label <- unname(dataset_labels[audit$dataset])
audit$display_value <- ifelse(
  audit$speed_eligible,
  log2(audit$speedup),
  NA_real_
)
audit$cell_label <- ifelse(
  audit$speed_eligible,
  sprintf("%.2fx", audit$speedup),
  ifelse(
    audit$evidence_status == "discordant_metric",
    "Metric\ndiscordant",
    ifelse(
      audit$evidence_status == "discordant_prediction",
      "Prediction\ndiscordant",
      "Prediction\nnot retained"
    )
  )
)

internal_theme <- theme_minimal(base_size = 10) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    plot.title = element_text(face = "bold", size = 11),
    plot.subtitle = element_text(size = 8.3),
    axis.title = element_text(size = 9),
    axis.text = element_text(size = 8),
    legend.position = "bottom",
    plot.margin = margin(6, 8, 6, 8)
  )

speed_colours <- c("#B2182B", "#F7F7F7", "#2166AC")
speed_breaks <- c(-4, -2, 0, 2, 4)

backend_panel <- function(data, accelerator, title) {
  eligible <- data[data$speed_eligible, , drop = FALSE]
  excluded <- data[!data$speed_eligible, , drop = FALSE]
  ggplot(data, aes(family_label, dataset_label)) +
    geom_tile(
      data = excluded,
      fill = "#D9D9D9",
      color = "white",
      linewidth = 0.65
    ) +
    geom_tile(
      data = eligible,
      aes(fill = display_value),
      color = "white",
      linewidth = 0.65
    ) +
    geom_text(aes(label = cell_label), size = 2.45) +
    scale_fill_gradient2(
      low = speed_colours[[1]],
      mid = speed_colours[[2]],
      high = speed_colours[[3]],
      midpoint = 0,
      breaks = speed_breaks,
      labels = sprintf("%.2fx", 2^speed_breaks),
      name = paste("CPU /", accelerator)
    ) +
    labs(
      title = title,
      subtitle = paste0(
        "Colored only when |metric difference| <=0.005 and ",
        "prediction agreement >=0.995"
      ),
      x = NULL,
      y = NULL
    ) +
    internal_theme +
    theme(axis.text.x = element_text(angle = 25, hjust = 1))
}

p_cuda <- backend_panel(
  audit[audit$accelerator == "CUDA", , drop = FALSE],
  "CUDA",
  "A  Numerically concordant CPU/CUDA runtime ratio"
)
p_metal <- backend_panel(
  audit[audit$accelerator == "Metal", , drop = FALSE],
  "Metal",
  "B  Numerically concordant CPU/Metal runtime ratio"
)

solver$dataset_label <- factor(
  solver$dataset_label,
  levels = rev(unique(solver$dataset_label))
)
p_solver <- ggplot(solver, aes(time_speedup, dataset_label)) +
  geom_vline(xintercept = 1, color = "grey45", linewidth = 0.45) +
  geom_segment(
    aes(x = 1, xend = time_speedup, yend = dataset_label),
    color = "#76B7B2",
    linewidth = 0.55
  ) +
  geom_point(size = 2.6, color = "#00877D") +
  geom_text(
    aes(label = sprintf("%.2fx", time_speedup)),
    hjust = -0.15,
    size = 2.7
  ) +
  scale_x_log10(
    limits = c(0.9, 40),
    breaks = c(1, 1.5, 3, 10, 30)
  ) +
  labs(
    title = "C  Workflow speed of the exploratory rSVD setting",
    subtitle = paste(
      "Matched float64 CPU SIMPLS; oversample=10, power=1, seed=123;",
      "101/117 audit checks met (not qualified)"
    ),
    x = "IRLBA time / rSVD time (log scale)",
    y = NULL
  ) +
  internal_theme

figure <- p_cuda | p_metal +
  plot_annotation(
    title = "CPU/accelerator runtime ratios for numerically concordant workflows",
    subtitle = paste(
      "Gray cells are excluded because the predictive metric or paired predictions",
      "were discordant, or paired predictions were not retained."
    ),
    theme = theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 9.5, hjust = 0.5)
    )
  )

ggsave(
  file.path(source_dir, "accelerator_concordance_speedups.png"),
  figure,
  width = 10.5,
  height = 5.4,
  units = "in",
  dpi = 320,
  bg = "white"
)
ggsave(
  file.path(source_dir, "accelerator_concordance_speedups.pdf"),
  figure,
  width = 10.5,
  height = 5.4,
  units = "in",
  device = cairo_pdf
)

solver_figure <- p_solver +
  plot_annotation(
    title = "Exploratory approximate-solver workflow speed",
    subtitle = paste(
      "This one-power rSVD setting did not meet the complete numerical audit",
      "and is not used as estimator-preservation evidence."
    ),
    theme = theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 9.5, hjust = 0.5)
    )
  )

ggsave(
  file.path(source_dir, "rsvd_workflow_speed_supp.png"),
  solver_figure,
  width = 7.2,
  height = 4.8,
  units = "in",
  dpi = 320,
  bg = "white"
)
ggsave(
  file.path(source_dir, "rsvd_workflow_speed_supp.pdf"),
  solver_figure,
  width = 7.2,
  height = 4.8,
  units = "in",
  device = cairo_pdf
)
