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
  "manuscript_revision_cycle57_20260726"
)
output_dir <- file.path(
  root,
  "benchmark_results",
  "manuscript_revision_cycle61_20260726"
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cuda <- read.csv(
  file.path(source_dir, "internal_cuda_speedup.csv"),
  stringsAsFactors = FALSE
)
metal <- read.csv(
  file.path(source_dir, "internal_metal_speedup.csv"),
  stringsAsFactors = FALSE
)
solver <- read.csv(
  file.path(source_dir, "internal_rsvd_irlba_speedup.csv"),
  stringsAsFactors = FALSE
)

family_levels <- c("PLS-SVD", "SIMPLS", "OPLS", "kernel PLS")
cuda$family <- factor(cuda$family, levels = family_levels)
metal$family <- factor(metal$family, levels = family_levels)

internal_theme <- theme_minimal(base_size = 10) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    plot.title = element_text(face = "bold", size = 11),
    plot.subtitle = element_text(size = 8.5),
    axis.title = element_text(size = 9),
    axis.text = element_text(size = 8),
    legend.position = "bottom",
    plot.margin = margin(6, 8, 6, 8)
  )

speed_colours <- c("#B2182B", "#F7F7F7", "#2166AC")
speed_breaks <- c(-4, -2, 0, 2, 4)

p_cuda <- ggplot(cuda, aes(family, dataset_label, fill = log2(time_speedup))) +
  geom_tile(color = "white", linewidth = 0.65) +
  geom_text(aes(label = sprintf("%.2fx", time_speedup)), size = 2.7) +
  scale_fill_gradient2(
    low = speed_colours[[1]],
    mid = speed_colours[[2]],
    high = speed_colours[[3]],
    midpoint = 0,
    breaks = speed_breaks,
    labels = sprintf("%.2fx", 2^speed_breaks),
    name = "CPU / CUDA"
  ) +
  labs(
    title = "A  CUDA speed-up over matched CPU",
    subtitle = "Family-specific training-selected rSVD workflows; NMR shown separately",
    x = NULL,
    y = NULL
  ) +
  internal_theme +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))

p_metal <- ggplot(metal, aes(family, dataset_label, fill = log2(time_speedup))) +
  geom_tile(color = "white", linewidth = 0.65) +
  geom_text(
    aes(
      label = paste0(
        sprintf("%.2fx", time_speedup),
        ifelse(metric_flag, "\u2020", "")
      )
    ),
    size = 2.8
  ) +
  scale_fill_gradient2(
    low = speed_colours[[1]],
    mid = speed_colours[[2]],
    high = speed_colours[[3]],
    midpoint = 0,
    breaks = speed_breaks,
    labels = sprintf("%.2fx", 2^speed_breaks),
    name = "CPU / Metal"
  ) +
  labs(
    title = "B  Metal speed-up over matched Apple CPU",
    subtitle = "\u2020Absolute predictive-metric difference >0.005",
    x = NULL,
    y = NULL
  ) +
  internal_theme +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))

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
    title = "C  rSVD speed-up over IRLBA",
    subtitle = "Matched float64 CPU SIMPLS; NMR fixed at 100 components",
    x = "IRLBA time / rSVD time (log scale)",
    y = NULL
  ) +
  internal_theme

internal_figure <- (p_cuda | p_metal) / p_solver +
  plot_layout(heights = c(1.05, 0.95)) +
  plot_annotation(
    title = "Internal execution and solver speed-ups",
    subtitle = "Values above one favor CUDA, Metal, or rSVD; no multicore speed-up is inferred",
    theme = theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5)
    )
  )

ggsave(
  file.path(output_dir, "internal_backend_solver_speedups_no_threads.png"),
  internal_figure,
  width = 10.5,
  height = 8.2,
  units = "in",
  dpi = 320,
  bg = "white"
)
ggsave(
  file.path(output_dir, "internal_backend_solver_speedups_no_threads.pdf"),
  internal_figure,
  width = 10.5,
  height = 8.2,
  units = "in",
  device = cairo_pdf
)
