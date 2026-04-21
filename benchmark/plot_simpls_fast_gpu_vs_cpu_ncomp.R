#!/usr/bin/env Rscript

if (!requireNamespace("data.table", quietly = TRUE)) stop("data.table must be installed")
if (!requireNamespace("ggplot2", quietly = TRUE)) stop("ggplot2 must be installed")
if (!requireNamespace("patchwork", quietly = TRUE)) stop("patchwork must be installed")

library(data.table)
library(ggplot2)
library(patchwork)

in_dir <- Sys.getenv("FASTPLS_BENCH_OUT", file.path(getwd(), "benchmark_results_simpls_fast_gpu_ncomp"))
summary_file <- file.path(in_dir, "simpls_fast_gpu_ncomp_summary.csv")
if (!file.exists(summary_file)) stop("Missing summary file: ", summary_file)

dt <- fread(summary_file)
dt <- dt[ok_runs > 0]
dt[, engine_label := fifelse(engine == "hybrid_cpu", "simpls_fast + cpu_rsvd", "simpls_gpu")]
dt[, engine_label := factor(engine_label, levels = c("simpls_fast + cpu_rsvd", "simpls_gpu"))]

cols <- c("simpls_fast + cpu_rsvd" = "#1b9e77", "simpls_gpu" = "#d95f02")

for (ds in unique(dt$dataset)) {
  sub <- dt[dataset == ds]
  if (!nrow(sub)) next

  time_plot <- ggplot(sub, aes(x = ncomp, y = train_time_seconds_median, color = engine_label)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2) +
    scale_color_manual(values = cols, drop = FALSE) +
    labs(
      title = ds,
      subtitle = sprintf("train_n=%s, test_n=%s, p=%s, classes=%s", sub$train_n[1], sub$test_n[1], sub$p[1], sub$classes[1]),
      x = "Number of components",
      y = "Median train time (seconds)",
      color = NULL
    ) +
    theme_bw(base_size = 12) +
    theme(legend.position = "top")

  acc_plot <- ggplot(sub, aes(x = ncomp, y = accuracy_median, color = engine_label)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2) +
    scale_color_manual(values = cols, drop = FALSE) +
    labs(
      x = "Number of components",
      y = "Median accuracy",
      color = NULL
    ) +
    theme_bw(base_size = 12) +
    theme(legend.position = "none")

  combo <- time_plot / acc_plot + plot_layout(heights = c(1, 1))
  ggsave(
    filename = file.path(in_dir, sprintf("%s_gpu_vs_cpu_ncomp.png", ds)),
    plot = combo,
    width = 9,
    height = 8,
    dpi = 160
  )
}

cat("Plots written to:", in_dir, "\n")
