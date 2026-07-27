#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args)) args[[1L]] else
  "benchmark_results/manuscript_revision_cycle80_20260727/imagenet_float32_simpls_lda_path"
root <- normalizePath(root, mustWork = TRUE)
path_csv <- file.path(root, "imagenet_float32_simpls_lda_path.csv")
time_log <- file.path(root, "imagenet_float32_simpls_lda_path.time")
gpu_log <- file.path(root, "imagenet_float32_simpls_lda_path_gpu_trace.csv")
out_png <- file.path(root, "imagenet_float32_simpls_lda_main_figure.png")
out_pdf <- file.path(root, "imagenet_float32_simpls_lda_main_figure.pdf")

dat <- read.csv(path_csv, check.names = FALSE)
stopifnot(
  nrow(dat) == 10L,
  identical(dat$ncomp_effective, seq.int(100L, 1000L, by = 100L)),
  all(dat$status == "success")
)

time_text <- paste(readLines(time_log, warn = FALSE), collapse = "\n")
peak_match <- regexec(
  "Maximum resident set size \\(kbytes\\):[[:space:]]*([0-9]+)",
  time_text
)
peak_values <- regmatches(time_text, peak_match)[[1L]]
host_peak_mb <- if (length(peak_values) > 1L) {
  as.numeric(peak_values[[2L]]) / 1024
} else {
  NA_real_
}
host_baseline_mb <- dat$rss_before_fit_mb[[1L]]

gpu <- read.csv(gpu_log, check.names = FALSE)
gpu$memory_used_mb <- as.numeric(gpu$memory_used_mb)
gpu_baseline_mb <- gpu$memory_used_mb[[1L]]
gpu_peak_mb <- max(gpu$memory_used_mb, na.rm = TRUE)

accuracy <- rbind(
  data.frame(
    ncomp = dat$ncomp_effective,
    metric = "Top-1 accuracy",
    value = dat$top1_accuracy
  ),
  data.frame(
    ncomp = dat$ncomp_effective,
    metric = "Top-5 accuracy",
    value = dat$top5_accuracy
  )
)
memory <- data.frame(
  resource = factor(
    c("Host RSS", "GPU memory"),
    levels = c("Host RSS", "GPU memory")
  ),
  peak = c(host_peak_mb, gpu_peak_mb),
  incremental = c(
    host_peak_mb - host_baseline_mb,
    gpu_peak_mb - gpu_baseline_mb
  )
)

theme_cmpb <- theme_classic(base_size = 10.5) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(size = 9.3, colour = "#333333"),
    axis.title = element_text(face = "bold"),
    legend.position = "bottom",
    legend.title = element_blank()
  )

p1 <- ggplot(
  accuracy,
  aes(ncomp, value, colour = metric, shape = metric)
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.4, fill = "white") +
  scale_colour_manual(
    values = c("Top-1 accuracy" = "#0072B2", "Top-5 accuracy" = "#D55E00")
  ) +
  scale_shape_manual(values = c("Top-1 accuracy" = 21, "Top-5 accuracy" = 22)) +
  scale_x_continuous(breaks = dat$ncomp_effective) +
  coord_cartesian(ylim = c(
    floor(min(accuracy$value) * 100) / 100,
    min(1, ceiling(max(accuracy$value) * 100) / 100)
  )) +
  labs(
    title = "A  Held-out classification",
    subtitle = "Requested prefixes; 1,000 is a boundary stress point",
    x = "SIMPLS components",
    y = "Accuracy"
  ) +
  theme_cmpb

p2 <- ggplot(dat, aes(ncomp_effective, prediction_time_sec)) +
  geom_line(linewidth = 0.8, colour = "#009E73") +
  geom_point(size = 2.5, shape = 21, fill = "#009E73", colour = "black") +
  scale_x_continuous(breaks = dat$ncomp_effective) +
  labs(
    title = "B  Held-out prediction",
    subtitle = sprintf(
      "Shared 100-1000 component fit: %.1f s",
      dat$shared_path_fit_time_sec[[1L]]
    ),
    x = "SIMPLS components",
    y = "Top-5 prediction time (s)"
  ) +
  theme_cmpb +
  theme(legend.position = "none")

p3 <- ggplot(memory, aes(resource, peak, fill = resource)) +
  geom_col(width = 0.62, colour = "black", linewidth = 0.35) +
  geom_text(
    aes(
      label = sprintf(
        "peak %.0f MB\nincrement %.0f MB",
        peak,
        incremental
      )
    ),
    vjust = -0.25,
    size = 3
  ) +
  scale_fill_manual(values = c("Host RSS" = "#56B4E9", "GPU memory" = "#E69F00")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(
    title = "C  Shared-path resources",
    x = NULL,
    y = "Memory (MB)"
  ) +
  theme_cmpb +
  theme(legend.position = "none")

caption <- paste0(
  "1,000,000 training and 281,167 held-out DINOv2 embeddings (p=1,024); ",
  "float32 label-aware SIMPLS/LDA\n",
  "Hybrid route: host deflation and score projection; CUDA rSVD ",
  "(oversampling 20, power 2, seed 123) and LDA"
)
figure <- (p1 | p2 | p3) +
  plot_annotation(
    title = "Exploratory ImageNet supervised dimension reduction",
    subtitle = caption,
    theme = theme(
      plot.title = element_text(face = "bold", size = 15),
      plot.subtitle = element_text(size = 9.5)
    )
  )

ggsave(out_png, figure, width = 12, height = 4.8, dpi = 300, bg = "white")
ggsave(out_pdf, figure, width = 12, height = 4.8, device = cairo_pdf)
cat(out_png, "\n", out_pdf, "\n", sep = "")
