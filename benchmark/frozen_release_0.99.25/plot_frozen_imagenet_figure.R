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

x <- read.csv(file.path(root, "imagenet", "imagenet_all_results.csv"),
              check.names = FALSE)
stopifnot(nrow(x) == 20L, all(x$status == "success"),
          all(x$package_version == "0.99.25"))
x$classifier <- factor(x$classifier, c("argmax", "lda"),
                       c("Argmax", "LDA"))
palette <- c(Argmax = "#0072B2", LDA = "#D55E00")
shapes <- c(Argmax = 16, LDA = 17)

theme_pub <- function() {
  theme_classic(base_size = 10, base_family = "sans") +
    theme(
      legend.position = "top", legend.title = element_blank(),
      plot.title = element_text(face = "bold", size = 11),
      plot.subtitle = element_text(size = 8.5, color = "#444444"),
      strip.background = element_rect(fill = "#F2F2F2", color = NA),
      strip.text = element_text(face = "bold")
    )
}

p1 <- ggplot(x, aes(ncomp, top1_accuracy, color = classifier,
                    shape = classifier, linetype = classifier)) +
  geom_line(linewidth = 0.7) + geom_point(size = 2.1) +
  scale_color_manual(values = palette) +
  scale_shape_manual(values = shapes) +
  scale_y_continuous(labels = function(z) sprintf("%.0f%%", 100 * z)) +
  scale_x_continuous(breaks = seq(100, 1000, 100)) +
  labs(title = "A  Top-1 accuracy", x = "Requested components", y = "Accuracy") +
  theme_pub()

p2 <- ggplot(x, aes(ncomp, top5_accuracy, color = classifier,
                    shape = classifier, linetype = classifier)) +
  geom_line(linewidth = 0.7) + geom_point(size = 2.1) +
  scale_color_manual(values = palette, guide = "none") +
  scale_shape_manual(values = shapes, guide = "none") +
  scale_y_continuous(labels = function(z) sprintf("%.0f%%", 100 * z)) +
  scale_x_continuous(breaks = seq(100, 1000, 100)) +
  labs(title = "B  Top-5 accuracy", x = "Requested components", y = "Accuracy") +
  theme_pub()

timing <- unique(x[c("classifier", "fit_time_sec", "predict_time_sec")])
timing <- rbind(
  data.frame(classifier = timing$classifier, stage = "Fit",
             seconds = timing$fit_time_sec),
  data.frame(classifier = timing$classifier, stage = "Blocked prediction",
             seconds = timing$predict_time_sec)
)
p3 <- ggplot(timing, aes(classifier, seconds, fill = stage)) +
  geom_col(color = "black", linewidth = 0.25, width = 0.65) +
  scale_fill_manual(values = c(Fit = "#009E73", `Blocked prediction` = "#E69F00")) +
  labs(title = "C  Shared-path runtime",
       subtitle = "One fit supplies all ten requested prefixes",
       x = NULL, y = "Seconds", fill = NULL) + theme_pub()

memory <- unique(x[c("classifier", "rss_peak_predict_mb", "gpu_peak_predict_mb")])
memory <- rbind(
  data.frame(classifier = memory$classifier, memory = "Host peak RSS",
             mb = memory$rss_peak_predict_mb),
  data.frame(classifier = memory$classifier, memory = "GPU used-memory peak",
             mb = memory$gpu_peak_predict_mb)
)
p4 <- ggplot(memory, aes(classifier, mb, fill = classifier)) +
  geom_col(color = "black", linewidth = 0.25, width = 0.65) +
  facet_wrap(~memory, scales = "free_y") +
  scale_fill_manual(values = palette, guide = "none") +
  labs(title = "D  Feasibility memory measurements",
       subtitle = "GPU values include runtime/context allocation",
       x = NULL, y = "MiB") + theme_pub()

figure <- (p1 | p2) / (p3 | p4) +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = "Exploratory ImageNet/DINOv2 SIMPLS stress test",
    subtitle = paste(
      "fastPLS 0.99.25; 1,000,000 training and 281,167 held-out embeddings;",
      "float32 CUDA rSVD; single-run noncanonical split"
    )
  )
ggsave(file.path(out, "Figure_5_frozen_imagenet.png"), figure,
       width = 10.4, height = 7.4, dpi = 400, bg = "white")
ggsave(file.path(out, "Figure_5_frozen_imagenet.pdf"), figure,
       width = 10.4, height = 7.4, device = cairo_pdf)
cat("Wrote frozen ImageNet figure to", normalizePath(out), "\n")
