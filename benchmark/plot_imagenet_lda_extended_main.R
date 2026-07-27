suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

root <- normalizePath(Sys.getenv("FASTPLS_REPO", unset = "."),
                      winslash = "/", mustWork = TRUE)
out_dir <- file.path(
  root, "benchmark_results", "manuscript_revision_cycle54_20260726"
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

pipeline4_path <- file.path(
  root,
  "benchmark_results",
  "manuscript_multidataset_summary_20260725",
  "source",
  "imagenet_pipeline4_summary.csv"
)
retrieval_path <- file.path(
  root,
  "benchmark_results",
  "imagenet_faiss_matched_1m_20260725",
  "imagenet_faiss_matched_main_table.csv"
)

pipeline4 <- read.csv(pipeline4_path, stringsAsFactors = FALSE)
retrieval <- read.csv(retrieval_path, stringsAsFactors = FALSE)

classification <- subset(
  pipeline4,
  classifier %in% c("argmax", "lda") &
    backend %in% c("cpu", "cuda") &
    status == "ok" &
    ncomp %in% seq(100, 1000, 100)
)
stopifnot(nrow(classification) == 40L)

classification$classifier <- factor(
  classification$classifier,
  levels = c("argmax", "lda"),
  labels = c("Argmax", "LDA")
)
classification$backend <- factor(
  classification$backend,
  levels = c("cpu", "cuda"),
  labels = c("CPU", "CUDA")
)

cuda_accuracy <- subset(classification, backend == "CUDA")

memory <- subset(classification, backend == "CUDA")
memory_host <- data.frame(
  ncomp = memory$ncomp,
  classifier = memory$classifier,
  memory_type = "Host RSS",
  memory_gb = memory$peak_host_rss_mb / 1024
)
memory_gpu <- data.frame(
  ncomp = memory$ncomp,
  classifier = memory$classifier,
  memory_type = "GPU process",
  memory_gb = memory$peak_gpu_compute_apps_mb / 1024
)
memory_long <- rbind(memory_host, memory_gpu)
memory_long$memory_type <- factor(
  memory_long$memory_type,
  levels = c("Host RSS", "GPU process")
)

retrieval_plot <- retrieval[, c(
  "feature_space", "n_features", "top1_accuracy", "top5_accuracy"
)]
retrieval_plot$representation <- factor(
  retrieval_plot$feature_space,
  levels = c("raw_dinov2", "pca_scores", "pls_scores"),
  labels = c("Raw DINOv2", "PCA", "PLS")
)
retrieval_long <- rbind(
  data.frame(
    representation = retrieval_plot$representation,
    n_features = retrieval_plot$n_features,
    metric = "Top-1",
    accuracy = retrieval_plot$top1_accuracy
  ),
  data.frame(
    representation = retrieval_plot$representation,
    n_features = retrieval_plot$n_features,
    metric = "Top-5",
    accuracy = retrieval_plot$top5_accuracy
  )
)
retrieval_long$metric <- factor(
  retrieval_long$metric,
  levels = c("Top-1", "Top-5")
)

classifier_colors <- c("Argmax" = "#4E79A7", "LDA" = "#E15759")
representation_colors <- c(
  "Raw DINOv2" = "#4E79A7",
  "PCA" = "#59A14F",
  "PLS" = "#F28E2B"
)
base_theme <- theme_bw(base_size = 10) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 11),
    legend.title = element_blank(),
    legend.key.width = grid::unit(1.25, "lines")
  )

panel_a <- ggplot(
  cuda_accuracy,
  aes(x = ncomp, y = accuracy, color = classifier)
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.1) +
  scale_color_manual(values = classifier_colors) +
  scale_x_continuous(breaks = seq(100, 1000, 100)) +
  scale_y_continuous(limits = c(0.60, 0.82)) +
  labs(
    title = "A  Classification accuracy",
    x = "Number of SIMPLS components",
    y = "Top-1 accuracy"
  ) +
  base_theme +
  theme(legend.position = "none")

panel_b <- ggplot(
  classification,
  aes(
    x = ncomp,
    y = total_fit_predict_sec,
    color = classifier,
    linetype = backend
  )
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.8) +
  scale_color_manual(values = classifier_colors) +
  scale_linetype_manual(values = c("CPU" = "dashed", "CUDA" = "solid")) +
  scale_x_continuous(breaks = seq(100, 1000, 200)) +
  scale_y_log10() +
  labs(
    title = "B  Complete workflow time",
    x = "Number of SIMPLS components",
    y = "Fit + prediction time (s, log scale)"
  ) +
  base_theme +
  guides(
    color = guide_legend(order = 1, nrow = 1),
    linetype = guide_legend(order = 2, nrow = 1)
  ) +
  theme(legend.position = "bottom")

panel_c <- ggplot(
  memory_long,
  aes(
    x = ncomp,
    y = memory_gb,
    color = classifier,
    linetype = memory_type
  )
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.8) +
  scale_color_manual(values = classifier_colors) +
  scale_linetype_manual(
    values = c("Host RSS" = "solid", "GPU process" = "dotted")
  ) +
  scale_x_continuous(breaks = seq(100, 1000, 200)) +
  labs(
    title = "C  CUDA-run memory footprint",
    x = "Number of SIMPLS components",
    y = "Peak process memory (GiB)"
  ) +
  base_theme +
  theme(legend.position = "none")

panel_d <- ggplot(
  retrieval_long,
  aes(
    x = n_features,
    y = accuracy,
    color = representation,
    shape = metric,
    group = interaction(representation, metric)
  )
) +
  geom_line(linewidth = 0.75) +
  geom_point(size = 2.2) +
  scale_color_manual(values = representation_colors) +
  scale_shape_manual(values = c("Top-1" = 16, "Top-5" = 17)) +
  scale_x_log10(breaks = c(50, 100, 200, 1024)) +
  scale_y_continuous(limits = c(0.58, 0.95)) +
  labs(
    title = "D  Independent FAISS retrieval",
    x = "Representation dimension (log scale)",
    y = "Accuracy"
  ) +
  base_theme +
  guides(
    color = guide_legend(order = 1, nrow = 1),
    shape = guide_legend(order = 2, nrow = 1)
  ) +
  theme(
    legend.position = c(0.62, 0.50),
    legend.box = "vertical",
    legend.background = element_rect(fill = "white", color = "grey70")
  )

combined <- (
  panel_a + panel_b + panel_c + panel_d +
    plot_layout(ncol = 2)
) +
  plot_annotation(
    title = "ImageNet/DINOv2 stress test and supervised reduction",
    theme = theme(plot.title = element_text(face = "bold", size = 16))
  )

png_path <- file.path(out_dir, "imagenet_lda_extended_main.png")
pdf_path <- file.path(out_dir, "imagenet_lda_extended_main.pdf")
ggsave(png_path, combined, width = 7.25, height = 7.45, dpi = 320)
ggsave(pdf_path, combined, width = 7.25, height = 7.45)

write.csv(
  classification,
  file.path(out_dir, "imagenet_argmax_lda_component_path.csv"),
  row.names = FALSE
)
write.csv(
  retrieval,
  file.path(out_dir, "imagenet_retrieval_table.csv"),
  row.names = FALSE
)

cat(png_path, "\n")
cat(pdf_path, "\n")
