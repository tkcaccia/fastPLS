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
out_dir <- file.path(
  root,
  "benchmark_results",
  "manuscript_revision_cycle52_20260726"
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

external_path <- file.path(
  root,
  "benchmark_results",
  "manuscript_multidataset_summary_20260725",
  "source",
  "external_float64_summary.csv"
)
precision_path <- file.path(
  root,
  "benchmark_results",
  "manuscript_revision_cycle13_20260725",
  "float32_float64_precision_memory_summary.csv"
)
nmr_dir <- file.path(
  root,
  "benchmark_results",
  "manuscript_revision_cycle13_20260725",
  "kernel_suite",
  "pipeline1",
  "real_datasets",
  "run_rows"
)

external <- read.csv(external_path, stringsAsFactors = FALSE)
precision <- read.csv(precision_path, stringsAsFactors = FALSE)

dataset_labels <- c(
  ccle = "CCLE",
  cifar100 = "CIFAR-100",
  gtex_v8 = "GTEx v8",
  metref = "MetRef",
  retina = "Retina",
  tabula = "Tabula Muris",
  tcga_brca = "TCGA-BRCA",
  tcga_hnsc_methylation = "TCGA-HNSC",
  tcga_pan_cancer = "TCGA Pan-Cancer",
  nmr = "NMR"
)

storage <- unique(precision[
  ,
  c("dataset", "precision", "input_storage_mb_median")
])
storage <- storage[storage$dataset %in% c("metref", "prism"), ]
storage$dataset <- factor(
  ifelse(storage$dataset == "metref", "MetRef", "PRISM"),
  levels = c("MetRef", "PRISM")
)
storage$precision <- factor(
  storage$precision,
  levels = c("float64", "float32"),
  labels = c("float64", "float32")
)

solver_irlba <- external[
  external$method_id == "fastPLS_simpls_cpu_irlba",
  c("dataset", "median_time_ms", "median_accuracy")
]
solver_rsvd <- external[
  external$method_id == "fastPLS_simpls_cpu_rsvd",
  c("dataset", "median_time_ms", "median_accuracy")
]
solver <- merge(
  solver_irlba,
  solver_rsvd,
  by = "dataset",
  suffixes = c("_irlba", "_rsvd")
)
solver$speedup <- solver$median_time_ms_irlba / solver$median_time_ms_rsvd
solver$metric_delta <- solver$median_accuracy_rsvd -
  solver$median_accuracy_irlba

read_nmr <- function(name) {
  read.csv(file.path(nmr_dir, name), stringsAsFactors = FALSE)
}
nmr_irlba <- read_nmr("nmr__cpp_simpls_irlba__n100__rep1.csv")
nmr_rsvd <- read_nmr("nmr__cpp_simpls_cpu_rsvd__n100__rep1.csv")
solver <- rbind(
  solver,
  data.frame(
    dataset = "nmr",
    median_time_ms_irlba = nmr_irlba$total_time_ms[[1]],
    median_accuracy_irlba = nmr_irlba$metric_value[[1]],
    median_time_ms_rsvd = nmr_rsvd$total_time_ms[[1]],
    median_accuracy_rsvd = nmr_rsvd$metric_value[[1]],
    speedup = nmr_irlba$total_time_ms[[1]] / nmr_rsvd$total_time_ms[[1]],
    metric_delta = nmr_rsvd$metric_value[[1]] -
      nmr_irlba$metric_value[[1]]
  )
)
solver$dataset_label <- unname(dataset_labels[solver$dataset])
solver$dataset_label <- factor(
  solver$dataset_label,
  levels = rev(unname(dataset_labels))
)

classifier_argmax <- external[
  external$method_id == "fastPLS_simpls_cpu_irlba",
  c("dataset", "median_time_ms", "median_accuracy")
]
classifier_lda <- external[
  external$method_id == "fastPLS_simpls_cpu_irlba_lda",
  c("dataset", "median_time_ms", "median_accuracy")
]
classifier <- merge(
  classifier_argmax,
  classifier_lda,
  by = "dataset",
  suffixes = c("_argmax", "_lda")
)
classifier$accuracy_gain_pp <- 100 * (
  classifier$median_accuracy_lda - classifier$median_accuracy_argmax
)
classifier$time_ratio <- classifier$median_time_ms_lda /
  classifier$median_time_ms_argmax
classifier$dataset_label <- unname(dataset_labels[classifier$dataset])
classifier$dataset_label <- factor(
  classifier$dataset_label,
  levels = rev(unname(dataset_labels))
)

write.csv(
  storage,
  file.path(out_dir, "float32_input_storage.csv"),
  row.names = FALSE
)
write.csv(
  solver,
  file.path(out_dir, "simpls_rsvd_vs_irlba.csv"),
  row.names = FALSE
)
write.csv(
  classifier,
  file.path(out_dir, "simpls_lda_vs_argmax.csv"),
  row.names = FALSE
)

theme_main <- theme_minimal(base_size = 10) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    plot.title = element_text(face = "bold", size = 11),
    axis.title = element_text(size = 9.5),
    axis.text = element_text(size = 8.5),
    legend.position = "bottom",
    legend.title = element_blank(),
    plot.margin = margin(6, 8, 5, 8)
  )

p_storage <- ggplot(
  storage,
  aes(dataset, input_storage_mb_median, fill = precision)
) +
  geom_col(position = position_dodge(width = 0.72), width = 0.64) +
  geom_text(
    aes(label = sprintf("%.1f", input_storage_mb_median)),
    position = position_dodge(width = 0.72),
    vjust = -0.35,
    size = 3
  ) +
  scale_fill_manual(values = c(float64 = "#4E79A7", float32 = "#E15759")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(
    title = "A  Stored input size",
    x = NULL,
    y = "Input matrices (MB)"
  ) +
  theme_main

p_solver <- ggplot(
  solver,
  aes(speedup, dataset_label)
) +
  geom_vline(xintercept = 1, color = "grey45", linewidth = 0.45) +
  geom_segment(
    aes(x = 1, xend = speedup, yend = dataset_label),
    linewidth = 0.55,
    color = "#76B7B2"
  ) +
  geom_point(size = 2.7, color = "#00877D") +
  geom_text(
    aes(label = sprintf("%.2fx", speedup)),
    hjust = -0.15,
    size = 2.8
  ) +
  scale_x_log10(
    breaks = c(1, 1.5, 3, 10, 30),
    limits = c(0.9, 40)
  ) +
  labs(
    title = "B  CPU rSVD speed-up",
    x = "IRLBA time / rSVD time (log scale)",
    y = NULL
  ) +
  theme_main

p_accuracy <- ggplot(
  classifier,
  aes(accuracy_gain_pp, dataset_label)
) +
  geom_vline(xintercept = 0, color = "grey45", linewidth = 0.45) +
  geom_col(
    aes(fill = accuracy_gain_pp >= 0),
    width = 0.62
  ) +
  geom_text(
    aes(
      label = sprintf("%+.1f", accuracy_gain_pp),
      hjust = ifelse(accuracy_gain_pp >= 0, -0.1, 1.1)
    ),
    size = 2.8
  ) +
  scale_fill_manual(
    values = c(`TRUE` = "#59A14F", `FALSE` = "#E15759"),
    guide = "none"
  ) +
  scale_x_continuous(expand = expansion(mult = c(0.18, 0.18))) +
  labs(
    title = "C  LDA gain over argmax",
    x = "Accuracy difference (percentage points)",
    y = NULL
  ) +
  theme_main

p_classifier_time <- ggplot(
  classifier,
  aes(time_ratio, dataset_label)
) +
  geom_vline(xintercept = 1, color = "grey45", linewidth = 0.45) +
  geom_segment(
    aes(x = 1, xend = time_ratio, yend = dataset_label),
    linewidth = 0.55,
    color = "#F28E2B"
  ) +
  geom_point(size = 2.7, color = "#D55E00") +
  geom_text(
    aes(label = sprintf("%.2fx", time_ratio)),
    hjust = ifelse(classifier$time_ratio >= 1, -0.15, 1.15),
    size = 2.8
  ) +
  scale_x_continuous(
    limits = c(0.82, 1.48),
    breaks = c(0.85, 1.0, 1.2, 1.4)
  ) +
  labs(
    title = "D  LDA total-time ratio",
    x = "LDA time / argmax time",
    y = NULL
  ) +
  theme_main

figure <- (p_storage | p_solver) / (p_accuracy | p_classifier_time) +
  plot_annotation(
    title = "Precision, solver, and classifier trade-offs",
    theme = theme(plot.title = element_text(face = "bold", size = 13))
  )

ggsave(
  file.path(out_dir, "main_precision_solver_classifier.png"),
  figure,
  width = 7.2,
  height = 7.0,
  dpi = 320,
  bg = "white"
)
ggsave(
  file.path(out_dir, "main_precision_solver_classifier.pdf"),
  figure,
  width = 7.2,
  height = 7.0,
  device = cairo_pdf
)
