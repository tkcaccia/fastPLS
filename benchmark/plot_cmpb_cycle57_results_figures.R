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
  "benchmark_results",
  "manuscript_revision_cycle57_20260726"
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

external_path <- file.path(
  root,
  "benchmark_results",
  "manuscript_multidataset_summary_20260725",
  "source",
  "external_float64_summary.csv"
)
cuda_path <- file.path(
  root,
  "benchmark_results",
  "manuscript_revision_cycle20_20260725",
  "paired_backend_selected_summary.csv"
)
metal_path <- file.path(
  root,
  "benchmark_results",
  "metal_validation_20260726",
  "summary",
  "metal_backend_paired.csv"
)
cpu1_path <- file.path(
  root,
  "benchmark_results_backend_reproducibility_20260722",
  "metref_cpu_threads1",
  "raw.csv"
)
cpu4_path <- file.path(
  root,
  "benchmark_results_backend_reproducibility_20260722",
  "metref_cpu_threads4",
  "raw.csv"
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
family_labels <- c(
  plssvd = "PLS-SVD",
  simpls = "SIMPLS",
  opls = "OPLS",
  kernelpls = "kernel PLS"
)
backend_dataset_labels <- c(
  dataset_labels,
  cbmc_citeseq = "CBMC\nCITE-seq",
  prism = "PRISM"
)

external <- read.csv(
  external_path,
  check.names = FALSE,
  stringsAsFactors = FALSE
)
selected <- external[
  external$method_id %in% names(method_labels) &
    external$dataset %in% names(dataset_labels),
  c(
    "dataset", "method_id", "package", "algorithm", "classifier",
    "ncomp_requested", "reps_ok", "median_time_ms", "iqr_time_ms",
    "median_peak_host_rss_mb", "iqr_peak_host_rss_mb",
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
external_plot_data <- merge(
  grid,
  selected,
  by = c("method", "dataset_label"),
  all.x = TRUE,
  sort = FALSE
)
external_plot_data$method <- factor(
  external_plot_data$method,
  levels = rev(unname(method_labels))
)
external_plot_data$dataset_label <- factor(
  external_plot_data$dataset_label,
  levels = unname(dataset_labels)
)
external_plot_data$accuracy_label <- ifelse(
  is.finite(external_plot_data$median_accuracy),
  sprintf("%.3f", external_plot_data$median_accuracy),
  "NE"
)
external_plot_data$time_label <- ifelse(
  !is.finite(external_plot_data$time_sec),
  "NE",
  ifelse(
    external_plot_data$time_sec < 1,
    sprintf("%.3f", external_plot_data$time_sec),
    ifelse(
      external_plot_data$time_sec < 100,
      sprintf("%.1f", external_plot_data$time_sec),
      sprintf("%.0f", external_plot_data$time_sec)
    )
  )
)
external_plot_data$rss_label <- ifelse(
  is.finite(external_plot_data$median_peak_host_rss_mb),
  sprintf("%.0f", external_plot_data$median_peak_host_rss_mb),
  "NE"
)

heatmap_theme <- theme_minimal(base_size = 12) +
  theme(
    axis.title = element_blank(),
    axis.text.x = element_text(size = 10.2, face = "bold"),
    axis.text.y = element_text(size = 10.4),
    panel.grid = element_blank(),
    plot.title = element_text(size = 13, face = "bold", hjust = 0),
    plot.subtitle = element_text(size = 10.2, hjust = 0),
    plot.margin = margin(5, 8, 5, 8),
    legend.position = "right",
    legend.title = element_text(size = 9),
    legend.text = element_text(size = 8.5)
  )

accuracy_plot <- ggplot(
  external_plot_data,
  aes(dataset_label, method, fill = median_accuracy)
) +
  geom_tile(color = "white", linewidth = 0.7) +
  geom_text(aes(label = accuracy_label), size = 3.25) +
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
  heatmap_theme

runtime_plot <- ggplot(
  external_plot_data,
  aes(dataset_label, method, fill = log10(time_sec))
) +
  geom_tile(color = "white", linewidth = 0.7) +
  geom_text(aes(label = time_label), size = 3.25) +
  scale_fill_gradientn(
    colours = c("#FFF7EC", "#FDD49E", "#FC8D59", "#D7301F", "#7F0000"),
    na.value = "#E6E6E6",
    name = expression(log[10] * " seconds")
  ) +
  labs(
    title = "B  Total fitting plus prediction time",
    subtitle = "Cell labels report seconds; three isolated float64 runs"
  ) +
  heatmap_theme

memory_plot <- ggplot(
  external_plot_data,
  aes(dataset_label, method, fill = log10(median_peak_host_rss_mb))
) +
  geom_tile(color = "white", linewidth = 0.7) +
  geom_text(aes(label = rss_label), size = 3.25) +
  scale_fill_gradientn(
    colours = c("#F7FCF5", "#C7E9C0", "#74C476", "#238B45", "#00441B"),
    na.value = "#E6E6E6",
    name = expression(log[10] * " MB")
  ) +
  labs(
    title = "C  Peak host memory",
    subtitle = "Cell labels report absolute process RSS (MB)"
  ) +
  heatmap_theme

external_figure <- accuracy_plot / runtime_plot / memory_plot +
  plot_annotation(
    title = "Single-CPU SIMPLS classification workflows",
    subtitle = paste(
      "fastPLS and independent R implementations; one effective BLAS thread,",
      "matched float64 inputs and fixed outer splits"
    ),
    theme = theme(
      plot.title = element_text(size = 15, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 11, hjust = 0.5)
    )
  )

write.csv(
  selected,
  file.path(output_dir, "external_single_cpu_accuracy_time_memory.csv"),
  row.names = FALSE
)
ggsave(
  file.path(output_dir, "external_single_cpu_accuracy_time_memory.png"),
  external_figure,
  width = 11,
  height = 14.5,
  units = "in",
  dpi = 320,
  bg = "white"
)
ggsave(
  file.path(output_dir, "external_single_cpu_accuracy_time_memory.pdf"),
  external_figure,
  width = 11,
  height = 14.5,
  units = "in",
  device = cairo_pdf
)

cuda <- read.csv(cuda_path, stringsAsFactors = FALSE)
cuda <- cuda[
  cuda$status == "ok" & cuda$dataset != "nmr",
  c(
    "dataset", "method_panel", "engine", "total_time_sec_median",
    "metric_median"
  )
]
cuda_wide <- reshape(
  cuda,
  idvar = c("dataset", "method_panel"),
  timevar = "engine",
  direction = "wide"
)
cuda_wide$time_speedup <- (
  cuda_wide$total_time_sec_median.CPU /
    cuda_wide$total_time_sec_median.CUDA
)
cuda_wide$metric_delta <- (
  cuda_wide$metric_median.CUDA - cuda_wide$metric_median.CPU
)
cuda_wide$dataset_label <- unname(backend_dataset_labels[cuda_wide$dataset])
cuda_wide$family <- unname(family_labels[cuda_wide$method_panel])

metal <- read.csv(metal_path, stringsAsFactors = FALSE)
metal <- metal[
  metal$precision == "float64" &
    metal$classifier == "argmax" &
    metal$svd_method == "rsvd" &
    metal$oversample == 10 &
    metal$power == 1,
]
metal$time_speedup <- metal$metal_speedup
metal$dataset_label <- unname(backend_dataset_labels[metal$dataset])
metal$family <- unname(family_labels[metal$method])
metal$metric_flag <- abs(metal$metric_delta_metal_minus_cpu) > 0.005

cpu1 <- read.csv(cpu1_path, stringsAsFactors = FALSE)
cpu4 <- read.csv(cpu4_path, stringsAsFactors = FALSE)
cpu1 <- cpu1[
  cpu1$classifier == "argmax",
  c("method_family", "total_time_ms", "accuracy")
]
cpu4 <- cpu4[
  cpu4$classifier == "argmax",
  c("method_family", "total_time_ms", "accuracy")
]
cpu_threads <- merge(
  cpu1,
  cpu4,
  by = "method_family",
  suffixes = c("_cpu1", "_cpu4")
)
cpu_threads$time_speedup <- (
  cpu_threads$total_time_ms_cpu1 / cpu_threads$total_time_ms_cpu4
)
cpu_threads$family <- unname(family_labels[cpu_threads$method_family])

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
solver$time_speedup <- (
  solver$median_time_ms_irlba / solver$median_time_ms_rsvd
)
solver$metric_delta <- (
  solver$median_accuracy_rsvd - solver$median_accuracy_irlba
)
nmr_irlba <- read.csv(
  file.path(nmr_dir, "nmr__cpp_simpls_irlba__n100__rep1.csv"),
  stringsAsFactors = FALSE
)
nmr_rsvd <- read.csv(
  file.path(nmr_dir, "nmr__cpp_simpls_cpu_rsvd__n100__rep1.csv"),
  stringsAsFactors = FALSE
)
solver <- rbind(
  solver,
  data.frame(
    dataset = "nmr",
    median_time_ms_irlba = nmr_irlba$total_time_ms[[1]],
    median_accuracy_irlba = nmr_irlba$metric_value[[1]],
    median_time_ms_rsvd = nmr_rsvd$total_time_ms[[1]],
    median_accuracy_rsvd = nmr_rsvd$metric_value[[1]],
    time_speedup = nmr_irlba$total_time_ms[[1]] /
      nmr_rsvd$total_time_ms[[1]],
    metric_delta = nmr_rsvd$metric_value[[1]] -
      nmr_irlba$metric_value[[1]]
  )
)
solver_labels <- c(
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
solver$dataset_label <- unname(solver_labels[solver$dataset])

write.csv(
  cuda_wide,
  file.path(output_dir, "internal_cuda_speedup.csv"),
  row.names = FALSE
)
write.csv(
  metal,
  file.path(output_dir, "internal_metal_speedup.csv"),
  row.names = FALSE
)
write.csv(
  cpu_threads,
  file.path(output_dir, "internal_cpu1_cpu4_speedup.csv"),
  row.names = FALSE
)
write.csv(
  solver,
  file.path(output_dir, "internal_rsvd_irlba_speedup.csv"),
  row.names = FALSE
)

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

p_cuda <- ggplot(
  cuda_wide,
  aes(family, dataset_label, fill = log2(time_speedup))
) +
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

p_metal <- ggplot(
  metal,
  aes(family, dataset_label, fill = log2(time_speedup))
) +
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

p_threads <- ggplot(
  cpu_threads,
  aes(family, time_speedup)
) +
  geom_hline(yintercept = 1, color = "grey45", linewidth = 0.45) +
  geom_col(width = 0.62, fill = "#7A5195") +
  geom_text(
    aes(label = sprintf("%.2fx", time_speedup)),
    vjust = ifelse(cpu_threads$time_speedup >= 1, -0.35, 1.25),
    size = 3
  ) +
  scale_y_continuous(
    limits = c(0, 1.38),
    breaks = c(0, 0.5, 1, 1.25),
    expand = expansion(mult = c(0, 0.04))
  ) +
  labs(
    title = "C  Four-thread request versus one thread",
    subtitle = "MetRef float32/argmax; one isolated run",
    x = NULL,
    y = "1-thread time / 4-thread-request time"
  ) +
  internal_theme +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))

solver$dataset_label <- factor(
  solver$dataset_label,
  levels = rev(unname(solver_labels))
)
p_solver <- ggplot(
  solver,
  aes(time_speedup, dataset_label)
) +
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
    title = "D  rSVD speed-up over IRLBA",
    subtitle = "Matched float64 CPU SIMPLS; NMR fixed at 100 components",
    x = "IRLBA time / rSVD time (log scale)",
    y = NULL
  ) +
  internal_theme

internal_figure <- (p_cuda | p_metal) / (p_threads | p_solver) +
  plot_annotation(
    title = "Internal execution and solver speed-ups",
    subtitle = "Values above one favor CUDA, Metal, the four-thread request, or rSVD",
    theme = theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5)
    )
  )

ggsave(
  file.path(output_dir, "internal_backend_solver_speedups.png"),
  internal_figure,
  width = 10.5,
  height = 9.6,
  units = "in",
  dpi = 320,
  bg = "white"
)
ggsave(
  file.path(output_dir, "internal_backend_solver_speedups.pdf"),
  internal_figure,
  width = 10.5,
  height = 9.6,
  units = "in",
  device = cairo_pdf
)
