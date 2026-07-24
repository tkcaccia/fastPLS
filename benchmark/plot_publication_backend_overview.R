#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) {
  stop("Usage: plot_publication_backend_overview.R RAW_CSV OUTPUT_DIR", call. = FALSE)
}

raw_file <- normalizePath(path.expand(args[[1L]]), mustWork = TRUE)
output_dir <- path.expand(args[[2L]])
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop("The ggplot2 package is required.", call. = FALSE)
}

x <- utils::read.csv(raw_file, check.names = FALSE)
required <- c(
  "dataset", "requested_method", "executed_method", "classifier", "backend",
  "requested_ncomp", "effective_ncomp", "total_time_ms", "metric_name",
  "metric_value", "peak_host_rss_mb", "status"
)
missing_fields <- setdiff(required, names(x))
if (length(missing_fields)) {
  stop("Missing fields: ", paste(missing_fields, collapse = ", "), call. = FALSE)
}

numeric_fields <- intersect(
  c(
    "requested_ncomp", "effective_ncomp", "total_time_ms", "metric_value",
    "peak_host_rss_mb", "rss_before_fit_mb"
  ),
  names(x)
)
x[numeric_fields] <- lapply(x[numeric_fields], function(z) suppressWarnings(as.numeric(z)))

x$backend_role <- ifelse(
  x$backend == "cpu_rsvd",
  "cpu_rsvd",
  ifelse(x$backend %in% c("cuda_rsvd", "gpu_native"), "cuda_rsvd", x$backend)
)

x <- x[
  x$status %in% c("ok", "capped") &
    x$backend_role %in% c("cpu_rsvd", "cuda_rsvd") &
    x$classifier %in% c("argmax", "lda"),
  ,
  drop = FALSE
]
if ("execution_precision" %in% names(x)) {
  x <- x[x$execution_precision == "float32", , drop = FALSE]
}
if (!nrow(x)) stop("No eligible matched CPU/CUDA rows were found.", call. = FALSE)

x$incremental_host_rss_mb <- if ("rss_before_fit_mb" %in% names(x)) {
  ifelse(
    is.finite(x$peak_host_rss_mb) & is.finite(x$rss_before_fit_mb),
    pmax(0, x$peak_host_rss_mb - x$rss_before_fit_mb),
    NA_real_
  )
} else {
  NA_real_
}

median_na <- function(z) if (all(is.na(z))) NA_real_ else stats::median(z, na.rm = TRUE)
group_fields <- c(
  "dataset", "requested_method", "executed_method", "classifier", "backend_role",
  "requested_ncomp", "effective_ncomp", "metric_name"
)
keys <- interaction(x[group_fields], drop = TRUE, lex.order = TRUE)
groups <- split(seq_len(nrow(x)), keys)
summary <- do.call(rbind, lapply(groups, function(idx) {
  row <- x[idx[[1L]], group_fields, drop = FALSE]
  row$total_time_ms <- median_na(x$total_time_ms[idx])
  row$metric_value <- median_na(x$metric_value[idx])
  row$peak_host_rss_mb <- median_na(x$peak_host_rss_mb[idx])
  row$incremental_host_rss_mb <- median_na(x$incremental_host_rss_mb[idx])
  row$n_replicates <- length(idx)
  row
}))
rownames(summary) <- NULL

identity_fields <- c(
  "dataset", "requested_method", "classifier", "requested_ncomp", "metric_name"
)
cpu <- summary[summary$backend_role == "cpu_rsvd", , drop = FALSE]
cuda <- summary[summary$backend_role == "cuda_rsvd", , drop = FALSE]
paired <- merge(cpu, cuda, by = identity_fields, suffixes = c("_cpu", "_cuda"))
paired <- paired[
  paired$executed_method_cpu == paired$executed_method_cuda &
    paired$effective_ncomp_cpu == paired$effective_ncomp_cuda,
  ,
  drop = FALSE
]
if (!nrow(paired)) stop("No estimator-matched CPU/CUDA pairs were found.", call. = FALSE)

paired$speedup_cpu_over_cuda <- paired$total_time_ms_cpu / paired$total_time_ms_cuda
paired$host_rss_ratio_cuda_to_cpu <- paired$peak_host_rss_mb_cuda / paired$peak_host_rss_mb_cpu
paired$incremental_host_rss_ratio_cuda_to_cpu <-
  paired$incremental_host_rss_mb_cuda / paired$incremental_host_rss_mb_cpu
paired$metric_delta_cuda_minus_cpu <- paired$metric_value_cuda - paired$metric_value_cpu
paired$metric_delta_display <- ifelse(
  paired$metric_name == "accuracy",
  100 * paired$metric_delta_cuda_minus_cpu,
  ifelse(
    paired$metric_name == "rmsd",
    100 * paired$metric_delta_cuda_minus_cpu / pmax(abs(paired$metric_value_cpu), .Machine$double.eps),
    paired$metric_delta_cuda_minus_cpu
  )
)
paired$metric_delta_label <- ifelse(
  paired$metric_name == "accuracy",
  "Accuracy difference (percentage points)",
  ifelse(paired$metric_name == "rmsd", "RMSD difference (%)", "Q2 difference")
)

utils::write.csv(
  paired,
  file.path(output_dir, "publication_cpu_cuda_matched_pairs.csv"),
  row.names = FALSE,
  na = ""
)

method_labels <- c(
  plssvd = "PLS-SVD",
  simpls = "SIMPLS",
  opls = "OPLS",
  kernelpls = "kernel-PLS"
)
method_colors <- c(
  plssvd = "#0073C2FF",
  simpls = "#CD534CFF",
  opls = "#00A087FF",
  kernelpls = "#7E6148FF"
)
paired$requested_method <- factor(
  paired$requested_method,
  levels = names(method_labels),
  labels = unname(method_labels)
)
names(method_colors) <- unname(method_labels)
paired$classifier <- factor(paired$classifier, levels = c("argmax", "lda"), labels = c("Argmax", "LDA"))
paired$dataset <- factor(paired$dataset, levels = unique(paired$dataset))

library(ggplot2)
base_theme <- theme_bw(base_size = 11.5) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold"),
    legend.position = "bottom",
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(face = "bold")
  )

time_limits <- range(c(paired$total_time_ms_cpu, paired$total_time_ms_cuda), finite = TRUE) / 1000
p_time <- ggplot(
  paired,
  aes(total_time_ms_cpu / 1000, total_time_ms_cuda / 1000, colour = requested_method, shape = classifier)
) +
  geom_abline(slope = 1, intercept = 0, colour = "grey45", linetype = 2) +
  geom_point(size = 2.2, alpha = 0.78) +
  scale_x_log10(limits = time_limits) +
  scale_y_log10(limits = time_limits) +
  scale_colour_manual(values = method_colors, drop = FALSE) +
  labs(
    title = "A  Matched total runtime",
    x = "CPU rSVD (s)", y = "CUDA rSVD (s)", colour = "PLS method", shape = "Decoder"
  ) +
  guides(colour = guide_legend(order = 1), shape = guide_legend(order = 2)) +
  coord_equal() +
  base_theme

speed_breaks <- c(0.1, 0.25, 0.5, 1, 2, 4, 10, 25, 100)
p_speed <- ggplot(
  paired,
  aes(dataset, speedup_cpu_over_cuda, colour = requested_method, shape = classifier)
) +
  geom_hline(yintercept = 1, colour = "grey45", linetype = 2) +
  geom_point(position = position_jitter(width = 0.18, height = 0), size = 1.8, alpha = 0.7) +
  scale_y_log10(breaks = speed_breaks) +
  scale_colour_manual(values = method_colors, drop = FALSE) +
  labs(
    title = "B  CUDA speedup across component grids",
    x = NULL, y = "CPU time / CUDA time", colour = "PLS method", shape = "Decoder"
  ) +
  guides(colour = guide_legend(order = 1), shape = guide_legend(order = 2)) +
  base_theme

p_prediction <- ggplot(
  paired,
  aes(dataset, metric_delta_display, colour = requested_method, shape = classifier)
) +
  geom_hline(yintercept = 0, colour = "grey45", linetype = 2) +
  geom_point(position = position_jitter(width = 0.18, height = 0), size = 1.8, alpha = 0.72) +
  facet_wrap(~metric_delta_label, scales = "free_y", ncol = 1) +
  scale_colour_manual(values = method_colors, drop = FALSE) +
  labs(
    title = "C  Predictive agreement",
    x = NULL, y = "CUDA minus CPU", colour = "PLS method", shape = "Decoder"
  ) +
  guides(colour = guide_legend(order = 1), shape = guide_legend(order = 2)) +
  base_theme

memory_limits <- range(c(paired$peak_host_rss_mb_cpu, paired$peak_host_rss_mb_cuda), finite = TRUE)
p_memory <- ggplot(
  paired,
  aes(peak_host_rss_mb_cpu, peak_host_rss_mb_cuda, colour = requested_method, shape = classifier)
) +
  geom_abline(slope = 1, intercept = 0, colour = "grey45", linetype = 2) +
  geom_point(size = 2.2, alpha = 0.78) +
  scale_x_log10(limits = memory_limits) +
  scale_y_log10(limits = memory_limits) +
  scale_colour_manual(values = method_colors, drop = FALSE) +
  labs(
    title = "D  Peak host memory",
    x = "CPU rSVD peak RSS (MB)", y = "CUDA rSVD peak RSS (MB)",
    colour = "PLS method", shape = "Decoder"
  ) +
  guides(colour = guide_legend(order = 1), shape = guide_legend(order = 2)) +
  coord_equal() +
  base_theme

plots <- list(time = p_time, speed = p_speed, prediction = p_prediction, memory = p_memory)
for (nm in names(plots)) {
  ggsave(
    file.path(output_dir, paste0("publication_backend_", nm, ".png")),
    plots[[nm]], width = 8.5, height = 6.5, dpi = 320
  )
  ggsave(
    file.path(output_dir, paste0("publication_backend_", nm, ".pdf")),
    plots[[nm]], width = 8.5, height = 6.5
  )
}

if (requireNamespace("patchwork", quietly = TRUE)) {
  legend_data <- expand.grid(
    requested_method = factor(unname(method_labels), levels = unname(method_labels)),
    classifier = factor(c("Argmax", "LDA"), levels = c("Argmax", "LDA"))
  )
  legend_plot <- ggplot(
    legend_data,
    aes(seq_len(nrow(legend_data)), 1, colour = requested_method, shape = classifier)
  ) +
    geom_point(size = 2.4) +
    scale_colour_manual(values = method_colors, drop = FALSE) +
    labs(colour = "PLS method", shape = "Decoder") +
    guides(colour = guide_legend(order = 1, nrow = 1), shape = guide_legend(order = 2, nrow = 1)) +
    theme_void() +
    theme(legend.position = "bottom", legend.box = "horizontal")
  legend_table <- ggplotGrob(legend_plot)
  legend_idx <- grep("^guide-box", legend_table$layout$name)
  legend_idx <- legend_idx[vapply(
    legend_table$grobs[legend_idx],
    function(g) inherits(g, "gtable") && length(g$grobs) > 0L,
    logical(1)
  )]
  if (!length(legend_idx)) stop("Could not construct the common publication legend.", call. = FALSE)
  legend_grob <- legend_table$grobs[[legend_idx[[1L]]]]

  panel_grid <- patchwork::wrap_plots(
    p_time + theme(legend.position = "none"),
    p_speed + theme(legend.position = "none"),
    p_prediction + theme(legend.position = "none"),
    p_memory + theme(legend.position = "none"),
    ncol = 2
  )
  combined <- panel_grid / patchwork::wrap_elements(full = legend_grob) +
    patchwork::plot_layout(heights = c(20, 1.2))
  ggsave(
    file.path(output_dir, "publication_backend_overview.png"),
    combined, width = 15, height = 12, dpi = 320
  )
  ggsave(
    file.path(output_dir, "publication_backend_overview.pdf"),
    combined, width = 15, height = 12
  )
}

message("Publication backend overview written to: ", normalizePath(output_dir))
