#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) {
  stop("Usage: plot_publication_precision_overview.R PAIRED_CSV OUTPUT_DIR", call. = FALSE)
}

paired_file <- normalizePath(path.expand(args[[1L]]), mustWork = TRUE)
output_dir <- path.expand(args[[2L]])
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop("The ggplot2 package is required.", call. = FALSE)
}
if (!requireNamespace("patchwork", quietly = TRUE)) {
  stop("The patchwork package is required.", call. = FALSE)
}

x <- utils::read.csv(paired_file, check.names = FALSE)
required <- c(
  "dataset", "requested_method", "classifier", "backend", "metric_name",
  "input_storage_reduction_pct", "incremental_host_rss_reduction_pct",
  "gpu_mem_reduction_pct", "float32_speedup",
  "metric_value_float32", "metric_value_float64"
)
missing_fields <- setdiff(required, names(x))
if (length(missing_fields)) {
  stop("Precision table is missing: ", paste(missing_fields, collapse = ", "), call. = FALSE)
}

numeric_fields <- setdiff(required, c("dataset", "requested_method", "classifier", "backend", "metric_name"))
x[numeric_fields] <- lapply(x[numeric_fields], function(z) suppressWarnings(as.numeric(z)))
x <- x[x$classifier %in% c("argmax", "lda"), , drop = FALSE]
if (!nrow(x)) stop("No argmax/LDA precision pairs were found.", call. = FALSE)

x$predictive_deviation <- ifelse(
  x$metric_name == "rmsd",
  100 * abs(x$metric_value_float32 - x$metric_value_float64) /
    pmax(abs(x$metric_value_float64), .Machine$double.eps),
  100 * abs(x$metric_value_float32 - x$metric_value_float64)
)
x$predictive_deviation_unit <- ifelse(
  x$metric_name == "rmsd",
  "relative percent",
  "percentage points"
)

utils::write.csv(
  x,
  file.path(output_dir, "publication_precision_matched_pairs.csv"),
  row.names = FALSE,
  na = ""
)

method_labels <- c(
  plssvd = "PLS-SVD",
  simpls = "SIMPLS",
  opls = "OPLS",
  kernelpls = "kernel-PLS"
)
method_colours <- c(
  plssvd = "#0073C2FF",
  simpls = "#CD534CFF",
  opls = "#00A087FF",
  kernelpls = "#7E6148FF"
)
x$requested_method <- factor(
  x$requested_method,
  levels = names(method_labels),
  labels = unname(method_labels)
)
names(method_colours) <- unname(method_labels)
x$classifier <- factor(x$classifier, levels = c("argmax", "lda"), labels = c("Argmax", "LDA"))
x$dataset <- factor(x$dataset, levels = unique(x$dataset))

library(ggplot2)

base_theme <- theme_bw(base_size = 11.5) +
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(face = "bold"),
    legend.position = "bottom"
  )

point_panel <- function(field, title, y_label, reference = NULL, log_y = FALSE) {
  p <- ggplot(
    x,
    aes(dataset, .data[[field]], colour = requested_method, shape = classifier)
  )
  if (!is.null(reference)) {
    p <- p + geom_hline(yintercept = reference, colour = "grey45", linetype = 2)
  }
  p <- p +
    geom_point(
      position = position_jitter(width = 0.18, height = 0),
      size = 1.8,
      alpha = 0.72,
      na.rm = TRUE
    ) +
    scale_colour_manual(values = method_colours, drop = FALSE) +
    labs(
      title = title, x = NULL, y = y_label,
      colour = "PLS method", shape = "Decoder"
    ) +
    guides(colour = guide_legend(order = 1), shape = guide_legend(order = 2)) +
    base_theme
  if (isTRUE(log_y)) p <- p + scale_y_log10()
  p
}

p_input <- point_panel(
  "input_storage_reduction_pct",
  "A  Input storage",
  "Float32 reduction relative to float64 (%)",
  reference = 0
)
p_host <- point_panel(
  "incremental_host_rss_reduction_pct",
  "B  Incremental host memory",
  "Float32 reduction relative to float64 (%)",
  reference = 0
)
p_gpu <- point_panel(
  "gpu_mem_reduction_pct",
  "C  Peak GPU memory",
  "Float32 reduction relative to float64 (%)",
  reference = 0
)
p_prediction <- point_panel(
  "predictive_deviation",
  "D  Predictive agreement",
  "Absolute deviation (percentage points; RMSD uses relative %)",
  reference = 0
)

panels <- list(p_input, p_host, p_gpu, p_prediction)
panel_grid <- patchwork::wrap_plots(
  lapply(panels, function(p) p + theme(legend.position = "none")),
  ncol = 2
)

legend_source <- p_input +
  guides(
    colour = guide_legend(order = 1, nrow = 1),
    shape = guide_legend(order = 2, nrow = 1)
  ) +
  theme(legend.position = "bottom", legend.box = "horizontal")
legend_table <- ggplotGrob(legend_source)
legend_idx <- grep("^guide-box", legend_table$layout$name)
legend_idx <- legend_idx[vapply(
  legend_table$grobs[legend_idx],
  function(g) inherits(g, "gtable") && length(g$grobs) > 0L,
  logical(1)
)]
if (!length(legend_idx)) stop("Could not construct the shared legend.", call. = FALSE)
legend <- legend_table$grobs[[legend_idx[[1L]]]]

combined <- panel_grid / patchwork::wrap_elements(full = legend) +
  patchwork::plot_layout(heights = c(18, 1.3)) +
  patchwork::plot_annotation(
    title = "Matched float32 versus float64 fastPLS execution",
    subtitle = "cKNN is excluded because its latent-score classifier cache is mixed precision"
  )

ggsave(
  file.path(output_dir, "publication_precision_overview.png"),
  combined,
  width = 15,
  height = 10.5,
  dpi = 320
)
ggsave(
  file.path(output_dir, "publication_precision_overview.pdf"),
  combined,
  width = 15,
  height = 10.5
)

message("Publication precision overview written to: ", normalizePath(output_dir))
