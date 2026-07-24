#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3L) {
  stop(
    "Usage: plot_publication_cknn_case_study.R PIPELINE1_RAW PIPELINE4_SUMMARY OUTPUT_DIR",
    call. = FALSE
  )
}

pipeline1_file <- normalizePath(path.expand(args[[1L]]), mustWork = TRUE)
pipeline4_file <- normalizePath(path.expand(args[[2L]]), mustWork = TRUE)
output_dir <- path.expand(args[[3L]])
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop("The ggplot2 package is required.", call. = FALSE)
}
if (!requireNamespace("patchwork", quietly = TRUE)) {
  stop("The patchwork package is required.", call. = FALSE)
}

median_na <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  if (all(is.na(x))) NA_real_ else stats::median(x, na.rm = TRUE)
}

primary <- utils::read.csv(pipeline1_file, check.names = FALSE)
required_primary <- c(
  "dataset", "requested_method", "backend", "classifier", "requested_ncomp",
  "metric_name", "metric_value", "total_time_ms", "status"
)
missing_primary <- setdiff(required_primary, names(primary))
if (length(missing_primary)) {
  stop("Pipeline 1 table is missing: ", paste(missing_primary, collapse = ", "), call. = FALSE)
}

primary <- primary[
  primary$status %in% c("ok", "capped") &
    primary$metric_name == "accuracy" &
    primary$classifier %in% c("argmax", "lda", "cknn"),
  ,
  drop = FALSE
]
if ("execution_precision" %in% names(primary)) {
  primary <- primary[primary$execution_precision == "float32", , drop = FALSE]
}

identity_fields <- c("dataset", "requested_method", "backend", "requested_ncomp")
if ("executed_method" %in% names(primary)) identity_fields <- c(identity_fields, "executed_method")
if ("effective_ncomp" %in% names(primary)) identity_fields <- c(identity_fields, "effective_ncomp")

aggregate_fields <- c(identity_fields, "classifier")
group_key <- interaction(primary[aggregate_fields], drop = TRUE, lex.order = TRUE)
primary_summary <- do.call(rbind, lapply(split(seq_len(nrow(primary)), group_key), function(idx) {
  out <- primary[idx[[1L]], aggregate_fields, drop = FALSE]
  out$accuracy <- median_na(primary$metric_value[idx])
  out$total_time_ms <- median_na(primary$total_time_ms[idx])
  out$n_replicates <- length(idx)
  out
}))
rownames(primary_summary) <- NULL

cknn <- primary_summary[primary_summary$classifier == "cknn", , drop = FALSE]
standard <- primary_summary[primary_summary$classifier %in% c("argmax", "lda"), , drop = FALSE]
standard_key <- interaction(standard[identity_fields], drop = TRUE, lex.order = TRUE)
best_standard <- do.call(rbind, lapply(split(seq_len(nrow(standard)), standard_key), function(idx) {
  candidate <- standard[idx, , drop = FALSE]
  candidate[which.max(candidate$accuracy), , drop = FALSE]
}))
rownames(best_standard) <- NULL

primary_pairs <- merge(
  cknn,
  best_standard,
  by = identity_fields,
  suffixes = c("_cknn", "_standard")
)
primary_pairs$accuracy_delta <- primary_pairs$accuracy_cknn - primary_pairs$accuracy_standard
primary_pairs$time_ratio_cknn_to_standard <-
  primary_pairs$total_time_ms_cknn / primary_pairs$total_time_ms_standard
utils::write.csv(
  primary_pairs,
  file.path(output_dir, "publication_cknn_primary_matched_pairs.csv"),
  row.names = FALSE,
  na = ""
)

primary_dataset <- do.call(rbind, lapply(split(primary_pairs, primary_pairs$dataset), function(z) {
  data.frame(
    dataset = z$dataset[[1L]],
    matched_configurations = nrow(z),
    median_accuracy_delta = median_na(z$accuracy_delta),
    best_accuracy_delta = if (all(is.na(z$accuracy_delta))) NA_real_ else max(z$accuracy_delta, na.rm = TRUE),
    median_time_ratio = median_na(z$time_ratio_cknn_to_standard)
  )
}))
rownames(primary_dataset) <- NULL
utils::write.csv(
  primary_dataset,
  file.path(output_dir, "publication_cknn_primary_dataset_summary.csv"),
  row.names = FALSE,
  na = ""
)

imagenet <- utils::read.csv(pipeline4_file, check.names = FALSE)
required_imagenet <- c(
  "backend", "classifier", "ncomp", "top1_accuracy", "top5_accuracy",
  "predict_time_sec", "total_time_sec"
)
missing_imagenet <- setdiff(required_imagenet, names(imagenet))
if (length(missing_imagenet)) {
  stop("Pipeline 4 table is missing: ", paste(missing_imagenet, collapse = ", "), call. = FALSE)
}
for (field in setdiff(required_imagenet, c("backend", "classifier"))) {
  imagenet[[field]] <- suppressWarnings(as.numeric(imagenet[[field]]))
}
if ("status" %in% names(imagenet)) {
  imagenet <- imagenet[imagenet$status %in% c("ok", "capped"), , drop = FALSE]
}
imagenet <- imagenet[imagenet$classifier %in% c("argmax", "lda", "cknn"), , drop = FALSE]
imagenet$compression_ratio <- 1024 / imagenet$ncomp
utils::write.csv(
  imagenet,
  file.path(output_dir, "publication_cknn_imagenet_summary.csv"),
  row.names = FALSE,
  na = ""
)

library(ggplot2)

classifier_labels <- c(argmax = "Argmax", lda = "LDA", cknn = "cKNN")
classifier_colours <- c(argmax = "#0072B2", lda = "#009E73", cknn = "#D55E00")
backend_linetypes <- c(CPU = "dashed", CUDA = "solid")
backend_shapes <- c(CPU = 21, CUDA = 24)

imagenet$classifier <- factor(
  imagenet$classifier,
  levels = names(classifier_labels),
  labels = unname(classifier_labels)
)
names(classifier_colours) <- unname(classifier_labels)
imagenet$backend <- factor(imagenet$backend, levels = c("cpu", "cuda"), labels = c("CPU", "CUDA"))

base_theme <- theme_bw(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 12.5),
    legend.position = "bottom"
  )

if (nrow(primary_pairs)) {
  primary_pairs$dataset <- factor(
    primary_pairs$dataset,
    levels = unique(primary_pairs$dataset[order(primary_pairs$accuracy_delta)])
  )
  p_primary <- ggplot(primary_pairs, aes(dataset, 100 * accuracy_delta)) +
    geom_hline(yintercept = 0, colour = "grey45", linetype = 2) +
    geom_boxplot(width = 0.58, outlier.shape = NA, fill = "#F0E6D2", colour = "#333333") +
    geom_jitter(width = 0.16, height = 0, size = 1.25, alpha = 0.45, colour = "#D55E00") +
    labs(
      title = "A  Matched ordinary-dataset accuracy",
      x = NULL,
      y = "cKNN minus best matched\nargmax/LDA accuracy (percentage points)"
    ) +
    base_theme +
    theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none")
} else {
  p_primary <- ggplot() +
    annotate("text", x = 0, y = 0, label = "No matched primary-dataset cKNN rows") +
    theme_void() +
    labs(title = "A  Primary-dataset comparison")
}

curve <- function(y, title, y_label, log_y = FALSE) {
  p <- ggplot(
    imagenet,
    aes(
      ncomp, .data[[y]], colour = classifier, linetype = backend,
      shape = backend, group = interaction(classifier, backend)
    )
  ) +
    geom_line(linewidth = 0.72, na.rm = TRUE) +
    geom_point(size = 2.25, fill = "white", stroke = 0.8, na.rm = TRUE) +
    scale_colour_manual(values = classifier_colours, drop = FALSE) +
    scale_linetype_manual(values = backend_linetypes, drop = FALSE) +
    scale_shape_manual(values = backend_shapes, drop = FALSE) +
    scale_x_continuous(breaks = sort(unique(imagenet$ncomp))) +
    labs(
      title = title, x = "Number of PLS components", y = y_label,
      colour = "Classifier", linetype = "Backend", shape = "Backend"
    ) +
    base_theme
  if (isTRUE(log_y)) p <- p + scale_y_log10()
  p
}

p_top1 <- curve("top1_accuracy", "B  ImageNet top-1 accuracy", "Top-1 accuracy")
p_top5 <- curve("top5_accuracy", "C  ImageNet top-5 accuracy", "Top-5 accuracy")
p_time <- curve(
  "predict_time_sec",
  "D  ImageNet prediction time",
  "Prediction time (s, log scale)",
  log_y = TRUE
)

panels <- list(p_primary, p_top1, p_top5, p_time)
combined <- patchwork::wrap_plots(
  lapply(panels, function(p) p + theme(legend.position = "none")),
  ncol = 2
)

legend_source <- p_top1 +
  guides(
    colour = guide_legend(order = 1, nrow = 1),
    linetype = guide_legend(order = 2, nrow = 1),
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

publication_figure <- combined / patchwork::wrap_elements(full = legend) +
  patchwork::plot_layout(heights = c(18, 1.4)) +
  patchwork::plot_annotation(
    title = "Candidate-kNN as a downstream classifier on PLS scores",
    subtitle = "Matched ordinary-dataset comparisons and the 1,000,000-sample ImageNet stress test"
  )

ggsave(
  file.path(output_dir, "publication_cknn_case_study.png"),
  publication_figure,
  width = 14.5,
  height = 10.5,
  dpi = 320
)
ggsave(
  file.path(output_dir, "publication_cknn_case_study.pdf"),
  publication_figure,
  width = 14.5,
  height = 10.5
)

message("Publication cKNN case-study outputs written to: ", normalizePath(output_dir))
