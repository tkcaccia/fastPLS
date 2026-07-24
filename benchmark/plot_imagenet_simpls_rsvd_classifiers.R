#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args)) args[[1L]] else "benchmark_results/pipeline4_imagenet"
raw_file <- file.path(out_dir, "pipeline4_imagenet_raw.csv")
summary_file <- file.path(out_dir, "pipeline4_imagenet_summary.csv")
if (!file.exists(raw_file)) stop("Missing Pipeline 4 results: ", raw_file)

raw <- read.csv(raw_file, stringsAsFactors = FALSE, check.names = FALSE)
numeric_fields <- intersect(
  c(
    "ncomp", "effective_ncomp", "fit_time_sec", "predict_time_sec", "total_time_sec",
    "top1_accuracy", "top5_accuracy", "balanced_accuracy", "macro_f1",
    "peak_host_rss_mb", "peak_gpu_mem_mb"
  ),
  names(raw)
)
raw[numeric_fields] <- lapply(raw[numeric_fields], function(x) suppressWarnings(as.numeric(x)))

group_fields <- intersect(
  c(
    "backend", "classifier", "ncomp", "requested_method", "executed_method",
    "input_precision", "classifier_numeric_path"
  ),
  names(raw)
)
ok <- raw[raw$status == "ok" & raw$process_status == "ok", , drop = FALSE]
median_na <- function(x) if (all(is.na(x))) NA_real_ else median(x, na.rm = TRUE)

if (nrow(ok)) {
  keys <- interaction(ok[group_fields], drop = TRUE, lex.order = TRUE)
  groups <- split(seq_len(nrow(ok)), keys)
  summary <- do.call(rbind, lapply(groups, function(idx) {
    result <- ok[idx[[1L]], group_fields, drop = FALSE]
    for (name in numeric_fields) result[[name]] <- median_na(ok[[name]][idx])
    result$completed_replicates <- length(idx)
    result
  }))
  rownames(summary) <- NULL
} else {
  summary <- raw[FALSE, unique(c(group_fields, numeric_fields)), drop = FALSE]
}
write.csv(summary, summary_file, row.names = FALSE, na = "")
write.csv(raw[raw$status != "ok" | raw$process_status != "ok", , drop = FALSE],
          file.path(out_dir, "pipeline4_imagenet_failures.csv"), row.names = FALSE, na = "")

if (!nrow(summary) || !requireNamespace("ggplot2", quietly = TRUE)) quit(save = "no")
library(ggplot2)

metric_labels <- c(
  total_time_sec = "Total fit + prediction time (s)",
  top1_accuracy = "Top-1 accuracy",
  top5_accuracy = "Top-5 accuracy",
  peak_host_rss_mb = "Peak host RSS (MB)",
  peak_gpu_mem_mb = "Peak GPU memory (MB)"
)
long <- do.call(rbind, lapply(names(metric_labels), function(metric) {
  data.frame(
    summary[c("backend", "classifier", "ncomp")],
    metric = metric_labels[[metric]],
    value = summary[[metric]],
    stringsAsFactors = FALSE
  )
}))
long$metric <- factor(long$metric, levels = unname(metric_labels))
long$classifier <- factor(long$classifier, levels = c("argmax", "lda", "cknn"))
long$backend <- factor(long$backend, levels = c("cpu", "cuda"))

palette <- c(cpu = "#0072B2", cuda = "#D55E00")
p <- ggplot(long, aes(ncomp, value, color = backend, linetype = classifier,
                      shape = classifier, group = interaction(backend, classifier))) +
  geom_line(linewidth = 0.7, na.rm = TRUE) +
  geom_point(size = 2.3, fill = "white", stroke = 0.8, na.rm = TRUE) +
  facet_wrap(~metric, ncol = 1, scales = "free_y") +
  scale_color_manual(values = palette, drop = FALSE) +
  scale_linetype_manual(values = c(argmax = "solid", lda = "dashed", cknn = "dotdash"), drop = FALSE) +
  scale_shape_manual(values = c(argmax = 21, lda = 22, cknn = 24), drop = FALSE) +
  scale_x_continuous(breaks = sort(unique(summary$ncomp))) +
  labs(
    title = "ImageNet SIMPLS-rSVD benchmark (1,000,000 training samples)",
    x = "Number of PLS components", y = NULL, color = "Backend",
    linetype = "Classifier", shape = "Classifier"
  ) +
  theme_bw(base_size = 12) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold")
  )

ggsave(file.path(out_dir, "pipeline4_imagenet_accuracy_time_memory.png"), p,
       width = 10, height = 15, dpi = 180)
ggsave(file.path(out_dir, "pipeline4_imagenet_accuracy_time_memory.pdf"), p,
       width = 10, height = 15)

