#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
arg <- function(key, default = "") {
  hit <- grep(paste0("^--", key, "="), args, value = TRUE)
  if (!length(hit)) return(default)
  sub(paste0("^--", key, "="), "", hit[[1L]])
}

results <- arg("results", "")
out_dir <- arg("out-dir", if (nzchar(results)) dirname(results) else file.path("benchmark_results", "pipeline2_single_cv"))
if (!nzchar(results)) results <- file.path(out_dir, "pipeline2_single_cv_raw.csv")
if (!file.exists(results)) stop("Results file not found: ", results)
dir.create(file.path(out_dir, "plots"), recursive = TRUE, showWarnings = FALSE)

d <- read.csv(results, stringsAsFactors = FALSE)
d <- d[d$status == "ok" & is.finite(d$elapsed_sec) & is.finite(d$metric_value), , drop = FALSE]
if (!nrow(d)) {
  message("No successful rows to plot.")
  quit(save = "no", status = 0)
}

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  message("ggplot2 is not installed; skipping plots.")
  quit(save = "no", status = 0)
}

library(ggplot2)

d$variant <- paste(d$backend, d$svd.method, d$classifier, sep = " / ")
d$metric_label <- ifelse(tolower(d$metric_name) %in% c("rmsd", "rmse", "mae", "mse"),
                         paste0(d$metric_name, " (lower better)"),
                         paste0(d$metric_name, " (higher better)"))

p_time <- ggplot(d, aes(x = variant, y = elapsed_sec, fill = backend)) +
  geom_col(width = 0.75) +
  facet_grid(dataset ~ method, scales = "free_y") +
  scale_y_log10() +
  coord_flip() +
  labs(x = NULL, y = "CV elapsed time (s, log scale)", title = "Pipeline 2: single.pls.cv speed") +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom", strip.text = element_text(face = "bold"))

p_metric <- ggplot(d, aes(x = variant, y = metric_value, fill = backend)) +
  geom_col(width = 0.75) +
  facet_grid(dataset ~ method, scales = "free_y") +
  coord_flip() +
  labs(x = NULL, y = "CV predictive metric", title = "Pipeline 2: single.pls.cv predictive performance") +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom", strip.text = element_text(face = "bold"))

ggsave(file.path(out_dir, "plots", "pipeline2_cv_time.png"), p_time, width = 16, height = max(8, 0.7 * length(unique(d$dataset))), dpi = 160)
ggsave(file.path(out_dir, "plots", "pipeline2_cv_metric.png"), p_metric, width = 16, height = max(8, 0.7 * length(unique(d$dataset))), dpi = 160)

message("Plots saved to ", file.path(out_dir, "plots"))
