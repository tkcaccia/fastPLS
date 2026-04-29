#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
results_dir <- normalizePath(if (length(args)) args[1] else getwd(), mustWork = TRUE)
raw_file <- file.path(results_dir, "dataset_memory_compare_raw.csv")
if (!file.exists(raw_file)) stop("Missing: ", raw_file)

need <- c("data.table", "ggplot2", "cowplot")
miss <- need[!vapply(need, requireNamespace, logical(1), quietly = TRUE)]
if (length(miss)) stop("Install packages: ", paste(miss, collapse = ", "))

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(cowplot)
})

out_dir <- file.path(results_dir, "plots")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

dt <- fread(raw_file)[status %in% c("ok", "capped")]
if (!nrow(dt)) stop("No successful benchmark rows to plot")

backend_cols <- c(
  irlba = "#1b9e77",
  rsvd = "#d95f02",
  pls_pkg = "#555555"
)

impl_lines <- c(
  R = "22",
  cpp = "solid",
  cuda = "dotdash",
  pls_pkg = "dotted"
)

dt[, backend_algorithm := fcase(
  backend == "pls_pkg", "pls_pkg",
  backend %in% c("cpu_rsvd", "gpu_native"), "rsvd",
  backend == "irlba", "irlba",
  default = backend
)]

dt[, implementation := fcase(
  grepl("^CUDA", implementation_label), "cuda",
  grepl("^R", implementation_label), "R",
  grepl("^Cpp", implementation_label), "cpp",
  default = "pls_pkg"
)]

metric_one <- function(col, label) {
  values <- suppressWarnings(as.numeric(dt[[col]]))
  d_metric <- copy(dt)
  d_metric[, plot_value := values]
  d_metric[is.finite(plot_value), .(
    value = as.numeric(median(plot_value, na.rm = TRUE)),
    task_type = as.character(first(na.omit(task_type))),
    perf_name = as.character(first(na.omit(metric_name))),
    n_train = as.numeric(first(na.omit(n_train))),
    n_test = as.numeric(first(na.omit(n_test))),
    p = as.numeric(first(na.omit(p))),
    n_classes = as.numeric(first(na.omit(n_classes)))
  ), by = .(
    dataset,
    method_panel,
    requested_ncomp,
    variant_name,
    implementation,
    backend_algorithm
  )][, metric := label]
}

sumdt <- rbindlist(list(
  metric_one("total_time_ms", "Total time (ms)"),
  metric_one("metric_value", "Performance"),
  metric_one("peak_host_rss_mb", "Peak host RSS (MB)"),
  metric_one("peak_gpu_mem_mb", "Peak GPU memory (MB)")
), fill = TRUE)

fwrite(sumdt, file.path(results_dir, "dataset_memory_compare_summary.csv"))

sumdt[, method_panel := factor(
  method_panel,
  c("plssvd", "simpls", "opls", "kernelpls")
)]
panel_labels <- c(
  plssvd = "plssvd",
  simpls = "simpls",
  opls = "opls (1 orth)",
  kernelpls = "kernelpls"
)
sumdt[, backend_algorithm := factor(backend_algorithm, names(backend_cols))]
sumdt[, implementation := factor(implementation, names(impl_lines))]

sumdt[, line_id := interaction(
  variant_name,
  implementation,
  backend_algorithm,
  drop = TRUE
)]

perf_label <- function(d) {
  m <- tolower(na.omit(d[metric == "Performance", perf_name])[1])
  if (!length(m) || is.na(m)) {
    "Performance"
  } else {
    switch(
      m,
      accuracy = "Accuracy",
      q2 = "Q2",
      rmsd = "RMSD",
      m
    )
  }
}

for (ds in unique(sumdt$dataset)) {
  d <- copy(sumdt[dataset == ds])
  perf_metric_label <- perf_label(d)
  d[metric == "Performance", metric := perf_metric_label]
  metric_order <- c(
    "Total time (ms)",
    perf_metric_label,
    "Peak host RSS (MB)",
    "Peak GPU memory (MB)"
  )

  title <- sprintf(
    "%s | %s | train_n=%s, test_n=%s, p=%s, classes/y=%s",
    ds,
    d$task_type[1],
    d$n_train[1],
    d$n_test[1],
    d$p[1],
    d$n_classes[1]
  )

  common_scales <- list(
    scale_x_continuous(
      breaks = sort(unique(d$requested_ncomp)),
      limits = range(d$requested_ncomp),
      expand = expansion(mult = 0.01)
    ),
    scale_color_manual(
      values = backend_cols,
      breaks = names(backend_cols),
      name = "SVD/backend"
    ),
    scale_linetype_manual(
      values = impl_lines,
      breaks = names(impl_lines),
      name = "Implementation"
    ),
    guides(
      color = guide_legend(
        nrow = 1,
        byrow = TRUE,
        override.aes = list(linewidth = 1.1, size = 2.5)
      ),
      linetype = guide_legend(nrow = 1, byrow = TRUE)
    )
  )

  row_plot <- function(metric_name, show_x = FALSE) {
    dd <- d[metric == metric_name]
    if (identical(metric_name, "Total time (ms)")) {
      dd <- dd[value > 0]
    }
    p <- ggplot(
      dd,
      aes(
        x = requested_ncomp,
        y = value,
        group = line_id,
        color = backend_algorithm,
        linetype = implementation
      )
    ) +
      geom_line(linewidth = 0.85, na.rm = TRUE) +
      geom_point(size = 1.9, stroke = 0.8, na.rm = TRUE) +
      facet_grid(. ~ method_panel, drop = FALSE, labeller = labeller(method_panel = panel_labels)) +
      common_scales +
      labs(
        x = if (show_x) "Requested components" else NULL,
        y = if (identical(metric_name, "Total time (ms)")) "Total time (ms, log scale)" else metric_name,
        caption = "OPLS panels use total requested components = predictive + 1 orthogonal when feasible."
      ) +
      theme_bw(base_size = 15) +
      theme(
        strip.text = element_text(face = "bold", size = 15),
        axis.text = element_text(size = 13),
        axis.title = element_text(size = 16),
        plot.caption = element_text(size = 11, hjust = 0),
        axis.text.x = if (show_x) element_text(size = 13) else element_blank(),
        axis.ticks.x = if (show_x) element_line() else element_blank(),
        panel.grid.minor = element_blank(),
        plot.background = element_rect(fill = "white", color = NA),
        legend.position = "none"
      )
    if (identical(metric_name, "Total time (ms)")) {
      p + scale_y_log10(labels = scales::label_number())
    } else {
      p + scale_y_continuous(labels = scales::label_number())
    }
  }

  plot_rows <- lapply(seq_along(metric_order), function(i) {
    row_plot(metric_order[[i]], show_x = i == length(metric_order))
  })

  title_plot <- cowplot::ggdraw() +
    cowplot::draw_label(title, fontface = "bold", size = 20, x = 0.5, hjust = 0.5)

  shared_legend <- cowplot::get_legend(
    row_plot("Total time (ms)", show_x = TRUE) +
      theme(
        legend.position = "bottom",
        legend.title = element_text(face = "bold", size = 14),
        legend.text = element_text(size = 13),
        legend.key.width = grid::unit(1.35, "cm"),
        legend.key.height = grid::unit(0.52, "cm"),
        legend.spacing.x = grid::unit(0.22, "cm"),
        legend.background = element_rect(fill = "white", color = NA),
        legend.box.background = element_rect(fill = "white", color = NA)
      )
  )

  p_final <- cowplot::plot_grid(
    title_plot,
    cowplot::plot_grid(plotlist = plot_rows, ncol = 1, align = "v"),
    shared_legend,
    ncol = 1,
    rel_heights = c(0.06, 1, 0.10)
  )

  ggsave(
    file.path(out_dir, sprintf("%s_4x4_methods_memory.png", ds)),
    p_final,
    width = 26,
    height = 21,
    dpi = 140,
    limitsize = FALSE,
    bg = "white"
  )
}

cat("Plots written to:", out_dir, "\n")
