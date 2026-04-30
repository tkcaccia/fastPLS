#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(cowplot)
})

args <- commandArgs(trailingOnly = TRUE)
results_dir <- normalizePath(if (length(args)) args[[1L]] else Sys.getenv("FASTPLS_SYNTH_VAR_OUTDIR", "benchmark_results_synthetic_variable_sweeps"), winslash = "/", mustWork = TRUE)
raw_file <- file.path(results_dir, "synthetic_variable_sweeps_raw.csv")
if (!file.exists(raw_file)) stop("Missing raw synthetic variable-sweep CSV: ", raw_file)

plot_dir <- file.path(results_dir, "plots")
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)
summary_file <- file.path(results_dir, "synthetic_variable_sweeps_summary.csv")

raw <- fread(raw_file)
ok <- raw[status == "ok"]
if (!nrow(ok)) stop("No successful synthetic variable-sweep rows to plot")
ok[, backend_algorithm := fifelse(
  backend_algorithm == "gpu_native",
  "rsvd",
  as.character(backend_algorithm)
)]

backend_cols <- c(
  irlba = "#1b9e77",
  rsvd = "#d95f02",
  flash_svd = "#7570b3",
  pls_pkg = "#555555"
)
impl_lines <- c(
  R = "22",
  cpp = "solid",
  cuda = "dotdash",
  pls_pkg = "dotted"
)
method_levels <- c("plssvd", "simpls", "opls", "kernelpls")
method_labels <- c(
  plssvd = "plssvd",
  simpls = "simpls",
  opls = "opls (1 orth)",
  kernelpls = "kernelpls"
)

ok[, method_panel := factor(method_panel, method_levels)]
backend_breaks <- intersect(names(backend_cols), unique(as.character(ok$backend_algorithm)))
impl_breaks <- intersect(names(impl_lines), unique(as.character(ok$implementation)))
ok[, backend_algorithm := factor(backend_algorithm, backend_breaks)]
ok[, implementation := factor(implementation, impl_breaks)]
ok[, line_id := interaction(variant_name, implementation, backend_algorithm, drop = TRUE)]

summ <- ok[, .(
  fit_time_ms_median = median(fit_time_ms, na.rm = TRUE),
  predict_time_ms_median = median(predict_time_ms, na.rm = TRUE),
  total_time_ms_median = median(total_time_ms, na.rm = TRUE),
  metric_value_median = median(metric_value, na.rm = TRUE),
  accuracy_median = median(accuracy, na.rm = TRUE),
  q2_median = median(q2, na.rm = TRUE),
  rmsd_median = median(rmsd, na.rm = TRUE),
  peak_host_rss_mb_median = median(peak_host_rss_mb, na.rm = TRUE),
  peak_gpu_mem_mb_median = median(peak_gpu_mem_mb, na.rm = TRUE),
  n_ok = .N
), by = .(
  family, task_type, swept_variable, x_value, x_label,
  variant_name, method_panel, engine, implementation, backend_algorithm,
  requested_ncomp, effective_ncomp, n_train, n_test, p, q, n_classes, noise, rank_true, metric_name
)]
summ[, method_panel := factor(method_panel, method_levels)]
summ[, backend_algorithm := factor(backend_algorithm, backend_breaks)]
summ[, implementation := factor(implementation, impl_breaks)]
summ[, line_id := interaction(variant_name, implementation, backend_algorithm, drop = TRUE)]
fwrite(summ, summary_file)

expand_limits <- function(x, log_y = FALSE) {
  x <- x[is.finite(x)]
  if (!length(x)) return(NULL)
  if (isTRUE(log_y)) x <- x[x > 0]
  if (!length(x)) return(NULL)
  rng <- range(x)
  if (identical(rng[[1L]], rng[[2L]])) {
    delta <- if (rng[[1L]] == 0) 1 else abs(rng[[1L]]) * 0.05
    rng <- c(rng[[1L]] - delta, rng[[2L]] + delta)
  }
  if (isTRUE(log_y)) rng <- pmax(rng, min(x[x > 0], na.rm = TRUE) * 0.5)
  rng
}

performance_label <- function(d) {
  metric_names <- unique(na.omit(as.character(d$metric_name)))
  if (length(metric_names) == 1L) return(metric_names[[1L]])
  if (any(metric_names == "accuracy")) return("accuracy")
  if (any(metric_names == "Q2")) return("Q2")
  if (any(metric_names == "RMSD")) return("RMSD")
  "Performance"
}

fmt_num <- function(x) {
  if (!is.finite(x)) return("NA")
  if (abs(x - round(x)) > 1e-8) {
    return(format(signif(x, 4), trim = TRUE, scientific = FALSE))
  }
  scales::comma(x)
}

fmt_range <- function(x) {
  x <- unique(x[is.finite(x)])
  if (!length(x)) return("NA")
  if (length(x) == 1L) return(fmt_num(x[[1L]]))
  sprintf("%s-%s", fmt_num(min(x)), fmt_num(max(x)))
}

family_title <- function(d, fam) {
  swept <- unique(as.character(d$swept_variable))[[1L]]
  task <- unique(as.character(d$task_type))[[1L]]
  bits <- c(
    fam,
    task,
    sprintf("x=%s (%s)", swept, fmt_range(d$x_value)),
    sprintf("ncomp=%s; effective=%s", fmt_range(d$requested_ncomp), fmt_range(d$effective_ncomp)),
    sprintf("train_n=%s", fmt_range(d$n_train)),
    sprintf("test_n=%s", fmt_range(d$n_test)),
    sprintf("p=%s", fmt_range(d$p))
  )
  if (identical(task, "classification")) {
    bits <- c(bits, sprintf("classes=%s", fmt_range(d$n_classes)))
  } else {
    bits <- c(bits, sprintf("q=%s", fmt_range(d$q)))
  }
  bits <- c(bits, sprintf("noise=%s", fmt_range(d$noise)))
  bits <- c(bits, sprintf("true_rank=%s", fmt_range(d$rank_true)))
  paste(bits, collapse = " | ")
}

long_for_family <- function(d) {
  perf <- performance_label(d)
  rbindlist(list(
    d[is.finite(total_time_ms_median), .(
      value = total_time_ms_median,
      metric = "Total time (ms)",
      family, task_type, swept_variable, x_value, x_label, method_panel, variant_name, implementation, backend_algorithm, line_id,
      n_train, n_test, p, q, n_classes, noise, rank_true, requested_ncomp, effective_ncomp
    )],
    d[is.finite(metric_value_median), .(
      value = metric_value_median,
      metric = perf,
      family, task_type, swept_variable, x_value, x_label, method_panel, variant_name, implementation, backend_algorithm, line_id,
      n_train, n_test, p, q, n_classes, noise, rank_true, requested_ncomp, effective_ncomp
    )],
    d[is.finite(peak_host_rss_mb_median), .(
      value = peak_host_rss_mb_median,
      metric = "Peak host RSS (MB)",
      family, task_type, swept_variable, x_value, x_label, method_panel, variant_name, implementation, backend_algorithm, line_id,
      n_train, n_test, p, q, n_classes, noise, rank_true, requested_ncomp, effective_ncomp
    )],
    d[is.finite(peak_gpu_mem_mb_median), .(
      value = peak_gpu_mem_mb_median,
      metric = "Peak GPU memory (MB)",
      family, task_type, swept_variable, x_value, x_label, method_panel, variant_name, implementation, backend_algorithm, line_id,
      n_train, n_test, p, q, n_classes, noise, rank_true, requested_ncomp, effective_ncomp
    )]
  ), fill = TRUE)
}

plot_family <- function(fam) {
  dsum <- copy(summ[family == fam])
  if (!nrow(dsum)) return(invisible(FALSE))
  d <- long_for_family(dsum)
  if (!nrow(d)) return(invisible(FALSE))

  perf <- performance_label(dsum)
  metric_order <- c("Total time (ms)", perf, "Peak host RSS (MB)", "Peak GPU memory (MB)")
  d[, metric := factor(metric, levels = metric_order)]
  d[, method_panel := factor(method_panel, method_levels)]

  x_breaks <- sort(unique(d$x_value))
  x_lab <- unique(d$x_label)[[1L]]
  title <- family_title(d, fam)

  common_scales <- list(
    scale_x_continuous(breaks = x_breaks, labels = scales::label_number()),
    scale_color_manual(values = backend_cols[backend_breaks], breaks = backend_breaks, name = "SVD/backend", drop = FALSE),
    scale_linetype_manual(values = impl_lines[impl_breaks], breaks = impl_breaks, name = "Implementation", drop = FALSE),
    guides(
      color = guide_legend(nrow = 1, byrow = TRUE, override.aes = list(linewidth = 1.1, size = 2.5)),
      linetype = guide_legend(nrow = 1, byrow = TRUE)
    )
  )

  row_plot <- function(metric_name, show_x = FALSE) {
    dd <- d[as.character(metric) == metric_name]
    if (identical(metric_name, "Total time (ms)")) dd <- dd[value > 0]
    y_lim <- expand_limits(dd$value, log_y = identical(metric_name, "Total time (ms)"))
    p <- ggplot(
      dd,
      aes(
        x = x_value,
        y = value,
        group = line_id,
        color = backend_algorithm,
        linetype = implementation
      )
    ) +
      geom_line(linewidth = 0.85, na.rm = TRUE) +
      geom_point(size = 1.9, stroke = 0.8, na.rm = TRUE) +
      facet_grid(. ~ method_panel, drop = FALSE, labeller = labeller(method_panel = method_labels)) +
      common_scales +
      labs(
        x = if (show_x) x_lab else NULL,
        y = if (identical(metric_name, "Total time (ms)")) "Total time (ms, log scale)" else metric_name,
        caption = "Rows share the same y-axis range across method columns."
      ) +
      theme_bw(base_size = 15) +
      theme(
        strip.text = element_text(face = "bold", size = 15),
        axis.text = element_text(size = 13),
        axis.title = element_text(size = 16),
        plot.caption = element_text(size = 11, hjust = 0),
        axis.text.x = if (show_x) element_text(size = 12, angle = 30, hjust = 1) else element_blank(),
        axis.ticks.x = if (show_x) element_line() else element_blank(),
        panel.grid.minor = element_blank(),
        plot.background = element_rect(fill = "white", color = NA),
        legend.position = "none"
      )
    if (identical(metric_name, "Total time (ms)") && !is.null(y_lim) && all(y_lim > 0)) {
      p + scale_y_log10(limits = y_lim, labels = scales::label_number())
    } else if (!is.null(y_lim)) {
      p + coord_cartesian(ylim = y_lim)
    } else {
      p
    }
  }

  plot_rows <- lapply(seq_along(metric_order), function(i) row_plot(metric_order[[i]], show_x = i == length(metric_order)))
  title_plot <- cowplot::ggdraw() + cowplot::draw_label(title, fontface = "bold", size = 20, x = 0.5, hjust = 0.5)
  shared_legend <- cowplot::get_legend(
    row_plot("Total time (ms)", show_x = TRUE) +
      theme(
        legend.position = "bottom",
        legend.title = element_text(face = "bold", size = 14),
        legend.text = element_text(size = 13),
        legend.key.width = grid::unit(1.35, "cm"),
        legend.key.height = grid::unit(0.52, "cm"),
        legend.background = element_rect(fill = "white", color = NA),
        legend.box.background = element_rect(fill = "white", color = NA)
      )
  )
  final <- cowplot::plot_grid(
    title_plot,
    cowplot::plot_grid(plotlist = plot_rows, ncol = 1, align = "v"),
    shared_legend,
    ncol = 1,
    rel_heights = c(0.06, 1, 0.10)
  )
  out <- file.path(plot_dir, sprintf("%s_4x4_synthetic_variable_sweep.png", fam))
  ggsave(out, final, width = 26, height = 21, dpi = 140, limitsize = FALSE, bg = "white")
  invisible(TRUE)
}

for (fam in unique(summ$family)) {
  plot_family(fam)
}

cat("Synthetic variable-sweep plots written to:", plot_dir, "\n")
