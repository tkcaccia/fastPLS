#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: summarize.R OUT_DIR")
out_dir <- normalizePath(args[[1L]], winslash = "/", mustWork = TRUE)
row_files <- list.files(file.path(out_dir, "rows"), pattern = "[.]csv$", full.names = TRUE)
if (!length(row_files)) stop("No result rows found.")
row_list <- lapply(row_files, read.csv, stringsAsFactors = FALSE)
all_names <- unique(unlist(lapply(row_list, names), use.names = FALSE))
row_list <- lapply(row_list, function(x) {
  missing <- setdiff(all_names, names(x))
  for (name in missing) x[[name]] <- NA
  x[all_names]
})
raw <- do.call(rbind, row_list)
raw <- raw[order(raw$factor_name, raw$factor_value, raw$route, raw$replicate), ]
write.csv(raw, file.path(out_dir, "controlled_scaling_raw.csv"), row.names = FALSE)

ok <- raw[raw$status == "success", , drop = FALSE]
keys <- c("scenario_id", "factor_name", "factor_value", "task_type", "route", "backend", "svd_method", "xprod_requested")
groups <- split(ok, interaction(ok[keys], drop = TRUE, lex.order = TRUE))
med <- function(x) if (any(is.finite(x))) median(x[is.finite(x)]) else NA_real_
iqr <- function(x) if (any(is.finite(x))) IQR(x[is.finite(x)]) else NA_real_
collapse_unique <- function(x) {
  x <- unique(x[!is.na(x) & nzchar(as.character(x))])
  if (length(x)) paste(x, collapse = "|") else NA_character_
}
summary <- do.call(rbind, lapply(groups, function(x) data.frame(
  scenario_id = x$scenario_id[1L],
  factor_name = x$factor_name[1L],
  factor_value = x$factor_value[1L],
  task_type = x$task_type[1L],
  route = x$route[1L],
  backend = x$backend[1L],
  svd_method = x$svd_method[1L],
  xprod_requested = x$xprod_requested[1L],
  xprod_used = collapse_unique(x$xprod_used),
  n_train = x$n_train[1L], n_test = x$n_test[1L], p = x$p[1L], q = x$q[1L],
  class_count = x$class_count[1L], latent_rank = x$latent_rank[1L],
  max_ncomp = x$max_ncomp[1L], requested_prefixes = x$requested_prefixes[1L],
  crosscov_mb = x$crosscov_mb[1L],
  median_fit_sec = med(x$fit_sec), iqr_fit_sec = iqr(x$fit_sec),
  median_prediction_sec = med(x$prediction_sec), iqr_prediction_sec = iqr(x$prediction_sec),
  median_total_sec = med(x$total_sec), iqr_total_sec = iqr(x$total_sec),
  median_incremental_rss_mb = med(x$incremental_peak_rss_mb),
  median_process_peak_rss_mb = med(x$process_peak_rss_mb),
  median_gpu_process_peak_mb = med(x$gpu_process_peak_mb),
  median_model_size_mb = med(x$model_size_mb),
  median_prediction_relative_error = med(x$prediction_relative_error),
  min_prediction_correlation = if (any(is.finite(x$prediction_correlation))) min(x$prediction_correlation, na.rm = TRUE) else NA_real_,
  min_label_agreement = if (any(is.finite(x$label_agreement))) min(x$label_agreement, na.rm = TRUE) else NA_real_,
  median_metric_absolute_difference = med(x$metric_absolute_difference),
  numerical_failures = sum(x$numerical_status == "outside_tolerance", na.rm = TRUE),
  completed_runs = nrow(x),
  stringsAsFactors = FALSE
)))
summary <- summary[order(summary$factor_name, summary$factor_value, summary$route), ]
write.csv(summary, file.path(out_dir, "controlled_scaling_summary.csv"), row.names = FALSE)

cross <- summary[summary$factor_name == "crosscov_mb" & grepl("_(explicit|implicit)$", summary$route), ]
crossovers <- data.frame()
if (nrow(cross)) {
  pair_keys <- unique(cross[c("backend", "factor_value")])
  rows <- lapply(seq_len(nrow(pair_keys)), function(i) {
    k <- pair_keys[i, ]
    z <- cross[cross$backend == k$backend & cross$factor_value == k$factor_value, ]
    ex <- z[z$xprod_requested == "explicit", ]
    im <- z[z$xprod_requested == "implicit", ]
    if (!nrow(ex) || !nrow(im)) return(NULL)
    data.frame(
      backend = k$backend,
      crosscov_mb = k$factor_value,
      explicit_total_sec = ex$median_total_sec[1L],
      implicit_total_sec = im$median_total_sec[1L],
      implicit_over_explicit_time = im$median_total_sec[1L] / ex$median_total_sec[1L],
      explicit_incremental_rss_mb = ex$median_incremental_rss_mb[1L],
      implicit_incremental_rss_mb = im$median_incremental_rss_mb[1L],
      implicit_rss_reduction_fraction = 1 - im$median_incremental_rss_mb[1L] / ex$median_incremental_rss_mb[1L],
      explicit_numerical_failures = ex$numerical_failures[1L],
      implicit_numerical_failures = im$numerical_failures[1L],
      qualified_time_preference = if (im$numerical_failures[1L] == 0L && ex$numerical_failures[1L] == 0L) {
        if (im$median_total_sec[1L] <= ex$median_total_sec[1L]) "implicit" else "explicit"
      } else "not_qualified",
      qualified_memory_preference = if (im$numerical_failures[1L] == 0L && ex$numerical_failures[1L] == 0L) {
        if (im$median_incremental_rss_mb[1L] <= ex$median_incremental_rss_mb[1L]) "implicit" else "explicit"
      } else "not_qualified",
      stringsAsFactors = FALSE
    )
  })
  crossovers <- do.call(rbind, rows)
  write.csv(crossovers, file.path(out_dir, "explicit_implicit_crossover.csv"), row.names = FALSE)
}

auto <- summary[grepl("_rsvd_auto$", summary$route), , drop = FALSE]
accelerator_pairs <- data.frame()
accelerator_crossovers <- data.frame()
if (nrow(auto)) {
  pair_rows <- list()
  for (scenario in unique(auto$scenario_id)) {
    z <- auto[auto$scenario_id == scenario, , drop = FALSE]
    cpu <- z[z$backend == "cpu", , drop = FALSE]
    if (!nrow(cpu)) next
    for (backend in intersect(c("cuda", "metal"), z$backend)) {
      acc <- z[z$backend == backend, , drop = FALSE]
      if (!nrow(acc)) next
      pair_rows[[length(pair_rows) + 1L]] <- data.frame(
        scenario_id = scenario,
        factor_name = acc$factor_name[1L],
        factor_value = acc$factor_value[1L],
        backend = backend,
        cpu_total_sec = cpu$median_total_sec[1L],
        accelerator_total_sec = acc$median_total_sec[1L],
        cpu_over_accelerator_speedup = cpu$median_total_sec[1L] / acc$median_total_sec[1L],
        cpu_incremental_rss_mb = cpu$median_incremental_rss_mb[1L],
        accelerator_incremental_rss_mb = acc$median_incremental_rss_mb[1L],
        accelerator_gpu_process_peak_mb = acc$median_gpu_process_peak_mb[1L],
        accelerator_numerical_failures = acc$numerical_failures[1L],
        qualified_and_faster = acc$numerical_failures[1L] == 0L &&
          is.finite(cpu$median_total_sec[1L]) && is.finite(acc$median_total_sec[1L]) &&
          acc$median_total_sec[1L] < cpu$median_total_sec[1L],
        stringsAsFactors = FALSE
      )
    }
  }
  if (length(pair_rows)) {
    accelerator_pairs <- do.call(rbind, pair_rows)
    write.csv(accelerator_pairs, file.path(out_dir, "cpu_accelerator_pairs.csv"), row.names = FALSE)
    crossover_rows <- lapply(split(accelerator_pairs, interaction(accelerator_pairs$factor_name, accelerator_pairs$backend, drop = TRUE)), function(x) {
      x <- x[order(x$factor_value), , drop = FALSE]
      eligible <- x[x$qualified_and_faster, , drop = FALSE]
      data.frame(
        factor_name = x$factor_name[1L],
        backend = x$backend[1L],
        first_tested_value_qualified_and_faster = if (nrow(eligible)) eligible$factor_value[1L] else NA_real_,
        qualified_faster_points = sum(x$qualified_and_faster),
        tested_points = nrow(x),
        note = if (nrow(eligible)) "Observed crossover within tested grid; not extrapolated beyond it" else "No qualified speed crossover in tested grid",
        stringsAsFactors = FALSE
      )
    })
    accelerator_crossovers <- do.call(rbind, crossover_rows)
    write.csv(accelerator_crossovers, file.path(out_dir, "cpu_accelerator_crossovers.csv"), row.names = FALSE)
  }
}

factor_overview <- do.call(rbind, lapply(unique(summary$factor_name), function(factor) {
  z <- summary[summary$factor_name == factor & summary$route %in% c("cpu_rsvd_auto", "cuda_rsvd_auto", "metal_rsvd_auto"), , drop = FALSE]
  cpu <- z[z$backend == "cpu", , drop = FALSE]
  cuda <- z[z$backend == "cuda", , drop = FALSE]
  metal <- z[z$backend == "metal", , drop = FALSE]
  cross <- accelerator_crossovers[accelerator_crossovers$factor_name == factor & accelerator_crossovers$backend == "cuda", , drop = FALSE]
  range_text <- function(x) {
    x <- sort(unique(signif(x[is.finite(x)], 6)))
    if (length(x)) paste(x, collapse = ", ") else NA_character_
  }
  data.frame(
    factor_name = factor,
    tested_values = range_text(z$factor_value),
    cpu_qualified_points = sum(cpu$numerical_failures == 0L),
    cpu_tested_points = nrow(cpu),
    cuda_qualified_points = sum(cuda$numerical_failures == 0L),
    cuda_tested_points = nrow(cuda),
    metal_qualified_points = sum(metal$numerical_failures == 0L),
    metal_tested_points = nrow(metal),
    first_cuda_value_qualified_and_faster = if (nrow(cross)) cross$first_tested_value_qualified_and_faster[1L] else NA_real_,
    cpu_total_time_range_sec = if (nrow(cpu)) sprintf("%.4g-%.4g", min(cpu$median_total_sec), max(cpu$median_total_sec)) else NA_character_,
    cuda_total_time_range_sec = if (nrow(cuda)) sprintf("%.4g-%.4g", min(cuda$median_total_sec), max(cuda$median_total_sec)) else NA_character_,
    stringsAsFactors = FALSE
  )
}))
write.csv(factor_overview, file.path(out_dir, "controlled_scaling_factor_overview.csv"), row.names = FALSE)

failures <- raw[raw$status != "success" | raw$numerical_status == "outside_tolerance", ]
write.csv(failures, file.path(out_dir, "failures_and_numerical_discordance.csv"), row.names = FALSE)

if (requireNamespace("ggplot2", quietly = TRUE)) {
  library(ggplot2)
  plot_data <- summary[summary$route != "cpu_irlba_explicit", ]
  route_cols <- c(
    cpu_rsvd_auto = "#2F5597", cpu_rsvd_explicit = "#5B9BD5", cpu_rsvd_implicit = "#1F4E79",
    cuda_rsvd_auto = "#C00000", cuda_rsvd_explicit = "#ED7D31", cuda_rsvd_implicit = "#A61C00",
    metal_rsvd_auto = "#70AD47", metal_rsvd_explicit = "#A5A5A5", metal_rsvd_implicit = "#548235"
  )
  theme_pub <- theme_bw(base_size = 9) + theme(legend.position = "bottom", panel.grid.minor = element_blank())
  p_time <- ggplot(plot_data, aes(factor_value, median_total_sec, color = route, group = route)) +
    geom_line() + geom_point(shape = 21, fill = "white", size = 1.8) +
    facet_wrap(~ factor_name, scales = "free", ncol = 3) +
    scale_x_log10() + scale_y_log10() + scale_color_manual(values = route_cols) +
    labs(x = "Swept factor value", y = "Median fit + prediction time (s)", color = "Route") + theme_pub
  ggsave(file.path(out_dir, "controlled_scaling_time.pdf"), p_time, width = 8.2, height = 7)

  p_mem <- ggplot(plot_data, aes(factor_value, median_incremental_rss_mb, color = route, group = route)) +
    geom_line() + geom_point(shape = 21, fill = "white", size = 1.8) +
    facet_wrap(~ factor_name, scales = "free", ncol = 3) +
    scale_x_log10() + scale_y_continuous() + scale_color_manual(values = route_cols) +
    labs(x = "Swept factor value", y = "Median incremental host RSS (MB)", color = "Route") + theme_pub
  ggsave(file.path(out_dir, "controlled_scaling_memory.pdf"), p_mem, width = 8.2, height = 7)

  p_num <- ggplot(plot_data[plot_data$task_type == "regression", ], aes(factor_value, median_prediction_relative_error, color = route, group = route)) +
    geom_hline(yintercept = 0.05, linetype = 2, color = "black") + geom_line() + geom_point() +
    facet_wrap(~ factor_name, scales = "free_x", ncol = 3) + scale_x_log10() + scale_y_log10() +
    scale_color_manual(values = route_cols) + labs(x = "Swept factor value", y = "Relative prediction error", color = "Route") + theme_pub
  ggsave(file.path(out_dir, "controlled_scaling_numerical_error.pdf"), p_num, width = 8.2, height = 7)

  p_cross <- NULL
  if (nrow(crossovers)) {
    p_cross <- ggplot(crossovers, aes(crosscov_mb, implicit_over_explicit_time, color = backend)) +
      geom_hline(yintercept = 1, linetype = 2) + geom_line() + geom_point(shape = 21, fill = "white", size = 2.2) +
      scale_x_log10() + scale_y_log10() +
      labs(x = "Explicit cross-covariance size (MB)", y = "Implicit / explicit total time", color = "Backend") + theme_pub
    ggsave(file.path(out_dir, "explicit_implicit_crossover.pdf"), p_cross, width = 6.5, height = 4.5)
  }
  if (requireNamespace("gridExtra", quietly = TRUE)) {
    panels <- list(p_time, p_mem, p_num)
    if (!is.null(p_cross)) panels <- c(panels, list(p_cross))
    overview <- do.call(gridExtra::arrangeGrob, c(panels, ncol = 2))
    ggsave(file.path(out_dir, "controlled_scaling_overview.pdf"), overview, width = 11, height = 9)
    ggsave(file.path(out_dir, "controlled_scaling_overview.png"), overview, width = 11, height = 9, dpi = 300)
  }
}

writeLines(capture.output(sessionInfo()), file.path(out_dir, "session_info.txt"))
cat("Summaries written to", out_dir, "\n")
