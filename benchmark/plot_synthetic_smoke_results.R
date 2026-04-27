#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(patchwork)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_file <- if (length(script_arg)) sub("^--file=", "", script_arg[[1]]) else file.path(getwd(), "benchmark", "plot_synthetic_smoke_results.R")
script_dir <- dirname(normalizePath(script_file, winslash = "/", mustWork = FALSE))
repo_root <- normalizePath(file.path(script_dir, ".."), winslash = "/", mustWork = FALSE)

source(file.path(repo_root, "R", "synthetic_smoke_generators.R"), local = TRUE)

out_dir <- path.expand(Sys.getenv("FASTPLS_SYNTH_SMOKE_OUTDIR", file.path(repo_root, "benchmark_results_synthetic_smoke_chiamaka")))
plot_dir <- file.path(out_dir, "plots")
panel_dir <- file.path(plot_dir, "panels")
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(panel_dir, recursive = TRUE, showWarnings = FALSE)

summary_path <- file.path(out_dir, "benchmark_results_synthetic_smoke_summary.csv")
fair50_path <- file.path(out_dir, "benchmark_results_synthetic_smoke_fair50.csv")

if (!file.exists(summary_path)) {
  stop("Summary CSV not found at: ", summary_path)
}
if (!file.exists(fair50_path)) {
  stop("Fair50 CSV not found at: ", fair50_path)
}

sum_dt <- fread(summary_path)
fair_dt <- fread(fair50_path)
noise_levels <- smoke_noise_levels()
sum_dt[, noise_regime := factor(noise_regime, levels = noise_levels, ordered = TRUE)]
fair_dt[, noise_regime := factor(noise_regime, levels = noise_levels, ordered = TRUE)]

decorate_plot_dt <- function(dt) {
  dt <- copy(dt)
  dt[, svd_family := fifelse(
    grepl("rsvd", svd_method, ignore.case = TRUE),
    "rsvd",
    fifelse(grepl("gpu", svd_method, ignore.case = TRUE),
      "gpu",
      fifelse(grepl("irlba", svd_method, ignore.case = TRUE),
        "irlba",
        fifelse(grepl("exact", svd_method, ignore.case = TRUE),
          "exact",
          fifelse(grepl("pls_pkg", svd_method, ignore.case = TRUE),
            "pls_pkg",
            tolower(svd_method)
          )
        )
      )
    )
  )]
  dt[, engine_shape := fifelse(
    engine == "pls_pkg",
    "pls_pkg",
    fifelse(engine == "R", "R", fifelse(engine == "Rcpp", "Rcpp", fifelse(engine == "GPU", "GPU", engine)))
  )]
  dt[, line_id := fifelse(
    engine == "GPU",
    "GPU",
    fifelse(engine == "pls_pkg", "pls_pkg", paste(engine, svd_family, sep = " / "))
  )]
  dt
}

sum_dt <- decorate_plot_dt(sum_dt)
fair_dt <- decorate_plot_dt(fair_dt)

svd_levels <- c("irlba", "rsvd", "gpu", "exact", "pls_pkg")
svd_levels <- svd_levels[svd_levels %in% unique(c(sum_dt$svd_family, fair_dt$svd_family))]
svd_palette <- c(
  irlba = "#1b9e77",
  rsvd = "#d95f02",
  gpu = "#e7298a",
  exact = "#7570b3",
  pls_pkg = "#666666"
)
svd_palette <- svd_palette[svd_levels]

shape_levels <- c("pls_pkg", "R", "Rcpp", "GPU")
shape_levels <- shape_levels[shape_levels %in% unique(c(sum_dt$engine_shape, fair_dt$engine_shape))]
shape_values <- c(
  pls_pkg = 15,
  R = 17,
  Rcpp = 16,
  GPU = 18
)
shape_values <- shape_values[shape_levels]

linetype_values <- c(
  pls_pkg = "solid",
  R = "22",
  Rcpp = "solid",
  GPU = "longdash"
)
linetype_values <- linetype_values[shape_levels]

base_theme <- function() {
  theme_bw(base_size = 11) +
    theme(
      legend.position = "bottom",
      legend.box = "horizontal",
      legend.justification = "center",
      panel.grid.minor = element_blank(),
      strip.background = element_rect(fill = "grey95"),
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
}

save_plot <- function(p, filename, width = 12, height = 7) {
  ggsave(filename = file.path(plot_dir, filename), plot = p, width = width, height = height, dpi = 200)
}

save_panel_plot <- function(p, filename, width = 15, height = 9) {
  ggsave(filename = file.path(panel_dir, filename), plot = p, width = width, height = height, dpi = 200)
}

time_scale <- function(p, dt) {
  if (nrow(dt) && all(dt$train_ms_median > 0, na.rm = TRUE)) {
    p + scale_y_log10()
  } else {
    p
  }
}

make_line_plot <- function(dt, x_var, y_var, x_label, y_label, title, facet_formula, filename) {
  if (!nrow(dt) || data.table::uniqueN(dt[[x_var]]) <= 1L) {
    message("Skipping plot ", filename, " because the required data are not present or the x-axis has <= 1 unique value.")
    return(invisible(FALSE))
  }
  p <- ggplot(dt, aes(
    x = .data[[x_var]],
    y = .data[[y_var]],
    color = svd_family,
    shape = engine_shape,
    linetype = engine_shape,
    group = line_id
  )) +
    geom_line(linewidth = 0.7) +
    geom_point(size = 1.7) +
    facet_grid(facet_formula, scales = "free_x") +
    labs(title = title, x = x_label, y = y_label, color = "SVD/backend", shape = "Implementation", linetype = "Implementation") +
    scale_color_manual(values = svd_palette, breaks = svd_levels, drop = FALSE) +
    scale_shape_manual(values = shape_values, breaks = shape_levels, drop = FALSE) +
    scale_linetype_manual(values = linetype_values, breaks = shape_levels, drop = FALSE) +
    guides(
      color = guide_legend(order = 1, nrow = 3, byrow = TRUE),
      shape = guide_legend(order = 2, nrow = 3, byrow = TRUE),
      linetype = "none"
    ) +
    base_theme()
  x_data <- dt[[x_var]]
  if (is.factor(x_data) || is.character(x_data)) {
    p <- p + scale_x_discrete(drop = FALSE)
  }
  p <- if (grepl("time", filename, fixed = TRUE)) time_scale(p, dt) else p
  save_plot(p, filename)
  invisible(TRUE)
}

plot_task_family <- function(dt, family_name, x_var, time_file, perf_file, x_label, perf_label, fair = FALSE) {
  sub <- dt[scenario_family == family_name]
  if (!nrow(sub) || data.table::uniqueN(sub[[x_var]]) <= 1L) {
    message("Skipping task plot for ", family_name, if (isTRUE(fair)) " (fair50)" else "", " because data are insufficient.")
    return(invisible(FALSE))
  }
  noise_formula <- stats::as.formula("noise_regime ~ .")
  make_line_plot(
    sub,
    x_var = x_var,
    y_var = "train_ms_median",
    x_label = x_label,
    y_label = "Median train time (ms)",
    title = paste0(family_name, if (isTRUE(fair)) " fair50" else "", " time"),
    facet_formula = noise_formula,
    filename = time_file
  )
  make_line_plot(
    sub,
    x_var = x_var,
    y_var = "metric_median",
    x_label = x_label,
    y_label = perf_label,
    title = paste0(family_name, if (isTRUE(fair)) " fair50" else "", " performance"),
    facet_formula = noise_formula,
    filename = perf_file
  )
  invisible(TRUE)
}

panel_title_text <- function(sub, family_name, x_var, subset_label = NULL, group_name = NULL) {
  parts <- c(family_name)
  if (!identical(x_var, "n_train")) parts <- c(parts, paste0("train_n=", sub$xtrain_nrow[1]))
  if (!identical(x_var, "p")) parts <- c(parts, paste0("p=", sub$xtrain_ncol[1]))
  if (!identical(x_var, "q")) parts <- c(parts, paste0("y_cols=", sub$ytrain_ncol[1]))
  if (!is.null(subset_label)) parts <- c(parts, subset_label)
  if (!is.null(group_name) && !identical(group_name, "all")) parts <- c(parts, gsub("_", " ", group_name, fixed = TRUE))
  paste(parts, collapse = " | ")
}

expand_limits_if_needed <- function(x) {
  rng <- range(x, finite = TRUE, na.rm = TRUE)
  if (!all(is.finite(rng))) return(NULL)
  if (identical(rng[[1]], rng[[2]])) {
    delta <- if (rng[[1]] == 0) 1 else abs(rng[[1]]) * 0.05
    rng <- c(rng[[1]] - delta, rng[[2]] + delta)
  }
  rng
}

empty_panel_plot <- function(title_text) {
  ggplot() +
    annotate("text", x = 0.5, y = 0.5, label = "Not run", size = 5) +
    xlim(0, 1) +
    ylim(0, 1) +
    labs(title = title_text, x = NULL, y = NULL) +
    theme_void(base_size = 12) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
}

build_algorithm_panel <- function(dt, algorithm_name, x_var, y_var, x_label, y_label, x_breaks,
                                  y_limits = NULL, log_y = FALSE, show_x = TRUE, show_y = TRUE) {
  alg_dt <- dt[algorithm == algorithm_name]
  alg_title <- switch(
    algorithm_name,
    plssvd = "plssvd",
    simpls = "simpls",
    simpls_fast = "simpls-fast",
    algorithm_name
  )
  if (!nrow(alg_dt)) {
    return(empty_panel_plot(alg_title))
  }

  p <- ggplot(
    alg_dt,
    aes(
      x = .data[[x_var]],
      y = .data[[y_var]],
      color = svd_family,
      shape = engine_shape,
      linetype = engine_shape,
      group = line_id
    )
  ) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 1.8) +
    labs(
      title = alg_title,
      x = if (isTRUE(show_x)) x_label else NULL,
      y = if (isTRUE(show_y)) y_label else NULL,
      color = "SVD/backend",
      shape = "Implementation",
      linetype = "Implementation"
    ) +
    scale_color_manual(values = svd_palette, breaks = svd_levels, drop = FALSE) +
    scale_shape_manual(values = shape_values, breaks = shape_levels, drop = FALSE) +
    scale_linetype_manual(values = linetype_values, breaks = shape_levels, drop = FALSE) +
    guides(
      color = guide_legend(order = 1, nrow = 3, byrow = TRUE),
      shape = guide_legend(order = 2, nrow = 3, byrow = TRUE),
      linetype = "none"
    ) +
    base_theme() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.title.x = if (isTRUE(show_x)) element_text() else element_blank(),
      axis.text.x = if (isTRUE(show_x)) element_text(angle = 0, hjust = 0.5) else element_blank(),
      axis.ticks.x = if (isTRUE(show_x)) element_line() else element_blank(),
      axis.title.y = if (isTRUE(show_y)) element_text() else element_blank(),
      axis.text.y = if (isTRUE(show_y)) element_text() else element_blank(),
      axis.ticks.y = if (isTRUE(show_y)) element_line() else element_blank()
    )

  if (is.factor(alg_dt[[x_var]]) || is.character(alg_dt[[x_var]])) {
    p <- p + scale_x_discrete(drop = FALSE, limits = x_breaks)
    if (isTRUE(show_x)) {
      p <- p + theme(axis.text.x = element_text(angle = 45, hjust = 1))
    }
  } else {
    p <- p + scale_x_continuous(breaks = x_breaks)
  }

  if (!is.null(y_limits)) {
    if (isTRUE(log_y) && all(is.finite(y_limits)) && all(y_limits > 0)) {
      p <- p + scale_y_log10(limits = y_limits)
    } else {
      p <- p + coord_cartesian(ylim = y_limits)
    }
  } else if (isTRUE(log_y) && all(alg_dt[[y_var]] > 0, na.rm = TRUE)) {
    p <- p + scale_y_log10()
  }

  p
}

build_task_panel <- function(dt, family_name, x_var, x_label, perf_label, filename_prefix, subset_label = NULL) {
  family_dt <- dt[scenario_family == family_name]
  if (!nrow(family_dt) || data.table::uniqueN(family_dt[[x_var]]) <= 1L) {
    message("Skipping panel plots for ", family_name, if (!is.null(subset_label)) paste0(" (", subset_label, ")") else "", " because data are insufficient.")
    return(invisible(FALSE))
  }

  algorithms <- c("plssvd", "simpls", "simpls_fast")

  panel_groups <- if (identical(x_var, "noise_regime")) {
    list(all = family_dt)
  } else {
    noise_values <- unique(as.character(family_dt$noise_regime))
    noise_values <- noise_values[order(match(noise_values, smoke_noise_levels()))]
    setNames(lapply(noise_values, function(noise_value) {
      family_dt[as.character(noise_regime) == noise_value]
    }), noise_values)
  }

  for (group_name in names(panel_groups)) {
    sub <- panel_groups[[group_name]]
    if (!nrow(sub) || data.table::uniqueN(sub[[x_var]]) <= 1L) next

    x_values <- sub[[x_var]]
    if (is.factor(x_values) || is.character(x_values)) {
      x_breaks <- unique(as.character(x_values))
      x_breaks <- x_breaks[order(match(x_breaks, smoke_noise_levels()))]
      sub[[x_var]] <- factor(as.character(sub[[x_var]]), levels = x_breaks, ordered = TRUE)
    } else {
      x_breaks <- sort(unique(x_values))
    }
    time_limits <- expand_limits_if_needed(sub[train_ms_median > 0, train_ms_median])
    perf_limits <- expand_limits_if_needed(sub$metric_median)

    top_row <- wrap_plots(lapply(seq_along(algorithms), function(i) {
      build_algorithm_panel(
        sub,
        algorithm_name = algorithms[[i]],
        x_var = x_var,
        y_var = "train_ms_median",
        x_label = x_label,
        y_label = if (i == 1L) "Median train time (ms)" else "",
        x_breaks = x_breaks,
        y_limits = time_limits,
        log_y = TRUE,
        show_x = FALSE,
        show_y = (i == 1L)
      )
    }), ncol = 3)

    bottom_row <- wrap_plots(lapply(seq_along(algorithms), function(i) {
      build_algorithm_panel(
        sub,
        algorithm_name = algorithms[[i]],
        x_var = x_var,
        y_var = "metric_median",
        x_label = x_label,
        y_label = if (i == 1L) perf_label else "",
        x_breaks = x_breaks,
        y_limits = perf_limits,
        log_y = FALSE,
        show_x = TRUE,
        show_y = (i == 1L)
      )
    }), ncol = 3)

    panel_title <- panel_title_text(sub, family_name, x_var, subset_label = subset_label, group_name = group_name)

    panel_plot <- (top_row / bottom_row) +
      plot_layout(guides = "collect", heights = c(1, 1)) +
      plot_annotation(title = panel_title) &
      theme(legend.position = "bottom")

    suffix <- if (identical(group_name, "all")) "" else paste0("_", group_name)
    save_panel_plot(panel_plot, paste0(filename_prefix, suffix, ".png"))
  }

  invisible(TRUE)
}

reg_sample <- sum_dt[scenario_family %in% c("sim_reg_n_p50", "sim_reg_n_p500", "sim_reg_n_p1000_q1000_ncomp500") & metric_name == "Q2"]
make_line_plot(
  reg_sample,
  x_var = "n_train",
  y_var = "train_ms_median",
  x_label = "n_train",
  y_label = "Median train time (ms)",
  title = "Regression sample sweep time",
  facet_formula = noise_regime ~ scenario_family,
  filename = "regression_sample_sweep_time.png"
)
make_line_plot(
  reg_sample,
  x_var = "n_train",
  y_var = "metric_median",
  x_label = "n_train",
  y_label = "Median Q2",
  title = "Regression sample sweep performance",
  facet_formula = noise_regime ~ scenario_family,
  filename = "regression_sample_sweep_performance.png"
)

reg_p_full <- sum_dt[scenario_family == "sim_reg_p_sweep" & metric_name == "Q2"]
make_line_plot(
  reg_p_full,
  x_var = "p",
  y_var = "train_ms_median",
  x_label = "p",
  y_label = "Median train time (ms)",
  title = "Regression X-variable sweep time",
  facet_formula = noise_regime ~ .,
  filename = "regression_xvar_sweep_time.png"
)
make_line_plot(
  reg_p_full,
  x_var = "p",
  y_var = "metric_median",
  x_label = "p",
  y_label = "Median Q2",
  title = "Regression X-variable sweep performance",
  facet_formula = noise_regime ~ .,
  filename = "regression_xvar_sweep_performance.png"
)

reg_p_fair <- fair_dt[scenario_family == "sim_reg_p_sweep" & metric_name == "Q2" & p >= 50]
make_line_plot(
  reg_p_fair,
  x_var = "p",
  y_var = "train_ms_median",
  x_label = "p",
  y_label = "Median train time (ms)",
  title = "Regression X-variable sweep time (fair50)",
  facet_formula = noise_regime ~ .,
  filename = "regression_xvar_sweep_time_fair50.png"
)
make_line_plot(
  reg_p_fair,
  x_var = "p",
  y_var = "metric_median",
  x_label = "p",
  y_label = "Median Q2",
  title = "Regression X-variable sweep performance (fair50)",
  facet_formula = noise_regime ~ .,
  filename = "regression_xvar_sweep_performance_fair50.png"
)

reg_q_full <- sum_dt[scenario_family == "sim_reg_q_sweep" & metric_name == "Q2"]
make_line_plot(
  reg_q_full,
  x_var = "q",
  y_var = "train_ms_median",
  x_label = "q",
  y_label = "Median train time (ms)",
  title = "Regression Y-variable sweep time",
  facet_formula = noise_regime ~ .,
  filename = "regression_yvar_sweep_time.png"
)
make_line_plot(
  reg_q_full,
  x_var = "q",
  y_var = "metric_median",
  x_label = "q",
  y_label = "Median Q2",
  title = "Regression Y-variable sweep performance",
  facet_formula = noise_regime ~ .,
  filename = "regression_yvar_sweep_performance.png"
)

reg_q_fair <- fair_dt[scenario_family == "sim_reg_q_sweep" & metric_name == "Q2" & q >= 50]
make_line_plot(
  reg_q_fair,
  x_var = "q",
  y_var = "train_ms_median",
  x_label = "q",
  y_label = "Median train time (ms)",
  title = "Regression Y-variable sweep time (fair50)",
  facet_formula = noise_regime ~ .,
  filename = "regression_yvar_sweep_time_fair50.png"
)
make_line_plot(
  reg_q_fair,
  x_var = "q",
  y_var = "metric_median",
  x_label = "q",
  y_label = "Median Q2",
  title = "Regression Y-variable sweep performance (fair50)",
  facet_formula = noise_regime ~ .,
  filename = "regression_yvar_sweep_performance_fair50.png"
)

reg_noise <- sum_dt[scenario_family == "sim_reg_noise_sweep" & metric_name == "Q2"]
make_line_plot(
  reg_noise,
  x_var = "noise_regime",
  y_var = "train_ms_median",
  x_label = "noise_regime",
  y_label = "Median train time (ms)",
  title = "Regression noise sweep time",
  facet_formula = . ~ .,
  filename = "regression_noise_sweep_time.png"
)
make_line_plot(
  reg_noise,
  x_var = "noise_regime",
  y_var = "metric_median",
  x_label = "noise_regime",
  y_label = "Median Q2",
  title = "Regression noise sweep performance",
  facet_formula = . ~ .,
  filename = "regression_noise_sweep_performance.png"
)

plot_task_family(sum_dt[metric_name == "Q2"], "sim_reg_n_p50", "n_train",
  "task_sim_reg_n_p50_time.png", "task_sim_reg_n_p50_performance.png", "n_train", "Median Q2")
plot_task_family(sum_dt[metric_name == "Q2"], "sim_reg_n_p500", "n_train",
  "task_sim_reg_n_p500_time.png", "task_sim_reg_n_p500_performance.png", "n_train", "Median Q2")
plot_task_family(sum_dt[metric_name == "Q2"], "sim_reg_n_p1000_q1000_ncomp500", "n_train",
  "task_sim_reg_n_p1000_q1000_ncomp500_time.png", "task_sim_reg_n_p1000_q1000_ncomp500_performance.png", "n_train", "Median Q2")
plot_task_family(sum_dt[metric_name == "Q2"], "sim_reg_p_sweep", "p",
  "task_sim_reg_p_sweep_time.png", "task_sim_reg_p_sweep_performance.png", "p", "Median Q2")
plot_task_family(fair_dt[metric_name == "Q2" & p >= 50], "sim_reg_p_sweep", "p",
  "task_sim_reg_p_sweep_time_fair50.png", "task_sim_reg_p_sweep_performance_fair50.png", "p", "Median Q2", fair = TRUE)
plot_task_family(sum_dt[metric_name == "Q2"], "sim_reg_q_sweep", "q",
  "task_sim_reg_q_sweep_time.png", "task_sim_reg_q_sweep_performance.png", "q", "Median Q2")
plot_task_family(fair_dt[metric_name == "Q2" & q >= 50], "sim_reg_q_sweep", "q",
  "task_sim_reg_q_sweep_time_fair50.png", "task_sim_reg_q_sweep_performance_fair50.png", "q", "Median Q2", fair = TRUE)
build_task_panel(sum_dt[metric_name == "Q2"], "sim_reg_n_p50", "n_train", "n_train", "Median Q2", "panel_sim_reg_n_p50")
build_task_panel(sum_dt[metric_name == "Q2"], "sim_reg_n_p500", "n_train", "n_train", "Median Q2", "panel_sim_reg_n_p500")
build_task_panel(sum_dt[metric_name == "Q2"], "sim_reg_n_p1000_q1000_ncomp500", "n_train", "n_train", "Median Q2", "panel_sim_reg_n_p1000_q1000_ncomp500")
build_task_panel(sum_dt[metric_name == "Q2"], "sim_reg_p_sweep", "p", "p", "Median Q2", "panel_sim_reg_p_sweep_full")
build_task_panel(fair_dt[metric_name == "Q2" & p >= 50], "sim_reg_p_sweep", "p", "p", "Median Q2", "panel_sim_reg_p_sweep_fair50", subset_label = "fair50")
build_task_panel(sum_dt[metric_name == "Q2"], "sim_reg_q_sweep", "q", "q", "Median Q2", "panel_sim_reg_q_sweep_full")
build_task_panel(fair_dt[metric_name == "Q2" & q >= 50], "sim_reg_q_sweep", "q", "q", "Median Q2", "panel_sim_reg_q_sweep_fair50", subset_label = "fair50")
build_task_panel(sum_dt[metric_name == "Q2"], "sim_reg_noise_sweep", "noise_regime", "noise_regime", "Median Q2", "panel_sim_reg_noise_sweep")

message("Synthetic smoke plots written to: ", plot_dir)
