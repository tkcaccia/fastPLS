#!/usr/bin/env Rscript

root <- "/Users/stefano/Documents/GPUPLS/fastPLS_fresh_github"
evidence <- file.path(
  root, "benchmark_results", "manuscript_revision_cycle17_20260725"
)
plot_dir <- file.path(evidence, "plots")
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

selected <- utils::read.csv(
  file.path(evidence, "selected_backend_cycle17_chosen.csv"),
  check.names = FALSE
)
selected <- selected[
  !selected$dataset %in% c("imagenet", "singlecell") &
    selected$status == "ok" & is.finite(selected$metric_median),
  , drop = FALSE
]

dataset_labels <- c(
  cbmc_citeseq = "CBMC CITE-seq", ccle = "CCLE", cifar100 = "CIFAR-100",
  gtex_v8 = "GTEx v8", metref = "MetRef", nmr = "NMR", prism = "PRISM",
  retina = "Retina", tabula = "Tabula Muris",
  tcga_brca = "TCGA-BRCA",
  tcga_hnsc_methylation = "TCGA-HNSC methylation",
  tcga_pan_cancer = "TCGA Pan-Cancer"
)
method_order <- c("plssvd", "simpls", "opls", "kernelpls")
method_labels <- c(
  plssvd = "PLS-SVD", simpls = "SIMPLS", opls = "OPLS",
  kernelpls = "kernel PLS"
)
method_colors <- c(
  plssvd = "#0072B2", simpls = "#D55E00", opls = "#009E73",
  kernelpls = "#CC79A7"
)

metric_label <- function(value) {
  switch(
    tolower(value),
    accuracy = "Accuracy (higher is better)",
    q2 = expression(Q^2 ~ "(higher is better)"),
    rmsd = "RMSD (lower is better)",
    value
  )
}

draw_selected <- function(device) {
  device()
  old <- graphics::par(
    mfrow = c(3, 4), mar = c(4.4, 4.1, 2.3, 0.7),
    oma = c(4.4, 0.8, 5.0, 0.5), mgp = c(2.25, 0.65, 0),
    tcl = -0.25, family = "sans"
  )
  on.exit({
    graphics::par(old)
    grDevices::dev.off()
  })
  dataset_order <- c(
    "metref", "ccle", "tcga_brca", "tcga_hnsc_methylation",
    "gtex_v8", "tcga_pan_cancer", "retina", "tabula",
    "cifar100", "cbmc_citeseq", "prism", "nmr"
  )
  datasets <- dataset_order[dataset_order %in% unique(selected$dataset)]
  for (dataset in datasets) {
    d <- selected[selected$dataset == dataset, , drop = FALSE]
    d <- d[order(match(d$method_panel, method_order)), , drop = FALSE]
    values <- d$metric_median
    metric <- d$metric_name[[1L]]
    if (tolower(metric) == "accuracy") {
      ylim <- c(max(0, min(values) - 0.08), min(1.02, max(values) + 0.11))
    } else {
      padding <- max(diff(range(values)) * 0.35, max(abs(values)) * 0.08, 1e-6)
      ylim <- range(values) + c(-padding, padding)
    }
    position <- seq_len(nrow(d))
    graphics::plot(
      position, values, type = "n", xaxt = "n", xlab = "",
      ylab = metric_label(metric), ylim = ylim, cex.lab = 0.82,
      cex.axis = 0.78
    )
    graphics::abline(h = graphics::axTicks(2), col = "#E2E2E2", lwd = 0.7)
    graphics::points(
      position, values, pch = 21, cex = 1.45,
      bg = method_colors[d$method_panel], col = "black", lwd = 0.7
    )
    value_labels <- if (tolower(metric) == "rmsd" && max(abs(values)) < 0.01) {
      format(values, scientific = TRUE, digits = 3)
    } else {
      sprintf("%.3f", values)
    }
    marker <- ifelse(
      d$selection_status == "interior tested value", "", "*"
    )
    graphics::text(
      position, values,
      sprintf(
        "%s\nk=%d%s", value_labels, as.integer(d$effective_ncomp), marker
      ),
      pos = 3, cex = 0.65, xpd = NA
    )
    graphics::axis(
      1, at = position, labels = method_labels[d$method_panel],
      las = 2, cex.axis = 0.67
    )
    graphics::title(
      main = unname(dataset_labels[dataset]), font.main = 2, cex.main = 0.92
    )
    graphics::box(bty = "l")
  }
  graphics::mtext(
    "Outer-test performance at the training-selected value within each evaluated grid",
    side = 3, outer = TRUE, line = 2.1, font = 2, cex = 1.12
  )
  graphics::mtext(
    "* tested-grid boundary or response-rank limit; these values are not global optima.",
    side = 1, outer = TRUE, line = 1.4, cex = 0.72
  )
}

draw_selected(function() {
  grDevices::png(
    file.path(plot_dir, "selected_performance_all_datasets.png"),
    width = 4200, height = 3000, res = 320
  )
})
draw_selected(function() {
  grDevices::pdf(
    file.path(plot_dir, "selected_performance_all_datasets.pdf"),
    width = 13.1, height = 9.4
  )
})

plssvd_raw <- utils::read.csv(
  file.path(
    root, "benchmark_results", "manuscript_revision_cycle16_20260725",
    "nmr_plssvd_extended_lower_grid", "results",
    "nmr_component_selection_raw.csv"
  ),
  check.names = FALSE
)
simpls_raw <- utils::read.csv(
  file.path(
    root, "benchmark_results", "review_nmr_extended_selection_20260725",
    "nmr_component_selection_raw.csv"
  ),
  check.names = FALSE
)

summarize_selection <- function(raw) {
  ok <- raw[raw$status == "ok", , drop = FALSE]
  rows <- lapply(sort(unique(ok$ncomp)), function(k) {
    values <- ok$RMSD[ok$ncomp == k]
    data.frame(
      ncomp = k, mean = mean(values), se = stats::sd(values) / sqrt(length(values)),
      median = stats::median(values),
      q25 = unname(stats::quantile(values, 0.25)),
      q75 = unname(stats::quantile(values, 0.75))
    )
  })
  do.call(rbind, rows)
}

draw_one_se_panel <- function(raw, selected_k, label, color, log_x) {
  summary <- summarize_selection(raw)
  minimizing_k <- summary$ncomp[which.min(summary$mean)]
  threshold <- summary$mean[summary$ncomp == minimizing_k] +
    summary$se[summary$ncomp == minimizing_k]
  eligible <- summary$ncomp[summary$mean <= threshold]
  split_ids <- sort(unique(raw$split[raw$status == "ok"]))
  ylim <- range(raw$RMSD[raw$status == "ok"], finite = TRUE)
  graphics::plot(
    summary$ncomp, summary$mean, type = "n",
    log = if (log_x) "x" else "", xaxt = "n",
    xlab = "Number of components", ylab = "Validation RMSD",
    xlim = range(summary$ncomp), ylim = ylim, main = label
  )
  ticks <- summary$ncomp
  if (length(ticks) > 12L) {
    ticks <- ticks[c(1:6, 8, 10, 12, 14, length(ticks))]
  }
  graphics::axis(1, at = ticks, labels = ticks, cex.axis = 0.78)
  for (split_id in split_ids) {
    d <- raw[raw$split == split_id & raw$status == "ok", , drop = FALSE]
    d <- d[order(d$ncomp), , drop = FALSE]
    graphics::lines(d$ncomp, d$RMSD, col = "#B8B8B8", lwd = 0.8)
  }
  graphics::polygon(
    c(summary$ncomp, rev(summary$ncomp)),
    c(summary$mean - summary$se, rev(summary$mean + summary$se)),
    border = NA, col = grDevices::adjustcolor(color, alpha.f = 0.18)
  )
  graphics::abline(h = threshold, lty = 3, col = "#555555", lwd = 1.1)
  graphics::lines(summary$ncomp, summary$mean, col = color, lwd = 2.2)
  graphics::points(
    summary$ncomp, summary$mean, pch = 21, bg = color,
    col = "black", cex = 0.9
  )
  graphics::points(
    eligible, summary$mean[match(eligible, summary$ncomp)],
    pch = 22, bg = "#F0E442", col = "black", cex = 1.05
  )
  graphics::abline(v = selected_k, lty = 2, col = "#222222", lwd = 1.2)
  graphics::legend(
    "topright",
    legend = sprintf(
      "One-SE selection: k=%d\nEligible: %s",
      selected_k, paste(eligible, collapse = ", ")
    ),
    bty = "n", cex = 0.78
  )
  graphics::box(bty = "l")
}

draw_nmr_selection <- function(device) {
  device()
  old <- graphics::par(
    mfrow = c(1, 2), mar = c(4.3, 4.7, 3.2, 1.0),
    oma = c(1.0, 0.3, 2.5, 0.2), family = "sans",
    mgp = c(2.6, 0.75, 0), tcl = -0.25
  )
  on.exit({
    graphics::par(old)
    grDevices::dev.off()
  })
  draw_one_se_panel(
    plssvd_raw, 5L, "A  PLS-SVD (extended lower grid)", "#0072B2", TRUE
  )
  draw_one_se_panel(
    simpls_raw, 50L, "B  SIMPLS", "#D55E00", FALSE
  )
  graphics::mtext(
    "Repeated training-only NMR component selection",
    side = 3, outer = TRUE, font = 2, line = 0.8, cex = 1.1
  )
}

draw_nmr_selection(function() {
  grDevices::png(
    file.path(plot_dir, "nmr_component_selection_one_se.png"),
    width = 3000, height = 1450, res = 300
  )
})
draw_nmr_selection(function() {
  grDevices::pdf(
    file.path(plot_dir, "nmr_component_selection_one_se.pdf"),
    width = 10.0, height = 4.8
  )
})
