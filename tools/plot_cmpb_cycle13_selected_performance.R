#!/usr/bin/env Rscript

root <- "/Users/stefano/Documents/GPUPLS/fastPLS_fresh_github"
input <- file.path(
  root, "benchmark_results", "manuscript_revision_cycle13_20260725",
  "selected_backend_cycle13_chosen.csv"
)
output_dir <- file.path(
  root, "benchmark_results", "manuscript_revision_cycle13_20260725", "plots"
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

dataset_labels <- c(
  cbmc_citeseq = "CBMC CITE-seq", ccle = "CCLE", cifar100 = "CIFAR-100",
  gtex_v8 = "GTEx v8", metref = "MetRef", nmr = "NMR", prism = "PRISM",
  singlecell = "SingleCell", retina = "Retina", tabula = "Tabula Muris",
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

x <- utils::read.csv(input, check.names = FALSE)
x <- x[
  !x$dataset %in% c("imagenet", "singlecell") &
    x$status == "ok" & is.finite(x$metric_median),
  , drop = FALSE
]
x$metric_value <- x$metric_median
x$total_time_sec <- suppressWarnings(as.numeric(x$total_time_sec_median))

metric_label <- function(value) {
  switch(
    tolower(value),
    accuracy = "Accuracy (higher is better)",
    q2 = expression(Q^2 ~ "(higher is better)"),
    rmsd = "RMSD (lower is better)",
    value
  )
}

draw_figure <- function(device) {
  device()
  old <- graphics::par(
    mfrow = c(3, 4), mar = c(4.4, 4.1, 2.3, 0.7),
    oma = c(4.0, 0.8, 4.8, 0.5), mgp = c(2.25, 0.65, 0),
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
  datasets <- dataset_order[dataset_order %in% unique(x$dataset)]
  for (dataset in datasets) {
    d <- x[x$dataset == dataset, , drop = FALSE]
    d <- d[order(match(d$method_panel, method_order)), , drop = FALSE]
    values <- d$metric_value
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
      ylab = metric_label(metric), ylim = ylim, main = "",
      cex.lab = 0.82, cex.axis = 0.78
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
    labels <- sprintf("%s\nk=%d", value_labels, as.integer(d$effective_ncomp))
    graphics::text(position, values, labels, pos = 3, cex = 0.65, xpd = NA)
    graphics::axis(
      1, at = position, labels = method_labels[d$method_panel],
      las = 2, cex.axis = 0.67
    )
    graphics::title(
      main = unname(dataset_labels[dataset]), font.main = 2, cex.main = 0.92
    )
    graphics::box(bty = "l")
  }
  for (unused in seq_len(max(0, 12L - length(datasets)))) graphics::plot.new()
  graphics::mtext(
    "Outer-test predictive performance at the training-selected component count",
    side = 3, outer = TRUE, line = 2.1, font = 2, cex = 1.15
  )
  graphics::mtext(
    "Points show the fastest completed matched CPU/CUDA rSVD row within each PLS family; k is the effective component count.",
    side = 1, outer = TRUE, line = 1.35, cex = 0.72
  )
}

draw_figure(function() {
  grDevices::png(
    file.path(output_dir, "selected_performance_all_datasets.png"),
    width = 4200, height = 3000, res = 320
  )
})
draw_figure(function() {
  grDevices::pdf(
    file.path(output_dir, "selected_performance_all_datasets.pdf"),
    width = 13.1, height = 9.4
  )
})
