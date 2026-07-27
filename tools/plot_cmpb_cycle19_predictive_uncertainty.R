#!/usr/bin/env Rscript

root <- "/Users/stefano/Documents/GPUPLS/fastPLS_fresh_github"
evidence <- file.path(
  root,
  "benchmark_results/manuscript_revision_cycle19_20260725"
)
plot_dir <- file.path(evidence, "plots")
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

selected <- utils::read.csv(
  file.path(
    root,
    "benchmark_results/manuscript_revision_cycle17_20260725",
    "selected_backend_cycle17_chosen.csv"
  ),
  check.names = FALSE
)
uncertainty <- utils::read.csv(
  file.path(evidence, "selected_predictive_uncertainty.csv"),
  check.names = FALSE
)
selected <- merge(
  selected,
  uncertainty[
    ,
    c(
      "dataset", "method_panel", "ci_lower", "ci_upper",
      "ci_method", "n_test"
    )
  ],
  by = c("dataset", "method_panel"),
  all.x = TRUE
)
selected <- selected[
  !selected$dataset %in% c("imagenet", "singlecell") &
    selected$status == "ok" &
    is.finite(selected$metric_median),
  ,
  drop = FALSE
]

dataset_labels <- c(
  metref = "MetRef",
  ccle = "CCLE",
  tcga_brca = "TCGA-BRCA",
  tcga_hnsc_methylation = "TCGA-HNSC methylation",
  gtex_v8 = "GTEx v8",
  tcga_pan_cancer = "TCGA Pan-Cancer",
  retina = "Retina",
  tabula = "Tabula Muris",
  cifar100 = "CIFAR-100",
  cbmc_citeseq = "CBMC CITE-seq",
  prism = "PRISM",
  nmr = "NMR"
)
method_order <- c("plssvd", "simpls", "opls", "kernelpls")
method_labels <- c(
  plssvd = "PLS-SVD",
  simpls = "SIMPLS",
  opls = "OPLS",
  kernelpls = "kernel PLS"
)
method_colors <- c(
  plssvd = "#0072B2",
  simpls = "#D55E00",
  opls = "#009E73",
  kernelpls = "#CC79A7"
)

draw <- function(device) {
  device()
  old <- graphics::par(
    mfrow = c(3, 4),
    mar = c(4.4, 4.1, 2.3, 0.7),
    oma = c(4.4, 0.8, 5, 0.5),
    mgp = c(2.25, 0.65, 0),
    tcl = -0.25,
    family = "sans"
  )
  on.exit({
    graphics::par(old)
    grDevices::dev.off()
  })

  dataset_order <- names(dataset_labels)
  for (dataset in dataset_order) {
    d <- selected[selected$dataset == dataset, , drop = FALSE]
    d <- d[order(match(d$method_panel, method_order)), , drop = FALSE]
    values <- d$metric_median
    limits <- range(c(d$ci_lower, d$ci_upper), finite = TRUE)
    if (d$metric_name[[1L]] == "accuracy") {
      padding <- 0.04
      ylim <- c(max(0, limits[[1L]] - padding), min(1.02, limits[[2L]] + padding))
      ylab <- "Accuracy (95% CI)"
    } else {
      padding <- max(diff(limits) * 0.2, max(abs(limits)) * 0.04, 1e-06)
      ylim <- limits + c(-padding, padding)
      ylab <- "RMSD (95% bootstrap CI)"
    }
    position <- seq_len(nrow(d))
    graphics::plot(
      position,
      values,
      type = "n",
      xaxt = "n",
      xlab = "",
      ylab = ylab,
      ylim = ylim,
      cex.lab = 0.80,
      cex.axis = 0.76
    )
    graphics::abline(h = graphics::axTicks(2), col = "#E2E2E2", lwd = 0.7)
    graphics::arrows(
      position,
      d$ci_lower,
      position,
      d$ci_upper,
      angle = 90,
      code = 3,
      length = 0.04,
      lwd = 1.1,
      col = method_colors[d$method_panel]
    )
    graphics::points(
      position,
      values,
      pch = 21,
      cex = 1.35,
      bg = method_colors[d$method_panel],
      col = "black",
      lwd = 0.7
    )
    marker <- ifelse(d$selection_status == "interior tested value", "", "*")
    labels <- if (d$metric_name[[1L]] == "rmsd" && max(abs(values)) < 0.01) {
      format(values, scientific = TRUE, digits = 3)
    } else {
      sprintf("%.3f", values)
    }
    graphics::text(
      position,
      d$ci_upper,
      sprintf("%s\nk=%d%s", labels, as.integer(d$effective_ncomp), marker),
      pos = 3,
      cex = 0.61,
      xpd = NA
    )
    graphics::axis(
      1,
      at = position,
      labels = method_labels[d$method_panel],
      las = 2,
      cex.axis = 0.66
    )
    graphics::title(
      main = unname(dataset_labels[[dataset]]),
      font.main = 2,
      cex.main = 0.90
    )
    graphics::box(bty = "l")
  }
  graphics::mtext(
    "Outer-test performance at the family-specific training-selected setting",
    side = 3,
    outer = TRUE,
    line = 2.1,
    font = 2,
    cex = 1.10
  )
  graphics::mtext(
    "Accuracy: Wilson 95% CI; RMSD: 10,000-resample held-out-sample bootstrap CI.",
    side = 1,
    outer = TRUE,
    line = 1.4,
    cex = 0.70
  )
}

draw(function() {
  grDevices::png(
    file.path(plot_dir, "selected_performance_all_datasets_with_ci.png"),
    width = 4200,
    height = 3000,
    res = 320
  )
})
draw(function() {
  grDevices::pdf(
    file.path(plot_dir, "selected_performance_all_datasets_with_ci.pdf"),
    width = 13.1,
    height = 9.4
  )
})
