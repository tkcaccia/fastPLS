#!/usr/bin/env Rscript

root <- normalizePath(
  Sys.getenv(
    "FASTPLS_REPO_ROOT",
    "/Users/stefano/Documents/GPUPLS/fastPLS_fresh_github"
  ),
  winslash = "/",
  mustWork = TRUE
)
evidence <- file.path(
  root,
  "benchmark_results/manuscript_revision_cycle20_20260725"
)
plot_dir <- file.path(evidence, "plots")
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

paired <- utils::read.csv(
  file.path(evidence, "paired_backend_selected_summary.csv"),
  stringsAsFactors = FALSE
)
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
engine_offset <- c(CPU = -0.12, CUDA = 0.12)
engine_pch <- c(CPU = 21, CUDA = 24)

draw <- function(device) {
  device()
  old <- graphics::par(
    mfrow = c(3, 4),
    mar = c(4.5, 4.2, 2.2, 0.7),
    oma = c(5.1, 0.8, 4.7, 0.5),
    mgp = c(2.3, 0.65, 0),
    tcl = -0.25,
    family = "sans"
  )
  on.exit({
    graphics::par(old)
    grDevices::dev.off()
  })

  for (dataset in names(dataset_labels)) {
    d <- paired[paired$dataset == dataset, , drop = FALSE]
    ok <- d[d$status == "ok" & is.finite(d$point_estimate), , drop = FALSE]
    limits <- range(c(ok$ci_lower, ok$ci_upper), finite = TRUE)
    metric <- ok$metric_name[[1L]]
    if (metric == "accuracy") {
      padding <- 0.04
      ylim <- c(
        max(0, limits[[1L]] - padding),
        min(1.02, limits[[2L]] + padding)
      )
      ylab <- "Accuracy (95% CI)"
    } else {
      padding <- max(
        diff(limits) * 0.2,
        max(abs(limits)) * 0.04,
        1e-06
      )
      ylim <- limits + c(-padding, padding)
      ylab <- "RMSD (95% bootstrap CI)"
    }

    graphics::plot(
      seq_along(method_order),
      rep(mean(ylim), length(method_order)),
      type = "n",
      xaxt = "n",
      xlab = "",
      ylab = ylab,
      ylim = ylim,
      xlim = c(0.55, 4.45),
      cex.lab = 0.8,
      cex.axis = 0.76
    )
    graphics::abline(h = graphics::axTicks(2), col = "#E2E2E2", lwd = 0.7)

    for (method_index in seq_along(method_order)) {
      method <- method_order[[method_index]]
      pair <- d[d$method_panel == method, , drop = FALSE]
      pair_ok <- pair[pair$status == "ok", , drop = FALSE]
      if (!nrow(pair_ok)) {
        graphics::text(
          method_index,
          mean(ylim),
          "NE",
          col = "#777777",
          cex = 0.65
        )
        next
      }
      pair_ok <- pair_ok[match(c("CPU", "CUDA"), pair_ok$engine), ]
      pair_ok <- pair_ok[!is.na(pair_ok$engine), , drop = FALSE]
      x <- method_index + engine_offset[pair_ok$engine]
      if (nrow(pair_ok) == 2L) {
        graphics::segments(
          x[[1L]],
          pair_ok$point_estimate[[1L]],
          x[[2L]],
          pair_ok$point_estimate[[2L]],
          col = "#555555",
          lwd = 0.9
        )
      }
      graphics::arrows(
        x,
        pair_ok$ci_lower,
        x,
        pair_ok$ci_upper,
        angle = 90,
        code = 3,
        length = 0.035,
        lwd = 1,
        col = method_colors[[method]]
      )
      graphics::points(
        x,
        pair_ok$point_estimate,
        pch = engine_pch[pair_ok$engine],
        cex = 1.25,
        bg = method_colors[[method]],
        col = "black",
        lwd = 0.7
      )
    }

    graphics::axis(
      1,
      at = seq_along(method_order),
      labels = method_labels[method_order],
      las = 2,
      cex.axis = 0.66
    )
    graphics::title(
      main = unname(dataset_labels[[dataset]]),
      font.main = 2,
      cex.main = 0.9
    )
    graphics::box(bty = "l")
  }

  graphics::mtext(
    "Matched CPU and CUDA performance at the family-specific training-selected setting",
    side = 3,
    outer = TRUE,
    line = 2,
    font = 2,
    cex = 1.05
  )
  graphics::mtext(
    "Segments join matched backends. Accuracy: Wilson 95% CI; RMSD: 10,000-resample held-out-sample bootstrap CI.",
    side = 1,
    outer = TRUE,
    line = 1.8,
    cex = 0.67
  )
  graphics::par(fig = c(0, 1, 0, 1), new = TRUE, mar = rep(0, 4))
  graphics::plot.new()
  graphics::legend(
    "bottom",
    inset = c(0, 0.006),
    legend = c("CPU", "CUDA", "NE: not evaluated"),
    pch = c(21, 24, NA),
    pt.bg = c("#777777", "#777777", NA),
    col = c("black", "black", "#777777"),
    bty = "n",
    horiz = TRUE,
    cex = 0.72,
    xpd = NA
  )
}

draw(function() {
  grDevices::png(
    file.path(plot_dir, "paired_backend_performance_all_datasets.png"),
    width = 4200,
    height = 3000,
    res = 320
  )
})
draw(function() {
  grDevices::pdf(
    file.path(plot_dir, "paired_backend_performance_all_datasets.pdf"),
    width = 13.1,
    height = 9.4
  )
})
