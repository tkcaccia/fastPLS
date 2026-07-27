#!/usr/bin/env Rscript

root <- normalizePath(
  Sys.getenv(
    "FASTPLS_REPO_ROOT",
    "/Users/stefano/Documents/GPUPLS/fastPLS_fresh_github"
  ),
  winslash = "/",
  mustWork = TRUE
)
out <- file.path(
  root,
  "benchmark_results/manuscript_revision_cycle46_20260726"
)
dir.create(out, recursive = TRUE, showWarnings = FALSE)

selected <- utils::read.csv(
  file.path(
    root,
    "benchmark_results/manuscript_revision_cycle20_20260725",
    "paired_backend_selected_summary.csv"
  ),
  stringsAsFactors = FALSE
)
metal <- utils::read.csv(
  file.path(
    root,
    "benchmark_results/metal_validation_20260726/summary",
    "metal_backend_paired.csv"
  ),
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
metal_datasets <- c("cifar100", "metref", "retina", "tabula")
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
backend_pch <- c(CPU = 21, CUDA = 24, Metal = 22)
backend_offset <- c(CPU = -0.14, CUDA = 0.14, Metal = 0.14)

wilson <- function(successes, total, z = 1.95996398454005) {
  if (!is.finite(successes) || !is.finite(total) || total <= 0) {
    return(c(NA_real_, NA_real_))
  }
  p <- successes / total
  denominator <- 1 + z^2 / total
  centre <- (p + z^2 / (2 * total)) / denominator
  half <- z * sqrt(p * (1 - p) / total + z^2 / (4 * total^2)) / denominator
  c(max(0, centre - half), min(1, centre + half))
}

draw_panel <- function(d, dataset, backends, title_suffix = "") {
  ok <- d[d$status == "ok" & is.finite(d$point_estimate), , drop = FALSE]
  if (!nrow(ok)) {
    graphics::plot.new()
    graphics::title(main = paste0(dataset_labels[[dataset]], title_suffix))
    graphics::text(0.5, 0.5, "No evaluated route", col = "#777777")
    return(invisible(NULL))
  }
  limits <- range(c(ok$ci_lower, ok$ci_upper), finite = TRUE)
  metric <- ok$metric_name[[1L]]
  if (metric == "accuracy") {
    ylim <- c(max(0, limits[[1L]] - 0.04), min(1.02, limits[[2L]] + 0.04))
    ylab <- "Accuracy (95% CI)"
  } else {
    padding <- max(diff(limits) * 0.2, max(abs(limits)) * 0.04, 1e-06)
    ylim <- limits + c(-padding, padding)
    ylab <- "RMSD (95% CI)"
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
    cex.lab = 0.72,
    cex.axis = 0.68
  )
  graphics::abline(h = graphics::axTicks(2), col = "#E2E2E2", lwd = 0.7)
  for (method_index in seq_along(method_order)) {
    method <- method_order[[method_index]]
    pair <- d[d$method_panel == method & d$engine %in% backends, , drop = FALSE]
    pair <- pair[match(backends, pair$engine), , drop = FALSE]
    pair <- pair[!is.na(pair$engine) & pair$status == "ok", , drop = FALSE]
    if (!nrow(pair)) {
      graphics::text(method_index, mean(ylim), "NE", col = "#777777", cex = 0.58)
      next
    }
    x <- method_index + backend_offset[pair$engine]
    if (nrow(pair) == 2L) {
      graphics::segments(
        x[[1L]], pair$point_estimate[[1L]],
        x[[2L]], pair$point_estimate[[2L]],
        col = "#555555", lwd = 0.9
      )
    }
    graphics::arrows(
      x, pair$ci_lower, x, pair$ci_upper,
      angle = 90, code = 3, length = 0.03, lwd = 0.9,
      col = method_colors[[method]]
    )
    graphics::points(
      x, pair$point_estimate,
      pch = backend_pch[pair$engine],
      cex = 1.05,
      bg = method_colors[[method]],
      col = "black",
      lwd = 0.65
    )
  }
  graphics::axis(
    1,
    at = seq_along(method_order),
    labels = method_labels[method_order],
    las = 2,
    cex.axis = 0.58
  )
  graphics::title(
    main = paste0(dataset_labels[[dataset]], title_suffix),
    font.main = 2,
    cex.main = 0.78
  )
  graphics::box(bty = "l")
}

# Convert the matched CPU/Metal validation archive to the selected-figure schema.
metal <- metal[
  metal$dataset %in% metal_datasets &
    metal$precision == "float64" &
    metal$classifier == "argmax" &
    metal$svd_method == "rsvd",
  ,
  drop = FALSE
]
metal <- metal[
  !duplicated(metal[c("dataset", "method", "ncomp")]),
  ,
  drop = FALSE
]
metal_rows <- list()
row_index <- 1L
for (i in seq_len(nrow(metal))) {
  row <- metal[i, ]
  for (backend in c("CPU", "Metal")) {
    suffix <- tolower(backend)
    point <- row[[paste0("median_metric_", suffix)]]
    n_test <- row$n_test
    interval <- wilson(round(point * n_test), n_test)
    metal_rows[[row_index]] <- data.frame(
      dataset = row$dataset,
      method_panel = row$method,
      engine = backend,
      status = "ok",
      metric_name = "accuracy",
      point_estimate = point,
      ci_lower = interval[[1L]],
      ci_upper = interval[[2L]],
      effective_ncomp = row$ncomp,
      stringsAsFactors = FALSE
    )
    row_index <- row_index + 1L
  }
}
metal_plot <- do.call(rbind, metal_rows)
utils::write.csv(
  metal_plot,
  file.path(out, "metal_matched_points_used_in_figure.csv"),
  row.names = FALSE
)

draw <- function(device) {
  device()
  old <- graphics::par(
    mfrow = c(4, 4),
    mar = c(4.0, 3.8, 2.0, 0.6),
    oma = c(5.0, 0.8, 5.2, 0.5),
    mgp = c(2.1, 0.58, 0),
    tcl = -0.22,
    family = "sans"
  )
  on.exit({
    graphics::par(old)
    grDevices::dev.off()
  })
  for (dataset in names(dataset_labels)) {
    d <- selected[selected$dataset == dataset, , drop = FALSE]
    draw_panel(d, dataset, c("CPU", "CUDA"))
  }
  for (dataset in metal_datasets) {
    d <- metal_plot[metal_plot$dataset == dataset, , drop = FALSE]
    components <- unique(d$effective_ncomp)
    suffix <- if (length(components)) paste0(" (Metal validation, A=", components[[1L]], ")") else ""
    draw_panel(d, dataset, c("CPU", "Metal"), suffix)
  }
  graphics::mtext(
    "Backend performance: selected CPU/CUDA benchmark and matched Metal validation",
    side = 3,
    outer = TRUE,
    line = 2.6,
    font = 2,
    cex = 1.0
  )
  graphics::mtext(
    "Rows 1-3: family-specific training-selected CPU/CUDA settings. Row 4: separately prespecified CPU/Metal validation settings; NE = not evaluated.",
    side = 3,
    outer = TRUE,
    line = 1.35,
    cex = 0.66
  )
  graphics::mtext(
    "Segments join matched backends within a row and setting. Error bars are conditional 95% intervals on the fixed held-out set.",
    side = 1,
    outer = TRUE,
    line = 1.8,
    cex = 0.62
  )
  graphics::par(fig = c(0, 1, 0, 1), new = TRUE, mar = rep(0, 4))
  graphics::plot.new()
  graphics::legend(
    "bottom",
    inset = c(0, 0.004),
    legend = c("CPU", "CUDA", "Metal", "NE: not evaluated"),
    pch = c(21, 24, 22, NA),
    pt.bg = c("#777777", "#777777", "#777777", NA),
    col = c("black", "black", "black", "#777777"),
    bty = "n",
    horiz = TRUE,
    cex = 0.68,
    xpd = NA
  )
}

draw(function() {
  grDevices::png(
    file.path(out, "supp_cpu_cuda_metal_outer_test.png"),
    width = 4400,
    height = 3900,
    res = 320
  )
})
draw(function() {
  grDevices::pdf(
    file.path(out, "supp_cpu_cuda_metal_outer_test.pdf"),
    width = 13.75,
    height = 12.2
  )
})
