#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1L) args[[1L]] else
  "benchmark_results/manuscript_revision_cycle13_20260725"
out_file <- if (length(args) >= 2L) args[[2L]] else
  file.path(root, "main_benchmark_opls_kernel_settings.csv")

read_if_present <- function(path) {
  if (!file.exists(path)) return(NULL)
  utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

primary <- read_if_present(file.path(root, "selected_backend_with_uncertainty.csv"))
retina_tabula <- read_if_present(file.path(
  root, "retina_tabula_selected_outer", "multidataset_selected_raw.csv"
))
retina_tabula_opls <- read_if_present(file.path(
  root, "retina_tabula_selected_opls", "multidataset_selected_raw.csv"
))

keep_selected <- function(x) {
  if (is.null(x) || !nrow(x)) return(NULL)
  x <- x[x$method_panel %in% c("opls", "kernelpls"), , drop = FALSE]
  if (!nrow(x)) return(NULL)
  data.frame(
    dataset = as.character(x$dataset),
    method = as.character(x$method_panel),
    reported_total_ncomp = as.integer(x$requested_ncomp),
    stringsAsFactors = FALSE
  )
}

selected <- do.call(rbind, Filter(
  Negate(is.null),
  list(
    keep_selected(primary),
    keep_selected(retina_tabula),
    keep_selected(retina_tabula_opls)
  )
))
selected <- selected[selected$dataset != "singlecell", , drop = FALSE]
selected <- unique(selected)
selected <- selected[order(selected$dataset, selected$method), , drop = FALSE]

all_datasets <- sort(unique(c(selected$dataset, "nmr")))
grid <- expand.grid(
  dataset = all_datasets,
  method = c("opls", "kernelpls"),
  stringsAsFactors = FALSE
)
settings <- merge(grid, selected, by = c("dataset", "method"), all.x = TRUE)
settings$status <- ifelse(is.na(settings$reported_total_ncomp), "not_evaluated", "evaluated")
settings$predictive_ncomp <- ifelse(
  settings$method == "opls" & settings$status == "evaluated",
  pmax(1L, settings$reported_total_ncomp - 1L),
  ifelse(settings$method == "kernelpls", settings$reported_total_ncomp, NA_integer_)
)
settings$orthogonal_ncomp <- ifelse(
  settings$method == "opls" & settings$status == "evaluated",
  pmin(1L, pmax(0L, settings$reported_total_ncomp - 1L)),
  ifelse(settings$method == "kernelpls" & settings$status == "evaluated", 0L, NA_integer_)
)
settings$kernel_type <- ifelse(
  settings$method == "kernelpls" & settings$status == "evaluated",
  "linear",
  NA_character_
)
settings$gamma <- NA_real_
settings$degree <- NA_integer_
settings$coef0 <- NA_real_
settings$selection_scope <- ifelse(
  settings$status != "evaluated",
  "not evaluated",
  ifelse(
    settings$method == "opls",
    "training-only total-component selection; north=1 prespecified",
    "training-only component selection; linear kernel prespecified"
  )
)
settings$interpretation <- ifelse(
  settings$status != "evaluated",
  "not evaluated",
  ifelse(
    settings$method == "kernelpls",
  "linear-kernel implementation control; not a nonlinear-kernel result",
  "one prespecified orthogonal component; reported k is predictive plus orthogonal"
  )
)

dir.create(dirname(out_file), recursive = TRUE, showWarnings = FALSE)
utils::write.csv(settings, out_file, row.names = FALSE, na = "")
cat(normalizePath(out_file, winslash = "/", mustWork = TRUE), "\n")
