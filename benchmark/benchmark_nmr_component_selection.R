#!/usr/bin/env Rscript

# Repeated training-only NMR component selection. One maximal model is fitted
# per split; requested prefixes are scored in response blocks so no
# validation-prediction cube is materialized.

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default = NULL) {
  key <- paste0("--", name, "=")
  value <- args[startsWith(args, key)]
  if (!length(value)) return(default)
  sub(key, "", value[[1L]], fixed = TRUE)
}

input <- get_arg("input")
out_dir <- get_arg("out")
backend <- match.arg(get_arg("backend", "cuda"), c("cpu", "cuda"))
method <- match.arg(get_arg("method", "simpls"), c("simpls", "plssvd"))
seeds <- as.integer(strsplit(get_arg("seeds", "123,456,789,1011,2027"), ",", fixed = TRUE)[[1L]])
grid <- sort(unique(as.integer(strsplit(
  get_arg("grid", "1,2,3,5,8,10,25,50,75,100,125,150,165,175,200,250,300"),
  ",", fixed = TRUE
)[[1L]])))
validation_fraction <- as.numeric(get_arg("validation_fraction", "0.2"))
response_block_size <- as.integer(get_arg("response_block_size", "2048"))
fit_seed <- as.integer(get_arg("fit_seed", "123"))
oversample_arg <- get_arg("oversample", "auto")
power_arg <- get_arg("power", "auto")
automatic_controls <- identical(oversample_arg, "auto") &&
  identical(power_arg, "auto")
if (xor(identical(oversample_arg, "auto"), identical(power_arg, "auto"))) {
  stop("oversample and power must both be 'auto' or both be numeric.",
       call. = FALSE)
}
oversample <- if (automatic_controls) NA_integer_ else as.integer(oversample_arg)
power <- if (automatic_controls) NA_integer_ else as.integer(power_arg)

if (is.null(input) || is.null(out_dir)) {
  stop("Provide --input=NMR.RData and --out=RESULT_DIR.", call. = FALSE)
}
if (any(!is.finite(grid)) || min(grid) < 1L || any(diff(grid) <= 0L)) {
  stop("The component grid must contain increasing positive integers.", call. = FALSE)
}
if (!is.finite(validation_fraction) || validation_fraction <= 0 || validation_fraction >= 0.5) {
  stop("validation_fraction must be greater than zero and less than 0.5.", call. = FALSE)
}

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_arg)) {
  normalizePath(sub("^--file=", "", script_arg[[1L]]), mustWork = TRUE)
} else {
  normalizePath("benchmark/benchmark_nmr_component_selection.R", mustWork = TRUE)
}
source(file.path(dirname(script_path), "nmr_protocol_helpers.R"))

fastpls_lib <- Sys.getenv("FASTPLS_LIB", unset = "")
if (nzchar(fastpls_lib)) .libPaths(c(fastpls_lib, .libPaths()))
suppressPackageStartupMessages(library(fastPLS))
if (identical(backend, "cuda") && !isTRUE(has_cuda())) {
  stop("The selected fastPLS installation has no CUDA backend.", call. = FALSE)
}

protocol <- fastpls_nmr_protocol(input)
Xtrain <- protocol$Xtrain
Ytrain <- protocol$Ytrain
protocol_metadata <- protocol$metadata
rm(protocol)
gc(full = TRUE)
if (max(grid) >= floor((1 - validation_fraction) * nrow(Xtrain))) {
  stop("The maximum ncomp must be smaller than the inner-training sample count.", call. = FALSE)
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

score_prefix <- function(model, X_validation, Y_validation, k, block_size) {
  R <- as.matrix(model$R)
  Q <- as.matrix(model$Q)
  if (ncol(R) < k || ncol(Q) < k) {
    stop("The fitted model does not contain the requested component prefix.", call. = FALSE)
  }
  X_centered <- sweep(X_validation, 2L, as.numeric(model$mX), "-")
  scale_values <- as.numeric(model$vX)
  if (length(scale_values) == ncol(X_centered) && any(scale_values != 1)) {
    X_centered <- sweep(X_centered, 2L, scale_values, "/")
  }
  scores <- X_centered %*% R[, seq_len(k), drop = FALSE]
  response_mean <- as.numeric(model$mY)
  n_values <- length(Y_validation)
  squared_error <- 0
  absolute_error <- 0
  total_sum_squares <- 0

  for (first in seq.int(1L, ncol(Y_validation), by = block_size)) {
    last <- min(ncol(Y_validation), first + block_size - 1L)
    index <- first:last
    predicted <- scores %*% t(Q[index, seq_len(k), drop = FALSE])
    predicted <- sweep(predicted, 2L, response_mean[index], "+")
    residual <- Y_validation[, index, drop = FALSE] - predicted
    squared_error <- squared_error + sum(residual * residual)
    absolute_error <- absolute_error + sum(abs(residual))
    centered_observed <- sweep(
      Y_validation[, index, drop = FALSE], 2L, response_mean[index], "-"
    )
    total_sum_squares <- total_sum_squares + sum(centered_observed * centered_observed)
  }

  c(
    RMSD = sqrt(squared_error / n_values),
    MAE = absolute_error / n_values,
    Q2 = if (total_sum_squares > 0) 1 - squared_error / total_sum_squares else NA_real_
  )
}

raw_rows <- list()
split_rows <- list()
effective_controls <- NULL
for (split_index in seq_along(seeds)) {
  split_seed <- seeds[[split_index]]
  set.seed(split_seed)
  validation_index <- sort(sample(
    seq_len(nrow(Xtrain)), floor(validation_fraction * nrow(Xtrain))
  ))
  training_index <- setdiff(seq_len(nrow(Xtrain)), validation_index)
  message(sprintf(
    "[%s] split=%d/%d seed=%d fit method=%s ncomp=%d backend=%s",
    format(Sys.time(), "%Y-%m-%d %H:%M:%S"), split_index, length(seeds),
    split_seed, method, max(grid), backend
  ))

  gc(full = TRUE)
  fit_error <- NULL
  fit_time <- system.time({
    fit_arguments <- list(
        Xtrain[training_index, , drop = FALSE],
        Ytrain[training_index, , drop = FALSE],
        ncomp = max(grid), method = method, backend = backend,
        svd.method = "rsvd", scaling = "centering", fit = FALSE,
        return_variance = FALSE, seed = fit_seed
    )
    if (!automatic_controls) {
      fit_arguments$oversample <- oversample
      fit_arguments$power <- power
    }
    model <- tryCatch(
      do.call(pls, fit_arguments),
      error = function(e) {
        fit_error <<- conditionMessage(e)
        NULL
      }
    )
  })[["elapsed"]]

  if (is.null(model)) {
    raw_rows[[length(raw_rows) + 1L]] <- data.frame(
      split = split_index, split_seed = split_seed, ncomp = grid,
      RMSD = NA_real_, MAE = NA_real_, Q2 = NA_real_,
      fit_time_sec = fit_time, score_time_sec = NA_real_,
      status = "failed", error = fit_error, stringsAsFactors = FALSE
    )
    next
  }
  if (is.null(effective_controls)) {
    diagnostics <- model$diagnostics$rsvd %||% list()
    effective_controls <- list(
      profile = diagnostics$control_profile %||% "explicit",
      oversample = diagnostics$oversample %||% oversample,
      power = diagnostics$power %||% power
    )
  }

  split_result <- lapply(grid, function(k) {
    score_time <- system.time({
      metrics <- score_prefix(
        model, Xtrain[validation_index, , drop = FALSE],
        Ytrain[validation_index, , drop = FALSE], k, response_block_size
      )
    })[["elapsed"]]
    data.frame(
      split = split_index, split_seed = split_seed, ncomp = k,
      RMSD = unname(metrics[["RMSD"]]), MAE = unname(metrics[["MAE"]]),
      Q2 = unname(metrics[["Q2"]]), fit_time_sec = fit_time,
      score_time_sec = score_time, status = "ok", error = "",
      stringsAsFactors = FALSE
    )
  })
  split_result <- do.call(rbind, split_result)
  raw_rows[[length(raw_rows) + 1L]] <- split_result
  best <- split_result[which.min(split_result$RMSD), , drop = FALSE]
  split_rows[[length(split_rows) + 1L]] <- data.frame(
    split = split_index, split_seed = split_seed,
    n_inner_train = length(training_index),
    n_inner_validation = length(validation_index),
    best_ncomp = best$ncomp, best_RMSD = best$RMSD, best_Q2 = best$Q2,
    stringsAsFactors = FALSE
  )
  rm(model)
  gc(full = TRUE)
}

raw <- do.call(rbind, raw_rows)
ok <- raw[raw$status == "ok", , drop = FALSE]
if (!nrow(ok)) stop("All repeated component-selection fits failed.", call. = FALSE)

summary_rows <- lapply(grid, function(k) {
  values <- ok[ok$ncomp == k, , drop = FALSE]
  data.frame(
    ncomp = k, n_success = nrow(values),
    RMSD_mean = mean(values$RMSD),
    RMSD_se = if (nrow(values) > 1L) {
      stats::sd(values$RMSD) / sqrt(nrow(values))
    } else {
      NA_real_
    },
    RMSD_median = stats::median(values$RMSD),
    RMSD_q25 = unname(stats::quantile(values$RMSD, 0.25)),
    RMSD_q75 = unname(stats::quantile(values$RMSD, 0.75)),
    RMSD_min = min(values$RMSD), RMSD_max = max(values$RMSD),
    Q2_median = stats::median(values$Q2),
    Q2_q25 = unname(stats::quantile(values$Q2, 0.25)),
    Q2_q75 = unname(stats::quantile(values$Q2, 0.75)),
    stringsAsFactors = FALSE
  )
})
summary <- do.call(rbind, summary_rows)
minimum_index <- which.min(summary$RMSD_mean)
one_se_threshold <- summary$RMSD_mean[[minimum_index]] +
  summary$RMSD_se[[minimum_index]]
eligible <- summary$ncomp[summary$RMSD_mean <= one_se_threshold]
selected_ncomp <- min(eligible)
selected_index <- match(selected_ncomp, summary$ncomp)
summary$one_se_eligible <- summary$ncomp %in% eligible
lower_boundary_selected <- selected_index == 1L
upper_boundary_selected <- selected_index == nrow(summary)
last_relative_improvement <- if (nrow(summary) > 1L) {
  with(summary, (RMSD_median[nrow(summary) - 1L] - RMSD_median[nrow(summary)]) /
         RMSD_median[nrow(summary) - 1L])
} else {
  NA_real_
}

selected_by_split <- if (length(split_rows)) do.call(rbind, split_rows) else data.frame()
selection <- data.frame(
  selected_ncomp = selected_ncomp,
  minimum_mean_ncomp = summary$ncomp[[minimum_index]],
  one_se_threshold = one_se_threshold,
  eligible_ncomp = paste(eligible, collapse = ","),
  selection_rule = "smallest component count within one standard error of the minimum mean validation RMSD",
  lower_boundary_selected = lower_boundary_selected,
  upper_boundary_selected = upper_boundary_selected,
  largest_tested_ncomp = max(grid),
  last_relative_improvement = last_relative_improvement,
  n_splits_requested = length(seeds),
  n_splits_successful = length(unique(ok$split)),
  stringsAsFactors = FALSE
)

utils::write.csv(raw, file.path(out_dir, "nmr_component_selection_raw.csv"), row.names = FALSE)
utils::write.csv(summary, file.path(out_dir, "nmr_component_selection_summary.csv"), row.names = FALSE)
utils::write.csv(selected_by_split, file.path(out_dir, "nmr_component_selection_by_split.csv"), row.names = FALSE)
utils::write.csv(selection, file.path(out_dir, "nmr_component_selection_decision.csv"), row.names = FALSE)
write_fastpls_nmr_manifest(
  protocol_metadata, file.path(out_dir, "nmr_protocol_manifest.txt"),
  extra = list(
    selection_scope = paste(
      "Xtrain/Ytrain only; Xtest/Ytest verified by the protocol helper but",
      "not used for fitting, scoring, or selection"
    ),
    backend = backend,
    svd_method = "rsvd",
    method = method,
    precision = "float64",
    split_seeds = seeds,
    component_grid = grid,
    validation_fraction = validation_fraction,
    fit_seed = fit_seed,
    control_profile = effective_controls$profile,
    oversample = effective_controls$oversample,
    power = effective_controls$power,
    fastPLS_version = as.character(utils::packageVersion("fastPLS")),
    selected_ncomp = selected_ncomp,
    selected_at_lower_boundary = lower_boundary_selected,
    selected_at_upper_boundary = upper_boundary_selected
  )
)
saveRDS(
  list(
    protocol = protocol_metadata, raw = raw, summary = summary,
    selected_by_split = selected_by_split, decision = selection,
    session = utils::sessionInfo()
  ),
  file.path(out_dir, "nmr_extended_component_selection.rds")
)
print(selection)
