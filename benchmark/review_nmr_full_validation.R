#!/usr/bin/env Rscript

# NMR manuscript-review validation.  It keeps the held-out test split untouched
# during component selection, applies the requested predictor water mask to both
# training and test predictors, and records global, per-spectrum, and
# per-response prediction errors for a final selected-component model.

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default = NULL) {
  key <- paste0("--", name, "=")
  value <- args[startsWith(args, key)]
  if (!length(value)) return(default)
  sub(key, "", value[[1L]], fixed = TRUE)
}

input <- get_arg("input")
out_dir <- get_arg("out")
mode <- match.arg(get_arg("mode", "select"), c("select", "final"))
backend <- match.arg(get_arg("backend", "cpu"), c("cpu", "cuda"))
selected_ncomp <- as.integer(get_arg("selected_ncomp", "0"))
seed <- as.integer(get_arg("seed", "123"))
grid <- as.integer(strsplit(get_arg("grid", "10,25,50,75,100"), ",", fixed = TRUE)[[1L]])

if (is.null(input) || is.null(out_dir)) {
  stop("Provide --input=NMR.RData and --out=RESULT_DIR.", call. = FALSE)
}
if (identical(mode, "final") && (!is.finite(selected_ncomp) || selected_ncomp < 1L)) {
  stop("final mode requires --selected_ncomp=INTEGER.", call. = FALSE)
}

fastpls_lib <- Sys.getenv("FASTPLS_LIB", unset = "")
if (nzchar(fastpls_lib)) .libPaths(c(fastpls_lib, .libPaths()))
suppressPackageStartupMessages(library(fastPLS))
if (identical(backend, "cuda") && !isTRUE(has_cuda())) {
  stop("The selected fastPLS installation has no CUDA backend.", call. = FALSE)
}

data_env <- new.env(parent = emptyenv())
load(input, envir = data_env)
required <- c("Xtrain", "Ytrain", "Xtest", "Ytest")
if (!all(required %in% ls(data_env))) {
  stop("Input must contain Xtrain, Ytrain, Xtest, and Ytest.", call. = FALSE)
}
Xtrain <- as.matrix(get("Xtrain", envir = data_env))
Ytrain <- as.matrix(get("Ytrain", envir = data_env))
Xtest <- as.matrix(get("Xtest", envir = data_env))
Ytest <- as.matrix(get("Ytest", envir = data_env))

# The water window is a predictor-only spectral region.  Applying the same mask
# to both splits prevents information leakage and preserves the test protocol.
x_axis <- suppressWarnings(as.numeric(colnames(Xtrain)))
water_columns <- which(is.finite(x_axis) & x_axis > 4.6 & x_axis < 4.8)
if (length(water_columns)) {
  Xtrain[, water_columns] <- 0
  Xtest[, water_columns] <- 0
}

extract_prediction <- function(x, k) {
  key <- paste0("ncomp=", k)
  if (is.list(x) && !is.data.frame(x)) return(as.matrix(x[[key]] %||% x[[length(x)]]))
  if (length(dim(x)) == 3L) {
    names_k <- dimnames(x)[[3L]]
    idx <- match(key, names_k)
    if (is.na(idx)) idx <- dim(x)[[3L]]
    return(x[, , idx, drop = FALSE][, , 1L])
  }
  as.matrix(x)
}
`%||%` <- function(x, y) if (is.null(x)) y else x

run_fit_predict <- function(X_fit, Y_fit, X_eval, Y_eval, ncomp) {
  gc(full = TRUE)
  fit_time <- system.time({
    model <- pls(
      X_fit, Y_fit, ncomp = ncomp, method = "simpls", backend = backend,
      svd.method = "rsvd", scaling = "centering", fit = FALSE,
      return_variance = FALSE, seed = seed
    )
  })[["elapsed"]]
  predict_time <- system.time({
    predicted <- predict(model, X_eval, Y_eval)
  })[["elapsed"]]
  list(model = model, prediction = predicted, fit_time_sec = fit_time,
       predict_time_sec = predict_time)
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
metadata <- list(
  input = normalizePath(input, winslash = "/", mustWork = TRUE),
  seed = seed,
  backend = backend,
  water_region_ppm = c(4.6, 4.8),
  water_columns_masked = length(water_columns),
  n_train = nrow(Xtrain), n_test = nrow(Xtest), p = ncol(Xtrain), q = ncol(Ytrain),
  component_grid = grid,
  package_version = as.character(utils::packageVersion("fastPLS"))
)

if (identical(mode, "select")) {
  set.seed(seed)
  inner_validation <- sample(seq_len(nrow(Xtrain)), floor(0.2 * nrow(Xtrain)))
  inner_training <- setdiff(seq_len(nrow(Xtrain)), inner_validation)
  selection <- do.call(rbind, lapply(grid, function(k) {
    # Fit each candidate separately. Some compact prediction paths retain only
    # the terminal component when a vector is supplied as ncomp.
    run <- run_fit_predict(
      Xtrain[inner_training, , drop = FALSE], Ytrain[inner_training, , drop = FALSE],
      Xtrain[inner_validation, , drop = FALSE], Ytrain[inner_validation, , drop = FALSE],
      k
    )
    pred <- extract_prediction(run$prediction$Ypred, k)
    metric <- evaluate(Ytrain[inner_validation, , drop = FALSE], pred,
                       ytrain = Ytrain[inner_training, , drop = FALSE])$metrics
    data.frame(
      ncomp = k, R2 = unname(metric[["R2"]]), Q2 = unname(metric[["Q2"]]),
      RMSD = unname(metric[["RMSD"]]), fit_time_sec = run$fit_time_sec,
      predict_time_sec = run$predict_time_sec,
      stringsAsFactors = FALSE
    )
  }))
  selected <- selection$ncomp[[which.min(selection$RMSD)]]
  utils::write.csv(selection, file.path(out_dir, "nmr_inner_component_selection.csv"), row.names = FALSE)
  saveRDS(list(metadata = metadata, inner_training = inner_training,
               inner_validation = inner_validation, selection = selection,
               selected_ncomp = selected, session = utils::sessionInfo()),
          file.path(out_dir, "nmr_inner_component_selection.rds"))
  cat(sprintf("selected_ncomp=%d\n", selected))
  quit(save = "no")
}

run <- run_fit_predict(Xtrain, Ytrain, Xtest, Ytest, selected_ncomp)
predicted <- extract_prediction(run$prediction$Ypred, selected_ncomp)
if (!identical(dim(predicted), dim(Ytest))) stop("Prediction dimensions do not match Ytest.", call. = FALSE)
metric <- evaluate(Ytest, predicted, ytrain = Ytrain, bycol = TRUE)
per_sample <- data.frame(
  test_sample = seq_len(nrow(Ytest)),
  RMSD = sqrt(rowMeans((Ytest - predicted)^2)),
  correlation = vapply(seq_len(nrow(Ytest)), function(i) {
    stats::cor(Ytest[i, ], predicted[i, ], use = "pairwise.complete.obs")
  }, numeric(1L))
)
per_response <- data.frame(
  response = colnames(Ytest) %||% as.character(seq_len(ncol(Ytest))),
  RMSD = sqrt(colMeans((Ytest - predicted)^2)),
  MAE = colMeans(abs(Ytest - predicted)),
  R2 = metric$per_response$R2,
  Q2 = metric$per_response$Q2
)
summary_row <- data.frame(
  backend = backend, ncomp = selected_ncomp,
  fit_time_sec = run$fit_time_sec, predict_time_sec = run$predict_time_sec,
  total_time_sec = run$fit_time_sec + run$predict_time_sec,
  R2 = unname(metric$metrics[["R2"]]), Q2 = unname(metric$metrics[["Q2"]]),
  RMSD = unname(metric$metrics[["RMSD"]]), MAE = unname(metric$metrics[["MAE"]]),
  median_sample_RMSD = stats::median(per_sample$RMSD),
  stringsAsFactors = FALSE
)
utils::write.csv(summary_row, file.path(out_dir, paste0("nmr_final_", backend, "_summary.csv")), row.names = FALSE)
utils::write.csv(per_sample, file.path(out_dir, paste0("nmr_final_", backend, "_per_sample.csv")), row.names = FALSE)
utils::write.csv(per_response, file.path(out_dir, paste0("nmr_final_", backend, "_per_response.csv")), row.names = FALSE)
saveRDS(list(metadata = metadata, summary = summary_row, per_sample = per_sample,
             per_response = per_response, observed = Ytest, predicted = predicted,
             session = utils::sessionInfo()),
        file.path(out_dir, paste0("nmr_final_", backend, ".rds")))
print(summary_row)
