#!/usr/bin/env Rscript

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_file <- if (length(script_arg)) sub("^--file=", "", script_arg[[1L]]) else file.path(getwd(), "benchmark_nmr_ycompress_simpls.R")
script_dir <- dirname(normalizePath(script_file, winslash = "/", mustWork = FALSE))
source(file.path(script_dir, "helpers_dataset_memory_compare.R"))

args <- parse_kv_args()
out_dir <- normalizePath(arg_value(args, "out_dir", default = "benchmark_results_nmr_ycompress"), winslash = "/", mustWork = FALSE)
lib_loc <- normalizePath(arg_value(args, "lib_loc", default = Sys.getenv("FASTPLS_BENCH_LIB", .libPaths()[[1L]])), winslash = "/", mustWork = FALSE)
ncomp <- suppressWarnings(as.integer(arg_value(args, "ncomp", default = "50")))
code_dim <- suppressWarnings(as.integer(arg_value(args, "code_dim", default = "50")))
variant <- arg_value(args, "variant", default = "cpp_rsvd")
response_mode <- tolower(arg_value(args, "response_mode", default = "full_y"))
split_seed <- suppressWarnings(as.integer(arg_value(args, "split_seed", default = "123")))

if (!is.finite(ncomp) || is.na(ncomp) || ncomp < 1L) ncomp <- 50L
if (!is.finite(code_dim) || is.na(code_dim) || code_dim < 1L) code_dim <- 50L
if (!is.finite(split_seed) || is.na(split_seed)) split_seed <- 123L

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
suppressPackageStartupMessages(library("fastPLS", lib.loc = lib_loc, character.only = TRUE))

as_pred_matrix <- function(pred, expected_cols) {
  yp <- pred$Ypred
  d <- dim(yp)
  if (length(d) == 3L) {
    if (d[[2L]] == expected_cols) {
      yp <- yp[, , d[[3L]], drop = TRUE]
    } else if (d[[3L]] == expected_cols) {
      yp <- yp[, d[[2L]], , drop = TRUE]
    } else {
      stop("Unexpected prediction dimensions: ", paste(d, collapse = " x "))
    }
  }
  yp <- as.matrix(yp)
  if (ncol(yp) != expected_cols) {
    stop("Prediction has ", ncol(yp), " columns, expected ", expected_cols)
  }
  yp
}

add_mean <- function(mat, center) {
  sweep(mat, 2L, center, "+", check.margin = FALSE)
}

ridge_decoder <- function(Z, Yc) {
  ztz <- crossprod(Z)
  ridge <- 1e-8 * mean(diag(ztz))
  if (!is.finite(ridge) || ridge <= 0) ridge <- 1e-8
  solve(ztz + diag(ridge, ncol(ztz)), crossprod(Z, Yc))
}

cuda_matmul_available <- function() {
  exists("cuda_matrix_multiply", envir = asNamespace("fastPLS"), inherits = FALSE) &&
    isTRUE(fastPLS::has_cuda())
}

cuda_matmul <- function(A, B) {
  get("cuda_matrix_multiply", envir = asNamespace("fastPLS"), inherits = FALSE)(
    as.matrix(A),
    as.matrix(B)
  )
}

cuda_thin_qr <- function(A) {
  get("cuda_thin_qr", envir = asNamespace("fastPLS"), inherits = FALSE)(
    as.matrix(A)
  )
}

ridge_decoder_backend <- function(Z, Yc, use_cuda = FALSE) {
  ztz <- crossprod(Z)
  ridge <- 1e-8 * mean(diag(ztz))
  if (!is.finite(ridge) || ridge <= 0) ridge <- 1e-8
  rhs <- if (isTRUE(use_cuda)) cuda_matmul(t(Z), Yc) else crossprod(Z, Yc)
  solve(ztz + diag(ridge, ncol(ztz)), rhs)
}

make_y_transform <- function(Ytrain, mode, code_dim, seed = 123L, use_cuda_projection = FALSE) {
  mode <- tolower(mode)
  Y <- as.matrix(Ytrain)
  n <- nrow(Y)
  q <- ncol(Y)
  d <- min(as.integer(code_dim), q, max(1L, n - 1L))
  projection_backend <- "cpu"
  prep_time <- system.time({
    y_center <- colMeans(Y)
    Yc <- sweep(Y, 2L, y_center, "-", check.margin = FALSE)

    if (mode %in% c("full_y", "full", "none")) {
      fit_y <- Y
      decoder <- function(pred_z) as.matrix(pred_z)
      effective_dim <- q
      decoder_name <- "identity"
    } else if (mode %in% c("y_pca50", "pca50", "y_pca")) {
      gram <- tcrossprod(Yc)
      eg <- eigen(gram, symmetric = TRUE)
      keep <- seq_len(min(d, sum(eg$values > .Machine$double.eps), ncol(eg$vectors)))
      vals <- pmax(eg$values[keep], .Machine$double.eps)
      U <- eg$vectors[, keep, drop = FALSE]
      sing <- sqrt(vals)
      fit_y <- sweep(U, 2L, sing, "*", check.margin = FALSE)
      loadings <- crossprod(Yc, sweep(U, 2L, 1 / sing, "*", check.margin = FALSE))
      decoder <- function(pred_z) add_mean(as.matrix(pred_z) %*% t(loadings), y_center)
      effective_dim <- ncol(fit_y)
      decoder_name <- "pca_scores_times_loadings"
    } else if (mode %in% c("gaussian50", "gaussian")) {
      set.seed(as.integer(seed))
      projection <- matrix(rnorm(q * d), nrow = q, ncol = d) / sqrt(d)
      if (isTRUE(use_cuda_projection)) {
        if (!cuda_matmul_available()) {
          stop("CUDA Gaussian compression requested, but cuda_matrix_multiply/CUDA is unavailable")
        }
        fit_y <- cuda_matmul(Yc, projection)
        decoder_mat <- ridge_decoder_backend(fit_y, Yc, use_cuda = TRUE)
        decoder <- function(pred_z) add_mean(cuda_matmul(as.matrix(pred_z), decoder_mat), y_center)
        projection_backend <- "cuda"
      } else {
        fit_y <- Yc %*% projection
        decoder_mat <- ridge_decoder(fit_y, Yc)
        decoder <- function(pred_z) add_mean(as.matrix(pred_z) %*% decoder_mat, y_center)
      }
      effective_dim <- ncol(fit_y)
      decoder_name <- "ridge_least_squares"
    } else if (mode %in% c("orthogonal_random50", "orthogonal_random", "orthogonal50")) {
      set.seed(as.integer(seed))
      raw_projection <- matrix(rnorm(q * d), nrow = q, ncol = d)
      if (isTRUE(use_cuda_projection)) {
        if (!cuda_matmul_available()) {
          stop("CUDA orthogonal compression requested, but CUDA helpers are unavailable")
        }
        projection <- cuda_thin_qr(raw_projection)
        fit_y <- cuda_matmul(Yc, projection)
        decoder_mat <- ridge_decoder_backend(fit_y, Yc, use_cuda = TRUE)
        decoder <- function(pred_z) add_mean(cuda_matmul(as.matrix(pred_z), decoder_mat), y_center)
        projection_backend <- "cuda"
      } else {
        projection <- qr.Q(qr(raw_projection), complete = FALSE)
        fit_y <- Yc %*% projection
        decoder_mat <- ridge_decoder(fit_y, Yc)
        decoder <- function(pred_z) add_mean(as.matrix(pred_z) %*% decoder_mat, y_center)
      }
      effective_dim <- ncol(fit_y)
      decoder_name <- "ridge_least_squares"
    } else {
      stop("Unknown response_mode: ", mode)
    }
  })[["elapsed"]] * 1000

  list(
    fit_y = fit_y,
    decoder = decoder,
    prep_time_ms = as.numeric(prep_time),
    effective_dim = as.integer(effective_dim),
    decoder_name = decoder_name,
    projection_backend = projection_backend,
    original_q = as.integer(q)
  )
}

fit_model <- function(task, Yfit, variant, ncomp, seed = 123L) {
  ncomp_eff <- min(as.integer(ncomp), ncol(Yfit), nrow(task$Xtrain) - 1L, ncol(task$Xtrain))
  if (identical(variant, "cpp_rsvd")) {
    return(fastPLS::pls(task$Xtrain, Yfit, ncomp = ncomp_eff, method = "simpls", backend = "cpp", svd.method = "cpu_rsvd", fit = FALSE, seed = seed))
  }
  if (identical(variant, "cpp_irlba")) {
    return(fastPLS::pls(task$Xtrain, Yfit, ncomp = ncomp_eff, method = "simpls", backend = "cpp", svd.method = "irlba", fit = FALSE, seed = seed))
  }
  if (identical(variant, "cuda")) {
    if (!isTRUE(fastPLS::has_cuda())) stop("CUDA backend is not available")
    return(fastPLS::pls(task$Xtrain, Yfit, ncomp = ncomp_eff, method = "simpls", backend = "cuda", fit = FALSE, seed = seed))
  }
  stop("Unknown variant: ", variant)
}

metric_regression <- function(Ytest, Ypred, Ytrain) {
  Ytest <- as.matrix(Ytest)
  Ypred <- as.matrix(Ypred)
  ymean <- colMeans(as.matrix(Ytrain))
  rss <- sum((Ytest - Ypred)^2, na.rm = TRUE)
  tss <- sum((sweep(Ytest, 2L, ymean, "-", check.margin = FALSE))^2, na.rm = TRUE)
  data.frame(
    rmsd = sqrt(mean((Ytest - Ypred)^2, na.rm = TRUE)),
    q2_global = if (is.finite(tss) && tss > 0) 1 - rss / tss else NA_real_,
    mae = mean(abs(Ytest - Ypred), na.rm = TRUE)
  )
}

task <- as_task(find_dataset_rdata("nmr"), dataset_id = "nmr", split_seed = split_seed)
transform <- make_y_transform(
  task$Ytrain,
  response_mode,
  code_dim,
  seed = split_seed,
  use_cuda_projection = identical(variant, "cuda") &&
    response_mode %in% c("gaussian50", "gaussian", "orthogonal_random50", "orthogonal_random", "orthogonal50")
)

row <- tryCatch({
  gc()
  fit_time <- system.time({
    model <- fit_model(task, transform$fit_y, variant, ncomp, seed = 1000L + split_seed)
  })[["elapsed"]] * 1000
  pred_time <- system.time({
    pred <- predict(model, task$Xtest, proj = FALSE)
    pred_z <- as_pred_matrix(pred, transform$effective_dim)
  })[["elapsed"]] * 1000
  recon_time <- system.time({
    y_pred <- transform$decoder(pred_z)
  })[["elapsed"]] * 1000
  metrics <- metric_regression(task$Ytest, y_pred, task$Ytrain)
  data.frame(
    dataset = "nmr",
    method = "simpls",
    variant = variant,
    response_mode = response_mode,
    requested_ncomp = as.integer(ncomp),
    effective_ncomp = as.integer(min(ncomp, transform$effective_dim, task$n_train - 1L, task$p)),
    requested_dim = as.integer(code_dim),
    effective_dim = as.integer(transform$effective_dim),
    original_q = as.integer(transform$original_q),
    decoder = transform$decoder_name,
    projection_backend = transform$projection_backend,
    n_train = as.integer(task$n_train),
    n_test = as.integer(task$n_test),
    p = as.integer(task$p),
    prep_time_ms = as.numeric(transform$prep_time_ms),
    fit_time_ms = as.numeric(fit_time),
    predict_time_ms = as.numeric(pred_time),
    reconstruct_time_ms = as.numeric(recon_time),
    total_time_ms = as.numeric(transform$prep_time_ms + fit_time + pred_time + recon_time),
    rmsd = metrics$rmsd,
    q2_global = metrics$q2_global,
    mae = metrics$mae,
    status = "ok",
    msg = "",
    stringsAsFactors = FALSE
  )
}, error = function(e) {
  data.frame(
    dataset = "nmr",
    method = "simpls",
    variant = variant,
    response_mode = response_mode,
    requested_ncomp = as.integer(ncomp),
    effective_ncomp = NA_integer_,
    requested_dim = as.integer(code_dim),
    effective_dim = NA_integer_,
    original_q = as.integer(ncol(task$Ytrain)),
    decoder = NA_character_,
    projection_backend = NA_character_,
    n_train = as.integer(task$n_train),
    n_test = as.integer(task$n_test),
    p = as.integer(task$p),
    prep_time_ms = NA_real_,
    fit_time_ms = NA_real_,
    predict_time_ms = NA_real_,
    reconstruct_time_ms = NA_real_,
    total_time_ms = NA_real_,
    rmsd = NA_real_,
    q2_global = NA_real_,
    mae = NA_real_,
    status = "error",
    msg = conditionMessage(e),
    stringsAsFactors = FALSE
  )
})

utils::write.csv(row, file.path(out_dir, "nmr_ycompress_row.csv"), row.names = FALSE)
print(row)
