#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(fastPLS)
})

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L || is.na(x[1L]) || !nzchar(as.character(x[1L]))) y else x
}

cmd_args_full <- commandArgs(FALSE)
file_arg <- sub("^--file=", "", cmd_args_full[grep("^--file=", cmd_args_full)][1L])
script_dir <- dirname(normalizePath(file_arg %||% "benchmark/benchmark_imagenet_class_bias_full_grid.R", mustWork = FALSE))
if (!nzchar(script_dir) || is.na(script_dir)) script_dir <- getwd()
helpers <- file.path(script_dir, "helpers_dataset_memory_compare.R")
if (!file.exists(helpers)) helpers <- file.path(getwd(), "benchmark", "helpers_dataset_memory_compare.R")
source(helpers)

csv_vec <- function(x, default, type = c("character", "integer", "numeric")) {
  type <- match.arg(type)
  x <- x %||% paste(default, collapse = ",")
  out <- trimws(strsplit(x, ",", fixed = TRUE)[[1L]])
  out <- out[nzchar(out)]
  switch(type, character = out, integer = as.integer(out), numeric = as.numeric(out))
}

arg_bool <- function(x, default = FALSE) {
  if (is.null(x)) return(default)
  tolower(as.character(x)[1L]) %in% c("1", "true", "t", "yes", "y")
}

timestamp <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")
log_msg <- function(...) message("[", timestamp(), "] ", paste0(..., collapse = ""))

top1_accuracy <- function(pred, truth) {
  mean(as.character(pred) == as.character(truth), na.rm = TRUE)
}

topk_accuracy <- function(top_mat, truth) {
  truth_chr <- as.character(truth)
  mean(vapply(seq_along(truth_chr), function(i) truth_chr[[i]] %in% as.character(top_mat[i, ]), logical(1)), na.rm = TRUE)
}

as_test_n <- function(x) {
  x <- tolower(trimws(as.character(x)[1L]))
  if (x %in% c("rest", "all", "remaining")) return(NA_integer_)
  as.integer(x)
}

imagenet_feature_cols <- function(nms) {
  feat_cols <- grep("^feat_", nms, value = TRUE)
  if (length(feat_cols)) return(feat_cols)
  meta <- intersect(
    c("image_path", "path", "file", "filename", "split", "label_idx", "label", "label_name", "class", "class_name", "synset"),
    nms
  )
  feat_cols <- setdiff(nms, meta)
  feat_cols <- setdiff(feat_cols, "label_idx")
  if (length(feat_cols) < 10L) {
    feat_cols <- setdiff(nms, nms[seq_len(min(3L, length(nms)))])
    feat_cols <- setdiff(feat_cols, "label_idx")
  }
  feat_cols
}

frame_rows_to_matrix <- function(x, rows, cols) {
  n <- length(rows)
  p <- length(cols)
  cells <- as.double(n) * as.double(p)
  if (cells > 5e7) {
    log_msg("Building matrix by streamed column fill: rows=", n, ", cols=", p)
    out <- matrix(NA_real_, nrow = n, ncol = p)
    colnames(out) <- cols
    for (j in seq_along(cols)) {
      v <- x[[cols[[j]]]]
      if (is.numeric(v) || is.integer(v) || is.logical(v)) {
        out[, j] <- as.numeric(v[rows])
      } else {
        out[, j] <- suppressWarnings(as.numeric(as.character(v[rows])))
      }
      if ((j %% 128L) == 0L) gc(FALSE)
    }
    return(out)
  }
  if (requireNamespace("data.table", quietly = TRUE) && data.table::is.data.table(x)) {
    return(as.matrix(x[rows, cols, with = FALSE]))
  }
  numeric_frame_to_matrix(x[rows, cols, drop = FALSE])
}

cache_expected_bytes <- function(n, p) {
  as.double(n) * as.double(p) * 8
}

cache_complete <- function(path, n, p) {
  file.exists(path) && isTRUE(file.info(path)$size == cache_expected_bytes(n, p))
}

cache_safe <- function(x) gsub("[^A-Za-z0-9_.-]+", "_", as.character(x))

task_cache_path <- function(matrix_cache_dir, train_n, test_n_arg, split_seed) {
  if (is.null(matrix_cache_dir) || !nzchar(matrix_cache_dir)) return("")
  file.path(
    matrix_cache_dir,
    paste0(
      "imagenet_seed", cache_safe(split_seed),
      "_train", cache_safe(train_n),
      "_test", cache_safe(test_n_arg),
      "_task.rds"
    )
  )
}

write_matrix_binary_cache <- function(x, rows, cols, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- paste0(path, ".tmp")
  if (file.exists(tmp)) unlink(tmp)
  con <- file(tmp, open = "wb")
  on.exit(close(con), add = TRUE)
  n <- length(rows)
  p <- length(cols)
  log_msg("Writing binary matrix cache: ", path, " rows=", n, ", cols=", p)
  for (j in seq_along(cols)) {
    v <- x[[cols[[j]]]]
    if (is.numeric(v) || is.integer(v) || is.logical(v)) {
      out <- as.double(v[rows])
    } else {
      out <- suppressWarnings(as.double(as.character(v[rows])))
    }
    writeBin(out, con, size = 8, endian = "little")
    if ((j %% 128L) == 0L) gc(FALSE)
  }
  close(con)
  on.exit(NULL, add = FALSE)
  file.rename(tmp, path)
  invisible(path)
}

read_matrix_binary_cache <- function(path, n, p, cols) {
  if (!cache_complete(path, n, p)) {
    stop("Matrix cache is missing or incomplete: ", path)
  }
  log_msg("Reading binary matrix cache: ", path, " rows=", n, ", cols=", p)
  con <- file(path, open = "rb")
  on.exit(close(con), add = TRUE)
  vals <- matrix(NA_real_, nrow = n, ncol = p)
  colnames(vals) <- cols
  for (j in seq_len(p)) {
    col <- readBin(con, what = "double", n = n, size = 8, endian = "little")
    if (length(col) != n) {
      stop("Unexpected matrix cache column length for ", path, " column ", j, ": got ", length(col), ", expected ", n)
    }
    vals[, j] <- col
    if ((j %% 128L) == 0L) gc(FALSE)
  }
  vals
}

read_matrix_binary_block <- function(path, n, p, start, stop, cols = NULL) {
  if (!cache_complete(path, n, p)) {
    stop("Matrix cache is missing or incomplete: ", path)
  }
  start <- as.integer(start)
  stop <- as.integer(stop)
  if (start < 1L || stop < start || stop > n) {
    stop("Invalid binary matrix block: start=", start, ", stop=", stop, ", n=", n)
  }
  nr <- stop - start + 1L
  out <- matrix(NA_real_, nrow = nr, ncol = p)
  if (!is.null(cols)) colnames(out) <- cols
  con <- file(path, open = "rb")
  on.exit(close(con), add = TRUE)
  for (j in seq_len(p)) {
    seek(con, where = as.double(((j - 1L) * n + (start - 1L)) * 8), origin = "start", rw = "read")
    out[, j] <- readBin(con, what = "double", n = nr, size = 8, endian = "little")
  }
  out
}

binary_block_size <- function(default = 8192L) {
  value <- suppressWarnings(as.integer(Sys.getenv("FASTPLS_IMAGENET_STREAM_BLOCK_SIZE", as.character(default)))[1L])
  if (!is.finite(value) || is.na(value) || value < 1L) value <- default
  value
}

sample_test_idx <- function(n, test_n) {
  if (is.na(test_n)) return(seq_len(n))
  sample_rows_n(n, min(test_n, n))
}

load_cached_imagenet_task <- function(matrix_cache_dir, train_n, test_n_arg, split_seed) {
  meta_path <- task_cache_path(matrix_cache_dir, train_n, test_n_arg, split_seed)
  if (!nzchar(meta_path) || !file.exists(meta_path)) return(NULL)
  task <- readRDS(meta_path)
  if (!identical(task$source_format, "binary_cache")) return(NULL)
  if (!cache_complete(task$train_bin, task$n_train, task$p)) return(NULL)
  if (!cache_complete(task$test_bin, task$n_test, task$p)) return(NULL)
  log_msg("Reusing cached ImageNet task metadata: ", meta_path)
  task
}

load_imagenet_task_custom <- function(path, train_n, test_n_arg = "rest", split_seed = 123L, lazy_matrices = TRUE, matrix_cache_dir = NULL) {
  cached_task <- load_cached_imagenet_task(matrix_cache_dir, train_n, test_n_arg, split_seed)
  if (!is.null(cached_task)) return(cached_task)

  e <- new.env(parent = emptyenv())
  objs <- load(path, envir = e)
  set.seed(as.integer(split_seed))
  test_n <- as_test_n(test_n_arg)

  if (all(c("Xtrain", "Ytrain", "Xtest", "Ytest") %in% objs)) {
    y_train_all <- safe_factor(e$Ytrain)
    train_idx <- sample_stratified_n(y_train_all, min(as.integer(train_n), nrow(e$Xtrain)))
    test_idx <- sample_test_idx(nrow(e$Xtest), test_n)
    y_train <- droplevels(y_train_all[train_idx])
    y_test <- factor(e$Ytest[test_idx], levels = levels(y_train))
    Xtrain <- if (lazy_matrices) NULL else as.matrix(e$Xtrain[train_idx, , drop = FALSE])
    Xtest <- if (lazy_matrices) NULL else as.matrix(e$Xtest[test_idx, , drop = FALSE])
    task <- list(
      dataset = "imagenet",
      task_type = "classification",
      dataset_path = normalizePath(path, winslash = "/", mustWork = TRUE),
      split_seed = as.integer(split_seed),
      Xtrain = Xtrain,
      Ytrain = y_train,
      Xtest = Xtest,
      Ytest = y_test,
      train_idx = train_idx,
      test_idx = test_idx,
      feat_cols = colnames(e$Xtrain),
      n_train = length(train_idx),
      n_test = length(test_idx),
      p = ncol(e$Xtrain),
      n_classes = nlevels(y_train),
      source_format = "train_test",
      lazy_matrices = lazy_matrices
    )
    rm(e)
    gc()
    return(task)
  }

  if ("r" %in% objs && is.data.frame(e$r) && "label_idx" %in% colnames(e$r)) {
    if (requireNamespace("data.table", quietly = TRUE)) data.table::setDT(e$r)
    y <- safe_factor(e$r[["label_idx"]])
    train_idx <- sample_stratified_n(y, min(as.integer(train_n), length(y) - 1L))
    rest_idx <- setdiff(seq_along(y), train_idx)
    test_idx <- if (is.na(test_n)) rest_idx else sort(sample(rest_idx, min(test_n, length(rest_idx))))
    feat_cols <- imagenet_feature_cols(names(e$r))
    y_train <- droplevels(y[train_idx])
    Xtrain <- NULL
    Xtest <- NULL
    keep <- rep(TRUE, length(feat_cols))
    if (!lazy_matrices) {
      Xtrain <- frame_rows_to_matrix(e$r, train_idx, feat_cols)
      keep <- colSums(is.finite(Xtrain)) > 0
      feat_cols <- feat_cols[keep]
      Xtrain <- as.matrix(Xtrain[, keep, drop = FALSE])
      Xtest <- frame_rows_to_matrix(e$r, test_idx, feat_cols)
    }
    source_format <- "data_frame"
    train_bin <- test_bin <- NA_character_
    if (lazy_matrices && !is.null(matrix_cache_dir) && nzchar(matrix_cache_dir)) {
      cache_tag <- paste0("seed", as.integer(split_seed), "_train", length(train_idx), "_test", length(test_idx), "_p", length(feat_cols))
      train_bin <- file.path(matrix_cache_dir, paste0("imagenet_", cache_tag, "_train.bin"))
      test_bin <- file.path(matrix_cache_dir, paste0("imagenet_", cache_tag, "_test.bin"))
      if (!cache_complete(train_bin, length(train_idx), length(feat_cols))) {
        write_matrix_binary_cache(e$r, train_idx, feat_cols, train_bin)
      } else {
        log_msg("Reusing binary matrix cache: ", train_bin)
      }
      if (!cache_complete(test_bin, length(test_idx), length(feat_cols))) {
        write_matrix_binary_cache(e$r, test_idx, feat_cols, test_bin)
      } else {
        log_msg("Reusing binary matrix cache: ", test_bin)
      }
      source_format <- "binary_cache"
    }
    task <- list(
      dataset = "imagenet",
      task_type = "classification",
      dataset_path = normalizePath(path, winslash = "/", mustWork = TRUE),
      split_seed = as.integer(split_seed),
      Xtrain = Xtrain,
      Ytrain = y_train,
      Xtest = Xtest,
      Ytest = factor(y[test_idx], levels = levels(y_train)),
      train_idx = train_idx,
      test_idx = test_idx,
      feat_cols = feat_cols,
      feature_keep = keep,
      train_bin = train_bin,
      test_bin = test_bin,
      n_train = length(train_idx),
      n_test = length(test_idx),
      p = length(feat_cols),
      n_classes = nlevels(y_train),
      source_format = source_format,
      lazy_matrices = lazy_matrices
    )
    if (identical(source_format, "binary_cache")) {
      meta_path <- task_cache_path(matrix_cache_dir, train_n, test_n_arg, split_seed)
      dir.create(dirname(meta_path), recursive = TRUE, showWarnings = FALSE)
      saveRDS(task, meta_path)
      log_msg("Saved cached ImageNet task metadata: ", meta_path)
    }
    rm(e, Xtrain, Xtest)
    gc()
    return(task)
  }

  if (all(c("data", "labels") %in% objs)) {
    y <- safe_factor(e$labels)
    train_idx <- sample_stratified_n(y, min(as.integer(train_n), length(y) - 1L))
    rest_idx <- setdiff(seq_along(y), train_idx)
    test_idx <- if (is.na(test_n)) rest_idx else sort(sample(rest_idx, min(test_n, length(rest_idx))))
    y_train <- droplevels(y[train_idx])
    Xtrain <- if (lazy_matrices) NULL else as.matrix(e$data[train_idx, , drop = FALSE])
    Xtest <- if (lazy_matrices) NULL else as.matrix(e$data[test_idx, , drop = FALSE])
    task <- list(
      dataset = "imagenet",
      task_type = "classification",
      dataset_path = normalizePath(path, winslash = "/", mustWork = TRUE),
      split_seed = as.integer(split_seed),
      Xtrain = Xtrain,
      Ytrain = y_train,
      Xtest = Xtest,
      Ytest = factor(y[test_idx], levels = levels(y_train)),
      train_idx = train_idx,
      test_idx = test_idx,
      feat_cols = colnames(e$data),
      n_train = length(train_idx),
      n_test = length(test_idx),
      p = ncol(e$data),
      n_classes = nlevels(y_train),
      source_format = "single_matrix",
      lazy_matrices = lazy_matrices
    )
    rm(e)
    gc()
    return(task)
  }

  stop("Unsupported imagenet.RData format: ", path)
}

load_imagenet_matrix <- function(task, part = c("train", "test")) {
  part <- match.arg(part)
  cached <- task[[if (identical(part, "train")) "Xtrain" else "Xtest"]]
  if (!is.null(cached)) return(cached)

  if (identical(task$source_format, "binary_cache")) {
    path <- if (identical(part, "train")) task$train_bin else task$test_bin
    n <- if (identical(part, "train")) task$n_train else task$n_test
    return(read_matrix_binary_cache(path, n, task$p, task$feat_cols))
  }

  e <- new.env(parent = emptyenv())
  objs <- load(task$dataset_path, envir = e)
  on.exit({
    rm(e)
    gc()
  }, add = TRUE)
  rows <- if (identical(part, "train")) task$train_idx else task$test_idx

  if (identical(task$source_format, "train_test") && all(c("Xtrain", "Xtest") %in% objs)) {
    src <- if (identical(part, "train")) e$Xtrain else e$Xtest
    return(as.matrix(src[rows, , drop = FALSE]))
  }

  if (identical(task$source_format, "data_frame") && "r" %in% objs && is.data.frame(e$r)) {
    if (requireNamespace("data.table", quietly = TRUE)) data.table::setDT(e$r)
    return(frame_rows_to_matrix(e$r, rows, task$feat_cols))
  }

  if (identical(task$source_format, "single_matrix") && "data" %in% objs) {
    return(as.matrix(e$data[rows, , drop = FALSE]))
  }

  stop("Cannot lazily load ", part, " matrix for source_format=", task$source_format)
}

append_row <- function(file, row) {
  write.table(
    row,
    file = file,
    sep = ",",
    row.names = FALSE,
    col.names = !file.exists(file),
    append = file.exists(file),
    qmethod = "double"
  )
}

make_key <- function(d) {
  paste(
    d$stage,
    d$method,
    d$backend,
    d$ncomp,
    d$scaling,
    d$gaussian_y,
    d$gaussian_y_dim,
    d$classifier,
    d$class_bias_method,
    d$class_bias_lambda,
    d$class_bias_iter,
    d$class_bias_clip,
    d$class_bias_eps,
    d$class_bias_calibration_fraction,
    sep = "|"
  )
}

fit_class_bias_model <- function(model, Xtrain, ytrain, backend, method, lambda, iter, clip, eps, calibration_fraction, seed) {
  model_cb <- model
  model_cb$classification_rule <- if (identical(backend, "cuda")) "class_bias_cuda" else "class_bias_cpp"
  model_cb$lda_backend <- model_cb$classification_rule
  model_cb$class_bias_backend <- backend
  fit_class_bias <- get(".fit_class_bias", envir = asNamespace("fastPLS"), inherits = FALSE)
  model_cb$class_bias <- fit_class_bias(
    model_cb,
    Xtrain,
    ytrain,
    backend = backend,
    method = method,
    lambda = lambda,
    iter = iter,
    clip = clip,
    eps = eps,
    calibration_fraction = calibration_fraction,
    seed = seed
  )
  model_cb
}

fit_plssvd_label_stream_binary <- function(task, ncomp, scaling = "centering", backend = "cuda") {
  if (!identical(task$source_format, "binary_cache")) {
    stop("Streaming label-aware PLSSVD requires a binary_cache ImageNet task")
  }
  scal <- pmatch(scaling, c("centering", "autoscaling", "none"))[1L]
  n <- as.integer(task$n_train)
  p <- as.integer(task$p)
  y <- factor(task$Ytrain)
  lev <- levels(y)
  y_code <- as.integer(y)
  m <- length(lev)
  ncomp <- pmin(pmax(as.integer(ncomp), 1L), min(n, p, m))
  max_rank <- max(ncomp)
  bs <- binary_block_size()

  sums <- numeric(p)
  sums_sq <- numeric(p)
  for (start in seq(1L, n, by = bs)) {
    stop <- min(n, start + bs - 1L)
    Xb <- read_matrix_binary_block(task$train_bin, n, p, start, stop, task$feat_cols)
    sums <- sums + colSums(Xb)
    if (scal == 2L) sums_sq <- sums_sq + colSums(Xb * Xb)
  }
  mX <- if (scal < 3L) sums / n else rep(0, p)
  vX <- rep(1, p)
  if (scal == 2L) {
    centered_ss <- pmax(sums_sq - n * mX * mX, 0)
    vX <- sqrt(centered_ss / max(1L, n - 1L))
    vX[!is.finite(vX) | vX == 0] <- 1
  }

  counts <- tabulate(y_code, nbins = m)
  mY <- counts / n
  class_sums <- matrix(0, nrow = p, ncol = m)
  for (start in seq(1L, n, by = bs)) {
    stop <- min(n, start + bs - 1L)
    rows <- start:stop
    Xb <- read_matrix_binary_block(task$train_bin, n, p, start, stop, task$feat_cols)
    if (scal < 3L) Xb <- sweep(Xb, 2L, mX, "-", check.margin = FALSE)
    if (scal == 2L) Xb <- sweep(Xb, 2L, vX, "/", check.margin = FALSE)
    rs <- rowsum(Xb, group = factor(y_code[rows], levels = seq_len(m)), reorder = FALSE)
    rs <- as.matrix(rs)
    if (nrow(rs) != m) {
      rs_full <- matrix(0, nrow = m, ncol = p)
      row_pos <- match(rownames(rs), as.character(seq_len(m)))
      rs_full[row_pos[!is.na(row_pos)], ] <- rs[!is.na(row_pos), , drop = FALSE]
      rs <- rs_full
    }
    class_sums <- class_sums + t(rs)
  }
  total_sums <- rowSums(class_sums)
  S <- class_sums - tcrossprod(total_sums, mY)
  sv <- svd(S, nu = max_rank, nv = max_rank)
  R <- sv$u[, seq_len(max_rank), drop = FALSE]
  Q <- sv$v[, seq_len(max_rank), drop = FALSE]
  d <- sv$d[seq_len(max_rank)]

  G <- matrix(0, nrow = max_rank, ncol = max_rank)
  for (start in seq(1L, n, by = bs)) {
    stop <- min(n, start + bs - 1L)
    Xb <- read_matrix_binary_block(task$train_bin, n, p, start, stop, task$feat_cols)
    if (scal < 3L) Xb <- sweep(Xb, 2L, mX, "-", check.margin = FALSE)
    if (scal == 2L) Xb <- sweep(Xb, 2L, vX, "/", check.margin = FALSE)
    Tb <- Xb %*% R
    G <- G + crossprod(Tb)
  }

  ns <- length(ncomp)
  C_latent <- array(0, dim = c(max_rank, max_rank, ns))
  W_latent <- array(0, dim = c(max_rank, m, ns))
  for (a in seq_along(ncomp)) {
    k <- ncomp[[a]]
    Gk <- G[seq_len(k), seq_len(k), drop = FALSE]
    Dk <- diag(d[seq_len(k)], nrow = k, ncol = k)
    ridge <- 1e-10 * mean(diag(Gk))
    if (!is.finite(ridge) || ridge <= 0) ridge <- 1e-10
    coeff <- tryCatch(solve(Gk + diag(ridge, k), Dk), error = function(e) qr.solve(Gk + diag(ridge, k), Dk))
    C_latent[seq_len(k), seq_len(k), a] <- coeff
    W_latent[seq_len(k), , a] <- coeff %*% t(Q[, seq_len(k), drop = FALSE])
  }

  model <- list(
    C_latent = C_latent,
    W_latent = W_latent,
    Q = Q,
    Ttrain = matrix(numeric(0), nrow = 0, ncol = max_rank),
    R = R,
    mX = matrix(mX, nrow = 1L),
    vX = matrix(vX, nrow = 1L),
    mY = matrix(mY, nrow = 1L),
    p = p,
    m = m,
    ncomp = ncomp,
    Yfit = array(numeric(0), dim = c(0L, 0L, 0L)),
    R2Y = rep(NA_real_, ns),
    pls_method = "plssvd",
    classification = TRUE,
    lev = lev,
    predict_latent_ok = TRUE,
    xprod_default = TRUE,
    xprod_mode = "label_aware_binary_stream",
    B_stored = FALSE,
    compact_prediction = TRUE,
    flash_svd = TRUE,
    flash_svd_backend = backend,
    predict_backend = if (identical(backend, "cuda")) "cuda_flash" else "cpu_flash",
    flash_block_size = bs,
    classification_rule = "argmax",
    lda_backend = "argmax"
  )
  class(model) <- "fastPLS"
  model
}

predict_stream_binary <- function(model, task, part = c("train", "test"), class_bias = NULL, top = 5L, backend = "cuda", row_keep = NULL) {
  part <- match.arg(part)
  n <- if (identical(part, "train")) task$n_train else task$n_test
  p <- task$p
  path <- if (identical(part, "train")) task$train_bin else task$test_bin
  y_true <- if (identical(part, "train")) task$Ytrain else task$Ytest
  keep <- if (is.null(row_keep)) rep(TRUE, n) else row_keep
  bs <- binary_block_size()
  predict_fun <- get(".class_bias_predict", envir = asNamespace("fastPLS"), inherits = FALSE)
  top <- as.integer(top)[1L]
  correct1 <- 0L
  correctk <- 0L
  total <- 0L
  for (start in seq(1L, n, by = bs)) {
    stop <- min(n, start + bs - 1L)
    rows <- start:stop
    take <- keep[rows]
    if (!any(take)) next
    Xb <- read_matrix_binary_block(path, n, p, start, stop, task$feat_cols)
    Xb <- Xb[take, , drop = FALSE]
    truth <- as.character(y_true[rows][take])
    pred <- predict_fun(model, Xb, class_bias = class_bias, top = top, proj = FALSE, backend = backend)
    yhat <- as.character(pred$Ypred[[1L]])
    correct1 <- correct1 + sum(yhat == truth, na.rm = TRUE)
    if (!is.null(pred$Ypred_top) && length(pred$Ypred_top)) {
      top_mat <- pred$Ypred_top[[1L]]
      correctk <- correctk + sum(vapply(seq_along(truth), function(i) truth[[i]] %in% as.character(top_mat[i, ]), logical(1)), na.rm = TRUE)
    }
    total <- total + length(truth)
  }
  list(top1_accuracy = correct1 / total, top5_accuracy = correctk / total, n = total)
}

fit_class_bias_stream_binary <- function(model, task, backend, method, lambda, iter, clip, eps, calibration_fraction, seed) {
  y <- factor(task$Ytrain, levels = model$lev)
  cal_idx <- get(".class_bias_calibration_indices", envir = asNamespace("fastPLS"), inherits = FALSE)(
    y,
    fraction = calibration_fraction,
    seed = seed
  )
  keep <- rep(FALSE, length(y))
  keep[cal_idx] <- TRUE
  n_pass <- if (identical(method, "count_ratio")) 1L else max(1L, as.integer(iter)[1L])
  bias <- matrix(0, nrow = length(model$lev), ncol = length(model$ncomp), dimnames = list(model$lev, paste0("ncomp=", model$ncomp)))
  truth_counts <- tabulate(as.integer(y[cal_idx]), nbins = length(model$lev))
  update_counts <- get(".class_bias_update_counts", envir = asNamespace("fastPLS"), inherits = FALSE)
  predict_fun <- get(".class_bias_predict", envir = asNamespace("fastPLS"), inherits = FALSE)
  bs <- binary_block_size()
  for (pass in seq_len(n_pass)) {
    pred_counts <- matrix(0, nrow = length(model$lev), ncol = length(model$ncomp))
    for (start in seq(1L, task$n_train, by = bs)) {
      stop <- min(task$n_train, start + bs - 1L)
      rows <- start:stop
      take <- keep[rows]
      if (!any(take)) next
      Xb <- read_matrix_binary_block(task$train_bin, task$n_train, task$p, start, stop, task$feat_cols)
      pred <- predict_fun(model, Xb[take, , drop = FALSE], class_bias = bias, top = 1L, proj = FALSE, backend = backend)
      for (a in seq_along(model$ncomp)) {
        pred_counts[, a] <- pred_counts[, a] + tabulate(as.integer(pred$Ypred_index[, a]), nbins = length(model$lev))
      }
    }
    for (a in seq_along(model$ncomp)) {
      bias[, a] <- bias[, a] + update_counts(truth_counts, pred_counts[, a], lambda, eps, clip)
      bias[, a] <- bias[, a] - mean(bias[, a])
    }
  }
  attr(bias, "parameters") <- list(
    method = method,
    lambda = lambda,
    iter = n_pass,
    clip = clip,
    eps = eps,
    calibration_fraction = calibration_fraction,
    n_calibration = length(cal_idx),
    seed = seed,
    backend = backend
  )
  bias
}

args <- parse_kv_args()
out_root <- normalize_path_if_exists(arg_value(args, "out_dir", ""))
if (!nzchar(out_root)) {
  out_root <- file.path(getwd(), "benchmark_results", "imagenet_full_class_bias_grid", format(Sys.time(), "%Y%m%d_%H%M%S"))
}
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

train_n <- as.integer(arg_value(args, "train_n", "1000000"))
test_n_arg <- arg_value(args, "test_n", "rest")
split_seed <- as.integer(arg_value(args, "seed", "123"))
ncomp_grid <- csv_vec(arg_value(args, "ncomp", "300,500,750,1000"), c(300L, 500L, 750L, 1000L), "integer")
method_grid <- csv_vec(arg_value(args, "methods", "plssvd,simpls"), c("plssvd", "simpls"), "character")
backend_grid <- csv_vec(arg_value(args, "backends", "cuda"), "cuda", "character")
scaling_grid <- csv_vec(arg_value(args, "scaling", "centering,none"), c("centering", "none"), "character")
class_bias_method_grid <- csv_vec(arg_value(args, "class_bias_method", "iter_count_ratio,count_ratio"), c("iter_count_ratio", "count_ratio"), "character")
lambda_grid <- csv_vec(arg_value(args, "lambda", "0,0.003,0.005,0.006,0.0075,0.01,0.0125,0.015"), c(0, 0.003, 0.005, 0.006, 0.0075, 0.01, 0.0125, 0.015), "numeric")
iter_grid <- csv_vec(arg_value(args, "iter", "1,2,3"), c(1L, 2L, 3L), "integer")
clip_grid <- csv_vec(arg_value(args, "clip", "Inf,0.25,0.5"), c(Inf, 0.25, 0.5), "numeric")
eps_grid <- csv_vec(arg_value(args, "eps", "0.5,1,2"), c(0.5, 1, 2), "numeric")
class_bias_fraction_grid <- csv_vec(arg_value(args, "class_bias_fraction", "1,0.5"), c(1, 0.5), "numeric")
classifiers <- csv_vec(arg_value(args, "classifiers", "argmax,class_bias"), c("argmax", "class_bias"), "character")
top_k <- as.integer(arg_value(args, "top", "5"))
max_runs <- as.integer(arg_value(args, "max_runs", "0"))
resume <- arg_bool(arg_value(args, "resume", "true"), TRUE)
return_variance <- arg_bool(arg_value(args, "return_variance", "false"), FALSE)
matrix_cache_dir <- normalize_path_if_exists(arg_value(args, "matrix_cache_dir", file.path(out_root, "matrix_cache")))
prepare_cache_only <- arg_bool(arg_value(args, "prepare_cache_only", "false"), FALSE)
gaussian_y <- arg_bool(arg_value(args, "gaussian_y", "false"), FALSE)
gaussian_y_dim_arg <- arg_value(args, "gaussian_y_dim", "ncomp")
if (gaussian_y && "class_bias" %in% classifiers) {
  log_msg("gaussian_y=true is incompatible with this benchmark's class_bias calibration; using argmax rows only.")
  classifiers <- setdiff(classifiers, "class_bias")
  if (!length(classifiers)) classifiers <- "argmax"
}

gaussian_y_dim_for <- function(ncomp) {
  if (!isTRUE(gaussian_y)) return(0L)
  if (tolower(as.character(gaussian_y_dim_arg)[1L]) %in% c("ncomp", "same", "auto")) {
    return(as.integer(ncomp))
  }
  as.integer(gaussian_y_dim_arg)
}

raw_file <- file.path(out_root, "imagenet_full_class_bias_grid_raw.csv")
best_file <- file.path(out_root, "imagenet_full_class_bias_grid_best.csv")
grid_file <- file.path(out_root, "imagenet_full_class_bias_grid.csv")

log_msg("Output: ", out_root)
imagenet_path <- find_dataset_rdata("imagenet")
log_msg("Loading ImageNet: train_n=", train_n, ", test_n=", test_n_arg, ", seed=", split_seed)
task <- load_imagenet_task_custom(
  imagenet_path,
  train_n = train_n,
  test_n_arg = test_n_arg,
  split_seed = split_seed,
  matrix_cache_dir = matrix_cache_dir
)
log_msg("Loaded ", task$n_train, " train x ", task$p, " features; ", task$n_test, " test; classes=", task$n_classes, "; source=", task$source_format)

if (prepare_cache_only) {
  log_msg("prepare-cache-only complete: ", matrix_cache_dir)
  quit(save = "no", status = 0L)
}

fit_grid <- expand.grid(
  method = method_grid,
  backend = backend_grid,
  ncomp = ncomp_grid,
  scaling = scaling_grid,
  stringsAsFactors = FALSE
)
fit_grid$gaussian_y <- gaussian_y
fit_grid$gaussian_y_dim <- vapply(fit_grid$ncomp, gaussian_y_dim_for, integer(1))
cb_grid <- expand.grid(
  class_bias_method = class_bias_method_grid,
  class_bias_lambda = lambda_grid,
  class_bias_iter = iter_grid,
  class_bias_clip = clip_grid,
  class_bias_eps = eps_grid,
  class_bias_calibration_fraction = class_bias_fraction_grid,
  stringsAsFactors = FALSE
)
cb_grid <- cb_grid[cb_grid$class_bias_lambda > 0, , drop = FALSE]

rows_grid <- list()
idx <- 0L
for (i in seq_len(nrow(fit_grid))) {
  fg <- fit_grid[i, , drop = FALSE]
  if ("argmax" %in% classifiers) {
    idx <- idx + 1L
    rows_grid[[idx]] <- cbind(
      stage = "predict",
      fg,
      classifier = "argmax",
      class_bias_method = NA_character_,
      class_bias_lambda = 0,
      class_bias_iter = 0L,
      class_bias_clip = Inf,
      class_bias_eps = NA_real_,
      class_bias_calibration_fraction = NA_real_
    )
  }
  if ("class_bias" %in% classifiers && nrow(cb_grid)) {
    for (j in seq_len(nrow(cb_grid))) {
      idx <- idx + 1L
      rows_grid[[idx]] <- cbind(stage = "predict", fg, classifier = "class_bias", cb_grid[j, , drop = FALSE])
    }
  }
}
param_grid <- do.call(rbind, rows_grid)
param_grid$ncomp <- as.integer(param_grid$ncomp)
param_grid$gaussian_y <- tolower(as.character(param_grid$gaussian_y)) %in% c("true", "t", "1", "yes", "y")
param_grid$gaussian_y_dim <- as.integer(param_grid$gaussian_y_dim)
param_grid$class_bias_method <- as.character(param_grid$class_bias_method)
param_grid$class_bias_calibration_fraction <- as.numeric(param_grid$class_bias_calibration_fraction)
param_grid$class_bias_iter <- as.integer(param_grid$class_bias_iter)
param_grid$class_bias_lambda <- as.numeric(param_grid$class_bias_lambda)
param_grid$class_bias_clip <- as.numeric(param_grid$class_bias_clip)
param_grid$class_bias_eps <- as.numeric(param_grid$class_bias_eps)
if (max_runs > 0L && nrow(param_grid) > max_runs) param_grid <- param_grid[seq_len(max_runs), , drop = FALSE]
param_grid$key <- make_key(param_grid)
write.csv(param_grid, grid_file, row.names = FALSE)

writeLines(capture.output(sessionInfo()), file.path(out_root, "session_info.txt"))
writeLines(c(
  paste("imagenet_path=", imagenet_path, sep = ""),
  paste("train_n=", task$n_train, sep = ""),
  paste("test_n=", task$n_test, sep = ""),
  paste("p=", task$p, sep = ""),
  paste("n_classes=", task$n_classes, sep = ""),
  paste("source_format=", task$source_format, sep = ""),
  paste("matrix_cache_dir=", matrix_cache_dir, sep = ""),
  paste("top_k=", top_k, sep = ""),
  paste("return_variance=", return_variance, sep = ""),
  paste("prepare_cache_only=", prepare_cache_only, sep = ""),
  paste("gaussian_y=", gaussian_y, sep = ""),
  paste("gaussian_y_dim_arg=", gaussian_y_dim_arg, sep = ""),
  paste("class_bias_method_grid=", paste(class_bias_method_grid, collapse = ","), sep = ""),
  paste("class_bias_fraction_grid=", paste(class_bias_fraction_grid, collapse = ","), sep = ""),
  paste("resume=", resume, sep = "")
), file.path(out_root, "parameters.txt"))

done_keys <- character(0)
if (resume && file.exists(raw_file)) {
  old <- read.csv(raw_file, stringsAsFactors = FALSE)
  if (nrow(old)) done_keys <- unique(old$key[old$status %in% c("ok", "error")])
}

fit_groups <- unique(param_grid[, c("method", "backend", "ncomp", "scaling", "gaussian_y", "gaussian_y_dim"), drop = FALSE])
for (g in seq_len(nrow(fit_groups))) {
  fg <- fit_groups[g, , drop = FALSE]
  group_idx <- which(
    param_grid$method == fg$method &
      param_grid$backend == fg$backend &
      param_grid$ncomp == fg$ncomp &
      param_grid$scaling == fg$scaling &
      param_grid$gaussian_y == fg$gaussian_y &
      param_grid$gaussian_y_dim == fg$gaussian_y_dim
  )
  group_grid <- param_grid[group_idx, , drop = FALSE]
  group_grid <- group_grid[!(group_grid$key %in% done_keys), , drop = FALSE]
  if (!nrow(group_grid)) next

  log_msg("Fit group ", g, "/", nrow(fit_groups), ": method=", fg$method, ", backend=", fg$backend, ", ncomp=", fg$ncomp, ", scaling=", fg$scaling, ", gaussian_y=", fg$gaussian_y, ", gaussian_y_dim=", fg$gaussian_y_dim, ", pending predictions=", nrow(group_grid))
  use_stream_binary <- identical(task$source_format, "binary_cache") &&
    identical(as.character(fg$method), "plssvd") &&
    !isTRUE(fg$gaussian_y) &&
    identical(as.character(fg$backend), "cuda")
  Xtrain <- NULL
  if (!use_stream_binary) {
    log_msg("Loading training matrix for this fit group")
    Xtrain <- load_imagenet_matrix(task, "train")
  } else {
    log_msg("Using label-aware binary-stream PLSSVD fit; full Xtrain and dense one-hot Y will not be loaded")
  }
  model <- NULL
  fit_error <- NULL
  fit_time <- NA_real_
  fit_time <- system.time({
    model <- tryCatch({
      if (use_stream_binary) {
        fit_plssvd_label_stream_binary(task, ncomp = fg$ncomp, scaling = fg$scaling, backend = fg$backend)
      } else {
        fastPLS::pls(
          Xtrain,
          task$Ytrain,
          ncomp = fg$ncomp,
          method = fg$method,
          backend = fg$backend,
          svd.method = "cpu_rsvd",
          scaling = fg$scaling,
          classifier = "argmax",
          gaussian_y = isTRUE(fg$gaussian_y),
          gaussian_y_dim = if (!isTRUE(fg$gaussian_y) || is.na(fg$gaussian_y_dim) || fg$gaussian_y_dim <= 0L) NULL else as.integer(fg$gaussian_y_dim),
          return_variance = return_variance,
          fit = FALSE,
          proj = FALSE
        )
      }
    }, error = function(e) {
      fit_error <<- conditionMessage(e)
      NULL
    })
  })[["elapsed"]]

  if (is.null(model)) {
    for (r in seq_len(nrow(group_grid))) {
      p <- group_grid[r, ]
      row <- data.frame(
        key = p$key, status = "error", method = p$method, backend = p$backend, ncomp = p$ncomp, scaling = p$scaling,
        gaussian_y = p$gaussian_y, gaussian_y_dim = p$gaussian_y_dim,
        classifier = p$classifier, class_bias_method = p$class_bias_method,
        class_bias_lambda = p$class_bias_lambda, class_bias_iter = p$class_bias_iter,
        class_bias_clip = p$class_bias_clip, class_bias_eps = p$class_bias_eps,
        class_bias_calibration_fraction = p$class_bias_calibration_fraction,
        train_n = task$n_train, test_n = task$n_test,
        p = task$p, n_classes = task$n_classes, fit_time_sec = fit_time, calibrate_time_sec = NA_real_,
        predict_time_sec = NA_real_, total_time_sec = fit_time, top1_accuracy = NA_real_, top5_accuracy = NA_real_,
        notes = paste("fit failed:", fit_error), stringsAsFactors = FALSE
      )
      append_row(raw_file, row)
      done_keys <- c(done_keys, p$key)
    }
    rm(Xtrain)
    gc()
    next
  }

  if (use_stream_binary) {
    for (r in seq_len(nrow(group_grid))) {
      p <- group_grid[r, ]
      row <- data.frame(
        key = p$key,
        status = "ok",
        method = p$method,
        backend = p$backend,
        ncomp = p$ncomp,
        scaling = p$scaling,
        gaussian_y = p$gaussian_y,
        gaussian_y_dim = p$gaussian_y_dim,
        classifier = p$classifier,
        class_bias_lambda = p$class_bias_lambda,
        class_bias_iter = p$class_bias_iter,
        class_bias_clip = p$class_bias_clip,
        class_bias_eps = p$class_bias_eps,
        class_bias_method = p$class_bias_method,
        class_bias_calibration_fraction = p$class_bias_calibration_fraction,
        train_n = task$n_train,
        test_n = task$n_test,
        p = task$p,
        n_classes = task$n_classes,
        fit_time_sec = fit_time,
        calibrate_time_sec = 0,
        predict_time_sec = NA_real_,
        total_time_sec = NA_real_,
        top1_accuracy = NA_real_,
        top5_accuracy = NA_real_,
        notes = "label_aware_binary_stream",
        stringsAsFactors = FALSE
      )
      bias <- NULL
      err <- NULL
      if (identical(p$classifier, "class_bias")) {
        log_msg("Stream calibrate class-bias: method=", p$method, ", ncomp=", p$ncomp, ", scaling=", p$scaling, ", class_bias_method=", p$class_bias_method, ", lambda=", p$class_bias_lambda, ", iter=", p$class_bias_iter, ", clip=", p$class_bias_clip, ", eps=", p$class_bias_eps, ", fraction=", p$class_bias_calibration_fraction)
        row$calibrate_time_sec <- system.time({
          bias <- tryCatch(
            fit_class_bias_stream_binary(
              model,
              task,
              backend = if (identical(p$backend, "cuda")) "cuda" else "cpp",
              method = p$class_bias_method,
              lambda = p$class_bias_lambda,
              iter = p$class_bias_iter,
              clip = p$class_bias_clip,
              eps = p$class_bias_eps,
              calibration_fraction = p$class_bias_calibration_fraction,
              seed = split_seed
            ),
            error = function(e) {
              err <<- conditionMessage(e)
              NULL
            }
          )
        })[["elapsed"]]
      }
      if (!is.null(err)) {
        row$status <- "error"
        row$notes <- paste("stream calibration failed:", err)
        row$total_time_sec <- fit_time + row$calibrate_time_sec
        append_row(raw_file, row)
        done_keys <- c(done_keys, row$key)
        next
      }
      log_msg("Stream predict: method=", row$method, ", ncomp=", row$ncomp, ", classifier=", row$classifier, ", lambda=", row$class_bias_lambda, ", iter=", row$class_bias_iter, ", clip=", row$class_bias_clip)
      pred_res <- NULL
      row$predict_time_sec <- system.time({
        pred_res <- tryCatch(
          predict_stream_binary(
            model,
            task,
            part = "test",
            class_bias = bias,
            top = top_k,
            backend = if (identical(p$backend, "cuda")) "cuda" else "cpp"
          ),
          error = function(e) {
            err <<- conditionMessage(e)
            NULL
          }
        )
      })[["elapsed"]]
      row$total_time_sec <- fit_time + row$calibrate_time_sec + row$predict_time_sec
      if (!is.null(err) || is.null(pred_res)) {
        row$status <- "error"
        row$notes <- paste("stream prediction failed:", err)
      } else {
        row$top1_accuracy <- pred_res$top1_accuracy
        row$top5_accuracy <- pred_res$top5_accuracy
      }
      append_row(raw_file, row)
      done_keys <- c(done_keys, row$key)
    }
    rm(model)
    gc()
    next
  }

  prediction_jobs <- vector("list", nrow(group_grid))
  for (r in seq_len(nrow(group_grid))) {
    p <- group_grid[r, ]
    row <- data.frame(
      key = p$key,
      status = "ok",
      method = p$method,
      backend = p$backend,
      ncomp = p$ncomp,
      scaling = p$scaling,
      gaussian_y = p$gaussian_y,
      gaussian_y_dim = p$gaussian_y_dim,
      classifier = p$classifier,
      class_bias_lambda = p$class_bias_lambda,
      class_bias_iter = p$class_bias_iter,
      class_bias_clip = p$class_bias_clip,
      class_bias_eps = p$class_bias_eps,
      class_bias_method = p$class_bias_method,
      class_bias_calibration_fraction = p$class_bias_calibration_fraction,
      train_n = task$n_train,
      test_n = task$n_test,
      p = task$p,
      n_classes = task$n_classes,
      fit_time_sec = fit_time,
      calibrate_time_sec = 0,
      predict_time_sec = NA_real_,
      total_time_sec = NA_real_,
      top1_accuracy = NA_real_,
      top5_accuracy = NA_real_,
      notes = "",
      stringsAsFactors = FALSE
    )
    pred_model <- model
    cal_err <- NULL
    if (identical(p$classifier, "class_bias")) {
      cb_backend <- if (identical(p$backend, "cuda")) "cuda" else "cpp"
      log_msg("Calibrate class-bias: method=", p$method, ", ncomp=", p$ncomp, ", scaling=", p$scaling, ", class_bias_method=", p$class_bias_method, ", lambda=", p$class_bias_lambda, ", iter=", p$class_bias_iter, ", clip=", p$class_bias_clip, ", eps=", p$class_bias_eps, ", fraction=", p$class_bias_calibration_fraction)
      row$calibrate_time_sec <- system.time({
        pred_model <- tryCatch(
          fit_class_bias_model(
            model,
            Xtrain,
            task$Ytrain,
            backend = cb_backend,
            method = p$class_bias_method,
            lambda = p$class_bias_lambda,
            iter = p$class_bias_iter,
            clip = p$class_bias_clip,
            eps = p$class_bias_eps,
            calibration_fraction = p$class_bias_calibration_fraction,
            seed = split_seed
          ),
          error = function(e) {
            cal_err <<- conditionMessage(e)
            NULL
          }
        )
      })[["elapsed"]]
    }
    if (!is.null(cal_err) || is.null(pred_model)) {
      row$status <- "error"
      row$notes <- paste("calibration failed:", cal_err)
      append_row(raw_file, row)
      done_keys <- c(done_keys, p$key)
    } else {
      prediction_jobs[[r]] <- list(row = row, model = pred_model)
    }
  }

  rm(Xtrain)
  gc()
  prediction_jobs <- Filter(Negate(is.null), prediction_jobs)
  if (!length(prediction_jobs)) {
    rm(model)
    gc()
    next
  }

  log_msg("Loading test matrix for ", length(prediction_jobs), " prediction job(s)")
  Xtest <- load_imagenet_matrix(task, "test")
  for (r in seq_along(prediction_jobs)) {
    job <- prediction_jobs[[r]]
    row <- job$row
    log_msg("Predict: method=", row$method, ", ncomp=", row$ncomp, ", scaling=", row$scaling, ", classifier=", row$classifier, ", class_bias_method=", row$class_bias_method, ", lambda=", row$class_bias_lambda, ", iter=", row$class_bias_iter, ", clip=", row$class_bias_clip, ", eps=", row$class_bias_eps, ", fraction=", row$class_bias_calibration_fraction)
    t0 <- proc.time()[["elapsed"]]
    pred <- NULL
    err <- NULL
    row$predict_time_sec <- system.time({
      pred <- tryCatch(
        predict(job$model, Xtest, top = top_k),
        error = function(e) {
          err <<- conditionMessage(e)
          NULL
        }
      )
    })[["elapsed"]]
    row$total_time_sec <- proc.time()[["elapsed"]] - t0 + fit_time + row$calibrate_time_sec
    if (!is.null(err) || is.null(pred)) {
      row$status <- "error"
      row$notes <- err
    } else {
      yhat <- pred$Ypred[[1L]]
      row$top1_accuracy <- top1_accuracy(yhat, task$Ytest)
      if (!is.null(pred$Ypred_top) && length(pred$Ypred_top)) {
        row$top5_accuracy <- topk_accuracy(pred$Ypred_top[[1L]], task$Ytest)
      }
    }
    append_row(raw_file, row)
    done_keys <- c(done_keys, row$key)
  }
  rm(model, Xtest, prediction_jobs)
  gc()
}

raw <- read.csv(raw_file, stringsAsFactors = FALSE)
best <- raw[order(-raw$top1_accuracy, -raw$top5_accuracy, raw$total_time_sec, na.last = TRUE), ]
write.csv(best, best_file, row.names = FALSE)

if (requireNamespace("ggplot2", quietly = TRUE)) {
  suppressPackageStartupMessages(library(ggplot2))
  ok <- raw[raw$status == "ok", , drop = FALSE]
  if (nrow(ok)) {
    ok$clip_label <- ifelse(is.infinite(ok$class_bias_clip), "Inf", as.character(ok$class_bias_clip))
    p1 <- ggplot(ok, aes(class_bias_lambda, top1_accuracy, color = factor(class_bias_iter), linetype = clip_label)) +
      geom_line() + geom_point(size = 1.8) +
      facet_grid(method + scaling ~ ncomp, scales = "free_y") +
      theme_bw(base_size = 13) +
      labs(x = "class_bias_lambda", y = "top-1 accuracy", color = "iter", linetype = "clip")
    ggsave(file.path(out_root, "imagenet_full_class_bias_top1.png"), p1, width = 14, height = 8, dpi = 150)
    p2 <- ggplot(ok, aes(total_time_sec, top1_accuracy, color = factor(ncomp), shape = classifier)) +
      geom_point(size = 2.2) +
      facet_grid(method ~ scaling) +
      theme_bw(base_size = 13) +
      labs(x = "total time (s)", y = "top-1 accuracy", color = "ncomp")
    ggsave(file.path(out_root, "imagenet_full_time_accuracy.png"), p2, width = 12, height = 7, dpi = 150)
  }
}

log_msg("Top configurations:")
print(utils::head(best[, c("method", "backend", "ncomp", "scaling", "gaussian_y", "gaussian_y_dim", "classifier", "class_bias_method", "class_bias_lambda", "class_bias_iter", "class_bias_clip", "class_bias_eps", "class_bias_calibration_fraction", "top1_accuracy", "top5_accuracy", "total_time_sec", "status", "notes")], 20), row.names = FALSE)
log_msg("Done: ", out_root)
