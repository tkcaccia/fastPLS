## Historical R IRLBA prototype retained only as a commented development note.
##  stopifnot(work>nu)
##  IRLB(X, nu, work, maxit, tol, eps, svtol)
##}

##r_orthog <- function(x, y) {
##  if (missing(y))
##    y <- runif(nrow(x))
##  y <- matrix(y)
##  xm <- nrow(x)
##  xn <- ncol(x)
##  yn <- ncol(y)
##  stopifnot(nrow(y)==xm)
##  stopifnot(yn==1)
##  initT <- matrix(0, xn+1, yn+1)
##  ORTHOG(x, y, initT, xm, xn, yn)
##}
# https://github.com/zdk123/irlba

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

.cap_plssvd_ncomp <- function(ncomp, nrows_x, ncols_x, ncols_y, warn = TRUE) {
  ncomp <- as.integer(ncomp)
  max_plssvd_rank <- min(as.integer(nrows_x), as.integer(ncols_x), as.integer(ncols_y))
  if (max_plssvd_rank < 1L) {
    stop("plssvd rank is < 1")
  }
  over <- max(ncomp, na.rm = TRUE) > max_plssvd_rank
  if (isTRUE(over) && isTRUE(warn)) {
    warning(
      sprintf(
        "plssvd rank is limited to %d; requested ncomp above this value will use %d components internally",
        max_plssvd_rank, max_plssvd_rank
      ),
      call. = FALSE
    )
  }
  ncomp <- pmin(pmax(ncomp, 1L), max_plssvd_rank)
  list(ncomp = ncomp, max_rank = max_plssvd_rank, capped = isTRUE(over))
}

.restore_env_scalar <- function(name, value) {
  stopifnot(length(name) == 1L, nzchar(name))
  if (length(value) != 1L || is.na(value)) {
    Sys.unsetenv(name)
  } else {
    val <- list(as.character(value))
    names(val) <- name
    do.call(Sys.setenv, val)
  }
}

.with_fastpls_fast_options <- function(expr,
                                       return_ttrain = FALSE) {
  old <- c(
    FASTPLS_FAST_CENTER_T = Sys.getenv("FASTPLS_FAST_CENTER_T", unset = NA_character_),
    FASTPLS_FAST_REORTH_V = Sys.getenv("FASTPLS_FAST_REORTH_V", unset = NA_character_),
    FASTPLS_FAST_INCREMENTAL = Sys.getenv("FASTPLS_FAST_INCREMENTAL", unset = NA_character_),
    FASTPLS_FAST_INC_ITERS = Sys.getenv("FASTPLS_FAST_INC_ITERS", unset = NA_character_),
    FASTPLS_FAST_DEFLCACHE = Sys.getenv("FASTPLS_FAST_DEFLCACHE", unset = NA_character_),
    FASTPLS_RETURN_TTRAIN = Sys.getenv("FASTPLS_RETURN_TTRAIN", unset = NA_character_)
  )
  on.exit({
    for (nm in names(old)) {
      val <- old[[nm]]
      .restore_env_scalar(nm, val)
    }
  }, add = TRUE)
  Sys.setenv(
    FASTPLS_FAST_CENTER_T = "0",
    FASTPLS_FAST_REORTH_V = "0",
    FASTPLS_FAST_INCREMENTAL = "1",
    FASTPLS_FAST_INC_ITERS = "2",
    FASTPLS_FAST_DEFLCACHE = "1",
    FASTPLS_RETURN_TTRAIN = if (isTRUE(return_ttrain)) "1" else "0"
  )
  force(expr)
}

.with_irlba_options <- function(expr,
                                irlba_work = 0L,
                                irlba_maxit = 1000L,
                                irlba_tol = 1e-5,
                                irlba_eps = 1e-9,
                                irlba_svtol = 1e-5) {
  old <- c(
    FASTPLS_IRLBA_WORK = Sys.getenv("FASTPLS_IRLBA_WORK", unset = NA_character_),
    FASTPLS_IRLBA_MAXIT = Sys.getenv("FASTPLS_IRLBA_MAXIT", unset = NA_character_),
    FASTPLS_IRLBA_TOL = Sys.getenv("FASTPLS_IRLBA_TOL", unset = NA_character_),
    FASTPLS_IRLBA_EPS = Sys.getenv("FASTPLS_IRLBA_EPS", unset = NA_character_),
    FASTPLS_IRLBA_SVTOL = Sys.getenv("FASTPLS_IRLBA_SVTOL", unset = NA_character_)
  )
  on.exit({
    for (nm in names(old)) {
      val <- old[[nm]]
      .restore_env_scalar(nm, val)
    }
  }, add = TRUE)
  Sys.setenv(
    FASTPLS_IRLBA_WORK = as.character(as.integer(irlba_work)),
    FASTPLS_IRLBA_MAXIT = as.character(as.integer(irlba_maxit)),
    FASTPLS_IRLBA_TOL = as.character(as.numeric(irlba_tol)),
    FASTPLS_IRLBA_EPS = as.character(as.numeric(irlba_eps)),
    FASTPLS_IRLBA_SVTOL = as.character(as.numeric(irlba_svtol))
  )
  force(expr)
}

.with_gpu_native_options <- function(expr,
                                     gpu_device_state = FALSE,
                                     gpu_qr = TRUE,
                                     gpu_eig = TRUE,
                                     gpu_qless_qr = FALSE,
                                     gpu_finalize_threshold = 32L) {
  old <- c(
    FASTPLS_GPU_DEVICE_STATE = Sys.getenv("FASTPLS_GPU_DEVICE_STATE", unset = NA_character_),
    FASTPLS_GPU_QR = Sys.getenv("FASTPLS_GPU_QR", unset = NA_character_),
    FASTPLS_GPU_EIG = Sys.getenv("FASTPLS_GPU_EIG", unset = NA_character_),
    FASTPLS_GPU_QLESS_QR = Sys.getenv("FASTPLS_GPU_QLESS_QR", unset = NA_character_),
    FASTPLS_GPU_FINALIZE_THRESHOLD = Sys.getenv("FASTPLS_GPU_FINALIZE_THRESHOLD", unset = NA_character_)
  )
  on.exit({
    for (nm in names(old)) {
      .restore_env_scalar(nm, old[[nm]])
    }
  }, add = TRUE)
  Sys.setenv(
    FASTPLS_GPU_DEVICE_STATE = if (isTRUE(gpu_device_state)) "1" else "0",
    FASTPLS_GPU_QR = if (isTRUE(gpu_qr)) "1" else "0",
    FASTPLS_GPU_EIG = if (isTRUE(gpu_eig)) "1" else "0",
    FASTPLS_GPU_QLESS_QR = if (isTRUE(gpu_qless_qr)) "1" else "0",
    FASTPLS_GPU_FINALIZE_THRESHOLD = as.character(as.integer(gpu_finalize_threshold))
  )
  force(expr)
}

.with_simpls_gpu_xprod <- function(expr) {
  old <- Sys.getenv("FASTPLS_GPU_SIMPLS_XPROD", unset = NA_character_)
  on.exit(.restore_env_scalar("FASTPLS_GPU_SIMPLS_XPROD", old), add = TRUE)
  Sys.setenv(FASTPLS_GPU_SIMPLS_XPROD = "1")
  force(expr)
}

.enable_flash_prediction <- function(model, backend = c("cpu", "cuda"), block_size = 4096L) {
  backend <- match.arg(backend)
  model$predict_backend <- if (identical(backend, "cuda")) "cuda_flash" else "cpu_flash"
  model$flash_svd <- TRUE
  model$flash_svd_backend <- backend
  model$flash_svd_mode <- "streamed_low_rank_prediction"
  model$flash_block_size <- as.integer(block_size)
  model
}

.attach_train_scores <- function(model, Xtrain) {
  if (is.null(model$R) || length(model$R) == 0L) return(model)
  if (!is.null(model$Ttrain) && length(model$Ttrain) > 0L && all(dim(model$Ttrain) > 0L)) {
    return(model)
  }
  model$Ttrain <- .fastpls_latent_scores(model, Xtrain, ncomp = max(model$ncomp), backend = "cpu")
  model
}

.fastpls_named_components <- function(x, prefix) {
  names(x) <- paste0(prefix, seq_along(x))
  x
}

.pls_x_variance_explained <- function(model, Xtrain) {
  if (is.null(model$R) || length(model$R) == 0L || is.null(model$ncomp)) {
    return(NULL)
  }
  k <- min(max(as.integer(model$ncomp), na.rm = TRUE), ncol(as.matrix(model$R)))
  if (!is.finite(k) || is.na(k) || k < 1L) {
    return(NULL)
  }
  Xscaled <- .fastpls_scaled_by_model(model, Xtrain)
  total_ss <- sum(Xscaled * Xscaled)
  if (!is.finite(total_ss) || total_ss <= 0) {
    return(NULL)
  }

  scores <- .fastpls_score_matrix(model, "Ttrain")
  if (is.null(scores) || ncol(scores) < k || nrow(scores) != nrow(Xscaled)) {
    scores <- .fastpls_latent_scores(model, Xtrain, ncomp = k, backend = "cpu")
  }
  scores <- as.matrix(scores)[, seq_len(k), drop = FALSE]

  explained_ss <- numeric(k)
  score_gram <- crossprod(scores)
  score_norms <- diag(score_gram)
  offdiag <- score_gram
  diag(offdiag) <- 0
  gram_scale <- max(abs(score_norms), 1)
  if (all(is.finite(score_norms)) &&
      all(score_norms > 0) &&
      max(abs(offdiag), na.rm = TRUE) <= sqrt(.Machine$double.eps) * gram_scale) {
    # SIMPLS/OPLS scores are orthogonal, so all one-dimensional projection
    # sums of squares can be obtained with one BLAS crossproduct.
    XtT <- crossprod(Xscaled, scores)
    explained_ss <- colSums(XtT * XtT) / score_norms
    explained_ss[!is.finite(explained_ss) | explained_ss < 0] <- 0
  } else {
    residual <- Xscaled
    for (j in seq_len(k)) {
      tj <- scores[, j, drop = FALSE]
      denom <- drop(crossprod(tj))
      if (!is.finite(denom) || denom <= 0) {
        explained_ss[j] <- 0
        next
      }
      before <- sum(residual * residual)
      pj <- crossprod(residual, tj) / denom
      residual <- residual - tj %*% t(pj)
      after <- sum(residual * residual)
      gain <- before - after
      explained_ss[j] <- if (is.finite(gain) && gain > 0) gain else 0
    }
  }

  denom_df <- max(1, nrow(Xscaled) - 1L)
  variance <- explained_ss / denom_df
  variance_explained <- explained_ss / total_ss
  variance <- .fastpls_named_components(variance, "LV")
  variance_explained <- .fastpls_named_components(variance_explained, "LV")
  cumulative <- .fastpls_named_components(cumsum(variance_explained), "LV")
  list(
    variance = variance,
    variance_explained = variance_explained,
    cumulative_variance_explained = cumulative,
    variance_total = total_ss / denom_df,
    variance_basis = "X"
  )
}

.maybe_attach_pls_variance_explained <- function(model, Xtrain, return_variance = TRUE) {
  if (!isTRUE(return_variance)) {
    return(model)
  }
  .attach_pls_variance_explained(model, Xtrain)
}

.attach_pls_variance_explained <- function(model, Xtrain) {
  stats <- try(.pls_x_variance_explained(model, Xtrain), silent = TRUE)
  if (inherits(stats, "try-error") || is.null(stats)) {
    return(model)
  }
  model$variance <- stats$variance
  model$variance_explained <- stats$variance_explained
  model$cumulative_variance_explained <- stats$cumulative_variance_explained
  model$variance_total <- stats$variance_total
  model$variance_basis <- stats$variance_basis
  model$x_variance <- stats$variance
  model$x_variance_explained <- stats$variance_explained
  model$x_cumulative_variance_explained <- stats$cumulative_variance_explained
  model$x_variance_total <- stats$variance_total
  model
}

.inherit_inner_variance_explained <- function(model, inner) {
  fields <- c(
    "variance",
    "variance_explained",
    "cumulative_variance_explained",
    "variance_total",
    "variance_basis",
    "x_variance",
    "x_variance_explained",
    "x_cumulative_variance_explained",
    "x_variance_total"
  )
  for (field in fields) {
    if (!is.null(inner[[field]])) {
      model[[field]] <- inner[[field]]
    }
  }
  model
}

.classifier_public_choices <- c("argmax", "lda", "cknn")
.classifier_internal_choices <- c(
  "argmax",
  "lda_cpp", "lda_cuda", "lda_metal",
  "candidate_knn_cpp", "candidate_knn_cuda", "candidate_knn_metal"
)

.normalize_classifier_public <- function(classifier) {
  if (length(classifier) > 1L) {
    classifier <- classifier[1L]
  }
  classifier <- as.character(classifier)
  if (identical(classifier, "candidate_knn")) {
    classifier <- "cknn"
  }
  match.arg(classifier, .classifier_public_choices)
}

.normalize_classifier <- function(classifier) {
  if (length(classifier) > 1L) {
    classifier <- classifier[1L]
  }
  classifier <- as.character(classifier)
  if (identical(classifier, "candidate_knn")) {
    classifier <- "cknn"
  }
  if (classifier %in% .classifier_public_choices) {
    classifier <- switch(
      classifier,
      argmax = "argmax",
      lda = "lda_cpp",
      cknn = "candidate_knn_cpp"
    )
  }
  match.arg(classifier, .classifier_internal_choices)
}

.resolve_classifier_for_backend <- function(classifier, backend) {
  if (length(classifier) > 1L) {
    classifier <- classifier[1L]
  }
  classifier <- as.character(classifier)
  if (classifier %in% .classifier_internal_choices) {
    return(.normalize_classifier(classifier))
  }
  classifier <- .normalize_classifier_public(classifier)
  backend <- .normalize_public_backend(backend)
  switch(
    classifier,
    argmax = "argmax",
    lda = switch(backend, cpu = "lda_cpp", cuda = "lda_cuda", metal = "lda_metal"),
    cknn = switch(
      backend,
      cpu = "candidate_knn_cpp",
      cuda = "candidate_knn_cuda",
      metal = "candidate_knn_metal"
    )
  )
}

.is_lda_classifier <- function(classifier) {
  !is.null(classifier) && classifier %in% c("lda_cpp", "lda_cuda", "lda_metal")
}

.is_class_bias_classifier <- function(classifier) {
  FALSE
}

.class_bias_backend <- function(classifier) {
  "cpp"
}

.is_candidate_knn_classifier <- function(classifier) {
  !is.null(classifier) &&
    classifier %in% c("candidate_knn_cpp", "candidate_knn_cuda", "candidate_knn_metal")
}

.candidate_knn_backend <- function(classifier) {
  if (identical(classifier, "candidate_knn_cuda")) {
    "cuda"
  } else if (identical(classifier, "candidate_knn_metal")) {
    "metal"
  } else {
    "cpp"
  }
}

.resolve_top_k <- function(top = 1L, top5 = FALSE) {
  top <- as.integer(top)[1L]
  if (!is.finite(top) || is.na(top) || top < 1L) {
    stop("top must be a positive integer", call. = FALSE)
  }
  if (isTRUE(top5)) top <- max(top, 5L)
  top
}

.class_bias_matrix <- function(class_bias, lev, ncomp) {
  nclass <- length(lev)
  nslice <- length(ncomp)
  if (is.null(class_bias)) {
    out <- matrix(0, nrow = nclass, ncol = nslice)
    rownames(out) <- lev
    return(out)
  }
  if (is.vector(class_bias) || is.factor(class_bias)) {
    nm <- names(class_bias)
    class_bias <- as.numeric(class_bias)
    if (!is.null(nm) && any(nzchar(nm))) {
      class_bias <- class_bias[match(lev, nm)]
    }
    if (length(class_bias) != nclass || anyNA(class_bias)) {
      stop("class_bias must have one numeric value per class", call. = FALSE)
    }
    out <- matrix(class_bias, nrow = nclass, ncol = nslice)
    rownames(out) <- lev
    return(out)
  }
  class_bias <- as.matrix(class_bias)
  if (nrow(class_bias) != nclass && ncol(class_bias) == nclass) {
    class_bias <- t(class_bias)
  }
  if (!is.null(rownames(class_bias))) {
    class_bias <- class_bias[match(lev, rownames(class_bias)), , drop = FALSE]
  }
  if (nrow(class_bias) != nclass || anyNA(class_bias)) {
    stop("class_bias must have one row per class", call. = FALSE)
  }
  if (ncol(class_bias) == 1L && nslice > 1L) {
    class_bias <- matrix(class_bias[, 1L], nrow = nclass, ncol = nslice)
  }
  if (ncol(class_bias) != nslice) {
    stop("class_bias must have one column or one column per ncomp", call. = FALSE)
  }
  rownames(class_bias) <- lev
  class_bias
}

.class_topk_to_labels <- function(top_index, top_score, lev, ncomp) {
  dims <- dim(top_index)
  labels <- array(lev[as.integer(top_index)], dim = dims)
  top1 <- as.data.frame(matrix(labels[, 1L, ], nrow = dims[1L], ncol = dims[3L]))
  colnames(top1) <- paste("ncomp=", ncomp, sep = "")
  for (j in seq_along(top1)) {
    top1[[j]] <- factor(top1[[j]], levels = lev)
  }
  out <- list(Ypred = top1)
  out$Ypred_index <- matrix(top_index[, 1L, ], nrow = dims[1L], ncol = dims[3L])
  colnames(out$Ypred_index) <- paste("ncomp=", ncomp, sep = "")
  if (dims[2L] > 1L) {
    top_list <- vector("list", dims[3L])
    score_list <- vector("list", dims[3L])
    names(top_list) <- names(score_list) <- paste("ncomp=", ncomp, sep = "")
    for (a in seq_len(dims[3L])) {
      top_list[[a]] <- matrix(
        labels[, , a],
        nrow = dims[1L],
        ncol = dims[2L],
        dimnames = list(NULL, paste0("rank", seq_len(dims[2L])))
      )
      score_list[[a]] <- matrix(
        top_score[, , a],
        nrow = dims[1L],
        ncol = dims[2L],
        dimnames = list(NULL, paste0("rank", seq_len(dims[2L])))
      )
    }
    out$Ypred_top <- top_list
    out$Ypred_top_score <- score_list
  }
  out
}

.class_topk_from_score_cube <- function(score_cube, lev, ncomp, class_bias = NULL, top = 1L) {
  dims <- dim(score_cube)
  top <- min(as.integer(top)[1L], dims[2L])
  bias <- .class_bias_matrix(class_bias, lev, ncomp)
  top_index <- array(NA_integer_, dim = c(dims[1L], top, dims[3L]))
  top_score <- array(NA_real_, dim = c(dims[1L], top, dims[3L]))
  for (a in seq_len(dims[3L])) {
    score <- sweep(score_cube[, , a, drop = FALSE][, , 1L], 2L, bias[, a], "+", check.margin = FALSE)
    if (top == 1L) {
      idx <- max.col(score, ties.method = "first")
      top_index[, 1L, a] <- idx
      top_score[, 1L, a] <- score[cbind(seq_len(nrow(score)), idx)]
    } else {
      for (i in seq_len(nrow(score))) {
        idx <- order(score[i, ], decreasing = TRUE)[seq_len(top)]
        top_index[i, , a] <- idx
        top_score[i, , a] <- score[i, idx]
      }
    }
  }
  .class_topk_to_labels(top_index, top_score, lev, ncomp)
}

.class_bias_predict <- function(model, Xtest, class_bias = NULL, top = 1L, proj = FALSE, backend = c("cpp", "cuda")) {
  backend <- match.arg(backend)
  bias <- .class_bias_matrix(class_bias, model$lev, model$ncomp)
  block_size <- model$flash_block_size
  if (is.null(block_size) || !length(block_size) || is.na(block_size)) {
    block_size <- 4096L
  }
  out <- if (identical(backend, "cuda") && isTRUE(has_cuda())) {
    pls_class_predict_topk_cuda(model, as.matrix(Xtest), bias, as.integer(top), isTRUE(proj))
  } else {
    pls_class_predict_topk_cpp(model, as.matrix(Xtest), bias, as.integer(top), isTRUE(proj), as.integer(block_size))
  }
  res <- .class_topk_to_labels(out$top_index, out$top_score, model$lev, model$ncomp)
  if (isTRUE(proj)) res$Ttest <- out$Ttest
  if (!is.null(out$predict_backend)) res$predict_backend <- out$predict_backend
  res
}

.fastpls_block_size <- function(option_name, env_name, default = 4096L) {
  value <- getOption(option_name, NULL)
  if (is.null(value)) {
    value <- Sys.getenv(env_name, unset = as.character(default))
  }
  value <- suppressWarnings(as.integer(value)[1L])
  if (!is.finite(value) || is.na(value) || value < 1L) {
    value <- as.integer(default)
  }
  value
}

.should_use_label_aware_plssvd <- function(n, q) {
  dense_y_mb <- as.numeric(n) * as.numeric(q) * 8 / 1024^2
  threshold <- suppressWarnings(as.numeric(Sys.getenv("FASTPLS_LABEL_AWARE_Y_THRESHOLD_MB", "512"))[1L])
  if (!is.finite(threshold) || threshold < 0) threshold <- 512
  isTRUE(dense_y_mb >= threshold)
}

.plssvd_label_aware_stream_model <- function(Xtrain,
                                             y_train,
                                             ncomp,
                                             scaling = 1L,
                                             backend = c("cpp", "cuda"),
                                             block_size = NULL) {
  backend <- match.arg(backend)
  Xtrain <- as.matrix(Xtrain)
  y_train <- factor(y_train)
  lev <- levels(y_train)
  y_code <- as.integer(y_train)
  n <- nrow(Xtrain)
  p <- ncol(Xtrain)
  m <- length(lev)
  if (n < 1L || p < 1L || m < 2L) {
    stop("label-aware PLSSVD requires non-empty X and at least two classes", call. = FALSE)
  }
  cap <- .cap_plssvd_ncomp(ncomp, n, p, m, warn = TRUE)
  ncomp <- as.integer(cap$ncomp)
  max_rank <- max(ncomp)
  if (is.null(block_size)) {
    block_size <- .fastpls_block_size(
      "fastPLS.label_aware_block_size",
      "FASTPLS_LABEL_AWARE_BLOCK_SIZE",
      default = 8192L
    )
  }
  block_size <- max(1L, as.integer(block_size)[1L])

  sums <- numeric(p)
  sums_sq <- numeric(p)
  for (start in seq(1L, n, by = block_size)) {
    stop <- min(n, start + block_size - 1L)
    Xb <- Xtrain[start:stop, , drop = FALSE]
    sums <- sums + colSums(Xb)
    if (as.integer(scaling) == 2L) {
      sums_sq <- sums_sq + colSums(Xb * Xb)
    }
  }
  mX <- if (as.integer(scaling) < 3L) sums / n else rep(0, p)
  vX <- rep(1, p)
  if (as.integer(scaling) == 2L) {
    centered_ss <- pmax(sums_sq - n * mX * mX, 0)
    vX <- sqrt(centered_ss / max(1L, n - 1L))
    vX[!is.finite(vX) | vX == 0] <- 1
  }

  counts <- tabulate(y_code, nbins = m)
  mY <- counts / n
  class_sums <- matrix(0, nrow = p, ncol = m)
  for (start in seq(1L, n, by = block_size)) {
    stop <- min(n, start + block_size - 1L)
    rows <- start:stop
    Xb <- Xtrain[rows, , drop = FALSE]
    if (as.integer(scaling) < 3L) {
      Xb <- sweep(Xb, 2L, mX, "-", check.margin = FALSE)
    }
    if (as.integer(scaling) == 2L) {
      Xb <- sweep(Xb, 2L, vX, "/", check.margin = FALSE)
    }
    rs <- rowsum(
      Xb,
      group = factor(y_code[rows], levels = seq_len(m)),
      reorder = FALSE
    )
    rs <- as.matrix(rs)
    if (nrow(rs) != m) {
      rs_full <- matrix(0, nrow = m, ncol = ncol(Xb))
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
  for (start in seq(1L, n, by = block_size)) {
    stop <- min(n, start + block_size - 1L)
    Xb <- Xtrain[start:stop, , drop = FALSE]
    if (as.integer(scaling) < 3L) {
      Xb <- sweep(Xb, 2L, mX, "-", check.margin = FALSE)
    }
    if (as.integer(scaling) == 2L) {
      Xb <- sweep(Xb, 2L, vX, "/", check.margin = FALSE)
    }
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
    coeff <- tryCatch(
      solve(Gk + diag(ridge, k), Dk),
      error = function(e) qr.solve(Gk + diag(ridge, k), Dk)
    )
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
    xprod_mode = "label_aware_stream",
    B_stored = FALSE,
    compact_prediction = TRUE,
    flash_svd = TRUE,
    flash_svd_backend = backend,
    predict_backend = if (identical(backend, "cuda")) "cuda_flash" else "cpu_flash",
    flash_block_size = block_size
  )
  class(model) <- "fastPLS"
  model
}

.candidate_row_l2 <- function(X) {
  X <- as.matrix(X)
  nr <- sqrt(rowSums(X * X))
  nr[!is.finite(nr) | nr == 0] <- 1
  sweep(X, 1L, nr, "/", check.margin = FALSE)
}

.candidate_top_indices <- function(scores, top_m) {
  top_m <- min(max(1L, as.integer(top_m)[1L]), ncol(scores))
  if (top_m == 1L) {
    return(matrix(max.col(scores, ties.method = "first"), ncol = 1L))
  }
  t(apply(scores, 1L, function(z) order(z, decreasing = TRUE)[seq_len(top_m)]))
}

.candidate_temp_knn_score <- function(sim, knn_k, tau) {
  if (!length(sim)) {
    return(-Inf)
  }
  kk <- min(max(1L, as.integer(knn_k)[1L]), length(sim))
  vals <- head(sort(sim, decreasing = TRUE), kk)
  tau <- as.numeric(tau)[1L]
  if (!is.finite(tau) || tau <= 0) {
    return(mean(vals))
  }
  mx <- max(vals)
  mx + tau * log(mean(exp((vals - mx) / tau)))
}

.candidate_centroids <- function(Ttrain_norm, y_codes, n_classes) {
  sums <- rowsum(Ttrain_norm, group = y_codes, reorder = FALSE)
  if (nrow(sums) < n_classes) {
    full <- matrix(0, nrow = n_classes, ncol = ncol(Ttrain_norm))
    present <- as.integer(rownames(sums))
    full[present, ] <- sums
    sums <- full
  }
  counts <- tabulate(y_codes, nbins = n_classes)
  counts[counts == 0L] <- 1L
  centroids <- sweep(sums, 1L, counts, "/", check.margin = FALSE)
  .candidate_row_l2(centroids)
}

.candidate_score_space <- function(model, T, ncomp) {
  T <- as.matrix(T)
  kk <- min(as.integer(ncomp)[1L], ncol(T))
  if (!is.finite(kk) || is.na(kk) || kk < 1L) {
    stop("candidate-kNN requires at least one latent component", call. = FALSE)
  }
  T <- T[, seq_len(kk), drop = FALSE]

  if (!identical(model$pls_method, "plssvd") ||
      is.null(model$C_latent) ||
      length(model$C_latent) == 0L) {
    return(T)
  }

  cd <- dim(model$C_latent)
  if (length(cd) != 3L || cd[1L] < kk || cd[2L] < kk || cd[3L] < 1L) {
    return(T)
  }
  model_ncomp <- as.integer(model$ncomp)
  slice <- match(kk, model_ncomp)
  if (is.na(slice)) {
    ge <- which(model_ncomp >= kk)
    slice <- if (length(ge)) ge[[1L]] else length(model_ncomp)
  }
  slice <- min(max(1L, as.integer(slice)[1L]), cd[3L])
  Ck <- model$C_latent[seq_len(kk), seq_len(kk), slice, drop = FALSE][, , 1L]
  if (!all(is.finite(Ck))) {
    return(T)
  }
  T %*% Ck
}

.candidate_knn_predict_core <- function(Ttest_norm,
                                        Ttrain_norm,
                                        y_codes,
                                        centroids,
                                        lev,
                                        knn_k = 3L,
                                        tau = 0.2,
                                        alpha = 0.5,
                                        top_m = 20L,
                                        candidate_bias = NULL,
                                        top = 1L,
                                        backend = c("cpp", "cuda", "metal")) {
  ntest <- nrow(Ttest_norm)
  n_classes <- length(lev)
  top <- min(max(1L, as.integer(top)[1L]), n_classes)
  top_m <- min(max(top, as.integer(top_m)[1L]), n_classes)
  backend <- match.arg(backend)
  if (is.null(candidate_bias)) {
    candidate_bias <- numeric(n_classes)
  }
  candidate_bias <- as.numeric(candidate_bias)
  if (length(candidate_bias) != n_classes) {
    stop("candidate_bias must have one value per class", call. = FALSE)
  }

  scorer_backend <- if (identical(backend, "metal")) "cpp" else backend
  scorer <- if (identical(scorer_backend, "cuda") && .cuda_matmul_available()) {
    candidate_knn_predict_cuda
  } else {
    candidate_knn_predict_cpp
  }
  native <- scorer(
    as.matrix(Ttest_norm),
    as.matrix(Ttrain_norm),
    as.integer(y_codes),
    as.matrix(centroids),
    candidate_bias,
    top,
    top_m,
    as.integer(knn_k)[1L],
    as.numeric(tau)[1L],
    as.numeric(alpha)[1L]
  )
  top_index <- as.matrix(native$top_index)
  top_score <- as.matrix(native$top_score)
  pred_index <- top_index[, 1L]

  list(
    Ypred = factor(lev[pred_index], levels = lev),
    Ypred_index = pred_index,
    Ypred_top = if (top > 1L) matrix(lev[top_index], nrow = ntest, ncol = top) else NULL,
    Ypred_top_score = if (top > 1L) top_score else NULL,
    top_index = top_index,
    top_score = top_score,
    predict_backend = if (identical(backend, "metal")) {
      "metal_candidate_knn_cpp"
    } else {
      native$predict_backend
    },
    n_reranked = native$n_reranked
  )
}

.fit_candidate_knn <- function(model,
                               Xtrain,
                               y_train,
                               backend = c("cpp", "cuda", "metal"),
                               knn_k = 3L,
                               tau = 0.2,
                               alpha = 0.5,
                               top_m = 20L) {
  backend <- match.arg(backend)
  if (identical(backend, "cuda") && !.cuda_matmul_available()) {
    warning("classifier='cknn' with backend='cuda' requested but CUDA projection is unavailable; using CPU cKNN.", call. = FALSE)
    backend <- "cpp"
  }
  if (!is.factor(y_train)) {
    stop("candidate-kNN classification requires factor Ytrain", call. = FALSE)
  }
  knn_k <- max(1L, as.integer(knn_k)[1L])
  tau <- as.numeric(tau)[1L]
  alpha <- as.numeric(alpha)[1L]
  top_m <- max(1L, as.integer(top_m)[1L])
  if (!is.finite(tau) || tau <= 0) stop("candidate_tau must be positive", call. = FALSE)
  if (!is.finite(alpha)) stop("candidate_alpha must be finite", call. = FALSE)

  model <- .attach_latent_projection_cache(model)
  score_backend <- if (identical(backend, "cuda")) {
    "cuda"
  } else if (identical(backend, "metal") && isTRUE(has_metal())) {
    "metal"
  } else {
    "cpu"
  }
  if (is.null(model$Ttrain) ||
      length(model$Ttrain) == 0L ||
      !all(dim(model$Ttrain) > 0L) ||
      ncol(as.matrix(model$Ttrain)) < max(as.integer(model$ncomp))) {
    model$Ttrain <- .fastpls_latent_scores(
      model,
      Xtrain,
      ncomp = max(model$ncomp),
      backend = score_backend
    )
  }
  Ttrain <- as.matrix(model$Ttrain)
  y_codes <- as.integer(factor(y_train, levels = model$lev))
  if (anyNA(y_codes)) {
    stop("candidate-kNN received labels outside the training levels", call. = FALSE)
  }

  ncomp_eff <- pmin(as.integer(model$ncomp), ncol(Ttrain))
  ncomp_eff <- pmax(ncomp_eff, 1L)
  unique_ncomp <- sort(unique(ncomp_eff))

  models <- vector("list", length(unique_ncomp))
  names(models) <- as.character(unique_ncomp)
  for (kk in unique_ncomp) {
    Tn <- .candidate_row_l2(.candidate_score_space(model, Ttrain, kk))
    cent <- .candidate_centroids(Tn, y_codes, length(model$lev))
    models[[as.character(kk)]] <- list(
      ncomp = kk,
      centroids = cent
    )
  }

  model$candidate_knn <- list(
    ncomp = unique_ncomp,
    models = models,
    Ttrain = Ttrain[, seq_len(max(unique_ncomp)), drop = FALSE],
    y_codes = y_codes,
    backend = backend,
    score_space = if (identical(model$pls_method, "plssvd")) "plssvd_prediction_latent" else "latent",
    parameters = list(
      knn_k = knn_k,
      tau = tau,
      alpha = alpha,
      top_m = top_m
    )
  )
  model
}

.candidate_knn_predictions <- function(object, Xtest, top = 1L, keep_ttest = FALSE) {
  if (is.null(object$candidate_knn) || is.null(object$candidate_knn$models)) {
    stop("This fastPLS object does not contain fitted candidate-kNN parameters", call. = FALSE)
  }
  par <- object$candidate_knn$parameters
  ncomp_eff <- pmin(as.integer(object$ncomp), max(as.integer(object$candidate_knn$ncomp)))
  ncomp_eff <- pmax(ncomp_eff, 1L)
  backend <- object$candidate_knn$backend %||% "cpp"
  use_cuda <- identical(backend, "cuda") && .cuda_matmul_available()
  use_metal <- identical(backend, "metal") && isTRUE(has_metal())
  Ttest <- .fastpls_latent_scores(
    object,
    Xtest,
    ncomp = max(ncomp_eff),
    backend = if (use_cuda) "cuda" else if (use_metal) "metal" else "cpu"
  )
  Ttrain <- as.matrix(object$candidate_knn$Ttrain)
  y_codes <- as.integer(object$candidate_knn$y_codes)
  Ypredlab <- as.data.frame(matrix(nrow = nrow(Ttest), ncol = length(object$ncomp)))
  colnames(Ypredlab) <- paste("ncomp=", object$ncomp, sep = "")
  Ypred_index <- matrix(NA_integer_, nrow = nrow(Ttest), ncol = length(object$ncomp))
  colnames(Ypred_index) <- colnames(Ypredlab)
  top <- min(max(1L, as.integer(top)[1L]), length(object$lev))
  top_list <- score_list <- vector("list", length(object$ncomp))
  names(top_list) <- names(score_list) <- colnames(Ypredlab)
  predict_backend <- character(length(object$ncomp))
  train_score_cache <- new.env(parent = emptyenv())
  test_score_cache <- new.env(parent = emptyenv())

  for (i in seq_along(object$ncomp)) {
    kk <- ncomp_eff[[i]]
    cm <- object$candidate_knn$models[[as.character(kk)]]
    if (is.null(cm)) {
      stop(sprintf("No fitted candidate-kNN classifier for ncomp=%s", kk), call. = FALSE)
    }
    cache_key <- as.character(kk)
    if (!exists(cache_key, envir = train_score_cache, inherits = FALSE)) {
      assign(cache_key, .candidate_score_space(object, Ttrain, kk), envir = train_score_cache)
      assign(cache_key, .candidate_score_space(object, Ttest, kk), envir = test_score_cache)
    }
    Ttrain_kk <- get(cache_key, envir = train_score_cache, inherits = FALSE)
    Ttest_kk <- get(cache_key, envir = test_score_cache, inherits = FALSE)
    pred <- .candidate_knn_predict_core(
      Ttest_norm = .candidate_row_l2(Ttest_kk),
      Ttrain_norm = .candidate_row_l2(Ttrain_kk),
      y_codes = y_codes,
      centroids = cm$centroids,
      lev = object$lev,
      knn_k = par$knn_k,
      tau = par$tau,
      alpha = par$alpha,
      top_m = par$top_m,
      candidate_bias = numeric(length(object$lev)),
      top = top,
      backend = backend
    )
    Ypredlab[[i]] <- pred$Ypred
    Ypred_index[, i] <- pred$Ypred_index
    predict_backend[[i]] <- pred$predict_backend %||% paste0(backend, "_candidate_knn")
    if (top > 1L) {
      top_list[[i]] <- pred$Ypred_top
      score_list[[i]] <- pred$Ypred_top_score
    }
  }

  out <- list(Ypred = Ypredlab, Ypred_index = Ypred_index)
  out$predict_backend <- unique(predict_backend[nzchar(predict_backend)])
  if (top > 1L) {
    out$Ypred_top <- top_list
    out$Ypred_top_score <- score_list
  }
  if (isTRUE(keep_ttest)) {
    out$Ttest <- Ttest
  }
  out
}

.fastpls_scaled_by_model <- function(object, X) {
  X <- as.matrix(X)
  if (!is.null(object$mX) && length(object$mX) == ncol(X)) {
    X <- sweep(X, 2L, as.numeric(object$mX), "-", check.margin = FALSE)
  }
  if (!is.null(object$vX) && length(object$vX) == ncol(X)) {
    scale <- as.numeric(object$vX)
    scale[!is.finite(scale) | scale == 0] <- 1
    X <- sweep(X, 2L, scale, "/", check.margin = FALSE)
  }
  X
}

.fastpls_latent_scores <- function(object, X, ncomp = max(object$ncomp), backend = c("cpu", "cuda", "metal")) {
  backend <- match.arg(backend)
  if (is.null(object$R) || length(object$R) == 0L) {
    stop("LDA classification requires latent projection matrix R", call. = FALSE)
  }
  R <- as.matrix(object$R)
  k <- min(as.integer(ncomp), ncol(R))
  if (!is.finite(k) || is.na(k) || k < 1L) {
    stop("LDA classification requires at least one latent component", call. = FALSE)
  }
  X <- as.matrix(X)
  if (!is.null(object$R_predict) &&
      length(object$R_predict) > 0L &&
      ncol(as.matrix(object$R_predict)) >= k &&
      nrow(as.matrix(object$R_predict)) == ncol(X)) {
    R_cached <- as.matrix(object$R_predict)[, seq_len(k), drop = FALSE]
    T <- if (identical(backend, "cuda") && .cuda_matmul_available()) {
      .cuda_matmul(X, R_cached)
    } else if (identical(backend, "metal") && isTRUE(has_metal())) {
      .metal_mm(X, R_cached)
    } else {
      X %*% R_cached
    }
    if (!is.null(object$R_offset) && length(object$R_offset) >= k) {
      offset <- as.numeric(object$R_offset)[seq_len(k)]
      if (any(offset != 0)) {
        T <- sweep(T, 2L, offset, "-", check.margin = FALSE)
      }
    }
    return(T)
  }
  R <- R[, seq_len(k), drop = FALSE]
  if (!is.null(object$vX) && length(object$vX) == nrow(R)) {
    scale <- as.numeric(object$vX)
    scale[!is.finite(scale) | scale == 0] <- 1
    R <- sweep(R, 1L, scale, "/", check.margin = FALSE)
  }
  offset <- NULL
  if (!is.null(object$mX) && length(object$mX) == nrow(R)) {
    offset <- drop(as.numeric(object$mX) %*% R)
  }
  if (identical(backend, "cuda") && .cuda_matmul_available()) {
    T <- .cuda_matmul(X, R)
  } else if (identical(backend, "metal") && isTRUE(has_metal())) {
    T <- .metal_mm(X, R)
  } else {
    T <- X %*% R
  }
  if (!is.null(offset) && any(offset != 0)) {
    T <- sweep(T, 2L, offset, "-", check.margin = FALSE)
  }
  T
}

.attach_latent_projection_cache <- function(model, ncomp = max(model$ncomp)) {
  if (is.null(model$R) || length(model$R) == 0L) {
    return(model)
  }
  R <- as.matrix(model$R)
  k <- min(as.integer(ncomp), ncol(R))
  if (!is.finite(k) || is.na(k) || k < 1L) {
    return(model)
  }
  R <- R[, seq_len(k), drop = FALSE]
  if (!is.null(model$vX) && length(model$vX) == nrow(R)) {
    scale <- as.numeric(model$vX)
    scale[!is.finite(scale) | scale == 0] <- 1
    R <- sweep(R, 1L, scale, "/", check.margin = FALSE)
  }
  offset <- rep(0, k)
  if (!is.null(model$mX) && length(model$mX) == nrow(R)) {
    offset <- drop(as.numeric(model$mX) %*% R)
  }
  model$R_predict <- R
  model$R_offset <- offset
  model
}

.fastpls_lda_predict_cuda <- function(Ttest, lda) {
  if (!.cuda_matmul_available() ||
      !exists("lda_predict_cuda", envir = asNamespace("fastPLS"), inherits = FALSE)) {
    return(lda_predict_cpp(Ttest, lda))
  }
  get("lda_predict_cuda", envir = asNamespace("fastPLS"), inherits = FALSE)(
    as.matrix(Ttest),
    lda
  )
}

.fastpls_lda_project_predict_cuda <- function(Xtest, R, offset, lda, return_scores = FALSE) {
  if (!.cuda_matmul_available() ||
      !exists("lda_project_predict_cuda", envir = asNamespace("fastPLS"), inherits = FALSE)) {
    constants <- as.numeric(lda$constants)
    linear <- as.matrix(lda$linear)
    if (length(offset) >= ncol(R)) {
      constants <- constants - drop(as.numeric(offset[seq_len(ncol(R))]) %*% t(linear))
    }
    scores <- (as.matrix(Xtest) %*% as.matrix(R)) %*% t(linear)
    scores <- sweep(scores, 2L, constants, "+", check.margin = FALSE)
    pred <- max.col(scores, ties.method = "first")
    if (isTRUE(return_scores)) {
      return(list(pred = pred, scores = scores))
    }
    return(list(pred = pred))
  }
  get("lda_project_predict_cuda", envir = asNamespace("fastPLS"), inherits = FALSE)(
    as.matrix(Xtest),
    as.matrix(R),
    as.numeric(offset),
    lda,
    isTRUE(return_scores)
  )
}

.fastpls_lda_project_predict_cpp <- function(Xtest, R, offset, lda) {
  if (!exists("lda_project_predict_labels_cpp", envir = asNamespace("fastPLS"), inherits = FALSE)) {
    Ttest <- sweep(
      as.matrix(Xtest) %*% as.matrix(R),
      2L,
      as.numeric(offset),
      "-",
      check.margin = FALSE
    )
    return(lda_predict_labels_cpp(Ttest, lda))
  }
  get("lda_project_predict_labels_cpp", envir = asNamespace("fastPLS"), inherits = FALSE)(
    as.matrix(Xtest),
    as.matrix(R),
    as.numeric(offset),
    lda
  )
}

.attach_lda_classifier <- function(model,
                                   Xtrain,
                                   Ytrain,
                                   classifier = "argmax",
                                   lda_ridge = 1e-8,
                                   candidate_knn_k = getOption("fastPLS.candidate_knn_k", 10L),
                                   candidate_tau = getOption("fastPLS.candidate_tau", 0.2),
                                   candidate_alpha = getOption("fastPLS.candidate_alpha", 0.75),
                                   candidate_top_m = getOption("fastPLS.candidate_top_m", 20L)) {
  classifier <- .resolve_classifier_for_backend(classifier, "cpu")
  model$classification_rule <- classifier
  model$lda_backend <- classifier
  if (!isTRUE(model$classification) || identical(classifier, "argmax")) {
    return(model)
  }
  if (!is.factor(Ytrain)) {
    stop("Classification head requires factor Ytrain", call. = FALSE)
  }
  if (.is_candidate_knn_classifier(classifier)) {
    backend <- .candidate_knn_backend(classifier)
    if (identical(backend, "cuda") && !.cuda_matmul_available()) {
      warning("classifier='cknn' with backend='cuda' requested but CUDA projection is unavailable; using CPU cKNN.", call. = FALSE)
      backend <- "cpp"
      model$classification_rule <- "candidate_knn_cpp"
      model$lda_backend <- "candidate_knn_cpp"
    } else if (identical(backend, "metal") && !isTRUE(has_metal())) {
      warning("classifier='cknn' with backend='metal' requested but Metal is unavailable; using CPU cKNN.", call. = FALSE)
      backend <- "cpp"
      model$classification_rule <- "candidate_knn_cpp"
      model$lda_backend <- "candidate_knn_cpp"
    }
    model <- .fit_candidate_knn(
      model,
      Xtrain,
      Ytrain,
      backend = backend,
      knn_k = candidate_knn_k,
      tau = candidate_tau,
      alpha = candidate_alpha,
      top_m = candidate_top_m
    )
    return(model)
  }
  if (identical(classifier, "lda_cuda") && !.cuda_matmul_available()) {
    warning("classifier='lda' with backend='cuda' requested but CUDA matrix multiply is unavailable; using CPU LDA.", call. = FALSE)
    classifier <- "lda_cpp"
    model$classification_rule <- classifier
    model$lda_backend <- classifier
  }
  if (identical(classifier, "lda_metal") && !isTRUE(has_metal())) {
    warning("classifier='lda' with backend='metal' requested but Metal is unavailable; using CPU LDA.", call. = FALSE)
    classifier <- "lda_cpp"
    model$classification_rule <- classifier
    model$lda_backend <- classifier
  }
  model <- .attach_latent_projection_cache(model)
  y_codes <- as.integer(factor(Ytrain, levels = model$lev))
  if (anyNA(y_codes)) {
    stop("LDA classification received labels outside the training levels", call. = FALSE)
  }

  if (identical(classifier, "lda_metal")) {
    Ttrain <- .fastpls_latent_scores(
      model,
      Xtrain,
      ncomp = max(as.integer(model$ncomp)),
      backend = "metal"
    )
    ncomp_eff <- pmin(as.integer(model$ncomp), ncol(Ttrain))
    ncomp_eff <- pmax(ncomp_eff, 1L)
    unique_ncomp <- sort(unique(ncomp_eff))
    lda_models <- lda_train_prefix_cpp(
      Ttrain[, seq_len(max(unique_ncomp)), drop = FALSE],
      y_codes,
      length(model$lev),
      unique_ncomp,
      as.numeric(lda_ridge)[1L]
    )
    names(lda_models) <- as.character(unique_ncomp)
    model$lda <- list(
      ncomp = unique_ncomp,
      models = lda_models,
      ridge = as.numeric(lda_ridge)[1L],
      train_backend = "metal_project_cpp_lda"
    )
    model$Ttrain <- Ttrain[, seq_len(max(unique_ncomp)), drop = FALSE]
    return(model)
  }

  if (identical(classifier, "lda_cpp") &&
      identical(model$flash_svd_backend, "cuda") &&
      .cuda_matmul_available()) {
    Ttrain <- .fastpls_latent_scores(
      model,
      Xtrain,
      ncomp = max(as.integer(model$ncomp)),
      backend = "cuda"
    )
    ncomp_eff <- pmin(as.integer(model$ncomp), ncol(Ttrain))
    ncomp_eff <- pmax(ncomp_eff, 1L)
    unique_ncomp <- sort(unique(ncomp_eff))
    lda_models <- lda_train_prefix_cpp(
      Ttrain[, seq_len(max(unique_ncomp)), drop = FALSE],
      y_codes,
      length(model$lev),
      unique_ncomp,
      as.numeric(lda_ridge)[1L]
    )
    names(lda_models) <- as.character(unique_ncomp)
    model$lda <- list(
      ncomp = unique_ncomp,
      models = lda_models,
      ridge = as.numeric(lda_ridge)[1L],
      train_backend = "cpp_on_cuda_scores"
    )
    return(model)
  }

  project_train_fun <- NULL
  project_train_backend <- NULL
  if (identical(classifier, "lda_cuda") &&
      .cuda_matmul_available() &&
      exists("lda_project_train_prefix_cuda", envir = asNamespace("fastPLS"), inherits = FALSE)) {
    project_train_fun <- get("lda_project_train_prefix_cuda", envir = asNamespace("fastPLS"), inherits = FALSE)
    project_train_backend <- "cuda_project"
  } else if (exists("lda_project_train_prefix_cpp", envir = asNamespace("fastPLS"), inherits = FALSE)) {
    project_train_fun <- get("lda_project_train_prefix_cpp", envir = asNamespace("fastPLS"), inherits = FALSE)
    project_train_backend <- "cpp_project"
  }

  if (!is.null(project_train_fun) &&
      !is.null(model$R_predict) &&
      !is.null(model$R_offset)) {
    R_predict <- as.matrix(model$R_predict)
    ncomp_eff <- pmin(as.integer(model$ncomp), ncol(R_predict))
    ncomp_eff <- pmax(ncomp_eff, 1L)
    unique_ncomp <- sort(unique(ncomp_eff))
    lda_models <- project_train_fun(
      as.matrix(Xtrain),
      R_predict[, seq_len(max(unique_ncomp)), drop = FALSE],
      as.numeric(model$R_offset)[seq_len(max(unique_ncomp))],
      y_codes,
      length(model$lev),
      unique_ncomp,
      as.numeric(lda_ridge)[1L]
    )
    names(lda_models) <- as.character(unique_ncomp)
    model$lda <- list(
      ncomp = unique_ncomp,
      models = lda_models,
      ridge = as.numeric(lda_ridge)[1L],
      train_backend = project_train_backend
    )
    return(model)
  }

  if (is.null(model$Ttrain) ||
      length(model$Ttrain) == 0L ||
      !all(dim(model$Ttrain) > 0L) ||
      ncol(as.matrix(model$Ttrain)) < max(as.integer(model$ncomp))) {
    model$Ttrain <- .fastpls_latent_scores(
      model,
      Xtrain,
      ncomp = max(model$ncomp),
      backend = "cpu"
    )
  }
  Ttrain <- as.matrix(model$Ttrain)

  ncomp_eff <- pmin(as.integer(model$ncomp), ncol(Ttrain))
  ncomp_eff <- pmax(ncomp_eff, 1L)
  unique_ncomp <- sort(unique(ncomp_eff))
  train_fun <- if (identical(classifier, "lda_cuda") &&
                   exists("lda_train_prefix_cuda", envir = asNamespace("fastPLS"), inherits = FALSE)) {
    get("lda_train_prefix_cuda", envir = asNamespace("fastPLS"), inherits = FALSE)
  } else {
    lda_train_prefix_cpp
  }
  lda_models <- train_fun(
    Ttrain[, seq_len(max(unique_ncomp)), drop = FALSE],
    y_codes,
    length(model$lev),
    unique_ncomp,
    as.numeric(lda_ridge)[1L]
  )
  names(lda_models) <- as.character(unique_ncomp)
  model$lda <- list(
    ncomp = unique_ncomp,
    models = lda_models,
    ridge = as.numeric(lda_ridge)[1L]
  )
  model
}

.fastpls_return_lda_scores <- function() {
  opt <- getOption("fastPLS.return_lda_scores", NULL)
  if (!is.null(opt)) {
    return(isTRUE(opt))
  }
  env <- tolower(Sys.getenv("FASTPLS_RETURN_LDA_SCORES", "false"))
  env %in% c("1", "true", "yes", "y")
}

.fastpls_one_hot_labels <- function(y, lev) {
  y <- as.character(y)
  idx <- match(y, lev)
  out <- matrix(0, nrow = length(y), ncol = length(lev))
  colnames(out) <- lev
  ok <- !is.na(idx)
  if (any(ok)) {
    out[cbind(which(ok), idx[ok])] <- 1
  }
  out
}

.fastpls_q2_from_class_labels <- function(object, Ytest, Ypredlab) {
  Ytest_transf <- .fastpls_one_hot_labels(Ytest, object$lev)
  vapply(seq_along(object$ncomp), function(i) {
    RQ(
      Ytest_transf,
      .fastpls_one_hot_labels(Ypredlab[[i]], object$lev)
    )
  }, numeric(1))
}

.try_cuda_native_lda_fit_predict <- function(method_id,
                                            method_name,
                                            Xtrain,
                                            Ytrain,
                                            Ytrain_original,
                                            Xtest,
                                            Ytest,
                                            ncomp,
                                            scaling_id,
                                            use_xprod_default,
                                            fit,
                                            proj,
                                            rsvd_oversample,
                                            rsvd_power,
                                            svds_tol,
                                            seed,
                                            lda_ridge,
                                            lev,
                                            gpu_device_state = FALSE,
                                            gpu_qr = TRUE,
                                            gpu_eig = TRUE,
                                            gpu_qless_qr = FALSE,
                                            gpu_finalize_threshold = 32L) {
  fused_enabled <- isTRUE(getOption("fastPLS.fused_cuda_lda", FALSE)) ||
    tolower(Sys.getenv("FASTPLS_FUSED_CUDA_LDA", "0")) %in% c("1", "true", "yes", "y")
  if (is.null(Xtest) ||
      !fused_enabled ||
      isTRUE(fit) ||
      isTRUE(proj) ||
      !is.factor(Ytrain_original) ||
      !has_cuda() ||
      !exists("pls_lda_gpu_native", envir = asNamespace("fastPLS"), inherits = FALSE)) {
    return(NULL)
  }
  y_codes <- as.integer(factor(Ytrain_original, levels = lev))
  if (anyNA(y_codes)) {
    return(NULL)
  }
  fit_expr <- function() {
    pls_lda_gpu_native(
      as.matrix(Xtrain),
      as.matrix(Ytrain),
      as.integer(y_codes),
      as.matrix(Xtest),
      as.integer(ncomp),
      length(lev),
      as.integer(method_id),
      as.integer(scaling_id),
      isTRUE(use_xprod_default),
      isTRUE(fit),
      as.integer(rsvd_oversample),
      as.integer(rsvd_power),
      as.numeric(svds_tol)[1L],
      as.integer(seed)[1L],
      as.numeric(lda_ridge)[1L]
    )
  }
  model <- tryCatch({
    .with_gpu_native_options(
      if (identical(as.integer(method_id), 3L) && isTRUE(use_xprod_default)) {
        .with_simpls_gpu_xprod(fit_expr())
      } else {
        fit_expr()
      },
      gpu_device_state = gpu_device_state,
      gpu_qr = gpu_qr,
      gpu_eig = gpu_eig,
      gpu_qless_qr = gpu_qless_qr,
      gpu_finalize_threshold = gpu_finalize_threshold
    )
  }, error = function(e) {
    warning("Native CUDA PLS+LDA fused path failed; falling back to standard CUDA path: ",
            conditionMessage(e), call. = FALSE)
    NULL
  })
  if (is.null(model)) {
    return(NULL)
  }
  cuda_reset_workspace()
  model$classification <- TRUE
  model$lev <- lev
  model$pls_method <- method_name
  model$predict_latent_ok <- TRUE
  model$xprod_default <- isTRUE(use_xprod_default)
  model <- .enable_flash_prediction(model, "cuda")
  model$predict_backend <- "cuda_fused_lda"
  model$flash_svd_mode <- "fused_pls_lda"
  pred_codes <- model$pred_codes
  model$pred_codes <- NULL
  if (!is.null(pred_codes)) {
    pred_codes <- as.matrix(pred_codes)
    Ypredlab <- as.data.frame(matrix(nrow = nrow(pred_codes), ncol = ncol(pred_codes)))
    colnames(Ypredlab) <- paste("ncomp=", model$ncomp, sep = "")
    for (i in seq_len(ncol(pred_codes))) {
      Ypredlab[, i] <- factor(lev[as.integer(pred_codes[, i])], levels = lev)
    }
    model$Ypred <- Ypredlab
    if (!is.null(Ytest)) {
      model$Q2Y <- .fastpls_q2_from_class_labels(model, Ytest, Ypredlab)
    }
  }
  class(model) <- "fastPLS"
  model
}

.fastpls_lda_direct_predict <- function(object,
                                        Xtest,
                                        ncomp_eff,
                                        use_cuda = FALSE,
                                        use_metal = FALSE,
                                        return_scores = FALSE) {
  if (length(unique(ncomp_eff)) != 1L ||
      is.null(object$R_predict) ||
      is.null(object$R_offset) ||
      is.null(object$lda) ||
      is.null(object$lda$models)) {
    return(NULL)
  }
  k <- as.integer(ncomp_eff[[1L]])
  lda <- object$lda$models[[as.character(k)]]
  if (is.null(lda) || is.null(lda$linear) || is.null(lda$constants)) {
    return(NULL)
  }
  R_predict <- as.matrix(object$R_predict)
  Xtest <- as.matrix(Xtest)
  linear <- as.matrix(lda$linear)
  constants <- as.numeric(lda$constants)
  if (k < 1L ||
      ncol(R_predict) < k ||
      nrow(R_predict) != ncol(Xtest) ||
      ncol(linear) != k ||
      length(constants) != nrow(linear)) {
    return(NULL)
  }

  n <- nrow(Xtest)
  p <- ncol(Xtest)
  n_classes <- nrow(linear)
  latent_ops <- as.numeric(n) * as.numeric(k) * (as.numeric(p) + as.numeric(n_classes))
  direct_ops <- as.numeric(n) * as.numeric(p) * as.numeric(n_classes)
  if (!is.finite(latent_ops) || !is.finite(direct_ops) || direct_ops >= 0.5 * latent_ops) {
    return(NULL)
  }

  Rk <- R_predict[, seq_len(k), drop = FALSE]
  W <- Rk %*% t(linear)
  offset <- as.numeric(object$R_offset)[seq_len(k)]
  constants <- constants - drop(offset %*% t(linear))
  scores <- if (isTRUE(use_cuda) && .cuda_matmul_available()) {
    .cuda_matmul(Xtest, W)
  } else if (isTRUE(use_metal) && isTRUE(has_metal())) {
    .metal_mm(Xtest, W)
  } else {
    Xtest %*% W
  }
  scores <- sweep(scores, 2L, constants, "+", check.margin = FALSE)
  pred <- max.col(scores, ties.method = "first")
  list(
    pred = pred,
    scores = if (isTRUE(return_scores)) scores else NULL,
    direct = TRUE
  )
}

.fastpls_lda_cuda_project_predict <- function(object,
                                             Xtest,
                                             ncomp_eff,
                                             return_scores = FALSE) {
  if (!identical(object$classification_rule, "lda_cuda") ||
      !.cuda_matmul_available() ||
      !exists("lda_project_predict_cuda", envir = asNamespace("fastPLS"), inherits = FALSE) ||
      is.null(object$R_predict) ||
      is.null(object$R_offset) ||
      is.null(object$lda) ||
      is.null(object$lda$models)) {
    return(NULL)
  }
  Xtest <- as.matrix(Xtest)
  R_predict <- as.matrix(object$R_predict)
  if (nrow(R_predict) != ncol(Xtest)) {
    return(NULL)
  }
  ncomp_eff <- pmax(as.integer(ncomp_eff), 1L)
  if (any(!is.finite(ncomp_eff)) || max(ncomp_eff) > ncol(R_predict)) {
    return(NULL)
  }

  Ypredlab <- as.data.frame(matrix(nrow = nrow(Xtest), ncol = length(object$ncomp)))
  colnames(Ypredlab) <- paste("ncomp=", object$ncomp, sep = "")
  score_cube <- if (isTRUE(return_scores)) {
    array(
      NA_real_,
      dim = c(nrow(Xtest), length(object$lev), length(object$ncomp)),
      dimnames = list(NULL, object$lev, NULL)
    )
  } else {
    NULL
  }

  for (i in seq_along(object$ncomp)) {
    k <- ncomp_eff[i]
    lda <- object$lda$models[[as.character(k)]]
    if (is.null(lda)) {
      return(NULL)
    }
    pred <- .fastpls_lda_project_predict_cuda(
      Xtest,
      R_predict[, seq_len(k), drop = FALSE],
      as.numeric(object$R_offset)[seq_len(k)],
      lda,
      return_scores = return_scores
    )
    Ypredlab[, i] <- factor(object$lev[as.integer(pred$pred)], levels = object$lev)
    if (isTRUE(return_scores)) {
      score_cube[, , i] <- as.matrix(pred$scores)
    }
  }

  list(Ypred = Ypredlab, lda_scores = score_cube, Ttest = NULL, direct = "cuda_project")
}

.fastpls_lda_predictions <- function(object,
                                     Xtest,
                                     Ttest = NULL,
                                     return_scores = .fastpls_return_lda_scores(),
                                     keep_ttest = FALSE) {
  if (is.null(object$lda) || is.null(object$lda$models)) {
    stop("This fastPLS object does not contain fitted LDA classifier parameters", call. = FALSE)
  }
  return_scores <- isTRUE(return_scores)
  ncomp_eff <- pmin(as.integer(object$ncomp), max(as.integer(object$lda$ncomp), na.rm = TRUE))
  ncomp_eff <- pmax(ncomp_eff, 1L)
  use_cuda <- identical(object$classification_rule, "lda_cuda") && .cuda_matmul_available()
  use_metal <- identical(object$classification_rule, "lda_metal") && isTRUE(has_metal())
  kmax <- max(ncomp_eff)
  cuda_project_res <- if (use_cuda && is.null(Ttest) && !isTRUE(keep_ttest)) {
    .fastpls_lda_cuda_project_predict(
      object,
      Xtest,
      ncomp_eff = ncomp_eff,
      return_scores = return_scores
    )
  } else {
    NULL
  }
  if (!is.null(cuda_project_res)) {
    return(cuda_project_res)
  }
  if (identical(object$classification_rule, "lda_cpp") &&
      is.null(Ttest) &&
      !isTRUE(keep_ttest) &&
      !return_scores &&
      !identical(object$flash_svd_backend, "cuda") &&
      !is.null(object$R_predict) &&
      !is.null(object$R_offset)) {
    Xtest_mat <- as.matrix(Xtest)
    R_predict <- as.matrix(object$R_predict)
    if (nrow(R_predict) == ncol(Xtest_mat) &&
        max(ncomp_eff) <= ncol(R_predict)) {
      Ypredlab <- as.data.frame(matrix(nrow = nrow(Xtest_mat), ncol = length(object$ncomp)))
      colnames(Ypredlab) <- paste("ncomp=", object$ncomp, sep = "")
      for (i in seq_along(object$ncomp)) {
        k <- ncomp_eff[i]
        lda <- object$lda$models[[as.character(k)]]
        if (is.null(lda)) {
          return(NULL)
        }
        pred <- .fastpls_lda_project_predict_cpp(
          Xtest_mat,
          R_predict[, seq_len(k), drop = FALSE],
          as.numeric(object$R_offset)[seq_len(k)],
          lda
        )
        Ypredlab[, i] <- factor(object$lev[as.integer(pred)], levels = object$lev)
      }
      return(list(Ypred = Ypredlab, lda_scores = NULL, Ttest = NULL, direct = "cpp_project"))
    }
  }
  direct_res <- if (is.null(Ttest) && !isTRUE(keep_ttest)) {
    .fastpls_lda_direct_predict(
      object,
      Xtest,
      ncomp_eff = ncomp_eff,
      use_cuda = use_cuda,
      use_metal = use_metal,
      return_scores = return_scores
    )
  } else {
    NULL
  }
  if (!is.null(direct_res)) {
    Ypredlab <- as.data.frame(matrix(nrow = nrow(as.matrix(Xtest)), ncol = length(object$ncomp)))
    colnames(Ypredlab) <- paste("ncomp=", object$ncomp, sep = "")
    for (i in seq_along(object$ncomp)) {
      Ypredlab[, i] <- factor(object$lev[as.integer(direct_res$pred)], levels = object$lev)
    }
    score_cube <- if (return_scores) {
      array(
        as.matrix(direct_res$scores),
        dim = c(nrow(as.matrix(Xtest)), length(object$lev), length(object$ncomp)),
        dimnames = list(NULL, object$lev, NULL)
      )
    } else {
      NULL
    }
    return(list(Ypred = Ypredlab, lda_scores = score_cube, Ttest = NULL, direct = TRUE))
  }
  if (is.null(Ttest) || length(Ttest) == 0L || ncol(as.matrix(Ttest)) < kmax) {
      score_backend <- if ((use_cuda || identical(object$flash_svd_backend, "cuda")) &&
	                         .cuda_matmul_available()) {
	      "cuda"
	    } else if (use_metal) {
	      "metal"
	    } else {
	      "cpu"
	    }
    Ttest <- .fastpls_latent_scores(
      object,
      Xtest,
      ncomp = kmax,
      backend = score_backend
    )
  } else {
    Ttest <- as.matrix(Ttest)[, seq_len(kmax), drop = FALSE]
  }

  Ypredlab <- as.data.frame(matrix(nrow = nrow(Ttest), ncol = length(object$ncomp)))
  colnames(Ypredlab) <- paste("ncomp=", object$ncomp, sep = "")
  score_cube <- if (return_scores) {
    array(
      NA_real_,
      dim = c(nrow(Ttest), length(object$lev), length(object$ncomp)),
      dimnames = list(NULL, object$lev, NULL)
    )
  } else {
    NULL
  }

  for (i in seq_along(object$ncomp)) {
    k <- ncomp_eff[i]
    lda <- object$lda$models[[as.character(k)]]
    if (is.null(lda)) {
      stop(sprintf("No fitted LDA classifier for ncomp=%s", k), call. = FALSE)
    }
    pred <- if (return_scores && use_cuda) {
      .fastpls_lda_predict_cuda(Ttest[, seq_len(k), drop = FALSE], lda)
    } else if (return_scores) {
      lda_predict_cpp(Ttest[, seq_len(k), drop = FALSE], lda)
    } else if (use_cuda &&
               exists("lda_predict_labels_cuda", envir = asNamespace("fastPLS"), inherits = FALSE)) {
      get("lda_predict_labels_cuda", envir = asNamespace("fastPLS"), inherits = FALSE)(
        Ttest[, seq_len(k), drop = FALSE],
        lda
      )
    } else if (exists("lda_predict_labels_cpp", envir = asNamespace("fastPLS"), inherits = FALSE)) {
      get("lda_predict_labels_cpp", envir = asNamespace("fastPLS"), inherits = FALSE)(
        Ttest[, seq_len(k), drop = FALSE],
        lda
      )
    } else {
      lda_predict_cpp(Ttest[, seq_len(k), drop = FALSE], lda)$pred
    }
    if (return_scores) {
      Ypredlab[, i] <- factor(object$lev[as.integer(pred$pred)], levels = object$lev)
      score_cube[, , i] <- as.matrix(pred$scores)
    } else {
      Ypredlab[, i] <- factor(object$lev[as.integer(pred)], levels = object$lev)
    }
  }

  list(Ypred = Ypredlab, lda_scores = score_cube, Ttest = Ttest)
}

.should_use_cpu_flash_prediction <- function(object, Xtest) {
  if (!isTRUE(object$flash_svd) || !identical(object$predict_backend, "cpu_flash")) {
    return(FALSE)
  }
  if (is.null(object$B)) {
    return(TRUE)
  }
  p <- suppressWarnings(as.numeric(ncol(Xtest)))
  m <- suppressWarnings(as.numeric(object$m))
  k <- suppressWarnings(max(as.integer(object$ncomp), na.rm = TRUE))
  if (!is.finite(p) || !is.finite(m) || !is.finite(k) || p <= 0 || m <= 0 || k <= 0) {
    return(FALSE)
  }
  dense_b_mb <- p * m * 8 / 1024^2
  min_b_mb <- suppressWarnings(as.numeric(Sys.getenv("FASTPLS_PREDICT_LATENT_MIN_B_MB", "256")))
  if (!is.finite(min_b_mb) || min_b_mb < 0) {
    min_b_mb <- 256
  }
  if (dense_b_mb >= min_b_mb) {
    return(TRUE)
  }
  # For small response dimension, dense X %*% B is often faster than X %*% R_k %*% W_k.
  k <= m
}

.normalize_svd_method <- function(method) {
  method
}

.normalize_public_backend <- function(backend) {
  if (length(backend) > 1L) {
    backend <- backend[[1L]]
  }
  backend <- as.character(backend)
  if (identical(backend, "cpp")) {
    backend <- "cpu"
  }
  match.arg(backend, c("cpu", "cuda", "metal"))
}

.compiled_backend <- function(backend) {
  backend <- .normalize_public_backend(backend)
  if (identical(backend, "cpu")) "cpp" else backend
}

.svd_control_defaults <- function() {
  list(
    svd.method = "irlba",
    rsvd_oversample = 10L,
    rsvd_power = 1L,
    svds_tol = 0,
    irlba_work = 0L,
    irlba_maxit = 1000L,
    irlba_tol = 1e-5,
    irlba_eps = 1e-9,
    irlba_svtol = 1e-5,
    seed = 1L
  )
}

.normalize_svd_parameter_list <- function(x, accepted, aliases, label) {
  if (is.null(x)) {
    x <- list()
  }
  if (!is.list(x) || is.data.frame(x)) {
    stop(sprintf("%s must be a named list.", label), call. = FALSE)
  }
  if (length(x) && (is.null(names(x)) || any(!nzchar(names(x))))) {
    stop(sprintf("All entries in %s must be named.", label), call. = FALSE)
  }
  if (length(x)) {
    nm <- names(x)
    hit <- match(nm, names(aliases), nomatch = 0L)
    nm[hit > 0L] <- unname(aliases[hit])
    names(x) <- nm
  }
  duplicated_names <- unique(names(x)[duplicated(names(x))])
  if (length(duplicated_names)) {
    stop(
      sprintf(
        "SVD control value%s supplied more than once in %s: %s",
        if (length(duplicated_names) == 1L) "" else "s",
        label,
        paste(duplicated_names, collapse = ", ")
      ),
      call. = FALSE
    )
  }
  unknown <- setdiff(names(x), accepted)
  if (length(unknown)) {
    stop(
      sprintf(
        "Unknown entr%s in %s: %s",
        if (length(unknown) == 1L) "y" else "ies",
        label,
        paste(unknown, collapse = ", ")
      ),
      call. = FALSE
    )
  }
  x
}

.svd_direct_aliases <- function() {
  c(
    oversample = "rsvd_oversample",
    power = "rsvd_power",
    work = "irlba_work",
    maxit = "irlba_maxit",
    tol = "irlba_tol",
    eps = "irlba_eps",
    svtol = "irlba_svtol"
  )
}

.svd_control_from_dots <- function(dots) {
  if (!is.list(dots)) {
    dots <- list()
  }
  list(dots = dots)
}

.resolve_svd_control <- function(svd.method = NULL,
                                 dots = list(),
                                 context = "pls()") {
  if (!is.list(dots)) {
    dots <- list()
  }

  defaults <- .svd_control_defaults()
  accepted <- names(defaults)
  dots <- .normalize_svd_parameter_list(
    dots,
    accepted = accepted,
    aliases = .svd_direct_aliases(),
    label = sprintf("... in %s", context)
  )
  direct <- list()
  if (!is.null(svd.method)) {
    direct$svd.method <- svd.method
  }

  direct_dots <- names(dots)
  supplied_sources <- list(
    direct = names(direct),
    dots = direct_dots
  )
  supplied_flat <- unlist(supplied_sources, use.names = FALSE)
  duplicated <- unique(supplied_flat[duplicated(supplied_flat)])
  if (length(duplicated)) {
    stop(
      sprintf(
        "SVD control value%s supplied more than once in %s: %s",
        if (length(duplicated) == 1L) "" else "s",
        context,
        paste(duplicated, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  out <- defaults
  if (length(direct)) {
    out[names(direct)] <- direct
  }
  if (length(direct_dots)) {
    out[direct_dots] <- dots[direct_dots]
  }
  supplied <- unique(supplied_flat)

  out$svd.method <- as.character(out$svd.method)[1L]
  if (identical(out$svd.method, "rsvd")) {
    out$svd.method <- "cpu_rsvd"
  }
  out$rsvd_oversample <- as.integer(out$rsvd_oversample)[1L]
  out$rsvd_power <- as.integer(out$rsvd_power)[1L]
  out$svds_tol <- as.numeric(out$svds_tol)[1L]
  out$irlba_work <- as.integer(out$irlba_work)[1L]
  out$irlba_maxit <- as.integer(out$irlba_maxit)[1L]
  out$irlba_tol <- as.numeric(out$irlba_tol)[1L]
  out$irlba_eps <- as.numeric(out$irlba_eps)[1L]
  out$irlba_svtol <- as.numeric(out$irlba_svtol)[1L]
  out$seed <- as.integer(out$seed)[1L]
  out$supplied <- supplied
  out
}

.should_use_xprod_default <- function(p, q, ncomp) {
  p <- as.numeric(p)
  q <- as.numeric(q)
  ncomp <- suppressWarnings(max(as.integer(ncomp), na.rm = TRUE))
  if (!is.finite(p) || !is.finite(q) || !is.finite(ncomp)) {
    return(FALSE)
  }
  s_mb <- p * q * 8 / 1024^2
  isTRUE(s_mb > 32) || (isTRUE(q >= 100) && isTRUE(ncomp <= 10))
}

.should_use_xprod_irlba_default <- function(n, p, q, ncomp) {
  n <- as.numeric(n)
  p <- as.numeric(p)
  q <- as.numeric(q)
  ncomp <- suppressWarnings(max(as.integer(ncomp), na.rm = TRUE))
  if (!is.finite(n) || !is.finite(p) || !is.finite(q) || !is.finite(ncomp)) {
    return(FALSE)
  }
  s_mb <- p * q * 8 / 1024^2
  isTRUE(s_mb > 32) && isTRUE(n >= 10000) && isTRUE(min(p, q) >= 1000)
}

.should_store_coefficients <- function(p, q, nslices = 1L, compact_prediction_available = TRUE) {
  mode <- tolower(Sys.getenv("FASTPLS_STORE_B", unset = "auto"))
  if (mode %in% c("always", "1", "true", "yes")) {
    return(TRUE)
  }
  if (mode %in% c("never", "0", "false", "no")) {
    return(FALSE)
  }
  if (!isTRUE(compact_prediction_available)) {
    return(TRUE)
  }
  max_mb <- suppressWarnings(as.numeric(Sys.getenv("FASTPLS_STORE_B_MAX_MB", unset = "256")))
  if (!is.finite(max_mb) || max_mb < 0) {
    max_mb <- 256
  }
  b_mb <- as.numeric(p) * as.numeric(q) * max(1L, as.integer(nslices)) * 8 / 1024^2
  isTRUE(b_mb <= max_mb)
}

.annotate_coefficient_storage <- function(model, store_B) {
  model$B_stored <- isTRUE(store_B)
  model$compact_prediction <- !isTRUE(store_B)
  model
}

.with_fastpls_seed <- function(seed, expr) {
  old_exists <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  old_seed <- if (old_exists) get(".Random.seed", envir = .GlobalEnv, inherits = FALSE) else NULL
  on.exit({
    if (old_exists) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)
  set.seed(as.integer(seed)[1L])
  force(expr)
}

.gaussian_y_default_dim <- function(Xtrain, gaussian_y_dim = NULL) {
  if (is.null(gaussian_y_dim) || length(gaussian_y_dim) == 0L || is.na(gaussian_y_dim[1L])) {
    return(max(1L, min(as.integer(ncol(as.matrix(Xtrain))), 100L)))
  }
  dim <- as.integer(gaussian_y_dim[1L])
  if (!is.finite(dim) || is.na(dim) || dim < 1L) {
    stop("gaussian_y_dim must be a positive integer", call. = FALSE)
  }
  dim
}

.cuda_matmul_available <- function() {
  exists("cuda_matrix_multiply", envir = asNamespace("fastPLS"), inherits = FALSE) &&
    isTRUE(has_cuda())
}

.cuda_matmul <- function(A, B) {
  get("cuda_matrix_multiply", envir = asNamespace("fastPLS"), inherits = FALSE)(
    as.matrix(A),
    as.matrix(B)
  )
}

.gaussian_y_ridge_decoder <- function(Z, Yc, backend = c("cpu", "cuda")) {
  backend <- match.arg(backend)
  ztz <- crossprod(Z)
  ridge <- 1e-8 * mean(diag(ztz))
  if (!is.finite(ridge) || ridge <= 0) {
    ridge <- 1e-8
  }
  rhs <- if (identical(backend, "cuda") && .cuda_matmul_available()) {
    .cuda_matmul(t(Z), Yc)
  } else {
    crossprod(Z, Yc)
  }
  solve(ztz + diag(ridge, ncol(ztz)), rhs)
}

.gaussian_y_class_codes <- function(lev, dim, seed) {
  codes <- .with_fastpls_seed(
    seed,
    matrix(rnorm(length(lev) * dim), nrow = length(lev), ncol = dim) / sqrt(dim)
  )
  codes <- sweep(codes, 2L, colMeans(codes), "-", check.margin = FALSE)
  sds <- sqrt(colSums(codes * codes))
  sds[!is.finite(sds) | sds == 0] <- 1
  codes <- sweep(codes, 2L, sds, "/", check.margin = FALSE)
  row_norm <- sqrt(rowSums(codes * codes))
  row_norm[!is.finite(row_norm) | row_norm == 0] <- 1
  codes <- sweep(codes, 1L, row_norm, "/", check.margin = FALSE)
  rownames(codes) <- lev
  colnames(codes) <- sprintf("gaussian_y_%03d", seq_len(ncol(codes)))
  codes
}

.prepare_gaussian_y <- function(Ytrain,
                                Xtrain,
                                gaussian_y = FALSE,
                                gaussian_y_dim = NULL,
                                seed = 1L,
                                backend = c("cpu", "cuda")) {
  backend <- match.arg(backend)
  classification <- is.factor(Ytrain)
  lev <- if (classification) levels(Ytrain) else NULL

  if (!isTRUE(gaussian_y)) {
    Ymat <- if (classification) transformy(Ytrain) else as.matrix(Ytrain)
    return(list(
      Ytrain = Ymat,
      classification = classification,
      lev = lev,
      gaussian = NULL
    ))
  }

  dim <- .gaussian_y_default_dim(Xtrain, gaussian_y_dim)
  if (classification) {
    codes <- .gaussian_y_class_codes(lev, dim, seed)
    Ymat <- codes[as.integer(Ytrain), , drop = FALSE]
    return(list(
      Ytrain = Ymat,
      classification = TRUE,
      lev = lev,
      gaussian = list(
        enabled = TRUE,
        task = "classification",
        dim = ncol(Ymat),
        original_dim = length(lev),
        codes = codes,
        backend = "cpu"
      )
    ))
  }

  Y <- as.matrix(Ytrain)
  y_center <- colMeans(Y)
  Yc <- sweep(Y, 2L, y_center, "-", check.margin = FALSE)
  projection <- .with_fastpls_seed(
    seed,
    matrix(rnorm(ncol(Y) * dim), nrow = ncol(Y), ncol = dim) / sqrt(dim)
  )
  use_cuda <- identical(backend, "cuda") && .cuda_matmul_available()
  Z <- if (use_cuda) .cuda_matmul(Yc, projection) else Yc %*% projection
  decoder <- .gaussian_y_ridge_decoder(Z, Yc, backend = if (use_cuda) "cuda" else "cpu")
  colnames(Z) <- sprintf("gaussian_y_%03d", seq_len(ncol(Z)))
  list(
    Ytrain = Z,
    classification = FALSE,
    lev = NULL,
    gaussian = list(
      enabled = TRUE,
      task = "regression",
      dim = ncol(Z),
      original_dim = ncol(Y),
      y_center = matrix(y_center, nrow = 1L),
      decoder = decoder,
      backend = if (use_cuda) "cuda" else "cpu"
    )
  )
}

.attach_gaussian_y <- function(model, spec) {
  if (is.null(spec)) {
    model$gaussian_y <- FALSE
    return(model)
  }
  model$gaussian_y <- TRUE
  model$gaussian_y_task <- spec$task
  model$gaussian_y_dim <- spec$dim
  model$gaussian_y_original_dim <- spec$original_dim
  model$gaussian_y_backend <- spec$backend
  if (identical(spec$task, "classification")) {
    model$gaussian_y_codes <- spec$codes
  } else {
    model$gaussian_y_center <- spec$y_center
    model$gaussian_y_decoder <- spec$decoder
  }
  model
}

.decode_gaussian_y_matrix <- function(object, Z) {
  Z <- as.matrix(Z)
  if (!isTRUE(object$gaussian_y)) {
    return(Z)
  }
  if (identical(object$gaussian_y_task, "classification")) {
    codes <- as.matrix(object$gaussian_y_codes)
    scores <- 2 * (Z %*% t(codes))
    scores <- sweep(scores, 2L, rowSums(codes * codes), "-", check.margin = FALSE)
    colnames(scores) <- rownames(codes)
    return(scores)
  }
  decoder <- as.matrix(object$gaussian_y_decoder)
  Y <- if (identical(object$gaussian_y_backend, "cuda") && .cuda_matmul_available()) {
    .cuda_matmul(Z, decoder)
  } else {
    Z %*% decoder
  }
  sweep(Y, 2L, as.numeric(object$gaussian_y_center[1L, ]), "+", check.margin = FALSE)
}

.decode_gaussian_y_cube <- function(object, Ycube) {
  if (!isTRUE(object$gaussian_y) || is.null(Ycube)) {
    return(Ycube)
  }
  dims <- dim(Ycube)
  if (length(dims) == 2L) {
    return(.decode_gaussian_y_matrix(object, Ycube))
  }
  if (length(dims) != 3L) {
    stop("Gaussian response decoder expected a matrix or 3D prediction array", call. = FALSE)
  }
  first <- .decode_gaussian_y_matrix(object, Ycube[, , 1L, drop = TRUE])
  out <- array(NA_real_, dim = c(nrow(first), ncol(first), dims[3L]))
  out[, , 1L] <- first
  if (dims[3L] >= 2L) {
    for (i in seq.int(2L, dims[3L])) {
      out[, , i] <- .decode_gaussian_y_matrix(object, Ycube[, , i, drop = TRUE])
    }
  }
  dimnames(out) <- list(NULL, colnames(first), NULL)
  out
}

.decode_gaussian_y_outputs <- function(model, original_Ytrain = NULL) {
  if (!isTRUE(model$gaussian_y)) {
    return(model)
  }
  if (!is.null(model$Yfit) && length(model$Yfit) > 0L && length(dim(model$Yfit)) >= 2L) {
    model$Yfit <- .decode_gaussian_y_cube(model, model$Yfit)
    if (!is.null(original_Ytrain) && !isTRUE(model$classification)) {
      model$R2Y <- vapply(seq_len(dim(model$Yfit)[3L]), function(i) {
        RQ(as.matrix(original_Ytrain), model$Yfit[, , i])
      }, numeric(1))
    }
  }
  model
}

.normalize_pls_method <- function(method) {
  method <- match.arg(method, c("simpls", "plssvd", "opls", "kernelpls"))
  switch(
    method,
    plssvd = 1L,
    simpls = 3L,
    opls = 4L,
    kernelpls = 5L
  )
}

.resolve_simpls_fast_rsvd_tuning <- function(n, p, q, svd.method) {
  stopifnot(length(n) == 1L, length(p) == 1L, length(q) == 1L, length(svd.method) == 1L)
  n <- as.integer(n)
  p <- as.integer(p)
  q <- as.integer(q)

  if (identical(svd.method, "cpu_rsvd")) {
    if (p >= 700L && n >= 20000L) {
      return(list(rsvd_oversample = 16L, rsvd_power = 0L))
    }
    if (p <= 128L && n >= 10000L) {
      return(list(rsvd_oversample = 8L, rsvd_power = 0L))
    }
    if (p >= 900L && n <= 5000L) {
      return(list(rsvd_oversample = 4L, rsvd_power = 2L))
    }
    if (p > n) {
      return(list(rsvd_oversample = 10L, rsvd_power = 2L))
    }
    if (p >= 512L) {
      return(list(rsvd_oversample = 10L, rsvd_power = 1L))
    }
    return(list(rsvd_oversample = 8L, rsvd_power = 1L))
  }

  if (identical(svd.method, "cuda_rsvd")) {
    if (p >= 700L && n >= 20000L) {
      return(list(rsvd_oversample = 4L, rsvd_power = 2L))
    }
    if (p <= 128L && n >= 10000L) {
      return(list(rsvd_oversample = 16L, rsvd_power = 1L))
    }
    if (p >= 900L && n <= 5000L) {
      return(list(rsvd_oversample = 16L, rsvd_power = 2L))
    }
    if (p > n) {
      return(list(rsvd_oversample = 8L, rsvd_power = 1L))
    }
    if (p >= 512L) {
      return(list(rsvd_oversample = 10L, rsvd_power = 2L))
    }
    return(list(rsvd_oversample = 4L, rsvd_power = 2L))
  }

  list(
    rsvd_oversample = as.integer(10L),
    rsvd_power = as.integer(1L)
  )
}

pls.model1 =
  function (Xtrain,
            Ytrain,
            ncomp,
            fit = FALSE,
            scaling = 1,
            svd.method = 1,
            rsvd_oversample = 10L,
            rsvd_power = 1L,
            svds_tol = 0,
            irlba_work = 0L,
            irlba_maxit = 1000L,
            irlba_tol = 1e-5,
            irlba_eps = 1e-9,
            irlba_svtol = 1e-5,
            seed = 1L)
  {
    Xtrain <- as.matrix(Xtrain)
    Ytrain <- as.matrix(Ytrain)
    cap <- .cap_plssvd_ncomp(ncomp, nrow(Xtrain), ncol(Xtrain), ncol(Ytrain), warn = TRUE)
    model <- .with_irlba_options(
      pls_model1(
        Xtrain,
        Ytrain,
        cap$ncomp,
        scaling,
        fit,
        svd.method,
        rsvd_oversample,
        rsvd_power,
        svds_tol,
        seed
      ),
      irlba_work = irlba_work,
      irlba_maxit = irlba_maxit,
      irlba_tol = irlba_tol,
      irlba_eps = irlba_eps,
      irlba_svtol = irlba_svtol
    )
    model$pls_method <- "plssvd"
    model$predict_latent_ok <- TRUE
    class(model) = "fastPLS"
    model
  }

pls.model1.gpu =
  function (Xtrain,
            Ytrain,
            ncomp,
            fit = FALSE,
            scaling = 1,
            rsvd_oversample = 10L,
            rsvd_power = 1L,
            svds_tol = 0,
            seed = 1L)
  {
    if (!has_cuda()) {
      stop("pls.model1.gpu requires CUDA support")
    }
    Xtrain <- as.matrix(Xtrain)
    Ytrain <- as.matrix(Ytrain)
    cap <- .cap_plssvd_ncomp(ncomp, nrow(Xtrain), ncol(Xtrain), ncol(Ytrain), warn = TRUE)
    model <- pls_model1_gpu(
      Xtrain,
      Ytrain,
      cap$ncomp,
      scaling,
      fit,
      rsvd_oversample,
      rsvd_power,
      svds_tol,
      seed
    )
    model$pls_method <- "plssvd"
    model$predict_latent_ok <- TRUE
    class(model) = "fastPLS"
    model
  }

pls.model1.gpu.implicit.xprod =
  function (Xtrain,
            Ytrain,
            ncomp,
            fit = FALSE,
            scaling = 1,
            rsvd_oversample = 10L,
            rsvd_power = 1L,
            svds_tol = 0,
            seed = 1L)
  {
    if (!has_cuda()) {
      stop("pls.model1.gpu.implicit.xprod requires CUDA support")
    }
    Xtrain <- as.matrix(Xtrain)
    Ytrain <- as.matrix(Ytrain)
    cap <- .cap_plssvd_ncomp(ncomp, nrow(Xtrain), ncol(Xtrain), ncol(Ytrain), warn = TRUE)
    model <- pls_model1_gpu_implicit_xprod(
      Xtrain,
      Ytrain,
      cap$ncomp,
      scaling,
      fit,
      rsvd_oversample,
      rsvd_power,
      svds_tol,
      seed
    )
    model$pls_method <- "plssvd"
    model$predict_latent_ok <- TRUE
    class(model) = "fastPLS"
    model
  }

pls.model2 =
  function (Xtrain,
            Ytrain,
            ncomp,
            fit = FALSE,
            scaling = 1,
            svd.method = 1,
            rsvd_oversample = 10L,
            rsvd_power = 1L,
            svds_tol = 0,
            irlba_work = 0L,
            irlba_maxit = 1000L,
            irlba_tol = 1e-5,
            irlba_eps = 1e-9,
            irlba_svtol = 1e-5,
            seed = 1L)
  {
    model <- .with_irlba_options(
      pls_model2(
        Xtrain,
        Ytrain,
        ncomp,
        scaling,
        fit,
        svd.method,
        rsvd_oversample,
        rsvd_power,
        svds_tol,
        seed
      ),
      irlba_work = irlba_work,
      irlba_maxit = irlba_maxit,
      irlba_tol = irlba_tol,
      irlba_eps = irlba_eps,
      irlba_svtol = irlba_svtol
    )
    model$pls_method <- "simpls"
    model$predict_latent_ok <- TRUE
    class(model) = "fastPLS"
    model
  }

pls.model2.fast =
  function (Xtrain,
            Ytrain,
            ncomp,
            fit = FALSE,
            scaling = 1,
            svd.method = 1,
            rsvd_oversample = 10L,
            rsvd_power = 1L,
            svds_tol = 0,
            irlba_work = 0L,
            irlba_maxit = 1000L,
            irlba_tol = 1e-5,
            irlba_eps = 1e-9,
            irlba_svtol = 1e-5,
            seed = 1L,
            return_ttrain = FALSE)
  {
    model <- .with_irlba_options(
      .with_fastpls_fast_options(
        pls_model2_fast(
          Xtrain,
          Ytrain,
          ncomp,
          scaling,
          fit,
          svd.method,
          rsvd_oversample,
          rsvd_power,
          svds_tol,
          seed
        ),
        return_ttrain = return_ttrain
      ),
      irlba_work = irlba_work,
      irlba_maxit = irlba_maxit,
      irlba_tol = irlba_tol,
      irlba_eps = irlba_eps,
      irlba_svtol = irlba_svtol
    )
    model$pls_method <- "simpls"
    model$predict_latent_ok <- TRUE
    class(model) = "fastPLS"
    model
  }

pls.model1.rsvd.xprod.precision =
  function (Xtrain,
            Ytrain,
            ncomp,
            fit = FALSE,
            scaling = 1,
            rsvd_oversample = 10L,
            rsvd_power = 1L,
            svds_tol = 0,
            irlba_work = 0L,
            irlba_maxit = 1000L,
            irlba_tol = 1e-5,
            irlba_eps = 1e-9,
            irlba_svtol = 1e-5,
            seed = 1L,
            xprod_precision = c("implicit64", "implicit_irlba", "double"))
  {
    xprod_precision <- match.arg(xprod_precision)
    precision_id <- switch(
      xprod_precision,
      double = 0L,
      implicit64 = 3L,
      implicit_irlba = 5L
    )
    Xtrain <- as.matrix(Xtrain)
    Ytrain <- as.matrix(Ytrain)
    cap <- .cap_plssvd_ncomp(ncomp, nrow(Xtrain), ncol(Xtrain), ncol(Ytrain), warn = TRUE)
    model <- .with_irlba_options(
      pls_model1_rsvd_xprod_precision(
        Xtrain,
        Ytrain,
        cap$ncomp,
        scaling,
        fit,
        as.integer(rsvd_oversample),
        as.integer(rsvd_power),
        svds_tol,
        as.integer(seed),
        as.integer(precision_id)
      ),
      irlba_work = irlba_work,
      irlba_maxit = irlba_maxit,
      irlba_tol = irlba_tol,
      irlba_eps = irlba_eps,
      irlba_svtol = irlba_svtol
    )
    model$pls_method <- "plssvd"
    model$predict_latent_ok <- TRUE
    class(model) = "fastPLS"
    model
  }

pls.model2.fast.rsvd.xprod.precision =
  function (Xtrain,
            Ytrain,
            ncomp,
            fit = FALSE,
            scaling = 1,
            rsvd_oversample = 10L,
            rsvd_power = 1L,
            svds_tol = 0,
            irlba_work = 0L,
            irlba_maxit = 1000L,
            irlba_tol = 1e-5,
            irlba_eps = 1e-9,
            irlba_svtol = 1e-5,
            seed = 1L,
            xprod_precision = c("implicit64", "implicit_irlba", "double"),
            return_ttrain = FALSE)
  {
    xprod_precision <- match.arg(xprod_precision)
    precision_id <- switch(
      xprod_precision,
      double = 0L,
      implicit64 = 3L,
      implicit_irlba = 5L
    )
    model <- .with_fastpls_fast_options(
      .with_irlba_options(
        pls_model2_fast_rsvd_xprod_precision(
          as.matrix(Xtrain),
          as.matrix(Ytrain),
          as.integer(ncomp),
          scaling,
          fit,
          as.integer(rsvd_oversample),
          as.integer(rsvd_power),
          svds_tol,
          as.integer(seed),
          as.integer(precision_id)
        ),
        irlba_work = irlba_work,
        irlba_maxit = irlba_maxit,
        irlba_tol = irlba_tol,
        irlba_eps = irlba_eps,
        irlba_svtol = irlba_svtol
      ),
      return_ttrain = return_ttrain
    )
    model$pls_method <- "simpls"
    model$predict_latent_ok <- TRUE
    class(model) = "fastPLS"
    model
  }

pls.model2.fast.gpu =
  function (Xtrain,
            Ytrain,
            ncomp,
            fit = FALSE,
            scaling = 1,
            rsvd_oversample = 10L,
            rsvd_power = 1L,
            svds_tol = 0,
            seed = 1L)
  {
    if (!has_cuda()) {
      stop("pls.model2.fast.gpu requires CUDA support")
    }
    model <- .with_fastpls_fast_options(
      pls_model2_fast_gpu(
        Xtrain,
        Ytrain,
        ncomp,
        scaling,
        fit,
        .svd_method_id("cuda_rsvd"),
        rsvd_oversample,
        rsvd_power,
        svds_tol,
        seed
      )
    )
    model$pls_method <- "simpls"
    model$predict_latent_ok <- TRUE
    class(model) = "fastPLS"
    model
  }


#' Predict from fitted fastPLS models
#'
#' Applies stored preprocessing and coefficient factors to produce test
#' predictions. For kernel PLS and OPLS fits, the corresponding preprocessing or
#' filtering step is applied before dispatching to the inner `fastPLS`
#' predictor. For classification models, response scores are converted to labels
#' by argmax over response columns unless an LDA classifier was fitted.
#'
#' @param object A fitted `fastPLS`, `fastPLSKernel`, or `fastPLSOpls` object.
#' @param newdata Numeric predictor matrix.
#' @param Ytest Optional observed response used to compute `Q2Y`.
#' @param proj Logical; return projected `Ttest` when `TRUE`.
#' @param backend Prediction backend. \code{auto} uses FlashSVD-style
#'   low-rank prediction when compact factors are available and the low-rank
#'   application is expected to be beneficial.
#' @param flash.block_size Row block size for \code{cpu_flash} prediction.
#' @param top Number of ranked classes to return for classification.
#' @param top5 Convenience flag equivalent to `top = max(top, 5)`.
#' @param raw_scores If `TRUE`, keep raw classification score cubes as
#'   `Yscore` when available.
#' @param ... Unused.
#' @return A list containing `Ypred`, optional `Q2Y`, optional `Ttest`, and
#'   optional LDA scores for LDA classification models.
#' @examples
#' X <- as.matrix(mtcars[, c("disp", "hp", "wt", "qsec")])
#' y <- mtcars$mpg
#' fit <- pls(X, y, ncomp = 2, method = "simpls", backend = "cpu",
#'            svd.method = "rsvd", return_variance = FALSE)
#' pred <- predict(fit, X[1:3, , drop = FALSE])
#' pred$Ypred
#' @export
predict.fastPLS = function(object, newdata, Ytest=NULL, proj=FALSE,
                           backend = c("auto", "cpu", "cpu_flash", "cuda_flash", "metal"),
                           flash.block_size = NULL, top = 1L, top5 = FALSE,
                           raw_scores = FALSE, ...) {
  if (!is(object, "fastPLS")) {
    stop("object is not a fastPLS object")
  }
  backend <- match.arg(backend)
  top <- .resolve_top_k(top, top5)
  Xtest=as.matrix(newdata)
  use_cuda_flash <- identical(backend, "cuda_flash") ||
    (identical(backend, "auto") &&
       identical(object$predict_backend, "cuda_flash") &&
       isTRUE(has_cuda()))
  use_cpu_flash <- identical(backend, "cpu_flash") ||
    (identical(backend, "auto") &&
       .should_use_cpu_flash_prediction(object, Xtest))
  use_metal <- (identical(backend, "metal") ||
    (identical(backend, "auto") &&
      identical(object$predict_backend, "metal"))) &&
    isTRUE(has_metal())
  if (is.null(flash.block_size)) {
    flash.block_size <- object$flash_block_size
  }
  if (is.null(flash.block_size) || !length(flash.block_size) || is.na(flash.block_size)) {
    flash.block_size <- 4096L
  }
	  if (isTRUE(object$classification) &&
	      !is.null(object$classification_rule) &&
	      .is_lda_classifier(object$classification_rule)) {
    lda_res <- .fastpls_lda_predictions(
      object,
      Xtest,
      return_scores = isTRUE(raw_scores) || top > 1L,
      keep_ttest = isTRUE(proj)
    )
    res <- list(Ypred = lda_res$Ypred, Q2Y = NULL)
    if (!is.null(lda_res$lda_scores)) {
      if (isTRUE(raw_scores)) {
        res$LDA_scores <- lda_res$lda_scores
      }
      if (top > 1L) {
        top_res <- .class_topk_from_score_cube(lda_res$lda_scores, object$lev, object$ncomp, top = top)
        res$Ypred <- top_res$Ypred
        res$Ypred_index <- top_res$Ypred_index
        res$Ypred_top <- top_res$Ypred_top
        res$Ypred_top_score <- top_res$Ypred_top_score
      }
    }
    if (isTRUE(proj)) {
      res$Ttest <- lda_res$Ttest
    }
    if (!is.null(Ytest)) {
      res$Q2Y <- .fastpls_q2_from_class_labels(object, Ytest, res$Ypred)
	    }
		    return(res)
		  }
	  if (isTRUE(object$classification) &&
	      !is.null(object$classification_rule) &&
	      .is_candidate_knn_classifier(object$classification_rule)) {
	    cand_res <- .candidate_knn_predictions(
	      object,
	      Xtest,
	      top = top,
	      keep_ttest = isTRUE(proj)
	    )
	    cand_res$Q2Y <- NULL
	    if (!is.null(Ytest)) {
	      cand_res$Q2Y <- .fastpls_q2_from_class_labels(object, Ytest, cand_res$Ypred)
	    }
	    return(cand_res)
	  }
	  if (isTRUE(object$classification) &&
	      is.null(Ytest) &&
	      !isTRUE(raw_scores) &&
	      !isTRUE(object$gaussian_y) &&
	      !isTRUE(use_metal) &&
	      (is.null(object$classification_rule) ||
	         identical(object$classification_rule, "argmax"))) {
	    pred_backend <- if (identical(object$predict_backend, "cuda_flash") && isTRUE(has_cuda())) {
	      "cuda"
	    } else {
	      "cpp"
    }
    bias_res <- .class_bias_predict(
	      object,
	      Xtest,
	      class_bias = NULL,
	      top = top,
      proj = proj,
      backend = pred_backend
    )
    bias_res$Q2Y <- NULL
    return(bias_res)
  }
	  res <- if (isTRUE(use_metal)) {
    .pls_predict_metal(object, Xtest, proj)
  } else if (isTRUE(use_cuda_flash)) {
    tryCatch(
      pls_predict_flash_cuda(object, Xtest, proj),
      error = function(e) {
        if (identical(backend, "cuda_flash")) {
          stop(e)
        }
        pls_predict(object, Xtest, proj)
      }
    )
  } else if (isTRUE(use_cpu_flash)) {
    tryCatch(
      pls_predict_flash_cpu(object, Xtest, proj, as.integer(flash.block_size)),
      error = function(e) {
        if (identical(backend, "cpu_flash")) {
          stop(e)
        }
        pls_predict(object, Xtest, proj)
      }
    )
  } else {
    pls_predict(object, Xtest, proj)
  }
  if (isTRUE(object$gaussian_y)) {
    res$Ypred <- .decode_gaussian_y_cube(object, res$Ypred)
  }
  res$Q2Y=NULL

  if (!is.null(Ytest)) {
    for (i in 1:length(object$ncomp)) {
      if(object$classification){
        Ytest_transf=matrix(0,ncol=length(object$lev),nrow=length(Ytest))
        colnames(Ytest_transf)=object$lev
        for(w in object$lev){
          Ytest_transf[Ytest==w,w]=1
        }
      } else{
        Ytest_transf=as.matrix(Ytest)
      }
      ypred_i <- matrix(
        res$Ypred[, , i],
        nrow = dim(res$Ypred)[1L],
        ncol = dim(res$Ypred)[2L]
      )
      res$Q2Y[i] = RQ(Ytest_transf, ypred_i)
    }
  }

  if(object$classification){
    if (!is.null(object$classification_rule) &&
        .is_lda_classifier(object$classification_rule)) {
      lda_res <- .fastpls_lda_predictions(
        object,
        Xtest,
        Ttest = if (!is.null(res$Ttest)) res$Ttest else NULL,
        return_scores = isTRUE(raw_scores) || top > 1L
      )
      res$Ypred <- lda_res$Ypred
      if (!is.null(lda_res$lda_scores)) {
        if (isTRUE(raw_scores)) {
          res$LDA_scores <- lda_res$lda_scores
        }
        if (top > 1L) {
          top_res <- .class_topk_from_score_cube(lda_res$lda_scores, object$lev, object$ncomp, top = top)
          res$Ypred <- top_res$Ypred
          res$Ypred_index <- top_res$Ypred_index
          res$Ypred_top <- top_res$Ypred_top
          res$Ypred_top_score <- top_res$Ypred_top_score
        }
      }
      if (isTRUE(proj) || !is.null(res$Ttest)) {
        res$Ttest <- lda_res$Ttest
      }
    } else {
	      score_cube <- res$Ypred
	      top_res <- .class_topk_from_score_cube(score_cube, object$lev, object$ncomp, class_bias = NULL, top = top)
      if (isTRUE(raw_scores)) {
        res$Yscore <- score_cube
      }
      res$Ypred <- top_res$Ypred
      res$Ypred_index <- top_res$Ypred_index
      if (!is.null(top_res$Ypred_top)) {
        res$Ypred_top <- top_res$Ypred_top
        res$Ypred_top_score <- top_res$Ypred_top_score
      }
    }
  }
  res
}

.fastpls_preprocess_train <- function(X, scaling) {
  X <- as.matrix(X)
  scal <- if (is.character(scaling)) pmatch(scaling, c("centering", "autoscaling", "none"))[1] else as.integer(scaling)
  mX <- rep(0, ncol(X))
  if (scal < 3L) {
    mX <- colMeans(X)
    X <- sweep(X, 2, mX, "-")
  }
  vX <- rep(1, ncol(X))
  if (scal == 2L) {
    vX <- apply(X, 2, sd)
    vX[!is.finite(vX) | vX == 0] <- 1
    X <- sweep(X, 2, vX, "/")
  }
  list(X = X, mX = matrix(mX, nrow = 1), vX = matrix(vX, nrow = 1), scaling = scal)
}

.fastpls_preprocess_test <- function(X, mX, vX) {
  X <- as.matrix(X)
  X <- sweep(X, 2, as.numeric(mX[1, ]), "-")
  sweep(X, 2, as.numeric(vX[1, ]), "/")
}

.kernel_pls_kernel_id <- function(kernel) {
  kernel <- match.arg(kernel, c("linear", "rbf", "poly"))
  switch(kernel, linear = 1L, rbf = 2L, poly = 3L)
}

.kernel_pls_gamma <- function(gamma, Xtrain) {
  if (is.null(gamma)) {
    gamma <- 1 / max(1L, ncol(Xtrain))
  }
  gamma <- as.numeric(gamma)[1]
  if (!is.finite(gamma) || gamma <= 0) {
    stop("gamma must be a finite positive number", call. = FALSE)
  }
  gamma
}

.center_kernel_train_base <- function(K) {
  col_means <- colMeans(K)
  row_means <- rowMeans(K)
  grand_mean <- mean(col_means)
  Kc <- sweep(K, 2, col_means, "-")
  Kc <- sweep(Kc, 1, row_means, "-")
  Kc <- Kc + grand_mean
  list(K = Kc, col_means = matrix(col_means, nrow = 1), grand_mean = grand_mean)
}

.center_kernel_test_base <- function(Ktest, train_col_means, train_grand_mean) {
  Kc <- sweep(Ktest, 2, as.numeric(train_col_means[1, ]), "-")
  Kc <- sweep(Kc, 1, rowMeans(Ktest), "-")
  Kc + train_grand_mean
}

.supervised_response_matrix <- function(Y) {
  if (is.factor(Y)) {
    return(transformy(Y))
  }
  as.matrix(Y)
}

.kernel_pls_fit <- function(Xtrain,
                            Ytrain,
                            Xtest,
                            Ytest,
                            ncomp,
                            scaling,
                            kernel,
                            gamma,
                            degree,
                            coef0,
                            fit,
                            proj,
                            kernel_engine,
                            fit_fun,
                            inner_args) {
  kernel <- match.arg(kernel, c("linear", "rbf", "poly"))
  if (identical(kernel, "linear")) {
    inner <- do.call(
      fit_fun,
      c(
        list(
          Xtrain = Xtrain,
          Ytrain = Ytrain,
          Xtest = NULL,
          Ytest = NULL,
          ncomp = ncomp,
          scaling = scaling,
          fit = fit,
          proj = FALSE
        ),
        inner_args
      )
    )
    inner$kernel <- "linear"
    inner$kernel_engine <- paste0(kernel_engine, "_direct")
    inner$kernel_linear_direct <- TRUE
    class(inner) <- "fastPLS"
    if (!is.null(Xtest)) {
      res <- predict.fastPLS(inner, as.matrix(Xtest), Ytest = Ytest, proj = proj)
      inner <- c(inner, res)
      class(inner) <- "fastPLS"
    }
    return(inner)
  }
  prep <- .fastpls_preprocess_train(Xtrain, scaling)
  gamma <- .kernel_pls_gamma(gamma, prep$X)
  kernel_id <- .kernel_pls_kernel_id(kernel)
  K <- kernel_matrix_cpp(prep$X, prep$X, kernel_id, gamma, as.integer(degree), coef0)
  kc <- center_kernel_train_cpp(K)
  inner <- do.call(
    fit_fun,
    c(
      list(
        Xtrain = kc$K,
        Ytrain = Ytrain,
        Xtest = NULL,
        Ytest = NULL,
        ncomp = ncomp,
        scaling = "none",
        fit = fit,
        proj = FALSE
      ),
      inner_args
    )
  )
  out <- list(
    inner_model = inner,
    Xref = prep$X,
    mX = prep$mX,
    vX = prep$vX,
    kernel = kernel,
    kernel_id = kernel_id,
    gamma = gamma,
    degree = as.integer(degree),
    coef0 = coef0,
    kernel_center = kc,
    kernel_engine = kernel_engine,
    ncomp = inner$ncomp,
    xprod_mode = inner$xprod_mode,
    gpu_resident = isTRUE(inner$gpu_resident)
  )
  out <- .inherit_inner_variance_explained(out, inner)
  class(out) <- c("fastPLSKernel", "fastPLS")
  if (!is.null(Xtest)) {
    res <- predict(out, Xtest, Ytest = Ytest, proj = proj)
    out <- c(out, res)
    class(out) <- c("fastPLSKernel", "fastPLS")
  }
  out
}

#' Kernel PLS
#'
#' Fits PLS on a centered training kernel. The CUDA variant uses the GPU PLS core
#' after host-side kernel construction and centering.
#'
#' @inheritParams pls
#' @param kernel Kernel type: \code{linear}, \code{rbf}, or \code{poly}.
#' @param gamma Kernel scale. Defaults to `1 / ncol(Xtrain)`.
#' @param degree Polynomial kernel degree.
#' @param coef0 Polynomial kernel offset.
#' @param ... Additional arguments passed to the inner PLS fit.
#' @return A `fastPLSKernel` object.
#' @noRd
kernel_pls_cpp <- function(Xtrain,
                           Ytrain,
                           Xtest = NULL,
                           Ytest = NULL,
                           ncomp = 2,
                           scaling = c("centering", "autoscaling", "none"),
                           kernel = c("linear", "rbf", "poly"),
                           gamma = NULL,
                           degree = 3L,
                           coef0 = 1,
                           svd.method = c("irlba", "cpu_rsvd"),
                           rsvd_oversample = 10L,
                           rsvd_power = 1L,
                           svds_tol = 0,
                           irlba_work = 0L,
                           irlba_maxit = 1000L,
                           irlba_tol = 1e-5,
                           irlba_eps = 1e-9,
                           irlba_svtol = 1e-5,
                           seed = 1L,
                           gaussian_y = FALSE,
                           gaussian_y_dim = NULL,
		                  classifier = c("argmax", "lda", "cknn"),
	                  lda_ridge = 1e-8,
	                  fit = FALSE,
                           return_variance = TRUE,
                           proj = FALSE) {
  classifier <- .resolve_classifier_for_backend(classifier, "cpu")
  svd.method <- match.arg(.normalize_svd_method(svd.method), c("irlba", "cpu_rsvd"))
  if (isTRUE(gaussian_y) && is.null(gaussian_y_dim)) {
    gaussian_y_dim <- .gaussian_y_default_dim(Xtrain, NULL)
  }
  .kernel_pls_fit(
    Xtrain, Ytrain, Xtest, Ytest, ncomp, match.arg(scaling), match.arg(kernel),
    gamma, degree, coef0, fit, proj, "cpp", pls,
    list(
      method = "simpls",
      svd.method = svd.method,
      rsvd_oversample = rsvd_oversample,
      rsvd_power = rsvd_power,
      svds_tol = svds_tol,
      irlba_work = irlba_work,
      irlba_maxit = irlba_maxit,
      irlba_tol = irlba_tol,
      irlba_eps = irlba_eps,
      irlba_svtol = irlba_svtol,
      seed = seed,
      gaussian_y = gaussian_y,
      gaussian_y_dim = gaussian_y_dim,
      classifier = classifier,
      lda_ridge = lda_ridge,
      return_variance = return_variance
    )
  )
}

#' @noRd
kernel_pls_cuda <- function(Xtrain,
                            Ytrain,
                            Xtest = NULL,
                            Ytest = NULL,
                            ncomp = 2,
                            scaling = c("centering", "autoscaling", "none"),
                            kernel = c("linear", "rbf", "poly"),
                            gamma = NULL,
                            degree = 3L,
                            coef0 = 1,
                            rsvd_oversample = 10L,
                            rsvd_power = 1L,
                            svds_tol = 0,
                            seed = 1L,
                            gaussian_y = FALSE,
                            gaussian_y_dim = NULL,
		                   classifier = c("argmax", "lda", "cknn"),
	                  lda_ridge = 1e-8,
	                  fit = FALSE,
                            return_variance = TRUE,
	  proj = FALSE,
                            ...) {
  classifier <- .resolve_classifier_for_backend(classifier, "cuda")
  fit_fun <- simpls_gpu
  if (isTRUE(gaussian_y) && is.null(gaussian_y_dim)) {
    gaussian_y_dim <- .gaussian_y_default_dim(Xtrain, NULL)
  }
  .kernel_pls_fit(
    Xtrain, Ytrain, Xtest, Ytest, ncomp, match.arg(scaling), match.arg(kernel),
    gamma, degree, coef0, fit, proj, "cuda", fit_fun,
    c(
      list(
        rsvd_oversample = rsvd_oversample,
        rsvd_power = rsvd_power,
        svds_tol = svds_tol,
        seed = seed,
        gaussian_y = gaussian_y,
        gaussian_y_dim = gaussian_y_dim,
        classifier = classifier,
        lda_ridge = lda_ridge,
        return_variance = return_variance
      ),
      list(...)
    )
  )
}

#' @rdname predict.fastPLS
#' @export
predict.fastPLSKernel <- function(object, newdata, Ytest = NULL, proj = FALSE, ...) {
  if (!is(object, "fastPLSKernel")) {
    stop("object is not a fastPLSKernel object", call. = FALSE)
  }
  Xnew <- .fastpls_preprocess_test(newdata, object$mX, object$vX)
  if (identical(object$kernel_engine, "metal")) {
    Ktest <- .kernel_matrix_metal(Xnew, object$Xref, object$kernel, object$gamma, object$degree, object$coef0)
    Ktest <- .center_kernel_test_base(Ktest, object$kernel_center$col_means, object$kernel_center$grand_mean)
  } else {
    Ktest <- kernel_matrix_cpp(Xnew, object$Xref, object$kernel_id, object$gamma, object$degree, object$coef0)
    Ktest <- center_kernel_test_cpp(Ktest, object$kernel_center$col_means, object$kernel_center$grand_mean)
  }
  predict.fastPLS(object$inner_model, Ktest, Ytest = Ytest, proj = proj, ...)
}

.opls_fit <- function(Xtrain,
                      Ytrain,
                      Xtest,
                      Ytest,
                      ncomp,
                      scaling,
                      north,
                      fit,
                      proj,
                      filter_engine,
                      fit_fun,
  inner_args) {
  Yfilter <- .supervised_response_matrix(Ytrain)
  filt <- opls_filter_cpp(as.matrix(Xtrain), Yfilter, as.integer(north), pmatch(scaling, c("centering", "autoscaling", "none"))[1])
  inner <- do.call(
    fit_fun,
    c(
      list(
        Xtrain = filt$X,
        Ytrain = Ytrain,
        Xtest = NULL,
        Ytest = NULL,
        ncomp = ncomp,
        scaling = "none",
        fit = fit,
        proj = FALSE
      ),
      inner_args
    )
  )
  out <- list(
    inner_model = inner,
    mX = filt$mX,
    vX = filt$vX,
    W_orth = filt$W_orth,
    P_orth = filt$P_orth,
    north = filt$north,
    opls_engine = filter_engine,
    ncomp = inner$ncomp,
    xprod_mode = inner$xprod_mode,
    gpu_resident = isTRUE(inner$gpu_resident)
  )
  out <- .inherit_inner_variance_explained(out, inner)
  class(out) <- c("fastPLSOpls", "fastPLS")
  if (!is.null(Xtest)) {
    res <- predict(out, Xtest, Ytest = Ytest, proj = proj)
    out <- c(out, res)
    class(out) <- c("fastPLSOpls", "fastPLS")
  }
  out
}

#' Orthogonal PLS
#'
#' Removes supervised orthogonal variation from `Xtrain`, then fits the SIMPLS
#' core. The CUDA variant uses the GPU SIMPLS core after CPU-side OPLS filtering.
#'
#' @inheritParams pls
#' @param north Number of orthogonal components to remove before PLS fitting.
#' @param ... Additional arguments passed to the inner PLS fit.
#' @return A `fastPLSOpls` object.
#' @noRd
opls_cpp <- function(Xtrain,
                     Ytrain,
                     Xtest = NULL,
                     Ytest = NULL,
                     ncomp = 2,
                     north = 1L,
                     scaling = c("centering", "autoscaling", "none"),
                     svd.method = c("irlba", "cpu_rsvd"),
                     rsvd_oversample = 10L,
                     rsvd_power = 1L,
                     svds_tol = 0,
                     irlba_work = 0L,
                     irlba_maxit = 1000L,
                     irlba_tol = 1e-5,
                     irlba_eps = 1e-9,
                     irlba_svtol = 1e-5,
                     seed = 1L,
                     gaussian_y = FALSE,
                     gaussian_y_dim = NULL,
			                  classifier = c("argmax", "lda", "cknn"),
                     lda_ridge = 1e-8,
                     fit = FALSE,
                     return_variance = TRUE,
                     proj = FALSE) {
  classifier <- .resolve_classifier_for_backend(classifier, "cpu")
  svd.method <- match.arg(.normalize_svd_method(svd.method), c("irlba", "cpu_rsvd"))
  .opls_fit(
    Xtrain, Ytrain, Xtest, Ytest, ncomp, match.arg(scaling), north, fit, proj,
    "cpp", pls,
    list(
      method = "simpls",
      svd.method = svd.method,
      rsvd_oversample = rsvd_oversample,
      rsvd_power = rsvd_power,
      svds_tol = svds_tol,
      irlba_work = irlba_work,
      irlba_maxit = irlba_maxit,
      irlba_tol = irlba_tol,
      irlba_eps = irlba_eps,
      irlba_svtol = irlba_svtol,
      seed = seed,
      gaussian_y = gaussian_y,
      gaussian_y_dim = gaussian_y_dim,
      classifier = classifier,
      lda_ridge = lda_ridge,
      return_variance = return_variance
    )
  )
}

#' @noRd
opls_cuda <- function(Xtrain,
                      Ytrain,
                      Xtest = NULL,
                      Ytest = NULL,
                      ncomp = 2,
                      north = 1L,
                      scaling = c("centering", "autoscaling", "none"),
                      rsvd_oversample = 10L,
                      rsvd_power = 1L,
                      svds_tol = 0,
                      seed = 1L,
                      gaussian_y = FALSE,
                      gaussian_y_dim = NULL,
			                      classifier = c("argmax", "lda", "cknn"),
	                  lda_ridge = 1e-8,
	                  fit = FALSE,
                      return_variance = TRUE,
	  proj = FALSE,
                      ...) {
				  classifier <- .resolve_classifier_for_backend(classifier, "cuda")
  fit_fun <- simpls_gpu
  .opls_fit(
    Xtrain, Ytrain, Xtest, Ytest, ncomp, match.arg(scaling), north, fit, proj,
    "cpp", fit_fun,
    c(
      list(
        rsvd_oversample = rsvd_oversample,
        rsvd_power = rsvd_power,
        svds_tol = svds_tol,
        seed = seed,
        gaussian_y = gaussian_y,
        gaussian_y_dim = gaussian_y_dim,
        classifier = classifier,
        lda_ridge = lda_ridge,
        return_variance = return_variance
      ),
      list(...)
    )
  )
}

#' @rdname predict.fastPLS
#' @export
predict.fastPLSOpls <- function(object, newdata, Ytest = NULL, proj = FALSE, ...) {
  if (!is(object, "fastPLSOpls")) {
    stop("object is not a fastPLSOpls object", call. = FALSE)
  }
  Xnew <- if (identical(object$opls_engine, "metal")) {
    .opls_apply_filter_metal(newdata, object$mX, object$vX, object$W_orth, object$P_orth)
  } else {
    opls_apply_filter_cpp(as.matrix(newdata), object$mX, object$vX, object$W_orth, object$P_orth)
  }
  predict.fastPLS(object$inner_model, Xnew, Ytest = Ytest, proj = proj, ...)
}

#' GPU-native SIMPLS fit
#'
#' Uses a CUDA-oriented `simpls` engine that keeps the training
#' matrices and deflated cross-covariance resident on device throughout the fit.
#'
#' @param Xtrain Numeric training predictor matrix.
#' @param Ytrain Training response (numeric or factor).
#' @param Xtest Optional test predictor matrix.
#' @param Ytest Optional observed response used to compute `Q2Y`.
#' @param ncomp Number of components (scalar or vector).
#' @param scaling One of \code{centering}, \code{autoscaling}, or \code{none}.
#' @param rsvd_oversample RSVD oversampling.
#' @param rsvd_power RSVD power iterations.
#' @param svds_tol Tolerance placeholder passed through to the backend.
#' @param seed Random seed.
#' @param fit Return fitted values and `R2Y` when `TRUE`.
#' @param return_variance Compute predictor-space latent-variable variance
#'   explained. Set to `FALSE` for timing/memory benchmarks that do not need
#'   plotting variance metadata.
#' @param proj Return projected `Ttest` when `TRUE`.
#' @param gpu_device_state Keep selected SIMPLS workspaces resident on the GPU when `TRUE`.
#' @param gpu_qr Use GPU QR finalization when available.
#' @param gpu_eig Use GPU eigensolver finalization when available.
#' @param gpu_qless_qr Use the q-less GPU QR path when available.
#' @param gpu_finalize_threshold Component threshold controlling GPU-side finalization.
#' @param gaussian_y Logical; when `TRUE`, fit to a Gaussian random response
#'   sketch and decode predictions back to the original response scale or class
#'   labels. The default is `FALSE`.
#' @param gaussian_y_dim Number of Gaussian response dimensions. When `NULL`,
#'   the default is `min(ncol(Xtrain), 100)`.
#' @return A `fastPLS` object.
#' @noRd
simpls_gpu = function(Xtrain,
                      Ytrain,
                      Xtest = NULL,
                      Ytest = NULL,
                      ncomp = 2,
                      scaling = c("centering", "autoscaling", "none"),
                      rsvd_oversample = 10L,
                      rsvd_power = 1L,
                      svds_tol = 0,
                      seed = 1L,
                      fit = FALSE,
                      proj = FALSE,
                      gpu_device_state = TRUE,
                      gpu_qr = TRUE,
                      gpu_eig = TRUE,
                      gpu_qless_qr = FALSE,
                      gpu_finalize_threshold = 32L,
                      gaussian_y = FALSE,
                      gaussian_y_dim = NULL,
			                      classifier = c("argmax", "lda", "cknn"),
	                      lda_ridge = 1e-8,
                          return_variance = TRUE) {
  if (!has_cuda()) {
    stop("simpls_gpu requires a CUDA-enabled fastPLS build")
  }
	  on.exit(try(cuda_reset_workspace(), silent = TRUE), add = TRUE)
  classifier <- .resolve_classifier_for_backend(classifier, "cuda")

	  scal <- pmatch(scaling, c("centering", "autoscaling", "none"))[1]
	  Xtrain <- as.matrix(Xtrain)
	  if (is.factor(Ytrain) &&
	      !isTRUE(gaussian_y) &&
	      !isTRUE(fit) &&
	      classifier %in% c("argmax", "candidate_knn_cpp", "candidate_knn_cuda") &&
	      .should_use_label_aware_plssvd(nrow(Xtrain), nlevels(Ytrain))) {
	    model <- .plssvd_label_aware_stream_model(
	      Xtrain,
	      Ytrain,
	      ncomp = as.integer(ncomp),
	      scaling = scal,
	      backend = "cuda"
	    )
	    model <- .attach_lda_classifier(
	      model,
	      Xtrain,
	      Ytrain,
	      classifier,
	      lda_ridge
	    )
	    model <- .maybe_attach_pls_variance_explained(model, Xtrain, return_variance)
	    return(model)
	  }
	  Ytrain_original <- Ytrain
  yprep <- .prepare_gaussian_y(
    Ytrain,
    Xtrain,
    gaussian_y = gaussian_y,
    gaussian_y_dim = gaussian_y_dim,
    seed = seed,
    backend = "cuda"
  )
  Ytrain <- yprep$Ytrain
  classification <- yprep$classification
  lev <- yprep$lev

  tuned <- .resolve_simpls_fast_rsvd_tuning(
    n = nrow(Xtrain),
    p = ncol(Xtrain),
    q = ncol(Ytrain),
    svd.method = "cuda_rsvd"
  )
  if (missing(rsvd_oversample)) rsvd_oversample <- tuned$rsvd_oversample
  if (missing(rsvd_power)) rsvd_power <- tuned$rsvd_power

  use_xprod_default <- .should_use_xprod_default(ncol(Xtrain), ncol(Ytrain), ncomp)
  fused_model <- if (classification && identical(classifier, "lda_cuda")) {
    .try_cuda_native_lda_fit_predict(
      method_id = 3L,
      method_name = "simpls",
      Xtrain = Xtrain,
      Ytrain = Ytrain,
      Ytrain_original = Ytrain_original,
      Xtest = Xtest,
      Ytest = Ytest,
      ncomp = ncomp,
      scaling_id = scal,
      use_xprod_default = use_xprod_default,
      fit = fit,
      proj = proj,
      rsvd_oversample = rsvd_oversample,
      rsvd_power = rsvd_power,
      svds_tol = svds_tol,
      seed = seed,
      lda_ridge = lda_ridge,
      lev = lev,
      gpu_device_state = gpu_device_state,
      gpu_qr = gpu_qr,
      gpu_eig = gpu_eig,
      gpu_qless_qr = gpu_qless_qr,
      gpu_finalize_threshold = gpu_finalize_threshold
    )
  } else {
    NULL
  }
  if (!is.null(fused_model)) {
    fused_model <- .attach_gaussian_y(fused_model, yprep$gaussian)
    fused_model <- .decode_gaussian_y_outputs(fused_model, Ytrain_original)
    fused_model <- .maybe_attach_pls_variance_explained(fused_model, Xtrain, return_variance)
    return(fused_model)
  }
  fit_expr <- function() {
    pls.model2.fast.gpu(
      Xtrain = Xtrain,
      Ytrain = Ytrain,
      ncomp = as.integer(ncomp),
      fit = fit,
      scaling = scal,
      rsvd_oversample = rsvd_oversample,
      rsvd_power = rsvd_power,
      svds_tol = svds_tol,
      seed = seed
    )
  }
  model <- .with_gpu_native_options(
    if (use_xprod_default) .with_simpls_gpu_xprod(fit_expr()) else fit_expr(),
    gpu_device_state = gpu_device_state,
    gpu_qr = gpu_qr,
    gpu_eig = gpu_eig,
    gpu_qless_qr = gpu_qless_qr,
    gpu_finalize_threshold = gpu_finalize_threshold
  )
  cuda_reset_workspace()
  model$classification <- classification
  model$lev <- lev
  model$pls_method <- "simpls"
  model$predict_latent_ok <- TRUE
  model$xprod_default <- use_xprod_default
  if (isTRUE(fit)) model <- .attach_train_scores(model, Xtrain)
  model <- .enable_flash_prediction(model, "cuda")
	  model <- .attach_gaussian_y(model, yprep$gaussian)
	  model <- .decode_gaussian_y_outputs(model, Ytrain_original)
	  model <- .attach_lda_classifier(
	    model,
	    Xtrain,
	    Ytrain_original,
	    classifier,
	    lda_ridge
	  )
  model <- .maybe_attach_pls_variance_explained(model, Xtrain, return_variance)

  if (!is.null(Xtest)) {
    Xtest <- as.matrix(Xtest)
    res <- predict.fastPLS(model, Xtest, Ytest = Ytest, proj = proj)
    model <- c(model, res)
  }

  if (classification && fit && !is.null(model$Yfit)) {
    Yfitlab <- as.data.frame(matrix(nrow = nrow(Xtrain), ncol = length(ncomp)))
    colnames(Yfitlab) <- paste("ncomp=", ncomp, sep = "")
    for (i in seq_along(ncomp)) {
      tt <- apply(model$Yfit[, , i], 1, which.max)
      Yfitlab[, i] <- factor(lev[tt], levels = lev)
    }
    model$Yfit <- Yfitlab
  }

  class(model) <- "fastPLS"
  model
}

#' GPU-native PLSSVD fit
#'
#' Uses a dedicated CUDA PLSSVD engine that keeps the cross-covariance SVD and
#' latent linear algebra on device, returning the standard `fastPLS` object
#' structure for prediction and plotting.
#'
#' @param Xtrain Numeric training predictor matrix.
#' @param Ytrain Training response (numeric or factor).
#' @param Xtest Optional test predictor matrix.
#' @param Ytest Optional observed response used to compute `Q2Y`.
#' @param ncomp Number of components (scalar or vector).
#' @param scaling One of \code{centering}, \code{autoscaling}, or \code{none}.
#' @param rsvd_oversample RSVD oversampling.
#' @param rsvd_power RSVD power iterations.
#' @param svds_tol Tolerance placeholder passed through to the backend.
#' @param seed Random seed.
#' @param fit Return fitted values and `R2Y` when `TRUE`.
#' @param proj Return projected `Ttest` when `TRUE`.
#' @param gpu_qr Use GPU QR finalization when available.
#' @param gpu_eig Use GPU eigensolver finalization when available.
#' @param gpu_qless_qr Use the q-less GPU QR path when available.
#' @param gpu_finalize_threshold Component threshold controlling GPU-side finalization.
#' @param gaussian_y Logical; when `TRUE`, fit to a Gaussian random response
#'   sketch and decode predictions back to the original response scale or class
#'   labels. The default is `FALSE`.
#' @param gaussian_y_dim Number of Gaussian response dimensions. When `NULL`,
#'   the default is `min(ncol(Xtrain), 100)`.
#' @return A `fastPLS` object fitted with GPU PLSSVD.
#' @noRd
plssvd_gpu = function(Xtrain,
                      Ytrain,
                      Xtest = NULL,
                      Ytest = NULL,
                      ncomp = 2,
                      scaling = c("centering", "autoscaling", "none"),
                      rsvd_oversample = 10L,
                      rsvd_power = 1L,
                      svds_tol = 0,
                      seed = 1L,
                      fit = FALSE,
                      proj = FALSE,
                      gpu_qr = TRUE,
                      gpu_eig = TRUE,
                      gpu_qless_qr = FALSE,
                      gpu_finalize_threshold = 32L,
                      gaussian_y = FALSE,
                      gaussian_y_dim = NULL,
	                      classifier = c("argmax", "lda", "cknn"),
	                      lda_ridge = 1e-8,
                          return_variance = TRUE) {
  if (!has_cuda()) {
    stop("plssvd_gpu requires a CUDA-enabled fastPLS build")
  }
	  on.exit(try(cuda_reset_workspace(), silent = TRUE), add = TRUE)
				  classifier <- .resolve_classifier_for_backend(classifier, "cuda")

  scal <- pmatch(scaling, c("centering", "autoscaling", "none"))[1]
  Xtrain <- as.matrix(Xtrain)
  Ytrain_original <- Ytrain
  yprep <- .prepare_gaussian_y(
    Ytrain,
    Xtrain,
    gaussian_y = gaussian_y,
    gaussian_y_dim = gaussian_y_dim,
    seed = seed,
    backend = "cuda"
  )
  Ytrain <- yprep$Ytrain
  classification <- yprep$classification
  lev <- yprep$lev

  use_xprod_default <- .should_use_xprod_default(ncol(Xtrain), ncol(Ytrain), ncomp)
  fused_model <- if (classification && identical(classifier, "lda_cuda")) {
    .try_cuda_native_lda_fit_predict(
      method_id = 1L,
      method_name = "plssvd",
      Xtrain = Xtrain,
      Ytrain = Ytrain,
      Ytrain_original = Ytrain_original,
      Xtest = Xtest,
      Ytest = Ytest,
      ncomp = ncomp,
      scaling_id = scal,
      use_xprod_default = use_xprod_default,
      fit = fit,
      proj = proj,
      rsvd_oversample = rsvd_oversample,
      rsvd_power = rsvd_power,
      svds_tol = svds_tol,
      seed = seed,
      lda_ridge = lda_ridge,
      lev = lev,
      gpu_device_state = FALSE,
      gpu_qr = gpu_qr,
      gpu_eig = gpu_eig,
      gpu_qless_qr = gpu_qless_qr,
      gpu_finalize_threshold = gpu_finalize_threshold
    )
  } else {
    NULL
  }
  if (!is.null(fused_model)) {
    fused_model <- .attach_gaussian_y(fused_model, yprep$gaussian)
    fused_model <- .decode_gaussian_y_outputs(fused_model, Ytrain_original)
    fused_model <- .maybe_attach_pls_variance_explained(fused_model, Xtrain, return_variance)
    return(fused_model)
  }
  fit_fun <- if (use_xprod_default) pls.model1.gpu.implicit.xprod else pls.model1.gpu
  model <- .with_gpu_native_options(
    fit_fun(
      Xtrain = Xtrain,
      Ytrain = Ytrain,
      ncomp = as.integer(ncomp),
      fit = fit,
      scaling = scal,
      rsvd_oversample = rsvd_oversample,
      rsvd_power = rsvd_power,
      svds_tol = svds_tol,
      seed = seed
    ),
    gpu_device_state = FALSE,
    gpu_qr = gpu_qr,
    gpu_eig = gpu_eig,
    gpu_qless_qr = gpu_qless_qr,
    gpu_finalize_threshold = gpu_finalize_threshold
  )
  cuda_reset_workspace()
  model$classification <- classification
  model$lev <- lev
  model$pls_method <- "plssvd"
  model$predict_latent_ok <- TRUE
  model$xprod_default <- use_xprod_default
  if (isTRUE(fit)) model <- .attach_train_scores(model, Xtrain)
  model <- .enable_flash_prediction(model, "cuda")
	  model <- .attach_gaussian_y(model, yprep$gaussian)
	  model <- .decode_gaussian_y_outputs(model, Ytrain_original)
	  model <- .attach_lda_classifier(
	    model,
	    Xtrain,
	    Ytrain_original,
	    classifier,
	    lda_ridge
	  )
  model <- .maybe_attach_pls_variance_explained(model, Xtrain, return_variance)

  if (!is.null(Xtest)) {
    Xtest <- as.matrix(Xtest)
    res <- predict.fastPLS(model, Xtest, Ytest = Ytest, proj = proj)
    model <- c(model, res)
  }

  if (classification && fit && !is.null(model$Yfit)) {
    Yfitlab <- as.data.frame(matrix(nrow = nrow(Xtrain), ncol = length(ncomp)))
    colnames(Yfitlab) <- paste("ncomp=", ncomp, sep = "")
    for (i in seq_along(ncomp)) {
      tt <- apply(model$Yfit[, , i], 1, which.max)
      Yfitlab[, i] <- factor(lev[tt], levels = lev)
    }
    model$Yfit <- Yfitlab
  }

  class(model) <- "fastPLS"
  model
}

.predict_flash_attach <- function(model, Xtest, Ytest, proj) {
  model$predict_backend <- "cuda_flash"
  model$flash_svd <- TRUE
  if (!is.null(Xtest)) {
    res <- predict.fastPLS(
      model,
      as.matrix(Xtest),
      Ytest = Ytest,
      proj = proj,
      backend = "cuda_flash"
    )
    model <- c(model, res)
  }
  model
}

#' GPU PLSSVD with FlashSVD-style low-rank CUDA prediction
#'
#' Fits with the standard GPU PLSSVD backend and marks the model so prediction
#' uses a CUDA low-rank path that applies `X %*% R %*% W` without materializing
#' the full coefficient matrix `B`.
#' @noRd
plssvd_flash_gpu <- function(Xtrain, Ytrain, Xtest = NULL, Ytest = NULL,
                             ncomp = 2, scaling = c("centering", "autoscaling", "none"),
                             rsvd_oversample = 10L, rsvd_power = 1L,
                             svds_tol = 0, seed = 1L, fit = FALSE,
                             proj = FALSE, gpu_qr = TRUE, gpu_eig = TRUE,
                             gpu_qless_qr = FALSE, gpu_finalize_threshold = 32L) {
  model <- plssvd_gpu(
    Xtrain = Xtrain, Ytrain = Ytrain, Xtest = NULL, Ytest = NULL,
    ncomp = ncomp, scaling = scaling, rsvd_oversample = rsvd_oversample,
    rsvd_power = rsvd_power, svds_tol = svds_tol, seed = seed,
    fit = fit, proj = FALSE, gpu_qr = gpu_qr, gpu_eig = gpu_eig,
    gpu_qless_qr = gpu_qless_qr, gpu_finalize_threshold = gpu_finalize_threshold
  )
  .predict_flash_attach(model, Xtest, Ytest, proj)
}

#' GPU SIMPLS with FlashSVD-style low-rank CUDA prediction
#' @noRd
simpls_flash_gpu <- function(Xtrain, Ytrain, Xtest = NULL, Ytest = NULL,
                             ncomp = 2, scaling = c("centering", "autoscaling", "none"),
                             rsvd_oversample = 10L, rsvd_power = 1L,
                             svds_tol = 0, seed = 1L, fit = FALSE,
                             proj = FALSE, gpu_device_state = TRUE,
                             gpu_qr = TRUE, gpu_eig = TRUE,
                             gpu_qless_qr = FALSE, gpu_finalize_threshold = 32L) {
  model <- simpls_gpu(
    Xtrain = Xtrain, Ytrain = Ytrain, Xtest = NULL, Ytest = NULL,
    ncomp = ncomp, scaling = scaling, rsvd_oversample = rsvd_oversample,
    rsvd_power = rsvd_power, svds_tol = svds_tol, seed = seed,
    fit = fit, proj = FALSE, gpu_device_state = gpu_device_state,
    gpu_qr = gpu_qr, gpu_eig = gpu_eig, gpu_qless_qr = gpu_qless_qr,
    gpu_finalize_threshold = gpu_finalize_threshold
  )
  .predict_flash_attach(model, Xtest, Ytest, proj)
}

#' GPU OPLS with FlashSVD-style low-rank CUDA prediction
#' @noRd
opls_flash_gpu <- function(Xtrain, Ytrain, Xtest = NULL, Ytest = NULL,
                           ncomp = 2, north = 1L,
                           scaling = c("centering", "autoscaling", "none"),
                           rsvd_oversample = 10L, rsvd_power = 1L,
                           svds_tol = 0, seed = 1L, fit = FALSE,
                           proj = FALSE, ...) {
  model <- opls_cuda(
    Xtrain = Xtrain, Ytrain = Ytrain, Xtest = NULL, Ytest = NULL,
    ncomp = ncomp, north = north, scaling = scaling,
    rsvd_oversample = rsvd_oversample,
    rsvd_power = rsvd_power, svds_tol = svds_tol, seed = seed,
    fit = fit, proj = FALSE, ...
  )
  model$inner_model$predict_backend <- "cuda_flash"
  model$inner_model$flash_svd <- TRUE
  model$flash_svd <- TRUE
  if (!is.null(Xtest)) {
    res <- predict(model, as.matrix(Xtest), Ytest = Ytest, proj = proj)
    model <- c(model, res)
    class(model) <- c("fastPLSOpls", "fastPLS")
  }
  model
}

#' GPU kernel PLS with FlashSVD-style low-rank CUDA prediction
#' @noRd
kernel_pls_flash_gpu <- function(Xtrain, Ytrain, Xtest = NULL, Ytest = NULL,
                                 ncomp = 2,
                                 scaling = c("centering", "autoscaling", "none"),
                                 kernel = c("linear", "rbf", "poly"),
                                 gamma = NULL, degree = 3L, coef0 = 1,
                                 rsvd_oversample = 10L, rsvd_power = 1L,
                                 svds_tol = 0, seed = 1L,
                                 fit = FALSE, proj = FALSE, ...) {
  model <- kernel_pls_cuda(
    Xtrain = Xtrain, Ytrain = Ytrain, Xtest = NULL, Ytest = NULL,
    ncomp = ncomp, scaling = scaling, kernel = kernel, gamma = gamma,
    degree = degree, coef0 = coef0,
    rsvd_oversample = rsvd_oversample, rsvd_power = rsvd_power,
    svds_tol = svds_tol, seed = seed,
    fit = fit, proj = FALSE, ...
  )
  model$inner_model$predict_backend <- "cuda_flash"
  model$inner_model$flash_svd <- TRUE
  model$flash_svd <- TRUE
  if (!is.null(Xtest)) {
    res <- predict(model, as.matrix(Xtest), Ytest = Ytest, proj = proj)
    model <- c(model, res)
    class(model) <- c("fastPLSKernel", "fastPLS")
  }
  model
}

.cv_metric_from_matrix <- function(Ytrue, Ypred, Ytrain = NULL) {
  Ytrue <- as.matrix(Ytrue)
  Ypred <- as.matrix(Ypred)
  if (ncol(Ytrue) == 1L) {
    center <- if (!is.null(Ytrain)) mean(as.numeric(as.matrix(Ytrain)), na.rm = TRUE) else mean(Ytrue[, 1L], na.rm = TRUE)
    press <- sum((Ypred[, 1L] - Ytrue[, 1L])^2, na.rm = TRUE)
    tss <- sum((Ytrue[, 1L] - center)^2, na.rm = TRUE)
    return(list(metric_name = "q2", metric_value = if (is.finite(tss) && tss > 0) 1 - press / tss else NA_real_))
  }
  list(metric_name = "rmsd", metric_value = sqrt(mean((Ypred - Ytrue)^2, na.rm = TRUE)))
}

.decode_cv_predictions <- function(Ypred, Ydata, classification, lev) {
  if (classification && is.null(Ypred)) {
    stop("Classification CV output is missing both class predictions and score predictions", call. = FALSE)
  }
  dims <- dim(Ypred)
  if (length(dims) != 3L) stop("Internal CV prediction output must be a 3D array")
  out <- vector("list", dims[[3L]])
  metrics <- data.frame(
    ncomp_index = seq_len(dims[[3L]]),
    metric_name = character(dims[[3L]]),
    metric_value = numeric(dims[[3L]]),
    stringsAsFactors = FALSE
  )
  for (i in seq_len(dims[[3L]])) {
    mat <- Ypred[, , i, drop = TRUE]
    if (classification) {
      pred <- factor(lev[max.col(mat, ties.method = "first")], levels = lev)
      out[[i]] <- pred
      metrics$metric_name[[i]] <- "accuracy"
      metrics$metric_value[[i]] <- mean(as.character(pred) == as.character(Ydata), na.rm = TRUE)
    } else {
      out[[i]] <- as.matrix(mat)
      metric <- .cv_metric_from_matrix(Ydata, mat, Ytrain = Ydata)
      metrics$metric_name[[i]] <- metric$metric_name
      metrics$metric_value[[i]] <- metric$metric_value
    }
  }
  list(
    pred = if (length(out) == 1L) out[[1L]] else out,
    metrics = metrics
  )
}

.decode_cv_class_predictions <- function(class_pred, Ydata, lev) {
  pred_mat <- as.matrix(class_pred)
  out <- vector("list", ncol(pred_mat))
  metrics <- data.frame(
    ncomp_index = seq_len(ncol(pred_mat)),
    metric_name = rep("accuracy", ncol(pred_mat)),
    metric_value = numeric(ncol(pred_mat)),
    stringsAsFactors = FALSE
  )
  for (i in seq_len(ncol(pred_mat))) {
    idx <- as.integer(pred_mat[, i])
    ok <- is.finite(idx) & idx >= 1L & idx <= length(lev)
    pred <- rep(NA_character_, length(idx))
    pred[ok] <- lev[idx[ok]]
    pred <- factor(pred, levels = lev)
    out[[i]] <- pred
    metrics$metric_value[[i]] <- mean(as.character(pred) == as.character(Ydata), na.rm = TRUE)
  }
  list(pred = out, metrics = metrics)
}

.pls_cv_compiled <- function(Xdata,
                             Ydata,
                             constrain = NULL,
                             ncomp = 2L,
                             kfold = 10L,
                             scaling = c("centering", "autoscaling", "none"),
                             method = c("plssvd", "simpls", "opls", "kernelpls"),
                             backend = c("cpp", "cuda", "metal"),
                             svd.method = c("cpu_rsvd", "irlba"),
                             rsvd_oversample = 10L,
                             rsvd_power = 1L,
                             svds_tol = 0,
                             irlba_work = 0L,
                             irlba_maxit = 1000L,
                             irlba_tol = 1e-5,
                             irlba_eps = 1e-9,
                             irlba_svtol = 1e-5,
                             seed = 1L,
                             xprod = NULL,
                             north = 1L,
                             return_scores = FALSE,
                             kodama_class_codes = NULL,
                             classifier = c("argmax", "lda", "cknn"),
                             lda_ridge = 1e-8,
                             candidate_knn_k = 10L,
                             candidate_tau = 0.2,
                             candidate_alpha = 0.75,
                             candidate_top_m = 20L,
                             gpu_qr = TRUE,
                             gpu_eig = TRUE,
                             gpu_qless_qr = FALSE,
                             gpu_finalize_threshold = 32L) {
  method <- match.arg(method)
  backend <- match.arg(backend)
  classifier <- .normalize_classifier_public(classifier)
  classifier_id <- switch(classifier, argmax = 0L, lda = 1L, cknn = 2L)
  candidate_knn_k <- max(1L, as.integer(candidate_knn_k)[1L])
  candidate_top_m <- max(1L, as.integer(candidate_top_m)[1L])
  candidate_tau <- as.numeric(candidate_tau)[1L]
  candidate_alpha <- as.numeric(candidate_alpha)[1L]
  if (!is.finite(candidate_tau) || candidate_tau <= 0) {
    stop("candidate_tau must be a finite positive number", call. = FALSE)
  }
  if (!is.finite(candidate_alpha)) {
    stop("candidate_alpha must be finite", call. = FALSE)
  }
  scal <- pmatch(scaling, c("centering", "autoscaling", "none"))[1]
  Xdata <- as.matrix(Xdata)
  if (is.null(constrain)) constrain <- seq_len(nrow(Xdata))
  constrain <- as.integer(as.factor(constrain))
  ncomp <- as.integer(ncomp)

  if (is.factor(Ydata)) {
    classification <- TRUE
    lev <- levels(Ydata)
    Yoriginal <- Ydata
    Ymat <- matrix(as.integer(Ydata), ncol = 1L)
    q_response <- length(lev)
  } else {
    classification <- FALSE
    lev <- NULL
    Yoriginal <- as.matrix(Ydata)
    Ymat <- as.matrix(Ydata)
    q_response <- ncol(Ymat)
  }
  class_codes <- matrix(numeric(0), nrow = 0L, ncol = 0L)
  q_backend <- q_response
  if (!is.null(kodama_class_codes)) {
    if (!classification) {
      stop("KODAMA Gaussian class-code CV is only available for classification factors.", call. = FALSE)
    }
    class_codes <- as.matrix(kodama_class_codes)
    if (nrow(class_codes) != q_response || ncol(class_codes) < 1L) {
      stop("kodama_class_codes must have one row per class and at least one column.", call. = FALSE)
    }
    q_backend <- ncol(class_codes)
  }

  if (identical(method, "plssvd")) {
    cap <- .cap_plssvd_ncomp(ncomp, nrow(Xdata), ncol(Xdata), q_response, warn = TRUE)
    ncomp <- cap$ncomp
  }

  if (identical(backend, "cuda") && !has_cuda()) {
    stop("CUDA CV requires a CUDA-enabled fastPLS build.", call. = FALSE)
  }
  if (identical(backend, "metal") && !isTRUE(has_metal())) {
    stop("Metal CV requires a fastPLS build with Apple Metal support.", call. = FALSE)
  }

  if (identical(backend, "cpp")) {
    svd.method <- .normalize_svd_method(match.arg(svd.method))
    svdmeth <- .svd_method_id(svd.method)
  } else if (identical(backend, "cuda")) {
    svdmeth <- .svd_method_id("cuda_rsvd")
  } else {
    svdmeth <- .svd_method_id("metal_rsvd")
  }
  if (is.null(xprod)) {
    xprod <- if (identical(backend, "cuda")) {
      .should_use_xprod_default(ncol(Xdata), q_backend, ncomp)
    } else if (identical(backend, "metal")) {
      FALSE
    } else if (identical(svd.method, "irlba")) {
      .should_use_xprod_irlba_default(nrow(Xdata), ncol(Xdata), q_backend, ncomp)
    } else if (identical(svd.method, "cpu_rsvd")) {
      .should_use_xprod_default(ncol(Xdata), q_backend, ncomp)
    } else {
      FALSE
    }
  } else {
    xprod <- isTRUE(xprod)
  }

  meth <- .normalize_pls_method(method)
  backend_id <- if (identical(backend, "cuda")) 1L else if (identical(backend, "metal")) 2L else 0L

  run_cv <- function() {
    if (!is.null(seed)) set.seed(as.integer(seed))
    pls_cv_predict_compiled(
      Xdata = Xdata,
      Ydata = Ymat,
      constrain = constrain,
      ncomp = ncomp,
      scaling = scal,
      kfold = .compiled_cv_kfold_arg(kfold, constrain),
      method = meth,
      backend = backend_id,
      svd_method = svdmeth,
      rsvd_oversample = as.integer(rsvd_oversample),
      rsvd_power = as.integer(rsvd_power),
      svds_tol = svds_tol,
      seed = as.integer(seed),
      classification = classification,
      n_response = as.integer(q_response),
      xprod = isTRUE(xprod),
      opls_north = as.integer(north),
      return_scores = isTRUE(return_scores),
      class_codes = class_codes,
      classifier = classifier_id,
      lda_ridge = lda_ridge,
      candidate_knn_k = candidate_knn_k,
      candidate_tau = candidate_tau,
      candidate_alpha = candidate_alpha,
      candidate_top_m = candidate_top_m
    )
  }

  if (method %in% c("simpls", "opls", "kernelpls")) {
    run_cv_profiled <- function() {
      .with_fastpls_fast_options(run_cv())
    }
  } else {
    run_cv_profiled <- run_cv
  }

  if (identical(backend, "cuda")) {
    on.exit(try(cuda_reset_workspace(), silent = TRUE), add = TRUE)
    cuda_simpls_family <- method %in% c("simpls", "opls", "kernelpls")
    if (cuda_simpls_family && isTRUE(xprod)) {
      res <- .with_simpls_gpu_xprod(
        .with_gpu_native_options(
          run_cv_profiled(),
          gpu_device_state = TRUE,
          gpu_qr = gpu_qr,
          gpu_eig = gpu_eig,
          gpu_qless_qr = gpu_qless_qr,
          gpu_finalize_threshold = gpu_finalize_threshold
        )
      )
    } else {
      res <- .with_gpu_native_options(
        run_cv_profiled(),
        gpu_device_state = cuda_simpls_family,
        gpu_qr = gpu_qr,
        gpu_eig = gpu_eig,
        gpu_qless_qr = gpu_qless_qr,
        gpu_finalize_threshold = gpu_finalize_threshold
      )
    }
    cuda_reset_workspace()
  } else if (identical(backend, "cpp")) {
    res <- .with_irlba_options(
      run_cv_profiled(),
      irlba_work = irlba_work,
      irlba_maxit = irlba_maxit,
      irlba_tol = irlba_tol,
      irlba_eps = irlba_eps,
      irlba_svtol = irlba_svtol
    )
  } else {
    res <- run_cv_profiled()
  }

  decoded <- if (classification && !is.null(res$class_pred)) {
    .decode_cv_class_predictions(res$class_pred, Yoriginal, lev)
  } else {
    .decode_cv_predictions(res$Ypred, Yoriginal, classification, lev)
  }
  res$pred <- decoded$pred
  res$metrics <- decoded$metrics
  res$classification <- classification
  res$levels <- lev
  if (!is.null(res$backend) && identical(res$backend, "cpp")) {
    res$backend <- "cpu"
  }
  res
}

.is_loocv_kfold <- function(kfold) {
  if (is.character(kfold)) {
    key <- tolower(trimws(kfold[[1L]]))
    return(key %in% c("loocv", "loo", "leave-one-out", "leave_one_out", "leave one out"))
  }
  FALSE
}

.cv_kfold_int <- function(kfold, n_groups, context = "cross-validation") {
  if (.is_loocv_kfold(kfold)) {
    return(as.integer(n_groups))
  }
  if (length(kfold) != 1L || is.na(kfold)) {
    stop(context, ": kfold must be a single integer or 'loocv'.", call. = FALSE)
  }
  kfold_int <- suppressWarnings(as.integer(kfold))
  if (is.na(kfold_int) || !is.finite(kfold_int)) {
    stop(context, ": kfold must be a finite integer or 'loocv'.", call. = FALSE)
  }
  if (kfold_int >= n_groups) {
    return(as.integer(n_groups))
  }
  max(2L, kfold_int)
}

.cv_is_leave_one_group_out <- function(kfold, n_groups) {
  .is_loocv_kfold(kfold) || .cv_kfold_int(kfold, n_groups) >= n_groups
}

.compiled_cv_kfold_arg <- function(kfold, constrain) {
  n_groups <- length(unique(as.integer(as.factor(constrain))))
  if (.cv_is_leave_one_group_out(kfold, n_groups)) {
    return(-1L)
  }
  .cv_kfold_int(kfold, n_groups, context = "compiled cross-validation")
}

.make_single_cv_folds <- function(Ydata, constrain, kfold, seed) {
  n <- if (is.matrix(Ydata) || is.data.frame(Ydata)) nrow(Ydata) else length(Ydata)
  if (is.null(constrain)) constrain <- seq_len(n)
  constrain <- as.integer(as.factor(constrain))
  groups <- sort(unique(constrain))
  n_groups <- length(groups)
  if (n_groups < 1L) {
    stop("cross-validation requires at least one constraint group.", call. = FALSE)
  }
  group_fold <- integer(length(groups))
  names(group_fold) <- as.character(groups)
  if (.cv_is_leave_one_group_out(kfold, n_groups)) {
    group_fold[] <- seq_along(groups) - 1L
    return(as.integer(group_fold[as.character(constrain)]))
  }
  kfold <- .cv_kfold_int(kfold, n_groups)
  set.seed(as.integer(seed))
  if (is.factor(Ydata)) {
    first_group_class <- vapply(groups, function(g) as.character(Ydata[which(constrain == g)[1L]]), character(1))
    for (cls in unique(first_group_class)) {
      idx <- which(first_group_class == cls)
      idx <- sample(idx, length(idx))
      group_fold[idx] <- (seq_along(idx) - 1L) %% kfold
    }
  } else {
    idx <- sample(seq_along(groups), length(groups))
    group_fold[idx] <- (seq_along(idx) - 1L) %% kfold
  }
  as.integer(group_fold[as.character(constrain)])
}

.cv_class_predictions_from_fit <- function(fit, component_index, ntest) {
  pred <- fit$Ypred
  if (is.data.frame(pred) || is.list(pred)) {
    return(as.character(pred[[component_index]]))
  }
  if (is.matrix(pred)) {
    if (ncol(pred) >= component_index) {
      return(as.character(pred[, component_index]))
    }
    if (ncol(pred) == 1L) {
      return(as.character(pred[, 1L]))
    }
  }
  if (length(pred) == ntest) {
    return(as.character(pred))
  }
  stop("Could not extract classification predictions from fold fit.", call. = FALSE)
}

.cv_regression_predictions_from_fit <- function(fit, component_index, ntest, q_response) {
  pred <- fit$Ypred
  dims <- dim(pred)
  if (length(dims) == 3L) {
    return(matrix(pred[, , component_index, drop = TRUE], nrow = ntest, ncol = q_response))
  }
  if (is.data.frame(pred)) {
    pred <- as.matrix(pred)
  }
  if (is.matrix(pred)) {
    if (q_response == 1L && ncol(pred) >= component_index) {
      return(matrix(pred[, component_index], nrow = ntest, ncol = 1L))
    }
    if (ncol(pred) == q_response) {
      return(matrix(pred, nrow = ntest, ncol = q_response))
    }
  }
  if (length(pred) == ntest * q_response) {
    return(matrix(pred, nrow = ntest, ncol = q_response))
  }
  stop("Could not extract regression predictions from fold fit.", call. = FALSE)
}

.pls_cv_via_pls <- function(Xdata,
                            Ydata,
                            constrain = NULL,
                            ncomp = 2L,
                            kfold = 10L,
                            scaling = c("centering", "autoscaling", "none"),
                            method = c("plssvd", "simpls", "opls", "kernelpls"),
                            backend = c("cpu", "cuda", "metal"),
                            svd.method = c("irlba", "cpu_rsvd"),
                            seed = 1L,
                            xprod = NULL,
                            north = 1L,
                            kernel = c("linear", "rbf", "poly"),
                            gamma = NULL,
                            degree = 3L,
                            coef0 = 1,
                            gaussian_y = FALSE,
                            gaussian_y_dim = NULL,
                            classifier = c("argmax", "lda", "cknn"),
                            lda_ridge = 1e-8,
                            candidate_knn_k = 10L,
                            candidate_tau = 0.2,
                            candidate_alpha = 0.75,
                            candidate_top_m = 20L,
                            return_scores = FALSE,
                            ...) {
  method <- match.arg(method)
  backend <- match.arg(backend)
  scaling <- match.arg(scaling)
  classifier <- .resolve_classifier_for_backend(classifier, backend)
  candidate_knn_k <- max(1L, as.integer(candidate_knn_k)[1L])
  candidate_tau <- as.numeric(candidate_tau)[1L]
  candidate_alpha <- as.numeric(candidate_alpha)[1L]
  candidate_top_m <- max(1L, as.integer(candidate_top_m)[1L])
  if (!is.finite(candidate_tau) || candidate_tau <= 0) {
    stop("candidate_tau must be a finite positive number", call. = FALSE)
  }
  if (!is.finite(candidate_alpha)) {
    stop("candidate_alpha must be finite", call. = FALSE)
  }
  dots <- .svd_control_from_dots(list(...))
  svd_ctl <- .resolve_svd_control(
    svd.method = if (missing(svd.method)) NULL else svd.method,
    dots = c(dots$dots, list(seed = seed)),
    context = ".pls_cv_via_pls()"
  )
  svd.method <- match.arg(.normalize_svd_method(svd_ctl$svd.method), c("irlba", "cpu_rsvd"))
  rsvd_oversample <- svd_ctl$rsvd_oversample
  rsvd_power <- svd_ctl$rsvd_power
  svds_tol <- svd_ctl$svds_tol
  irlba_work <- svd_ctl$irlba_work
  irlba_maxit <- svd_ctl$irlba_maxit
  irlba_tol <- svd_ctl$irlba_tol
  irlba_eps <- svd_ctl$irlba_eps
  irlba_svtol <- svd_ctl$irlba_svtol
  seed <- svd_ctl$seed
  kernel <- match.arg(kernel)
  Xdata <- as.matrix(Xdata)
  if (is.null(constrain)) constrain <- seq_len(nrow(Xdata))
  constrain <- as.integer(as.factor(constrain))
  ncomp <- as.integer(ncomp)
  if (identical(backend, "metal") && !isTRUE(has_metal())) {
    stop("Metal CV requires a fastPLS build with Apple Metal support.", call. = FALSE)
  }
  if (!is.null(xprod)) {
    warning("Explicit xprod is ignored in classifier CV; pls() applies its backend defaults inside each fold.", call. = FALSE)
  }

  classification <- is.factor(Ydata)
  Yoriginal <- Ydata
  if (classification) {
    lev <- levels(Ydata)
    q_response <- length(lev)
  } else {
    lev <- NULL
    Yoriginal <- as.matrix(Ydata)
    q_response <- ncol(Yoriginal)
  }
  if (identical(method, "plssvd")) {
    cap <- .cap_plssvd_ncomp(ncomp, nrow(Xdata), ncol(Xdata), q_response, warn = TRUE)
    ncomp <- cap$ncomp
  }

  fold <- .make_single_cv_folds(
    Ydata = if (classification) Ydata else Yoriginal[, 1L],
    constrain = constrain,
    kfold = kfold,
    seed = as.integer(seed)
  )
  nslice <- length(ncomp)
  class_pred <- if (classification) {
    matrix(NA_integer_, nrow = nrow(Xdata), ncol = nslice)
  } else {
    NULL
  }
  score_pred <- if (classification && isTRUE(return_scores)) {
    array(NA_real_, dim = c(nrow(Xdata), q_response, nslice))
  } else if (!classification) {
    array(NA_real_, dim = c(nrow(Xdata), q_response, nslice))
  } else {
    NULL
  }

  for (f in sort(unique(fold))) {
    test_idx <- which(fold == f)
    train_idx <- which(fold != f)
    if (!length(test_idx) || !length(train_idx)) {
      next
    }
    Ytrain <- if (classification) Ydata[train_idx] else Yoriginal[train_idx, , drop = FALSE]
    Ytest <- if (classification) Ydata[test_idx] else Yoriginal[test_idx, , drop = FALSE]
    if (classification && length(unique(Ytrain)) < 2L) {
      fallback <- names(which.max(table(Ytrain)))
      class_pred[test_idx, ] <- match(fallback, lev)
      next
    }

    fit <- pls(
      Xtrain = Xdata[train_idx, , drop = FALSE],
      Ytrain = Ytrain,
      Xtest = Xdata[test_idx, , drop = FALSE],
      Ytest = Ytest,
      ncomp = ncomp,
      scaling = scaling,
      method = method,
      svd.method = svd.method,
      rsvd_oversample = rsvd_oversample,
      rsvd_power = rsvd_power,
      svds_tol = svds_tol,
      seed = as.integer(seed) + as.integer(f),
      irlba_work = irlba_work,
      irlba_maxit = irlba_maxit,
      irlba_tol = irlba_tol,
      irlba_eps = irlba_eps,
      irlba_svtol = irlba_svtol,
      gaussian_y = gaussian_y,
      gaussian_y_dim = gaussian_y_dim,
      fit = FALSE,
      proj = FALSE,
      return_variance = FALSE,
      backend = backend,
      north = north,
      kernel = kernel,
      gamma = gamma,
      degree = degree,
      coef0 = coef0,
      classifier = classifier,
      lda_ridge = lda_ridge,
      candidate_knn_k = candidate_knn_k,
      candidate_tau = candidate_tau,
      candidate_alpha = candidate_alpha,
      candidate_top_m = candidate_top_m
    )

    if (classification) {
      for (j in seq_len(nslice)) {
        pred_chr <- .cv_class_predictions_from_fit(fit, j, length(test_idx))
        class_pred[test_idx, j] <- match(pred_chr, lev)
      }
    } else {
      for (j in seq_len(nslice)) {
        score_pred[test_idx, , j] <- .cv_regression_predictions_from_fit(
          fit,
          component_index = j,
          ntest = length(test_idx),
          q_response = q_response
        )
      }
    }
  }

  res <- list(
    Ypred = score_pred,
    class_pred = class_pred,
    fold = fold,
    ncomp = ncomp,
    method = method,
    backend = backend,
    classification = classification,
    levels = lev,
    status = "ok"
  )
  decoded <- if (classification) {
    .decode_cv_class_predictions(class_pred, Ydata, lev)
  } else {
    .decode_cv_predictions(score_pred, Yoriginal, FALSE, NULL)
  }
  res$pred <- decoded$pred
  res$metrics <- decoded$metrics
  res
}

#' Single grouped cross-validation for PLS
#'
#' Runs one fixed-component grouped cross-validation using the high-level
#' fastPLS algorithm router. This is the user-facing replacement for the older
#' backend-specific CV helpers.
#'
#' @inheritParams pls
#' @param Xdata Predictor matrix.
#' @param Ydata Response vector/matrix or factor.
#' @param constrain Optional grouping vector for grouped cross-validation. It
#'   must have one value per sample. Samples with the same value are assigned to
#'   the same fold, so they are kept together either in the training set or in
#'   the test set. This is useful when several rows come from the same patient,
#'   subject, batch, or technical replicate and must not be split across
#'   training and test folds. When `NULL`, each sample is treated as its own
#'   group.
#' @param kfold Number of folds, or `"loocv"` for leave-one-out
#'   cross-validation. When `constrain` is supplied, LOOCV means
#'   leave-one-constraint-group-out: all samples sharing the same constraint
#'   value are held out together.
#' @param classifier Classification rule for factor responses: `"argmax"` uses
#'   the largest predicted response score, `"lda"` fits an LDA head on the PLS
#'   scores inside each training fold, and `"cknn"` uses the candidate-kNN head.
#' @param lda_ridge Ridge added to the pooled LDA covariance diagonal.
#' @param candidate_knn_k,candidate_tau,candidate_alpha,candidate_top_m
#'   Candidate-kNN controls. `candidate_knn_k` is the number of neighbours,
#'   `candidate_tau` is the softmax temperature, `candidate_alpha` weights the
#'   neighbour evidence against the PLS class score, and `candidate_top_m`
#'   limits reranking to the strongest candidate classes.
#' @param return_scores Store score predictions for classification when `TRUE`.
#' @return A list with fold assignments, predictions, metrics, status, and
#'   backend metadata.
#' @examples
#' idx <- c(1:12, 51:62, 101:112)
#' X <- as.matrix(iris[idx, 1:4])
#' y <- factor(iris[idx, 5])
#' cv <- pls.single.cv(X, y, ncomp = 2, kfold = 3, method = "simpls",
#'                     backend = "cpu", svd.method = "rsvd", seed = 1)
#' cv$metrics
#' @export
pls.single.cv <- function(Xdata,
                          Ydata,
                          constrain = NULL,
                          ncomp = 2L,
                          kfold = 10L,
                          scaling = c("centering", "autoscaling", "none"),
                          method = c("simpls", "plssvd", "opls", "kernelpls"),
                          backend = c("cpu", "cuda", "metal"),
                          svd.method = c("irlba", "rsvd"),
                          seed = 1L,
                          north = 1L,
                          kernel = c("linear", "rbf", "poly"),
                          gamma = NULL,
                          degree = 3L,
                          coef0 = 1,
                          gaussian_y = FALSE,
                          gaussian_y_dim = NULL,
                          classifier = c("argmax", "lda", "cknn"),
                          lda_ridge = 1e-8,
                          candidate_knn_k = 10L,
                          candidate_tau = 0.2,
                          candidate_alpha = 0.75,
                          candidate_top_m = 20L,
                          return_scores = FALSE,
                          xprod = NULL,
                          ...) {
  method <- match.arg(method)
  backend <- .normalize_public_backend(backend)
  backend_compiled <- .compiled_backend(backend)
  scaling <- match.arg(scaling)
  dots <- .svd_control_from_dots(list(...))
  svd_ctl <- .resolve_svd_control(
    svd.method = if (missing(svd.method)) NULL else svd.method,
    dots = c(dots$dots, list(seed = seed)),
    context = "pls.single.cv()"
  )
  svd.method <- match.arg(.normalize_svd_method(svd_ctl$svd.method), c("irlba", "cpu_rsvd"))
  rsvd_oversample <- svd_ctl$rsvd_oversample
  rsvd_power <- svd_ctl$rsvd_power
  svds_tol <- svd_ctl$svds_tol
  irlba_work <- svd_ctl$irlba_work
  irlba_maxit <- svd_ctl$irlba_maxit
  irlba_tol <- svd_ctl$irlba_tol
  irlba_eps <- svd_ctl$irlba_eps
  irlba_svtol <- svd_ctl$irlba_svtol
  seed <- svd_ctl$seed
  kernel <- match.arg(kernel)
  classifier <- .normalize_classifier_public(classifier)
  if (!identical(kernel, "linear") && !identical(backend, "metal")) {
    stop("Nonlinear kernel CV is not available in the compiled CV helper; use kernel='linear'.", call. = FALSE)
  }
  if ((identical(backend, "metal") && !identical(classifier, "argmax")) ||
      (!identical(classifier, "argmax") && isTRUE(gaussian_y)) ||
      (identical(backend, "metal") && (!identical(kernel, "linear") || isTRUE(gaussian_y)))) {
    return(.pls_cv_via_pls(
      Xdata = Xdata,
      Ydata = Ydata,
      constrain = constrain,
      ncomp = ncomp,
      kfold = kfold,
      scaling = scaling,
      method = method,
      backend = backend,
      svd.method = svd.method,
      rsvd_oversample = rsvd_oversample,
      rsvd_power = rsvd_power,
      svds_tol = svds_tol,
      irlba_work = irlba_work,
      irlba_maxit = irlba_maxit,
      irlba_tol = irlba_tol,
      irlba_eps = irlba_eps,
      irlba_svtol = irlba_svtol,
      seed = seed,
      xprod = xprod,
      north = north,
      kernel = kernel,
      gamma = gamma,
      degree = degree,
      coef0 = coef0,
      gaussian_y = gaussian_y,
      gaussian_y_dim = gaussian_y_dim,
      classifier = classifier,
      lda_ridge = lda_ridge,
      candidate_knn_k = candidate_knn_k,
      candidate_tau = candidate_tau,
      candidate_alpha = candidate_alpha,
      candidate_top_m = candidate_top_m,
      return_scores = return_scores
    ))
  }
  .pls_cv_compiled(
    Xdata = Xdata,
    Ydata = Ydata,
    constrain = constrain,
    ncomp = ncomp,
    kfold = kfold,
    scaling = scaling,
    method = method,
    backend = backend_compiled,
    svd.method = if (missing(svd.method)) NULL else svd.method,
    rsvd_oversample = rsvd_oversample,
    rsvd_power = rsvd_power,
    svds_tol = svds_tol,
    irlba_work = irlba_work,
    irlba_maxit = irlba_maxit,
    irlba_tol = irlba_tol,
    irlba_eps = irlba_eps,
    irlba_svtol = irlba_svtol,
    seed = seed,
    xprod = xprod,
    north = north,
    return_scores = return_scores,
    classifier = classifier,
    lda_ridge = lda_ridge,
    candidate_knn_k = candidate_knn_k,
    candidate_tau = candidate_tau,
    candidate_alpha = candidate_alpha,
    candidate_top_m = candidate_top_m
  )
}

#' Fast grouped PLS cross-validation for compiled backends
#'
#' These fixed-component helpers perform grouped k-fold cross-validation with
#' compiled fastPLS models only. They accept classification factors or numeric
#' regression responses and return fold predictions plus accuracy, Q2, or RMSD.
#'
#' @param Xdata Numeric predictor matrix.
#' @param Ydata Factor response for classification, or numeric vector/matrix for regression.
#' @param constrain Optional grouping vector; samples with the same value stay in the same fold.
#' @param ncomp Number of PLS components.
#' @param kfold Number of CV folds.
#' @param scaling Scaling mode.
#' @param svd.method CPU SVD backend for Cpp functions.
#' @param xprod Use the matrix-free xprod backend where available. The default
#'   `NULL` applies the same size thresholds used by [pls()]; `TRUE` forces the
#'   route and `FALSE` disables it.
#' @param ... Additional backend tuning arguments.
#' @return A list with `Ypred`, decoded `pred`, `metrics`, `fold`, and status.
#' @noRd
plssvd_cv_cpp <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                          scaling = c("centering", "autoscaling", "none"),
                          svd.method = c("cpu_rsvd", "irlba"), xprod = NULL, ...) {
  .pls_cv_compiled(Xdata, Ydata, constrain, ncomp, kfold, scaling, "plssvd", "cpp", svd.method, xprod = xprod, ...)
}

#' @noRd
simpls_cv_cpp <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                          scaling = c("centering", "autoscaling", "none"),
                          svd.method = c("cpu_rsvd", "irlba"), xprod = NULL, ...) {
  .pls_cv_compiled(Xdata, Ydata, constrain, ncomp, kfold, scaling, "simpls", "cpp", svd.method, xprod = xprod, ...)
}

simpls_fast_cv_cpp <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                               scaling = c("centering", "autoscaling", "none"),
                               svd.method = c("cpu_rsvd", "irlba"), xprod = NULL, ...) {
  simpls_cv_cpp(Xdata, Ydata, constrain, ncomp, kfold, scaling, svd.method, xprod = xprod, ...)
}

#' @noRd
opls_cv_cpp <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                        north = 1L,
                        scaling = c("centering", "autoscaling", "none"),
                        svd.method = c("cpu_rsvd", "irlba"), xprod = NULL, ...) {
  pred_ncomp <- pmax(1L, as.integer(ncomp) - as.integer(north))
  .pls_cv_compiled(Xdata, Ydata, constrain, pred_ncomp, kfold, scaling, "opls", "cpp", svd.method, xprod = xprod, north = north, ...)
}

#' @noRd
kernelpls_cv_cpp <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                             scaling = c("centering", "autoscaling", "none"),
                             svd.method = c("cpu_rsvd", "irlba"), xprod = NULL, ...) {
  .pls_cv_compiled(Xdata, Ydata, constrain, ncomp, kfold, scaling, "kernelpls", "cpp", svd.method, xprod = xprod, ...)
}

kernel_pls_cv_cpp <- kernelpls_cv_cpp

#' @noRd
plssvd_cv_cuda <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                           scaling = c("centering", "autoscaling", "none"),
                           xprod = NULL, ...) {
  .pls_cv_compiled(Xdata, Ydata, constrain, ncomp, kfold, scaling, "plssvd", "cuda", xprod = xprod, ...)
}

#' @noRd
simpls_cv_cuda <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                           scaling = c("centering", "autoscaling", "none"),
                           xprod = NULL, ...) {
  .pls_cv_compiled(Xdata, Ydata, constrain, ncomp, kfold, scaling, "simpls", "cuda", xprod = xprod, ...)
}

simpls_fast_cv_cuda <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                                scaling = c("centering", "autoscaling", "none"),
                                xprod = NULL, ...) {
  simpls_cv_cuda(Xdata, Ydata, constrain, ncomp, kfold, scaling, xprod = xprod, ...)
}

#' @noRd
opls_cv_cuda <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                         north = 1L,
                         scaling = c("centering", "autoscaling", "none"),
                         xprod = NULL, ...) {
  pred_ncomp <- pmax(1L, as.integer(ncomp) - as.integer(north))
  .pls_cv_compiled(Xdata, Ydata, constrain, pred_ncomp, kfold, scaling, "opls", "cuda", xprod = xprod, north = north, ...)
}

#' @noRd
kernelpls_cv_cuda <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                              scaling = c("centering", "autoscaling", "none"),
                              xprod = NULL, ...) {
  .pls_cv_compiled(Xdata, Ydata, constrain, ncomp, kfold, scaling, "kernelpls", "cuda", xprod = xprod, ...)
}

kernel_pls_cv_cuda <- kernelpls_cv_cuda

#' @noRd
plssvd_cv_metal <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                            scaling = c("centering", "autoscaling", "none"),
                            xprod = NULL, ...) {
  .pls_cv_compiled(Xdata, Ydata, constrain, ncomp, kfold, scaling, "plssvd", "metal", xprod = xprod, ...)
}

#' @noRd
simpls_cv_metal <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                            scaling = c("centering", "autoscaling", "none"),
                            xprod = NULL, ...) {
  .pls_cv_compiled(Xdata, Ydata, constrain, ncomp, kfold, scaling, "simpls", "metal", xprod = xprod, ...)
}

simpls_fast_cv_metal <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                                 scaling = c("centering", "autoscaling", "none"),
                                 xprod = NULL, ...) {
  simpls_cv_metal(Xdata, Ydata, constrain, ncomp, kfold, scaling, xprod = xprod, ...)
}

#' @noRd
opls_cv_metal <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                          north = 1L,
                          scaling = c("centering", "autoscaling", "none"),
                          xprod = NULL, ...) {
  pred_ncomp <- pmax(1L, as.integer(ncomp) - as.integer(north))
  .pls_cv_compiled(Xdata, Ydata, constrain, pred_ncomp, kfold, scaling, "opls", "metal", xprod = xprod, north = north, ...)
}

#' @noRd
kernelpls_cv_metal <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                               scaling = c("centering", "autoscaling", "none"),
                               xprod = NULL, ...) {
  .pls_cv_compiled(Xdata, Ydata, constrain, ncomp, kfold, scaling, "kernelpls", "metal", xprod = xprod, ...)
}

kernel_pls_cv_metal <- kernelpls_cv_metal

.svd_methods_internal <- c("exact", "irlba", "cpu_rsvd", "cuda_rsvd", "metal_rsvd")
.svd_methods_public <- c("irlba", "rsvd")
.svd_methods_cpu <- c("irlba", "cpu_rsvd")

.svd_method_id <- function(method) {
  method <- .normalize_svd_method(method)
  method <- match.arg(method, .svd_methods_internal)
  switch(
    method,
    exact = 3L,
    irlba = 1L,
    cpu_rsvd = 4L,
    cuda_rsvd = 5L,
    metal_rsvd = 6L
  )
}

#' List available SVD backends
#'
#' Reports backend labels accepted by high-level APIs and whether each backend
#' is currently available.
#'
#' @return Data frame with columns `backend`, `method`, `svd.method`, and
#'   `enabled`.
#' @noRd
svd_methods <- function() {
  combos <- data.frame(
    backend = c("cpu", "cpu", "cuda", "metal"),
    method = c("irlba", "rsvd", "rsvd", "rsvd"),
    svd.method = c("irlba", "cpu_rsvd", "cuda_rsvd", "metal_rsvd"),
    enabled = c(TRUE, TRUE, isTRUE(has_cuda()), isTRUE(has_metal())),
    stringsAsFactors = FALSE
  )
  combos
}

.resolve_fastsvd_backend_method <- function(backend = c("cpu", "cuda", "metal"),
                                            method = c("irlba", "rsvd")) {
  backend <- match.arg(backend)
  method <- .normalize_svd_method(method)
  method <- match.arg(method, c("irlba", "rsvd"))
  if (identical(method, "irlba") && !identical(backend, "cpu")) {
    stop("fastsvd(method='irlba') is only available with backend='cpu'. Use method='rsvd' with backend='cuda' or backend='metal'.", call. = FALSE)
  }
  svd_method <- if (identical(method, "irlba")) {
    "irlba"
  } else {
    switch(
      backend,
      cpu = "cpu_rsvd",
      cuda = "cuda_rsvd",
      metal = "metal_rsvd"
    )
  }
  list(backend = backend, method = method, svd.method = svd_method)
}

.fastsvd_args_from_svd_method <- function(svd.method) {
  svd.method <- match.arg(.normalize_svd_method(svd.method), c("irlba", "cpu_rsvd", "cuda_rsvd", "metal_rsvd"))
  switch(
    svd.method,
    irlba = list(backend = "cpu", method = "irlba"),
    cpu_rsvd = list(backend = "cpu", method = "rsvd"),
    cuda_rsvd = list(backend = "cuda", method = "rsvd"),
    metal_rsvd = list(backend = "metal", method = "rsvd")
  )
}

.truncated_rsvd_metal <- function(A,
                                  k,
                                  rsvd_oversample = 10L,
                                  rsvd_power = 1L,
                                  seed = 1L,
                                  left_only = FALSE) {
  if (!isTRUE(has_metal())) {
    stop("method='metal_rsvd' requires a macOS build with Apple Metal support.", call. = FALSE)
  }

  A <- as.matrix(A)
  max_rank <- min(nrow(A), ncol(A))
  target <- min(max_rank, max(1L, as.integer(k)[1L]))
  sketch_rank <- min(max_rank, target + max(0L, as.integer(rsvd_oversample)[1L]))

  if (max_rank <= .metal_exact_max_rank() || sketch_rank >= max_rank) {
    exact <- svd(A, nu = target, nv = if (isTRUE(left_only)) 0L else target)
    return(list(
      U = exact$u[, seq_len(target), drop = FALSE],
      s = exact$d[seq_len(target)],
      Vt = if (isTRUE(left_only)) NULL else t(exact$v[, seq_len(target), drop = FALSE])
    ))
  }

  set.seed(as.integer(seed)[1L])
  omega <- matrix(rnorm(ncol(A) * sketch_rank), nrow = ncol(A), ncol = sketch_rank)
  Y <- metal_matrix_multiply_cpp(A, omega)

  power_iters <- max(0L, as.integer(rsvd_power)[1L])
  if (power_iters == 1L) {
    Y <- metal_matrix_multiply_cpp(A, metal_crossprod_cpp(A, Y))
  } else if (power_iters > 1L) {
    for (i in seq_len(power_iters)) {
      Z <- metal_crossprod_cpp(A, Y)
      Qz <- qr.Q(qr(Z))
      Y <- metal_matrix_multiply_cpp(A, Qz)
    }
  }

  Q <- qr.Q(qr(Y))
  B <- metal_crossprod_cpp(Q, A)
  small <- svd(B, nu = target, nv = if (isTRUE(left_only)) 0L else target)

  usable <- min(target, length(small$d), ncol(small$u))
  U <- Q %*% small$u[, seq_len(usable), drop = FALSE]
  Vt <- if (isTRUE(left_only)) {
    NULL
  } else {
    t(small$v[, seq_len(usable), drop = FALSE])
  }

  list(
    U = U,
    s = small$d[seq_len(usable)],
    Vt = Vt
  )
}

.svd_dispatch <- function(A,
                          k,
                          method = c("cpu_rsvd", "irlba", "cuda_rsvd", "metal_rsvd"),
                          rsvd_oversample = 10L,
                          rsvd_power = 1L,
                          svds_tol = 0,
                          seed = 1L,
                          left_only = FALSE) {
  method <- .normalize_svd_method(method)
  method <- match.arg(method)
  if (identical(method, "cuda_rsvd") && !has_cuda()) {
    stop("method='cuda_rsvd' requires a CUDA-enabled fastPLS build.", call. = FALSE)
  }
  if (identical(method, "metal_rsvd") && !has_metal()) {
    stop("method='metal_rsvd' requires a macOS build with Apple Metal support.", call. = FALSE)
  }
  if (identical(method, "metal_rsvd")) {
    A <- as.matrix(A)
    t_elapsed <- system.time({
      out <- .truncated_rsvd_metal(
        A = A,
        k = as.integer(k),
        rsvd_oversample = as.integer(rsvd_oversample),
        rsvd_power = as.integer(rsvd_power),
        seed = as.integer(seed),
        left_only = isTRUE(left_only)
      )
    })["elapsed"]
    return(list(
      U = out$U,
      s = as.vector(out$s),
      Vt = out$Vt,
      method = method,
      elapsed = as.numeric(t_elapsed)
    ))
  }
  svdmeth <- .svd_method_id(method)
  if (is.na(svdmeth)) {
    stop("Unknown method")
  }
  A <- as.matrix(A)
  t_elapsed <- system.time({
    out <- truncated_svd_debug(
      A = A,
      k = as.integer(k),
      svd_method = as.integer(svdmeth),
      rsvd_oversample = as.integer(rsvd_oversample),
      rsvd_power = as.integer(rsvd_power),
      svds_tol = as.numeric(svds_tol),
      seed = as.integer(seed),
      left_only = isTRUE(left_only)
    )
  })["elapsed"]
  out_norm <- list(
    U = out$u,
    s = as.vector(out$d),
    Vt = out$vt,
    method = method,
    elapsed = as.numeric(t_elapsed)
  )
  out_norm
}

#' Singular value decomposition through fastPLS backends
#'
#' Public interface to the truncated singular value decomposition engines used
#' internally by fastPLS. The function returns a base-SVD-compatible object from
#' the same CPU and accelerator implementations used by the PCA and PLS
#' routines, making it useful for direct low-rank decompositions, backend
#' checks, and numerical validation without masking `base::svd()`.
#'
#' @param x Numeric matrix to decompose, with observations or rows in rows and
#'   variables or columns in columns. Sparse matrices should be converted by the
#'   caller; `fastsvd()` currently works on a dense numeric matrix.
#' @param nu Number of left singular vectors to return. If `NULL`, the function
#'   uses the largest feasible rank implied by the matrix dimensions. When
#'   `ncomp` is supplied, `ncomp` controls the decomposition rank and `nu`
#'   controls only how many left vectors are kept in the returned object.
#' @param nv Number of right singular vectors to return. If `NULL`, the
#'   function uses the largest feasible rank implied by the matrix dimensions.
#'   When `ncomp` is supplied, `ncomp` controls the decomposition rank and `nv`
#'   controls only how many right vectors are kept in the returned object.
#' @param ncomp Optional truncated rank. When supplied, it overrides the rank
#'   implied by `nu` and `nv`; the final rank is always capped at
#'   `min(nrow(x), ncol(x))`.
#' @param backend Compute backend. \code{cpu} runs on the host CPU. \code{cuda}
#'   dispatches randomized SVD to the CUDA-native backend and requires
#'   `has_cuda()` to be `TRUE`. \code{metal} dispatches randomized SVD to the Apple
#'   Metal backend and requires `has_metal()` to be `TRUE`.
#' @param method SVD algorithm family. \code{irlba} uses the bundled iterative
#'   IRLBA-style CPU backend and is valid only with \code{backend = cpu}. \code{rsvd}
#'   uses randomized SVD on the selected backend.
#' @param oversample Non-negative oversampling dimension used by
#'   randomized SVD. The sketch dimension is approximately
#'   `ncomp + oversample`, capped by the matrix rank. Larger values can improve
#'   approximation accuracy at the cost of extra time and memory.
#' @param power Number of randomized-SVD power iterations. Larger values improve
#'   accuracy when singular values decay slowly, but each iteration adds
#'   additional matrix multiplications.
#' @param svds_tol Tolerance forwarded to iterative SVD backends. A value of
#'   `0` keeps the backend default.
#' @param work IRLBA working subspace size. A value of `0` lets the bundled
#'   IRLBA backend choose its default workspace.
#' @param maxit Maximum number of IRLBA iterations before the CPU IRLBA backend
#'   stops.
#' @param tol IRLBA residual convergence tolerance. Smaller values can
#'   improve numerical convergence but may increase runtime.
#' @param eps IRLBA orthogonality threshold used internally by the bundled
#'   implementation.
#' @param svtol IRLBA singular-value convergence tolerance.
#' @param seed Random seed used by randomized backends to generate the Gaussian
#'   sketch. It affects \code{rsvd} results and is ignored by deterministic
#'   backends.
#' @return A list compatible with `base::svd()` containing `d`, `u`, and `v`,
#'   plus backend metadata.
#' @examples
#' set.seed(1)
#' x <- matrix(rnorm(12 * 5), 12, 5)
#' s <- fastsvd(x, ncomp = 2, backend = "cpu", method = "rsvd", seed = 1)
#' s$d
#' s_irlba <- fastsvd(x, ncomp = 2, backend = "cpu", method = "irlba")
#' s_irlba$svd.method
#' @export
fastsvd <- function(x,
                    nu = NULL,
                    nv = NULL,
                    ncomp = NULL,
                    backend = c("cpu", "cuda", "metal"),
                    method = c("irlba", "rsvd"),
                    oversample = 10L,
                    power = 1L,
                    svds_tol = 0,
                    work = 0L,
                    maxit = 1000L,
                    tol = 1e-5,
                    eps = 1e-9,
                    svtol = 1e-5,
                    seed = 1L) {
  x <- as.matrix(x)
  n <- nrow(x)
  p <- ncol(x)
  if (is.null(nu)) nu <- min(n, p)
  if (is.null(nv)) nv <- min(n, p)
  resolved <- .resolve_fastsvd_backend_method(backend, method)
  backend <- resolved$backend
  method <- resolved$method
  svd.method <- resolved$svd.method
  if (identical(svd.method, "cuda_rsvd") && !has_cuda()) {
    stop("method='cuda_rsvd' requires a CUDA-enabled fastPLS build.", call. = FALSE)
  }
  if (identical(svd.method, "metal_rsvd") && !has_metal()) {
    stop("method='metal_rsvd' requires a macOS build with Apple Metal support.", call. = FALSE)
  }
  if (is.null(ncomp)) {
    k <- max(as.integer(nu), as.integer(nv), 1L)
  } else {
    k <- as.integer(ncomp)[1L]
  }
  k <- max(1L, min(k, n, p))

  out <- .with_irlba_options(
    .svd_dispatch(
      A = x,
      k = k,
      method = svd.method,
      rsvd_oversample = oversample,
      rsvd_power = power,
      svds_tol = svds_tol,
      seed = seed,
      left_only = FALSE
    ),
    irlba_work = work,
    irlba_maxit = maxit,
    irlba_tol = tol,
    irlba_eps = eps,
    irlba_svtol = svtol
  )
  u <- out$U
  v <- if (is.null(out$Vt) || length(out$Vt) == 0L) NULL else t(out$Vt)
  if (!is.null(u) && ncol(u) > nu) u <- u[, seq_len(nu), drop = FALSE]
  if (!is.null(v) && ncol(v) > nv) v <- v[, seq_len(nv), drop = FALSE]
  list(
    d = out$s,
    u = u,
    v = v,
    method = method,
    backend = backend,
    svd.method = svd.method,
    elapsed = out$elapsed,
    ncomp = k
  )
}

#' Principal component analysis through fastPLS SVD backends
#'
#' Computes PCA from the selected SVD backend and returns scores/loadings in a
#' compact object with a base-graphics plot method.
#'
#' @param x Numeric matrix with samples in rows and variables in columns.
#' @param ncomp Number of principal components.
#' @param center Logical; center columns before SVD.
#' @param scale Logical; scale columns before SVD.
#' @param backend Compute backend. \code{cpu} runs on the host CPU. \code{cuda} and
#'   \code{metal} use the corresponding native randomized-SVD backend when
#'   available.
#' @param method SVD algorithm family. \code{irlba} is available only with
#'   \code{backend = cpu}. \code{rsvd} uses randomized SVD on the selected backend.
#' @param ... Additional arguments passed to [fastsvd()].
#' @return A `fastPLSPCA` object with scores, loadings, and per-component
#'   `variance_explained` plus cumulative variance explained.
#' @examples
#' pc <- pca(as.matrix(iris[, 1:4]), ncomp = 2, backend = "cpu",
#'           method = "rsvd", seed = 1)
#' head(pc$scores)
#' pc$variance_explained
#' @export
pca <- function(x,
                ncomp = 2L,
                center = TRUE,
                scale = FALSE,
                backend = c("cpu", "cuda", "metal"),
                method = c("irlba", "rsvd"),
                ...) {
  x <- as.matrix(x)
  ncomp <- max(1L, min(as.integer(ncomp)[1L], nrow(x), ncol(x)))
  scaled <- base::scale(x, center = center, scale = scale)
  x_center <- attr(scaled, "scaled:center")
  x_scale <- attr(scaled, "scaled:scale")
  if (is.null(x_center)) x_center <- rep(0, ncol(x))
  if (is.null(x_scale)) x_scale <- rep(1, ncol(x))
  x_scaled <- as.matrix(scaled)

  resolved <- .resolve_fastsvd_backend_method(backend, method)
  decomp <- do.call(
    fastsvd,
    c(
      list(
        x = x_scaled,
        nu = ncomp,
        nv = ncomp,
        ncomp = ncomp,
        backend = resolved$backend,
        method = resolved$method
      ),
      list(...)
    )
  )
  scores <- x_scaled %*% decomp$v[, seq_len(ncomp), drop = FALSE]
  colnames(scores) <- paste0("PC", seq_len(ncomp))
  loadings <- decomp$v[, seq_len(ncomp), drop = FALSE]
  rownames(loadings) <- colnames(x)
  colnames(loadings) <- colnames(scores)
  sdev <- decomp$d[seq_len(ncomp)] / sqrt(max(1, nrow(x_scaled) - 1L))
  variance <- .fastpls_named_components(sdev^2, "PC")
  total_variance <- sum(x_scaled^2 / max(1, nrow(x_scaled) - 1L))
  variance_explained <- if (is.finite(total_variance) && total_variance > 0) {
    variance / total_variance
  } else {
    rep(NA_real_, length(variance))
  }
  variance_explained <- .fastpls_named_components(as.numeric(variance_explained), "PC")
  out <- list(
    scores = scores,
    loadings = loadings,
    sdev = sdev,
    variance = variance,
    variance_explained = variance_explained,
    cumulative_variance_explained = .fastpls_named_components(cumsum(variance_explained), "PC"),
    variance_total = total_variance,
    variance_basis = "X",
    center = x_center,
    scale = x_scale,
    svd = decomp,
    svd.method = decomp$svd.method %||% decomp$method,
    ncomp = ncomp
  )
  class(out) <- "fastPLSPCA"
  out
}

.fastpls_ellipse <- function(scores, conf = 0.95, type = c("confidence", "hotelling"), npoints = 100L) {
  type <- match.arg(type)
  scores <- as.matrix(scores)
  scores <- scores[stats::complete.cases(scores), , drop = FALSE]
  if (nrow(scores) < 3L || ncol(scores) < 2L) return(NULL)
  center <- colMeans(scores)
  cov2 <- stats::cov(scores)
  if (any(!is.finite(cov2)) || qr(cov2)$rank < 2L) return(NULL)
  radius <- if (identical(type, "hotelling")) {
    sqrt(2 * (nrow(scores) - 1) / (nrow(scores) - 2) * stats::qf(conf, 2, nrow(scores) - 2))
  } else {
    sqrt(stats::qchisq(conf, df = 2))
  }
  theta <- seq(0, 2 * pi, length.out = npoints)
  circle <- cbind(cos(theta), sin(theta))
  eig <- eigen(cov2, symmetric = TRUE)
  transform <- eig$vectors %*% diag(sqrt(pmax(eig$values, 0)), 2)
  sweep(radius * circle %*% t(transform), 2L, center, "+")
}

.fastpls_plot_palette <- function(n) {
  n <- as.integer(n)
  base <- c(
    "#0073C2FF", "#EFC000FF", "#CD534CFF", "#009E73FF",
    "#868686FF", "#56B4E9FF", "#D55E00FF", "#CC79A7FF",
    "#003C67FF", "#8F7700FF", "#A73030FF", "#005F45FF"
  )
  if (n <= length(base)) {
    return(base[seq_len(n)])
  }
  grDevices::hcl.colors(n, "Dark 3")
}

.fastpls_plot_call <- function(x, y, args) {
  do.call(graphics::plot, c(list(x = x, y = y), args))
}

.fastpls_plot_args <- function(xlab, ylab, main, dots) {
  if (is.null(dots$xlab)) dots$xlab <- xlab
  if (is.null(dots$ylab)) dots$ylab <- ylab
  if (is.null(dots$main)) dots$main <- main
  dots
}

.fastpls_plot_scores <- function(scores,
                                 comps = c(1L, 2L),
                                 groups = NULL,
                                 ellipse = FALSE,
                                 ellipse.type = c("confidence", "hotelling"),
                                 conf = 0.95,
                                 main = NULL,
                                 xlab = NULL,
                                 ylab = NULL,
                                 ...) {
  scores <- as.matrix(scores)
  comps <- as.integer(comps)
  if (length(comps) != 2L || any(comps < 1L) || max(comps) > ncol(scores)) {
    stop("comps must contain two valid component indices.", call. = FALSE)
  }
  xy <- scores[, comps, drop = FALSE]
  if (is.null(xlab)) xlab <- colnames(scores)[comps[1L]]
  if (is.null(ylab)) ylab <- colnames(scores)[comps[2L]]
  if (is.null(xlab) || is.na(xlab)) xlab <- paste0("Component ", comps[1L])
  if (is.null(ylab) || is.na(ylab)) ylab <- paste0("Component ", comps[2L])
  dots <- list(...)
  if (is.null(groups)) {
    if (is.null(dots$pch)) dots$pch <- 21
    if (is.null(dots$col)) dots$col <- "black"
    if (is.null(dots$bg)) dots$bg <- "#0073C2FF"
    .fastpls_plot_call(
      xy[, 1L],
      xy[, 2L],
      .fastpls_plot_args(xlab, ylab, main, dots)
    )
    if (isTRUE(ellipse)) {
      el <- .fastpls_ellipse(xy, conf = conf, type = ellipse.type)
      if (!is.null(el)) graphics::lines(el[, 1L], el[, 2L], col = "firebrick", lwd = 2)
    }
    return(invisible(xy))
  }
  groups <- as.factor(groups)
  pal <- .fastpls_plot_palette(nlevels(groups))
  bg <- pal[as.integer(groups)]
  if (is.null(dots$pch)) dots$pch <- 21
  if (is.null(dots$col)) dots$col <- "black"
  if (is.null(dots$bg)) dots$bg <- bg
  .fastpls_plot_call(
    xy[, 1L],
    xy[, 2L],
    .fastpls_plot_args(xlab, ylab, main, dots)
  )
  graphics::legend("topright", legend = levels(groups), pt.bg = pal, col = "black", pch = 21, bty = "n")
  if (isTRUE(ellipse)) {
    for (lev in levels(groups)) {
      idx <- which(groups == lev)
      el <- .fastpls_ellipse(xy[idx, , drop = FALSE], conf = conf, type = ellipse.type)
      if (!is.null(el)) graphics::lines(el[, 1L], el[, 2L], col = pal[match(lev, levels(groups))], lwd = 2)
    }
  }
  invisible(xy)
}

#' Plot PCA or PLS scores
#'
#' Draws a two-component score plot for `fastPLSPCA` and `fastPLS` objects.
#' Optional ellipses are computed either as a data confidence ellipse or a
#' Hotelling T2 score ellipse.
#' By default, grouped points use filled symbols with the group color in `bg`
#' and a black contour in `col`.
#' Axis labels include the predictor-space variance explained by each plotted
#' PCA component or PLS latent variable when available.
#'
#' @param x A `fastPLSPCA` object.
#' @param comps Two component indices.
#' @param groups Optional grouping vector for color and grouped ellipses.
#' @param score.set For PLS objects, plot \code{train} scores, \code{test} scores,
#'   or \code{auto} to use training scores when available.
#' @param ellipse Logical; draw confidence ellipses when `TRUE`.
#' @param ellipse.type \code{confidence} or \code{hotelling}.
#' @param conf Confidence level.
#' @param ... Additional arguments passed to `plot()`.
#' @return Invisibly returns the plotted score matrix.
#' @examples
#' pc <- pca(as.matrix(iris[, 1:4]), ncomp = 2, backend = "cpu",
#'           method = "rsvd", seed = 1)
#' plot(pc, groups = iris$Species, ellipse = TRUE)
#' @export
plot.fastPLSPCA <- function(x,
                            comps = c(1L, 2L),
                            groups = NULL,
                            ellipse = FALSE,
                            ellipse.type = c("confidence", "hotelling"),
                            conf = 0.95,
                            ...) {
  dots <- list(...)
  main <- if (is.null(dots$main)) "fastPLS PCA scores" else dots$main
  xlab <- if (is.null(dots$xlab)) {
    sprintf("PC%d (%.1f%%)", comps[1L], 100 * x$variance_explained[comps[1L]])
  } else {
    dots$xlab
  }
  ylab <- if (is.null(dots$ylab)) {
    sprintf("PC%d (%.1f%%)", comps[2L], 100 * x$variance_explained[comps[2L]])
  } else {
    dots$ylab
  }
  dots$main <- NULL
  dots$xlab <- NULL
  dots$ylab <- NULL
  do.call(
    .fastpls_plot_scores,
    c(
      list(
        scores = x$scores,
        comps = comps,
        groups = groups,
        ellipse = ellipse,
        ellipse.type = match.arg(ellipse.type),
        conf = conf,
        main = main,
        xlab = xlab,
        ylab = ylab
      ),
      dots
    )
  )
}

.fastpls_score_matrix <- function(x, slot) {
  scores <- x[[slot]]
  if (!is.null(scores) && length(scores) > 0L && all(dim(scores) > 0L)) {
    scores <- as.matrix(scores)
    colnames(scores) <- paste0("LV", seq_len(ncol(scores)))
    return(scores)
  }
  NULL
}

.fastpls_model_variance_explained <- function(x) {
  vx <- x$variance_explained
  if (!is.null(vx) && length(vx) > 0L) {
    return(as.numeric(vx))
  }
  vx <- x$x_variance_explained
  if (!is.null(vx) && length(vx) > 0L) {
    return(as.numeric(vx))
  }
  if (!is.null(x$inner_model)) {
    return(.fastpls_model_variance_explained(x$inner_model))
  }
  NULL
}

.fastpls_model_scores <- function(x, score.set = c("auto", "train", "test")) {
  score.set <- match.arg(score.set)
  if (identical(score.set, "train")) {
    scores <- .fastpls_score_matrix(x, "Ttrain")
    if (!is.null(scores)) return(scores)
    if (!is.null(x$inner_model)) return(.fastpls_model_scores(x$inner_model, score.set = "train"))
    return(NULL)
  }
  if (identical(score.set, "test")) {
    scores <- .fastpls_score_matrix(x, "Ttest")
    if (!is.null(scores)) return(scores)
    if (!is.null(x$inner_model)) return(.fastpls_model_scores(x$inner_model, score.set = "test"))
    return(NULL)
  }
  if (!is.null(x$Ttrain) && length(x$Ttrain) > 0L && all(dim(x$Ttrain) > 0L)) {
    scores <- as.matrix(x$Ttrain)
    colnames(scores) <- paste0("LV", seq_len(ncol(scores)))
    return(scores)
  }
  if (!is.null(x$inner_model)) {
    scores <- .fastpls_model_scores(x$inner_model, score.set = "auto")
    if (!is.null(scores)) return(scores)
  }
  if (!is.null(x$Ttest) && length(x$Ttest) > 0L && all(dim(x$Ttest) > 0L)) {
    scores <- as.matrix(x$Ttest)
    colnames(scores) <- paste0("LV", seq_len(ncol(scores)))
    return(scores)
  }
  NULL
}

#' @rdname plot.fastPLSPCA
#' @export
plot.fastPLS <- function(x,
                         comps = c(1L, 2L),
                         groups = NULL,
                         score.set = c("auto", "train", "test"),
                         ellipse = FALSE,
                         ellipse.type = c("confidence", "hotelling"),
                         conf = 0.95,
                         ...) {
  score.set <- match.arg(score.set)
  scores <- .fastpls_model_scores(x, score.set = score.set)
  if (is.null(scores)) {
    stop("The requested PLS scores are not stored. Refit with fit=TRUE for training scores or proj=TRUE for test scores.", call. = FALSE)
  }
  dots <- list(...)
  main <- if (is.null(dots$main)) "fastPLS scores" else dots$main
  var_exp <- .fastpls_model_variance_explained(x)
  xlab <- dots$xlab
  ylab <- dots$ylab
  if (is.null(xlab) && !is.null(var_exp) && length(var_exp) >= comps[1L] && is.finite(var_exp[comps[1L]])) {
    xlab <- sprintf("LV%d (%.1f%%)", comps[1L], 100 * var_exp[comps[1L]])
  }
  if (is.null(ylab) && !is.null(var_exp) && length(var_exp) >= comps[2L] && is.finite(var_exp[comps[2L]])) {
    ylab <- sprintf("LV%d (%.1f%%)", comps[2L], 100 * var_exp[comps[2L]])
  }
  dots$main <- NULL
  dots$xlab <- NULL
  dots$ylab <- NULL
  do.call(
    .fastpls_plot_scores,
    c(
      list(
        scores = scores,
        comps = comps,
        groups = groups,
        ellipse = ellipse,
        ellipse.type = match.arg(ellipse.type),
        conf = conf,
        main = main,
        xlab = xlab,
        ylab = ylab
      ),
      dots
    )
  )
}

.metal_mm <- function(A, B) {
  if (!isTRUE(has_metal())) {
    stop("backend='metal' requires Apple Metal support.", call. = FALSE)
  }
  A <- as.matrix(A)
  B <- as.matrix(B)
  metal_matrix_multiply_cpp(A, B)
}

.metal_crossprod <- function(A, B) {
  if (!isTRUE(has_metal())) {
    stop("backend='metal' requires Apple Metal support.", call. = FALSE)
  }
  A <- as.matrix(A)
  B <- as.matrix(B)
  metal_crossprod_cpp(A, B)
}

.metal_outer <- function(a, b) {
  tcrossprod(as.numeric(a), as.numeric(b))
}

.metal_min_flops <- function() {
  val <- suppressWarnings(as.numeric(Sys.getenv("FASTPLS_METAL_MIN_FLOPS", "200000000")))
  if (!is.finite(val) || val < 0) 2e8 else val
}

.metal_exact_max_rank <- function() {
  val <- suppressWarnings(as.integer(Sys.getenv("FASTPLS_METAL_EXACT_MAX_RANK", "256")))
  if (!is.finite(val) || is.na(val) || val < 0L) 256L else val
}

.metal_should_use_mm <- function(m, k, n) {
  m <- as.numeric(m); k <- as.numeric(k); n <- as.numeric(n)
  if (!is.finite(m) || !is.finite(k) || !is.finite(n)) return(FALSE)
  if (m <= 0 || k <= 0 || n <= 0) return(FALSE)
  # Matrix-vector and very thin products spend more time copying/dispatching
  # than computing unless the matrix is very large. BLAS is safer there.
  if (min(m, n) <= 1 && (m * k * n) < (.metal_min_flops() * 4)) return(FALSE)
  (2 * m * k * n) >= .metal_min_flops()
}

.metal_experimental_iterative_enabled <- function() {
  tolower(Sys.getenv("FASTPLS_METAL_EXPERIMENTAL_ITERATIVE", "false")) %in%
    c("1", "true", "yes", "y")
}

.metal_resident_simpls_enabled <- function() {
  !tolower(Sys.getenv("FASTPLS_METAL_RESIDENT_SIMPLS", "true")) %in%
    c("0", "false", "no", "n")
}

.pls_model1_metal <- function(Xtrain,
                              Ytrain,
                              ncomp,
                              scaling,
                              fit,
                              rsvd_oversample,
                              rsvd_power,
                              seed) {
  n <- nrow(Xtrain); p <- ncol(Xtrain); m <- ncol(Ytrain)
  ncomp <- as.integer(ncomp)
  max_ncomp <- max(ncomp)
  max_ncomp_eff <- min(max_ncomp, n, p, m)
  if (max_ncomp_eff < 1L) stop("plssvd effective rank is < 1")
  length_ncomp <- length(ncomp)

  mX <- matrix(0, nrow = 1, ncol = p)
  if (scaling < 3L) {
    mX <- matrix(colMeans(Xtrain), nrow = 1)
    Xtrain <- sweep(Xtrain, 2, mX[1, ], "-")
  }
  vX <- matrix(1, nrow = 1, ncol = p)
  if (scaling == 2L) {
    vX <- matrix(apply(Xtrain, 2, sd), nrow = 1)
    vX[!is.finite(vX) | vX == 0] <- 1
    Xtrain <- sweep(Xtrain, 2, vX[1, ], "/")
  }

  mY <- matrix(colMeans(Ytrain), nrow = 1)
  Yc <- sweep(Ytrain, 2, mY[1, ], "-")

  S <- .metal_crossprod(Xtrain, Yc)
  s <- .truncated_rsvd_metal(
    S,
    max_ncomp_eff,
    rsvd_oversample = rsvd_oversample,
    rsvd_power = rsvd_power,
    seed = seed
  )
  max_ncomp_eff <- min(max_ncomp_eff, ncol(s$U), nrow(s$Vt))
  R <- s$U[, seq_len(max_ncomp_eff), drop = FALSE]
  Q <- t(s$Vt[seq_len(max_ncomp_eff), , drop = FALSE])
  Ttrain <- .metal_mm(Xtrain, R)
  G_full <- .metal_crossprod(Ttrain, Ttrain)

  store_B <- .should_store_coefficients(p, m, length_ncomp, TRUE)
  B <- if (store_B) array(0, dim = c(p, m, length_ncomp)) else NULL
  C_latent <- array(0, dim = c(max_ncomp_eff, max_ncomp_eff, length_ncomp))
  W_latent <- array(0, dim = c(max_ncomp_eff, m, length_ncomp))
  Yfit <- if (fit) array(0, dim = c(n, m, length_ncomp)) else NULL
  R2Y <- rep(NA_real_, length_ncomp)

  for (i in seq_len(length_ncomp)) {
    mc <- min(ncomp[i], max_ncomp_eff)
    R_mc <- R[, seq_len(mc), drop = FALSE]
    Q_mc <- Q[, seq_len(mc), drop = FALSE]
    G_mc <- G_full[seq_len(mc), seq_len(mc), drop = FALSE]
    D_mc <- diag(s$s[seq_len(mc)], nrow = mc, ncol = mc)
    coeff_latent <- solve(G_mc, D_mc)
    C_i <- matrix(0, nrow = max_ncomp_eff, ncol = max_ncomp_eff)
    C_i[seq_len(mc), seq_len(mc)] <- coeff_latent
    C_latent[, , i] <- C_i
    W_i <- coeff_latent %*% t(Q_mc)
    W_latent[seq_len(mc), , i] <- W_i
    if (store_B) {
      B[, , i] <- .metal_mm(R_mc, W_i)
    }
    if (fit) {
      yf <- .metal_mm(Ttrain[, seq_len(mc), drop = FALSE], W_i)
      R2Y[i] <- RQ(Yc, yf)
      Yfit[, , i] <- sweep(yf, 2, mY[1, ], "+")
    }
  }

  out <- list(
    C_latent = C_latent,
    W_latent = W_latent,
    Q = Q,
    Ttrain = Ttrain,
    R = R,
    mX = mX,
    vX = vX,
    mY = mY,
    p = p,
    m = m,
    ncomp = ncomp,
    Yfit = Yfit,
    R2Y = R2Y,
    backend = "metal",
    svd.method = "metal_rsvd"
  )
  if (store_B) {
    out$B <- B
  }
  out <- .annotate_coefficient_storage(out, store_B)
  class(out) <- "fastPLS"
  out
}

.pls_model2_fast_metal <- function(Xtrain,
                                   Ytrain,
                                   ncomp,
                                   scaling,
                                   fit,
                                   rsvd_oversample,
                                   rsvd_power,
                                   seed) {
  n <- nrow(Xtrain); p <- ncol(Xtrain); m <- ncol(Ytrain)
  ncomp <- sort(unique(as.integer(ncomp)))
  max_ncomp <- max(ncomp)
  length_ncomp <- length(ncomp)

  mX <- matrix(0, nrow = 1, ncol = p)
  if (scaling < 3L) {
    mX <- matrix(colMeans(Xtrain), nrow = 1)
    Xtrain <- sweep(Xtrain, 2, mX[1, ], "-")
  }
  vX <- matrix(1, nrow = 1, ncol = p)
  if (scaling == 2L) {
    vX <- matrix(apply(Xtrain, 2, sd), nrow = 1)
    vX[!is.finite(vX) | vX == 0] <- 1
    Xtrain <- sweep(Xtrain, 2, vX[1, ], "/")
  }

  mY <- matrix(colMeans(Ytrain), nrow = 1)
  Y <- sweep(Ytrain, 2, mY[1, ], "-")
  max_ncomp_eff <- min(max_ncomp, n - 1L, p)
  if (max_ncomp_eff < 1L) {
    stop("SIMPLS Metal effective rank is < 1", call. = FALSE)
  }
  native <- metal_simpls_resident_cpp(
    Xtrain,
    Y,
    as.integer(max_ncomp_eff),
    as.integer(max(2L, rsvd_power + 1L)),
    as.integer(seed)
  )
  RR <- native$R
  QQ <- native$Q
  Bfull <- native$B
  max_ncomp_eff <- dim(Bfull)[3L]
  ncomp <- pmin(ncomp, max_ncomp_eff)
  store_B <- TRUE
  B <- array(0, dim = c(p, m, length_ncomp))
  Yfit <- if (fit) array(0, dim = c(n, m, length_ncomp)) else NULL
  R2Y <- rep(NA_real_, length_ncomp)
  for (i in seq_len(length_ncomp)) {
    mc <- max(1L, min(ncomp[i], max_ncomp_eff))
    B[, , i] <- Bfull[, , mc]
    if (fit) {
      yf <- Xtrain %*% B[, , i]
      Yfit[, , i] <- sweep(yf, 2, mY[1, ], "+")
      R2Y[i] <- RQ(Ytrain, matrix(Yfit[, , i], nrow = n, ncol = m))
    }
  }

  out <- list(
    P = matrix(0, nrow = 0, ncol = 0),
    Q = QQ,
    Ttrain = matrix(0, nrow = 0, ncol = 0),
    R = RR,
    mX = mX,
    vX = vX,
    mY = mY,
    p = p,
    m = m,
    ncomp = ncomp,
    Yfit = Yfit,
    R2Y = R2Y,
    backend = "metal",
    svd.method = "metal_resident_simpls"
  )
  if (store_B) {
    out$B <- B
  }
  out <- .annotate_coefficient_storage(out, store_B)
  class(out) <- "fastPLS"
  out
}

.pls_predict_metal <- function(object, Xtest, proj = FALSE) {
  Xscaled <- .fastpls_preprocess_test(Xtest, object$mX, object$vX)
  ncomp <- as.integer(object$ncomp)
  ns <- length(ncomp)
  n <- nrow(Xscaled)
  m <- as.integer(object$m)
  Ypred <- array(0, dim = c(n, m, ns))

  Tfull <- NULL
  if (!is.null(object$R) && length(object$R) > 0L) {
    maxc <- min(max(ncomp), ncol(object$R))
    Tfull <- .metal_mm(Xscaled, object$R[, seq_len(maxc), drop = FALSE])
  }

  for (i in seq_len(ns)) {
    mc <- min(ncomp[i], if (!is.null(Tfull)) ncol(Tfull) else dim(object$B)[3L])
    if (!is.null(object$W_latent) && !is.null(Tfull)) {
      W <- matrix(object$W_latent[seq_len(mc), , i], nrow = mc, ncol = m)
      y <- .metal_mm(Tfull[, seq_len(mc), drop = FALSE], W)
    } else if (!is.null(object$B)) {
      B_i <- matrix(object$B[, , i], nrow = object$p, ncol = object$m)
      y <- .metal_mm(Xscaled, B_i)
    } else if (!is.null(object$R) && !is.null(object$Q) && !is.null(Tfull)) {
      B_i <- .metal_mm(
        object$R[, seq_len(mc), drop = FALSE],
        t(object$Q[, seq_len(mc), drop = FALSE])
      )
      y <- .metal_mm(Xscaled, B_i)
    } else {
      stop("Metal prediction requires compact factors or coefficients.", call. = FALSE)
    }
    Ypred[, , i] <- sweep(y, 2, as.numeric(object$mY[1, ]), "+")
  }
  out <- list(Ypred = Ypred)
  if (isTRUE(proj) && !is.null(Tfull)) {
    out$Ttest <- Tfull
  }
  out
}

.opls_filter_metal <- function(X, Y, north, scaling) {
  prep <- .fastpls_preprocess_train(X, scaling)
  Xf <- prep$X
  Yc <- sweep(as.matrix(Y), 2, colMeans(as.matrix(Y)), "-")
  north <- as.integer(north)
  W_orth <- matrix(0, nrow = ncol(Xf), ncol = max(0L, north))
  P_orth <- matrix(0, nrow = ncol(Xf), ncol = max(0L, north))
  used <- 0L
  if (north > 0L) {
    for (a in seq_len(north)) {
      s <- fastsvd(.metal_crossprod(Xf, Yc), ncomp = 1L, backend = "metal", method = "rsvd", power = 1L)
      w <- s$u[, 1L, drop = FALSE]
      w_norm <- sqrt(sum(w * w))
      if (!is.finite(w_norm) || w_norm <= 0) break
      w <- w / w_norm
      tt <- .metal_mm(Xf, w)
      tt_ss <- drop(crossprod(tt))
      if (!is.finite(tt_ss) || tt_ss <= 0) break
      pp <- .metal_crossprod(Xf, tt) / tt_ss
      w_orth <- pp - w %*% crossprod(w, pp) / drop(crossprod(w))
      wo_norm <- sqrt(sum(w_orth * w_orth))
      if (!is.finite(wo_norm) || wo_norm <= 0) break
      w_orth <- w_orth / wo_norm
      t_orth <- .metal_mm(Xf, w_orth)
      to_ss <- drop(crossprod(t_orth))
      if (!is.finite(to_ss) || to_ss <= 0) break
      p_orth <- .metal_crossprod(Xf, t_orth) / to_ss
      Xf <- Xf - .metal_outer(t_orth, p_orth)
      used <- used + 1L
      W_orth[, used] <- w_orth[, 1L]
      P_orth[, used] <- p_orth[, 1L]
    }
  }
  if (used == 0L) {
    W_orth <- matrix(0, nrow = ncol(Xf), ncol = 0L)
    P_orth <- matrix(0, nrow = ncol(Xf), ncol = 0L)
  } else {
    W_orth <- W_orth[, seq_len(used), drop = FALSE]
    P_orth <- P_orth[, seq_len(used), drop = FALSE]
  }
  list(X = Xf, mX = prep$mX, vX = prep$vX, W_orth = W_orth, P_orth = P_orth, north = used)
}

.opls_apply_filter_metal <- function(X, mX, vX, W_orth, P_orth) {
  Xf <- .fastpls_preprocess_test(X, mX, vX)
  if (ncol(W_orth) > 0L) {
    for (a in seq_len(ncol(W_orth))) {
      t_orth <- .metal_mm(Xf, W_orth[, a, drop = FALSE])
      Xf <- Xf - .metal_outer(t_orth, P_orth[, a, drop = FALSE])
    }
  }
  Xf
}

.kernel_matrix_metal <- function(X1, X2, kernel, gamma, degree, coef0) {
  dots <- .metal_mm(X1, t(X2))
  if (identical(kernel, "linear")) {
    return(dots)
  }
  if (identical(kernel, "poly")) {
    return((gamma * dots + coef0)^as.integer(degree))
  }
  n1 <- rowSums(X1 * X1)
  n2 <- rowSums(X2 * X2)
  dist2 <- outer(n1, n2, "+") - 2 * dots
  dist2[dist2 < 0 & dist2 > -1e-10] <- 0
  exp(-gamma * dist2)
}

.pls_metal_fit_core <- function(Xtrain,
                                Ytrain,
                                ncomp,
                                scaling,
                                method,
                                fit,
                                rsvd_oversample,
                                rsvd_power,
                                seed) {
  if (identical(method, "plssvd")) {
    cap <- .cap_plssvd_ncomp(ncomp, nrow(Xtrain), ncol(Xtrain), ncol(Ytrain), warn = TRUE)
    return(.pls_model1_metal(
      Xtrain, Ytrain, cap$ncomp, scaling, fit,
      rsvd_oversample = rsvd_oversample,
      rsvd_power = rsvd_power,
      seed = seed
    ))
  }
  if (!isTRUE(.metal_resident_simpls_enabled()) &&
      !isTRUE(.metal_experimental_iterative_enabled())) {
    stop(
      "backend='metal' requires the Metal SIMPLS-family path; enable FASTPLS_METAL_RESIDENT_SIMPLS or use backend='cpu'.",
      call. = FALSE
    )
  }
  .pls_model2_fast_metal(
    Xtrain, Ytrain, ncomp, scaling, fit,
    rsvd_oversample = rsvd_oversample,
    rsvd_power = rsvd_power,
    seed = seed
  )
}

.pls_metal_finish <- function(model,
                              Xtrain,
                              Ytrain_original,
                              yprep,
                              classifier,
                              lda_ridge,
                              return_variance,
                              Xtest,
                              Ytest,
                              proj) {
  model$predict_backend <- "metal"
  model$backend <- "metal"
  model$svd.method <- "metal_rsvd"
  model$predict_latent_ok <- TRUE
  model <- .enable_flash_prediction(model, "cpu")
  model$predict_backend <- "metal"
  model <- .attach_gaussian_y(model, yprep$gaussian)
  model$classification <- yprep$classification
  model$lev <- yprep$lev
  model <- .decode_gaussian_y_outputs(model, Ytrain_original)
  model <- .attach_lda_classifier(model, Xtrain, Ytrain_original, classifier, lda_ridge)
  model <- .maybe_attach_pls_variance_explained(model, Xtrain, return_variance)
  class(model) <- "fastPLS"
  if (!is.null(Xtest)) {
    res <- predict(model, Xtest, Ytest = Ytest, proj = proj)
    model <- c(model, res)
    class(model) <- "fastPLS"
  }
  model
}

.pls_metal <- function(Xtrain,
                       Ytrain,
                       Xtest = NULL,
                       Ytest = NULL,
                       ncomp = 2,
                       scaling = c("centering", "autoscaling", "none"),
                       method = c("simpls", "plssvd", "opls", "kernelpls"),
                       north = 1L,
                       kernel = c("linear", "rbf", "poly"),
                       gamma = NULL,
                       degree = 3L,
                       coef0 = 1,
                       rsvd_oversample = 10L,
                       rsvd_power = 1L,
                       seed = 1L,
                       gaussian_y = FALSE,
                       gaussian_y_dim = NULL,
                       classifier = c("argmax", "lda", "cknn"),
                       lda_ridge = 1e-8,
                       fit = FALSE,
                       return_variance = TRUE,
                       proj = FALSE) {
  if (!isTRUE(has_metal())) {
    stop("backend='metal' requires Apple Metal support.", call. = FALSE)
  }
  method <- match.arg(method)
  scaling <- match.arg(scaling)
  kernel <- match.arg(kernel)
  classifier <- .resolve_classifier_for_backend(classifier, "metal")

  Xtrain <- as.matrix(Xtrain)
  Ytrain_original <- Ytrain
  yprep <- .prepare_gaussian_y(
    Ytrain,
    Xtrain,
    gaussian_y = gaussian_y,
    gaussian_y_dim = gaussian_y_dim,
    seed = seed,
    backend = "cpu"
  )
  Ymat <- yprep$Ytrain
  scal <- pmatch(scaling, c("centering", "autoscaling", "none"))[1]

  if (identical(method, "opls")) {
    filt <- .opls_filter_metal(Xtrain, .supervised_response_matrix(Ytrain_original), north, scaling)
    inner <- .pls_metal_fit_core(
      filt$X, Ymat, ncomp, 3L, "simpls", fit,
      rsvd_oversample, rsvd_power, seed
    )
    inner <- .pls_metal_finish(
      inner, filt$X, Ytrain_original, yprep, classifier, lda_ridge,
      return_variance, NULL, NULL, FALSE
    )
    out <- list(
      inner_model = inner,
      mX = filt$mX,
      vX = filt$vX,
      W_orth = filt$W_orth,
      P_orth = filt$P_orth,
      north = filt$north,
      opls_engine = "metal",
      ncomp = inner$ncomp,
      backend = "metal",
      predict_backend = "metal",
      svd.method = inner$svd.method
    )
    out <- .inherit_inner_variance_explained(out, inner)
    class(out) <- c("fastPLSOpls", "fastPLS")
    if (!is.null(Xtest)) {
      res <- predict(out, Xtest, Ytest = Ytest, proj = proj)
      out <- c(out, res)
      class(out) <- c("fastPLSOpls", "fastPLS")
    }
    return(out)
  }

  if (identical(method, "kernelpls")) {
    if (identical(kernel, "linear")) {
      model <- .pls_metal_fit_core(
        Xtrain, Ymat, ncomp, scal, "simpls", fit,
        rsvd_oversample, rsvd_power, seed
      )
      model$kernel <- "linear"
      model$kernel_engine <- "metal_direct"
      model$kernel_linear_direct <- TRUE
      return(.pls_metal_finish(
        model, Xtrain, Ytrain_original, yprep, classifier, lda_ridge,
        return_variance, Xtest, Ytest, proj
      ))
    }

    prep <- .fastpls_preprocess_train(Xtrain, scaling)
    gamma <- .kernel_pls_gamma(gamma, prep$X)
    K <- .kernel_matrix_metal(prep$X, prep$X, kernel, gamma, degree, coef0)
    kc <- .center_kernel_train_base(K)
    inner <- .pls_metal_fit_core(
      kc$K, Ymat, ncomp, 3L, "simpls", fit,
      rsvd_oversample, rsvd_power, seed
    )
    inner <- .pls_metal_finish(
      inner, kc$K, Ytrain_original, yprep, classifier, lda_ridge,
      return_variance, NULL, NULL, FALSE
    )
    out <- list(
      inner_model = inner,
      Xref = prep$X,
      mX = prep$mX,
      vX = prep$vX,
      kernel = kernel,
      kernel_id = .kernel_pls_kernel_id(kernel),
      gamma = gamma,
      degree = as.integer(degree),
      coef0 = coef0,
      kernel_center = kc,
      kernel_engine = "metal",
      ncomp = inner$ncomp,
      backend = "metal",
      predict_backend = "metal",
      svd.method = inner$svd.method
    )
    out <- .inherit_inner_variance_explained(out, inner)
    class(out) <- c("fastPLSKernel", "fastPLS")
    if (!is.null(Xtest)) {
      res <- predict(out, Xtest, Ytest = Ytest, proj = proj)
      out <- c(out, res)
      class(out) <- c("fastPLSKernel", "fastPLS")
    }
    return(out)
  }

  model <- .pls_metal_fit_core(
    Xtrain, Ymat, ncomp, scal, method, fit,
    rsvd_oversample, rsvd_power, seed
  )
  model$pls_method <- method
  .pls_metal_finish(
    model, Xtrain, Ytrain_original, yprep, classifier, lda_ridge,
    return_variance, Xtest, Ytest, proj
  )
}

#' Partial Least Squares with selectable model family and backend
#'
#' `pls()` is the main fastPLS user entry point. It routes PLSSVD, SIMPLS, OPLS,
#' and kernel PLS through the selected CPU, CUDA, or Metal backend while keeping
#' low-level implementation functions internal.
#'
#' @param Xtrain Numeric training predictor matrix.
#' @param Ytrain Training response (numeric or factor).
#' @param Xtest Optional test predictor matrix.
#' @param Ytest Optional test response for `Q2Y`.
#' @param ncomp Number of components (scalar or vector).
#' @param scaling One of \code{centering}, \code{autoscaling}, or \code{none}.
#' @param method One of \code{simpls}, \code{plssvd}, \code{opls}, or \code{kernelpls}.
#'   `simpls` uses the fastPLS accelerated SIMPLS core.
#' @param svd.method SVD algorithm family for compiled CPU fits: \code{irlba} or
#'   \code{rsvd}. Use CUDA or Metal backends for native GPU
#'   fits where available.
#' @param gaussian_y Logical; when `TRUE`, fit PLS to a Gaussian random
#'   low-dimensional response sketch and decode predictions back to the original
#'   response scale or class labels. The default is `FALSE`.
#' @param gaussian_y_dim Number of Gaussian response dimensions. When `NULL`,
#'   the default is `min(ncol(Xtrain), 100)`.
#' @param classifier Classification decision rule. \code{argmax} keeps the
#'   standard PLS-DA response-score argmax. \code{lda} fits an LDA classifier on
#'   the PLS latent scores. \code{cknn} is the compact name for the PLS-score
#'   candidate-kNN classifier: class centroids in PLS score space choose
#'   candidate classes, then every sample is reranked by within-candidate kNN in
#'   the same PLS score space. The compiled implementation is selected automatically from
#'   `backend`: C++ for \code{cpu}, CUDA for \code{cuda}, and Metal for \code{metal}
#'   where available.
#' @param candidate_knn_k Number of same-class PLS-score neighbours used by
#'   the candidate-kNN classifier.
#' @param candidate_tau Positive temperature used to smooth the neighbour
#'   similarities in candidate-kNN scoring.
#' @param candidate_alpha Weight of the centroid/prototype candidate score
#'   added to the local kNN score.
#' @param candidate_top_m Number of centroid-ranked candidate classes passed to
#'   the kNN reranker.
#' @param lda_ridge Relative diagonal ridge added to the pooled LDA covariance.
#' @param fit Return fitted values and `R2Y` when `TRUE`.
#' @param return_variance Compute predictor-space latent-variable variance
#'   explained. Set to `FALSE` for timing/memory benchmarks that do not need
#'   plotting variance metadata.
#' @param proj Return projected `Ttest` when `TRUE`.
#' @param perm.test Run permutation test.
#' @param times Number of permutations.
#' @param backend Implementation backend: \code{cpu} for compiled CPU, \code{cuda}
#'   for CUDA-native fitting, or experimental \code{metal} for Apple Metal
#'   randomized-SVD/GEMM acceleration.
#' @param north Number of orthogonal components removed by OPLS.
#' @param kernel Kernel type for kernel PLS: \code{linear}, \code{rbf}, or \code{poly}.
#' @param gamma Kernel scale. Defaults internally to `1 / ncol(Xtrain)`.
#' @param degree Polynomial kernel degree.
#' @param coef0 Polynomial kernel offset.
#' @param ... Optional SVD tuning controls forwarded to the selected backend.
#'   Use the same compact names documented in [fastsvd()], such as
#'   `oversample`, `power`, `svds_tol`, `work`, `maxit`, `tol`, `eps`,
#'   `svtol`, and `seed`.
#' @return A `fastPLS` object. Fitted objects include `variance_explained`,
#'   `cumulative_variance_explained`, and matching `x_*` aliases containing the
#'   fraction of training predictor variance explained by each latent variable.
#' @examples
#' X <- as.matrix(mtcars[, c("disp", "hp", "wt", "qsec")])
#' y <- mtcars$mpg
#' fit <- pls(X, y, ncomp = 2, method = "simpls", backend = "cpu",
#'            svd.method = "rsvd", return_variance = FALSE)
#' head(predict(fit, X)$Ypred)
#' @export
pls =  function (Xtrain,
                 Ytrain,
                 Xtest = NULL,
                 Ytest = NULL,
                 ncomp=2,
                 scaling = c("centering", "autoscaling","none"),
                 method = c("simpls", "plssvd", "opls", "kernelpls"),
                 svd.method = c("irlba", "rsvd"),
                 gaussian_y = FALSE,
                 gaussian_y_dim = NULL,
		                 classifier = c("argmax", "lda", "cknn"),
		                 lda_ridge = 1e-8,
			                 candidate_knn_k = 10L,
			                 candidate_tau = 0.2,
			                 candidate_alpha = 0.75,
			                 candidate_top_m = 20L,
	                 fit = FALSE,
                 return_variance = TRUE,
                 proj = FALSE,
                 perm.test = FALSE,
                 times = 100,
                 backend = c("cpu", "cuda", "metal"),
                 north = 1L,
                 kernel = c("linear", "rbf", "poly"),
                 gamma = NULL,
                 degree = 3L,
                 coef0 = 1,
                 ...)
{

  scal = pmatch(scaling, c("centering", "autoscaling","none"))[1]
  dots <- .svd_control_from_dots(list(...))
  svd_ctl <- .resolve_svd_control(
    svd.method = if (missing(svd.method)) NULL else svd.method,
    dots = dots$dots,
    context = "pls()"
  )
  svd.method <- svd_ctl$svd.method
  rsvd_oversample <- svd_ctl$rsvd_oversample
  rsvd_power <- svd_ctl$rsvd_power
  svds_tol <- svd_ctl$svds_tol
  irlba_work <- svd_ctl$irlba_work
  irlba_maxit <- svd_ctl$irlba_maxit
  irlba_tol <- svd_ctl$irlba_tol
  irlba_eps <- svd_ctl$irlba_eps
  irlba_svtol <- svd_ctl$irlba_svtol
  seed <- svd_ctl$seed
  requested_method <- match.arg(method, c("simpls", "plssvd", "opls", "kernelpls"))
		  backend <- .normalize_public_backend(backend)
		  backend_compiled <- .compiled_backend(backend)
			  classifier <- .resolve_classifier_for_backend(classifier, backend)
		  candidate_knn_k <- max(1L, as.integer(candidate_knn_k)[1L])
		  candidate_tau <- as.numeric(candidate_tau)[1L]
		  candidate_alpha <- as.numeric(candidate_alpha)[1L]
		  candidate_top_m <- max(1L, as.integer(candidate_top_m)[1L])
		  if (!is.finite(candidate_tau) || candidate_tau <= 0) {
		    stop("candidate_tau must be a finite positive number", call. = FALSE)
		  }
		  if (!is.finite(candidate_alpha)) {
		    stop("candidate_alpha must be finite", call. = FALSE)
		  }
		  old_class_bias_options <- options(
		    fastPLS.candidate_knn_k = candidate_knn_k,
		    fastPLS.candidate_tau = candidate_tau,
		    fastPLS.candidate_alpha = candidate_alpha,
		    fastPLS.candidate_top_m = candidate_top_m
	  )
	  on.exit(options(old_class_bias_options), add = TRUE)

  if (identical(backend, "metal")) {
    return(.pls_metal(
      Xtrain = Xtrain,
      Ytrain = Ytrain,
      Xtest = Xtest,
      Ytest = Ytest,
      ncomp = ncomp,
      scaling = scaling,
      method = requested_method,
      north = north,
      kernel = kernel,
      gamma = gamma,
      degree = degree,
      coef0 = coef0,
      rsvd_oversample = rsvd_oversample,
      rsvd_power = rsvd_power,
      seed = seed,
      gaussian_y = gaussian_y,
      gaussian_y_dim = gaussian_y_dim,
      classifier = classifier,
      lda_ridge = lda_ridge,
      fit = fit,
      return_variance = return_variance,
      proj = proj
    ))
  }

  if (identical(requested_method, "opls")) {
    fit_fun <- switch(backend_compiled, cpp = opls_cpp, cuda = opls_cuda)
    args <- list(
      Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest,
      ncomp = ncomp, north = north, scaling = scaling,
      rsvd_oversample = rsvd_oversample, rsvd_power = rsvd_power,
      svds_tol = svds_tol, seed = seed,
      gaussian_y = gaussian_y, gaussian_y_dim = gaussian_y_dim,
      fit = fit, proj = proj
    )
    args <- c(args, list(classifier = classifier, lda_ridge = lda_ridge))
    args$return_variance <- return_variance
    if (!identical(backend, "cuda")) {
      args <- c(args, list(
        svd.method = svd.method,
        irlba_work = irlba_work,
        irlba_maxit = irlba_maxit,
        irlba_tol = irlba_tol,
        irlba_eps = irlba_eps,
        irlba_svtol = irlba_svtol
      ))
    }
    return(do.call(fit_fun, args))
  }

  if (identical(requested_method, "kernelpls")) {
    fit_fun <- switch(backend_compiled, cpp = kernel_pls_cpp, cuda = kernel_pls_cuda)
    args <- list(
      Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest,
      ncomp = ncomp, scaling = scaling, kernel = kernel, gamma = gamma,
      degree = degree, coef0 = coef0,
      rsvd_oversample = rsvd_oversample, rsvd_power = rsvd_power,
      svds_tol = svds_tol, seed = seed,
      gaussian_y = gaussian_y, gaussian_y_dim = gaussian_y_dim,
      fit = fit, proj = proj
    )
    args <- c(args, list(classifier = classifier, lda_ridge = lda_ridge))
    args$return_variance <- return_variance
    if (!identical(backend, "cuda")) {
      args <- c(args, list(
        svd.method = svd.method,
        irlba_work = irlba_work,
        irlba_maxit = irlba_maxit,
        irlba_tol = irlba_tol,
        irlba_eps = irlba_eps,
        irlba_svtol = irlba_svtol
      ))
    }
    return(do.call(fit_fun, args))
  }

  if (identical(backend, "cuda")) {
    if (identical(requested_method, "plssvd")) {
      return(plssvd_gpu(
        Xtrain = Xtrain,
        Ytrain = Ytrain,
        Xtest = Xtest,
        Ytest = Ytest,
        ncomp = ncomp,
        scaling = scaling,
        rsvd_oversample = rsvd_oversample,
        rsvd_power = rsvd_power,
        svds_tol = svds_tol,
        seed = seed,
        fit = fit,
        proj = proj,
        gaussian_y = gaussian_y,
        gaussian_y_dim = gaussian_y_dim,
	        classifier = classifier,
	        lda_ridge = lda_ridge,
            return_variance = return_variance
	      ))
    }
    return(simpls_gpu(
      Xtrain = Xtrain,
      Ytrain = Ytrain,
      Xtest = Xtest,
      Ytest = Ytest,
      ncomp = ncomp,
      scaling = scaling,
      rsvd_oversample = rsvd_oversample,
      rsvd_power = rsvd_power,
      svds_tol = svds_tol,
      seed = seed,
      fit = fit,
      proj = proj,
      gaussian_y = gaussian_y,
      gaussian_y_dim = gaussian_y_dim,
	      classifier = classifier,
	      lda_ridge = lda_ridge,
          return_variance = return_variance
	    ))
  }

  meth = .normalize_pls_method(requested_method)
  svd.method <- .normalize_svd_method(svd.method)
  svd.method <- match.arg(svd.method, c("irlba", "cpu_rsvd"))
  svdmeth <- .svd_method_id(svd.method)

  Xtrain = as.matrix(Xtrain)
  Ytrain_original <- Ytrain
  yprep <- .prepare_gaussian_y(
    Ytrain,
    Xtrain,
    gaussian_y = gaussian_y,
    gaussian_y_dim = gaussian_y_dim,
    seed = seed,
    backend = "cpu"
  )
  Ytrain <- yprep$Ytrain
  classification <- yprep$classification
  lev <- yprep$lev

  if (meth == 3L && svd.method %in% c("cpu_rsvd", "cuda_rsvd")) {
    tuned <- .resolve_simpls_fast_rsvd_tuning(
      n = nrow(Xtrain),
      p = ncol(Xtrain),
      q = ncol(Ytrain),
      svd.method = svd.method
    )
    if (!("rsvd_oversample" %in% svd_ctl$supplied)) rsvd_oversample <- tuned$rsvd_oversample
    if (!("rsvd_power" %in% svd_ctl$supplied)) rsvd_power <- tuned$rsvd_power
  }

  use_xprod_default <- meth %in% c(1L, 3L) && (
    (identical(svd.method, "cpu_rsvd") &&
       .should_use_xprod_default(ncol(Xtrain), ncol(Ytrain), ncomp)) ||
      (identical(svd.method, "irlba") &&
         .should_use_xprod_irlba_default(nrow(Xtrain), ncol(Xtrain), ncol(Ytrain), ncomp))
  )
  xprod_precision_default <- if (identical(svd.method, "irlba")) "implicit_irlba" else "implicit64"

  if(meth==1){
    cap <- .cap_plssvd_ncomp(ncomp, nrow(Xtrain), ncol(Xtrain), ncol(Ytrain), warn = TRUE)
    ncomp <- cap$ncomp
    if (use_xprod_default) {
      model=pls.model1.rsvd.xprod.precision(
        Xtrain,
        Ytrain,
        ncomp=ncomp,
        fit=fit,
        scaling=scal,
        rsvd_oversample=rsvd_oversample,
        rsvd_power=rsvd_power,
        svds_tol=svds_tol,
        irlba_work=irlba_work,
        irlba_maxit=irlba_maxit,
        irlba_tol=irlba_tol,
        irlba_eps=irlba_eps,
        irlba_svtol=irlba_svtol,
        seed=seed,
        xprod_precision=xprod_precision_default
      )
    } else {
      model=pls.model1(
        Xtrain,
        Ytrain,
        ncomp=ncomp,
        fit=fit,
        scaling=scal,
        svd.method=svdmeth,
        rsvd_oversample=rsvd_oversample,
        rsvd_power=rsvd_power,
        svds_tol=svds_tol,
        irlba_work=irlba_work,
        irlba_maxit=irlba_maxit,
        irlba_tol=irlba_tol,
        irlba_eps=irlba_eps,
        irlba_svtol=irlba_svtol,
        seed=seed
      )
    }
  }
  if(meth==2){
    model=pls.model2(
      Xtrain,
      Ytrain,
      ncomp=ncomp,
      fit=fit,
      scaling=scal,
      svd.method=svdmeth,
      rsvd_oversample=rsvd_oversample,
      rsvd_power=rsvd_power,
      svds_tol=svds_tol,
      irlba_work=irlba_work,
      irlba_maxit=irlba_maxit,
      irlba_tol=irlba_tol,
      irlba_eps=irlba_eps,
      irlba_svtol=irlba_svtol,
      seed=seed
    )
  }
  if(meth==3){
    if (use_xprod_default) {
      model=pls.model2.fast.rsvd.xprod.precision(
        Xtrain,
        Ytrain,
        ncomp=ncomp,
        fit=fit,
        scaling=scal,
        rsvd_oversample=rsvd_oversample,
        rsvd_power=rsvd_power,
        svds_tol=svds_tol,
        irlba_work=irlba_work,
        irlba_maxit=irlba_maxit,
        irlba_tol=irlba_tol,
        irlba_eps=irlba_eps,
        irlba_svtol=irlba_svtol,
        seed=seed,
        xprod_precision=xprod_precision_default,
        return_ttrain=FALSE
      )
    } else {
      model=pls.model2.fast(
        Xtrain,
        Ytrain,
        ncomp=ncomp,
        fit=fit,
        scaling=scal,
        svd.method=svdmeth,
        rsvd_oversample=rsvd_oversample,
        rsvd_power=rsvd_power,
        svds_tol=svds_tol,
        irlba_work=irlba_work,
        irlba_maxit=irlba_maxit,
        irlba_tol=irlba_tol,
        irlba_eps=irlba_eps,
        irlba_svtol=irlba_svtol,
        seed=seed,
        return_ttrain=FALSE
      )
    }
  }
  model$xprod_default=use_xprod_default
  model$pls_method <- if (meth == 1L) "plssvd" else "simpls"
  model$predict_latent_ok <- TRUE
  if (isTRUE(fit)) model <- .attach_train_scores(model, Xtrain)
  model <- .enable_flash_prediction(model, "cpu")
  model <- .attach_gaussian_y(model, yprep$gaussian)
  model$classification=classification
  model$lev=lev
	  model <- .decode_gaussian_y_outputs(model, Ytrain_original)
	  model <- .attach_lda_classifier(
	    model,
	    Xtrain,
	    Ytrain_original,
	    classifier,
	    lda_ridge
	  )
  model <- .maybe_attach_pls_variance_explained(model, Xtrain, return_variance)


#  model$R2Y[i] = 1 - sum(((Ytrain - model$Yfit[, , i]))^2)/sum(t(t(Ytrain) -  colMeans(Ytrain))^2)



  # PLS analysis
  if(!is.null(Xtest)){
    Xtest = as.matrix(Xtest)
    res=predict(model,Xtest,Ytest,proj=proj)
    model=c(model,res)
    # output


      #    o$scoreXtest=as.matrix(Xtest) %*% o$R[,1:ncomp]
      if (perm.test) {
        v = matrix(NA,nrow=times,ncol=length(ncomp))
        for (i in 1:times) {
          ss = sample(1:nrow(Xtrain))
          Xtrain_permuted = Xtrain[ss, ]

          if(meth==1){
            model_perm=pls.model1(
              Xtrain_permuted,
              Ytrain,
              ncomp=ncomp,
              scaling=scal,
              svd.method=svdmeth,
              rsvd_oversample=rsvd_oversample,
              rsvd_power=rsvd_power,
              svds_tol=svds_tol,
              irlba_work=irlba_work,
              irlba_maxit=irlba_maxit,
              irlba_tol=irlba_tol,
              irlba_eps=irlba_eps,
              irlba_svtol=irlba_svtol,
              seed=seed
            )
          }
          if(meth==2){
            model_perm=pls.model2(
              Xtrain_permuted,
              Ytrain,
              ncomp=ncomp,
              scaling=scal,
              svd.method=svdmeth,
              rsvd_oversample=rsvd_oversample,
              rsvd_power=rsvd_power,
              svds_tol=svds_tol,
              irlba_work=irlba_work,
              irlba_maxit=irlba_maxit,
              irlba_tol=irlba_tol,
              irlba_eps=irlba_eps,
              irlba_svtol=irlba_svtol,
              seed=seed
            )
          }
          if(meth==3){
            model_perm=pls.model2.fast(
              Xtrain_permuted,
              Ytrain,
              ncomp=ncomp,
              scaling=scal,
              svd.method=svdmeth,
              rsvd_oversample=rsvd_oversample,
              rsvd_power=rsvd_power,
              svds_tol=svds_tol,
              irlba_work=irlba_work,
              irlba_maxit=irlba_maxit,
              irlba_tol=irlba_tol,
              irlba_eps=irlba_eps,
              irlba_svtol=irlba_svtol,
              seed=seed
            )
          }

          model_perm <- .attach_gaussian_y(model_perm, yprep$gaussian)
          model_perm$classification <- classification
          model_perm$lev <- lev
          res_perm=predict(model_perm,Xtest,Ytest)

          v[i,]=res_perm$Q2Y
        }
        model$pval=NULL
        for(j in 1:length(ncomp)){
          model$pval[j] = sum(v[,j] > model$Q2Y)/times
        }


      }
  }
    if(classification){

      if(fit){
        Yfitlab = as.data.frame(matrix(nrow = nrow(Xtrain), ncol = length(ncomp)))
        colnames(Yfitlab)=paste("ncomp=",ncomp,sep="")
        for (i in 1:length(ncomp)) {
          t = apply(model$Yfit[, , i], 1, which.max)
          Yfitlab[, i] = factor(lev[t], levels = lev)
        }
        model$Yfit=Yfitlab
      }
    }



  class(model)="fastPLS"
  model
}

.cv_best_index <- function(metrics) {
  values <- as.numeric(metrics$metric_value)
  metric_names <- tolower(as.character(metrics$metric_name))
  finite <- is.finite(values)
  if (!any(finite)) {
    return(1L)
  }
  loss_metric <- any(metric_names[finite] %in% c("rmsd", "rmse", "mae", "mse"))
  idx <- if (loss_metric) {
    which.min(ifelse(finite, values, Inf))
  } else {
    which.max(ifelse(finite, values, -Inf))
  }
  as.integer(idx[1L])
}

.cv_numeric_metric_values <- function(metrics) {
  values <- as.numeric(metrics$metric_value)
  names(values) <- as.character(metrics$metric_name)
  values
}

.cv_extract_prediction_at <- function(cv_res, idx) {
  if (!is.null(cv_res$class_pred)) {
    return(cv_res$pred[[idx]])
  }
  if (!is.null(cv_res$Ypred)) {
    return(cv_res$Ypred[, , idx, drop = FALSE])
  }
  cv_res$pred[[idx]]
}

.cv_metric_name_at <- function(metrics, idx) {
  as.character(metrics$metric_name[[idx]])
}

#' Cross-validation component optimization for PLS
#'
#' Performs k-fold CV over candidate component counts using the compiled
#' C++/CUDA/Metal CV core shared with [pls.single.cv()]. Nonlinear Metal
#' kernel CV still uses the public [pls()] backend inside an R-level fold loop.
#'
#' @inheritParams pls
#' @param Xdata Predictor matrix.
#' @param Ydata Response (numeric or factor).
#' @param constrain Optional grouping vector for grouped cross-validation. It
#'   must have one value per sample. Samples with the same value are assigned to
#'   the same fold, so all rows from the same patient, subject, batch, or
#'   technical replicate stay together in training or test data. When `NULL`,
#'   each sample is treated as its own group.
#' @param kfold Number of folds, or `"loocv"` for leave-one-out
#'   cross-validation. When `constrain` is supplied, LOOCV means
#'   leave-one-constraint-group-out: samples sharing the same constraint value
#'   are always held out together and are never split across training and test.
#' @param method One of \code{simpls}, \code{plssvd}, \code{opls}, or \code{kernelpls}.
#' @param backend Implementation backend: \code{cpu}, \code{cuda}, or \code{metal}.
#' @param seed Random seed used for fold assignment and randomized SVD steps.
#' @param gamma Kernel scale. Defaults internally to `1 / ncol(Xdata)`.
#' @param gaussian_y_dim Number of Gaussian response dimensions. When `NULL`,
#'   the default is `min(ncol(Xdata), 100)`.
#' @param classifier Classification rule for factor responses: `"argmax"`,
#'   `"lda"`, or `"cknn"`.
#' @param lda_ridge Ridge added to the pooled LDA covariance diagonal.
#' @param candidate_knn_k,candidate_tau,candidate_alpha,candidate_top_m
#'   Candidate-kNN controls used when `classifier = "cknn"`.
#' @param return_scores Store score predictions for classification when `TRUE`.
#' @param xprod Use the matrix-free cross-product route where available.
#'   `NULL` applies fastPLS defaults.
#' @param ... Optional SVD tuning controls forwarded to the selected backend.
#'   Use the same compact names documented in [fastsvd()], such as
#'   `oversample`, `power`, `svds_tol`, `work`, `maxit`, `tol`, `eps`,
#'   and `svtol`.
#' @return List with `optim_comp`, decoded `pred`, `metrics`, legacy
#'   `Q2Y`/`R2Y` metric vectors, `fold`, backend metadata, and `Ypred` when
#'   score predictions are stored.
#' @examples
#' idx <- c(1:12, 51:62, 101:112)
#' X <- as.matrix(iris[idx, 1:4])
#' y <- factor(iris[idx, 5])
#' opt <- optim.pls.cv(X, y, ncomp = 1:2, kfold = 3, method = "simpls",
#'                     backend = "cpu", svd.method = "rsvd", seed = 1)
#' opt$optim_comp
#' @export
optim.pls.cv =  function (Xdata,
                          Ydata,
                          ncomp=2,
                          constrain=NULL,
                          scaling = c("centering", "autoscaling","none"),
                          method = c("simpls", "plssvd", "opls", "kernelpls"),
                          backend = c("cpu", "cuda", "metal"),
                          svd.method = c("irlba", "rsvd"),
                          seed = 1L,
                          kfold=10,
                          north = 1L,
                          kernel = c("linear", "rbf", "poly"),
                          gamma = NULL,
                          degree = 3L,
                          coef0 = 1,
                          gaussian_y = FALSE,
                          gaussian_y_dim = NULL,
                          classifier = c("argmax", "lda", "cknn"),
                          lda_ridge = 1e-8,
                          candidate_knn_k = 10L,
                          candidate_tau = 0.2,
                          candidate_alpha = 0.75,
                          candidate_top_m = 20L,
                          return_scores = FALSE,
                          xprod = NULL,
                          ...)
{
  if (sum(is.na(Xdata)) > 0) {
    stop("Missing values are present")
  }
  method <- match.arg(method)
  backend <- .normalize_public_backend(backend)
  backend_compiled <- .compiled_backend(backend)
  scaling <- match.arg(scaling)
  dots <- .svd_control_from_dots(list(...))
  svd_ctl <- .resolve_svd_control(
    svd.method = svd.method,
    dots = c(dots$dots, list(seed = seed)),
    context = "optim.pls.cv()"
  )
  svd.method <- match.arg(.normalize_svd_method(svd_ctl$svd.method), c("irlba", "cpu_rsvd"))
  rsvd_oversample <- svd_ctl$rsvd_oversample
  rsvd_power <- svd_ctl$rsvd_power
  svds_tol <- svd_ctl$svds_tol
  irlba_work <- svd_ctl$irlba_work
  irlba_maxit <- svd_ctl$irlba_maxit
  irlba_tol <- svd_ctl$irlba_tol
  irlba_eps <- svd_ctl$irlba_eps
  irlba_svtol <- svd_ctl$irlba_svtol
  seed <- svd_ctl$seed
  kernel <- match.arg(kernel)
  classifier <- .normalize_classifier_public(classifier)
  Xdata <- as.matrix(Xdata)
  if (is.null(constrain)) constrain <- seq_len(nrow(Xdata))

  if (!identical(kernel, "linear") && !identical(backend, "metal")) {
    stop("Nonlinear kernel CV is not available in the compiled CV helper; use backend='metal' or kernel='linear'.", call. = FALSE)
  }

  res <- if ((identical(backend, "metal") && !identical(classifier, "argmax")) ||
             (!identical(classifier, "argmax") && isTRUE(gaussian_y)) ||
             (identical(backend, "metal") && (!identical(kernel, "linear") || isTRUE(gaussian_y)))) {
    .pls_cv_via_pls(
      Xdata = Xdata,
      Ydata = Ydata,
      constrain = constrain,
      ncomp = as.integer(ncomp),
      kfold = kfold,
      scaling = scaling,
      method = method,
      backend = backend,
      svd.method = svd.method,
      rsvd_oversample = rsvd_oversample,
      rsvd_power = rsvd_power,
      svds_tol = svds_tol,
      irlba_work = irlba_work,
      irlba_maxit = irlba_maxit,
      irlba_tol = irlba_tol,
      irlba_eps = irlba_eps,
      irlba_svtol = irlba_svtol,
      seed = seed,
      xprod = xprod,
      north = north,
      kernel = kernel,
      gamma = gamma,
      degree = degree,
      coef0 = coef0,
      gaussian_y = gaussian_y,
      gaussian_y_dim = gaussian_y_dim,
      classifier = classifier,
      lda_ridge = lda_ridge,
      candidate_knn_k = candidate_knn_k,
      candidate_tau = candidate_tau,
      candidate_alpha = candidate_alpha,
      candidate_top_m = candidate_top_m,
      return_scores = return_scores
    )
  } else {
    .pls_cv_compiled(
    Xdata = Xdata,
    Ydata = Ydata,
    constrain = constrain,
    ncomp = as.integer(ncomp),
    kfold = kfold,
    scaling = scaling,
    method = method,
    backend = backend_compiled,
    svd.method = svd.method,
    rsvd_oversample = rsvd_oversample,
    rsvd_power = rsvd_power,
    svds_tol = svds_tol,
    irlba_work = irlba_work,
    irlba_maxit = irlba_maxit,
    irlba_tol = irlba_tol,
    irlba_eps = irlba_eps,
    irlba_svtol = irlba_svtol,
    seed = seed,
    xprod = xprod,
    north = north,
    return_scores = return_scores,
    classifier = classifier,
    lda_ridge = lda_ridge,
    candidate_knn_k = candidate_knn_k,
    candidate_tau = candidate_tau,
    candidate_alpha = candidate_alpha,
    candidate_top_m = candidate_top_m
    )
  }
  best_idx <- .cv_best_index(res$metrics)
  values <- as.numeric(res$metrics$metric_value)
  res$optim_comp <- as.integer(res$ncomp[[best_idx]])
  res$best_index <- best_idx
  res$best_metric_name <- .cv_metric_name_at(res$metrics, best_idx)
  res$best_metric_value <- values[[best_idx]]
  res$Q2Y <- values
  res$R2Y <- values
  res$Ypred_optim <- .cv_extract_prediction_at(res, best_idx)
  res
}






#' Nested cross-validation for PLS
#'
#' Runs outer/inner CV loops for performance estimation and component selection.
#' Inner component optimization uses the same compiled C++/CUDA/Metal CV core as
#' [pls.single.cv()] and [optim.pls.cv()]. Nonlinear Metal kernel CV still uses
#' the public [pls()] backend inside R-level folds.
#'
#' @inheritParams pls
#' @param Xdata Predictor matrix.
#' @param Ydata Response (numeric or factor).
#' @param constrain Grouping vector for grouped cross-validation. It must have
#'   one value per sample. Samples with the same value are assigned to the same
#'   fold, so all rows from the same patient, subject, batch, or technical
#'   replicate stay together in training or test data. The default
#'   `1:nrow(Xdata)` treats every sample as an independent group.
#' @param runn Number of repeated runs.
#' @param kfold_inner Inner-fold count, or `"loocv"` to leave out one
#'   constraint group at a time inside each outer training set.
#' @param kfold_outer Outer-fold count, or `"loocv"` to leave out one
#'   constraint group at a time in the outer loop. In both loops, samples
#'   sharing the same constraint value are never split across training and test.
#' @param method One of \code{simpls}, \code{plssvd}, \code{opls}, or \code{kernelpls}.
#' @param backend Implementation backend: \code{cpu}, \code{cuda}, or \code{metal}.
#' @param seed Random seed used for outer/inner fold assignment and randomized
#'   SVD steps.
#' @param gamma Kernel scale. Defaults internally to `1 / ncol(Xdata)`.
#' @param xprod Use the matrix-free cross-product route where available for
#'   inner component optimization. `NULL` applies fastPLS defaults.
#' @param ... Optional SVD tuning controls forwarded to the selected backend.
#'   Use the same compact names documented in [fastsvd()], such as
#'   `oversample`, `power`, `svds_tol`, `work`, `maxit`, `tol`, `eps`,
#'   and `svtol`.
#' @return List of nested CV outputs and summaries.
#' @examples
#' idx <- c(1:10, 51:60, 101:110)
#' X <- as.matrix(iris[idx, 1:4])
#' y <- factor(iris[idx, 5])
#' dcv <- pls.double.cv(X, y, ncomp = 1:2, runn = 1, kfold_inner = 2,
#'                      kfold_outer = 2, method = "simpls", backend = "cpu",
#'                      svd.method = "rsvd", seed = 1)
#' names(dcv)
#' @export
pls.double.cv = function(Xdata,
                         Ydata,
                         ncomp=2,
                         constrain=1:nrow(Xdata),
                         scaling = c("centering", "autoscaling","none"),
                         method = c("simpls", "plssvd", "opls", "kernelpls"),
                         backend = c("cpu", "cuda", "metal"),
                         svd.method = c("irlba", "rsvd"),
                         seed = 1L,
                         perm.test=FALSE,
                         times=100,
                         runn=10,
                         kfold_inner=10,
                         kfold_outer=10,
                         north = 1L,
                         kernel = c("linear", "rbf", "poly"),
                         gamma = NULL,
                         degree = 3L,
                         coef0 = 1,
                         classifier = c("argmax", "lda", "cknn"),
                         lda_ridge = 1e-8,
                         xprod = NULL,
                         ...){

  if(sum(is.na(Xdata))>0) {
    stop("Missing values are present")
  }
  method <- match.arg(method)
  backend <- .normalize_public_backend(backend)
  scaling <- match.arg(scaling)
  dots <- .svd_control_from_dots(list(...))
  svd_ctl <- .resolve_svd_control(
    svd.method = svd.method,
    dots = c(dots$dots, list(seed = seed)),
    context = "pls.double.cv()"
  )
  svd.method <- match.arg(.normalize_svd_method(svd_ctl$svd.method), c("irlba", "cpu_rsvd"))
  rsvd_oversample <- svd_ctl$rsvd_oversample
  rsvd_power <- svd_ctl$rsvd_power
  svds_tol <- svd_ctl$svds_tol
  irlba_work <- svd_ctl$irlba_work
  irlba_maxit <- svd_ctl$irlba_maxit
  irlba_tol <- svd_ctl$irlba_tol
  irlba_eps <- svd_ctl$irlba_eps
  irlba_svtol <- svd_ctl$irlba_svtol
  seed <- svd_ctl$seed
  kernel <- match.arg(kernel)
			  classifier <- .resolve_classifier_for_backend(classifier, backend)
  if (!identical(kernel, "linear") && !identical(backend, "metal")) {
    stop("Nonlinear kernel double CV is not available in the compiled CV helper; use backend='metal' or kernel='linear'.", call. = FALSE)
  }

  Xdata <- as.matrix(Xdata)
  constrain <- as.integer(as.factor(constrain))
  classification <- is.factor(Ydata)
  Ydata_original <- Ydata
  if (classification) {
    lev <- levels(Ydata)
    conf_tot <- matrix(0, ncol = length(lev), nrow = length(lev), dimnames = list(lev, lev))
  } else {
    lev <- NULL
    Ydata <- as.matrix(Ydata)
  }
  ncomp <- as.integer(ncomp)

  res <- list(results = vector("list", as.integer(runn)))
  Q2Y <- rep(NA_real_, as.integer(runn))
  R2Y <- rep(NA_real_, as.integer(runn))
  metric_name <- rep(NA_character_, as.integer(runn))
  bb <- integer(0)

  if (classification) {
    vote_tot <- matrix(0, nrow = nrow(Xdata), ncol = length(lev), dimnames = list(NULL, lev))
  } else {
    Ypred_tot <- matrix(0, nrow = nrow(Xdata), ncol = ncol(Ydata))
  }

  for (j in seq_len(as.integer(runn))) {
    fold <- .make_single_cv_folds(
      Ydata = if (classification) Ydata_original else Ydata,
      constrain = constrain,
      kfold = kfold_outer,
      seed = as.integer(seed) + j - 1L
    )
    fold_values <- sort(unique(fold))
    nfold_outer <- length(fold_values)
    best_comp <- integer(nfold_outer)
    inner_results <- vector("list", nfold_outer)
    if (classification) {
      run_pred_chr <- rep(NA_character_, nrow(Xdata))
    } else {
      run_pred <- matrix(NA_real_, nrow = nrow(Xdata), ncol = ncol(Ydata))
    }

    for (f in seq_along(fold_values)) {
      fold_value <- fold_values[[f]]
      test_idx <- which(fold == fold_value)
      train_idx <- which(fold != fold_value)
      if (!length(test_idx) || !length(train_idx)) {
        next
      }
      Ytrain <- if (classification) Ydata_original[train_idx] else Ydata[train_idx, , drop = FALSE]
      Ytest <- if (classification) Ydata_original[test_idx] else Ydata[test_idx, , drop = FALSE]
      if (classification && length(unique(Ytrain)) < 2L) {
        fallback <- names(which.max(table(Ytrain)))
        run_pred_chr[test_idx] <- fallback
        best_comp[f] <- min(ncomp)
        next
      }

      inner <- optim.pls.cv(
        Xdata = Xdata[train_idx, , drop = FALSE],
        Ydata = Ytrain,
        ncomp = ncomp,
        constrain = constrain[train_idx],
        scaling = scaling,
        method = method,
        backend = backend,
        svd.method = svd.method,
        rsvd_oversample = rsvd_oversample,
        rsvd_power = rsvd_power,
        svds_tol = svds_tol,
        seed = as.integer(seed) + 1000L * j + f,
        irlba_work = irlba_work,
        irlba_maxit = irlba_maxit,
        irlba_tol = irlba_tol,
        irlba_eps = irlba_eps,
        irlba_svtol = irlba_svtol,
        kfold = kfold_inner,
        north = north,
        kernel = kernel,
        gamma = gamma,
        degree = degree,
        coef0 = coef0,
        return_scores = FALSE,
        xprod = xprod
      )
      best_comp[f] <- as.integer(inner$optim_comp[[1L]])
      inner_results[[f]] <- inner

      fit <- pls(
        Xtrain = Xdata[train_idx, , drop = FALSE],
        Ytrain = Ytrain,
        Xtest = Xdata[test_idx, , drop = FALSE],
        Ytest = Ytest,
        ncomp = best_comp[f],
        scaling = scaling,
        method = method,
        svd.method = svd.method,
        rsvd_oversample = rsvd_oversample,
        rsvd_power = rsvd_power,
        svds_tol = svds_tol,
        seed = as.integer(seed) + 2000L * j + f,
        irlba_work = irlba_work,
        irlba_maxit = irlba_maxit,
        irlba_tol = irlba_tol,
        irlba_eps = irlba_eps,
        irlba_svtol = irlba_svtol,
        fit = FALSE,
        proj = FALSE,
        backend = backend,
        north = north,
        kernel = kernel,
        gamma = gamma,
        degree = degree,
        coef0 = coef0,
        classifier = classifier,
        lda_ridge = lda_ridge,
      )

      if (classification) {
        pred <- fit$Ypred
        pred <- if (is.data.frame(pred)) pred[[1L]] else pred
        run_pred_chr[test_idx] <- as.character(pred)
      } else {
        pred <- fit$Ypred
        if (length(dim(pred)) == 3L) {
          pred <- pred[, , 1L, drop = TRUE]
        }
        run_pred[test_idx, ] <- as.matrix(pred)
      }
    }

    if (classification) {
      pred_factor <- factor(run_pred_chr, levels = lev)
      tab <- table(pred_factor, factor(Ydata_original, levels = lev))
      conf_tot <- conf_tot + tab
      idx <- match(run_pred_chr, lev)
      ok <- is.finite(idx)
      if (any(ok)) {
        vote_tot[cbind(which(ok), idx[ok])] <- vote_tot[cbind(which(ok), idx[ok])] + 1
      }
      Q2Y[j] <- mean(as.character(pred_factor) == as.character(Ydata_original), na.rm = TRUE)
      R2Y[j] <- Q2Y[j]
      metric_name[j] <- "accuracy"
      res$results[[j]] <- list(
        Ypred = pred_factor,
        pred = pred_factor,
        fold = fold + 1L,
        optim_comp = best_comp,
        inner = inner_results,
        metric_name = "accuracy",
        metric_value = Q2Y[j],
        backend = backend,
        method = method
      )
      bb <- c(bb, best_comp)
    } else {
      Ypred_tot <- Ypred_tot + run_pred
      metric <- .cv_metric_from_matrix(Ydata, run_pred, Ytrain = Ydata)
      Q2Y[j] <- metric$metric_value
      R2Y[j] <- metric$metric_value
      metric_name[j] <- metric$metric_name
      res$results[[j]] <- list(
        Ypred = run_pred,
        pred = run_pred,
        fold = fold + 1L,
        optim_comp = best_comp,
        inner = inner_results,
        metric_name = metric$metric_name,
        metric_value = metric$metric_value,
        backend = backend,
        method = method
      )
      bb <- c(bb, best_comp)
    }
  }

  if (classification) {
    final_idx <- max.col(vote_tot, ties.method = "first")
    final_idx[rowSums(vote_tot) <= 0] <- NA_integer_
    Ypredlab <- factor(ifelse(is.na(final_idx), NA_character_, lev[final_idx]), levels = lev)
    res$Ypred <- Ypredlab
    conf_final <- table(Ypredlab, factor(Ydata_original, levels = lev))
    acc_tot <- round(sum(diag(conf_final)), digits = 1)
    acc_tot_perc <- 100 * acc_tot / nrow(Xdata)
    res$acc_tot <- paste(acc_tot, " (", acc_tot_perc, "%)", sep = "")
    conf_perc <- suppressWarnings(t(t(conf_final) / colSums(conf_final)) * 100)
    conf_perc[!is.finite(conf_perc)] <- 0
    conf_txt <- matrix(
      paste(round(conf_final, digits = 1), " (", round(conf_perc, digits = 1), "%)", sep = ""),
      ncol = length(lev),
      dimnames = list(lev, lev)
    )
    res$conf <- conf_txt
    res$vote_counts <- vote_tot
  } else {
    res$Ypred <- Ypred_tot / as.integer(runn)
  }

  res$Q2Y <- Q2Y
  res$R2Y <- R2Y
  res$metric_name <- metric_name
  res$medianR2Y <- median(R2Y, na.rm = TRUE)
  res$CI95R2Y <- as.numeric(quantile(R2Y, c(0.025, 0.975), na.rm = TRUE))
  res$medianQ2Y <- median(Q2Y, na.rm = TRUE)
  res$CI95Q2Y <- as.numeric(quantile(Q2Y, c(0.025, 0.975), na.rm = TRUE))
  res$bcomp <- names(which.max(table(bb)))
  res$backend <- backend
  res$method <- method

  if (perm.test) {
    sampled <- numeric(as.integer(times))
    for (i in seq_len(as.integer(times))) {
      ss <- sample(seq_len(nrow(Xdata)))
      sampled[i] <- median(pls.double.cv(
        Xdata = Xdata[ss, , drop = FALSE],
        Ydata = Ydata_original,
        ncomp = ncomp,
        constrain = constrain,
        scaling = scaling,
        method = method,
        backend = backend,
        svd.method = svd.method,
        rsvd_oversample = rsvd_oversample,
        rsvd_power = rsvd_power,
        svds_tol = svds_tol,
        seed = as.integer(seed) + 3000L + i,
        irlba_work = irlba_work,
        irlba_maxit = irlba_maxit,
        irlba_tol = irlba_tol,
        irlba_eps = irlba_eps,
        irlba_svtol = irlba_svtol,
        perm.test = FALSE,
        runn = runn,
        kfold_inner = kfold_inner,
        kfold_outer = kfold_outer,
        north = north,
        kernel = kernel,
        gamma = gamma,
        degree = degree,
        coef0 = coef0,
        classifier = classifier,
        lda_ridge = lda_ridge,
        xprod = xprod
      )$Q2Y, na.rm = TRUE)
    }
    loss_metric <- any(tolower(metric_name) %in% c("rmsd", "rmse", "mae", "mse"))
    observed <- median(Q2Y, na.rm = TRUE)
    res$Q2Ysampled <- sampled
    res$p.value <- if (loss_metric) {
      mean(sampled <= observed, na.rm = TRUE)
    } else {
      mean(sampled >= observed, na.rm = TRUE)
    }
  }

  res
}







Vip <- function(object) {

  SS <- c(object$Q)^2 * colSums(object$Ttrain^2)
  Wnorm2 <- colSums(object$R^2)
  SSW <- sweep(object$R^2, 2, SS / Wnorm2, "*")
  sqrt(nrow(SSW) * apply(SSW, 1, cumsum) / cumsum(SS))
}


#' Variable importance in projection (VIP)
#'
#' Computes VIP trajectories from fitted model components.
#'
#' @param model Fitted `fastPLS` model.
#' @return Numeric matrix (single response) or list of matrices (multi-response).
#' @examples
#' X <- as.matrix(mtcars[, c("disp", "hp", "wt", "qsec")])
#' y <- mtcars$mpg
#' fit <- pls(X, y, ncomp = 1, method = "plssvd", backend = "cpu",
#'            svd.method = "rsvd", return_variance = FALSE)
#' ViP(fit)
#' @export
ViP <- function(model) {

  u <- nrow(model$Q)
  if (u==1) return (as.matrix(Vip(model)))
  V <- list ()
  for (i in 1:u) V[[i]] <- Vip(list(Q=model$Q[i,], Ttrain=model$Ttrain, R=model$R))
  return (V)
}


fastcor <- function(a, b=NULL, byrow=TRUE, diag=TRUE) {

  ## if byrow == T rows are correlated (much faster) else columns
  ## if diag == T only the diagonal of the cor matrix is returned (much faster)
  ## b can be NULL

  if (!byrow) a <- t(a)
  a <- a - rowMeans(a)
  a <- a / sqrt(rowSums(a*a))
  if (!is.null(b)) {
    if (!byrow) b <- t(b)
    b <- b - rowMeans(b)
    b <- b / sqrt(rowSums(b*b))
    if (diag) return (rowSums(a*b)) else return (tcrossprod(a,b))
  } else return (tcrossprod(a))
}
