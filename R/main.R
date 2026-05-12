## Historical R IRLBA prototype retained only as a commented development note.
##  stopifnot(work>nu)
##  IRLB(X, nu, work, maxit, tol, eps, svtol)
##}

###' @import irlba
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

.simpls_fast_profile_notice <- function(context = "simpls_fast") {
  sprintf(
    "%s ignores deprecated low-level fast_* tuning arguments except fast_block, which is still honored for accuracy-sensitive refresh control.",
    context
  )
}

.resolve_simpls_fast_profile <- function(fast_block,
                                         fast_center_t,
                                         fast_reorth_v,
                                         fast_incremental,
                                         fast_inc_iters,
                                         fast_defl_cache,
                                         missing_fast_block,
                                         missing_fast_center_t,
                                         missing_fast_reorth_v,
                                         missing_fast_incremental,
                                         missing_fast_inc_iters,
                                         missing_fast_defl_cache,
                                         context = "simpls_fast") {
  profile <- list(
    fast_block = 1L,
    fast_center_t = FALSE,
    fast_reorth_v = FALSE,
    fast_incremental = TRUE,
    fast_inc_iters = 2L,
    fast_defl_cache = TRUE
  )
  warn <- FALSE
  if (!missing_fast_block) {
    profile$fast_block <- max(1L, as.integer(fast_block))
  }
  if (!missing_fast_center_t && !identical(isTRUE(fast_center_t), FALSE)) warn <- TRUE
  if (!missing_fast_reorth_v && !identical(isTRUE(fast_reorth_v), FALSE)) warn <- TRUE
  if (!missing_fast_incremental && !identical(isTRUE(fast_incremental), TRUE)) warn <- TRUE
  if (!missing_fast_inc_iters && !identical(as.integer(fast_inc_iters), 2L)) warn <- TRUE
  if (!missing_fast_defl_cache && !identical(isTRUE(fast_defl_cache), TRUE)) warn <- TRUE
  if (warn) {
    warning(.simpls_fast_profile_notice(context), call. = FALSE)
  }
  profile
}

.with_fastpls_fast_options <- function(expr,
                                       fast_block = 1L,
                                       fast_center_t = FALSE,
                                       fast_reorth_v = FALSE,
                                       fast_incremental = TRUE,
                                       fast_inc_iters = 2L,
                                       fast_defl_cache = TRUE,
                                       return_ttrain = FALSE) {
  old <- c(
    FASTPLS_FAST_BLOCK = Sys.getenv("FASTPLS_FAST_BLOCK", unset = NA_character_),
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
    FASTPLS_FAST_BLOCK = as.character(as.integer(fast_block)),
    FASTPLS_FAST_CENTER_T = if (isTRUE(fast_center_t)) "1" else "0",
    FASTPLS_FAST_REORTH_V = if (isTRUE(fast_reorth_v)) "1" else "0",
    FASTPLS_FAST_INCREMENTAL = "1",
    FASTPLS_FAST_INC_ITERS = as.character(as.integer(fast_inc_iters)),
    FASTPLS_FAST_DEFLCACHE = if (isTRUE(fast_defl_cache)) "1" else "0",
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

.normalize_classifier <- function(classifier) {
  if (length(classifier) > 1L) {
    classifier <- classifier[1L]
  }
  classifier <- match.arg(classifier, c("argmax", "lda_cpp", "lda_cuda", "class_bias", "class_bias_cpp", "class_bias_cuda"))
  if (identical(classifier, "class_bias")) "class_bias_cpp" else classifier
}

.is_class_bias_classifier <- function(classifier) {
  !is.null(classifier) && classifier %in% c("class_bias_cpp", "class_bias_cuda")
}

.class_bias_backend <- function(classifier) {
  if (identical(classifier, "class_bias_cuda")) "cuda" else "cpp"
}

.normalize_class_bias_method <- function(class_bias_method) {
  match.arg(class_bias_method, c("iter_count_ratio", "count_ratio"))
}

.class_bias_calibration_indices <- function(y, fraction = 1, seed = 1L) {
  y <- factor(y)
  n <- length(y)
  fraction <- as.numeric(fraction)[1L]
  if (!is.finite(fraction) || fraction <= 0 || fraction > 1) {
    stop("class_bias_calibration_fraction must be in (0, 1]", call. = FALSE)
  }
  if (fraction >= 1) {
    return(seq_len(n))
  }

  old_seed <- if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
    get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  } else {
    NULL
  }
  on.exit({
    if (is.null(old_seed)) {
      if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
        rm(".Random.seed", envir = .GlobalEnv)
      }
    } else {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    }
  }, add = TRUE)

  set.seed(as.integer(seed)[1L])
  idx <- unlist(lapply(split(seq_len(n), y), function(ii) {
    ni <- length(ii)
    take <- max(1L, floor(ni * fraction))
    sample(ii, min(ni, take))
  }), use.names = FALSE)
  sort(unique(as.integer(idx)))
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

.class_bias_update <- function(y_true, pred_index, lev, scale, eps, clip) {
  truth <- tabulate(as.integer(factor(y_true, levels = lev)), nbins = length(lev))
  pred <- tabulate(as.integer(pred_index), nbins = length(lev))
  .class_bias_update_counts(truth, pred, scale, eps, clip)
}

.class_bias_update_counts <- function(truth, pred, scale, eps, clip) {
  delta <- scale * log((truth + eps) / (pred + eps))
  delta <- delta - mean(delta)
  if (is.finite(clip)) {
    delta <- pmax(pmin(delta, clip), -clip)
  }
  delta
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

.fit_class_bias <- function(model,
                            Xtrain,
                            y_train,
                            backend = c("cpp", "cuda"),
                            method = c("iter_count_ratio", "count_ratio"),
                            lambda = 0.05,
                            iter = 1L,
                            clip = Inf,
                            eps = 1,
                            calibration_fraction = 1,
                            seed = 1L) {
  backend <- match.arg(backend)
  method <- .normalize_class_bias_method(method)
  iter <- max(1L, as.integer(iter)[1L])
  cal_idx <- .class_bias_calibration_indices(
    y_train,
    fraction = calibration_fraction,
    seed = seed
  )
  bias <- matrix(0, nrow = length(model$lev), ncol = length(model$ncomp))
  rownames(bias) <- model$lev
  n_pass <- if (identical(method, "count_ratio")) 1L else iter
  truth_counts <- tabulate(as.integer(factor(y_train[cal_idx], levels = model$lev)), nbins = length(model$lev))
  block_size <- .fastpls_block_size(
    "fastPLS.class_bias_block_size",
    "FASTPLS_CLASS_BIAS_BLOCK_SIZE",
    default = 4096L
  )
  Xmat <- as.matrix(Xtrain)
  for (i in seq_len(n_pass)) {
    pred_counts <- matrix(0, nrow = length(model$lev), ncol = length(model$ncomp))
    for (start in seq(1L, length(cal_idx), by = block_size)) {
      stop <- min(length(cal_idx), start + block_size - 1L)
      rows <- cal_idx[start:stop]
      pred <- .class_bias_predict(
        model,
        Xmat[rows, , drop = FALSE],
        bias,
        top = 1L,
        proj = FALSE,
        backend = backend
      )
      for (a in seq_along(model$ncomp)) {
        pred_counts[, a] <- pred_counts[, a] +
          tabulate(as.integer(pred$Ypred_index[, a]), nbins = length(model$lev))
      }
    }
    for (a in seq_along(model$ncomp)) {
      bias[, a] <- bias[, a] + .class_bias_update_counts(
        truth = truth_counts,
        pred = pred_counts[, a],
        scale = lambda,
        eps = eps,
        clip = clip
      )
      bias[, a] <- bias[, a] - mean(bias[, a])
    }
  }
  colnames(bias) <- paste("ncomp=", model$ncomp, sep = "")
  attr(bias, "parameters") <- list(
    method = method,
    lambda = lambda,
    iter = n_pass,
    clip = clip,
    eps = eps,
    calibration_fraction = as.numeric(calibration_fraction)[1L],
    n_calibration = length(cal_idx),
    seed = as.integer(seed)[1L],
    backend = backend
  )
  bias
}

.normalize_regression_head <- function(regression_head) {
  match.arg(regression_head, c("standard", "linear_cpp", "linear_cuda"))
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

.fastpls_latent_scores <- function(object, X, ncomp = max(object$ncomp), backend = c("cpu", "cuda")) {
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
                                   class_bias_method = getOption("fastPLS.class_bias_method", "iter_count_ratio"),
                                   class_bias_lambda = getOption("fastPLS.class_bias_lambda", 0.05),
                                   class_bias_iter = getOption("fastPLS.class_bias_iter", 1L),
                                   class_bias_clip = getOption("fastPLS.class_bias_clip", Inf),
                                   class_bias_eps = getOption("fastPLS.class_bias_eps", 1),
                                   class_bias_calibration_fraction = getOption("fastPLS.class_bias_calibration_fraction", 1),
                                   class_bias_seed = getOption("fastPLS.class_bias_seed", 1L)) {
  classifier <- .normalize_classifier(classifier)
  model$classification_rule <- classifier
  model$lda_backend <- classifier
  if (!isTRUE(model$classification) || identical(classifier, "argmax")) {
    return(model)
  }
  if (!is.factor(Ytrain)) {
    stop("Classification head requires factor Ytrain", call. = FALSE)
  }
  if (.is_class_bias_classifier(classifier)) {
    backend <- .class_bias_backend(classifier)
    if (identical(backend, "cuda") && !isTRUE(has_cuda())) {
      warning("classifier='class_bias_cuda' requested but CUDA is unavailable; using class_bias_cpp.", call. = FALSE)
      backend <- "cpp"
      model$classification_rule <- "class_bias_cpp"
      model$lda_backend <- "class_bias_cpp"
    }
    model$class_bias_backend <- backend
    class_bias <- .fit_class_bias(
      model,
      Xtrain,
      Ytrain,
      backend = backend,
      method = class_bias_method,
      lambda = as.numeric(class_bias_lambda)[1L],
      iter = as.integer(class_bias_iter)[1L],
      clip = as.numeric(class_bias_clip)[1L],
      eps = as.numeric(class_bias_eps)[1L],
      calibration_fraction = as.numeric(class_bias_calibration_fraction)[1L],
      seed = as.integer(class_bias_seed)[1L]
    )
    model$class_bias <- class_bias
    model$class_bias_parameters <- attr(class_bias, "parameters", exact = TRUE)
    return(model)
  }
  if (identical(classifier, "lda_cuda") && !.cuda_matmul_available()) {
    warning("classifier='lda_cuda' requested but CUDA matrix multiply is unavailable; using lda_cpp.", call. = FALSE)
    classifier <- "lda_cpp"
    model$classification_rule <- classifier
    model$lda_backend <- classifier
  }
  model <- .attach_latent_projection_cache(model)
  y_codes <- as.integer(factor(Ytrain, levels = model$lev))
  if (anyNA(y_codes)) {
    stop("LDA classification received labels outside the training levels", call. = FALSE)
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

.attach_linear_regression_head <- function(model,
                                           Xtrain,
                                           Ytrain,
                                           regression_head = "standard") {
  regression_head <- .normalize_regression_head(regression_head)
  model$regression_head <- regression_head
  if (isTRUE(model$classification) || identical(regression_head, "standard")) {
    return(model)
  }
  if (is.factor(Ytrain)) {
    stop("Linear regression head requires numeric Ytrain", call. = FALSE)
  }
  if (identical(regression_head, "linear_cuda") && !.cuda_matmul_available()) {
    warning("regression_head='linear_cuda' requested but CUDA matrix multiply is unavailable; using linear_cpp.", call. = FALSE)
    regression_head <- "linear_cpp"
    model$regression_head <- regression_head
  }

  model <- .attach_latent_projection_cache(model)
  backend <- if (identical(regression_head, "linear_cuda") && .cuda_matmul_available()) "cuda" else "cpu"
  if (is.null(model$Ttrain) ||
      length(model$Ttrain) == 0L ||
      !all(dim(model$Ttrain) > 0L) ||
      ncol(as.matrix(model$Ttrain)) < max(as.integer(model$ncomp))) {
    model$Ttrain <- .fastpls_latent_scores(
      model,
      Xtrain,
      ncomp = max(model$ncomp),
      backend = backend
    )
  }
  Ttrain <- as.matrix(model$Ttrain)
  ncomp_eff <- pmin(as.integer(model$ncomp), ncol(Ttrain))
  ncomp_eff <- pmax(ncomp_eff, 1L)
  unique_ncomp <- sort(unique(ncomp_eff))
  linear_models <- linear_train_prefix_cpp(
    Ttrain[, seq_len(max(unique_ncomp)), drop = FALSE],
    as.matrix(Ytrain),
    unique_ncomp
  )
  names(linear_models) <- as.character(unique_ncomp)
  model$linear_head <- list(
    ncomp = unique_ncomp,
    models = linear_models
  )
  model
}

.fastpls_linear_predictions <- function(object,
                                        Xtest,
                                        Ttest = NULL,
                                        keep_ttest = FALSE) {
  if (is.null(object$linear_head) || is.null(object$linear_head$models)) {
    stop("This fastPLS object does not contain fitted linear-regression head parameters", call. = FALSE)
  }
  use_cuda <- identical(object$regression_head, "linear_cuda") && .cuda_matmul_available()
  ncomp_eff <- pmin(as.integer(object$ncomp), max(as.integer(object$linear_head$ncomp)))
  ncomp_eff <- pmax(ncomp_eff, 1L)
  if (is.null(Ttest)) {
    Ttest <- .fastpls_latent_scores(
      object,
      Xtest,
      ncomp = max(ncomp_eff),
      backend = if (use_cuda) "cuda" else "cpu"
    )
  } else {
    Ttest <- as.matrix(Ttest)
  }

  first_model <- object$linear_head$models[[as.character(ncomp_eff[1L])]]
  if (is.null(first_model)) {
    stop(sprintf("No fitted linear-regression head for ncomp=%s", ncomp_eff[1L]), call. = FALSE)
  }
  q <- length(as.numeric(first_model$intercept))
  Ypred <- array(
    NA_real_,
    dim = c(nrow(Ttest), q, length(object$ncomp))
  )
  for (i in seq_along(object$ncomp)) {
    k <- ncomp_eff[i]
    lm <- object$linear_head$models[[as.character(k)]]
    if (is.null(lm)) {
      stop(sprintf("No fitted linear-regression head for ncomp=%s", k), call. = FALSE)
    }
    pred <- if (use_cuda && exists("linear_predict_cuda", envir = asNamespace("fastPLS"), inherits = FALSE)) {
      get("linear_predict_cuda", envir = asNamespace("fastPLS"), inherits = FALSE)(
        Ttest[, seq_len(k), drop = FALSE],
        lm
      )
    } else {
      linear_predict_cpp(Ttest[, seq_len(k), drop = FALSE], lm)
    }
    Ypred[, , i] <- as.matrix(pred)
  }
  out <- list(Ypred = Ypred)
  if (isTRUE(keep_ttest)) {
    out$Ttest <- Ttest
  }
  out
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
  if (length(method) == 1L && !is.na(method) && as.character(method) %in% c("arpack", "dc")) {
    stop("svd.method='arpack' has been removed from fastPLS; use 'irlba' or 'cpu_rsvd'.", call. = FALSE)
  }
  method
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
                                gaussian_y_seed = 1L,
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
    codes <- .gaussian_y_class_codes(lev, dim, gaussian_y_seed)
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
    gaussian_y_seed,
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

.guard_removed_hybrid_cuda <- function(svd.method, context = "pls()") {
  if (length(svd.method) == 1L &&
      !is.na(svd.method) &&
      identical(as.character(svd.method), "cuda_rsvd")) {
    stop(
      sprintf(
        "The hybrid CUDA path via svd.method='cuda_rsvd' has been removed from %s; use pls(..., backend='cuda') for GPU-native fits instead.",
        context
      ),
      call. = FALSE
    )
  }
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
            fast_block = 1L,
            fast_center_t = FALSE,
            fast_reorth_v = FALSE,
            fast_incremental = TRUE,
            fast_inc_iters = 2L,
            fast_defl_cache = TRUE,
            return_ttrain = FALSE)
  {
    profile <- .resolve_simpls_fast_profile(
      fast_block = fast_block,
      fast_center_t = fast_center_t,
      fast_reorth_v = fast_reorth_v,
      fast_incremental = fast_incremental,
      fast_inc_iters = fast_inc_iters,
      fast_defl_cache = fast_defl_cache,
      missing_fast_block = missing(fast_block),
      missing_fast_center_t = missing(fast_center_t),
      missing_fast_reorth_v = missing(fast_reorth_v),
      missing_fast_incremental = missing(fast_incremental),
      missing_fast_inc_iters = missing(fast_inc_iters),
      missing_fast_defl_cache = missing(fast_defl_cache),
      context = "pls.model2.fast"
    )
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
        fast_block = profile$fast_block,
        fast_center_t = profile$fast_center_t,
        fast_reorth_v = profile$fast_reorth_v,
        fast_incremental = profile$fast_incremental,
        fast_inc_iters = profile$fast_inc_iters,
        fast_defl_cache = profile$fast_defl_cache,
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
            fast_block = 1L,
            fast_center_t = FALSE,
            fast_reorth_v = FALSE,
            fast_incremental = TRUE,
            fast_inc_iters = 2L,
            fast_defl_cache = TRUE,
            return_ttrain = FALSE)
  {
    xprod_precision <- match.arg(xprod_precision)
    precision_id <- switch(
      xprod_precision,
      double = 0L,
      implicit64 = 3L,
      implicit_irlba = 5L
    )
    profile <- .resolve_simpls_fast_profile(
      fast_block = fast_block,
      fast_center_t = fast_center_t,
      fast_reorth_v = fast_reorth_v,
      fast_incremental = fast_incremental,
      fast_inc_iters = fast_inc_iters,
      fast_defl_cache = fast_defl_cache,
      missing_fast_block = missing(fast_block),
      missing_fast_center_t = missing(fast_center_t),
      missing_fast_reorth_v = missing(fast_reorth_v),
      missing_fast_incremental = missing(fast_incremental),
      missing_fast_inc_iters = missing(fast_inc_iters),
      missing_fast_defl_cache = missing(fast_defl_cache),
      context = "pls.model2.fast.rsvd.xprod.precision"
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
      fast_block = profile$fast_block,
      fast_center_t = profile$fast_center_t,
      fast_reorth_v = profile$fast_reorth_v,
      fast_incremental = profile$fast_incremental,
      fast_inc_iters = profile$fast_inc_iters,
      fast_defl_cache = profile$fast_defl_cache,
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
            seed = 1L,
            fast_block = 1L,
            fast_center_t = FALSE,
            fast_reorth_v = FALSE,
            fast_incremental = TRUE,
            fast_inc_iters = 2L,
            fast_defl_cache = TRUE)
  {
    if (!has_cuda()) {
      stop("pls.model2.fast.gpu requires CUDA support")
    }
    profile <- .resolve_simpls_fast_profile(
      fast_block = fast_block,
      fast_center_t = fast_center_t,
      fast_reorth_v = fast_reorth_v,
      fast_incremental = fast_incremental,
      fast_inc_iters = fast_inc_iters,
      fast_defl_cache = fast_defl_cache,
      missing_fast_block = missing(fast_block),
      missing_fast_center_t = missing(fast_center_t),
      missing_fast_reorth_v = missing(fast_reorth_v),
      missing_fast_incremental = missing(fast_incremental),
      missing_fast_inc_iters = missing(fast_inc_iters),
      missing_fast_defl_cache = missing(fast_defl_cache),
      context = "pls.model2.fast.gpu"
    )
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
      ),
      fast_block = profile$fast_block,
      fast_center_t = profile$fast_center_t,
      fast_reorth_v = profile$fast_reorth_v,
      fast_incremental = profile$fast_incremental,
      fast_inc_iters = profile$fast_inc_iters,
      fast_defl_cache = profile$fast_defl_cache
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
#' @param predict.backend Prediction backend. `"auto"` uses FlashSVD-style
#'   low-rank prediction when compact factors are available and the low-rank
#'   application is expected to be beneficial.
#' @param flash.block_size Row block size for `"cpu_flash"` prediction.
#' @param top Number of ranked classes to return for classification.
#' @param top5 Convenience flag equivalent to `top = max(top, 5)`.
#' @param raw_scores If `TRUE`, keep raw classification score cubes as
#'   `Yscore` when available.
#' @param ... Unused.
#' @return A list containing `Ypred`, optional `Q2Y`, optional `Ttest`, and
#'   optional LDA scores for LDA classification models.
#' @export
predict.fastPLS = function(object, newdata, Ytest=NULL, proj=FALSE,
                           predict.backend = c("auto", "cpu", "cpu_flash", "cuda_flash"),
                           flash.block_size = NULL, top = 1L, top5 = FALSE,
                           raw_scores = FALSE, ...) {
  if (!is(object, "fastPLS")) {
    stop("object is not a fastPLS object")
  }
  predict.backend <- match.arg(predict.backend)
  top <- .resolve_top_k(top, top5)
  Xtest=as.matrix(newdata)
  use_cuda_flash <- identical(predict.backend, "cuda_flash") ||
    (identical(predict.backend, "auto") &&
       identical(object$predict_backend, "cuda_flash") &&
       isTRUE(has_cuda()))
  use_cpu_flash <- identical(predict.backend, "cpu_flash") ||
    (identical(predict.backend, "auto") &&
       .should_use_cpu_flash_prediction(object, Xtest))
  if (is.null(flash.block_size)) {
    flash.block_size <- object$flash_block_size
  }
  if (is.null(flash.block_size) || !length(flash.block_size) || is.na(flash.block_size)) {
    flash.block_size <- 4096L
  }
	  if (isTRUE(object$classification) &&
	      !is.null(object$classification_rule) &&
	      object$classification_rule %in% c("lda_cpp", "lda_cuda")) {
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
      is.null(Ytest) &&
      !isTRUE(raw_scores) &&
      !isTRUE(object$gaussian_y) &&
      (is.null(object$classification_rule) ||
         identical(object$classification_rule, "argmax") ||
         .is_class_bias_classifier(object$classification_rule))) {
    pred_backend <- if (!is.null(object$classification_rule) &&
                        .is_class_bias_classifier(object$classification_rule)) {
      object$class_bias_backend %||% .class_bias_backend(object$classification_rule)
    } else if (identical(object$predict_backend, "cuda_flash") && isTRUE(has_cuda())) {
      "cuda"
    } else {
      "cpp"
    }
    bias_res <- .class_bias_predict(
      object,
      Xtest,
      class_bias = if (!is.null(object$classification_rule) &&
                       .is_class_bias_classifier(object$classification_rule)) object$class_bias else NULL,
      top = top,
      proj = proj,
      backend = pred_backend
    )
    bias_res$Q2Y <- NULL
    return(bias_res)
  }
	  if (!isTRUE(object$classification) &&
	      !is.null(object$regression_head) &&
	      object$regression_head %in% c("linear_cpp", "linear_cuda")) {
	    lin_res <- .fastpls_linear_predictions(object, Xtest, keep_ttest = isTRUE(proj))
	    res <- list(Ypred = lin_res$Ypred, Q2Y = NULL)
	    if (isTRUE(proj)) {
	      res$Ttest <- lin_res$Ttest
	    }
	    if (!is.null(Ytest)) {
	      Ytest_transf <- as.matrix(Ytest)
	      for (i in seq_along(object$ncomp)) {
	        ypred_i <- matrix(
	          res$Ypred[, , i],
	          nrow = dim(res$Ypred)[1L],
	          ncol = dim(res$Ypred)[2L]
	        )
	        res$Q2Y[i] <- RQ(Ytest_transf, ypred_i)
	      }
	    }
	    return(res)
	  }
	  res <- if (isTRUE(use_cuda_flash)) {
    tryCatch(
      pls_predict_flash_cuda(object, Xtest, proj),
      error = function(e) {
        if (identical(predict.backend, "cuda_flash")) {
          stop(e)
        }
        pls_predict(object, Xtest, proj)
      }
    )
  } else if (isTRUE(use_cpu_flash)) {
    tryCatch(
      pls_predict_flash_cpu(object, Xtest, proj, as.integer(flash.block_size)),
      error = function(e) {
        if (identical(predict.backend, "cpu_flash")) {
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
        object$classification_rule %in% c("lda_cpp", "lda_cuda")) {
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
      bias <- if (!is.null(object$classification_rule) &&
                  .is_class_bias_classifier(object$classification_rule)) {
        object$class_bias
      } else {
        NULL
      }
      score_cube <- res$Ypred
      top_res <- .class_topk_from_score_cube(score_cube, object$lev, object$ncomp, class_bias = bias, top = top)
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

.kernel_matrix_r <- function(X1, X2, kernel, gamma, degree, coef0) {
  dots <- X1 %*% t(X2)
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

.center_kernel_train_r <- function(K) {
  col_means <- colMeans(K)
  row_means <- rowMeans(K)
  grand_mean <- mean(col_means)
  Kc <- sweep(K, 2, col_means, "-")
  Kc <- sweep(Kc, 1, row_means, "-")
  Kc <- Kc + grand_mean
  list(K = Kc, col_means = matrix(col_means, nrow = 1), grand_mean = grand_mean)
}

.center_kernel_test_r <- function(Ktest, train_col_means, train_grand_mean) {
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

.opls_filter_r <- function(X, Y, north, scaling) {
  prep <- .fastpls_preprocess_train(X, scaling)
  Xf <- prep$X
  Yc <- sweep(as.matrix(Y), 2, colMeans(as.matrix(Y)), "-")
  north <- as.integer(north)
  W_orth <- matrix(0, nrow = ncol(Xf), ncol = max(0L, north))
  P_orth <- matrix(0, nrow = ncol(Xf), ncol = max(0L, north))
  used <- 0L
  if (north > 0L) {
    for (a in seq_len(north)) {
      s <- base::svd(crossprod(Xf, Yc), nu = 1L, nv = 0L)
      w <- s$u[, 1L, drop = FALSE]
      w_norm <- sqrt(sum(w * w))
      if (!is.finite(w_norm) || w_norm <= 0) break
      w <- w / w_norm
      tt <- Xf %*% w
      tt_ss <- drop(crossprod(tt))
      if (!is.finite(tt_ss) || tt_ss <= 0) break
      pp <- crossprod(Xf, tt) / tt_ss
      w_orth <- pp - w %*% crossprod(w, pp) / drop(crossprod(w))
      wo_norm <- sqrt(sum(w_orth * w_orth))
      if (!is.finite(wo_norm) || wo_norm <= 0) break
      w_orth <- w_orth / wo_norm
      t_orth <- Xf %*% w_orth
      to_ss <- drop(crossprod(t_orth))
      if (!is.finite(to_ss) || to_ss <= 0) break
      p_orth <- crossprod(Xf, t_orth) / to_ss
      Xf <- Xf - t_orth %*% t(p_orth)
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

.opls_apply_filter_r <- function(X, mX, vX, W_orth, P_orth) {
  Xf <- .fastpls_preprocess_test(X, mX, vX)
  if (ncol(W_orth) > 0L) {
    for (a in seq_len(ncol(W_orth))) {
      t_orth <- Xf %*% W_orth[, a, drop = FALSE]
      Xf <- Xf - t_orth %*% t(P_orth[, a, drop = FALSE])
    }
  }
  Xf
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
  K <- if (identical(kernel_engine, "R")) {
    .kernel_matrix_r(prep$X, prep$X, kernel, gamma, degree, coef0)
  } else {
    kernel_matrix_cpp(prep$X, prep$X, kernel_id, gamma, as.integer(degree), coef0)
  }
  kc <- if (identical(kernel_engine, "R")) .center_kernel_train_r(K) else center_kernel_train_cpp(K)
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
#' @param kernel Kernel type: `"linear"`, `"rbf"`, or `"poly"`.
#' @param gamma Kernel scale. Defaults to `1 / ncol(Xtrain)`.
#' @param degree Polynomial kernel degree.
#' @param coef0 Polynomial kernel offset.
#' @param ... Additional arguments passed to the inner PLS fit.
#' @return A `fastPLSKernel` object.
#' @keywords internal
kernel_pls_r <- function(Xtrain,
                         Ytrain,
                         Xtest = NULL,
                         Ytest = NULL,
                         ncomp = 2,
                         scaling = c("centering", "autoscaling", "none"),
                         kernel = c("linear", "rbf", "poly"),
                         gamma = NULL,
                         degree = 3L,
                         coef0 = 1,
                         method = c("simpls", "plssvd"),
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
                         fast_block = 1L,
                         gaussian_y = FALSE,
                         gaussian_y_dim = NULL,
                          gaussian_y_seed = seed,
	                  classifier = c("argmax", "lda_cpp", "lda_cuda", "class_bias_cpp", "class_bias_cuda"),
	                  lda_ridge = 1e-8,
	                  regression_head = c("standard", "linear_cpp", "linear_cuda"),
	                  fit = FALSE,
                           return_variance = TRUE,
                           proj = FALSE) {
  stop("The pure-R kernel PLS backend has been removed; use backend='cpp' or backend='cuda'.", call. = FALSE)
}

#' @rdname kernel_pls_r
#' @keywords internal
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
                           method = c("simpls", "plssvd"),
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
                           fast_block = 1L,
                           gaussian_y = FALSE,
                           gaussian_y_dim = NULL,
	                  gaussian_y_seed = seed,
	                  classifier = c("argmax", "lda_cpp", "lda_cuda", "class_bias_cpp", "class_bias_cuda"),
	                  lda_ridge = 1e-8,
	                  class_bias_method = c("iter_count_ratio", "count_ratio"),
	                  class_bias_lambda = 0.05,
	                  class_bias_iter = 1L,
	                  class_bias_clip = Inf,
	                  class_bias_eps = 1,
	                  class_bias_calibration_fraction = 1,
	                  class_bias_seed = seed,
	                  regression_head = c("standard", "linear_cpp", "linear_cuda"),
	                  fit = FALSE,
                           return_variance = TRUE,
                           proj = FALSE) {
  method <- match.arg(method)
  classifier <- .normalize_classifier(classifier)
  class_bias_method <- .normalize_class_bias_method(class_bias_method)
  svd.method <- match.arg(.normalize_svd_method(svd.method), c("irlba", "cpu_rsvd"))
  if (isTRUE(gaussian_y) && is.null(gaussian_y_dim)) {
    gaussian_y_dim <- .gaussian_y_default_dim(Xtrain, NULL)
  }
  .kernel_pls_fit(
    Xtrain, Ytrain, Xtest, Ytest, ncomp, match.arg(scaling), match.arg(kernel),
    gamma, degree, coef0, fit, proj, "cpp", pls,
    list(
      method = method,
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
      fast_block = fast_block,
      gaussian_y = gaussian_y,
      gaussian_y_dim = gaussian_y_dim,
      gaussian_y_seed = gaussian_y_seed,
      classifier = classifier,
      lda_ridge = lda_ridge,
      class_bias_method = class_bias_method,
      class_bias_lambda = class_bias_lambda,
      class_bias_iter = class_bias_iter,
      class_bias_clip = class_bias_clip,
      class_bias_eps = class_bias_eps,
      class_bias_calibration_fraction = class_bias_calibration_fraction,
      class_bias_seed = class_bias_seed,
      return_variance = return_variance
    )
  )
}

#' @rdname kernel_pls_r
#' @keywords internal
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
                            method = c("simpls", "plssvd"),
                            rsvd_oversample = 10L,
                            rsvd_power = 1L,
                            svds_tol = 0,
                            seed = 1L,
                            fast_block = 1L,
                            gaussian_y = FALSE,
                            gaussian_y_dim = NULL,
	                  gaussian_y_seed = seed,
	                  classifier = c("argmax", "lda_cpp", "lda_cuda", "class_bias_cpp", "class_bias_cuda"),
	                  lda_ridge = 1e-8,
	                  regression_head = c("standard", "linear_cpp", "linear_cuda"),
	                  fit = FALSE,
                            return_variance = TRUE,
	  proj = FALSE,
                            ...) {
  method <- match.arg(method)
  classifier <- .normalize_classifier(classifier)
  fit_fun <- if (identical(method, "plssvd")) plssvd_gpu else simpls_gpu
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
        gaussian_y_seed = gaussian_y_seed,
        classifier = classifier,
        lda_ridge = lda_ridge,
        return_variance = return_variance
      ),
      if (identical(method, "simpls")) list(fast_block = fast_block) else list(),
      list(...)
    )
  )
}

.fastpls_call_fixed_method <- function(fun, method, ...) {
  args <- list(...)
  args$method <- NULL
  do.call(fun, c(args, list(method = method)))
}

#' @rdname kernel_pls_r
#' @keywords internal
kernel_pls_fast_r <- function(...) {
  .fastpls_call_fixed_method(kernel_pls_r, "simpls", ...)
}

#' @rdname kernel_pls_r
#' @keywords internal
kernel_pls_fast_cpp <- function(...) {
  .fastpls_call_fixed_method(kernel_pls_cpp, "simpls", ...)
}

#' @rdname kernel_pls_r
#' @keywords internal
kernel_pls_fast_cuda <- function(...) {
  .fastpls_call_fixed_method(kernel_pls_cuda, "simpls", ...)
}

#' @rdname predict.fastPLS
#' @export
predict.fastPLSKernel <- function(object, newdata, Ytest = NULL, proj = FALSE, ...) {
  if (!is(object, "fastPLSKernel")) {
    stop("object is not a fastPLSKernel object", call. = FALSE)
  }
  Xnew <- .fastpls_preprocess_test(newdata, object$mX, object$vX)
  if (identical(object$kernel_engine, "R")) {
    Ktest <- .kernel_matrix_r(Xnew, object$Xref, object$kernel, object$gamma, object$degree, object$coef0)
    Ktest <- .center_kernel_test_r(Ktest, object$kernel_center$col_means, object$kernel_center$grand_mean)
  } else {
    Ktest <- kernel_matrix_cpp(Xnew, object$Xref, object$kernel_id, object$gamma, object$degree, object$coef0)
    Ktest <- center_kernel_test_cpp(Ktest, object$kernel_center$col_means, object$kernel_center$grand_mean)
  }
  predict.fastPLS(object$inner_model, Ktest, Ytest = Ytest, proj = proj)
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
  filt <- if (identical(filter_engine, "R")) {
    .opls_filter_r(Xtrain, Yfilter, north, scaling)
  } else {
    opls_filter_cpp(as.matrix(Xtrain), Yfilter, as.integer(north), pmatch(scaling, c("centering", "autoscaling", "none"))[1])
  }
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
#' Removes supervised orthogonal variation from `Xtrain`, then fits the requested
#' PLS core. The CUDA variant uses the GPU PLS core after CPU-side OPLS filtering.
#'
#' @inheritParams pls
#' @param north Number of orthogonal components to remove before PLS fitting.
#' @param ... Additional arguments passed to the inner PLS fit.
#' @return A `fastPLSOpls` object.
#' @keywords internal
opls_r <- function(Xtrain,
                   Ytrain,
                   Xtest = NULL,
                   Ytest = NULL,
                   ncomp = 2,
                   north = 1L,
                   scaling = c("centering", "autoscaling", "none"),
                   method = c("simpls", "plssvd"),
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
                   fast_block = 1L,
                   gaussian_y = FALSE,
                   gaussian_y_dim = NULL,
                   gaussian_y_seed = seed,
                   classifier = c("argmax", "lda_cpp", "lda_cuda", "class_bias_cpp", "class_bias_cuda"),
                   lda_ridge = 1e-8,
                   fit = FALSE,
                   return_variance = TRUE,
                   proj = FALSE) {
  stop("The pure-R OPLS backend has been removed; use backend='cpp' or backend='cuda'.", call. = FALSE)
}

#' @rdname opls_r
#' @keywords internal
opls_cpp <- function(Xtrain,
                     Ytrain,
                     Xtest = NULL,
                     Ytest = NULL,
                     ncomp = 2,
                     north = 1L,
                     scaling = c("centering", "autoscaling", "none"),
                     method = c("simpls", "plssvd"),
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
                     fast_block = 1L,
                     gaussian_y = FALSE,
                     gaussian_y_dim = NULL,
                     gaussian_y_seed = seed,
                     classifier = c("argmax", "lda_cpp", "lda_cuda", "class_bias_cpp", "class_bias_cuda"),
                     lda_ridge = 1e-8,
                     class_bias_method = c("iter_count_ratio", "count_ratio"),
                     class_bias_lambda = 0.05,
                     class_bias_iter = 1L,
                     class_bias_clip = Inf,
                     class_bias_eps = 1,
                     class_bias_calibration_fraction = 1,
                     class_bias_seed = seed,
                     fit = FALSE,
                     return_variance = TRUE,
                     proj = FALSE) {
  method <- match.arg(method)
  classifier <- .normalize_classifier(classifier)
  class_bias_method <- .normalize_class_bias_method(class_bias_method)
  svd.method <- match.arg(.normalize_svd_method(svd.method), c("irlba", "cpu_rsvd"))
  .opls_fit(
    Xtrain, Ytrain, Xtest, Ytest, ncomp, match.arg(scaling), north, fit, proj,
    "cpp", pls,
    list(
      method = method,
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
      fast_block = fast_block,
      gaussian_y = gaussian_y,
      gaussian_y_dim = gaussian_y_dim,
      gaussian_y_seed = gaussian_y_seed,
      classifier = classifier,
      lda_ridge = lda_ridge,
      class_bias_method = class_bias_method,
      class_bias_lambda = class_bias_lambda,
      class_bias_iter = class_bias_iter,
      class_bias_clip = class_bias_clip,
      class_bias_eps = class_bias_eps,
      class_bias_calibration_fraction = class_bias_calibration_fraction,
      class_bias_seed = class_bias_seed,
      return_variance = return_variance
    )
  )
}

#' @rdname opls_r
#' @keywords internal
opls_cuda <- function(Xtrain,
                      Ytrain,
                      Xtest = NULL,
                      Ytest = NULL,
                      ncomp = 2,
                      north = 1L,
                      scaling = c("centering", "autoscaling", "none"),
                      method = c("plssvd", "simpls"),
                      rsvd_oversample = 10L,
                      rsvd_power = 1L,
                      svds_tol = 0,
                      seed = 1L,
                      fast_block = 1L,
                      gaussian_y = FALSE,
                      gaussian_y_dim = NULL,
                      gaussian_y_seed = seed,
	                  classifier = c("argmax", "lda_cpp", "lda_cuda", "class_bias_cpp", "class_bias_cuda"),
	                  lda_ridge = 1e-8,
	                  regression_head = c("standard", "linear_cpp", "linear_cuda"),
	                  fit = FALSE,
                      return_variance = TRUE,
	  proj = FALSE,
                      ...) {
  method <- match.arg(method)
  classifier <- .normalize_classifier(classifier)
  fit_fun <- if (identical(method, "plssvd")) plssvd_gpu else simpls_gpu
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
        gaussian_y_seed = gaussian_y_seed,
        classifier = classifier,
        lda_ridge = lda_ridge,
        return_variance = return_variance
      ),
      if (identical(method, "simpls")) list(fast_block = fast_block) else list(),
      list(...)
    )
  )
}

#' @rdname opls_r
#' @keywords internal
opls_fast_r <- function(...) {
  .fastpls_call_fixed_method(opls_r, "simpls", ...)
}

#' @rdname opls_r
#' @keywords internal
opls_fast_cpp <- function(...) {
  .fastpls_call_fixed_method(opls_cpp, "simpls", ...)
}

#' @rdname opls_r
#' @keywords internal
opls_fast_cuda <- function(...) {
  .fastpls_call_fixed_method(opls_cuda, "simpls", ...)
}

#' @rdname predict.fastPLS
#' @export
predict.fastPLSOpls <- function(object, newdata, Ytest = NULL, proj = FALSE, ...) {
  if (!is(object, "fastPLSOpls")) {
    stop("object is not a fastPLSOpls object", call. = FALSE)
  }
  Xnew <- if (identical(object$opls_engine, "R")) {
    .opls_apply_filter_r(newdata, object$mX, object$vX, object$W_orth, object$P_orth)
  } else {
    opls_apply_filter_cpp(as.matrix(newdata), object$mX, object$vX, object$W_orth, object$P_orth)
  }
  predict.fastPLS(object$inner_model, Xnew, Ytest = Ytest, proj = proj)
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
#' @param scaling One of `"centering"`, `"autoscaling"`, `"none"`.
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
#' @param fast_block Refresh block size for the fastPLS `simpls` core.
#' @param gaussian_y Logical; when `TRUE`, fit to a Gaussian random response
#'   sketch and decode predictions back to the original response scale or class
#'   labels. The default is `FALSE`.
#' @param gaussian_y_dim Number of Gaussian response dimensions. When `NULL`,
#'   the default is `min(ncol(Xtrain), 100)`.
#' @param gaussian_y_seed Random seed used to generate the Gaussian response
#'   sketch.
#' @param fast_center_t Deprecated and ignored.
#' @param fast_reorth_v Deprecated and ignored.
#' @param fast_incremental Deprecated and ignored.
#' @param fast_inc_iters Deprecated and ignored.
#' @param fast_defl_cache Deprecated and ignored.
#' @return A `fastPLS` object.
#' @keywords internal
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
                      gpu_qr = FALSE,
                      gpu_eig = FALSE,
                      gpu_qless_qr = TRUE,
                      gpu_finalize_threshold = 32L,
                      fast_block = 1L,
                      fast_center_t = FALSE,
                      fast_reorth_v = FALSE,
                      fast_incremental = TRUE,
                      fast_inc_iters = 2L,
                      fast_defl_cache = TRUE,
                      gaussian_y = FALSE,
                      gaussian_y_dim = NULL,
	                      gaussian_y_seed = seed,
	                      classifier = c("argmax", "lda_cpp", "lda_cuda", "class_bias_cpp", "class_bias_cuda"),
	                      lda_ridge = 1e-8,
	                      class_bias_method = c("iter_count_ratio", "count_ratio"),
	                      class_bias_lambda = 0.05,
	                      class_bias_iter = 1L,
	                      class_bias_clip = Inf,
	                      class_bias_eps = 1,
	                      class_bias_calibration_fraction = 1,
	                      class_bias_seed = seed,
	                      regression_head = c("standard", "linear_cpp", "linear_cuda"),
                          return_variance = TRUE) {
  if (!has_cuda()) {
    stop("simpls_gpu requires a CUDA-enabled fastPLS build")
  }
	  on.exit(try(cuda_reset_workspace(), silent = TRUE), add = TRUE)
	  classifier <- .normalize_classifier(classifier)
	  regression_head <- .normalize_regression_head(regression_head)
	  class_bias_method <- .normalize_class_bias_method(class_bias_method)

	  scal <- pmatch(scaling, c("centering", "autoscaling", "none"))[1]
	  Xtrain <- as.matrix(Xtrain)
	  if (is.factor(Ytrain) &&
	      !isTRUE(gaussian_y) &&
	      !isTRUE(fit) &&
	      classifier %in% c("argmax", "class_bias_cpp", "class_bias_cuda") &&
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
	      lda_ridge,
	      class_bias_method = class_bias_method,
	      class_bias_lambda = class_bias_lambda,
	      class_bias_iter = class_bias_iter,
	      class_bias_clip = class_bias_clip,
	      class_bias_eps = class_bias_eps,
	      class_bias_calibration_fraction = class_bias_calibration_fraction,
	      class_bias_seed = class_bias_seed
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
    gaussian_y_seed = gaussian_y_seed,
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
      seed = seed,
      fast_block = fast_block,
      fast_center_t = fast_center_t,
      fast_reorth_v = fast_reorth_v,
      fast_incremental = fast_incremental,
      fast_inc_iters = fast_inc_iters,
      fast_defl_cache = fast_defl_cache
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
	    lda_ridge,
	    class_bias_method = class_bias_method,
	    class_bias_lambda = class_bias_lambda,
	    class_bias_iter = class_bias_iter,
	    class_bias_clip = class_bias_clip,
	    class_bias_eps = class_bias_eps,
	    class_bias_calibration_fraction = class_bias_calibration_fraction,
	    class_bias_seed = class_bias_seed
	  )
	  model <- .attach_linear_regression_head(model, Xtrain, Ytrain_original, regression_head)
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
#' @param scaling One of `"centering"`, `"autoscaling"`, `"none"`.
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
#' @param gaussian_y_seed Random seed used to generate the Gaussian response
#'   sketch.
#' @return A `fastPLS` object fitted with GPU PLSSVD.
#' @keywords internal
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
	                      gaussian_y_seed = seed,
	                      classifier = c("argmax", "lda_cpp", "lda_cuda", "class_bias_cpp", "class_bias_cuda"),
	                      lda_ridge = 1e-8,
	                      class_bias_method = c("iter_count_ratio", "count_ratio"),
	                      class_bias_lambda = 0.05,
	                      class_bias_iter = 1L,
	                      class_bias_clip = Inf,
	                      class_bias_eps = 1,
	                      class_bias_calibration_fraction = 1,
	                      class_bias_seed = seed,
	                      regression_head = c("standard", "linear_cpp", "linear_cuda"),
                          return_variance = TRUE) {
  if (!has_cuda()) {
    stop("plssvd_gpu requires a CUDA-enabled fastPLS build")
  }
	  on.exit(try(cuda_reset_workspace(), silent = TRUE), add = TRUE)
	  classifier <- .normalize_classifier(classifier)
	  regression_head <- .normalize_regression_head(regression_head)
	  class_bias_method <- .normalize_class_bias_method(class_bias_method)

  scal <- pmatch(scaling, c("centering", "autoscaling", "none"))[1]
  Xtrain <- as.matrix(Xtrain)
  Ytrain_original <- Ytrain
  yprep <- .prepare_gaussian_y(
    Ytrain,
    Xtrain,
    gaussian_y = gaussian_y,
    gaussian_y_dim = gaussian_y_dim,
    gaussian_y_seed = gaussian_y_seed,
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
	    lda_ridge,
	    class_bias_method = class_bias_method,
	    class_bias_lambda = class_bias_lambda,
	    class_bias_iter = class_bias_iter,
	    class_bias_clip = class_bias_clip,
	    class_bias_eps = class_bias_eps,
	    class_bias_calibration_fraction = class_bias_calibration_fraction,
	    class_bias_seed = class_bias_seed
	  )
	  model <- .attach_linear_regression_head(model, Xtrain, Ytrain_original, regression_head)
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
      predict.backend = "cuda_flash"
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
#' @rdname plssvd_gpu
#' @keywords internal
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
#' @rdname simpls_gpu
#' @keywords internal
simpls_flash_gpu <- function(Xtrain, Ytrain, Xtest = NULL, Ytest = NULL,
                             ncomp = 2, scaling = c("centering", "autoscaling", "none"),
                             rsvd_oversample = 10L, rsvd_power = 1L,
                             svds_tol = 0, seed = 1L, fit = FALSE,
                             proj = FALSE, gpu_device_state = TRUE,
                             gpu_qr = FALSE, gpu_eig = FALSE,
                             gpu_qless_qr = TRUE, gpu_finalize_threshold = 32L,
                             fast_block = 1L, fast_center_t = FALSE,
                             fast_reorth_v = FALSE, fast_incremental = TRUE,
                             fast_inc_iters = 2L, fast_defl_cache = TRUE) {
  model <- simpls_gpu(
    Xtrain = Xtrain, Ytrain = Ytrain, Xtest = NULL, Ytest = NULL,
    ncomp = ncomp, scaling = scaling, rsvd_oversample = rsvd_oversample,
    rsvd_power = rsvd_power, svds_tol = svds_tol, seed = seed,
    fit = fit, proj = FALSE, gpu_device_state = gpu_device_state,
    gpu_qr = gpu_qr, gpu_eig = gpu_eig, gpu_qless_qr = gpu_qless_qr,
    gpu_finalize_threshold = gpu_finalize_threshold, fast_block = fast_block,
    fast_center_t = fast_center_t, fast_reorth_v = fast_reorth_v,
    fast_incremental = fast_incremental, fast_inc_iters = fast_inc_iters,
    fast_defl_cache = fast_defl_cache
  )
  .predict_flash_attach(model, Xtest, Ytest, proj)
}

#' GPU OPLS with FlashSVD-style low-rank CUDA prediction
#' @rdname opls_r
#' @keywords internal
opls_flash_gpu <- function(Xtrain, Ytrain, Xtest = NULL, Ytest = NULL,
                           ncomp = 2, north = 1L,
                           scaling = c("centering", "autoscaling", "none"),
                           method = c("simpls", "plssvd"),
                           rsvd_oversample = 10L, rsvd_power = 1L,
                           svds_tol = 0, seed = 1L, fit = FALSE,
                           proj = FALSE, fast_block = 1L, ...) {
  model <- opls_cuda(
    Xtrain = Xtrain, Ytrain = Ytrain, Xtest = NULL, Ytest = NULL,
    ncomp = ncomp, north = north, scaling = scaling,
    method = match.arg(method), rsvd_oversample = rsvd_oversample,
    rsvd_power = rsvd_power, svds_tol = svds_tol, seed = seed,
    fit = fit, proj = FALSE, fast_block = fast_block, ...
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
#' @rdname kernel_pls_r
#' @keywords internal
kernel_pls_flash_gpu <- function(Xtrain, Ytrain, Xtest = NULL, Ytest = NULL,
                                 ncomp = 2,
                                 scaling = c("centering", "autoscaling", "none"),
                                 kernel = c("linear", "rbf", "poly"),
                                 gamma = NULL, degree = 3L, coef0 = 1,
                                 method = "simpls",
                                 rsvd_oversample = 10L, rsvd_power = 1L,
                                 svds_tol = 0, seed = 1L, fast_block = 1L,
                                 fit = FALSE, proj = FALSE, ...) {
  model <- kernel_pls_cuda(
    Xtrain = Xtrain, Ytrain = Ytrain, Xtest = NULL, Ytest = NULL,
    ncomp = ncomp, scaling = scaling, kernel = kernel, gamma = gamma,
    degree = degree, coef0 = coef0, method = method,
    rsvd_oversample = rsvd_oversample, rsvd_power = rsvd_power,
    svds_tol = svds_tol, seed = seed, fast_block = fast_block,
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
                             backend = c("cpp", "cuda"),
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
                             gpu_qr = TRUE,
                             gpu_eig = TRUE,
                             gpu_qless_qr = FALSE,
                             gpu_finalize_threshold = 32L) {
  method <- match.arg(method)
  backend <- match.arg(backend)
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

  if (!identical(backend, "cuda")) {
    .guard_removed_hybrid_cuda(svd.method, "PLS CV")
    svd.method <- .normalize_svd_method(match.arg(svd.method))
    svdmeth <- .svd_method_id(svd.method)
  } else {
    svdmeth <- .svd_method_id("cuda_rsvd")
  }
  if (is.null(xprod)) {
    xprod <- if (identical(backend, "cuda")) {
      .should_use_xprod_default(ncol(Xdata), q_backend, ncomp)
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
  backend_id <- if (identical(backend, "cuda")) 1L else 0L

  run_cv <- function() {
    if (!is.null(seed)) set.seed(as.integer(seed))
    pls_cv_predict_compiled(
      Xdata = Xdata,
      Ydata = Ymat,
      constrain = constrain,
      ncomp = ncomp,
      scaling = scal,
      kfold = as.integer(kfold),
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
      class_codes = class_codes
    )
  }

  if (method %in% c("simpls", "opls", "kernelpls")) {
    profile <- .resolve_simpls_fast_profile(
      fast_block = 1L,
      fast_center_t = FALSE,
      fast_reorth_v = FALSE,
      fast_incremental = TRUE,
      fast_inc_iters = 2L,
      fast_defl_cache = TRUE,
      missing_fast_block = TRUE,
      missing_fast_center_t = TRUE,
      missing_fast_reorth_v = TRUE,
      missing_fast_incremental = TRUE,
      missing_fast_inc_iters = TRUE,
      missing_fast_defl_cache = TRUE,
      context = "simpls_fast"
    )
    run_cv_profiled <- function() {
      .with_fastpls_fast_options(
        run_cv(),
        fast_block = profile$fast_block,
        fast_center_t = profile$fast_center_t,
        fast_reorth_v = profile$fast_reorth_v,
        fast_incremental = profile$fast_incremental,
        fast_inc_iters = profile$fast_inc_iters,
        fast_defl_cache = profile$fast_defl_cache
      )
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
  } else {
    res <- .with_irlba_options(
      run_cv_profiled(),
      irlba_work = irlba_work,
      irlba_maxit = irlba_maxit,
      irlba_tol = irlba_tol,
      irlba_eps = irlba_eps,
      irlba_svtol = irlba_svtol
    )
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
  res
}

.make_single_cv_folds <- function(Ydata, constrain, kfold, seed) {
  n <- length(Ydata)
  if (is.null(constrain)) constrain <- seq_len(n)
  constrain <- as.integer(as.factor(constrain))
  groups <- sort(unique(constrain))
  group_fold <- integer(length(groups))
  names(group_fold) <- as.character(groups)
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

.pls_single_cv_r <- function(Xdata,
                             Ydata,
                             constrain,
                             ncomp,
                             kfold,
                             scaling,
                             method,
                             svd.method,
                             rsvd_oversample,
                             rsvd_power,
                             svds_tol,
                             seed,
                             inner.method,
                             north,
                             kernel,
                             gamma,
                             degree,
                             coef0,
                             gaussian_y,
                             gaussian_y_dim,
                             gaussian_y_seed) {
  stop("The pure-R PLS CV backend has been removed; use backend='cpp' or backend='cuda'.", call. = FALSE)
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
#' @param constrain Optional grouping vector; samples with the same value stay
#'   in the same fold.
#' @param kfold Number of folds.
#' @param return_scores Store score predictions for classification when `TRUE`.
#' @param irlba_work IRLBA work subspace size; `0` lets the backend choose.
#' @param irlba_maxit Maximum IRLBA iterations.
#' @param irlba_tol IRLBA convergence tolerance.
#' @param irlba_eps IRLBA orthogonality threshold.
#' @param irlba_svtol IRLBA singular-value convergence tolerance.
#' @return A list with fold assignments, predictions, metrics, status, and
#'   backend metadata.
#' @export
pls.single.cv <- function(Xdata,
                          Ydata,
                          constrain = NULL,
                          ncomp = 2L,
                          kfold = 10L,
                          scaling = c("centering", "autoscaling", "none"),
                          method = c("simpls", "plssvd", "opls", "kernelpls"),
                          backend = c("cpp", "cuda"),
                          inner.method = c("simpls", "plssvd"),
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
                          north = 1L,
                          kernel = c("linear", "rbf", "poly"),
                          gamma = NULL,
                          degree = 3L,
                          coef0 = 1,
                          gaussian_y = FALSE,
                          gaussian_y_dim = NULL,
                          gaussian_y_seed = seed,
                          return_scores = FALSE,
                          xprod = NULL,
                          ...) {
  method <- match.arg(method)
  backend <- match.arg(backend)
  inner.method <- match.arg(inner.method)
  scaling <- match.arg(scaling)
  svd.method <- match.arg(.normalize_svd_method(svd.method), c("irlba", "cpu_rsvd"))
  kernel <- match.arg(kernel)
  if (!identical(kernel, "linear")) {
    stop("Nonlinear kernel CV is not available after removal of the pure-R PLS backend; use kernel='linear'.", call. = FALSE)
  }
  .pls_cv_compiled(
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
    return_scores = return_scores,
    ...
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
#' @keywords internal
plssvd_cv_cpp <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                          scaling = c("centering", "autoscaling", "none"),
                          svd.method = c("cpu_rsvd", "irlba"), xprod = NULL, ...) {
  .pls_cv_compiled(Xdata, Ydata, constrain, ncomp, kfold, scaling, "plssvd", "cpp", svd.method, xprod = xprod, ...)
}

#' @rdname plssvd_cv_cpp
#' @keywords internal
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

#' @rdname plssvd_cv_cpp
#' @keywords internal
opls_cv_cpp <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                        north = 1L,
                        scaling = c("centering", "autoscaling", "none"),
                        svd.method = c("cpu_rsvd", "irlba"), xprod = NULL, ...) {
  pred_ncomp <- pmax(1L, as.integer(ncomp) - as.integer(north))
  .pls_cv_compiled(Xdata, Ydata, constrain, pred_ncomp, kfold, scaling, "opls", "cpp", svd.method, xprod = xprod, north = north, ...)
}

#' @rdname plssvd_cv_cpp
#' @keywords internal
kernelpls_cv_cpp <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                             scaling = c("centering", "autoscaling", "none"),
                             svd.method = c("cpu_rsvd", "irlba"), xprod = NULL, ...) {
  .pls_cv_compiled(Xdata, Ydata, constrain, ncomp, kfold, scaling, "kernelpls", "cpp", svd.method, xprod = xprod, ...)
}

kernel_pls_cv_cpp <- kernelpls_cv_cpp

#' @rdname plssvd_cv_cpp
#' @keywords internal
plssvd_cv_cuda <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                           scaling = c("centering", "autoscaling", "none"),
                           xprod = NULL, ...) {
  .pls_cv_compiled(Xdata, Ydata, constrain, ncomp, kfold, scaling, "plssvd", "cuda", xprod = xprod, ...)
}

#' @rdname plssvd_cv_cpp
#' @keywords internal
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

#' @rdname plssvd_cv_cpp
#' @keywords internal
opls_cv_cuda <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                         north = 1L,
                         scaling = c("centering", "autoscaling", "none"),
                         xprod = NULL, ...) {
  pred_ncomp <- pmax(1L, as.integer(ncomp) - as.integer(north))
  .pls_cv_compiled(Xdata, Ydata, constrain, pred_ncomp, kfold, scaling, "opls", "cuda", xprod = xprod, north = north, ...)
}

#' @rdname plssvd_cv_cpp
#' @keywords internal
kernelpls_cv_cuda <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                              scaling = c("centering", "autoscaling", "none"),
                              xprod = NULL, ...) {
  .pls_cv_compiled(Xdata, Ydata, constrain, ncomp, kfold, scaling, "kernelpls", "cuda", xprod = xprod, ...)
}

kernel_pls_cv_cuda <- kernelpls_cv_cuda

.svd_methods_internal <- c("exact", "irlba", "cpu_rsvd", "cuda_rsvd")
.svd_methods_public <- c("irlba", "cpu_rsvd")
.svd_methods_cpu <- c("irlba", "cpu_rsvd")

.svd_method_id <- function(method) {
  method <- .normalize_svd_method(method)
  method <- match.arg(method, .svd_methods_internal)
  switch(
    method,
    exact = 3L,
    irlba = 1L,
    cpu_rsvd = 4L,
    cuda_rsvd = 5L
  )
}

#' List available SVD backends
#'
#' Reports backend labels accepted by high-level APIs and whether each backend
#' is currently available.
#'
#' @return Data frame with columns `method` and `enabled`.
#' @keywords internal
svd_methods <- function() {
  methods <- .svd_methods_public
  enabled <- rep(TRUE, length(methods))
  names(enabled) <- methods
  data.frame(
    method = methods,
    enabled = as.logical(enabled),
    stringsAsFactors = FALSE
  )
}

#' Run a single truncated SVD through fastPLS dispatch
#'
#' Utility wrapper around the internal truncated SVD dispatcher used by fastPLS.
#'
#' @param A Numeric matrix.
#' @param k Requested truncated rank.
#' @param method Backend label.
#' @param rsvd_oversample RSVD oversampling.
#' @param rsvd_power RSVD power iterations.
#' @param svds_tol Reserved backend tolerance placeholder.
#' @param seed RSVD seed.
#' @param left_only Return left singular vectors only.
#' @return List with `U`, `s`, `Vt`, `method`, and elapsed time.
#' @keywords internal
svd_run <- function(A,
                    k,
                    method = c("cpu_rsvd", "irlba", "cuda_rsvd"),
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

#' Benchmark truncated SVD backends
#'
#' Repeats truncated SVD calls for selected backend labels.
#'
#' @param A Numeric matrix.
#' @param k Requested truncated rank.
#' @param methods Backend labels to test.
#' @param reps Repetitions per method.
#' @param rsvd_oversample RSVD oversampling.
#' @param rsvd_power RSVD power iterations.
#' @param svds_tol Reserved backend tolerance placeholder.
#' @param seed RSVD seed.
#' @param left_only Return left singular vectors only.
#' @return Data frame with `method`, `rep`, `elapsed`, and `status`.
#' @keywords internal
svd_benchmark <- function(A,
                          k,
                          methods = c("irlba", "cpu_rsvd"),
                          reps = 3L,
                          rsvd_oversample = 10L,
                          rsvd_power = 1L,
                          svds_tol = 0,
                          seed = 1L,
                          left_only = FALSE) {
  A <- as.matrix(A)
  reps <- as.integer(reps)
  if (reps < 1) {
    stop("reps must be >= 1")
  }

  methods <- unique(methods)
  out <- vector("list", length(methods))
  names(out) <- methods

  for (i in seq_along(methods)) {
    method <- methods[i]
    svdmeth <- .svd_method_id(method)
    if (is.na(svdmeth)) {
      stop(paste0("Unknown method: ", method))
    }

    elapsed <- numeric(reps)
    status <- rep("ok", reps)
    for (r in seq_len(reps)) {
      tr <- try(
        system.time(
          truncated_svd_debug(
            A = A,
            k = as.integer(k),
            svd_method = as.integer(svdmeth),
            rsvd_oversample = as.integer(rsvd_oversample),
            rsvd_power = as.integer(rsvd_power),
            svds_tol = as.numeric(svds_tol),
            seed = as.integer(seed + r - 1L),
            left_only = isTRUE(left_only)
          )
        )["elapsed"],
        silent = TRUE
      )
      if (inherits(tr, "try-error")) {
        elapsed[r] <- NA_real_
        status[r] <- "error"
      } else {
        elapsed[r] <- as.numeric(tr)
      }
    }
    out[[i]] <- data.frame(
      method = rep(method, reps),
      rep = seq_len(reps),
      elapsed = elapsed,
      status = status,
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, out)
}

#' Singular value decomposition through fastPLS backends
#'
#' User-facing SVD wrapper for the bundled IRLBA-style and randomized SVD
#' implementations; CUDA is available when fastPLS was built with CUDA support.
#'
#' @param x Numeric matrix.
#' @param nu Number of left singular vectors to return.
#' @param nv Number of right singular vectors to return.
#' @param ncomp Optional truncated rank. When supplied, it overrides `nu`/`nv`
#'   for the decomposition rank.
#' @param method One of `"irlba"`, `"cpu_rsvd"`, or `"cuda_rsvd"`.
#' @param rsvd_oversample Randomized SVD oversampling.
#' @param rsvd_power Randomized SVD power iterations.
#' @param svds_tol Iterative backend tolerance.
#' @param seed Random seed used by randomized backends.
#' @return A list compatible with `base::svd()` containing `d`, `u`, and `v`,
#'   plus backend metadata.
#' @export
fastsvd <- function(x,
                    nu = NULL,
                    nv = NULL,
                    ncomp = NULL,
                    method = c("irlba", "cpu_rsvd", "cuda_rsvd"),
                    rsvd_oversample = 10L,
                    rsvd_power = 1L,
                    svds_tol = 0,
                    seed = 1L) {
  x <- as.matrix(x)
  n <- nrow(x)
  p <- ncol(x)
  if (is.null(nu)) nu <- min(n, p)
  if (is.null(nv)) nv <- min(n, p)
  method <- match.arg(.normalize_svd_method(method), c("irlba", "cpu_rsvd", "cuda_rsvd"))
  if (identical(method, "cuda_rsvd") && !has_cuda()) {
    stop("method='cuda_rsvd' requires a CUDA-enabled fastPLS build.", call. = FALSE)
  }
  if (is.null(ncomp)) {
    k <- max(as.integer(nu), as.integer(nv), 1L)
  } else {
    k <- as.integer(ncomp)[1L]
  }
  k <- max(1L, min(k, n, p))

  out <- svd_run(
    A = x,
    k = k,
    method = method,
    rsvd_oversample = rsvd_oversample,
    rsvd_power = rsvd_power,
    svds_tol = svds_tol,
    seed = seed,
    left_only = FALSE
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
#' @param scale. Logical; scale columns before SVD.
#' @param svd.method SVD backend used by [fastsvd()].
#' @param ... Additional arguments passed to [fastsvd()].
#' @return A `fastPLSPCA` object with scores, loadings, and per-component
#'   `variance_explained` plus cumulative variance explained.
#' @export
pca <- function(x,
                ncomp = 2L,
                center = TRUE,
                scale. = FALSE,
                svd.method = c("cpu_rsvd", "irlba", "cuda_rsvd"),
                ...) {
  x <- as.matrix(x)
  ncomp <- max(1L, min(as.integer(ncomp)[1L], nrow(x), ncol(x)))
  scaled <- scale(x, center = center, scale = scale.)
  x_center <- attr(scaled, "scaled:center")
  x_scale <- attr(scaled, "scaled:scale")
  if (is.null(x_center)) x_center <- rep(0, ncol(x))
  if (is.null(x_scale)) x_scale <- rep(1, ncol(x))
  x_scaled <- as.matrix(scaled)

  decomp <- fastsvd(
    x_scaled,
    nu = ncomp,
    nv = ncomp,
    ncomp = ncomp,
    method = match.arg(svd.method),
    ...
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
    svd.method = decomp$method,
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
#' @param score.set For PLS objects, plot `"train"` scores, `"test"` scores,
#'   or `"auto"` to use training scores when available.
#' @param ellipse Logical; draw confidence ellipses when `TRUE`.
#' @param ellipse.type `"confidence"` or `"hotelling"`.
#' @param conf Confidence level.
#' @param ... Additional arguments passed to `plot()`.
#' @return Invisibly returns the plotted score matrix.
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

.truncated_svd_r <- function(A,
                             k,
                             svd.method = c("irlba", "cpu_rsvd"),
                             rsvd_oversample = 10L,
                             rsvd_power = 1L,
                             svds_tol = 0,
                             irlba_work = 0L,
                             irlba_maxit = 1000L,
                             irlba_tol = 1e-5,
                             irlba_eps = 1e-9,
                             irlba_svtol = 1e-5,
                             seed = 1L) {
  svd.method <- .normalize_svd_method(svd.method)
  svd.method <- match.arg(svd.method, c("irlba", "cpu_rsvd"))
  A <- as.matrix(A)
  k <- as.integer(k)
  if (k < 1L) stop("k must be >= 1")
  max_rank <- min(nrow(A), ncol(A))
  k <- min(k, max_rank)

  full_svd <- function() {
    s <- base::svd(A, nu = k, nv = k)
    list(
      u = s$u[, seq_len(k), drop = FALSE],
      d = s$d[seq_len(k)],
      v = s$v[, seq_len(k), drop = FALSE]
    )
  }

  if (max_rank < 6L) {
    return(full_svd())
  }

  if (svd.method == "irlba") {
    work <- as.integer(irlba_work)
    if (!is.finite(work) || is.na(work) || work <= k) {
      work <- max(k + 7L, 8L)
    }
    work <- min(work, max_rank)
    out <- IRLB(
      A,
      nu = k,
      work = work,
      maxit = as.integer(irlba_maxit),
      tol = as.numeric(irlba_tol),
      eps = as.numeric(irlba_eps),
      svtol = as.numeric(irlba_svtol)
    )
    return(list(
      u = out$u[, seq_len(k), drop = FALSE],
      d = out$d[seq_len(k)],
      v = out$v[, seq_len(k), drop = FALSE]
    ))
  }

  if (svd.method != "cpu_rsvd") {
    return(full_svd())
  }

  l <- min(max_rank, k + max(0L, as.integer(rsvd_oversample)))
  if (l >= max_rank) {
    s <- base::svd(A, nu = k, nv = k)
    return(list(
      u = s$u[, seq_len(k), drop = FALSE],
      d = s$d[seq_len(k)],
      v = s$v[, seq_len(k), drop = FALSE]
    ))
  }

  set.seed(as.integer(seed))
  Omega <- matrix(rnorm(ncol(A) * l), nrow = ncol(A), ncol = l)
  Y <- A %*% Omega
  q <- max(0L, as.integer(rsvd_power))
  if (q > 0L) {
    for (i in seq_len(q)) {
      Z <- crossprod(A, Y)
      Qz <- qr.Q(qr(Z), complete = FALSE)
      Y <- A %*% Qz
    }
  }
  Q <- qr.Q(qr(Y), complete = FALSE)
  B <- crossprod(Q, A)
  s_small <- base::svd(B, nu = min(nrow(B), ncol(B)), nv = min(nrow(B), ncol(B)))
  U <- Q %*% s_small$u
  list(
    u = U[, seq_len(k), drop = FALSE],
    d = s_small$d[seq_len(k)],
    v = s_small$v[, seq_len(k), drop = FALSE]
  )
}

.project_deflated_left_r <- function(M, V = NULL) {
  if (!is.null(V) && ncol(V) > 0L) {
    M <- M - V %*% crossprod(V, M)
  }
  M
}

.truncated_rsvd_crossprod_r <- function(X,
                                        Y,
                                        k,
                                        rsvd_oversample = 10L,
                                        rsvd_power = 1L,
                                        seed = 1L) {
  p <- ncol(X)
  m <- ncol(Y)
  max_rank <- min(p, m)
  k <- min(max(as.integer(k), 1L), max_rank)
  l <- min(max_rank, k + max(0L, as.integer(rsvd_oversample)))

  if (max_rank < 6L || l >= max_rank) {
    S <- crossprod(X, Y)
    s <- base::svd(S, nu = k, nv = k)
    return(list(
      u = s$u[, seq_len(k), drop = FALSE],
      d = s$d[seq_len(k)],
      v = s$v[, seq_len(k), drop = FALSE]
    ))
  }

  a_times <- function(M) crossprod(X, Y %*% M)
  at_times <- function(M) crossprod(Y, X %*% M)

  set.seed(as.integer(seed))
  Ysample <- a_times(matrix(rnorm(m * l), nrow = m, ncol = l))
  q <- max(0L, as.integer(rsvd_power))
  if (q == 1L) {
    Ysample <- a_times(at_times(Ysample))
  } else if (q > 1L) {
    for (i in seq_len(q)) {
      Z <- at_times(Ysample)
      Qz <- qr.Q(qr(Z), complete = FALSE)
      Ysample <- a_times(Qz)
    }
  }

  Q <- qr.Q(qr(Ysample), complete = FALSE)
  B <- crossprod(X %*% Q, Y)
  s_small <- base::svd(B, nu = min(nrow(B), ncol(B)), nv = min(nrow(B), ncol(B)))
  U <- Q %*% s_small$u
  list(
    u = U[, seq_len(k), drop = FALSE],
    d = s_small$d[seq_len(k)],
    v = s_small$v[, seq_len(k), drop = FALSE]
  )
}

.refresh_deflated_crossprod_left_r <- function(X,
                                               Y,
                                               V,
                                               k_block,
                                               rsvd_power = 1L,
                                               seed = 1L,
                                               warm_start = NULL) {
  p <- ncol(X)
  m <- ncol(Y)
  k_block <- min(max(as.integer(k_block), 1L), min(p, m))

  a_times <- function(M) .project_deflated_left_r(crossprod(X, Y %*% M), V)
  at_times <- function(M) crossprod(Y, X %*% .project_deflated_left_r(M, V))

  set.seed(as.integer(seed))
  Ysample <- matrix(rnorm(p * k_block), nrow = p, ncol = k_block)
  if (!is.null(warm_start) && length(warm_start) == p) {
    Ysample[, 1L] <- warm_start
  }
  Ysample <- .project_deflated_left_r(Ysample, V)

  q <- max(0L, as.integer(rsvd_power))
  if (q > 0L) {
    for (i in seq_len(q)) {
      Ysample <- a_times(at_times(Ysample))
    }
  }

  Q <- qr.Q(qr(Ysample), complete = FALSE)
  Q <- .project_deflated_left_r(Q, V)
  Q <- qr.Q(qr(Q), complete = FALSE)
  if (!ncol(Q)) {
    return(matrix(0, nrow = p, ncol = 0L))
  }

  B <- crossprod(X %*% Q, Y)
  s_small <- base::svd(B, nu = min(nrow(B), ncol(B)), nv = 0L)
  U <- Q %*% s_small$u
  U[, seq_len(min(k_block, ncol(U))), drop = FALSE]
}

.pls_model1_r_xprod <- function(Xtrain,
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
  Ytrain <- sweep(Ytrain, 2, mY[1, ], "-")

  s <- .truncated_rsvd_crossprod_r(
    Xtrain, Ytrain, max_ncomp_eff,
    rsvd_oversample = rsvd_oversample,
    rsvd_power = rsvd_power,
    seed = seed
  )

  max_ncomp_eff <- min(max_ncomp_eff, ncol(s$u), ncol(s$v))
  R <- s$u[, seq_len(max_ncomp_eff), drop = FALSE]
  Q <- s$v[, seq_len(max_ncomp_eff), drop = FALSE]
  Ttrain <- Xtrain %*% R
  G_full <- crossprod(Ttrain)

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
    D_mc <- diag(s$d[seq_len(mc)], nrow = mc, ncol = mc)
    coeff_latent <- solve(G_mc, D_mc)
    C_i <- C_latent[, , i, drop = FALSE][, , 1]
    C_i[seq_len(mc), seq_len(mc)] <- coeff_latent
    C_latent[, , i] <- C_i
    W_i <- coeff_latent %*% t(Q_mc)
    W_latent[seq_len(mc), , i] <- W_i
    if (store_B) {
      B[, , i] <- R_mc %*% W_i
    }
    if (fit) {
      yf <- Ttrain[, seq_len(mc), drop = FALSE] %*% W_i
      R2Y[i] <- RQ(Ytrain, yf)
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
    xprod_precision = "implicit64"
  )
  if (store_B) {
    out$B <- B
  }
  out <- .annotate_coefficient_storage(out, store_B)
  class(out) <- "fastPLS"
  out
}

.pls_model2_fast_r_xprod <- function(Xtrain,
                                     Ytrain,
                                     ncomp,
                                     scaling,
                                     fit,
                                     rsvd_power,
                                     seed,
                                     fast_block = 1L) {
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

  RR <- matrix(0, nrow = p, ncol = max_ncomp)
  QQ <- matrix(0, nrow = m, ncol = max_ncomp)
  VV <- matrix(0, nrow = p, ncol = max_ncomp)
  store_B <- .should_store_coefficients(p, m, length_ncomp, TRUE)
  B <- if (store_B) array(0, dim = c(p, m, length_ncomp)) else NULL
  Yfit <- if (fit) array(0, dim = c(n, m, length_ncomp)) else NULL
  R2Y <- rep(NA_real_, length_ncomp)
  Yfit_cur <- if (fit) matrix(0, nrow = n, ncol = m) else NULL
  Bcur <- if (store_B) matrix(0, nrow = p, ncol = m) else NULL

  i_out <- 1L
  a <- 1L
  refresh_block <- max(1L, as.integer(fast_block))
  rr_prev <- NULL

  while (a <= max_ncomp) {
    k_block <- min(refresh_block, max_ncomp - a + 1L)
    Vprev <- if (a > 1L) VV[, seq_len(a - 1L), drop = FALSE] else NULL
    Ublock <- .refresh_deflated_crossprod_left_r(
      Xtrain, Y, Vprev, k_block,
      rsvd_power = rsvd_power,
      seed = seed + a - 1L,
      warm_start = rr_prev
    )
    if (!ncol(Ublock)) break

    use_cols <- min(ncol(Ublock), k_block)
    stop_now <- FALSE
    for (j in seq_len(use_cols)) {
      rr <- Ublock[, j, drop = FALSE]
      tt <- Xtrain %*% rr
      tnorm <- sqrt(sum(tt * tt))
      if (!is.finite(tnorm) || tnorm <= 0) {
        stop_now <- TRUE
        break
      }
      tt <- tt / tnorm
      rr <- rr / tnorm
      pp <- crossprod(Xtrain, tt)
      qq <- crossprod(Y, tt)
      vv <- pp
      if (a > 1L) {
        Vprev <- VV[, seq_len(a - 1L), drop = FALSE]
        vv <- vv - Vprev %*% crossprod(Vprev, pp)
      }
      vnorm <- sqrt(sum(vv * vv))
      if (!is.finite(vnorm) || vnorm <= 0) {
        stop_now <- TRUE
        break
      }
      vv <- vv / vnorm

      RR[, a] <- rr[, 1]
      QQ[, a] <- qq[, 1]
      VV[, a] <- vv[, 1]
      if (store_B) {
        Bcur <- Bcur + rr %*% t(qq)
      }
      rr_prev <- rr[, 1]

      while (i_out <= length_ncomp && a == ncomp[i_out]) {
        if (store_B) {
          B[, , i_out] <- Bcur
        }
        if (fit) {
          Yfit_cur <- Yfit_cur + tt %*% t(qq)
          R2Y[i_out] <- RQ(Y, Yfit_cur)
          Yfit[, , i_out] <- sweep(Yfit_cur, 2, mY[1, ], "+")
        }
        i_out <- i_out + 1L
      }

      a <- a + 1L
      if (a > max_ncomp) break
    }
    if (stop_now) break
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
    xprod_precision = "implicit64"
  )
  if (store_B) {
    out$B <- B
  }
  out <- .annotate_coefficient_storage(out, store_B)
  class(out) <- "fastPLS"
  out
}

.pls_model1_r <- function(Xtrain,
                          Ytrain,
                          ncomp,
                          scaling,
                          fit,
                          svd.method,
                          rsvd_oversample,
                          rsvd_power,
                          svds_tol,
                          irlba_work,
                          irlba_maxit,
                          irlba_tol,
                          irlba_eps,
                          irlba_svtol,
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
  Ytrain <- sweep(Ytrain, 2, mY[1, ], "-")

  S <- crossprod(Xtrain, Ytrain)
  s <- .truncated_svd_r(
    S,
    max_ncomp_eff,
    svd.method,
    rsvd_oversample,
    rsvd_power,
    svds_tol,
    irlba_work,
    irlba_maxit,
    irlba_tol,
    irlba_eps,
    irlba_svtol,
    seed
  )
  max_ncomp_eff <- min(max_ncomp_eff, ncol(s$u), ncol(s$v))
  R <- s$u[, seq_len(max_ncomp_eff), drop = FALSE]
  Q <- s$v[, seq_len(max_ncomp_eff), drop = FALSE]
  Ttrain <- Xtrain %*% R

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
    T_mc <- Ttrain[, seq_len(mc), drop = FALSE]
    G_mc <- crossprod(T_mc)
    D_mc <- diag(s$d[seq_len(mc)], nrow = mc, ncol = mc)
    coeff_latent <- solve(G_mc, D_mc)
    C_i <- C_latent[, , i, drop = FALSE][, , 1]
    C_i[seq_len(mc), seq_len(mc)] <- coeff_latent
    C_latent[, , i] <- C_i
    W_i <- coeff_latent %*% t(Q_mc)
    W_latent[seq_len(mc), , i] <- W_i
    if (store_B) {
      B[, , i] <- R_mc %*% W_i
    }
    if (fit) {
      yf <- T_mc %*% W_i
      R2Y[i] <- RQ(Ytrain, yf)
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
    R2Y = R2Y
  )
  if (store_B) {
    out$B <- B
  }
  out <- .annotate_coefficient_storage(out, store_B)
  class(out) <- "fastPLS"
  out
}

.pls_model2_r <- function(Xtrain,
                          Ytrain,
                          ncomp,
                          scaling,
                          fit,
                          svd.method,
                          rsvd_oversample,
                          rsvd_power,
                          svds_tol,
                          irlba_work,
                          irlba_maxit,
                          irlba_tol,
                          irlba_eps,
                          irlba_svtol,
                          seed) {
  n <- nrow(Xtrain); p <- ncol(Xtrain); m <- ncol(Ytrain)
  ncomp <- as.integer(ncomp)
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

  X <- Xtrain
  mY <- matrix(colMeans(Ytrain), nrow = 1)
  Y <- sweep(Ytrain, 2, mY[1, ], "-")
  S <- crossprod(X, Y)

  RR <- matrix(0, nrow = p, ncol = max_ncomp)
  PP <- matrix(0, nrow = p, ncol = max_ncomp)
  QQ <- matrix(0, nrow = m, ncol = max_ncomp)
  TT <- matrix(0, nrow = n, ncol = max_ncomp)
  VV <- matrix(0, nrow = p, ncol = max_ncomp)
  store_B <- .should_store_coefficients(p, m, length_ncomp, TRUE)
  B <- if (store_B) array(0, dim = c(p, m, length_ncomp)) else NULL
  Yfit <- if (fit) array(0, dim = c(n, m, length_ncomp)) else NULL
  R2Y <- rep(NA_real_, length_ncomp)

  i_out <- 1L
  for (a in seq_len(max_ncomp)) {
    rr <- .truncated_svd_r(
      S, 1L, svd.method, rsvd_oversample, rsvd_power, svds_tol,
      irlba_work, irlba_maxit, irlba_tol, irlba_eps, irlba_svtol,
      seed + a - 1L
    )$u[, 1, drop = FALSE]
    tt <- X %*% rr
    tt <- tt - mean(tt)
    tnorm <- sqrt(sum(tt * tt))
    tt <- tt / tnorm
    rr <- rr / tnorm
    pp <- crossprod(X, tt)
    qq <- crossprod(Y, tt)
    vv <- pp
    if (a > 1L) {
      VV_prev <- VV[, seq_len(a - 1L), drop = FALSE]
      vv <- vv - VV_prev %*% crossprod(VV_prev, pp)
    }
    vv <- vv / sqrt(sum(vv * vv))
    S <- S - vv %*% crossprod(vv, S)

    RR[, a] <- rr[, 1]
    TT[, a] <- tt[, 1]
    PP[, a] <- pp[, 1]
    QQ[, a] <- qq[, 1]
    VV[, a] <- vv[, 1]

    if (a == ncomp[i_out]) {
      RR_a <- RR[, seq_len(a), drop = FALSE]
      QQ_a <- QQ[, seq_len(a), drop = FALSE]
      if (store_B) {
        B[, , i_out] <- RR_a %*% t(QQ_a)
      }
      if (fit) {
        yf <- TT[, seq_len(a), drop = FALSE] %*% t(QQ_a)
        Yfit[, , i_out] <- sweep(yf, 2, mY[1, ], "+")
        R2Y[i_out] <- RQ(Ytrain, matrix(Yfit[, , i_out], nrow = n, ncol = m))
      }
      i_out <- i_out + 1L
      if (i_out > length_ncomp) break
    }
  }

  out <- list(
    P = PP,
    Q = QQ,
    Ttrain = TT,
    R = RR,
    mX = mX,
    vX = vX,
    mY = mY,
    p = p,
    m = m,
    ncomp = ncomp,
    Yfit = Yfit,
    R2Y = R2Y
  )
  if (store_B) {
    out$B <- B
  }
  out <- .annotate_coefficient_storage(out, store_B)
  class(out) <- "fastPLS"
  out
}

.pls_model2_fast_r <- function(Xtrain,
                               Ytrain,
                               ncomp,
                               scaling,
                               fit,
                               svd.method,
                               rsvd_oversample,
                               rsvd_power,
                               svds_tol,
                               irlba_work,
                               irlba_maxit,
                               irlba_tol,
                               irlba_eps,
                               irlba_svtol,
                               seed,
                               fast_block = 1L) {
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

  X <- Xtrain
  Xt <- t(Xtrain)
  mY <- matrix(colMeans(Ytrain), nrow = 1)
  Y <- sweep(Ytrain, 2, mY[1, ], "-")
  S <- crossprod(X, Y)

  RR <- matrix(0, nrow = p, ncol = max_ncomp)
  QQ <- matrix(0, nrow = m, ncol = max_ncomp)
  VV <- matrix(0, nrow = p, ncol = max_ncomp)
  store_B <- .should_store_coefficients(p, m, length_ncomp, TRUE)
  B <- if (store_B) array(0, dim = c(p, m, length_ncomp)) else NULL
  Yfit <- if (fit) array(0, dim = c(n, m, length_ncomp)) else NULL
  R2Y <- rep(NA_real_, length_ncomp)
  Yfit_cur <- if (fit) matrix(0, nrow = n, ncol = m) else NULL

  Bcur <- if (store_B) matrix(0, nrow = p, ncol = m) else NULL
  i_out <- 1L
  a <- 1L
  refresh_block <- max(1L, as.integer(fast_block))

  while (a <= max_ncomp) {
    k_block <- min(refresh_block, max_ncomp - a + 1L)
    Ublock <- .truncated_svd_r(
      S,
      k = k_block,
      svd.method = svd.method,
      rsvd_oversample = rsvd_oversample,
      rsvd_power = rsvd_power,
      svds_tol = svds_tol,
      irlba_work = irlba_work,
      irlba_maxit = irlba_maxit,
      irlba_tol = irlba_tol,
      irlba_eps = irlba_eps,
      irlba_svtol = irlba_svtol,
      seed = seed + a - 1L
    )$u
    if (!is.matrix(Ublock)) {
      Ublock <- matrix(Ublock, ncol = 1L)
    }
    if (!ncol(Ublock)) break

    use_cols <- min(ncol(Ublock), k_block)
    stop_now <- FALSE
    for (j in seq_len(use_cols)) {
      rr <- Ublock[, j, drop = FALSE]
      pp <- crossprod(X, X %*% rr)
      tnorm_sq <- drop(crossprod(rr, pp))
      if (!is.finite(tnorm_sq) || tnorm_sq <= 0) {
        stop_now <- TRUE
        break
      }
      tnorm <- sqrt(tnorm_sq)
      rr <- rr / tnorm
      pp <- pp / tnorm
      tt <- X %*% rr
      qq <- crossprod(Y, tt)

      vv <- pp
      if (a > 1L) {
        Vprev <- VV[, seq_len(a - 1L), drop = FALSE]
        vv <- vv - Vprev %*% crossprod(Vprev, pp)
      }
      vnorm <- sqrt(sum(vv * vv))
      if (!is.finite(vnorm) || vnorm <= 0) {
        stop_now <- TRUE
        break
      }
      vv <- vv / vnorm
      S <- S - vv %*% crossprod(vv, S)

      RR[, a] <- rr[, 1]
      QQ[, a] <- qq[, 1]
      VV[, a] <- vv[, 1]
      if (store_B) {
        Bcur <- Bcur + rr %*% t(qq)
      }

      while (i_out <= length_ncomp && a == ncomp[i_out]) {
        if (store_B) {
          B[, , i_out] <- Bcur
        }
        if (fit) {
          Yfit_cur <- Yfit_cur + tt %*% t(qq)
          Yfit[, , i_out] <- sweep(Yfit_cur, 2, mY[1, ], "+")
          R2Y[i_out] <- RQ(Ytrain, matrix(Yfit[, , i_out], nrow = n, ncol = m))
        }
        i_out <- i_out + 1L
      }

      a <- a + 1L
      if (a > max_ncomp) break
    }
    if (stop_now) break
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
    R2Y = R2Y
  )
  if (store_B) {
    out$B <- B
  }
  out <- .annotate_coefficient_storage(out, store_B)
  class(out) <- "fastPLS"
  out
}

#' Removed pure-R PLS reference implementation
#'
#' The pure-R PLS implementation was removed from the public package. Use
#' `pls(..., backend = "cpp")` for CPU IRLBA/rSVD or `backend = "cuda"` for
#' CUDA rSVD.
#'
#' @inheritParams pls
#' @param method One of `"simpls"` or `"plssvd"`. `simpls` uses the fastPLS SIMPLS core.
#' @param svd.method One of `"irlba"` or `"cpu_rsvd"`.
#' @return A `fastPLS` object.
#' @keywords internal
pls_r <- function(...) {
  stop("The pure-R PLS backend has been removed; use backend='cpp' or backend='cuda'.", call. = FALSE)
}



#' Partial Least Squares with selectable model family and backend
#'
#' `pls()` is the main fastPLS user entry point. It routes PLSSVD, SIMPLS, OPLS,
#' and kernel PLS through the selected compiled CPU or CUDA backend while
#' keeping low-level implementation functions internal.
#'
#' @param Xtrain Numeric training predictor matrix.
#' @param Ytrain Training response (numeric or factor).
#' @param Xtest Optional test predictor matrix.
#' @param Ytest Optional test response for `Q2Y`.
#' @param ncomp Number of components (scalar or vector).
#' @param scaling One of `"centering"`, `"autoscaling"`, `"none"`.
#' @param method One of `"simpls"`, `"plssvd"`, `"opls"`, or `"kernelpls"`.
#'   `simpls` uses the fastPLS accelerated SIMPLS core.
#' @param svd.method One of `"irlba"` or `"cpu_rsvd"`.
#'   The former hybrid CUDA route via `svd.method = "cuda_rsvd"` has been removed
#'   from `pls()`; use `backend = "cuda"` for GPU-native fits.
#' @param rsvd_oversample RSVD oversampling.
#' @param rsvd_power RSVD power iterations.
#' @param svds_tol Reserved backend tolerance placeholder.
#' @param irlba_work IRLBA work subspace size; `0` lets the backend choose.
#' @param irlba_maxit Maximum IRLBA iterations.
#' @param irlba_tol IRLBA convergence tolerance.
#' @param irlba_eps IRLBA orthogonality threshold.
#' @param irlba_svtol IRLBA singular-value convergence tolerance.
#' @param seed RSVD seed.
#' @param fast_block Refresh block size for the fastPLS `simpls` core; use `1L` for the
#'   most accuracy-stable per-component refresh.
#' @param gaussian_y Logical; when `TRUE`, fit PLS to a Gaussian random
#'   low-dimensional response sketch and decode predictions back to the original
#'   response scale or class labels. The default is `FALSE`.
#' @param gaussian_y_dim Number of Gaussian response dimensions. When `NULL`,
#'   the default is `min(ncol(Xtrain), 100)`.
#' @param gaussian_y_seed Random seed used to generate the Gaussian response
#'   sketch.
#' @param classifier Classification decision rule. `"argmax"` keeps the
#'   standard PLS-DA response-score argmax. `"lda_cpp"` fits an LDA classifier
#'   on the PLS latent scores with compiled C++ code. `"lda_cuda"` uses the
#'   same LDA model and CUDA matrix multiplication for latent projection and
#'   discriminant scoring when CUDA is available. For high-throughput PLS-DA on
#'   GPU, the recommended route is `method = "plssvd"`, `backend = "cuda"`,
#'   and `classifier = "lda_cuda"`; the compiled CPU fallback is
#'   `classifier = "lda_cpp"`. `"class_bias_cpp"` and `"class_bias_cuda"` keep
#'   the PLS response-score classifier but add calibrated per-class score
#'   offsets before ranking classes. The experimental fused CUDA PLS+LDA path
#'   can be enabled with `FASTPLS_FUSED_CUDA_LDA=1`, but the optimized standard
#'   CUDA LDA route remains the default.
#' @param lda_ridge Relative diagonal ridge added to the pooled LDA covariance.
#' @param class_bias_method Calibration rule for class-bias prediction.
#'   `"iter_count_ratio"` repeats the count-balancing update
#'   `class_bias_iter` times; `"count_ratio"` applies one
#'   `lambda * log(true_count / predicted_count)` update.
#' @param class_bias_lambda Calibration strength for
#'   `classifier = "class_bias_cpp"` or `"class_bias_cuda"`.
#' @param class_bias_iter Number of iterative class-bias calibration passes.
#' @param class_bias_clip Optional absolute clipping value for class-bias
#'   offsets; `Inf` disables clipping.
#' @param class_bias_eps Positive smoothing constant used when estimating
#'   class-bias offsets from prediction counts.
#' @param class_bias_calibration_fraction Stratified fraction of the training
#'   set used to estimate class-bias offsets. Use values below 1 to mimic a
#'   held-out calibration split while still fitting PLS on all training rows.
#' @param class_bias_seed Seed used for the stratified calibration split.
#' @param regression_head Regression prediction head. `"standard"` keeps the
#'   usual PLS coefficient-path prediction. `"linear_cpp"` fits an ordinary
#'   least-squares linear model on the latent scores with compiled C++ code.
#'   `"linear_cuda"` uses CUDA matrix multiplication for latent projection and
#'   prediction when CUDA is available.
#' @param fast_center_t Deprecated and ignored. `simpls` now permanently uses
#'   the former incdefl profile.
#' @param fast_reorth_v Deprecated and ignored. `simpls` now permanently uses
#'   the former incdefl profile.
#' @param fast_incremental Deprecated and ignored. `simpls` now permanently uses
#'   the former incdefl profile.
#' @param fast_inc_iters Deprecated and ignored. `simpls` now permanently uses
#'   the former incdefl profile.
#' @param fast_defl_cache Deprecated and ignored. `simpls` now permanently uses
#'   the former incdefl profile.
#' @param fit Return fitted values and `R2Y` when `TRUE`.
#' @param return_variance Compute predictor-space latent-variable variance
#'   explained. Set to `FALSE` for timing/memory benchmarks that do not need
#'   plotting variance metadata.
#' @param proj Return projected `Ttest` when `TRUE`.
#' @param perm.test Run permutation test.
#' @param times Number of permutations.
#' @param backend Implementation backend: `"cpp"` for compiled CPU or `"cuda"`
#'   for GPU-native fitting.
#' @param inner.method Inner PLS core used by `method = "opls"` or
#'   `method = "kernelpls"`.
#' @param north Number of orthogonal components removed by OPLS.
#' @param kernel Kernel type for kernel PLS: `"linear"`, `"rbf"`, or `"poly"`.
#' @param gamma Kernel scale. Defaults internally to `1 / ncol(Xtrain)`.
#' @param degree Polynomial kernel degree.
#' @param coef0 Polynomial kernel offset.
#' @param gpu_device_state Keep selected SIMPLS GPU workspaces resident when
#'   using `backend = "cuda"`.
#' @param gpu_qr Use GPU QR finalization when available.
#' @param gpu_eig Use GPU eigensolver finalization when available.
#' @param gpu_qless_qr Use the q-less GPU QR path when available.
#' @param gpu_finalize_threshold Component threshold controlling GPU-side
#'   finalization.
#' @return A `fastPLS` object. Fitted objects include `variance_explained`,
#'   `cumulative_variance_explained`, and matching `x_*` aliases containing the
#'   fraction of training predictor variance explained by each latent variable.
#' @export
pls =  function (Xtrain,
                 Ytrain,
                 Xtest = NULL,
                 Ytest = NULL,
                 ncomp=2,
                 scaling = c("centering", "autoscaling","none"),
                 method = c("simpls", "plssvd", "opls", "kernelpls"),
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
                 fast_block = 1L,
                 fast_center_t = FALSE,
                 fast_reorth_v = FALSE,
                 fast_incremental = TRUE,
                 fast_inc_iters = 2L,
                 fast_defl_cache = TRUE,
                 gaussian_y = FALSE,
                 gaussian_y_dim = NULL,
	                 gaussian_y_seed = seed,
	                 classifier = c("argmax", "lda_cpp", "lda_cuda", "class_bias_cpp", "class_bias_cuda"),
	                 lda_ridge = 1e-8,
	                 class_bias_method = c("iter_count_ratio", "count_ratio"),
	                 class_bias_lambda = 0.05,
	                 class_bias_iter = 1L,
	                 class_bias_clip = Inf,
	                 class_bias_eps = 1,
	                 class_bias_calibration_fraction = 1,
	                 class_bias_seed = seed,
	                 regression_head = c("standard", "linear_cpp", "linear_cuda"),
	                 fit = FALSE,
                 return_variance = TRUE,
                 proj = FALSE,
                 perm.test = FALSE,
                 times = 100,
                 backend = c("cpp", "cuda"),
                 inner.method = c("simpls", "plssvd"),
                 north = 1L,
                 kernel = c("linear", "rbf", "poly"),
                 gamma = NULL,
                 degree = 3L,
                 coef0 = 1,
                 gpu_device_state = TRUE,
                 gpu_qr = TRUE,
                 gpu_eig = TRUE,
                 gpu_qless_qr = FALSE,
                 gpu_finalize_threshold = 32L)
{

  scal = pmatch(scaling, c("centering", "autoscaling","none"))[1]
  requested_method <- match.arg(method, c("simpls", "plssvd", "opls", "kernelpls"))
	  backend <- match.arg(backend)
	  inner.method <- match.arg(inner.method)
	  classifier <- .normalize_classifier(classifier)
	  regression_head <- .normalize_regression_head(regression_head)
	  class_bias_method <- .normalize_class_bias_method(class_bias_method)
	  class_bias_lambda <- as.numeric(class_bias_lambda)[1L]
	  class_bias_iter <- max(1L, as.integer(class_bias_iter)[1L])
	  class_bias_clip <- as.numeric(class_bias_clip)[1L]
	  class_bias_eps <- as.numeric(class_bias_eps)[1L]
	  class_bias_calibration_fraction <- as.numeric(class_bias_calibration_fraction)[1L]
	  class_bias_seed <- as.integer(class_bias_seed)[1L]
	  if (!is.finite(class_bias_lambda) || class_bias_lambda < 0) {
	    stop("class_bias_lambda must be a finite non-negative number", call. = FALSE)
	  }
	  if (is.na(class_bias_iter) || class_bias_iter < 1L) {
	    stop("class_bias_iter must be a positive integer", call. = FALSE)
	  }
	  if (is.na(class_bias_clip) || class_bias_clip < 0) {
	    stop("class_bias_clip must be non-negative or Inf", call. = FALSE)
	  }
	  if (!is.finite(class_bias_eps) || class_bias_eps <= 0) {
	    stop("class_bias_eps must be a finite positive number", call. = FALSE)
	  }
	  if (!is.finite(class_bias_calibration_fraction) ||
	      class_bias_calibration_fraction <= 0 ||
	      class_bias_calibration_fraction > 1) {
	    stop("class_bias_calibration_fraction must be in (0, 1]", call. = FALSE)
	  }
	  if (is.na(class_bias_seed)) {
	    stop("class_bias_seed must be an integer seed", call. = FALSE)
	  }
	  old_class_bias_options <- options(
	    fastPLS.class_bias_method = class_bias_method,
	    fastPLS.class_bias_lambda = class_bias_lambda,
	    fastPLS.class_bias_iter = class_bias_iter,
	    fastPLS.class_bias_clip = class_bias_clip,
	    fastPLS.class_bias_eps = class_bias_eps,
	    fastPLS.class_bias_calibration_fraction = class_bias_calibration_fraction,
	    fastPLS.class_bias_seed = class_bias_seed
	  )
	  on.exit(options(old_class_bias_options), add = TRUE)

  if (identical(requested_method, "opls")) {
    fit_fun <- switch(backend, cpp = opls_cpp, cuda = opls_cuda)
    args <- list(
      Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest,
      ncomp = ncomp, north = north, scaling = scaling, method = inner.method,
      rsvd_oversample = rsvd_oversample, rsvd_power = rsvd_power,
      svds_tol = svds_tol, seed = seed, fast_block = fast_block,
      gaussian_y = gaussian_y, gaussian_y_dim = gaussian_y_dim,
      gaussian_y_seed = gaussian_y_seed, fit = fit, proj = proj,
      class_bias_method = class_bias_method,
      class_bias_lambda = class_bias_lambda,
      class_bias_iter = class_bias_iter,
      class_bias_clip = class_bias_clip,
      class_bias_eps = class_bias_eps,
      class_bias_calibration_fraction = class_bias_calibration_fraction,
      class_bias_seed = class_bias_seed
    )
    args <- c(args, list(classifier = classifier, lda_ridge = lda_ridge))
    args$return_variance <- return_variance
    if (identical(backend, "cuda")) {
      args <- c(args, list(
        gpu_qr = gpu_qr,
        gpu_eig = gpu_eig,
        gpu_qless_qr = gpu_qless_qr,
        gpu_finalize_threshold = gpu_finalize_threshold
      ))
    } else {
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
    fit_fun <- switch(backend, cpp = kernel_pls_cpp, cuda = kernel_pls_cuda)
    args <- list(
      Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest,
      ncomp = ncomp, scaling = scaling, kernel = kernel, gamma = gamma,
      degree = degree, coef0 = coef0, method = inner.method,
      rsvd_oversample = rsvd_oversample, rsvd_power = rsvd_power,
      svds_tol = svds_tol, seed = seed, fast_block = fast_block,
      gaussian_y = gaussian_y, gaussian_y_dim = gaussian_y_dim,
      gaussian_y_seed = gaussian_y_seed, fit = fit, proj = proj,
      class_bias_method = class_bias_method,
      class_bias_lambda = class_bias_lambda,
      class_bias_iter = class_bias_iter,
      class_bias_clip = class_bias_clip,
      class_bias_eps = class_bias_eps,
      class_bias_calibration_fraction = class_bias_calibration_fraction,
      class_bias_seed = class_bias_seed
    )
    args <- c(args, list(classifier = classifier, lda_ridge = lda_ridge))
    args$return_variance <- return_variance
    if (identical(backend, "cuda")) {
      args <- c(args, list(
        gpu_qr = gpu_qr,
        gpu_eig = gpu_eig,
        gpu_qless_qr = gpu_qless_qr,
        gpu_finalize_threshold = gpu_finalize_threshold
      ))
    } else {
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
        gpu_qr = gpu_qr,
        gpu_eig = gpu_eig,
        gpu_qless_qr = gpu_qless_qr,
        gpu_finalize_threshold = gpu_finalize_threshold,
        gaussian_y = gaussian_y,
        gaussian_y_dim = gaussian_y_dim,
        gaussian_y_seed = gaussian_y_seed,
	        classifier = classifier,
	        lda_ridge = lda_ridge,
	        class_bias_method = class_bias_method,
	        class_bias_lambda = class_bias_lambda,
	        class_bias_iter = class_bias_iter,
	        class_bias_clip = class_bias_clip,
	        class_bias_eps = class_bias_eps,
	        class_bias_calibration_fraction = class_bias_calibration_fraction,
	        class_bias_seed = class_bias_seed,
	        regression_head = regression_head,
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
      gpu_device_state = gpu_device_state,
      gpu_qr = gpu_qr,
      gpu_eig = gpu_eig,
      gpu_qless_qr = gpu_qless_qr,
      gpu_finalize_threshold = gpu_finalize_threshold,
      fast_block = fast_block,
      fast_center_t = fast_center_t,
      fast_reorth_v = fast_reorth_v,
      fast_incremental = fast_incremental,
      fast_inc_iters = fast_inc_iters,
      fast_defl_cache = fast_defl_cache,
      gaussian_y = gaussian_y,
      gaussian_y_dim = gaussian_y_dim,
      gaussian_y_seed = gaussian_y_seed,
	      classifier = classifier,
	      lda_ridge = lda_ridge,
	      class_bias_method = class_bias_method,
	      class_bias_lambda = class_bias_lambda,
	      class_bias_iter = class_bias_iter,
	      class_bias_clip = class_bias_clip,
	      class_bias_eps = class_bias_eps,
	      class_bias_calibration_fraction = class_bias_calibration_fraction,
	      class_bias_seed = class_bias_seed,
	      regression_head = regression_head,
          return_variance = return_variance
	    ))
  }

  meth = .normalize_pls_method(requested_method)
  .guard_removed_hybrid_cuda(svd.method, "pls()")
  svd.method <- .normalize_svd_method(svd.method)
  svd.method <- match.arg(svd.method)
  svdmeth <- .svd_method_id(svd.method)
  if (meth == 3L) {
    profile <- .resolve_simpls_fast_profile(
      fast_block = fast_block,
      fast_center_t = fast_center_t,
      fast_reorth_v = fast_reorth_v,
      fast_incremental = fast_incremental,
      fast_inc_iters = fast_inc_iters,
      fast_defl_cache = fast_defl_cache,
      missing_fast_block = missing(fast_block),
      missing_fast_center_t = missing(fast_center_t),
      missing_fast_reorth_v = missing(fast_reorth_v),
      missing_fast_incremental = missing(fast_incremental),
      missing_fast_inc_iters = missing(fast_inc_iters),
      missing_fast_defl_cache = missing(fast_defl_cache),
      context = "simpls_fast"
    )
    fast_block <- profile$fast_block
    fast_center_t <- profile$fast_center_t
    fast_reorth_v <- profile$fast_reorth_v
    fast_incremental <- profile$fast_incremental
    fast_inc_iters <- profile$fast_inc_iters
    fast_defl_cache <- profile$fast_defl_cache
  }

  Xtrain = as.matrix(Xtrain)
  Ytrain_original <- Ytrain
  yprep <- .prepare_gaussian_y(
    Ytrain,
    Xtrain,
    gaussian_y = gaussian_y,
    gaussian_y_dim = gaussian_y_dim,
    gaussian_y_seed = gaussian_y_seed,
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
    if (missing(rsvd_oversample)) rsvd_oversample <- tuned$rsvd_oversample
    if (missing(rsvd_power)) rsvd_power <- tuned$rsvd_power
  }

  use_xprod_default <- meth %in% c(1L, 3L) && (
    (identical(svd.method, "cpu_rsvd") &&
       .should_use_xprod_default(ncol(Xtrain), ncol(Ytrain), ncomp)) ||
      (identical(svd.method, "irlba") &&
         .should_use_xprod_irlba_default(nrow(Xtrain), ncol(Xtrain), ncol(Ytrain), ncomp))
  )
  xprod_precision_default <- if (identical(svd.method, "irlba")) "implicit_irlba" else "implicit64"
	  return_ttrain_for_head <- (!isTRUE(classification) && regression_head %in% c("linear_cpp", "linear_cuda"))

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
        fast_block=fast_block,
        fast_center_t=fast_center_t,
        fast_reorth_v=fast_reorth_v,
        fast_incremental=fast_incremental,
        fast_inc_iters=fast_inc_iters,
        fast_defl_cache=fast_defl_cache,
        return_ttrain=return_ttrain_for_head
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
        fast_block=fast_block,
        fast_center_t=fast_center_t,
        fast_reorth_v=fast_reorth_v,
        fast_incremental=fast_incremental,
        fast_inc_iters=fast_inc_iters,
        fast_defl_cache=fast_defl_cache,
        return_ttrain=return_ttrain_for_head
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
	    lda_ridge,
	    class_bias_method = class_bias_method,
	    class_bias_lambda = class_bias_lambda,
	    class_bias_iter = class_bias_iter,
	    class_bias_clip = class_bias_clip,
	    class_bias_eps = class_bias_eps,
	    class_bias_calibration_fraction = class_bias_calibration_fraction,
	    class_bias_seed = class_bias_seed
	  )
		  model <- .attach_linear_regression_head(model, Xtrain, Ytrain_original, regression_head)
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
              seed=seed,
              fast_block=fast_block,
              fast_center_t=fast_center_t,
              fast_reorth_v=fast_reorth_v,
              fast_incremental=fast_incremental,
              fast_inc_iters=fast_inc_iters,
              fast_defl_cache=fast_defl_cache
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
#' C++/CUDA CV core shared with [pls.single.cv()].
#'
#' @inheritParams pls
#' @param Xdata Predictor matrix.
#' @param Ydata Response (numeric or factor).
#' @param constrain Optional grouping vector for constrained splitting.
#' @param kfold Number of folds.
#' @param method One of `"simpls"`, `"plssvd"`, `"opls"`, or `"kernelpls"`.
#' @param backend Implementation backend: `"cpp"` or `"cuda"`.
#' @param fast_block Refresh block size for the fastPLS `simpls` core; use `1L` for the
#'   most accuracy-stable per-component refresh.
#' @param fast_center_t Deprecated and ignored. `simpls` now permanently uses
#'   the former incdefl profile.
#' @param fast_reorth_v Deprecated and ignored. `simpls` now permanently uses
#'   the former incdefl profile.
#' @param fast_incremental Deprecated and ignored. `simpls` now permanently uses
#'   the former incdefl profile.
#' @param fast_inc_iters Deprecated and ignored. `simpls` now permanently uses
#'   the former incdefl profile.
#' @param fast_defl_cache Deprecated and ignored. `simpls` now permanently uses
#'   the former incdefl profile.
#' @param return_scores Store score predictions for classification when `TRUE`.
#' @param xprod Use the matrix-free cross-product route where available.
#'   `NULL` applies fastPLS defaults.
#' @param ... Additional backend tuning arguments.
#' @return List with `optim_comp`, decoded `pred`, `metrics`, legacy
#'   `Q2Y`/`R2Y` metric vectors, `fold`, backend metadata, and `Ypred` when
#'   score predictions are stored.
#' @export
optim.pls.cv =  function (Xdata,
                          Ydata,
                          ncomp=2,
                          constrain=NULL,
                          scaling = c("centering", "autoscaling","none"),
                          method = c("simpls", "plssvd", "opls", "kernelpls"),
                          backend = c("cpp", "cuda"),
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
                          fast_block = 1L,
                          fast_center_t = FALSE,
                          fast_reorth_v = FALSE,
                          fast_incremental = TRUE,
                          fast_inc_iters = 2L,
                          fast_defl_cache = TRUE,
                          kfold=10,
                          north = 1L,
                          return_scores = FALSE,
                          xprod = NULL,
                          ...)
{
  if (sum(is.na(Xdata)) > 0) {
    stop("Missing values are present")
  }
  method <- match.arg(method)
  backend <- match.arg(backend)
  scaling <- match.arg(scaling)
  svd.method <- match.arg(.normalize_svd_method(svd.method), c("irlba", "cpu_rsvd"))
  Xdata <- as.matrix(Xdata)
  if (is.null(constrain)) constrain <- seq_len(nrow(Xdata))

  res <- .pls_cv_compiled(
    Xdata = Xdata,
    Ydata = Ydata,
    constrain = constrain,
    ncomp = as.integer(ncomp),
    kfold = as.integer(kfold),
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
    return_scores = return_scores,
    ...
  )
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
#' Inner component optimization uses the same compiled C++/CUDA CV core as
#' [pls.single.cv()] and [optim.pls.cv()].
#'
#' @inheritParams pls
#' @param Xdata Predictor matrix.
#' @param Ydata Response (numeric or factor).
#' @param constrain Grouping vector for constrained splitting.
#' @param runn Number of repeated runs.
#' @param kfold_inner Inner-fold count.
#' @param kfold_outer Outer-fold count.
#' @param method One of `"simpls"`, `"plssvd"`, `"opls"`, or `"kernelpls"`.
#' @param backend Implementation backend: `"cpp"` or `"cuda"`.
#' @param fast_block Refresh block size for the fastPLS `simpls` core; use `1L` for the
#'   most accuracy-stable per-component refresh.
#' @param fast_center_t Deprecated and ignored. `simpls` now permanently uses
#'   the former incdefl profile.
#' @param fast_reorth_v Deprecated and ignored. `simpls` now permanently uses
#'   the former incdefl profile.
#' @param fast_incremental Deprecated and ignored. `simpls` now permanently uses
#'   the former incdefl profile.
#' @param fast_inc_iters Deprecated and ignored. `simpls` now permanently uses
#'   the former incdefl profile.
#' @param fast_defl_cache Deprecated and ignored. `simpls` now permanently uses
#'   the former incdefl profile.
#' @param xprod Use the matrix-free cross-product route where available for
#'   inner component optimization. `NULL` applies fastPLS defaults.
#' @param ... Additional backend tuning arguments.
#' @return List of nested CV outputs and summaries.
#' @export
pls.double.cv = function(Xdata,
                         Ydata,
                         ncomp=2,
                         constrain=1:nrow(Xdata),
                         scaling = c("centering", "autoscaling","none"),
                         method = c("simpls", "plssvd", "opls", "kernelpls"),
                         backend = c("cpp", "cuda"),
                         inner.method = c("simpls", "plssvd"),
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
                         fast_block = 1L,
                         fast_center_t = FALSE,
                         fast_reorth_v = FALSE,
                         fast_incremental = TRUE,
                         fast_inc_iters = 2L,
                         fast_defl_cache = TRUE,
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
                         classifier = c("argmax", "lda_cpp", "lda_cuda", "class_bias_cpp", "class_bias_cuda"),
                         lda_ridge = 1e-8,
                         regression_head = c("standard", "linear_cpp", "linear_cuda"),
                         xprod = NULL,
                         ...){

  if(sum(is.na(Xdata))>0) {
    stop("Missing values are present")
  }
  method <- match.arg(method)
  backend <- match.arg(backend)
  inner.method <- match.arg(inner.method)
  scaling <- match.arg(scaling)
  svd.method <- match.arg(.normalize_svd_method(svd.method), c("irlba", "cpu_rsvd"))
  kernel <- match.arg(kernel)
  classifier <- .normalize_classifier(classifier)
  regression_head <- .normalize_regression_head(regression_head)
  if (!identical(kernel, "linear")) {
    stop("Nonlinear kernel double CV is not available after removal of the pure-R PLS backend; use kernel='linear'.", call. = FALSE)
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
      kfold = as.integer(kfold_outer),
      seed = as.integer(seed) + j - 1L
    )
    best_comp <- integer(as.integer(kfold_outer))
    inner_results <- vector("list", as.integer(kfold_outer))
    if (classification) {
      run_pred_chr <- rep(NA_character_, nrow(Xdata))
    } else {
      run_pred <- matrix(NA_real_, nrow = nrow(Xdata), ncol = ncol(Ydata))
    }

    for (f in seq_len(as.integer(kfold_outer))) {
      test_idx <- which(fold == (f - 1L))
      train_idx <- which(fold != (f - 1L))
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
        irlba_work = irlba_work,
        irlba_maxit = irlba_maxit,
        irlba_tol = irlba_tol,
        irlba_eps = irlba_eps,
        irlba_svtol = irlba_svtol,
        seed = as.integer(seed) + 1000L * j + f,
        kfold = kfold_inner,
        north = north,
        return_scores = FALSE,
        xprod = xprod,
        ...
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
        irlba_work = irlba_work,
        irlba_maxit = irlba_maxit,
        irlba_tol = irlba_tol,
        irlba_eps = irlba_eps,
        irlba_svtol = irlba_svtol,
        seed = as.integer(seed) + 2000L * j + f,
        fast_block = fast_block,
        fast_center_t = fast_center_t,
        fast_reorth_v = fast_reorth_v,
        fast_incremental = fast_incremental,
        fast_inc_iters = fast_inc_iters,
        fast_defl_cache = fast_defl_cache,
        fit = FALSE,
        proj = FALSE,
        backend = backend,
        inner.method = inner.method,
        north = north,
        kernel = kernel,
        gamma = gamma,
        degree = degree,
        coef0 = coef0,
        classifier = classifier,
        lda_ridge = lda_ridge,
        regression_head = regression_head
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
        inner.method = inner.method,
        svd.method = svd.method,
        rsvd_oversample = rsvd_oversample,
        rsvd_power = rsvd_power,
        svds_tol = svds_tol,
        irlba_work = irlba_work,
        irlba_maxit = irlba_maxit,
        irlba_tol = irlba_tol,
        irlba_eps = irlba_eps,
        irlba_svtol = irlba_svtol,
        seed = as.integer(seed) + 3000L + i,
        fast_block = fast_block,
        fast_center_t = fast_center_t,
        fast_reorth_v = fast_reorth_v,
        fast_incremental = fast_incremental,
        fast_inc_iters = fast_inc_iters,
        fast_defl_cache = fast_defl_cache,
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
        regression_head = regression_head,
        xprod = xprod,
        ...
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
