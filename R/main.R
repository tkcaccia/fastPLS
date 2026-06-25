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

.backend_control_env_defaults <- c(
  FASTPLS_STORE_B = "auto",
  FASTPLS_STORE_B_MAX_MB = "256",
  FASTPLS_PREDICT_LATENT_MIN_B_MB = "256",
  FASTPLS_COMPACT_CLASS_BLOCK_SIZE = "4096",
  FASTPLS_LABEL_AWARE_Y_THRESHOLD_MB = "512",
  FASTPLS_LABEL_AWARE_BLOCK_SIZE = "8192",
  FASTPLS_PLSSVD_SMALL_EXACT_MAX_RANK = "32",
  FASTPLS_PLSSVD_OPTIMIZED = "1",
  FASTPLS_LEADING_LEFT_MAX_ITERS = NA_character_,
  FASTPLS_FAST_CENTER_T = "0",
  FASTPLS_FAST_REORTH_V = "0",
  FASTPLS_FAST_INCREMENTAL = "1",
  FASTPLS_FAST_INC_ITERS = "2",
  FASTPLS_FAST_DEFLCACHE = "1",
  FASTPLS_FAST_OPTIMIZED = "1",
  FASTPLS_FAST_RSVD_TOP1 = "0",
  FASTPLS_FAST_RSVD_TOP1_OVERSAMPLE = "10",
  FASTPLS_FAST_RSVD_TOP1_POWER = "1",
  FASTPLS_FAST_CROSSPROD_MIN_NCOMP = "20",
  FASTPLS_FAST_CROSSPROD_MAX_P = "512",
  FASTPLS_FAST_CROSSPROD_MIN_N_TO_P_RATIO = "8",
  FASTPLS_FAST_ADAPTIVE_RSVD = "0",
  FASTPLS_FAST_ADAPTIVE_MIN_BLOCK = NA_character_,
  FASTPLS_FAST_ADAPTIVE_MAX_BLOCK = NA_character_,
  FASTPLS_FAST_ADAPTIVE_MIN_POWER = NA_character_,
  FASTPLS_FAST_ADAPTIVE_MAX_POWER = NA_character_,
  FASTPLS_FAST_ADAPTIVE_FLAT_RATIO = "0.55",
  FASTPLS_FAST_ADAPTIVE_STEEP_RATIO = "0.12",
  FASTPLS_RETURN_TTRAIN = "0",
  FASTPLS_IRLBA_WORK = "0",
  FASTPLS_IRLBA_MAXIT = "1000",
  FASTPLS_IRLBA_TOL = "1e-5",
  FASTPLS_IRLBA_EPS = "1e-9",
  FASTPLS_IRLBA_SVTOL = "1e-5",
  FASTPLS_GPU_DEVICE_STATE = "0",
  FASTPLS_GPU_QR = "1",
  FASTPLS_GPU_EIG = "1",
  FASTPLS_GPU_FINALIZE_THRESHOLD = "32",
  FASTPLS_GPU_SIMPLS_XPROD = "0",
  FASTPLS_CUDA_WORKSPACE_STREAMS = "0",
  FASTPLS_FAST_GPU_MIN_M = "512",
  FASTPLS_FAST_GPU_MIN_N = "16",
  FASTPLS_FAST_GPU_MIN_WORK = "200000",
  FASTPLS_CUDA_RSVD_RESIDENT_PUBLIC = "0",
  FASTPLS_CUDA_RSVD_RESIDENT_MIN_L = "48",
  FASTPLS_CUDA_RSVD_RESIDENT_MIN_WORK = "1000000",
  FASTPLS_METAL_MIN_FLOPS = "200000000",
  FASTPLS_METAL_EXACT_MAX_RANK = "256",
  FASTPLS_METAL_EXPERIMENTAL_ITERATIVE = "false",
  FASTPLS_METAL_RESIDENT_SIMPLS = "true",
  FASTPLS_RETURN_LDA_SCORES = "false",
  FASTPLS_FUSED_CUDA_LDA = "0"
)

.backend_control_env_groups <- c(
  FASTPLS_STORE_B = "storage",
  FASTPLS_STORE_B_MAX_MB = "storage",
  FASTPLS_PREDICT_LATENT_MIN_B_MB = "prediction",
  FASTPLS_COMPACT_CLASS_BLOCK_SIZE = "prediction",
  FASTPLS_LABEL_AWARE_Y_THRESHOLD_MB = "response",
  FASTPLS_LABEL_AWARE_BLOCK_SIZE = "response",
  FASTPLS_PLSSVD_SMALL_EXACT_MAX_RANK = "plssvd",
  FASTPLS_PLSSVD_OPTIMIZED = "plssvd",
  FASTPLS_LEADING_LEFT_MAX_ITERS = "plssvd",
  FASTPLS_FAST_CENTER_T = "simpls",
  FASTPLS_FAST_REORTH_V = "simpls",
  FASTPLS_FAST_INCREMENTAL = "simpls",
  FASTPLS_FAST_INC_ITERS = "simpls",
  FASTPLS_FAST_DEFLCACHE = "simpls",
  FASTPLS_FAST_OPTIMIZED = "simpls",
  FASTPLS_FAST_RSVD_TOP1 = "simpls",
  FASTPLS_FAST_RSVD_TOP1_OVERSAMPLE = "simpls",
  FASTPLS_FAST_RSVD_TOP1_POWER = "simpls",
  FASTPLS_FAST_CROSSPROD_MIN_NCOMP = "simpls",
  FASTPLS_FAST_CROSSPROD_MAX_P = "simpls",
  FASTPLS_FAST_CROSSPROD_MIN_N_TO_P_RATIO = "simpls",
  FASTPLS_FAST_ADAPTIVE_RSVD = "simpls",
  FASTPLS_FAST_ADAPTIVE_MIN_BLOCK = "simpls",
  FASTPLS_FAST_ADAPTIVE_MAX_BLOCK = "simpls",
  FASTPLS_FAST_ADAPTIVE_MIN_POWER = "simpls",
  FASTPLS_FAST_ADAPTIVE_MAX_POWER = "simpls",
  FASTPLS_FAST_ADAPTIVE_FLAT_RATIO = "simpls",
  FASTPLS_FAST_ADAPTIVE_STEEP_RATIO = "simpls",
  FASTPLS_RETURN_TTRAIN = "simpls",
  FASTPLS_IRLBA_WORK = "irlba",
  FASTPLS_IRLBA_MAXIT = "irlba",
  FASTPLS_IRLBA_TOL = "irlba",
  FASTPLS_IRLBA_EPS = "irlba",
  FASTPLS_IRLBA_SVTOL = "irlba",
  FASTPLS_GPU_DEVICE_STATE = "gpu",
  FASTPLS_GPU_QR = "gpu",
  FASTPLS_GPU_EIG = "gpu",
  FASTPLS_GPU_FINALIZE_THRESHOLD = "gpu",
  FASTPLS_GPU_SIMPLS_XPROD = "gpu",
  FASTPLS_CUDA_WORKSPACE_STREAMS = "cuda",
  FASTPLS_FAST_GPU_MIN_M = "cuda",
  FASTPLS_FAST_GPU_MIN_N = "cuda",
  FASTPLS_FAST_GPU_MIN_WORK = "cuda",
  FASTPLS_CUDA_RSVD_RESIDENT_PUBLIC = "cuda",
  FASTPLS_CUDA_RSVD_RESIDENT_MIN_L = "cuda",
  FASTPLS_CUDA_RSVD_RESIDENT_MIN_WORK = "cuda",
  FASTPLS_METAL_MIN_FLOPS = "metal",
  FASTPLS_METAL_EXACT_MAX_RANK = "metal",
  FASTPLS_METAL_EXPERIMENTAL_ITERATIVE = "metal",
  FASTPLS_METAL_RESIDENT_SIMPLS = "metal",
  FASTPLS_RETURN_LDA_SCORES = "classifier",
  FASTPLS_FUSED_CUDA_LDA = "classifier"
)

.backend_control_package_version <- function() {
  desc <- utils::packageDescription("fastPLS", fields = "Version")
  if (length(desc) == 1L && !is.na(desc)) {
    as.character(desc)
  } else {
    NA_character_
  }
}

.backend_control_snapshot <- function(context = NULL, overrides = NULL) {
  names_env <- names(.backend_control_env_defaults)
  raw <- Sys.getenv(names_env, unset = NA_character_)
  defaults <- .backend_control_env_defaults
  values <- raw
  missing <- is.na(values)
  values[missing] <- defaults[missing]
  groups <- unname(.backend_control_env_groups[names_env])
  groups[is.na(groups)] <- "other"
  env <- data.frame(
    name = names_env,
    group = groups,
    value = unname(values),
    default = unname(defaults),
    overridden = !is.na(raw),
    stringsAsFactors = FALSE
  )
  out <- list(
    context = context %||% NA_character_,
    timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
    fastPLS_version = .backend_control_package_version(),
    env = env,
    overrides = overrides %||% list()
  )
  class(out) <- "fastPLSBackendControl"
  out
}

.with_backend_env <- function(expr, values) {
  if (is.null(values) || length(values) == 0L) {
    return(force(expr))
  }
  if (is.null(names(values)) || any(!nzchar(names(values)))) {
    stop("backend-control environment values must be named", call. = FALSE)
  }
  values <- as.list(values)
  old <- Sys.getenv(names(values), unset = NA_character_)
  on.exit({
    for (nm in names(old)) {
      .restore_env_scalar(nm, old[[nm]])
    }
  }, add = TRUE)
  for (nm in names(values)) {
    .restore_env_scalar(nm, values[[nm]])
  }
  force(expr)
}

.attach_backend_control <- function(model, backend_control = NULL) {
  model
}

.with_fastpls_fast_options <- function(expr,
                                       return_ttrain = FALSE) {
  .with_backend_env(expr, c(
    FASTPLS_FAST_CENTER_T = "0",
    FASTPLS_FAST_REORTH_V = "0",
    FASTPLS_FAST_INCREMENTAL = "1",
    FASTPLS_FAST_INC_ITERS = "2",
    FASTPLS_FAST_DEFLCACHE = "1",
    FASTPLS_RETURN_TTRAIN = if (isTRUE(return_ttrain)) "1" else "0"
  ))
}

.with_irlba_options <- function(expr,
                                irlba_work = 0L,
                                irlba_maxit = 1000L,
                                irlba_tol = 1e-5,
                                irlba_eps = 1e-9,
                                irlba_svtol = 1e-5) {
  .with_backend_env(expr, c(
    FASTPLS_IRLBA_WORK = as.character(as.integer(irlba_work)),
    FASTPLS_IRLBA_MAXIT = as.character(as.integer(irlba_maxit)),
    FASTPLS_IRLBA_TOL = as.character(as.numeric(irlba_tol)),
    FASTPLS_IRLBA_EPS = as.character(as.numeric(irlba_eps)),
    FASTPLS_IRLBA_SVTOL = as.character(as.numeric(irlba_svtol))
  ))
}

.with_gpu_native_options <- function(expr,
                                     gpu_device_state = FALSE,
                                     gpu_qr = TRUE,
                                     gpu_eig = TRUE,
                                     gpu_finalize_threshold = 32L) {
  .with_backend_env(expr, c(
    FASTPLS_GPU_DEVICE_STATE = if (isTRUE(gpu_device_state)) "1" else "0",
    FASTPLS_GPU_QR = if (isTRUE(gpu_qr)) "1" else "0",
    FASTPLS_GPU_EIG = if (isTRUE(gpu_eig)) "1" else "0",
    FASTPLS_GPU_FINALIZE_THRESHOLD = as.character(as.integer(gpu_finalize_threshold))
  ))
}

.with_simpls_gpu_xprod <- function(expr) {
  .with_backend_env(expr, c(FASTPLS_GPU_SIMPLS_XPROD = "1"))
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

.maybe_attach_x_loadings <- function(model, Xtrain, return_loadings = FALSE) {
  empty_p <- matrix(numeric(0), 0L, 0L)
  if (is.null(model) || !is.list(model)) {
    return(model)
  }
  if (!isTRUE(return_loadings)) {
    model$P <- empty_p
    if (!is.null(model$inner_model) && is.list(model$inner_model)) {
      model$inner_model <- .maybe_attach_x_loadings(model$inner_model, Xtrain, FALSE)
    }
    return(model)
  }
  if (is.null(model$R) || length(model$R) == 0L || is.null(model$ncomp)) {
    if (!is.null(model$inner_model) && is.list(model$inner_model)) {
      model$inner_model <- .maybe_attach_x_loadings(model$inner_model, Xtrain, TRUE)
    }
    return(model)
  }
  R <- as.matrix(model$R)
  Xtrain <- as.matrix(Xtrain)
  if (nrow(R) != ncol(Xtrain)) {
    return(model)
  }
  k <- min(max(as.integer(model$ncomp), na.rm = TRUE), ncol(R))
  if (!is.finite(k) || is.na(k) || k < 1L) {
    return(model)
  }
  Xscaled <- .fastpls_scaled_by_model(model, Xtrain)
  scores <- .fastpls_score_matrix(model, "Ttrain")
  if (is.null(scores) || ncol(scores) < k || nrow(scores) != nrow(Xscaled)) {
    scores <- .fastpls_latent_scores(model, Xtrain, ncomp = k, backend = "cpu")
  }
  scores <- as.matrix(scores)[, seq_len(k), drop = FALSE]
  denom <- colSums(scores * scores)
  ok <- is.finite(denom) & denom > 0
  P <- matrix(0, nrow = ncol(Xscaled), ncol = k)
  if (any(ok)) {
    P[, ok] <- sweep(crossprod(Xscaled, scores[, ok, drop = FALSE]), 2L, denom[ok], "/", check.margin = FALSE)
  }
  rownames(P) <- colnames(Xtrain)
  colnames(P) <- paste0("LV", seq_len(k))
  model$P <- P
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

.normalize_cknn_memory <- function(cknn_memory = c("auto", "standard", "blocked", "streaming")) {
  cknn_memory <- as.character(cknn_memory)
  if (!length(cknn_memory) || is.na(cknn_memory[[1L]])) {
    cknn_memory <- "auto"
  }
  match.arg(cknn_memory[[1L]], c("auto", "standard", "blocked", "streaming"))
}

.cknn_prediction_block_size <- function() {
  as.integer(getOption("fastPLS.cknn_block_size", 2000L))[[1L]]
}

.cknn_train_block_size <- function() {
  as.integer(getOption("fastPLS.cknn_train_block_size", 10000L))[[1L]]
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
  model <- .attach_backend_control(model)
  model
}

.plssvd_label_aware_scores_fast_model <- function(Xtrain,
                                                  y_train,
                                                  ncomp,
                                                  scaling = 1L) {
  Xtrain <- as.matrix(Xtrain)
  y_train <- factor(y_train)
  lev <- levels(y_train)
  y_code <- as.integer(y_train)
  n <- nrow(Xtrain)
  p <- ncol(Xtrain)
  m <- length(lev)
  cap <- .cap_plssvd_ncomp(ncomp, n, p, m, warn = TRUE)
  ncomp <- as.integer(cap$ncomp)
  max_rank <- max(ncomp)
  stats <- label_crossprod_scaled_cpp(Xtrain, y_code, m, as.integer(scaling))
  sv <- svd(as.matrix(stats$S), nu = max_rank, nv = 0)
  R <- sv$u[, seq_len(max_rank), drop = FALSE]
  model <- list(
    P = matrix(numeric(0), 0L, 0L),
    Q = matrix(numeric(0), m, max_rank),
    Ttrain = matrix(numeric(0), 0L, 0L),
    R = R,
    mX = matrix(as.numeric(stats$mX), nrow = 1L),
    vX = matrix(as.numeric(stats$vX), nrow = 1L),
    mY = matrix(as.numeric(stats$mY), nrow = 1L),
    p = p,
    m = m,
    ncomp = ncomp,
    Yfit = array(numeric(0), dim = c(0L, 0L, 0L)),
    R2Y = rep(NA_real_, length(ncomp)),
    pls_method = "plssvd",
    classification = TRUE,
    lev = lev,
    predict_latent_ok = TRUE,
    xprod_default = TRUE,
    xprod_mode = "label_aware_class_sums",
    B_stored = FALSE,
    compact_prediction = TRUE,
    flash_svd = TRUE,
    flash_svd_backend = "cuda",
    predict_backend = "cuda_flash"
  )
  class(model) <- "fastPLS"
  .attach_backend_control(model)
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
                               top_m = 20L,
                               cknn_memory = c("auto", "standard", "blocked", "streaming")) {
  backend <- match.arg(backend)
  cknn_memory <- .normalize_cknn_memory(cknn_memory)
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
  if (!is.finite(tau) || tau <= 0) stop("tau must be positive", call. = FALSE)
  if (!is.finite(alpha)) stop("alpha must be finite", call. = FALSE)

  model <- .attach_latent_projection_cache(model)
  score_backend <- if (identical(backend, "cuda")) {
    "cuda"
  } else if (identical(backend, "metal") && isTRUE(has_metal())) {
    "metal"
  } else {
    "cpu"
  }
  y_codes <- as.integer(factor(y_train, levels = model$lev))
  if (anyNA(y_codes)) {
    stop("candidate-kNN received labels outside the training levels", call. = FALSE)
  }

  max_requested <- max(as.integer(model$ncomp))
  ncomp_eff <- pmin(as.integer(model$ncomp), max_requested)
  ncomp_eff <- pmax(ncomp_eff, 1L)
  unique_ncomp <- sort(unique(ncomp_eff))
  if (identical(cknn_memory, "auto")) {
    score_mb <- as.double(nrow(Xtrain)) * as.double(max(unique_ncomp)) * 8 / 1024^2
    cknn_memory <- if (score_mb >= 512 && length(unique_ncomp) == 1L) "streaming" else if (score_mb >= 128) "blocked" else "standard"
  }
  if (identical(cknn_memory, "streaming") && length(unique_ncomp) > 1L) {
    warning(
      "cknn_memory = 'streaming' currently streams the training-score cache only for scalar ncomp; using blocked prediction for this component grid.",
      call. = FALSE
    )
    cknn_memory <- "blocked"
  }

  if (identical(cknn_memory, "streaming") && length(unique_ncomp) == 1L) {
    kk <- unique_ncomp[[1L]]
    block_size <- max(1L, .cknn_train_block_size())
    n <- nrow(Xtrain)
    Ttrain_norm <- matrix(NA_real_, nrow = n, ncol = kk)
    for (start in seq.int(1L, n, by = block_size)) {
      stop <- min(n, start + block_size - 1L)
      idx <- start:stop
      Tb <- .fastpls_latent_scores(
        model,
        Xtrain[idx, , drop = FALSE],
        ncomp = kk,
        backend = score_backend
      )
      Ttrain_norm[idx, ] <- .candidate_row_l2(.candidate_score_space(model, Tb, kk))
    }
    cent <- .candidate_centroids(Ttrain_norm, y_codes, length(model$lev))
    models <- list(list(ncomp = kk, centroids = cent))
    names(models) <- as.character(kk)
    model$candidate_knn <- list(
      ncomp = unique_ncomp,
      models = models,
      Ttrain = Ttrain_norm,
      y_codes = y_codes,
      backend = backend,
      score_space = "precomputed_norm",
      memory = cknn_memory,
      parameters = list(
        knn_k = knn_k,
        tau = tau,
        alpha = alpha,
        top_m = top_m
      )
    )
    return(model)
  }

  if (is.null(model$Ttrain) ||
      length(model$Ttrain) == 0L ||
      !all(dim(model$Ttrain) > 0L) ||
      ncol(as.matrix(model$Ttrain)) < max_requested) {
    model$Ttrain <- .fastpls_latent_scores(
      model,
      Xtrain,
      ncomp = max_requested,
      backend = score_backend
    )
  }
  Ttrain <- as.matrix(model$Ttrain)
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
    memory = cknn_memory,
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
  memory <- .normalize_cknn_memory(object$candidate_knn$memory %||% "standard")
  block_size <- max(1L, .cknn_prediction_block_size())
  if (memory %in% c("blocked", "streaming") && nrow(Xtest) > block_size) {
    Xtest <- as.matrix(Xtest)
    Ttrain <- as.matrix(object$candidate_knn$Ttrain)
    y_codes <- as.integer(object$candidate_knn$y_codes)
    ntest <- nrow(Xtest)
    Ypredlab <- as.data.frame(matrix(nrow = ntest, ncol = length(object$ncomp)))
    colnames(Ypredlab) <- paste("ncomp=", object$ncomp, sep = "")
    Ypred_index <- matrix(NA_integer_, nrow = ntest, ncol = length(object$ncomp))
    colnames(Ypred_index) <- colnames(Ypredlab)
    top <- min(max(1L, as.integer(top)[1L]), length(object$lev))
    top_list <- score_list <- vector("list", length(object$ncomp))
    names(top_list) <- names(score_list) <- colnames(Ypredlab)
    if (top > 1L) {
      for (i in seq_along(object$ncomp)) {
        top_list[[i]] <- matrix(NA_character_, nrow = ntest, ncol = top)
        score_list[[i]] <- matrix(NA_real_, nrow = ntest, ncol = top)
      }
    }
    predict_backend <- character(length(object$ncomp))
    train_score_cache <- new.env(parent = emptyenv())
    for (start in seq.int(1L, ntest, by = block_size)) {
      stop <- min(ntest, start + block_size - 1L)
      idx <- start:stop
      Ttest_block <- .fastpls_latent_scores(
        object,
        Xtest[idx, , drop = FALSE],
        ncomp = max(ncomp_eff),
        backend = if (use_cuda) "cuda" else if (use_metal) "metal" else "cpu"
      )
      for (i in seq_along(object$ncomp)) {
        kk <- ncomp_eff[[i]]
        cm <- object$candidate_knn$models[[as.character(kk)]]
        if (is.null(cm)) {
          stop(sprintf("No fitted candidate-kNN classifier for ncomp=%s", kk), call. = FALSE)
        }
        cache_key <- as.character(kk)
        if (!exists(cache_key, envir = train_score_cache, inherits = FALSE)) {
          train_kk <- if (identical(object$candidate_knn$score_space, "precomputed_norm")) {
            Ttrain
          } else {
            .candidate_row_l2(.candidate_score_space(object, Ttrain, kk))
          }
          assign(cache_key, train_kk, envir = train_score_cache)
        }
        Ttest_kk <- .candidate_row_l2(.candidate_score_space(object, Ttest_block, kk))
        pred <- .candidate_knn_predict_core(
          Ttest_norm = Ttest_kk,
          Ttrain_norm = get(cache_key, envir = train_score_cache, inherits = FALSE),
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
        Ypredlab[idx, i] <- as.character(pred$Ypred)
        Ypred_index[idx, i] <- as.integer(pred$Ypred_index)
        predict_backend[[i]] <- pred$predict_backend %||% paste0(backend, "_candidate_knn")
        if (top > 1L) {
          top_list[[i]][idx, ] <- pred$Ypred_top
          score_list[[i]][idx, ] <- pred$Ypred_top_score
        }
      }
    }
    out <- list(Ypred = Ypredlab, Ypred_index = Ypred_index)
    out$predict_backend <- unique(predict_backend[nzchar(predict_backend)])
    if (top > 1L) {
      out$Ypred_top <- top_list
      out$Ypred_top_score <- score_list
    }
    return(out)
  }
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

.lda_train_projected_stream <- function(Xtrain,
                                        R,
                                        offset,
                                        y_codes,
                                        n_classes,
                                        ncomp,
                                        ridge = 1e-8,
                                        block_size = NULL,
                                        backend = c("cpu", "cuda", "metal")) {
  backend <- match.arg(backend)
  Xtrain <- as.matrix(Xtrain)
  R <- as.matrix(R)
  n <- nrow(Xtrain)
  p <- ncol(Xtrain)
  if (n < 1L || p < 1L || nrow(R) != p || ncol(R) < 1L) {
    stop("streamed LDA training received incompatible X/projection dimensions", call. = FALSE)
  }
  y_codes <- as.integer(y_codes)
  if (length(y_codes) != n || anyNA(y_codes) || any(y_codes < 1L | y_codes > n_classes)) {
    stop("streamed LDA training requires labels encoded as 1..n_classes", call. = FALSE)
  }
  ncomp <- as.integer(ncomp)
  kmax <- max(ncomp, na.rm = TRUE)
  if (!is.finite(kmax) || is.na(kmax) || kmax < 1L || kmax > ncol(R)) {
    stop("streamed LDA training component counts must be in 1..ncol(R)", call. = FALSE)
  }
  R <- R[, seq_len(kmax), drop = FALSE]
  offset <- as.numeric(offset)
  if (length(offset) < kmax) offset <- c(offset, rep(0, kmax - length(offset)))
  offset <- offset[seq_len(kmax)]
  if (is.null(block_size)) {
    block_size <- .fastpls_block_size(
      "fastPLS.label_aware_block_size",
      "FASTPLS_LABEL_AWARE_BLOCK_SIZE",
      default = 8192L
    )
  }
  block_size <- max(1L, as.integer(block_size)[1L])
  use_cuda <- identical(backend, "cuda") && .cuda_matmul_available()
  use_metal <- identical(backend, "metal") && isTRUE(has_metal())

  counts <- tabulate(y_codes, nbins = n_classes)
  if (any(counts <= 0L)) {
    stop("streamed LDA training received an empty class", call. = FALSE)
  }
  class_sums <- matrix(0, nrow = n_classes, ncol = kmax)
  gram <- matrix(0, nrow = kmax, ncol = kmax)

  for (start in seq(1L, n, by = block_size)) {
    stop <- min(n, start + block_size - 1L)
    rows <- start:stop
    Xb <- Xtrain[rows, , drop = FALSE]
    Tb <- if (use_cuda) {
      .cuda_matmul(Xb, R)
    } else if (use_metal) {
      .metal_mm(Xb, R)
    } else {
      Xb %*% R
    }
    if (any(offset != 0)) {
      Tb <- sweep(Tb, 2L, offset, "-", check.margin = FALSE)
    }
    gram <- gram + crossprod(Tb)
    rs <- rowsum(
      Tb,
      group = factor(y_codes[rows], levels = seq_len(n_classes)),
      reorder = FALSE
    )
    rs <- as.matrix(rs)
    if (nrow(rs) != n_classes) {
      full <- matrix(0, nrow = n_classes, ncol = kmax)
      pos <- match(rownames(rs), as.character(seq_len(n_classes)))
      full[pos[!is.na(pos)], ] <- rs[!is.na(pos), , drop = FALSE]
      rs <- full
    }
    class_sums <- class_sums + rs
    rm(Xb, Tb, rs)
    if ((start %/% block_size) %% 16L == 0L) gc(FALSE)
  }

  means <- sweep(class_sums, 1L, pmax(as.numeric(counts), 1), "/", check.margin = FALSE)
  pooled_full <- gram
  for (c in seq_len(n_classes)) {
    mu <- means[c, , drop = FALSE]
    pooled_full <- pooled_full - counts[[c]] * crossprod(mu)
  }
  pooled_full <- (pooled_full + t(pooled_full)) / 2
  pooled_full <- pooled_full / max(1, n - n_classes)

  unique_ncomp <- sort(unique(pmax(1L, pmin(ncomp, kmax))))
  models <- vector("list", length(unique_ncomp))
  names(models) <- as.character(unique_ncomp)
  ridge <- as.numeric(ridge)[1L]
  if (!is.finite(ridge) || ridge < 0) ridge <- 1e-8

  for (i in seq_along(unique_ncomp)) {
    kk <- unique_ncomp[[i]]
    pooled <- pooled_full[seq_len(kk), seq_len(kk), drop = FALSE]
    means_k <- means[, seq_len(kk), drop = FALSE]
    ridge_scale <- sum(diag(pooled)) / max(1L, kk)
    if (!is.finite(ridge_scale) || ridge_scale <= 0) ridge_scale <- 1
    lambda <- ridge * ridge_scale
    diag(pooled) <- diag(pooled) + lambda
    inv_cov <- tryCatch(
      solve(pooled),
      error = function(e) qr.solve(pooled)
    )
    linear <- means_k %*% inv_cov
    constants <- numeric(n_classes)
    priors <- as.numeric(counts) / n
    for (c in seq_len(n_classes)) {
      constants[[c]] <- -0.5 * drop(means_k[c, , drop = FALSE] %*% t(linear[c, , drop = FALSE])) +
        log(max(priors[[c]], .Machine$double.xmin))
    }
    models[[i]] <- list(
      means = means_k,
      inv_cov = inv_cov,
      linear = linear,
      constants = matrix(constants, nrow = 1L),
      priors = matrix(priors, ncol = 1L),
      ridge = lambda
    )
  }
  models
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
                                   k = getOption("fastPLS.k", 10L),
                                   tau = getOption("fastPLS.tau", 0.2),
                                   alpha = getOption("fastPLS.alpha", 0.75),
                                   top_m = getOption("fastPLS.top_m", 20L),
                                   cknn_memory = getOption("fastPLS.cknn_memory", "auto")) {
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
      knn_k = k,
      tau = tau,
      alpha = alpha,
      top_m = top_m,
      cknn_memory = cknn_memory
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

  if (.is_lda_classifier(classifier) &&
      !is.null(model$R_predict) &&
      !is.null(model$R_offset)) {
    backend <- if (identical(classifier, "lda_cuda")) {
      "cuda"
    } else if (identical(classifier, "lda_metal")) {
      "metal"
    } else {
      "cpu"
    }
    R_predict <- as.matrix(model$R_predict)
    ncomp_eff <- pmin(as.integer(model$ncomp), ncol(R_predict))
    ncomp_eff <- pmax(ncomp_eff, 1L)
    unique_ncomp <- sort(unique(ncomp_eff))
    score_mb <- as.numeric(nrow(Xtrain)) * as.numeric(max(unique_ncomp)) * 8 / 1024^2
    use_stream_project <- identical(backend, "cuda") ||
      identical(backend, "metal") ||
      isTRUE(score_mb >= 512)
    if (isTRUE(use_stream_project)) {
      lda_models <- .lda_train_projected_stream(
        Xtrain = Xtrain,
        R = R_predict[, seq_len(max(unique_ncomp)), drop = FALSE],
        offset = as.numeric(model$R_offset)[seq_len(max(unique_ncomp))],
        y_codes = y_codes,
        n_classes = length(model$lev),
        ncomp = unique_ncomp,
        ridge = as.numeric(lda_ridge)[1L],
        backend = backend
      )
      names(lda_models) <- as.character(unique_ncomp)
      model$lda <- list(
        ncomp = unique_ncomp,
        models = lda_models,
        ridge = as.numeric(lda_ridge)[1L],
        train_backend = paste0(backend, "_stream_project")
      )
      return(model)
    }
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

.fastpls_accuracy_from_class_labels <- function(lev, Ytest, Ypredlab) {
  vapply(seq_along(Ypredlab), function(i) {
    pred <- factor(as.character(Ypredlab[[i]]), levels = lev)
    obs <- factor(as.character(Ytest), levels = lev)
    mean(pred == obs, na.rm = TRUE)
  }, numeric(1))
}

.fastpls_ncomp_names <- function(ncomp) {
  paste0("ncomp=", as.integer(ncomp))
}

.fastpls_name_metric_path <- function(x, ncomp) {
  if (is.null(x)) {
    return(x)
  }
  if (length(x) != length(ncomp)) {
    return(x)
  }
  out <- as.numeric(x)
  names(out) <- .fastpls_ncomp_names(ncomp)
  out
}

.fastpls_name_pls_metric_paths <- function(x, ncomp = NULL) {
  if (is.null(ncomp)) {
    ncomp <- x$ncomp
  }
  if (is.null(ncomp)) {
    return(x)
  }
  for (field in c("accuracy", "Q2Y", "R2Y")) {
    if (!is.null(x[[field]])) {
      x[[field]] <- .fastpls_name_metric_path(x[[field]], ncomp)
    }
  }
  x
}

.fastpls_hidden_output_fields <- c(
  "B_stored",
  "compact_prediction",
  "pls_method",
  "predict_latent_ok",
  "xprod_default",
  "predict_backend",
  "flash_svd",
  "flash_svd_backend",
  "flash_svd_mode",
  "flash_block_size",
  "classification"
)

.fastpls_hide_internal_output_fields <- function(x) {
  present <- intersect(.fastpls_hidden_output_fields, names(x))
  if (!length(present)) {
    return(x)
  }
  internal <- attr(x, "fastPLS_internal", exact = TRUE)
  if (is.null(internal)) {
    internal <- list()
  }
  internal[present] <- x[present]
  x[present] <- NULL
  attr(x, "fastPLS_internal") <- internal
  x
}

.fastpls_restore_internal_output_fields <- function(x) {
  internal <- attr(x, "fastPLS_internal", exact = TRUE)
  if (is.null(internal) || !length(internal)) {
    return(x)
  }
  missing <- setdiff(names(internal), names(x))
  if (length(missing)) {
    x[missing] <- internal[missing]
  }
  x
}

print.fastPLS <- function(x, ...) {
  out <- .fastpls_hide_internal_output_fields(x)
  attr(out, "fastPLS_internal") <- NULL
  class(out) <- setdiff(class(out), "fastPLS")
  print(out, ...)
  invisible(x)
}

.fastpls_public_pls_output <- function(x, ncomp = NULL) {
  x <- .fastpls_name_pls_metric_paths(x, ncomp)
  .fastpls_hide_internal_output_fields(x)
}

.fastpls_permutation_cor <- function(Y, idx) {
  Y <- as.matrix(Y)
  if (nrow(Y) != length(idx)) {
    return(NA_real_)
  }
  y0 <- as.numeric(Y)
  yp <- as.numeric(Y[idx, , drop = FALSE])
  ok <- is.finite(y0) & is.finite(yp)
  if (sum(ok) < 2L || stats::sd(y0[ok]) == 0 || stats::sd(yp[ok]) == 0) {
    return(NA_real_)
  }
  as.numeric(stats::cor(y0[ok], yp[ok]))
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
      model$accuracy <- .fastpls_accuracy_from_class_labels(lev, Ytest, Ypredlab)
      model$Q2Y <- rep(NA_real_, length(model$ncomp))
    }
  }
  class(model) <- "fastPLS"
  model <- .attach_backend_control(model)
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
    svd.method = "rsvd",
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

.prepare_response <- function(Ytrain) {
  classification <- is.factor(Ytrain)
  lev <- if (classification) levels(Ytrain) else NULL
  list(
    Ytrain = if (classification) transformy(Ytrain) else as.matrix(Ytrain),
    classification = classification,
    lev = lev
  )
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
#' Generates predictions for new samples from fitted PLSSVD, SIMPLS, OPLS, or
#' kernel PLS models. Stored centering, scaling, latent projections, and
#' model-specific filtering are applied before producing numeric response
#' predictions or classification labels.
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
  object <- .fastpls_restore_internal_output_fields(object)
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
      res$accuracy <- .fastpls_accuracy_from_class_labels(object$lev, Ytest, res$Ypred)
      res$Q2Y <- rep(NA_real_, length(object$ncomp))
	    }
		    return(.fastpls_name_pls_metric_paths(res, object$ncomp))
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
	      cand_res$accuracy <- .fastpls_accuracy_from_class_labels(object$lev, Ytest, cand_res$Ypred)
	      cand_res$Q2Y <- rep(NA_real_, length(object$ncomp))
	    }
	    return(.fastpls_name_pls_metric_paths(cand_res, object$ncomp))
	  }
	  if (isTRUE(object$classification) &&
	      is.null(Ytest) &&
	      !isTRUE(raw_scores) &&
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
    if (!is.null(Ytest)) {
      bias_res$accuracy <- .fastpls_accuracy_from_class_labels(object$lev, Ytest, bias_res$Ypred)
      bias_res$Q2Y <- rep(NA_real_, length(object$ncomp))
    }
    return(.fastpls_name_pls_metric_paths(bias_res, object$ncomp))
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
    if (!is.null(Ytest)) {
      res$accuracy <- .fastpls_accuracy_from_class_labels(object$lev, Ytest, res$Ypred)
    }
  }
  .fastpls_name_pls_metric_paths(res, object$ncomp)
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
  out <- .attach_backend_control(out)
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
.kernel_pls_cpp <- function(Xtrain,
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
		                  classifier = c("argmax", "lda", "cknn"),
	                  lda_ridge = 1e-8,
	                  fit = FALSE,
                           return_variance = TRUE,
                           proj = FALSE) {
  classifier <- .resolve_classifier_for_backend(classifier, "cpu")
  svd.method <- match.arg(.normalize_svd_method(svd.method), c("irlba", "cpu_rsvd"))
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
      classifier = classifier,
      lda_ridge = lda_ridge,
      return_variance = return_variance
    )
  )
}

#' @noRd
.kernel_pls_cuda <- function(Xtrain,
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
		                   classifier = c("argmax", "lda", "cknn"),
	                  lda_ridge = 1e-8,
	                  fit = FALSE,
                            return_variance = TRUE,
	  proj = FALSE,
                            ...) {
  classifier <- .resolve_classifier_for_backend(classifier, "cuda")
  fit_fun <- .simpls_gpu
  .kernel_pls_fit(
    Xtrain, Ytrain, Xtest, Ytest, ncomp, match.arg(scaling), match.arg(kernel),
    gamma, degree, coef0, fit, proj, "cuda", fit_fun,
    c(
      list(
        rsvd_oversample = rsvd_oversample,
        rsvd_power = rsvd_power,
        svds_tol = svds_tol,
        seed = seed,
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
  out <- .attach_backend_control(out)
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
.opls_cpp <- function(Xtrain,
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
      classifier = classifier,
      lda_ridge = lda_ridge,
      return_variance = return_variance
    )
  )
}

#' @noRd
.opls_cuda <- function(Xtrain,
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
			                      classifier = c("argmax", "lda", "cknn"),
	                  lda_ridge = 1e-8,
	                  fit = FALSE,
                      return_variance = TRUE,
	  proj = FALSE,
                      ...) {
				  classifier <- .resolve_classifier_for_backend(classifier, "cuda")
  fit_fun <- .simpls_gpu
  .opls_fit(
    Xtrain, Ytrain, Xtest, Ytest, ncomp, match.arg(scaling), north, fit, proj,
    "cpp", fit_fun,
    c(
      list(
        rsvd_oversample = rsvd_oversample,
        rsvd_power = rsvd_power,
        svds_tol = svds_tol,
        seed = seed,
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
#'   Alternatively, a result returned by [pls.single.cv()]. In that case
#'   `pls()` refits the selected model on the full cross-validation training
#'   set and predicts `Xtest` using `best_ncomp` and the selected tuning
#'   settings.
#' @param Ytrain Training response (numeric or factor).
#'   When `Xtrain` is a [pls.single.cv()] result, the second positional argument
#'   may be used as `Xtest`.
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
#' @param return_loadings Compute and store predictor loadings `P`. The default
#'   is `FALSE` because `P` is mainly used for interpretation/loading plots and
#'   is not needed for prediction, VIP, or the benchmark pipelines.
#' @param proj Return projected `Ttest` when `TRUE`.
#' @param gpu_device_state Keep selected SIMPLS workspaces resident on the GPU when `TRUE`.
#' @param gpu_qr Use GPU QR finalization when available.
#' @param gpu_eig Use GPU eigensolver finalization when available.
#' @param gpu_finalize_threshold Component threshold controlling GPU-side finalization.
#' @return A `fastPLS` object.
#' @noRd
.simpls_gpu = function(Xtrain,
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
                      gpu_finalize_threshold = 32L,
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
  if (is.factor(Ytrain_original) &&
      !isTRUE(fit) &&
      identical(classifier, "lda_cuda")) {
    lev <- levels(Ytrain_original)
    model <- .plssvd_label_aware_scores_fast_model(
      Xtrain,
      Ytrain_original,
      ncomp = as.integer(ncomp),
      scaling = scal
    )
    cuda_reset_workspace()
    model$classification <- TRUE
    model$lev <- lev
    model$predict_latent_ok <- TRUE
    model <- .enable_flash_prediction(model, "cuda")
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
    return(model)
  }
  yprep <- .prepare_response(Ytrain)
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
      gpu_finalize_threshold = gpu_finalize_threshold
    )
  } else {
    NULL
  }
  if (!is.null(fused_model)) {
    fused_model <- .maybe_attach_pls_variance_explained(fused_model, Xtrain, return_variance)
    fused_model <- .attach_backend_control(fused_model)
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
  model <- .attach_backend_control(model)
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
#' @param gpu_finalize_threshold Component threshold controlling GPU-side finalization.
#' @return A `fastPLS` object fitted with GPU PLSSVD.
#' @noRd
.plssvd_gpu = function(Xtrain,
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
                      gpu_finalize_threshold = 32L,
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
  yprep <- .prepare_response(Ytrain)
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
      gpu_finalize_threshold = gpu_finalize_threshold
    )
  } else {
    NULL
  }
  if (!is.null(fused_model)) {
    fused_model <- .maybe_attach_pls_variance_explained(fused_model, Xtrain, return_variance)
    fused_model <- .attach_backend_control(fused_model)
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
  model <- .attach_backend_control(model)
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
.plssvd_flash_gpu <- function(Xtrain, Ytrain, Xtest = NULL, Ytest = NULL,
                             ncomp = 2, scaling = c("centering", "autoscaling", "none"),
                             rsvd_oversample = 10L, rsvd_power = 1L,
                             svds_tol = 0, seed = 1L, fit = FALSE,
                             proj = FALSE, gpu_qr = TRUE, gpu_eig = TRUE,
                             gpu_finalize_threshold = 32L) {
  model <- .plssvd_gpu(
    Xtrain = Xtrain, Ytrain = Ytrain, Xtest = NULL, Ytest = NULL,
    ncomp = ncomp, scaling = scaling, rsvd_oversample = rsvd_oversample,
    rsvd_power = rsvd_power, svds_tol = svds_tol, seed = seed,
    fit = fit, proj = FALSE, gpu_qr = gpu_qr, gpu_eig = gpu_eig,
    gpu_finalize_threshold = gpu_finalize_threshold
  )
  .predict_flash_attach(model, Xtest, Ytest, proj)
}

#' GPU SIMPLS with FlashSVD-style low-rank CUDA prediction
#' @noRd
.simpls_flash_gpu <- function(Xtrain, Ytrain, Xtest = NULL, Ytest = NULL,
                             ncomp = 2, scaling = c("centering", "autoscaling", "none"),
                             rsvd_oversample = 10L, rsvd_power = 1L,
                             svds_tol = 0, seed = 1L, fit = FALSE,
                             proj = FALSE, gpu_device_state = TRUE,
                             gpu_qr = TRUE, gpu_eig = TRUE,
                             gpu_finalize_threshold = 32L) {
  model <- .simpls_gpu(
    Xtrain = Xtrain, Ytrain = Ytrain, Xtest = NULL, Ytest = NULL,
    ncomp = ncomp, scaling = scaling, rsvd_oversample = rsvd_oversample,
    rsvd_power = rsvd_power, svds_tol = svds_tol, seed = seed,
    fit = fit, proj = FALSE, gpu_device_state = gpu_device_state,
    gpu_qr = gpu_qr, gpu_eig = gpu_eig,
    gpu_finalize_threshold = gpu_finalize_threshold
  )
  .predict_flash_attach(model, Xtest, Ytest, proj)
}

#' GPU OPLS with FlashSVD-style low-rank CUDA prediction
#' @noRd
.opls_flash_gpu <- function(Xtrain, Ytrain, Xtest = NULL, Ytest = NULL,
                           ncomp = 2, north = 1L,
                           scaling = c("centering", "autoscaling", "none"),
                           rsvd_oversample = 10L, rsvd_power = 1L,
                           svds_tol = 0, seed = 1L, fit = FALSE,
                           proj = FALSE, ...) {
  model <- .opls_cuda(
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
.kernel_pls_flash_gpu <- function(Xtrain, Ytrain, Xtest = NULL, Ytest = NULL,
                                 ncomp = 2,
                                 scaling = c("centering", "autoscaling", "none"),
                                 kernel = c("linear", "rbf", "poly"),
                                 gamma = NULL, degree = 3L, coef0 = 1,
                                 rsvd_oversample = 10L, rsvd_power = 1L,
                                 svds_tol = 0, seed = 1L,
                                 fit = FALSE, proj = FALSE, ...) {
  model <- .kernel_pls_cuda(
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

.cv_normalize_selection_metric <- function(selection_metric = NULL) {
  if (is.null(selection_metric) || !length(selection_metric)) {
    return("auto")
  }
  metric <- tolower(gsub("[[:space:]-]+", "_", as.character(selection_metric[[1L]])))
  aliases <- c(
    auto = "auto",
    acc = "accuracy",
    cv_accuracy = "accuracy",
    accuracy = "accuracy",
    r2 = "r2",
    r_squared = "r2",
    rsquared = "r2",
    q2 = "q2",
    q_squared = "q2",
    rmsd = "rmsd",
    rmse = "rmsd"
  )
  if (metric %in% names(aliases)) {
    metric <- unname(aliases[[metric]])
  }
  if (!metric %in% c("auto", "accuracy", "r2", "q2", "rmsd")) {
    stop(
      "selection_metric must be one of 'auto', 'accuracy', 'r2', 'q2', or 'rmsd'.",
      call. = FALSE
    )
  }
  metric
}

.cv_selection_metric_from_dots <- function(dots) {
  if (!is.list(dots)) {
    dots <- list()
  }
  keys <- intersect(c("selection_metric", "metric", "opt_metric", "criterion"), names(dots))
  if (!length(keys)) {
    return(list(metric = "auto", dots = dots))
  }
  metric <- dots[[keys[[1L]]]]
  dots[keys] <- NULL
  list(metric = .cv_normalize_selection_metric(metric), dots = dots)
}

.cv_metric_from_matrix <- function(Ytrue, Ypred, Ytrain = NULL, metric = "auto") {
  metric <- .cv_normalize_selection_metric(metric)
  Ytrue <- as.matrix(Ytrue)
  Ypred <- as.matrix(Ypred)
  if (!all(dim(Ytrue) == dim(Ypred))) {
    stop("Ytrue and Ypred must have the same dimensions for CV metric calculation.", call. = FALSE)
  }
  if (identical(metric, "auto")) {
    metric <- if (ncol(Ytrue) == 1L) "q2" else "rmsd"
  }
  if (identical(metric, "accuracy")) {
    stop("Accuracy selection is only available for factor responses.", call. = FALSE)
  }
  if (identical(metric, "rmsd")) {
    return(list(metric_name = "rmsd", metric_value = sqrt(mean((Ypred - Ytrue)^2, na.rm = TRUE))))
  }
  Ytrain_mat <- if (!is.null(Ytrain)) as.matrix(Ytrain) else Ytrue
  center <- colMeans(Ytrain_mat, na.rm = TRUE)
  press <- sum((Ypred - Ytrue)^2, na.rm = TRUE)
  tss <- sum(sweep(Ytrue, 2L, center, "-")^2, na.rm = TRUE)
  list(
    metric_name = metric,
    metric_value = if (is.finite(tss) && tss > 0) 1 - press / tss else NA_real_
  )
}

.cv_regression_q2_rmsd <- function(Ytrue, Ypred, Ytrain = NULL) {
  q2 <- .cv_metric_from_matrix(Ytrue, Ypred, Ytrain = Ytrain, metric = "q2")$metric_value
  rmsd <- .cv_metric_from_matrix(Ytrue, Ypred, Ytrain = Ytrain, metric = "rmsd")$metric_value
  list(Q2Y = q2, RMSD = rmsd)
}

.cv_classification_q2_path <- function(Ytrue, Ypred, lev) {
  dims <- dim(Ypred)
  if (length(dims) != 3L) {
    return(NA_real_)
  }
  Ymat <- .fastpls_one_hot_labels(Ytrue, lev)
  vapply(seq_len(dims[[3L]]), function(i) {
    pred_i <- matrix(Ypred[, , i], nrow = dims[[1L]], ncol = dims[[2L]])
    if (!any(is.finite(pred_i))) {
      return(NA_real_)
    }
    .cv_metric_from_matrix(
      Ytrue = Ymat,
      Ypred = pred_i,
      Ytrain = Ymat,
      metric = "q2"
    )$metric_value
  }, numeric(1))
}

.cv_training_fit_summary <- function(Xdata,
                                     Ydata,
                                     ncomp,
                                     scaling,
                                     method,
                                     backend,
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
                                     north,
                                     kernel,
                                     gamma,
                                     degree,
                                     coef0) {
  out <- tryCatch({
    fit <- pls(
      Xtrain = Xdata,
      Ytrain = Ydata,
      ncomp = ncomp,
      scaling = scaling,
      method = method,
      svd.method = svd.method,
      rsvd_oversample = rsvd_oversample,
      rsvd_power = rsvd_power,
      svds_tol = svds_tol,
      seed = seed,
      irlba_work = irlba_work,
      irlba_maxit = irlba_maxit,
      irlba_tol = irlba_tol,
      irlba_eps = irlba_eps,
      irlba_svtol = irlba_svtol,
      fit = TRUE,
      return_variance = FALSE,
      proj = FALSE,
      backend = backend,
      north = north,
      kernel = kernel,
      gamma = gamma,
      degree = degree,
      coef0 = coef0
    )
    list(
      R2Y = .fastpls_name_metric_path(fit$R2Y, ncomp),
      Yfit = fit$Yfit
    )
  }, error = function(e) {
    list(
      R2Y = .fastpls_name_metric_path(rep(NA_real_, length(ncomp)), ncomp),
      Yfit = NULL
    )
  })
  if (length(out$R2Y) != length(ncomp)) {
    out$R2Y <- rep_len(out$R2Y, length(ncomp))
  }
  out$R2Y <- .fastpls_name_metric_path(out$R2Y, ncomp)
  out
}

.is_single_pls_cv_result <- function(x) {
  inherits(x, "fastPLSCV") ||
    (is.list(x) && !is.null(x$best_ncomp) && !is.null(x$tuning_config))
}

.cv_attach_fit_data <- function(res, Xdata, Ydata) {
  attr(res, "fit_data") <- list(Xdata = Xdata, Ydata = Ydata)
  class(res) <- unique(c("fastPLSCV", class(res)))
  res
}

.cv_drop_fit_data <- function(res) {
  attr(res, "fit_data") <- NULL
  class(res) <- setdiff(class(res), "fastPLSCV")
  res
}

.pls_from_single_cv_result <- function(cv,
                                       Xtest = NULL,
                                       Ytest = NULL,
                                       fit = FALSE,
                                       return_variance = TRUE,
                                       return_loadings = FALSE,
                                       proj = FALSE,
                                       perm.test = FALSE,
                                       times = 100) {
  fit_data <- attr(cv, "fit_data", exact = TRUE)
  if (is.null(fit_data) || is.null(fit_data$Xdata) || is.null(fit_data$Ydata)) {
    stop(
      "This pls.single.cv() result does not contain the training data needed ",
      "for automatic refitting. Please rerun pls.single.cv() with the current ",
      "fastPLS version, or call pls(Xtrain, Ytrain, ...) manually using ",
      "cv$best_parameters.",
      call. = FALSE
    )
  }
  cfg <- cv$tuning_config
  if (is.null(cfg)) {
    stop("The pls.single.cv() result does not contain tuning_config.", call. = FALSE)
  }
  params <- .cv_config_list(cfg)
  svd_dots <- cfg$svd_dots %||% list()
  args <- c(
    list(
      Xtrain = fit_data$Xdata,
      Ytrain = fit_data$Ydata,
      Xtest = Xtest,
      Ytest = Ytest,
      ncomp = as.integer(cv$best_ncomp[[1L]]),
      scaling = params$scaling %||% "centering",
      method = params$method %||% "simpls",
      svd.method = params$svd.method %||% "irlba",
      classifier = params$classifier %||% "argmax",
      lda_ridge = params$lda_ridge %||% 1e-8,
      k = params$k %||% 10L,
      tau = params$tau %||% 0.2,
      alpha = params$alpha %||% 0.75,
      top_m = params$top_m %||% 20L,
      cknn_memory = params$cknn_memory %||% "auto",
      fit = fit,
      return_variance = return_variance,
      return_loadings = return_loadings,
      proj = proj,
      perm.test = perm.test,
      times = times,
      backend = params$backend %||% "cpu",
      north = params$north %||% 1L,
      kernel = params$kernel %||% "linear",
      gamma = params$gamma,
      degree = params$degree %||% 3L,
      coef0 = params$coef0 %||% 1
    ),
    svd_dots
  )
  args <- args[!vapply(args, is.null, logical(1L))]
  model <- do.call(pls, args)
  model$cv_best_parameters <- cv$best_parameters
  model$cv_best_metric_name <- cv$best_metric_name
  model$cv_best_metric_value <- cv$best_metric_value
  model
}

.cv_selection_metrics <- function(cv_res, Ydata, classification, selection_metric = "auto") {
  selection_metric <- .cv_normalize_selection_metric(selection_metric)
  if (classification) {
    if (identical(selection_metric, "auto")) {
      selection_metric <- "accuracy"
    }
    if (identical(selection_metric, "accuracy")) {
      return(cv_res$metrics)
    }
    if (identical(selection_metric, "q2")) {
      if (is.null(cv_res$Ypred)) {
        stop("Stored classification score predictions are required to optimize selection_metric = 'q2'.", call. = FALSE)
      }
      q2 <- .cv_classification_q2_path(Ydata, cv_res$Ypred, cv_res$levels)
      return(data.frame(
        ncomp_index = seq_along(q2),
        metric_name = rep("q2", length(q2)),
        metric_value = q2,
        stringsAsFactors = FALSE
      ))
    }
    if (!identical(selection_metric, "r2")) {
      stop(
        "Classification CV can optimize selection_metric = 'accuracy' or 'q2'.",
        call. = FALSE
      )
    }
    stop("Classification selection_metric = 'r2' is based on the full-data training fit and cannot be optimized by held-out folds.", call. = FALSE)
  }
  if (identical(selection_metric, "auto")) {
    selection_metric <- "rmsd"
  }
  if (identical(selection_metric, "accuracy")) {
    stop(
      "Regression CV can only optimize selection_metric = 'r2', 'q2', or 'rmsd'.",
      call. = FALSE
    )
  }
  if (!is.null(cv_res$metrics) && is.null(cv_res$Ypred)) {
    return(cv_res$metrics)
  }
  if (is.null(cv_res$Ypred)) {
    stop("Stored CV predictions are required to optimize the requested regression metric.", call. = FALSE)
  }
  dims <- dim(cv_res$Ypred)
  if (length(dims) != 3L) {
    stop("Internal CV prediction output must be a 3D array.", call. = FALSE)
  }
  metrics <- data.frame(
    ncomp_index = seq_len(dims[[3L]]),
    metric_name = character(dims[[3L]]),
    metric_value = numeric(dims[[3L]]),
    stringsAsFactors = FALSE
  )
  for (i in seq_len(dims[[3L]])) {
    mat <- cv_res$Ypred[, , i, drop = TRUE]
    metric <- .cv_metric_from_matrix(Ydata, mat, Ytrain = Ydata, metric = selection_metric)
    metrics$metric_name[[i]] <- metric$metric_name
    metrics$metric_value[[i]] <- metric$metric_value
  }
  metrics
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
                             k = 10L,
                            tau = 0.2,
                            alpha = 0.75,
                            top_m = 20L,
                            gpu_qr = TRUE,
                            gpu_eig = TRUE,
                             gpu_finalize_threshold = 32L,
                             store_predictions = TRUE,
                             selection_metric = "auto") {
  method <- match.arg(method)
  backend <- match.arg(backend)
  classifier <- .normalize_classifier_public(classifier)
  classifier_id <- switch(classifier, argmax = 0L, lda = 1L, cknn = 2L)
  k <- max(1L, as.integer(k)[1L])
  top_m <- max(1L, as.integer(top_m)[1L])
  tau <- as.numeric(tau)[1L]
  alpha <- as.numeric(alpha)[1L]
  if (!is.finite(tau) || tau <= 0) {
    stop("tau must be a finite positive number", call. = FALSE)
  }
  if (!is.finite(alpha)) {
    stop("alpha must be finite", call. = FALSE)
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
      k = k,
      tau = tau,
      alpha = alpha,
      top_m = top_m,
      store_predictions = isTRUE(store_predictions),
      metric_id = .cv_metric_id(selection_metric, classification)
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
          gpu_finalize_threshold = gpu_finalize_threshold
        )
      )
    } else {
      res <- .with_gpu_native_options(
        run_cv_profiled(),
        gpu_device_state = cuda_simpls_family,
        gpu_qr = gpu_qr,
        gpu_eig = gpu_eig,
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
  } else if (!is.null(res$Ypred)) {
    .decode_cv_predictions(res$Ypred, Yoriginal, classification, lev)
  } else {
    list(pred = NULL, metrics = res$metrics)
  }
  if (classification && !is.null(res$Ypred)) {
    res$Yscore <- res$Ypred
    res$Q2Y <- .cv_classification_q2_path(Yoriginal, res$Ypred, lev)
    if (!isTRUE(return_scores)) {
      res$Ypred <- NULL
    }
  }
  if (classification && !is.null(decoded$metrics)) {
    res$accuracy <- as.numeric(decoded$metrics$metric_value)
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
                            classifier = c("argmax", "lda", "cknn"),
                            lda_ridge = 1e-8,
                            k = 10L,
                            tau = 0.2,
                            alpha = 0.75,
                            top_m = 20L,
                            cknn_memory = c("auto", "standard", "blocked", "streaming"),
                            return_scores = TRUE,
                            store_predictions = TRUE,
                            selection_metric = "auto",
                            ...) {
  method <- match.arg(method)
  backend <- match.arg(backend)
  scaling <- match.arg(scaling)
  classifier <- .resolve_classifier_for_backend(classifier, backend)
  k <- max(1L, as.integer(k)[1L])
  tau <- as.numeric(tau)[1L]
  alpha <- as.numeric(alpha)[1L]
  top_m <- max(1L, as.integer(top_m)[1L])
  cknn_memory <- .normalize_cknn_memory(cknn_memory)
  if (!is.finite(tau) || tau <= 0) {
    stop("tau must be a finite positive number", call. = FALSE)
  }
  if (!is.finite(alpha)) {
    stop("alpha must be finite", call. = FALSE)
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
  score_pred <- if (classification) {
    array(NA_real_, dim = c(nrow(Xdata), q_response, nslice))
  } else if (!classification) {
    if (isTRUE(store_predictions)) array(NA_real_, dim = c(nrow(Xdata), q_response, nslice)) else NULL
  } else {
    NULL
  }
  if (classification && !isTRUE(store_predictions)) {
    class_pred <- NULL
  }
  metric_id <- .cv_metric_id(selection_metric, classification)
  metric_correct <- metric_total <- numeric(nslice)
  metric_sse <- metric_count <- numeric(nslice)
  metric_tss <- if (!classification && metric_id %in% c(2L, 3L)) {
    center <- colMeans(Yoriginal, na.rm = TRUE)
    sum(sweep(Yoriginal, 2L, center, "-")^2, na.rm = TRUE)
  } else {
    NA_real_
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
      fallback_idx <- match(fallback, lev)
      if (!is.null(class_pred)) {
        class_pred[test_idx, ] <- fallback_idx
      }
      for (j in seq_len(nslice)) {
        pred_chr <- rep(fallback, length(test_idx))
        metric_correct[[j]] <- metric_correct[[j]] +
          sum(pred_chr == as.character(Ydata[test_idx]), na.rm = TRUE)
        metric_total[[j]] <- metric_total[[j]] + length(test_idx)
      }
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
      k = k,
      tau = tau,
      alpha = alpha,
      top_m = top_m,
      cknn_memory = cknn_memory
    )

    if (classification) {
      raw_scores <- tryCatch(
        predict(fit, Xdata[test_idx, , drop = FALSE], raw_scores = TRUE),
        error = function(e) NULL
      )
      score_cube <- if (!is.null(raw_scores$Yscore)) {
        raw_scores$Yscore
      } else if (length(dim(raw_scores$Ypred)) == 3L) {
        raw_scores$Ypred
      } else {
        NULL
      }
      if (!is.null(score_cube) && length(dim(score_cube)) == 3L) {
        for (j in seq_len(nslice)) {
          score_pred[test_idx, , j] <- matrix(
            score_cube[, , j],
            nrow = length(test_idx),
            ncol = q_response
          )
        }
      }
      for (j in seq_len(nslice)) {
        pred_chr <- .cv_class_predictions_from_fit(fit, j, length(test_idx))
        if (!is.null(class_pred)) {
          class_pred[test_idx, j] <- match(pred_chr, lev)
        }
        metric_correct[[j]] <- metric_correct[[j]] +
          sum(pred_chr == as.character(Ydata[test_idx]), na.rm = TRUE)
        metric_total[[j]] <- metric_total[[j]] + length(test_idx)
      }
    } else {
      for (j in seq_len(nslice)) {
        pred_mat <- .cv_regression_predictions_from_fit(
          fit,
          component_index = j,
          ntest = length(test_idx),
          q_response = q_response
        )
        if (!is.null(score_pred)) {
          score_pred[test_idx, , j] <- pred_mat
        }
        diff <- pred_mat - Ytest
        metric_sse[[j]] <- metric_sse[[j]] + sum(diff^2, na.rm = TRUE)
        metric_count[[j]] <- metric_count[[j]] + sum(is.finite(diff))
      }
    }
  }

  metric_name <- if (classification) {
    rep("accuracy", nslice)
  } else if (metric_id == 2L) {
    rep("r2", nslice)
  } else if (metric_id == 3L) {
    rep("q2", nslice)
  } else {
    rep("rmsd", nslice)
  }
  metric_value <- if (classification) {
    ifelse(metric_total > 0, metric_correct / metric_total, NA_real_)
  } else if (metric_id %in% c(2L, 3L)) {
    if (is.finite(metric_tss) && metric_tss > 0) {
      1 - metric_sse / metric_tss
    } else {
      rep(NA_real_, nslice)
    }
  } else {
    sqrt(metric_sse / pmax(metric_count, 1))
  }
  q2_value <- if (!classification && is.finite(metric_tss) && metric_tss > 0) {
    1 - metric_sse / metric_tss
  } else if (classification) {
    .cv_classification_q2_path(Ydata, score_pred, lev)
  } else {
    rep(NA_real_, nslice)
  }
  rmsd_value <- if (!classification) {
    sqrt(metric_sse / pmax(metric_count, 1))
  } else {
    rep(NA_real_, nslice)
  }
  online_metrics <- data.frame(
    ncomp_index = seq_len(nslice),
    metric_name = metric_name,
    metric_value = metric_value,
    stringsAsFactors = FALSE
  )

  res <- list(
    Ypred = score_pred,
    Yscore = if (classification) score_pred else NULL,
    class_pred = class_pred,
    fold = fold,
    ncomp = ncomp,
    method = method,
    backend = backend,
    classification = classification,
    levels = lev,
    status = "ok"
  )
  decoded <- if (classification && !is.null(class_pred)) {
    .decode_cv_class_predictions(class_pred, Ydata, lev)
  } else if (!is.null(score_pred)) {
    .decode_cv_predictions(score_pred, Yoriginal, FALSE, NULL)
  } else {
    list(pred = NULL, metrics = online_metrics)
  }
  res$pred <- decoded$pred
  res$metrics <- if (is.null(decoded$metrics)) online_metrics else decoded$metrics
  res$Q2Y <- as.numeric(q2_value)
  res$RMSD <- as.numeric(rmsd_value)
  res
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
.plssvd_cv_cpp <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                          scaling = c("centering", "autoscaling", "none"),
                          svd.method = c("cpu_rsvd", "irlba"), xprod = NULL, ...) {
  .pls_cv_compiled(Xdata, Ydata, constrain, ncomp, kfold, scaling, "plssvd", "cpp", svd.method, xprod = xprod, ...)
}

#' @noRd
.simpls_cv_cpp <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                          scaling = c("centering", "autoscaling", "none"),
                          svd.method = c("cpu_rsvd", "irlba"), xprod = NULL, ...) {
  .pls_cv_compiled(Xdata, Ydata, constrain, ncomp, kfold, scaling, "simpls", "cpp", svd.method, xprod = xprod, ...)
}

.simpls_fast_cv_cpp <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                               scaling = c("centering", "autoscaling", "none"),
                               svd.method = c("cpu_rsvd", "irlba"), xprod = NULL, ...) {
  .simpls_cv_cpp(Xdata, Ydata, constrain, ncomp, kfold, scaling, svd.method, xprod = xprod, ...)
}

#' @noRd
.opls_cv_cpp <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                        north = 1L,
                        scaling = c("centering", "autoscaling", "none"),
                        svd.method = c("cpu_rsvd", "irlba"), xprod = NULL, ...) {
  pred_ncomp <- pmax(1L, as.integer(ncomp) - as.integer(north))
  .pls_cv_compiled(Xdata, Ydata, constrain, pred_ncomp, kfold, scaling, "opls", "cpp", svd.method, xprod = xprod, north = north, ...)
}

#' @noRd
.kernelpls_cv_cpp <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                             scaling = c("centering", "autoscaling", "none"),
                             svd.method = c("cpu_rsvd", "irlba"), xprod = NULL, ...) {
  .pls_cv_compiled(Xdata, Ydata, constrain, ncomp, kfold, scaling, "kernelpls", "cpp", svd.method, xprod = xprod, ...)
}

#' @noRd
.plssvd_cv_cuda <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                           scaling = c("centering", "autoscaling", "none"),
                           xprod = NULL, ...) {
  .pls_cv_compiled(Xdata, Ydata, constrain, ncomp, kfold, scaling, "plssvd", "cuda", xprod = xprod, ...)
}

#' @noRd
.simpls_cv_cuda <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                           scaling = c("centering", "autoscaling", "none"),
                           xprod = NULL, ...) {
  .pls_cv_compiled(Xdata, Ydata, constrain, ncomp, kfold, scaling, "simpls", "cuda", xprod = xprod, ...)
}

.simpls_fast_cv_cuda <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                                scaling = c("centering", "autoscaling", "none"),
                                xprod = NULL, ...) {
  .simpls_cv_cuda(Xdata, Ydata, constrain, ncomp, kfold, scaling, xprod = xprod, ...)
}

#' @noRd
.opls_cv_cuda <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                         north = 1L,
                         scaling = c("centering", "autoscaling", "none"),
                         xprod = NULL, ...) {
  pred_ncomp <- pmax(1L, as.integer(ncomp) - as.integer(north))
  .pls_cv_compiled(Xdata, Ydata, constrain, pred_ncomp, kfold, scaling, "opls", "cuda", xprod = xprod, north = north, ...)
}

#' @noRd
.kernelpls_cv_cuda <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                              scaling = c("centering", "autoscaling", "none"),
                              xprod = NULL, ...) {
  .pls_cv_compiled(Xdata, Ydata, constrain, ncomp, kfold, scaling, "kernelpls", "cuda", xprod = xprod, ...)
}

#' @noRd
.plssvd_cv_metal <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                            scaling = c("centering", "autoscaling", "none"),
                            xprod = NULL, ...) {
  .pls_cv_compiled(Xdata, Ydata, constrain, ncomp, kfold, scaling, "plssvd", "metal", xprod = xprod, ...)
}

#' @noRd
.simpls_cv_metal <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                            scaling = c("centering", "autoscaling", "none"),
                            xprod = NULL, ...) {
  .pls_cv_compiled(Xdata, Ydata, constrain, ncomp, kfold, scaling, "simpls", "metal", xprod = xprod, ...)
}

.simpls_fast_cv_metal <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                                 scaling = c("centering", "autoscaling", "none"),
                                 xprod = NULL, ...) {
  .simpls_cv_metal(Xdata, Ydata, constrain, ncomp, kfold, scaling, xprod = xprod, ...)
}

#' @noRd
.opls_cv_metal <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                          north = 1L,
                          scaling = c("centering", "autoscaling", "none"),
                          xprod = NULL, ...) {
  pred_ncomp <- pmax(1L, as.integer(ncomp) - as.integer(north))
  .pls_cv_compiled(Xdata, Ydata, constrain, pred_ncomp, kfold, scaling, "opls", "metal", xprod = xprod, north = north, ...)
}

#' @noRd
.kernelpls_cv_metal <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                               scaling = c("centering", "autoscaling", "none"),
                               xprod = NULL, ...) {
  .pls_cv_compiled(Xdata, Ydata, constrain, ncomp, kfold, scaling, "kernelpls", "metal", xprod = xprod, ...)
}

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
.svd_methods <- function() {
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
                                            method = c("rsvd", "irlba")) {
  backend <- match.arg(backend)
  method <- match.arg(method)
  method <- .normalize_svd_method(method)
  method <- match.arg(method, c("rsvd", "irlba"))
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

.truncated_rsvd_metal_xprod <- function(X,
                                        Y,
                                        k,
                                        rsvd_oversample = 10L,
                                        rsvd_power = 1L,
                                        seed = 1L,
                                        left_only = FALSE) {
  if (!isTRUE(has_metal())) {
    stop("method='metal_rsvd' requires a macOS build with Apple Metal support.", call. = FALSE)
  }

  X <- as.matrix(X)
  Y <- as.matrix(Y)
  if (nrow(X) != nrow(Y)) {
    stop("Metal matrix-free xprod SVD requires X and Y to have the same number of rows.", call. = FALSE)
  }

  p <- ncol(X)
  q <- ncol(Y)
  max_rank <- min(p, q)
  target <- min(max_rank, max(1L, as.integer(k)[1L]))
  sketch_rank <- min(max_rank, target + max(0L, as.integer(rsvd_oversample)[1L]))

  multiply <- function(V) {
    .metal_crossprod(X, .metal_mm(Y, V))
  }
  tmultiply <- function(U) {
    .metal_crossprod(Y, .metal_mm(X, U))
  }

  set.seed(as.integer(seed)[1L])
  omega <- matrix(rnorm(q * sketch_rank), nrow = q, ncol = sketch_rank)
  Ysk <- multiply(omega)

  power_iters <- max(0L, as.integer(rsvd_power)[1L])
  if (power_iters == 1L) {
    Ysk <- multiply(tmultiply(Ysk))
  } else if (power_iters > 1L) {
    for (i in seq_len(power_iters)) {
      Z <- tmultiply(Ysk)
      Qz <- qr.Q(qr(Z))
      Ysk <- multiply(Qz)
    }
  }

  Q <- qr.Q(qr(Ysk))
  B <- t(tmultiply(Q))
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
#' Computes a truncated singular value decomposition of a dense numeric matrix
#' with a selected CPU, CUDA, or Metal backend. The result contains singular
#' values, requested singular vectors, decomposition rank, elapsed time, and
#' backend metadata.
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
                    method = c("rsvd", "irlba"),
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
#' @param xtest Optional independent matrix to project using the PCA loadings
#'   learned from `x`. The same centering and scaling estimated from `x` are
#'   applied to `xtest`.
#' @param center Logical; center columns before SVD.
#' @param scale Logical; scale columns before SVD.
#' @param backend Compute backend. \code{cpu} runs on the host CPU. \code{cuda} and
#'   \code{metal} use the corresponding native randomized-SVD backend when
#'   available.
#' @param method SVD algorithm family. \code{irlba} is available only with
#'   \code{backend = cpu}. \code{rsvd} uses randomized SVD on the selected backend.
#' @param ... Additional arguments passed to [fastsvd()].
#' @return A `fastPLSPCA` object with training `scores`, optional
#'   `scores_test`, loadings, preprocessing values, and per-component
#'   `variance_explained` plus cumulative variance explained.
#' @examples
#' pc <- pca(as.matrix(iris[, 1:4]), ncomp = 2, backend = "cpu",
#'           method = "rsvd", seed = 1)
#' head(pc$scores)
#' pc$variance_explained
#' head(predict(pc, as.matrix(iris[1:5, 1:4])))
#' @export
pca <- function(x,
                ncomp = 2L,
                xtest = NULL,
                center = TRUE,
                scale = FALSE,
                backend = c("cpu", "cuda", "metal"),
                method = c("rsvd", "irlba"),
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
  if (!is.null(xtest)) {
    out$scores_test <- predict(out, xtest)
  }
  out
}

.predict_fastplspca_scores <- function(object, newdata, ncomp = NULL) {
  if (!inherits(object, "fastPLSPCA")) {
    stop("object must be a fastPLSPCA object.", call. = FALSE)
  }
  if (missing(newdata) || is.null(newdata)) {
    stop("newdata must be supplied.", call. = FALSE)
  }
  newdata <- as.matrix(newdata)
  loadings <- as.matrix(object$loadings)
  if (ncol(newdata) != nrow(loadings)) {
    stop(
      "newdata must have the same number of columns used to fit the PCA object.",
      call. = FALSE
    )
  }
  if (!is.null(colnames(newdata)) && !is.null(rownames(loadings)) &&
      !identical(colnames(newdata), rownames(loadings))) {
    warning(
      "newdata column names differ from the PCA loading names; projection uses column order.",
      call. = FALSE
    )
  }
  k <- if (is.null(ncomp)) {
    ncol(loadings)
  } else {
    max(1L, min(as.integer(ncomp)[1L], ncol(loadings)))
  }
  center <- object$center %||% rep(0, nrow(loadings))
  scale <- object$scale %||% rep(1, nrow(loadings))
  scale[!is.finite(scale) | scale == 0] <- 1
  projected <- sweep(newdata, 2L, center, "-")
  projected <- sweep(projected, 2L, scale, "/")
  scores <- projected %*% loadings[, seq_len(k), drop = FALSE]
  colnames(scores) <- paste0("PC", seq_len(k))
  rownames(scores) <- rownames(newdata)
  scores
}

#' Project new data with a fitted fastPLS PCA model
#'
#' Applies the centering, scaling, and loading matrix stored in a
#' `fastPLSPCA` object to an independent dataset.
#'
#' @param object A `fastPLSPCA` object returned by [pca()].
#' @param newdata Numeric matrix with the same columns, in the same order, as
#'   the matrix used to fit `object`.
#' @param ncomp Optional number of principal components to return. By default
#'   all components stored in `object` are returned.
#' @param ... Ignored.
#' @return Matrix of projected PCA scores for `newdata`.
#' @examples
#' pc <- pca(as.matrix(iris[, 1:4]), ncomp = 2, backend = "cpu",
#'           method = "rsvd", seed = 1)
#' predict(pc, as.matrix(iris[1:3, 1:4]))
#' @export
predict.fastPLSPCA <- function(object, newdata, ncomp = NULL, ...) {
  .predict_fastplspca_scores(object, newdata, ncomp = ncomp)
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
  graphics::legend("topright", legend = levels(groups), pt.bg = pal, col = "black", pch = dots$pch, bty = "n")
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
#' and a black contour in `col`; PCA plots use `pch = 22` unless another
#' plotting character is supplied through `...`.
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
  if (is.null(dots$pch)) dots$pch <- 22
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

#' Plot PLS permutation-test R2 and Q2 values
#'
#' Draws the permutation-test diagnostic plot produced by `pls(...,
#' perm.test = TRUE)`. The x-axis is the correlation between the original and
#' permuted response structure; the y-axis is the observed or permuted R2/Q2
#' value. R2 is shown in blue and Q2 in red.
#'
#' @param x A `fastPLS` model fitted with `perm.test = TRUE`, or a permutation
#'   data frame stored in `model$permutation`.
#' @param ncomp Component count to plot. Defaults to the largest component
#'   stored in the permutation table.
#' @param main,xlab,ylab Plot title and axis labels.
#' @param col,pch Colors and point symbols for R2 and Q2.
#' @param legend_position Legend position passed to [legend()].
#' @param ... Additional graphical parameters passed to [plot()].
#' @return Invisibly returns the plotted permutation data.
#' @examples
#' set.seed(1)
#' X <- as.matrix(iris[, 1:4])
#' y <- iris$Sepal.Length
#' idx <- sample(seq_len(nrow(X)), 30)
#' fit <- pls(X[idx, ], y[idx], X[idx, ], y[idx],
#'            ncomp = 2, perm.test = TRUE, times = 5)
#' plot.permutation(fit)
#' @export
plot.permutation <- function(x,
                             ncomp = NULL,
                             main = NULL,
                             xlab = "Cor",
                             ylab = "Value",
                             col = c(R2 = "#3155B7", Q2 = "#E5332A"),
                             pch = c(R2 = 16, Q2 = 15),
                             legend_position = "bottomright",
                             ...) {
  perm <- if (is.data.frame(x)) {
    x
  } else {
    x$permutation
  }
  if (is.null(perm) || !is.data.frame(perm) || !nrow(perm)) {
    stop("No permutation table found. Refit pls() with perm.test = TRUE.", call. = FALSE)
  }
  required <- c("type", "ncomp", "metric", "cor", "value")
  missing_cols <- setdiff(required, names(perm))
  if (length(missing_cols)) {
    stop("Permutation table is missing required columns: ",
         paste(missing_cols, collapse = ", "), call. = FALSE)
  }
  if (is.null(ncomp)) {
    ncomp <- max(perm$ncomp, na.rm = TRUE)
  }
  ncomp <- as.integer(ncomp)[1L]
  dat <- perm[perm$ncomp == ncomp & perm$metric %in% c("R2", "Q2"), , drop = FALSE]
  dat <- dat[is.finite(dat$cor) & is.finite(dat$value), , drop = FALSE]
  if (!nrow(dat)) {
    stop("No finite permutation values available for ncomp = ", ncomp, ".", call. = FALSE)
  }
  if (is.null(main)) {
    main <- paste("Permutation test, ncomp =", ncomp)
  }
  xlim <- range(c(0, 1, dat$cor), finite = TRUE)
  ylim <- range(dat$value, finite = TRUE)
  pad <- diff(ylim) * 0.08
  if (!is.finite(pad) || pad == 0) pad <- 0.1
  ylim <- ylim + c(-pad, pad)
  graphics::plot(
    dat$cor,
    dat$value,
    type = "n",
    xlim = xlim,
    ylim = ylim,
    xlab = xlab,
    ylab = ylab,
    main = main,
    ...
  )
  for (metric in c("R2", "Q2")) {
    d <- dat[dat$metric == metric & dat$type == "permutation", , drop = FALSE]
    if (nrow(d)) {
      graphics::points(d$cor, d$value, col = col[[metric]], pch = pch[[metric]])
    }
    obs <- dat[dat$metric == metric & dat$type == "observed", , drop = FALSE]
    if (nrow(obs)) {
      graphics::points(obs$cor, obs$value, col = col[[metric]], pch = pch[[metric]], cex = 1.3)
      if (nrow(d)) {
        graphics::segments(
          x0 = mean(d$cor, na.rm = TRUE),
          y0 = mean(d$value, na.rm = TRUE),
          x1 = obs$cor[[1L]],
          y1 = obs$value[[1L]],
          col = col[[metric]],
          lty = 2
        )
      }
    }
  }
  graphics::legend(
    legend_position,
    legend = c("R2", "Q2"),
    col = col[c("R2", "Q2")],
    pch = pch[c("R2", "Q2")],
    bty = "o"
  )
  invisible(dat)
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

  use_xprod <- .should_use_xprod_default(p, m, ncomp)
  s <- if (isTRUE(use_xprod)) {
    .truncated_rsvd_metal_xprod(
      Xtrain,
      Yc,
      max_ncomp_eff,
      rsvd_oversample = rsvd_oversample,
      rsvd_power = rsvd_power,
      seed = seed
    )
  } else {
    S <- .metal_crossprod(Xtrain, Yc)
    .truncated_rsvd_metal(
      S,
      max_ncomp_eff,
      rsvd_oversample = rsvd_oversample,
      rsvd_power = rsvd_power,
      seed = seed
    )
  }
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
    svd.method = "metal_rsvd",
    xprod_default = use_xprod,
    xprod_mode = if (isTRUE(use_xprod)) "metal_implicit" else "materialized"
  )
  if (store_B) {
    out$B <- B
  }
  out <- .annotate_coefficient_storage(out, store_B)
  class(out) <- "fastPLS"
  out <- .attach_backend_control(out)
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
    as.integer(max(1L, rsvd_power)),
    as.integer(seed)
  )
  RR <- native$R
  QQ <- native$Q
  if (is.null(RR) || is.null(QQ) || !length(RR) || !length(QQ)) {
    stop("SIMPLS Metal did not return any latent components.", call. = FALSE)
  }
  active <- which(colSums(abs(RR)) > 0 & colSums(abs(QQ)) > 0)
  if (!length(active)) {
    stop("SIMPLS Metal did not return any non-zero latent components.", call. = FALSE)
  }
  max_ncomp_eff <- min(max(active), ncol(RR), ncol(QQ))
  RR <- RR[, seq_len(max_ncomp_eff), drop = FALSE]
  QQ <- QQ[, seq_len(max_ncomp_eff), drop = FALSE]
  ncomp <- pmin(ncomp, max_ncomp_eff)
  store_B <- .should_store_coefficients(p, m, length_ncomp, TRUE)
  B <- if (store_B) array(0, dim = c(p, m, length_ncomp)) else NULL
  Yfit <- if (fit) array(0, dim = c(n, m, length_ncomp)) else NULL
  R2Y <- rep(NA_real_, length_ncomp)
  Tfull <- if (fit) .metal_mm(Xtrain, RR) else NULL
  for (i in seq_len(length_ncomp)) {
    mc <- max(1L, min(ncomp[i], max_ncomp_eff))
    if (store_B) {
      B[, , i] <- RR[, seq_len(mc), drop = FALSE] %*% t(QQ[, seq_len(mc), drop = FALSE])
    }
    if (fit) {
      yf <- .metal_mm(Tfull[, seq_len(mc), drop = FALSE], t(QQ[, seq_len(mc), drop = FALSE]))
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
    svd.method = "metal_resident_simpls",
    compact_prediction = !isTRUE(store_B)
  )
  if (store_B) {
    out$B <- B
  }
  out <- .annotate_coefficient_storage(out, store_B)
  class(out) <- "fastPLS"
  out <- .attach_backend_control(out)
  out
}

.pls_predict_metal <- function(object, Xtest, proj = FALSE) {
  Xscaled <- .fastpls_preprocess_test(Xtest, object$mX, object$vX)
  ncomp <- as.integer(object$ncomp)
  ns <- length(ncomp)
  n <- nrow(Xscaled)
  m <- as.integer(object$m)
  Ypred <- array(0, dim = c(n, m, ns))
  B_obj <- object[["B", exact = TRUE]]

  Tfull <- NULL
  if (!is.null(object$R) && length(object$R) > 0L) {
    maxc <- min(max(ncomp), ncol(object$R))
    Tfull <- .metal_mm(Xscaled, object$R[, seq_len(maxc), drop = FALSE])
  }

  for (i in seq_len(ns)) {
    mc <- min(ncomp[i], if (!is.null(Tfull)) ncol(Tfull) else dim(B_obj)[3L])
    if (!is.null(object$W_latent) && !is.null(Tfull)) {
      W <- matrix(object$W_latent[seq_len(mc), , i], nrow = mc, ncol = m)
      y <- .metal_mm(Tfull[, seq_len(mc), drop = FALSE], W)
    } else if (!is.null(B_obj)) {
      B_i <- matrix(B_obj[, , i], nrow = object$p, ncol = object$m)
      y <- .metal_mm(Xscaled, B_i)
    } else if (!is.null(object$R) && !is.null(object$Q) && !is.null(Tfull)) {
      y <- .metal_mm(
        Tfull[, seq_len(mc), drop = FALSE],
        t(object$Q[, seq_len(mc), drop = FALSE])
      )
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
  model$classification <- yprep$classification
  model$lev <- yprep$lev
  model <- .attach_lda_classifier(model, Xtrain, Ytrain_original, classifier, lda_ridge)
  model <- .maybe_attach_pls_variance_explained(model, Xtrain, return_variance)
  class(model) <- "fastPLS"
  if (!is.null(Xtest)) {
    res <- predict(model, Xtest, Ytest = Ytest, proj = proj)
    model <- c(model, res)
    class(model) <- "fastPLS"
  }
  model <- .attach_backend_control(model)
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
  yprep <- .prepare_response(Ytrain)
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
    out <- .attach_backend_control(out)
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
    out <- .attach_backend_control(out)
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
#' Fits PLSSVD, SIMPLS, OPLS, or kernel PLS models for regression or
#' classification using a selected CPU, CUDA, or Metal backend. The fitted model
#' can include predictions for held-out samples, latent scores, fitted values,
#' variance summaries, and optional classification heads.
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
#' @param classifier Classification decision rule. \code{argmax} keeps the
#'   standard PLS-DA response-score argmax. \code{lda} fits an LDA classifier on
#'   the PLS latent scores. \code{cknn} is the compact name for the PLS-score
#'   candidate-kNN classifier: class centroids in PLS score space choose
#'   candidate classes, then every sample is reranked by within-candidate kNN in
#'   the same PLS score space. The compiled implementation is selected automatically from
#'   `backend`: C++ for \code{cpu}, CUDA for \code{cuda}, and Metal for \code{metal}
#'   where available.
#' @param k Number of same-class PLS-score neighbours used by
#'   the candidate-kNN classifier.
#' @param tau Positive temperature used to smooth the neighbour
#'   similarities in candidate-kNN scoring.
#' @param alpha Weight of the centroid/prototype candidate score
#'   added to the local kNN score.
#' @param top_m Number of centroid-ranked candidate classes passed to
#'   the kNN reranker.
#' @param cknn_memory Memory strategy for \code{classifier = "cknn"}.
#'   \code{standard} uses the historical one-pass candidate-kNN path.
#'   \code{blocked} predicts test samples in blocks, reducing test-side
#'   latent-score memory. \code{streaming} additionally builds the training
#'   candidate-score cache in blocks for scalar component counts. \code{auto}
#'   chooses a memory-aware strategy from the data size.
#' @param lda_ridge Relative diagonal ridge added to the pooled LDA covariance.
#' @param fit Return fitted values and `R2Y` when `TRUE`.
#' @param return_variance Compute predictor-space latent-variable variance
#'   explained. Set to `FALSE` for timing/memory benchmarks that do not need
#'   plotting variance metadata.
#' @param proj Return projected `Ttest` when `TRUE`.
#' @param perm.test Run a single-split permutation test when `Xtest` and
#'   `Ytest` are supplied. The rows of `Xtrain` are randomly permuted, the
#'   model is refitted, and the permuted test-set `Q2Y` path is compared with
#'   the observed `Q2Y` path.
#' @param times Number of permutations. For `pls()`, the empirical p-value for
#'   each component is `mean(Q2Y_permuted > Q2Y_observed)` and is returned in
#'   `pval`. No +1 correction is applied.
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
#' @return A `fastPLS` object. The object is a list whose fields depend on the
#'   selected method, backend, classifier, and whether test data or optional
#'   summaries were requested. Common fields are:
#'
#'   * `P`: predictor loadings, with one column per latent component.
#'   * `Q`: response loadings or response-side latent coefficients.
#'   * `R`: predictor weights/rotations used to project new samples into the PLS
#'     latent space.
#'   * `Ttrain`: training latent scores. This is returned when the backend stores
#'     scores or when they are needed for fitted values, classifiers, variance
#'     summaries, or compact prediction.
#'   * `C_latent`, `W_latent`: low-rank latent prediction factors used by
#'     PLSSVD-style compact prediction when a full coefficient array is avoided.
#'   * `B`: regression coefficient matrix or coefficient array, when stored.
#'     For vector-valued `ncomp`, a three-dimensional array may contain the
#'     coefficient path for all requested component counts.
#'   * `mX`, `vX`: training predictor centering and scaling values. `vX` is one
#'     when no scaling is applied.
#'   * `mY`: response centering values for regression or dummy-coded PLS-DA.
#'   * `p`, `m`: number of predictor and response columns used internally.
#'   * `ncomp`: requested/effective component count vector. For PLSSVD this may
#'     be capped by the numerical rank of the response.
#'   * `lev`: factor levels used for classification.
#'   * `classification_rule`: classification head used for factor responses:
#'     `"argmax"`, `"lda"`, or `"cknn"` internally resolved to the selected
#'     backend.
#'   * `lda_backend`: backend used by the latent-space LDA classifier, when
#'     `classifier = "lda"`.
#'   * `Yfit`: fitted training responses or fitted class labels, returned when
#'     `fit = TRUE`.
#'   * `R2Y`: training-set coefficient of determination path when `fit = TRUE`;
#'     otherwise `NA` placeholders may be returned for compatibility. Elements
#'     are named by component count, for example `"ncomp=2"`.
#'   * `Ypred`: predictions for `Xtest`, returned only when `Xtest` is supplied
#'     to `pls()`. For classification this contains predicted factor labels; for
#'     regression it contains numeric predictions.
#'   * `Ypred_index`: integer class indices for classification predictions, when
#'     available.
#'   * `Ttest`: test-set latent scores, returned when `proj = TRUE`.
#'   * `Q2Y`: test-set Q2 for numeric `Ytest`, or dummy-response PLS-DA Q2 for
#'     factor `Ytest`, returned when response scores are available. Elements are
#'     named by component count.
#'   * `accuracy`: decoded-label accuracy for factor `Ytest`, returned when
#'     classification predictions are available. Elements are named by component
#'     count.
#'   * `pval`: single-split permutation-test p-values by component, returned
#'     when `perm.test = TRUE`. Each p-value is the fraction of permuted
#'     `Q2Y` values larger than the observed `Q2Y`.
#'   * `permutation`: long-format permutation table, returned when
#'     `perm.test = TRUE`, with observed and permuted `R2`/`Q2` values and the
#'     permutation correlation used by `plot.permutation()`.
#'   * `variance`, `variance_explained`, `cumulative_variance_explained`,
#'     `variance_total`, `variance_basis`: predictor-space variance summaries
#'     returned when `return_variance = TRUE`.
#'   * `x_variance`, `x_variance_explained`,
#'     `x_cumulative_variance_explained`, `x_variance_total`: aliases of the
#'     predictor-space variance summaries.
#'   * `inner_model`: fitted inner PLS model used by OPLS.
#'   * `W_orth`, `P_orth`, `north`, `opls_engine`, `xprod_mode`,
#'     `gpu_resident`: OPLS-specific orthogonal-component and backend metadata.
#'   * `kernel`, `kernel_engine`, `kernel_linear_direct`: kernelPLS-specific
#'     kernel settings and execution metadata.
#' @examples
#' X <- as.matrix(mtcars[, c("disp", "hp", "wt", "qsec")])
#' y <- mtcars$mpg
#' fit <- pls(X, y, ncomp = 2, method = "simpls", backend = "cpu",
#'            svd.method = "rsvd", return_variance = FALSE)
#' head(predict(fit, X)$Ypred)
#'
#' cv <- pls.single.cv(X, y, ncomp = 1:2, kfold = 3, method = "simpls",
#'                     backend = "cpu", svd.method = "rsvd", seed = 1)
#' fit_cv <- pls(cv, Xtest = X, return_variance = FALSE)
#' cv$best_ncomp
#' head(fit_cv$Ypred)
#' @export
pls =  function (Xtrain,
                 Ytrain,
                 Xtest = NULL,
                 Ytest = NULL,
                 ncomp=2,
                 scaling = c("centering", "autoscaling","none"),
                 method = c("simpls", "plssvd", "opls", "kernelpls"),
                 svd.method = c("rsvd", "irlba"),
		                 classifier = c("argmax", "lda", "cknn"),
		                 lda_ridge = 1e-8,
			                 k = 10L,
			                 tau = 0.2,
			                 alpha = 0.75,
			                 top_m = 20L,
                         cknn_memory = c("auto", "standard", "blocked", "streaming"),
	                 fit = FALSE,
                 return_variance = TRUE,
                 return_loadings = FALSE,
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
  if (.is_single_pls_cv_result(Xtrain)) {
    cv_Xtest <- if (!missing(Ytrain) && missing(Xtest)) {
      Ytrain
    } else if (missing(Xtest)) {
      NULL
    } else {
      Xtest
    }
    cv_Ytest <- if (missing(Ytest)) NULL else Ytest
    return(.pls_from_single_cv_result(
      cv = Xtrain,
      Xtest = cv_Xtest,
      Ytest = cv_Ytest,
      fit = fit,
      return_variance = return_variance,
      return_loadings = return_loadings,
      proj = proj,
      perm.test = perm.test,
      times = times
    ))
  }

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
		  k <- max(1L, as.integer(k)[1L])
		  tau <- as.numeric(tau)[1L]
		  alpha <- as.numeric(alpha)[1L]
		  top_m <- max(1L, as.integer(top_m)[1L])
      cknn_memory <- .normalize_cknn_memory(cknn_memory)
		  if (!is.finite(tau) || tau <= 0) {
		    stop("tau must be a finite positive number", call. = FALSE)
		  }
		  if (!is.finite(alpha)) {
		    stop("alpha must be finite", call. = FALSE)
		  }
		  old_class_bias_options <- options(
		    fastPLS.k = k,
		    fastPLS.tau = tau,
		    fastPLS.alpha = alpha,
		    fastPLS.top_m = top_m,
        fastPLS.cknn_memory = cknn_memory
	  )
	  on.exit(options(old_class_bias_options), add = TRUE)

  backend_control <- NULL

  if (identical(backend, "metal")) {
    model <- .pls_metal(
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
      classifier = classifier,
      lda_ridge = lda_ridge,
      fit = fit,
      return_variance = return_variance,
      proj = proj
    )
    model <- .maybe_attach_x_loadings(model, Xtrain, return_loadings)
    model <- .attach_backend_control(model, backend_control)
    return(.fastpls_public_pls_output(model, model$ncomp))
  }

  if (identical(requested_method, "opls")) {
    fit_fun <- switch(backend_compiled, cpp = .opls_cpp, cuda = .opls_cuda)
    args <- list(
      Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest,
      ncomp = ncomp, north = north, scaling = scaling,
      rsvd_oversample = rsvd_oversample, rsvd_power = rsvd_power,
      svds_tol = svds_tol, seed = seed,
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
    model <- do.call(fit_fun, args)
    model <- .maybe_attach_x_loadings(model, Xtrain, return_loadings)
    model <- .attach_backend_control(model, backend_control)
    return(.fastpls_public_pls_output(model, model$ncomp))
  }

  if (identical(requested_method, "kernelpls")) {
    fit_fun <- switch(backend_compiled, cpp = .kernel_pls_cpp, cuda = .kernel_pls_cuda)
    args <- list(
      Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest,
      ncomp = ncomp, scaling = scaling, kernel = kernel, gamma = gamma,
      degree = degree, coef0 = coef0,
      rsvd_oversample = rsvd_oversample, rsvd_power = rsvd_power,
      svds_tol = svds_tol, seed = seed,
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
    model <- do.call(fit_fun, args)
    model <- .maybe_attach_x_loadings(model, Xtrain, return_loadings)
    model <- .attach_backend_control(model, backend_control)
    return(.fastpls_public_pls_output(model, model$ncomp))
  }

  if (identical(backend, "cuda")) {
    if (identical(requested_method, "plssvd")) {
      model <- .plssvd_gpu(
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
	      classifier = classifier,
	      lda_ridge = lda_ridge,
            return_variance = return_variance
	      )
      model <- .maybe_attach_x_loadings(model, Xtrain, return_loadings)
      model <- .attach_backend_control(model, backend_control)
      return(.fastpls_public_pls_output(model, model$ncomp))
    }
    model <- .simpls_gpu(
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
	      classifier = classifier,
	      lda_ridge = lda_ridge,
          return_variance = return_variance
	    )
    model <- .maybe_attach_x_loadings(model, Xtrain, return_loadings)
    model <- .attach_backend_control(model, backend_control)
    return(.fastpls_public_pls_output(model, model$ncomp))
  }

  meth = .normalize_pls_method(requested_method)
  svd.method <- .normalize_svd_method(svd.method)
  svd.method <- match.arg(svd.method, c("irlba", "cpu_rsvd"))
  svdmeth <- .svd_method_id(svd.method)

  Xtrain = as.matrix(Xtrain)
  Ytrain_original <- Ytrain
  yprep <- .prepare_response(Ytrain)
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
  model$classification=classification
  model$lev=lev
	  model <- .attach_lda_classifier(
	    model,
	    Xtrain,
	    Ytrain_original,
	    classifier,
	    lda_ridge
		  )
  model <- .maybe_attach_pls_variance_explained(model, Xtrain, return_variance)
  model <- .maybe_attach_x_loadings(model, Xtrain, return_loadings)
  if (!isTRUE(fit) && !is.null(model$R2Y)) {
    model$R2Y <- rep(NA_real_, length(ncomp))
  }


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
        r2_perm = matrix(NA_real_, nrow = times, ncol = length(ncomp))
        cor_perm = rep(NA_real_, times)
        for (i in 1:times) {
          ss = sample(1:nrow(Xtrain))
          Xtrain_permuted = Xtrain[ss, ]
          cor_perm[[i]] <- .fastpls_permutation_cor(Ytrain, ss)

          if(meth==1){
            model_perm=pls.model1(
              Xtrain_permuted,
              Ytrain,
              ncomp=ncomp,
              fit=TRUE,
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
              fit=TRUE,
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
              fit=TRUE,
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

          model_perm$classification <- classification
          model_perm$lev <- lev
          if (!is.null(model_perm$R2Y)) {
            r2_perm[i, ] <- as.numeric(model_perm$R2Y)
          }
          res_perm=predict(model_perm,Xtest,Ytest)

          v[i,]=res_perm$Q2Y
        }
        model$pval=NULL
        for(j in 1:length(ncomp)){
          model$pval[j] = sum(v[,j] > model$Q2Y)/times
        }
        perm_df <- data.frame(
          type = rep("permutation", times * length(ncomp) * 2L),
          permutation = rep(seq_len(times), times = length(ncomp) * 2L),
          ncomp = rep(rep(as.integer(ncomp), each = times), times = 2L),
          metric = rep(c("R2", "Q2"), each = times * length(ncomp)),
          cor = rep(rep(cor_perm, times = length(ncomp)), times = 2L),
          value = c(as.numeric(r2_perm), as.numeric(v)),
          stringsAsFactors = FALSE
        )
        obs_df <- data.frame(
          type = "observed",
          permutation = NA_integer_,
          ncomp = rep(as.integer(ncomp), times = 2L),
          metric = rep(c("R2", "Q2"), each = length(ncomp)),
          cor = 1,
          value = c(as.numeric(model$R2Y), as.numeric(model$Q2Y)),
          stringsAsFactors = FALSE
        )
        model$permutation <- rbind(perm_df, obs_df)


      }
  }
    if(classification){

      if(fit){
        train_model <- model
        class(train_model) <- "fastPLS"
        model$Yfit <- predict.fastPLS(train_model, Xtrain)$Ypred
      }
    }



  class(model)="fastPLS"
  model <- .attach_backend_control(model, backend_control)
  .fastpls_public_pls_output(model, model$ncomp)
}

.cv_best_index <- function(metrics, selection_metric = "auto") {
  selection_metric <- .cv_normalize_selection_metric(selection_metric)
  values <- as.numeric(metrics$metric_value)
  metric_names <- tolower(as.character(metrics$metric_name))
  finite <- is.finite(values)
  if (!any(finite)) {
    return(1L)
  }
  if (!identical(selection_metric, "auto")) {
    target_names <- switch(
      selection_metric,
      accuracy = "accuracy",
      r2 = c("r2", "q2"),
      q2 = c("q2", "r2"),
      rmsd = c("rmsd", "rmse"),
      character(0)
    )
    finite <- finite & metric_names %in% target_names
    if (!any(finite)) {
      stop(
        sprintf(
          "selection_metric = '%s' is not available in these CV results. Available metrics: %s.",
          selection_metric,
          paste(unique(metric_names), collapse = ", ")
        ),
        call. = FALSE
      )
    }
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
  class_pred <- cv_res[["class_pred", exact = TRUE]]
  score_pred <- cv_res[["Ypred", exact = TRUE]]
  pred <- cv_res[["pred", exact = TRUE]]
  if (!is.null(class_pred)) {
    return(pred[[idx]])
  }
  if (!is.null(score_pred)) {
    return(score_pred[, , idx, drop = FALSE])
  }
  if (is.null(pred) || length(pred) < idx) {
    return(NULL)
  }
  pred[[idx]]
}

.cv_metric_name_at <- function(metrics, idx) {
  as.character(metrics$metric_name[[idx]])
}

.cv_metric_id <- function(metric, classification) {
  metric <- .cv_normalize_selection_metric(metric)
  if (isTRUE(classification)) {
    return(1L)
  }
  switch(
    metric,
    r2 = 2L,
    q2 = 3L,
    rmsd = 4L,
    auto = 4L,
    4L
  )
}

.cv_grid_choice_values <- function(value,
                                   missing_arg,
                                   choices,
                                   default = choices[[1L]],
                                   name = "argument",
                                   normalizer = NULL) {
  if (isTRUE(missing_arg)) {
    value <- default
  }
  value <- as.character(value)
  if (!length(value)) {
    stop(name, " must contain at least one value.", call. = FALSE)
  }
  if (!is.null(normalizer)) {
    value <- vapply(value, normalizer, character(1L), USE.NAMES = FALSE)
  }
  bad <- setdiff(value, choices)
  if (length(bad)) {
    stop(
      sprintf(
        "%s must use values from: %s. Invalid: %s.",
        name,
        paste(choices, collapse = ", "),
        paste(bad, collapse = ", ")
      ),
      call. = FALSE
    )
  }
  as.list(unique(value))
}

.cv_grid_scalar_values <- function(value,
                                   missing_arg = FALSE,
                                   default = NULL,
                                   name = "argument",
                                   cast = identity,
                                   allow_null = TRUE) {
  if (isTRUE(missing_arg)) {
    value <- default
  }
  if (is.null(value)) {
    if (!allow_null) {
      stop(name, " cannot be NULL.", call. = FALSE)
    }
    return(list(NULL))
  }
  if (!length(value)) {
    stop(name, " must contain at least one value.", call. = FALSE)
  }
  value <- cast(value)
  as.list(unique(value))
}

.cv_expand_prediction_grid <- function(params) {
  lens <- vapply(params, length, integer(1L))
  if (!length(lens) || any(lens < 1L)) {
    return(list(list()))
  }
  idx <- expand.grid(
    lapply(lens, seq_len),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  names(idx) <- names(params)
  lapply(seq_len(nrow(idx)), function(i) {
    cfg <- vector("list", length(params))
    names(cfg) <- names(params)
    for (nm in names(params)) {
      cfg[[nm]] <- params[[nm]][[idx[[nm]][[i]]]]
    }
    cfg
  })
}

.cv_normalize_svd_grid_dots <- function(dots, context) {
  if (!is.list(dots)) {
    dots <- list()
  }
  .normalize_svd_parameter_list(
    dots,
    accepted = names(.svd_control_defaults()),
    aliases = .svd_direct_aliases(),
    label = sprintf("... in %s", context)
  )
}

.cv_canonicalize_prediction_config <- function(cfg) {
  if (!identical(cfg$method, "kernelpls")) {
    cfg$kernel <- "linear"
    cfg["gamma"] <- list(NULL)
    cfg$degree <- 3L
    cfg$coef0 <- 1
  } else if (identical(cfg$kernel, "linear")) {
    cfg["gamma"] <- list(NULL)
    cfg$degree <- 3L
    cfg$coef0 <- 1
  } else if (identical(cfg$kernel, "rbf")) {
    cfg$degree <- 3L
    cfg$coef0 <- 1
  }
  if (!identical(cfg$method, "opls")) {
    cfg$north <- 1L
  }
  if (!identical(cfg$classifier, "cknn")) {
    cfg$k <- 10L
    cfg$tau <- 0.2
    cfg$alpha <- 0.75
    cfg$top_m <- 20L
    cfg$cknn_memory <- "auto"
  }
  if (!identical(cfg$classifier, "lda")) {
    cfg$lda_ridge <- 1e-8
  }
  cfg
}

.cv_config_key <- function(cfg) {
  rec <- .cv_config_record(cfg)
  paste(names(rec), vapply(rec, function(x) as.character(x[[1L]]), character(1L)), sep = "=", collapse = "|")
}

.cv_make_prediction_grid <- function(scaling, scaling_missing,
                                     method, method_missing,
                                     backend, backend_missing,
                                     svd.method, svd_missing,
                                     north,
                                     kernel, kernel_missing,
                                     gamma,
                                     degree,
                                     coef0,
                                     classifier, classifier_missing,
                                     lda_ridge,
                                     k,
                                     tau,
                                     alpha,
                                     top_m,
                                     cknn_memory,
                                     cknn_memory_missing,
                                     xprod,
                                     dots = list(),
                                     context = "cross-validation") {
  classifier_normalizer <- function(x) {
    x <- as.character(x)
    if (identical(x, "candidate_knn")) x <- "cknn"
    x
  }
  svd_normalizer <- function(x) {
    x <- as.character(x)
    if (identical(x, "rsvd")) x <- "cpu_rsvd"
    x
  }
  dots <- .cv_normalize_svd_grid_dots(dots, context = context)
  dots <- dots[setdiff(names(dots), "seed")]
  dot_params <- lapply(names(dots), function(nm) {
    .cv_grid_scalar_values(dots[[nm]], name = nm, allow_null = FALSE)
  })
  names(dot_params) <- names(dots)

  params <- c(
    list(
      scaling = .cv_grid_choice_values(
        scaling, scaling_missing,
        choices = c("centering", "autoscaling", "none"),
        default = "centering",
        name = "scaling"
      ),
      method = .cv_grid_choice_values(
        method, method_missing,
        choices = c("simpls", "plssvd", "opls", "kernelpls"),
        default = "simpls",
        name = "method"
      ),
      backend = .cv_grid_choice_values(
        backend, backend_missing,
        choices = c("cpu", "cuda", "metal"),
        default = "cpu",
        name = "backend",
        normalizer = .normalize_public_backend
      ),
      svd.method = .cv_grid_choice_values(
        svd.method, svd_missing,
        choices = c("irlba", "cpu_rsvd"),
        default = "irlba",
        name = "svd.method",
        normalizer = svd_normalizer
      ),
      north = .cv_grid_scalar_values(north, name = "north", cast = as.integer, allow_null = FALSE),
      kernel = .cv_grid_choice_values(
        kernel, kernel_missing,
        choices = c("linear", "rbf", "poly"),
        default = "linear",
        name = "kernel"
      ),
      gamma = .cv_grid_scalar_values(gamma, name = "gamma"),
      degree = .cv_grid_scalar_values(degree, name = "degree", cast = as.integer, allow_null = FALSE),
      coef0 = .cv_grid_scalar_values(coef0, name = "coef0", cast = as.numeric, allow_null = FALSE),
      classifier = .cv_grid_choice_values(
        classifier, classifier_missing,
        choices = .classifier_public_choices,
        default = "argmax",
        name = "classifier",
        normalizer = classifier_normalizer
      ),
      lda_ridge = .cv_grid_scalar_values(lda_ridge, name = "lda_ridge", cast = as.numeric, allow_null = FALSE),
      k = .cv_grid_scalar_values(k, name = "k", cast = as.integer, allow_null = FALSE),
      tau = .cv_grid_scalar_values(tau, name = "tau", cast = as.numeric, allow_null = FALSE),
      alpha = .cv_grid_scalar_values(alpha, name = "alpha", cast = as.numeric, allow_null = FALSE),
      top_m = .cv_grid_scalar_values(top_m, name = "top_m", cast = as.integer, allow_null = FALSE),
      cknn_memory = .cv_grid_choice_values(
        cknn_memory, cknn_memory_missing,
        choices = c("auto", "standard", "blocked", "streaming"),
        default = "auto",
        name = "cknn_memory",
        normalizer = .normalize_cknn_memory
      ),
      xprod = .cv_grid_scalar_values(xprod, name = "xprod")
    ),
    dot_params
  )

  configs <- .cv_expand_prediction_grid(params)
  dot_names <- names(dot_params)
  configs <- lapply(configs, function(cfg) {
    cfg$svd_dots <- cfg[dot_names]
    cfg[dot_names] <- NULL
    .cv_canonicalize_prediction_config(cfg)
  })
  configs[!duplicated(vapply(configs, .cv_config_key, character(1L)))]
}

.cv_config_record <- function(cfg) {
  svd_dots <- cfg$svd_dots
  cfg <- cfg[setdiff(names(cfg), "svd_dots")]
  if (length(svd_dots)) {
    cfg <- c(cfg, svd_dots)
  }
  as.data.frame(
    lapply(cfg, function(x) {
      if (is.null(x) || !length(x)) {
        return(NA)
      }
      x[[1L]]
    }),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

.cv_config_list <- function(cfg) {
  rec <- .cv_config_record(cfg)
  as.list(rec[1L, , drop = FALSE])
}

.cv_prune_config_for_output <- function(cfg) {
  keep <- c("scaling", "method", "backend", "svd.method", "classifier", "xprod")
  if (identical(cfg$method, "opls")) {
    keep <- c(keep, "north")
  }
  if (identical(cfg$method, "kernelpls")) {
    keep <- c(keep, "kernel")
    if (identical(cfg$kernel, "rbf")) {
      keep <- c(keep, "gamma")
    } else if (identical(cfg$kernel, "poly")) {
      keep <- c(keep, "gamma", "degree", "coef0")
    }
  }
  if (identical(cfg$classifier, "lda")) {
    keep <- c(keep, "lda_ridge")
  } else if (identical(cfg$classifier, "cknn")) {
    keep <- c(keep, "k", "tau", "alpha", "top_m", "cknn_memory")
  }
  keep <- intersect(unique(keep), names(cfg))
  out <- cfg[keep]
  if (!is.null(cfg$svd_dots) && length(cfg$svd_dots)) {
    out$svd_dots <- cfg$svd_dots
  }
  out
}

.cv_varied_parameter_names <- function(configs) {
  if (length(configs) <= 1L) {
    return(character(0))
  }
  recs <- do.call(rbind, lapply(configs, .cv_config_record))
  keep <- vapply(recs, function(x) {
    x <- x[!is.na(x)]
    length(unique(as.character(x))) > 1L
  }, logical(1L))
  names(recs)[keep]
}

.cv_selected_parameters <- function(cfg, configs, best_ncomp) {
  full <- .cv_config_list(cfg)
  varied <- .cv_varied_parameter_names(configs)
  selected <- full[intersect(varied, names(full))]
  c(list(ncomp = as.integer(best_ncomp[[1L]])), selected)
}

.cv_select_best_result_from_grid <- function(results,
                                             summaries,
                                             metrics,
                                             selection_metric = "auto") {
  ok <- vapply(results, function(x) is.list(x) && identical(x$status, "ok"), logical(1L))
  if (!any(ok)) {
    stop(
      "All CV tuning configurations failed. First errors: ",
      paste(head(summaries$error[!is.na(summaries$error)], 5L), collapse = " | "),
      call. = FALSE
    )
  }
  pick_df <- summaries[ok, , drop = FALSE]
  pick_idx <- .cv_best_index(
    data.frame(
      metric_name = pick_df$best_metric_name,
      metric_value = pick_df$best_metric_value,
      stringsAsFactors = FALSE
    ),
    selection_metric = selection_metric
  )
  best_grid_id <- pick_df$grid_id[[pick_idx]]
  best <- results[[best_grid_id]]
  best$tuning_results <- results
  best$tuning_summary <- summaries
  best$tuning_metrics <- metrics
  best$best_grid_id <- best_grid_id
  full_configs <- lapply(results, function(x) x$tuning_config_full %||% x$tuning_config)
  best_full_config <- results[[best_grid_id]]$tuning_config_full %||% results[[best_grid_id]]$tuning_config
  best$best_parameters <- .cv_selected_parameters(
    best_full_config,
    full_configs,
    best$best_ncomp
  )
  best$tuning_config_full <- NULL
  if (length(best$tuning_results)) {
    best$tuning_results <- lapply(best$tuning_results, function(x) {
      x$tuning_config_full <- NULL
      x
    })
  }
  best
}

.cv_value_or_default <- function(params, name, default) {
  if (!is.null(params) && name %in% names(params)) {
    value <- params[[name]]
    if (!is.null(value) && length(value) && !is.na(value[[1L]])) {
      return(value[[1L]])
    }
  }
  default
}

.cv_grid_arg_values <- function(configs, name) {
  vals <- lapply(configs, function(cfg) cfg[[name]])
  nonnull <- !vapply(vals, is.null, logical(1L))
  if (!any(nonnull)) {
    return(NULL)
  }
  unique(unlist(vals[nonnull], recursive = FALSE, use.names = FALSE))
}

.cv_grid_dot_values <- function(configs, name) {
  vals <- lapply(configs, function(cfg) cfg$svd_dots[[name]])
  nonnull <- !vapply(vals, is.null, logical(1L))
  if (!any(nonnull)) {
    return(NULL)
  }
  unique(unlist(vals[nonnull], recursive = FALSE, use.names = FALSE))
}

#' Single cross-validation for PLS component optimization
#'
#' Performs grouped k-fold or leave-one-out cross-validation over candidate
#' component counts and, when vector-valued predictive arguments are supplied,
#' over a compact hyperparameter grid. The selection can be based on
#' cross-validated accuracy, R2, Q2, or RMSD.
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
#' @param method One or more of \code{simpls}, \code{plssvd}, \code{opls}, or
#'   \code{kernelpls}. Multiple values are treated as a tuning grid.
#' @param backend Implementation backend: \code{cpu}, \code{cuda}, or
#'   \code{metal}. Multiple values are treated as a tuning grid.
#' @param seed Random seed used for fold assignment and randomized SVD steps.
#' @param gamma Kernel scale. Defaults internally to `1 / ncol(Xdata)`. For
#'   \code{method = "kernelpls"}, multiple values are treated as a tuning grid.
#' @param classifier Classification rule for factor responses: `"argmax"`,
#'   `"lda"`, or `"cknn"`. Multiple values are treated as a tuning grid.
#' @param lda_ridge Ridge added to the pooled LDA covariance diagonal. Multiple
#'   values are used only when `classifier = "lda"`.
#' @param k,tau,alpha,top_m
#'   Candidate-kNN controls used when `classifier = "cknn"`.
#' @param fit Fit one additional model on the full dataset and return its
#'   fitted values (`Yfit`) and training `R2Y` path. The default is `TRUE` for
#'   backward compatibility. Set to `FALSE` to skip this extra full-data fit;
#'   held-out cross-validated `Q2Y` and `RMSD` are still calculated.
#' @param xprod Use the matrix-free cross-product route where available.
#'   `NULL` applies fastPLS defaults.
#' @param ... Optional SVD tuning controls forwarded to the selected backend.
#'   Use the same compact names documented in [fastsvd()], such as
#'   `oversample`, `power`, `svds_tol`, `work`, `maxit`, `tol`, `eps`,
#'   and `svtol`. Vector values are included in the tuning grid. Component
#'   selection can also be controlled with `selection_metric = "auto"`,
#'   `"accuracy"`, `"r2"`, `"q2"`, or `"rmsd"`; the selection metric itself is
#'   scalar.
#' @return A list describing the cross-validation run and selected model:
#'   \itemize{
#'   \item `best_ncomp`: number of components selected by the chosen metric.
#'   \item `best_index`: position of `best_ncomp` in the tested component grid.
#'   \item `selection_metric`: metric used for optimization. With `"auto"`,
#'   classification uses accuracy and regression uses the default prediction
#'   error rule.
#'   \item `best_metric_name` and `best_metric_value`: name and value of the
#'   metric at the selected component count.
#'   \item `Q2Y`: held-out cross-validated Q2. For factor responses, Q2 is
#'   calculated on the dummy-coded PLS-DA response scores.
#'   \item `accuracy`: held-out decoded-label accuracy for factor responses.
#'   \item `RMSD`: held-out root mean squared deviation for regression. It is
#'   `NA` for classification.
#'   \item `Yfit`: fitted values from the full-data model when `fit = TRUE`.
#'   \item `R2Y`: training-set explained-variance path from a model fitted on
#'   the full dataset when `fit = TRUE`; otherwise `NA`. For factor
#'   responses, this is calculated on the dummy-coded PLS-DA response scores,
#'   not on the decoded class labels.
#'   \item `fold`: fold assignment used for each sample.
#'   \item `pred`: decoded cross-validated predictions when predictions are
#'   stored.
#'   \item `Ypred`: raw prediction array when score predictions are stored.
#'   \item `metrics`: per-component metric table returned by the CV backend.
#'   \item `best_parameters`: compact list containing only `ncomp` plus the
#'   arguments that were actually optimized, for example `classifier` when
#'   `classifier = c("argmax", "lda")`.
#'   \item `tuning_config`: relevant selected configuration used for the run.
#'   Irrelevant classifier- or method-specific defaults are omitted; for
#'   example, cKNN controls are not shown when `classifier = "argmax"`.
#'   \item `tuning_summary` and `tuning_metrics`: tables for all tested
#'   configurations when more than one predictive configuration is supplied.
#'   \item The returned object can be passed as the first argument to [pls()] to
#'   refit the selected model on the full training data and predict new samples.
#'   }
#' @examples
#' idx <- c(1:12, 51:62, 101:112)
#' X <- as.matrix(iris[idx, 1:4])
#' y <- factor(iris[idx, 5])
#' opt <- pls.single.cv(X, y, ncomp = 1:2, kfold = 3, method = "simpls",
#'                      backend = "cpu", svd.method = "rsvd", seed = 1)
#' opt$best_ncomp
#' opt_kernel <- pls.single.cv(X, y, ncomp = 1:2, kfold = 3,
#'                             method = "kernelpls", backend = "cpu",
#'                             svd.method = "rsvd",
#'                             kernel = c("linear", "rbf"),
#'                             gamma = c(0.1, 1), seed = 1)
#' opt_kernel$best_parameters
#' @export
pls.single.cv =  function (Xdata,
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
                          classifier = c("argmax", "lda", "cknn"),
                          lda_ridge = 1e-8,
                          k = 10L,
                          tau = 0.2,
                          alpha = 0.75,
                          top_m = 20L,
                          cknn_memory = c("auto", "standard", "blocked", "streaming"),
                          fit = TRUE,
                          xprod = NULL,
                          ...)
{
  if (sum(is.na(Xdata)) > 0) {
    stop("Missing values are present")
  }
  selection_ctl <- .cv_selection_metric_from_dots(list(...))
  tuning_grid <- .cv_make_prediction_grid(
    scaling = scaling,
    scaling_missing = missing(scaling),
    method = method,
    method_missing = missing(method),
    backend = backend,
    backend_missing = missing(backend),
    svd.method = svd.method,
    svd_missing = missing(svd.method),
    north = north,
    kernel = kernel,
    kernel_missing = missing(kernel),
    gamma = gamma,
    degree = degree,
    coef0 = coef0,
    classifier = classifier,
    classifier_missing = missing(classifier),
    lda_ridge = lda_ridge,
    k = k,
    tau = tau,
    alpha = alpha,
    top_m = top_m,
    cknn_memory = cknn_memory,
    cknn_memory_missing = missing(cknn_memory),
    xprod = xprod,
    dots = selection_ctl$dots,
    context = "pls.single.cv()"
  )
  if (length(tuning_grid) > 1L) {
    grid_results <- vector("list", length(tuning_grid))
    grid_summary <- vector("list", length(tuning_grid))
    grid_metrics <- vector("list", length(tuning_grid))
    for (grid_id in seq_along(tuning_grid)) {
      cfg <- tuning_grid[[grid_id]]
      rec <- .cv_config_record(cfg)
      run_args <- c(
        list(
          Xdata = Xdata,
          Ydata = Ydata,
          ncomp = ncomp,
          constrain = constrain,
          scaling = cfg$scaling,
          method = cfg$method,
          backend = cfg$backend,
          svd.method = cfg$svd.method,
          seed = seed,
          kfold = kfold,
          north = cfg$north,
          kernel = cfg$kernel,
          gamma = cfg$gamma,
          degree = cfg$degree,
          coef0 = cfg$coef0,
          classifier = cfg$classifier,
          lda_ridge = cfg$lda_ridge,
          k = cfg$k,
          tau = cfg$tau,
          alpha = cfg$alpha,
          top_m = cfg$top_m,
          cknn_memory = cfg$cknn_memory,
          fit = fit,
          xprod = cfg$xprod,
          selection_metric = selection_ctl$metric
        ),
        cfg$svd_dots
      )
      one <- tryCatch(
        do.call(pls.single.cv, run_args),
        error = function(e) {
          list(status = "error", error = conditionMessage(e), tuning_config = cfg)
        }
      )
      if (!identical(one$status, "error")) {
        one$cv_status <- one$status
        one$status <- "ok"
      }
      one <- .cv_drop_fit_data(one)
      one$tuning_config_full <- cfg
      one$tuning_config <- .cv_prune_config_for_output(cfg)
      grid_results[[grid_id]] <- one
      status <- if (identical(one$status, "ok")) "ok" else "error"
      err_msg <- if (identical(status, "ok")) NA_character_ else (one$error %||% "configuration failed")
      best_ncomp_val <- if (identical(status, "ok") && length(one$best_ncomp)) one$best_ncomp[[1L]] else NA_integer_
      best_metric_name_val <- if (identical(status, "ok") && length(one$best_metric_name)) one$best_metric_name[[1L]] else NA_character_
      best_metric_value_val <- if (identical(status, "ok") && length(one$best_metric_value)) one$best_metric_value[[1L]] else NA_real_
      grid_summary[[grid_id]] <- cbind(
        data.frame(
          grid_id = grid_id,
          status = status,
          best_ncomp = best_ncomp_val,
          best_metric_name = best_metric_name_val,
          best_metric_value = best_metric_value_val,
          error = err_msg,
          stringsAsFactors = FALSE
        ),
        rec
      )
      if (identical(status, "ok")) {
        metric_df <- one$selection_metrics
        metric_df$ncomp <- one$ncomp
        grid_metrics[[grid_id]] <- cbind(
          data.frame(grid_id = grid_id, stringsAsFactors = FALSE),
          rec[rep(1L, nrow(metric_df)), , drop = FALSE],
          metric_df
        )
      }
    }
    summaries <- do.call(rbind, grid_summary)
    metrics <- if (length(Filter(Negate(is.null), grid_metrics))) {
      do.call(rbind, Filter(Negate(is.null), grid_metrics))
    } else {
      data.frame()
    }
    best <- .cv_select_best_result_from_grid(
      results = grid_results,
      summaries = summaries,
      metrics = metrics,
      selection_metric = selection_ctl$metric
    )
    return(.cv_attach_fit_data(best, Xdata, Ydata))
  }

  cfg <- tuning_grid[[1L]]
  scaling <- cfg$scaling
  method <- cfg$method
  backend <- cfg$backend
  svd.method <- cfg$svd.method
  north <- cfg$north
  kernel <- cfg$kernel
  gamma <- cfg$gamma
  degree <- cfg$degree
  coef0 <- cfg$coef0
  classifier <- cfg$classifier
  lda_ridge <- cfg$lda_ridge
  k <- cfg$k
  tau <- cfg$tau
  alpha <- cfg$alpha
  top_m <- cfg$top_m
  cknn_memory <- cfg$cknn_memory
  xprod <- cfg$xprod
  backend <- .normalize_public_backend(backend)
  backend_compiled <- .compiled_backend(backend)
  dots <- .svd_control_from_dots(cfg$svd_dots)
  svd_ctl <- .resolve_svd_control(
    svd.method = svd.method,
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
  Xdata <- as.matrix(Xdata)
  if (is.null(constrain)) constrain <- seq_len(nrow(Xdata))
  classification <- is.factor(Ydata)

  res <- if ((identical(classifier, "cknn") && !identical(cknn_memory, "standard")) ||
             !identical(kernel, "linear")) {
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
      north = north,
      kernel = kernel,
      gamma = gamma,
      degree = degree,
      coef0 = coef0,
      classifier = classifier,
      lda_ridge = lda_ridge,
      k = k,
      tau = tau,
      alpha = alpha,
      top_m = top_m,
      cknn_memory = cknn_memory,
      return_scores = TRUE,
      store_predictions = !classification ||
        classification ||
        (classification && !identical(classifier, "argmax")),
      selection_metric = selection_ctl$metric
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
    return_scores = TRUE,
    classifier = classifier,
    lda_ridge = lda_ridge,
    k = k,
    tau = tau,
    alpha = alpha,
    top_m = top_m,
    store_predictions = !classification ||
      classification ||
      (classification && !identical(classifier, "argmax")),
    selection_metric = selection_ctl$metric
    )
  }
  selection_metrics <- .cv_selection_metrics(
    cv_res = res,
    Ydata = Ydata,
    classification = classification,
    selection_metric = selection_ctl$metric
  )
  best_idx <- .cv_best_index(selection_metrics, selection_metric = selection_ctl$metric)
  values <- as.numeric(res$metrics$metric_value)
  q2_values <- res$Q2Y
  rmsd_values <- res$RMSD
  accuracy_values <- if (classification) {
    if (!is.null(res$accuracy)) as.numeric(res$accuracy) else values
  } else {
    NULL
  }
  if (classification &&
      (is.null(q2_values) || length(q2_values) != length(values) || all(!is.finite(q2_values)))) {
    score_cube <- res$Yscore %||% res$Ypred
    if (!is.null(score_cube)) {
      q2_values <- .cv_classification_q2_path(Ydata, score_cube, res$levels)
    } else {
      q2_values <- rep(NA_real_, length(values))
    }
  }
  if (!classification &&
      (is.null(q2_values) || is.null(rmsd_values) ||
       all(!is.finite(q2_values)) || all(!is.finite(rmsd_values)))) {
    if (!is.null(res$Ypred)) {
      dims <- dim(res$Ypred)
      q2_values <- rmsd_values <- rep(NA_real_, dims[[3L]])
      for (i in seq_len(dims[[3L]])) {
        metric_pair <- .cv_regression_q2_rmsd(
          Ytrue = Ydata,
          Ypred = res$Ypred[, , i, drop = TRUE],
          Ytrain = Ydata
        )
        q2_values[[i]] <- metric_pair$Q2Y
        rmsd_values[[i]] <- metric_pair$RMSD
      }
    } else {
      q2_values <- if (all(tolower(res$metrics$metric_name) == "q2")) values else rep(NA_real_, length(values))
      rmsd_values <- if (all(tolower(res$metrics$metric_name) == "rmsd")) values else rep(NA_real_, length(values))
    }
  }
  selection_values <- as.numeric(selection_metrics$metric_value)
  res$best_ncomp <- as.integer(res$ncomp[[best_idx]])
  res$best_index <- best_idx
  res$selection_metric <- selection_ctl$metric
  res$selection_metrics <- selection_metrics
  res$selection_values <- selection_values
  res$best_metric_name <- .cv_metric_name_at(selection_metrics, best_idx)
  res$best_metric_value <- selection_values[[best_idx]]
  if (classification) {
    res$accuracy <- accuracy_values
  }
  res$Q2Y <- as.numeric(q2_values)
  res$RMSD <- if (classification) rep(NA_real_, length(values)) else as.numeric(rmsd_values)
  training_fit <- if (!isTRUE(fit)) {
    list(R2Y = rep(NA_real_, length(values)), Yfit = NULL)
  } else {
    .cv_training_fit_summary(
      Xdata = Xdata,
      Ydata = Ydata,
      ncomp = as.integer(res$ncomp),
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
      north = north,
      kernel = kernel,
      gamma = gamma,
      degree = degree,
      coef0 = coef0
    )
  }
  res$R2Y <- training_fit$R2Y
  res$Yfit <- training_fit$Yfit
  res$Ypred_optim <- .cv_extract_prediction_at(res, best_idx)
  res$tuning_config <- .cv_prune_config_for_output(cfg)
  res$best_parameters <- .cv_selected_parameters(cfg, tuning_grid, res$best_ncomp)
  .cv_attach_fit_data(res, Xdata, Ydata)
}






#' Nested cross-validation for PLS
#'
#' Performs nested grouped cross-validation with an outer loop for unbiased
#' performance estimation and an inner loop for component and hyperparameter
#' selection. Constraint groups are respected in both loops so related samples
#' remain in the same fold.
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
#' @param method One or more of \code{simpls}, \code{plssvd}, \code{opls}, or
#'   \code{kernelpls}. Multiple values are tuned in the inner loop.
#' @param backend Implementation backend: \code{cpu}, \code{cuda}, or
#'   \code{metal}. Multiple values are tuned in the inner loop.
#' @param seed Random seed used for outer/inner fold assignment and randomized
#'   SVD steps.
#' @param gamma Kernel scale. Defaults internally to `1 / ncol(Xdata)`. For
#'   \code{method = "kernelpls"}, multiple values are tuned in the inner loop.
#' @param xprod Use the matrix-free cross-product route where available for
#'   inner component optimization. `NULL` applies fastPLS defaults.
#' @param perm.test Run a nested-CV permutation test. For each permutation, the
#'   rows of `Xdata` are shuffled, the complete double cross-validation is
#'   repeated, and the median permuted `Q2Y` is compared with the observed
#'   median `Q2Y`.
#' @param times Number of permutations. For predictive metrics where larger is
#'   better, such as `Q2Y`, the empirical p-value is
#'   `mean(Q2Y_permuted >= Q2Y_observed)`. For loss metrics where smaller is
#'   better, such as RMSD, it is `mean(loss_permuted <= loss_observed)`. No +1
#'   correction is applied.
#' @param ... Optional SVD tuning controls forwarded to the selected backend.
#'   Use the same compact names documented in [fastsvd()], such as
#'   `oversample`, `power`, `svds_tol`, `work`, `maxit`, `tol`, `eps`,
#'   and `svtol`. Vector values are tuned in the inner loop. Inner selection can
#'   also be controlled with `selection_metric = "auto"`, `"accuracy"`, `"r2"`,
#'   `"q2"`, or `"rmsd"`; the selection metric itself is scalar.
#' @return A list with the following elements:
#'
#'   * `results`: list with one element per repeated run. Each run stores
#'     `Ypred`/`pred`, the outer `fold` assignment, `best_ncomp` selected in
#'     each outer fold, fold-level `best_parameters`, the complete inner-CV
#'     objects in `inner`, run-level `metric_name` and `metric_value`, and the
#'     default `backend` and `method`.
#'   * `Ypred`: final cross-validated predictions. For classification, repeated
#'     runs are combined by voting; for regression, numeric predictions are
#'     averaged across runs.
#'   * `Q2Y`: one held-out Q2 value per repeated run. For factor responses this
#'     is calculated on dummy-coded PLS-DA response scores.
#'   * `R2Y`: one training-fit R2 value per repeated run, averaged across the
#'     selected outer-fold models.
#'   * `RMSD`: one held-out RMSD value per repeated run for numeric responses;
#'     `NA` for classification.
#'   * `metric_name`: held-out metric used for each repeated run.
#'   * `bcomp`: most frequently selected component count across outer folds and
#'     repeated runs.
#'   * `backend`, `method`: default backend and PLS method supplied to the call.
#'     If vector-valued methods or backends are tuned, selected fold-level values
#'     are stored in `results[[run]]$best_parameters`.
#'   * `selection_metric`: criterion used by the inner CV loop.
#'   * `acc_tot`: classification-only text summary of correctly classified
#'     samples and percentage accuracy.
#'   * `conf`: classification-only confusion matrix printed as counts and
#'     column percentages.
#'   * `vote_counts`: classification-only vote-count matrix with one row per
#'     sample and one column per class.
#'   * `accuracy`: classification-only decoded-label accuracy, one value per
#'     repeated run.
#'   * `medianR2Y`, `CI95R2Y`, `medianQ2Y`, `CI95Q2Y`, `medianRMSD`,
#'     `CI95RMSD`: repeated-run summaries returned only when `runn > 1`.
#'   * `Q2Ysampled`: permutation median-Q2 values returned when
#'     `perm.test = TRUE`.
#'   * `p.value`: permutation-test p-value returned when `perm.test = TRUE`.
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
                         runn=1,
                         kfold_inner=10,
                         kfold_outer=10,
                         north = 1L,
                         kernel = c("linear", "rbf", "poly"),
                         gamma = NULL,
                         degree = 3L,
                         coef0 = 1,
                         classifier = c("argmax", "lda", "cknn"),
                         lda_ridge = 1e-8,
                         k = 10L,
                         tau = 0.2,
                         alpha = 0.75,
                         top_m = 20L,
                         cknn_memory = c("auto", "standard", "blocked", "streaming"),
                         xprod = NULL,
                         ...){

  if(sum(is.na(Xdata))>0) {
    stop("Missing values are present")
  }
  selection_ctl <- .cv_selection_metric_from_dots(list(...))
  tuning_grid <- .cv_make_prediction_grid(
    scaling = scaling,
    scaling_missing = missing(scaling),
    method = method,
    method_missing = missing(method),
    backend = backend,
    backend_missing = missing(backend),
    svd.method = svd.method,
    svd_missing = missing(svd.method),
    north = north,
    kernel = kernel,
    kernel_missing = missing(kernel),
    gamma = gamma,
    degree = degree,
    coef0 = coef0,
    classifier = classifier,
    classifier_missing = missing(classifier),
    lda_ridge = lda_ridge,
    k = k,
    tau = tau,
    alpha = alpha,
    top_m = top_m,
    cknn_memory = cknn_memory,
    cknn_memory_missing = missing(cknn_memory),
    xprod = xprod,
    dots = selection_ctl$dots,
    context = "pls.double.cv()"
  )
  base_cfg <- tuning_grid[[1L]]
  scaling_grid <- .cv_grid_arg_values(tuning_grid, "scaling")
  method_grid <- .cv_grid_arg_values(tuning_grid, "method")
  backend_grid <- .cv_grid_arg_values(tuning_grid, "backend")
  svd_method_grid <- .cv_grid_arg_values(tuning_grid, "svd.method")
  north_grid <- .cv_grid_arg_values(tuning_grid, "north")
  kernel_grid <- .cv_grid_arg_values(tuning_grid, "kernel")
  gamma_grid <- .cv_grid_arg_values(tuning_grid, "gamma")
  degree_grid <- .cv_grid_arg_values(tuning_grid, "degree")
  coef0_grid <- .cv_grid_arg_values(tuning_grid, "coef0")
  classifier_grid <- .cv_grid_arg_values(tuning_grid, "classifier")
  lda_ridge_grid <- .cv_grid_arg_values(tuning_grid, "lda_ridge")
  k_grid <- .cv_grid_arg_values(tuning_grid, "k")
  tau_grid <- .cv_grid_arg_values(tuning_grid, "tau")
  alpha_grid <- .cv_grid_arg_values(tuning_grid, "alpha")
  top_m_grid <- .cv_grid_arg_values(tuning_grid, "top_m")
  cknn_memory_grid <- .cv_grid_arg_values(tuning_grid, "cknn_memory")
  xprod_grid <- .cv_grid_arg_values(tuning_grid, "xprod")
  svd_dot_names <- unique(unlist(lapply(tuning_grid, function(cfg) names(cfg$svd_dots)), use.names = FALSE))
  svd_dot_args <- lapply(svd_dot_names, function(nm) .cv_grid_dot_values(tuning_grid, nm))
  names(svd_dot_args) <- svd_dot_names

  method <- base_cfg$method
  backend <- base_cfg$backend
  scaling <- base_cfg$scaling
  north <- base_cfg$north
  kernel <- base_cfg$kernel
  gamma <- base_cfg$gamma
  degree <- base_cfg$degree
  coef0 <- base_cfg$coef0
  classifier_public <- base_cfg$classifier
  classifier <- .resolve_classifier_for_backend(classifier_public, backend)
  lda_ridge <- base_cfg$lda_ridge
  k <- base_cfg$k
  tau <- base_cfg$tau
  alpha <- base_cfg$alpha
  top_m <- base_cfg$top_m
  cknn_memory <- base_cfg$cknn_memory
  xprod <- base_cfg$xprod

  dots <- .svd_control_from_dots(base_cfg$svd_dots)
  svd_ctl <- .resolve_svd_control(
    svd.method = base_cfg$svd.method,
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
  RMSD <- rep(NA_real_, as.integer(runn))
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
    best_parameters <- vector("list", nfold_outer)
    outer_train_r2 <- rep(NA_real_, nfold_outer)
    outer_q2 <- rep(NA_real_, nfold_outer)
    outer_accuracy <- rep(NA_real_, nfold_outer)
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

      inner_args <- c(
        list(
          Xdata = Xdata[train_idx, , drop = FALSE],
          Ydata = Ytrain,
          ncomp = ncomp,
          constrain = constrain[train_idx],
          scaling = scaling_grid,
          method = method_grid,
          backend = backend_grid,
          svd.method = svd_method_grid,
          seed = as.integer(seed) + 1000L * j + f,
          kfold = kfold_inner,
          north = north_grid,
          kernel = kernel_grid,
          gamma = gamma_grid,
          degree = degree_grid,
          coef0 = coef0_grid,
          classifier = classifier_grid,
          lda_ridge = lda_ridge_grid,
          k = k_grid,
          tau = tau_grid,
          alpha = alpha_grid,
          top_m = top_m_grid,
          cknn_memory = cknn_memory_grid,
          xprod = xprod_grid,
          selection_metric = selection_ctl$metric
        ),
        svd_dot_args
      )
      inner <- do.call(pls.single.cv, inner_args)
      best_comp[f] <- as.integer(inner$best_ncomp[[1L]])
      inner_results[[f]] <- inner
      selected <- inner$best_parameters
      best_parameters[[f]] <- selected
      fit_scaling <- .cv_value_or_default(selected, "scaling", scaling)
      fit_method <- .cv_value_or_default(selected, "method", method)
      fit_backend <- .cv_value_or_default(selected, "backend", backend)
      fit_svd_method <- .cv_value_or_default(selected, "svd.method", svd.method)
      fit_north <- .cv_value_or_default(selected, "north", north)
      fit_kernel <- .cv_value_or_default(selected, "kernel", kernel)
      fit_gamma <- .cv_value_or_default(selected, "gamma", gamma)
      fit_degree <- .cv_value_or_default(selected, "degree", degree)
      fit_coef0 <- .cv_value_or_default(selected, "coef0", coef0)
      fit_classifier_public <- .cv_value_or_default(selected, "classifier", classifier_public)
      fit_classifier <- .resolve_classifier_for_backend(fit_classifier_public, fit_backend)
      fit_lda_ridge <- .cv_value_or_default(selected, "lda_ridge", lda_ridge)
      fit_k <- .cv_value_or_default(selected, "k", k)
      fit_tau <- .cv_value_or_default(selected, "tau", tau)
      fit_alpha <- .cv_value_or_default(selected, "alpha", alpha)
      fit_top_m <- .cv_value_or_default(selected, "top_m", top_m)
      fit_cknn_memory <- .cv_value_or_default(selected, "cknn_memory", cknn_memory)
      fit_rsvd_oversample <- .cv_value_or_default(selected, "rsvd_oversample", rsvd_oversample)
      fit_rsvd_power <- .cv_value_or_default(selected, "rsvd_power", rsvd_power)
      fit_svds_tol <- .cv_value_or_default(selected, "svds_tol", svds_tol)
      fit_irlba_work <- .cv_value_or_default(selected, "irlba_work", irlba_work)
      fit_irlba_maxit <- .cv_value_or_default(selected, "irlba_maxit", irlba_maxit)
      fit_irlba_tol <- .cv_value_or_default(selected, "irlba_tol", irlba_tol)
      fit_irlba_eps <- .cv_value_or_default(selected, "irlba_eps", irlba_eps)
      fit_irlba_svtol <- .cv_value_or_default(selected, "irlba_svtol", irlba_svtol)

      fit <- pls(
        Xtrain = Xdata[train_idx, , drop = FALSE],
        Ytrain = Ytrain,
        Xtest = Xdata[test_idx, , drop = FALSE],
        Ytest = Ytest,
        ncomp = best_comp[f],
        scaling = fit_scaling,
        method = fit_method,
        svd.method = fit_svd_method,
        rsvd_oversample = fit_rsvd_oversample,
        rsvd_power = fit_rsvd_power,
        svds_tol = fit_svds_tol,
        seed = as.integer(seed) + 2000L * j + f,
        irlba_work = fit_irlba_work,
        irlba_maxit = fit_irlba_maxit,
        irlba_tol = fit_irlba_tol,
        irlba_eps = fit_irlba_eps,
        irlba_svtol = fit_irlba_svtol,
        fit = TRUE,
        proj = FALSE,
        backend = fit_backend,
        north = fit_north,
        kernel = fit_kernel,
        gamma = fit_gamma,
        degree = fit_degree,
        coef0 = fit_coef0,
        classifier = fit_classifier,
        lda_ridge = fit_lda_ridge,
        k = fit_k,
        tau = fit_tau,
        alpha = fit_alpha,
        top_m = fit_top_m,
        cknn_memory = fit_cknn_memory
      )

      if (classification) {
        if (!is.null(fit$R2Y) && length(fit$R2Y)) {
          outer_train_r2[[f]] <- as.numeric(tail(fit$R2Y, 1L))
        }
        if (!is.null(fit$Q2Y) && length(fit$Q2Y)) {
          outer_q2[[f]] <- as.numeric(tail(fit$Q2Y, 1L))
        }
        if (!is.null(fit$accuracy) && length(fit$accuracy)) {
          outer_accuracy[[f]] <- as.numeric(tail(fit$accuracy, 1L))
        }
        pred <- fit$Ypred
        pred <- if (is.data.frame(pred)) pred[[1L]] else pred
        run_pred_chr[test_idx] <- as.character(pred)
      } else {
        if (!is.null(fit$R2Y) && length(fit$R2Y)) {
          outer_train_r2[[f]] <- as.numeric(tail(fit$R2Y, 1L))
        }
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
      accuracy_j <- mean(as.character(pred_factor) == as.character(Ydata_original), na.rm = TRUE)
      if (!any(is.finite(outer_accuracy))) {
        outer_accuracy[] <- accuracy_j
      }
      Q2Y[j] <- if (any(is.finite(outer_q2))) {
        mean(outer_q2, na.rm = TRUE)
      } else {
        NA_real_
      }
      R2Y[j] <- if (any(is.finite(outer_train_r2))) {
        mean(outer_train_r2, na.rm = TRUE)
      } else {
        NA_real_
      }
      metric_name[j] <- "accuracy"
      res$results[[j]] <- list(
        Ypred = pred_factor,
        pred = pred_factor,
        fold = fold + 1L,
        best_ncomp = best_comp,
        best_parameters = best_parameters,
        inner = inner_results,
        metric_name = "accuracy",
        metric_value = accuracy_j,
        accuracy = accuracy_j,
        backend = backend,
        method = method
      )
      bb <- c(bb, best_comp)
    } else {
      Ypred_tot <- Ypred_tot + run_pred
      run_selection_metric <- if (selection_ctl$metric %in% c("r2", "q2", "rmsd")) selection_ctl$metric else "auto"
      metric <- .cv_metric_from_matrix(Ydata, run_pred, Ytrain = Ydata, metric = run_selection_metric)
      metric_pair <- .cv_regression_q2_rmsd(Ydata, run_pred, Ytrain = Ydata)
      Q2Y[j] <- metric_pair$Q2Y
      RMSD[j] <- metric_pair$RMSD
      R2Y[j] <- if (any(is.finite(outer_train_r2))) {
        mean(outer_train_r2, na.rm = TRUE)
      } else {
        NA_real_
      }
      metric_name[j] <- metric$metric_name
      res$results[[j]] <- list(
        Ypred = run_pred,
        pred = run_pred,
        fold = fold + 1L,
        best_ncomp = best_comp,
        best_parameters = best_parameters,
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
    res$accuracy <- vapply(res$results, function(x) {
      val <- x$accuracy
      if (is.null(val) || !length(val)) NA_real_ else as.numeric(val[[1L]])
    }, numeric(1))
  } else {
    res$Ypred <- Ypred_tot / as.integer(runn)
  }

  res$Q2Y <- Q2Y
  res$R2Y <- R2Y
  res$RMSD <- RMSD
  res$metric_name <- metric_name
  if (as.integer(runn) > 1L) {
    res$medianR2Y <- median(R2Y, na.rm = TRUE)
    res$CI95R2Y <- as.numeric(quantile(R2Y, c(0.025, 0.975), na.rm = TRUE))
    res$medianQ2Y <- median(Q2Y, na.rm = TRUE)
    res$CI95Q2Y <- as.numeric(quantile(Q2Y, c(0.025, 0.975), na.rm = TRUE))
    res$medianRMSD <- median(RMSD, na.rm = TRUE)
    res$CI95RMSD <- as.numeric(quantile(RMSD, c(0.025, 0.975), na.rm = TRUE))
  }
  res$bcomp <- names(which.max(table(bb)))
  res$backend <- backend
  res$method <- method
  res$selection_metric <- selection_ctl$metric

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
        k = k,
        tau = tau,
        alpha = alpha,
        top_m = top_m,
        cknn_memory = cknn_memory,
        xprod = xprod,
        selection_metric = selection_ctl$metric
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







#' Evaluate prediction performance
#'
#' Computes common classification or regression performance metrics from
#' observed and predicted values. The function accepts two vectors, two matrices,
#' or classification score matrices. For NMR-style multivariate regression, it
#' reports RMSE/RMSD, R2, Q2, MAE, median relative error percentage, RPD, and
#' correlations. For classification, it reports accuracy, balanced accuracy,
#' macro precision, macro recall, macro F1, Cohen's kappa, and the confusion
#' matrix.
#'
#' @param observed Observed response values. Use a factor/character vector for
#'   classification, a numeric vector/matrix for regression, or a one-hot
#'   matrix for classification.
#' @param predicted Predicted values. Use a factor/character vector for
#'   predicted classes, a numeric vector/matrix for regression, or a class-score
#'   matrix for classification.
#' @param task One of `"auto"`, `"classification"`, or `"regression"`.
#'   `"auto"` treats factor/character observed values as classification and
#'   numeric values as regression, unless a one-hot observed matrix is detected.
#' @param ytrain Optional training response for regression Q2. When supplied,
#'   Q2 is computed relative to the training-set response mean. When omitted,
#'   Q2 uses the observed response mean and is therefore identical to R2.
#' @param top_k Integer vector of top-k classification accuracies to compute
#'   when `predicted` is a class-score matrix.
#' @param relative_epsilon Values with absolute observed response below this
#'   threshold are ignored for relative-error metrics.
#' @param na.rm Remove incomplete observations before computing metrics.
#' @return A list with `task`, `metrics`, and optionally `per_response`,
#'   `confusion`, `topk`, and `notes`.
#' @examples
#' evaluate(iris$Species, iris$Species)
#'
#' set.seed(1)
#' y <- mtcars$mpg
#' pred <- y + rnorm(length(y), sd = 2)
#' evaluate(y, pred)$metrics
#' @export
evaluate <- function(observed,
                     predicted,
                     task = c("auto", "classification", "regression"),
                     ytrain = NULL,
                     top_k = c(1L, 5L),
                     relative_epsilon = .Machine$double.eps,
                     na.rm = TRUE) {
  task <- match.arg(task)
  notes <- character()

  is_onehot <- function(x) {
    is.matrix(x) &&
      is.numeric(x) &&
      ncol(x) > 1L &&
      all(is.finite(x) | is.na(x)) &&
      all(abs(rowSums(x, na.rm = TRUE) - 1) < 1e-8, na.rm = TRUE) &&
      all(x[!is.na(x)] %in% c(0, 1))
  }

  if (identical(task, "auto")) {
    task <- if (is.factor(observed) || is.character(observed) || is_onehot(observed)) {
      "classification"
    } else {
      "regression"
    }
  }

  if (identical(task, "classification")) {
    class_from_input <- function(x, levels_ref = NULL) {
      if (is.factor(x) || is.character(x)) {
        return(as.character(x))
      }
      if (is.matrix(x) || is.data.frame(x)) {
        x <- as.matrix(x)
        if (!is.numeric(x)) {
          return(as.character(x[, 1L]))
        }
        lev <- colnames(x)
        if (is.null(lev)) {
          lev <- if (is.null(levels_ref)) as.character(seq_len(ncol(x))) else levels_ref
        }
        return(lev[max.col(x, ties.method = "first")])
      }
      as.character(x)
    }

    obs_levels <- if (is.factor(observed)) levels(observed) else NULL
    if (is.matrix(observed) && !is.null(colnames(observed))) {
      obs_levels <- colnames(observed)
    }
    obs <- class_from_input(observed, obs_levels)
    pred <- class_from_input(predicted, obs_levels)
    lev <- unique(c(obs_levels, obs, pred))
    lev <- lev[!is.na(lev)]
    if (!length(lev)) {
      stop("No valid class labels were found.", call. = FALSE)
    }
    obs <- factor(obs, levels = lev)
    pred <- factor(pred, levels = lev)
    keep <- rep(TRUE, length(obs))
    if (na.rm) {
      keep <- !is.na(obs) & !is.na(pred)
      obs <- obs[keep]
      pred <- pred[keep]
    }
    if (length(obs) != length(pred)) {
      stop("observed and predicted must have the same number of samples.", call. = FALSE)
    }

    conf <- table(predicted = pred, observed = obs)
    tp <- diag(conf)
    support <- colSums(conf)
    predicted_support <- rowSums(conf)
    recall <- tp / support
    precision <- tp / predicted_support
    recall[!is.finite(recall)] <- NA_real_
    precision[!is.finite(precision)] <- NA_real_
    f1 <- 2 * precision * recall / (precision + recall)
    f1[!is.finite(f1)] <- NA_real_
    n <- sum(conf)
    accuracy <- if (n > 0) sum(tp) / n else NA_real_
    expected_accuracy <- if (n > 0) sum(rowSums(conf) * colSums(conf)) / n^2 else NA_real_
    kappa <- if (is.finite(expected_accuracy) && expected_accuracy < 1) {
      (accuracy - expected_accuracy) / (1 - expected_accuracy)
    } else {
      NA_real_
    }

    topk <- NULL
    if ((is.matrix(predicted) || is.data.frame(predicted)) && is.numeric(as.matrix(predicted))) {
      score <- as.matrix(predicted)
      score_levels <- colnames(score)
      if (is.null(score_levels)) {
        score_levels <- if (!is.null(obs_levels)) obs_levels else as.character(seq_len(ncol(score)))
      }
      obs_chr <- as.character(class_from_input(observed, score_levels))
      if (na.rm) {
        obs_chr <- obs_chr[keep]
        score <- score[keep, , drop = FALSE]
      }
      topk <- data.frame(
        k = as.integer(top_k),
        accuracy = vapply(as.integer(top_k), function(k) {
          k <- min(max(1L, k), ncol(score))
          top_idx <- t(matrix(
            vapply(
              seq_len(nrow(score)),
              function(i) order(score[i, ], decreasing = TRUE)[seq_len(k)],
              integer(k)
            ),
            nrow = k
          ))
          mean(vapply(seq_len(nrow(score)), function(i) obs_chr[[i]] %in% score_levels[top_idx[i, ]], logical(1L)), na.rm = TRUE)
        }, numeric(1L))
      )
    }

    return(list(
      task = "classification",
      metrics = data.frame(
        n = as.integer(n),
        accuracy = accuracy,
        balanced_accuracy = mean(recall, na.rm = TRUE),
        macro_precision = mean(precision, na.rm = TRUE),
        macro_recall = mean(recall, na.rm = TRUE),
        macro_f1 = mean(f1, na.rm = TRUE),
        kappa = kappa,
        stringsAsFactors = FALSE
      ),
      per_class = data.frame(
        class = names(tp),
        support = as.integer(support),
        precision = as.numeric(precision),
        recall = as.numeric(recall),
        f1 = as.numeric(f1),
        stringsAsFactors = FALSE
      ),
      confusion = conf,
      topk = topk,
      notes = notes
    ))
  }

  obs <- as.matrix(observed)
  pred <- as.matrix(predicted)
  if (!is.numeric(obs) || !is.numeric(pred)) {
    stop("Regression evaluation requires numeric observed and predicted values.", call. = FALSE)
  }
  if (!all(dim(obs) == dim(pred))) {
    stop("observed and predicted must have the same dimensions.", call. = FALSE)
  }
  train <- if (!is.null(ytrain)) {
    as.matrix(ytrain)
  } else {
    notes <- c(notes, "Q2 uses the observed response mean because ytrain was not supplied; Q2 equals R2 in this case.")
    obs
  }
  if (ncol(train) != ncol(obs)) {
    stop("ytrain must have the same number of response columns as observed.", call. = FALSE)
  }

  metric_one <- function(o, p, tr) {
    keep <- is.finite(o) & is.finite(p)
    if (na.rm) {
      o <- o[keep]
      p <- p[keep]
    }
    n <- length(o)
    if (!n) {
      return(rep(NA_real_, 12L))
    }
    err <- p - o
    sse <- sum(err^2, na.rm = TRUE)
    rmse <- sqrt(mean(err^2, na.rm = TRUE))
    mae <- mean(abs(err), na.rm = TRUE)
    bias <- mean(err, na.rm = TRUE)
    tss_obs <- sum((o - mean(o, na.rm = TRUE))^2, na.rm = TRUE)
    tr_center <- mean(tr, na.rm = TRUE)
    tss_train <- sum((o - tr_center)^2, na.rm = TRUE)
    rel_ok <- is.finite(o) & abs(o) > relative_epsilon
    rel_err_pct <- abs(err[rel_ok] / o[rel_ok]) * 100
    mre_pct <- if (length(rel_err_pct)) stats::median(rel_err_pct, na.rm = TRUE) else NA_real_
    mape_pct <- if (length(rel_err_pct)) mean(rel_err_pct, na.rm = TRUE) else NA_real_
    r <- suppressWarnings(stats::cor(o, p, method = "pearson", use = "complete.obs"))
    rho <- suppressWarnings(stats::cor(o, p, method = "spearman", use = "complete.obs"))
    sd_obs <- stats::sd(o, na.rm = TRUE)
    c(
      n = n,
      R2 = if (is.finite(tss_obs) && tss_obs > 0) 1 - sse / tss_obs else NA_real_,
      Q2 = if (is.finite(tss_train) && tss_train > 0) 1 - sse / tss_train else NA_real_,
      RMSD = rmse,
      RMSE = rmse,
      MAE = mae,
      bias = bias,
      MRE_percent = mre_pct,
      MAPE_percent = mape_pct,
      RPD = if (is.finite(rmse) && rmse > 0) sd_obs / rmse else NA_real_,
      Pearson_r = r,
      Spearman_r = rho
    )
  }

  per <- t(vapply(seq_len(ncol(obs)), function(j) {
    metric_one(obs[, j], pred[, j], train[, j])
  }, numeric(12L)))
  per <- as.data.frame(per)
  per$response <- colnames(obs) %||% paste0("Y", seq_len(ncol(obs)))
  per <- per[, c("response", setdiff(names(per), "response")), drop = FALSE]
  overall <- as.data.frame(as.list(metric_one(as.vector(obs), as.vector(pred), as.vector(train))))

  list(
    task = "regression",
    metrics = overall,
    per_response = per,
    notes = notes
  )
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
