##r_irlba <- function(X, nu, work=nu+7, maxit=1000, tol=1e-5, eps=1e-10, svtol=tol) {
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
                                       fast_defl_cache = TRUE) {
  old <- c(
    FASTPLS_FAST_BLOCK = Sys.getenv("FASTPLS_FAST_BLOCK", unset = NA_character_),
    FASTPLS_FAST_CENTER_T = Sys.getenv("FASTPLS_FAST_CENTER_T", unset = NA_character_),
    FASTPLS_FAST_REORTH_V = Sys.getenv("FASTPLS_FAST_REORTH_V", unset = NA_character_),
    FASTPLS_FAST_INCREMENTAL = Sys.getenv("FASTPLS_FAST_INCREMENTAL", unset = NA_character_),
    FASTPLS_FAST_INC_ITERS = Sys.getenv("FASTPLS_FAST_INC_ITERS", unset = NA_character_),
    FASTPLS_FAST_DEFLCACHE = Sys.getenv("FASTPLS_FAST_DEFLCACHE", unset = NA_character_)
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
    FASTPLS_FAST_DEFLCACHE = if (isTRUE(fast_defl_cache)) "1" else "0"
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
    stop("svd.method='arpack' has been removed from fastPLS; use 'exact', 'irlba', or 'cpu_rsvd'.", call. = FALSE)
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
        "The hybrid CUDA path via svd.method='cuda_rsvd' has been removed from %s; use simpls_gpu() or plssvd_gpu() for GPU-native fits instead.",
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
            fast_defl_cache = TRUE)
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
        fast_defl_cache = profile$fast_defl_cache
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
            fast_defl_cache = TRUE)
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
      fast_defl_cache = profile$fast_defl_cache
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


#' Predict from a fitted fastPLS object
#'
#' Applies stored preprocessing (`mX`, `vX`) and coefficient slices (`B`) to
#' produce test predictions. For classification models, one-hot predictions are
#' converted to labels by argmax over response columns.
#'
#' @param object A fitted `fastPLS` object.
#' @param newdata Numeric predictor matrix.
#' @param Ytest Optional observed response used to compute `Q2Y`.
#' @param proj Logical; return projected `Ttest` when `TRUE`.
#' @param predict.backend Prediction backend. `"auto"` uses FlashSVD-style
#'   low-rank prediction when compact factors are available and the low-rank
#'   application is expected to be beneficial.
#' @param flash.block_size Row block size for `"cpu_flash"` prediction.
#' @param ... Unused.
#' @return A list containing `Ypred`, optional `Q2Y`, and optional `Ttest`.
#' @export
predict.fastPLS = function(object, newdata, Ytest=NULL, proj=FALSE,
                           predict.backend = c("auto", "cpu", "cpu_flash", "cuda_flash"),
                           flash.block_size = NULL, ...) {
  if (!is(object, "fastPLS")) {
    stop("object is not a fastPLS object")
  }
  predict.backend <- match.arg(predict.backend)
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
    Ypredlab = as.data.frame(matrix(nrow = nrow(Xtest), ncol = length(object$ncomp)))

    for (i in 1:length(object$ncomp)) {
      t = apply(res$Ypred[, , i], 1, which.max)
      Ypredlab[, i] = (factor(object$lev[t], levels = object$lev))
    }
    res$Ypred=Ypredlab

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
      s <- svd(crossprod(Xf, Yc), nu = 1L, nv = 0L)
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
#' @export
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
                         svd.method = c("irlba", "cpu_rsvd", "exact"),
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
                         fit = FALSE,
                         proj = FALSE) {
  method <- match.arg(method)
  svd.method <- match.arg(.normalize_svd_method(svd.method), c("irlba", "cpu_rsvd", "exact"))
  if (isTRUE(gaussian_y) && is.null(gaussian_y_dim)) {
    gaussian_y_dim <- .gaussian_y_default_dim(Xtrain, NULL)
  }
  .kernel_pls_fit(
    Xtrain, Ytrain, Xtest, Ytest, ncomp, match.arg(scaling), match.arg(kernel),
    gamma, degree, coef0, fit, proj, "R", pls_r,
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
      gaussian_y_seed = gaussian_y_seed
    )
  )
}

#' @rdname kernel_pls_r
#' @export
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
                           svd.method = c("irlba", "cpu_rsvd", "exact"),
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
                           fit = FALSE,
                           proj = FALSE) {
  method <- match.arg(method)
  svd.method <- match.arg(.normalize_svd_method(svd.method), c("irlba", "cpu_rsvd", "exact"))
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
      gaussian_y_seed = gaussian_y_seed
    )
  )
}

#' @rdname kernel_pls_r
#' @export
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
                            fit = FALSE,
                            proj = FALSE,
                            ...) {
  method <- match.arg(method)
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
        gaussian_y_seed = gaussian_y_seed
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
#' @export
kernel_pls_fast_r <- function(...) {
  .fastpls_call_fixed_method(kernel_pls_r, "simpls", ...)
}

#' @rdname kernel_pls_r
#' @export
kernel_pls_fast_cpp <- function(...) {
  .fastpls_call_fixed_method(kernel_pls_cpp, "simpls", ...)
}

#' @rdname kernel_pls_r
#' @export
kernel_pls_fast_cuda <- function(...) {
  .fastpls_call_fixed_method(kernel_pls_cuda, "simpls", ...)
}

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
#' @export
opls_r <- function(Xtrain,
                   Ytrain,
                   Xtest = NULL,
                   Ytest = NULL,
                   ncomp = 2,
                   north = 1L,
                   scaling = c("centering", "autoscaling", "none"),
                   method = c("simpls", "plssvd"),
                   svd.method = c("irlba", "cpu_rsvd", "exact"),
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
                   fit = FALSE,
                   proj = FALSE) {
  method <- match.arg(method)
  svd.method <- match.arg(.normalize_svd_method(svd.method), c("irlba", "cpu_rsvd", "exact"))
  .opls_fit(
    Xtrain, Ytrain, Xtest, Ytest, ncomp, match.arg(scaling), north, fit, proj,
    "R", pls_r,
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
      gaussian_y_seed = gaussian_y_seed
    )
  )
}

#' @rdname opls_r
#' @export
opls_cpp <- function(Xtrain,
                     Ytrain,
                     Xtest = NULL,
                     Ytest = NULL,
                     ncomp = 2,
                     north = 1L,
                     scaling = c("centering", "autoscaling", "none"),
                     method = c("simpls", "plssvd"),
                     svd.method = c("irlba", "cpu_rsvd", "exact"),
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
                     fit = FALSE,
                     proj = FALSE) {
  method <- match.arg(method)
  svd.method <- match.arg(.normalize_svd_method(svd.method), c("irlba", "cpu_rsvd", "exact"))
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
      gaussian_y_seed = gaussian_y_seed
    )
  )
}

#' @rdname opls_r
#' @export
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
                      fit = FALSE,
                      proj = FALSE,
                      ...) {
  method <- match.arg(method)
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
        gaussian_y_seed = gaussian_y_seed
      ),
      if (identical(method, "simpls")) list(fast_block = fast_block) else list(),
      list(...)
    )
  )
}

#' @rdname opls_r
#' @export
opls_fast_r <- function(...) {
  .fastpls_call_fixed_method(opls_r, "simpls", ...)
}

#' @rdname opls_r
#' @export
opls_fast_cpp <- function(...) {
  .fastpls_call_fixed_method(opls_cpp, "simpls", ...)
}

#' @rdname opls_r
#' @export
opls_fast_cuda <- function(...) {
  .fastpls_call_fixed_method(opls_cuda, "simpls", ...)
}

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
#' @export
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
                      gaussian_y_seed = seed) {
  if (!has_cuda()) {
    stop("simpls_gpu requires a CUDA-enabled fastPLS build")
  }
  on.exit(try(cuda_reset_workspace(), silent = TRUE), add = TRUE)

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

  tuned <- .resolve_simpls_fast_rsvd_tuning(
    n = nrow(Xtrain),
    p = ncol(Xtrain),
    q = ncol(Ytrain),
    svd.method = "cuda_rsvd"
  )
  if (missing(rsvd_oversample)) rsvd_oversample <- tuned$rsvd_oversample
  if (missing(rsvd_power)) rsvd_power <- tuned$rsvd_power

  use_xprod_default <- .should_use_xprod_default(ncol(Xtrain), ncol(Ytrain), ncomp)
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
  model <- .enable_flash_prediction(model, "cuda")
  model <- .attach_gaussian_y(model, yprep$gaussian)
  model <- .decode_gaussian_y_outputs(model, Ytrain_original)

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
#' @export
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
                      gaussian_y_seed = seed) {
  if (!has_cuda()) {
    stop("plssvd_gpu requires a CUDA-enabled fastPLS build")
  }
  on.exit(try(cuda_reset_workspace(), silent = TRUE), add = TRUE)

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
  model <- .enable_flash_prediction(model, "cuda")
  model <- .attach_gaussian_y(model, yprep$gaussian)
  model <- .decode_gaussian_y_outputs(model, Ytrain_original)

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
#' @export
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
#' @export
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
#' @export
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
#' @export
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
#' @export
plssvd_cv_cpp <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                          scaling = c("centering", "autoscaling", "none"),
                          svd.method = c("cpu_rsvd", "irlba"), xprod = NULL, ...) {
  .pls_cv_compiled(Xdata, Ydata, constrain, ncomp, kfold, scaling, "plssvd", "cpp", svd.method, xprod = xprod, ...)
}

#' @rdname plssvd_cv_cpp
#' @export
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
#' @export
kodama_cpp_simpls_rsvd <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                                   scaling = c("centering", "autoscaling", "none"),
                                   gaussian_y_seed = seed,
                                   seed = 1L,
                                   xprod = NULL, ...) {
  if (!is.factor(Ydata)) {
    stop("kodama_cpp_simpls_rsvd only supports classification factors.", call. = FALSE)
  }
  lev <- levels(Ydata)
  codes <- NULL
  if (length(lev) > 50L) {
    codes <- .gaussian_y_class_codes(lev, 50L, gaussian_y_seed)
  }
  res <- .pls_cv_compiled(
    Xdata, Ydata, constrain, ncomp, kfold, scaling, "simpls", "cpp",
    svd.method = "cpu_rsvd", seed = seed, xprod = xprod,
    kodama_class_codes = codes, ...
  )
  res$kodama_gaussian_y <- !is.null(codes)
  res$kodama_gaussian_y_dim <- if (is.null(codes)) length(lev) else ncol(codes)
  res
}

#' @rdname plssvd_cv_cpp
#' @export
opls_cv_cpp <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                        north = 1L,
                        scaling = c("centering", "autoscaling", "none"),
                        svd.method = c("cpu_rsvd", "irlba"), xprod = NULL, ...) {
  pred_ncomp <- pmax(1L, as.integer(ncomp) - as.integer(north))
  .pls_cv_compiled(Xdata, Ydata, constrain, pred_ncomp, kfold, scaling, "opls", "cpp", svd.method, xprod = xprod, north = north, ...)
}

#' @rdname plssvd_cv_cpp
#' @export
kernelpls_cv_cpp <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                             scaling = c("centering", "autoscaling", "none"),
                             svd.method = c("cpu_rsvd", "irlba"), xprod = NULL, ...) {
  .pls_cv_compiled(Xdata, Ydata, constrain, ncomp, kfold, scaling, "kernelpls", "cpp", svd.method, xprod = xprod, ...)
}

kernel_pls_cv_cpp <- kernelpls_cv_cpp

#' @rdname plssvd_cv_cpp
#' @export
plssvd_cv_cuda <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                           scaling = c("centering", "autoscaling", "none"),
                           xprod = NULL, ...) {
  .pls_cv_compiled(Xdata, Ydata, constrain, ncomp, kfold, scaling, "plssvd", "cuda", xprod = xprod, ...)
}

#' @rdname plssvd_cv_cpp
#' @export
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
#' @export
kodama_cuda_simpls_rsvd <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                                    scaling = c("centering", "autoscaling", "none"),
                                    gaussian_y_seed = seed,
                                    seed = 1L,
                                    xprod = NULL, ...) {
  if (!is.factor(Ydata)) {
    stop("kodama_cuda_simpls_rsvd only supports classification factors.", call. = FALSE)
  }
  lev <- levels(Ydata)
  codes <- NULL
  if (length(lev) > 50L) {
    codes <- .gaussian_y_class_codes(lev, 50L, gaussian_y_seed)
  }
  res <- .pls_cv_compiled(
    Xdata, Ydata, constrain, ncomp, kfold, scaling, "simpls", "cuda",
    seed = seed, xprod = xprod, kodama_class_codes = codes, ...
  )
  res$kodama_gaussian_y <- !is.null(codes)
  res$kodama_gaussian_y_dim <- if (is.null(codes)) length(lev) else ncol(codes)
  res
}

#' @rdname plssvd_cv_cpp
#' @export
opls_cv_cuda <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                         north = 1L,
                         scaling = c("centering", "autoscaling", "none"),
                         xprod = NULL, ...) {
  pred_ncomp <- pmax(1L, as.integer(ncomp) - as.integer(north))
  .pls_cv_compiled(Xdata, Ydata, constrain, pred_ncomp, kfold, scaling, "opls", "cuda", xprod = xprod, north = north, ...)
}

#' @rdname plssvd_cv_cpp
#' @export
kernelpls_cv_cuda <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
                              scaling = c("centering", "autoscaling", "none"),
                              xprod = NULL, ...) {
  .pls_cv_compiled(Xdata, Ydata, constrain, ncomp, kfold, scaling, "kernelpls", "cuda", xprod = xprod, ...)
}

kernel_pls_cv_cuda <- kernelpls_cv_cuda

.svd_methods_internal <- c("exact", "irlba", "cpu_rsvd", "cuda_rsvd")
.svd_methods_public <- c("exact", "irlba", "cpu_rsvd")
.svd_methods_cpu <- c("exact", "irlba", "cpu_rsvd")

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
#' @export
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
#' @export
svd_run <- function(A,
                    k,
                    method = c("cpu_rsvd", "irlba", "exact"),
                    rsvd_oversample = 10L,
                    rsvd_power = 1L,
                    svds_tol = 0,
                    seed = 1L,
                    left_only = FALSE) {
  method <- .normalize_svd_method(method)
  method <- match.arg(method)
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
#' @export
svd_benchmark <- function(A,
                          k,
                          methods = c("irlba", "cpu_rsvd", "exact"),
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

.truncated_svd_r <- function(A,
                             k,
                             svd.method = c("irlba", "cpu_rsvd", "exact"),
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
  svd.method <- match.arg(svd.method)
  A <- as.matrix(A)
  k <- as.integer(k)
  if (k < 1L) stop("k must be >= 1")
  max_rank <- min(nrow(A), ncol(A))
  k <- min(k, max_rank)

  full_svd <- function() {
    s <- svd(A, nu = k, nv = k)
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
    s <- svd(A, nu = k, nv = k)
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
  s_small <- svd(B, nu = min(nrow(B), ncol(B)), nv = min(nrow(B), ncol(B)))
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
    s <- svd(S, nu = k, nv = k)
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
  s_small <- svd(B, nu = min(nrow(B), ncol(B)), nv = min(nrow(B), ncol(B)))
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
  s_small <- svd(B, nu = min(nrow(B), ncol(B)), nv = 0L)
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
        R2Y[i_out] <- RQ(Ytrain, Yfit[, , i_out])
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
          R2Y[i_out] <- RQ(Ytrain, Yfit[, , i_out])
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

#' Pure-R PLS reference implementation
#'
#' Pure-R implementation of `plssvd` and the fastPLS `simpls` core with CPU-only SVD choices.
#'
#' @inheritParams pls
#' @param method One of `"simpls"` or `"plssvd"`. `simpls` uses the fastPLS SIMPLS core.
#' @param svd.method One of `"irlba"` or `"cpu_rsvd"`.
#' @return A `fastPLS` object.
#' @export
pls_r = function (Xtrain,
                  Ytrain,
                  Xtest = NULL,
                  Ytest = NULL,
                  ncomp=2,
                  scaling = c("centering", "autoscaling","none"),
                  method = c("simpls", "plssvd"),
                  svd.method = c("irlba", "cpu_rsvd", "exact"),
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
                  fit = FALSE,
  proj = FALSE,
  perm.test = FALSE,
  times = 100) {
  scal <- pmatch(scaling, c("centering", "autoscaling", "none"))[1]
  requested_method <- match.arg(method, c("simpls", "plssvd"))
  meth <- .normalize_pls_method(requested_method)
  svdmeth <- .normalize_svd_method(svd.method)
  svdmeth <- match.arg(svdmeth, c("irlba", "cpu_rsvd", "exact"))
  Xtrain <- as.matrix(Xtrain)
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

  ncomp <- as.integer(ncomp)
  use_xprod_default <- identical(svdmeth, "cpu_rsvd") &&
    meth %in% c(1L, 3L) &&
    .should_use_xprod_default(ncol(Xtrain), ncol(Ytrain), ncomp)
  if (meth == 1L) {
    cap <- .cap_plssvd_ncomp(ncomp, nrow(Xtrain), ncol(Xtrain), ncol(Ytrain), warn = TRUE)
    ncomp <- cap$ncomp
    if (use_xprod_default) {
      model <- .pls_model1_r_xprod(
        Xtrain, Ytrain, ncomp, scal, fit,
        rsvd_oversample, rsvd_power, seed
      )
    } else {
      model <- .pls_model1_r(
        Xtrain, Ytrain, ncomp, scal, fit,
        svdmeth, rsvd_oversample, rsvd_power, svds_tol,
        irlba_work, irlba_maxit, irlba_tol, irlba_eps, irlba_svtol,
        seed
      )
    }
  } else if (meth == 2L) {
    model <- .pls_model2_r(
      Xtrain, Ytrain, ncomp, scal, fit,
      svdmeth, rsvd_oversample, rsvd_power, svds_tol,
      irlba_work, irlba_maxit, irlba_tol, irlba_eps, irlba_svtol,
      seed
    )
  } else {
    if (use_xprod_default) {
      model <- .pls_model2_fast_r_xprod(
        Xtrain, Ytrain, ncomp, scal, fit,
        rsvd_power = rsvd_power,
        seed = seed,
        fast_block = fast_block
      )
    } else {
      model <- .pls_model2_fast_r(
        Xtrain, Ytrain, ncomp, scal, fit,
        svdmeth, rsvd_oversample, rsvd_power, svds_tol,
        irlba_work, irlba_maxit, irlba_tol, irlba_eps, irlba_svtol,
        seed,
        fast_block = fast_block
      )
    }
  }
  model$xprod_default <- use_xprod_default
  model$pls_method <- if (meth == 1L) "plssvd" else "simpls"
  model$predict_latent_ok <- TRUE
  model <- .enable_flash_prediction(model, "cpu")
  model <- .attach_gaussian_y(model, yprep$gaussian)
  model$classification <- classification
  model$lev <- lev
  model <- .decode_gaussian_y_outputs(model, Ytrain_original)

  if (!is.null(Xtest)) {
    Xtest <- as.matrix(Xtest)
    res <- predict.fastPLS(model, Xtest, Ytest = Ytest, proj = proj)
    model <- c(model, res)
    if (perm.test) {
      v <- matrix(NA_real_, nrow = times, ncol = length(ncomp))
      for (i in seq_len(times)) {
        ss <- sample(seq_len(nrow(Xtrain)))
        Xperm <- Xtrain[ss, , drop = FALSE]
        if (meth == 1L) {
          mperm <- .pls_model1_r(Xperm, Ytrain, ncomp, scal, FALSE, svdmeth,
                                 rsvd_oversample, rsvd_power, svds_tol,
                                 irlba_work, irlba_maxit, irlba_tol, irlba_eps, irlba_svtol,
                                 seed + i)
        } else if (meth == 2L) {
          mperm <- .pls_model2_r(Xperm, Ytrain, ncomp, scal, FALSE, svdmeth,
                                 rsvd_oversample, rsvd_power, svds_tol,
                                 irlba_work, irlba_maxit, irlba_tol, irlba_eps, irlba_svtol,
                                 seed + i)
        } else {
          mperm <- .pls_model2_fast_r(Xperm, Ytrain, ncomp, scal, FALSE, svdmeth,
                                      rsvd_oversample, rsvd_power, svds_tol,
                                      irlba_work, irlba_maxit, irlba_tol, irlba_eps, irlba_svtol,
                                      seed + i, fast_block = fast_block)
        }
        mperm <- .attach_gaussian_y(mperm, yprep$gaussian)
        mperm$classification <- classification
        mperm$lev <- lev
        rperm <- predict.fastPLS(mperm, Xtest, Ytest = Ytest, proj = FALSE)
        v[i, ] <- rperm$Q2Y
      }
      model$pval <- vapply(seq_along(ncomp), function(j) sum(v[, j] > model$Q2Y[j]) / times, numeric(1))
    }
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



#' Partial Least Squares with selectable outer algorithm and SVD backend
#'
#' Outer algorithm (`method`) and inner linear algebra backend (`svd.method`) are
#' configurable independently.
#'
#' @param Xtrain Numeric training predictor matrix.
#' @param Ytrain Training response (numeric or factor).
#' @param Xtest Optional test predictor matrix.
#' @param Ytest Optional test response for `Q2Y`.
#' @param ncomp Number of components (scalar or vector).
#' @param scaling One of `"centering"`, `"autoscaling"`, `"none"`.
#' @param method One of `"simpls"` or `"plssvd"`. `simpls` uses the fastPLS SIMPLS core.
#' @param svd.method One of `"exact"`, `"irlba"` or `"cpu_rsvd"`.
#'   The former hybrid CUDA route via `svd.method = "cuda_rsvd"` has been removed
#'   from `pls()`; use [simpls_gpu()] for the experimental GPU-native fit.
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
#' @param proj Return projected `Ttest` when `TRUE`.
#' @param perm.test Run permutation test.
#' @param times Number of permutations.
#' @return A `fastPLS` object.
#' @export
pls =  function (Xtrain,
                 Ytrain,
                 Xtest = NULL,
                 Ytest = NULL,
                 ncomp=2,
                 scaling = c("centering", "autoscaling","none"),
                 method = c("simpls", "plssvd"),
                 svd.method = c("irlba", "cpu_rsvd", "exact"),
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
                 fit = FALSE,
                 proj = FALSE,
                 perm.test = FALSE,
  times = 100)
{

  scal = pmatch(scaling, c("centering", "autoscaling","none"))[1]
  requested_method <- match.arg(method, c("simpls", "plssvd"))
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
        fast_defl_cache=fast_defl_cache
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
        fast_defl_cache=fast_defl_cache
      )
    }
  }
  model$xprod_default=use_xprod_default
  model$pls_method <- if (meth == 1L) "plssvd" else "simpls"
  model$predict_latent_ok <- TRUE
  model <- .enable_flash_prediction(model, "cpu")
  model <- .attach_gaussian_y(model, yprep$gaussian)
  model$classification=classification
  model$lev=lev
  model <- .decode_gaussian_y_outputs(model, Ytrain_original)


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

#' Cross-validation component optimization for PLS
#'
#' Performs k-fold CV over candidate component counts for `simpls` or `plssvd`.
#'
#' @inheritParams pls
#' @param Xdata Predictor matrix.
#' @param Ydata Response (numeric or factor).
#' @param constrain Optional grouping vector for constrained splitting.
#' @param kfold Number of folds.
#' @param method One of `"simpls"` or `"plssvd"`. `simpls` uses the fastPLS SIMPLS core.
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
#' @return List with `optim_comp`, `Ypred`, `Q2Y`, `R2Y`, and `fold`.
#' @export
optim.pls.cv =  function (Xdata,
                          Ydata,
                          ncomp=2,
                          constrain=NULL,
                          scaling = c("centering", "autoscaling","none"),
                          method = c("simpls", "plssvd"),
                          svd.method = c("irlba", "cpu_rsvd", "exact"),
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
                          kfold=10)
{
  scal = pmatch(scaling, c("centering", "autoscaling","none"))[1]
  meth = .normalize_pls_method(method)

  .guard_removed_hybrid_cuda(svd.method, "optim.pls.cv()")
  svd.method <- .normalize_svd_method(svd.method)
  svd.method <- match.arg(svd.method)
  svdmeth <- .svd_method_id(svd.method)
  if(is.null(constrain))
    constrain=1:nrow(Xdata)
  Xdata=as.matrix(Xdata)
  ncomp <- as.integer(ncomp)

  if (is.factor(Ydata)){
    classification=TRUE # classification
    lev = levels(Ydata)
    Ydata = transformy(Ydata)
  } else{
    classification=FALSE   # regression
  }
  if (meth == 1L) {
    cap <- .cap_plssvd_ncomp(ncomp, nrow(Xdata), ncol(Xdata), ncol(Ydata), warn = TRUE)
    ncomp <- cap$ncomp
  }
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
    res <- .with_irlba_options(
      .with_fastpls_fast_options(
        optim_pls_cv(
          Xdata=Xdata,
          Ydata=Ydata,
          constrain=constrain,
          ncomp=ncomp,
          scaling=scal,
          kfold=kfold,
          method=meth,
          svd_method=svdmeth,
          rsvd_oversample=rsvd_oversample,
          rsvd_power=rsvd_power,
          svds_tol=svds_tol,
          seed=seed
        ),
        fast_block = profile$fast_block,
        fast_center_t = profile$fast_center_t,
        fast_reorth_v = profile$fast_reorth_v,
        fast_incremental = profile$fast_incremental,
        fast_inc_iters = profile$fast_inc_iters,
        fast_defl_cache = profile$fast_defl_cache
      ),
      irlba_work = irlba_work,
      irlba_maxit = irlba_maxit,
      irlba_tol = irlba_tol,
      irlba_eps = irlba_eps,
      irlba_svtol = irlba_svtol
    )
  } else {
    res <- .with_irlba_options(
      optim_pls_cv(
        Xdata=Xdata,
        Ydata=Ydata,
        constrain=constrain,
        ncomp=ncomp,
        scaling=scal,
        kfold=kfold,
        method=meth,
        svd_method=svdmeth,
        rsvd_oversample=rsvd_oversample,
        rsvd_power=rsvd_power,
        svds_tol=svds_tol,
        seed=seed
      ),
      irlba_work = irlba_work,
      irlba_maxit = irlba_maxit,
      irlba_tol = irlba_tol,
      irlba_eps = irlba_eps,
      irlba_svtol = irlba_svtol
    )
  }
  res
}






#' Nested cross-validation for PLS
#'
#' Runs outer/inner CV loops for performance estimation and component selection.
#'
#' @inheritParams pls
#' @param Xdata Predictor matrix.
#' @param Ydata Response (numeric or factor).
#' @param constrain Grouping vector for constrained splitting.
#' @param runn Number of repeated runs.
#' @param kfold_inner Inner-fold count.
#' @param kfold_outer Outer-fold count.
#' @param method One of `"simpls"` or `"plssvd"`. `simpls` uses the fastPLS SIMPLS core.
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
#' @return List of nested CV outputs and summaries.
#' @export
pls.double.cv = function(Xdata,
                         Ydata,
                         ncomp=2,
                         constrain=1:nrow(Xdata),
                         scaling = c("centering", "autoscaling","none"),
                         method = c("simpls", "plssvd"),
                         svd.method = c("irlba", "cpu_rsvd", "exact"),
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
                         kfold_outer=10){

  if(sum(is.na(Xdata))>0) {
    stop("Missing values are present")
  }
  scal=pmatch(scaling,c("centering","autoscaling","none"))[1]
  meth = .normalize_pls_method(method)

  .guard_removed_hybrid_cuda(svd.method, "pls.double.cv()")
  svd.method <- .normalize_svd_method(svd.method)
  svd.method <- match.arg(svd.method)
  svdmeth <- .svd_method_id(svd.method)

  if (is.factor(Ydata)){
    classification=TRUE # classification
    lev = levels(Ydata)
    Ydata_original = Ydata
    Ydata = transformy(Ydata)
    conf_tot=matrix(0,ncol=length(lev),nrow=length(lev))
    colnames(conf_tot)=lev
    rownames(conf_tot)=lev
  } else{
    classification=FALSE   # regression
    Ydata = as.matrix(Ydata)
  }
  ncomp <- as.integer(ncomp)
  if (meth == 1L) {
    cap <- .cap_plssvd_ncomp(ncomp, nrow(Xdata), ncol(Xdata), ncol(Ydata), warn = TRUE)
    ncomp <- cap$ncomp
  }

  Xdata=as.matrix(Xdata)
  constrain=as.numeric(as.factor(constrain))

  res=list()
  Q2Y=NULL
  R2Y=NULL
  bcomp=NULL

  bb=NULL
  Ypred_tot=matrix(0,nrow=nrow(Xdata),ncol=ncol(Ydata))
  for(j in 1:runn){


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
      o <- .with_irlba_options(
        .with_fastpls_fast_options(
          double_pls_cv(
            Xdata,
            Ydata,
            ncomp,
            constrain,
            scal,
            kfold_inner,
            kfold_outer,
            meth,
            svd_method=svdmeth,
            rsvd_oversample=rsvd_oversample,
            rsvd_power=rsvd_power,
            svds_tol=svds_tol,
            seed=seed
          ),
          fast_block = profile$fast_block,
          fast_center_t = profile$fast_center_t,
          fast_reorth_v = profile$fast_reorth_v,
          fast_incremental = profile$fast_incremental,
          fast_inc_iters = profile$fast_inc_iters,
          fast_defl_cache = profile$fast_defl_cache
        ),
        irlba_work = irlba_work,
        irlba_maxit = irlba_maxit,
        irlba_tol = irlba_tol,
        irlba_eps = irlba_eps,
        irlba_svtol = irlba_svtol
      )
    } else {
      o <- .with_irlba_options(
        double_pls_cv(
          Xdata,
          Ydata,
          ncomp,
          constrain,
          scal,
          kfold_inner,
          kfold_outer,
          meth,
          svd_method=svdmeth,
          rsvd_oversample=rsvd_oversample,
          rsvd_power=rsvd_power,
          svds_tol=svds_tol,
          seed=seed
        ),
        irlba_work = irlba_work,
        irlba_maxit = irlba_maxit,
        irlba_tol = irlba_tol,
        irlba_eps = irlba_eps,
        irlba_svtol = irlba_svtol
      )
    }
    Ypred_tot=Ypred_tot+o$Ypred
    if(classification){
      t = apply(o$Ypred, 1, which.max)
      Ypredlab = factor(lev[t], levels = lev)

      o$Ypred=Ypredlab

      o$conf=table(Ypredlab,Ydata_original)
      conf_tot=conf_tot+o$conf
      o$acc=(sum(diag(o$conf))*100)/length(Ydata)
    }
    #  o$R2X=diag((t(o$T)%*%(o$T))%*%(t(o$P)%*%(o$P)))/sum(scale(Xdata,TRUE,TRUE)^2)
    Q2Y[j]=o$Q2Y
    R2Y[j]=mean(o$R2Y)
    res$results[[j]]=o
    bb=c(bb,o$optim_comp)
  }
  Ypred_tot=Ypred_tot/runn

  if(classification){
    conf_tot=conf_tot/runn
    acc_tot=round(sum(diag(conf_tot)),digits=1)
    acc_tot_perc=100*acc_tot/nrow(Xdata)
    acc_tot_txt=paste(acc_tot," (",acc_tot_perc,"%)",sep="")


    conf_tot_perc=t(t(conf_tot)/colSums(conf_tot))*100
    conf_tot=round(conf_tot,digits=1)
    conf_tot_perc=round(conf_tot_perc,digits=1)

    conf_txt=matrix(paste(conf_tot," (",conf_tot_perc,"%)",sep=""),ncol=length(lev))
    colnames(conf_txt)=lev
    rownames(conf_txt)=lev
    res$acc_tot=acc_tot_txt
    res$conf=conf_txt

    t = apply(Ypred_tot, 1, which.max)
    Ypredlab = factor(lev[t], levels = lev)

    res$Ypred=Ypredlab
  }

    res$Q2Y=Q2Y
    res$R2Y=R2Y
    res$medianR2Y=median(R2Y)
    res$CI95R2Y=as.numeric(quantile(R2Y,c(0.025,0.975)))
    res$medianQ2Y=median(Q2Y)
    res$CI95Q2Y=as.numeric(quantile(Q2Y,c(0.025,0.975)))


    res$bcomp=names(which.max(table(bb)))

    if(perm.test){

      v=NULL

      for(i in 1:times){
        ss=sample(1:nrow(Xdata))
        w=NULL
        for(ii in 1:runn)
          if (meth == 3L) {
            w[ii] <- .with_irlba_options(
              .with_fastpls_fast_options(
                double_pls_cv(
                  Xdata[ss,],
                  Ydata,
                  ncomp,
                  constrain,
                  scal,
                  kfold_inner,
                  kfold_outer,
                  meth,
                  svd_method=svdmeth,
                  rsvd_oversample=rsvd_oversample,
                  rsvd_power=rsvd_power,
                  svds_tol=svds_tol,
                  seed=seed
                ),
                fast_block = fast_block,
                fast_center_t = fast_center_t,
                fast_reorth_v = fast_reorth_v,
                fast_incremental = fast_incremental,
                fast_inc_iters = fast_inc_iters,
                fast_defl_cache = fast_defl_cache
              ),
              irlba_work = irlba_work,
              irlba_maxit = irlba_maxit,
              irlba_tol = irlba_tol,
              irlba_eps = irlba_eps,
              irlba_svtol = irlba_svtol
            )$Q2Y
          } else {
            w[ii] <- .with_irlba_options(
              double_pls_cv(
                Xdata[ss,],
                Ydata,
                ncomp,
                constrain,
                scal,
                kfold_inner,
                kfold_outer,
                meth,
                svd_method=svdmeth,
                rsvd_oversample=rsvd_oversample,
                rsvd_power=rsvd_power,
                svds_tol=svds_tol,
                seed=seed
              ),
              irlba_work = irlba_work,
              irlba_maxit = irlba_maxit,
              irlba_tol = irlba_tol,
              irlba_eps = irlba_eps,
              irlba_svtol = irlba_svtol
            )$Q2Y
          }

        v[i]=median(w)
      }
      pval=pnorm(median(Q2Y), mean=mean(v), sd=sqrt(((length(v)-1)/length(v))*var(v)), lower.tail=FALSE)
      res$Q2Ysampled=v
      res$p.value=pval
    }


    if (classification) {
      conf_tot=matrix(0,ncol=length(lev),nrow=length(lev))
      colnames(conf_tot)=lev
      rownames(conf_tot)=lev
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
