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

.with_fastpls_fast_options <- function(expr,
                                       fast_block = 4L,
                                       fast_center_t = FALSE,
                                       fast_reorth_v = TRUE,
                                       fast_incremental = FALSE,
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
    FASTPLS_FAST_INCREMENTAL = if (isTRUE(fast_incremental)) "1" else "0",
    FASTPLS_FAST_INC_ITERS = as.character(as.integer(fast_inc_iters)),
    FASTPLS_FAST_DEFLCACHE = if (isTRUE(fast_defl_cache)) "1" else "0"
  )
  force(expr)
}

.with_irlba_options <- function(expr,
                                irlba_work = 0L,
                                irlba_maxit = 2000L,
                                irlba_tol = 1e-6,
                                irlba_eps = 1e-9,
                                irlba_svtol = 1e-6) {
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

.normalize_svd_method <- function(method) {
  if (length(method) == 1L && !is.na(method) && identical(as.character(method), "dc")) {
    warning("svd.method='dc' is deprecated; use 'arpack' instead.", call. = FALSE)
    return("arpack")
  }
  method
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
            irlba_maxit = 2000L,
            irlba_tol = 1e-6,
            irlba_eps = 1e-9,
            irlba_svtol = 1e-6,
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
            irlba_maxit = 2000L,
            irlba_tol = 1e-6,
            irlba_eps = 1e-9,
            irlba_svtol = 1e-6,
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
            irlba_maxit = 2000L,
            irlba_tol = 1e-6,
            irlba_eps = 1e-9,
            irlba_svtol = 1e-6,
            seed = 1L,
            fast_block = 4L,
            fast_center_t = FALSE,
            fast_reorth_v = TRUE,
            fast_incremental = FALSE,
            fast_inc_iters = 2L,
            fast_defl_cache = TRUE)
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
    )
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
#' @param ... Unused.
#' @return A list containing `Ypred`, optional `Q2Y`, and optional `Ttest`.
#' @export
predict.fastPLS = function(object, newdata, Ytest=NULL, proj=FALSE, ...) {
  if (!is(object, "fastPLS")) {
    stop("object is not a fastPLS object")
  }
  Xtest=newdata
  res=pls_predict(object, Xtest,proj)
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
      res$Ypred[, , i]=t(t(res$Ypred[, , i])+as.numeric(object$mY))
      res$Q2Y[i] = RQ(Ytest_transf,res$Ypred[, , i])
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

.svd_methods_all <- c("irlba", "arpack", "cpu_rsvd", "cuda_rsvd", "dc")
.svd_methods_cpu <- c("irlba", "arpack", "cpu_rsvd", "dc")

.svd_method_id <- function(method) {
  method <- match.arg(method, .svd_methods_all)
  if (identical(method, "dc")) {
    warning("svd.method='dc' is deprecated; use 'arpack' instead.", call. = FALSE)
    method <- "arpack"
  }
  switch(
    method,
    irlba = 1L,
    arpack = 2L,
    dc = 2L,
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
  methods <- setdiff(.svd_methods_all, "dc")
  enabled <- rep(TRUE, length(methods))
  names(enabled) <- methods
  enabled["cuda_rsvd"] <- has_cuda()
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
#' @param svds_tol Tolerance passed to ARPACK `svds()` when `svd.method = "arpack"`.
#'   Larger values can reduce runtime but may reduce numerical precision. Ignored by
#'   non-ARPACK backends.
#' @param seed RSVD seed.
#' @param left_only Return left singular vectors only.
#' @return List with `U`, `s`, `Vt`, `method`, and elapsed time.
#' @export
svd_run <- function(A,
                    k,
                    method = c("arpack", "cpu_rsvd", "irlba", "cuda_rsvd"),
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
  if (method == "cuda_rsvd" && !has_cuda()) {
    stop("svd.method='cuda_rsvd' requested, but CUDA backend is not available")
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
#' @param svds_tol Tolerance passed to ARPACK `svds()` when applicable.
#' @param seed RSVD seed.
#' @param left_only Return left singular vectors only.
#' @return Data frame with `method`, `rep`, `elapsed`, and `status`.
#' @export
svd_benchmark <- function(A,
                          k,
                          methods = c("irlba", "arpack", "cpu_rsvd", "cuda_rsvd"),
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
    if (method == "cuda_rsvd" && !has_cuda()) {
      out[[i]] <- data.frame(
        method = method,
        rep = NA_integer_,
        elapsed = NA_real_,
        status = "unavailable",
        stringsAsFactors = FALSE
      )
      next
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
                             svd.method = c("irlba", "arpack", "cpu_rsvd"),
                             rsvd_oversample = 10L,
                             rsvd_power = 1L,
                             svds_tol = 0,
                             seed = 1L) {
  svd.method <- .normalize_svd_method(svd.method)
  svd.method <- match.arg(svd.method)
  A <- as.matrix(A)
  k <- as.integer(k)
  if (k < 1L) stop("k must be >= 1")
  max_rank <- min(nrow(A), ncol(A))
  k <- min(k, max_rank)

  if (svd.method != "cpu_rsvd") {
    s <- svd(A, nu = k, nv = k)
    return(list(
      u = s$u[, seq_len(k), drop = FALSE],
      d = s$d[seq_len(k)],
      v = s$v[, seq_len(k), drop = FALSE]
    ))
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

.pls_model1_r <- function(Xtrain,
                          Ytrain,
                          ncomp,
                          scaling,
                          fit,
                          svd.method,
                          rsvd_oversample,
                          rsvd_power,
                          svds_tol,
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
  s <- .truncated_svd_r(S, max_ncomp_eff, svd.method, rsvd_oversample, rsvd_power, svds_tol, seed)
  max_ncomp_eff <- min(max_ncomp_eff, ncol(s$u), ncol(s$v))
  R <- s$u[, seq_len(max_ncomp_eff), drop = FALSE]
  Q <- s$v[, seq_len(max_ncomp_eff), drop = FALSE]
  Ttrain <- Xtrain %*% R

  B <- array(0, dim = c(p, m, length_ncomp))
  Yfit <- if (fit) array(0, dim = c(n, m, length_ncomp)) else NULL
  R2Y <- rep(NA_real_, length_ncomp)

  for (i in seq_len(length_ncomp)) {
    mc <- min(ncomp[i], max_ncomp_eff)
    R_mc <- R[, seq_len(mc), drop = FALSE]
    Q_mc <- Q[, seq_len(mc), drop = FALSE]
    T_mc <- Ttrain[, seq_len(mc), drop = FALSE]
    U_mc <- Ytrain %*% Q_mc
    B[, , i] <- R_mc %*% (solve(crossprod(T_mc), t(T_mc) %*% U_mc)) %*% t(Q_mc)
    if (fit) {
      yf <- Xtrain %*% B[, , i]
      R2Y[i] <- RQ(Ytrain, yf)
      Yfit[, , i] <- sweep(yf, 2, mY[1, ], "+")
    }
  }

  out <- list(
    B = B,
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
  B <- array(0, dim = c(p, m, length_ncomp))
  Yfit <- if (fit) array(0, dim = c(n, m, length_ncomp)) else NULL
  R2Y <- rep(NA_real_, length_ncomp)

  i_out <- 1L
  for (a in seq_len(max_ncomp)) {
    rr <- .truncated_svd_r(S, 1L, svd.method, rsvd_oversample, rsvd_power, svds_tol, seed + a - 1L)$u[, 1, drop = FALSE]
    tt <- X %*% rr
    tt <- sweep(tt, 2, colMeans(tt), "-")
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
      B[, , i_out] <- RR_a %*% t(QQ_a)
      if (fit) {
        yf <- Xtrain %*% B[, , i_out]
        Yfit[, , i_out] <- sweep(yf, 2, mY[1, ], "+")
        R2Y[i_out] <- RQ(Ytrain, Yfit[, , i_out])
      }
      i_out <- i_out + 1L
      if (i_out > length_ncomp) break
    }
  }

  out <- list(
    B = B,
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
  class(out) <- "fastPLS"
  out
}

#' Pure-R PLS reference implementation
#'
#' Pure-R implementation of `plssvd` and `simpls` with CPU-only SVD choices.
#'
#' @inheritParams pls
#' @param method One of `"simpls"` or `"plssvd"`.
#' @param svd.method One of `"irlba"`, `"arpack"`, `"cpu_rsvd"` (with `"dc"` kept as a deprecated alias for `"arpack"`).
#' @return A `fastPLS` object.
#' @export
pls_r = function (Xtrain,
                  Ytrain,
                  Xtest = NULL,
                  Ytest = NULL,
                  ncomp=2,
                  scaling = c("centering", "autoscaling","none"),
                  method = c("simpls", "plssvd"),
                  svd.method = c("irlba", "arpack", "cpu_rsvd"),
                  rsvd_oversample = 10L,
                  rsvd_power = 1L,
                  svds_tol = 0,
                  irlba_work = 0L,
                  irlba_maxit = 2000L,
                  irlba_tol = 1e-6,
                  irlba_eps = 1e-9,
                  irlba_svtol = 1e-6,
                  seed = 1L,
                  fit = FALSE,
                  proj = FALSE,
                  perm.test = FALSE,
                  times = 100) {
  scal <- pmatch(scaling, c("centering", "autoscaling", "none"))[1]
  meth <- pmatch(method, c("plssvd", "simpls"))[1]
  svdmeth <- .normalize_svd_method(svd.method)
  svdmeth <- match.arg(svdmeth, c("irlba", "arpack", "cpu_rsvd"))
  Xtrain <- as.matrix(Xtrain)

  if (is.factor(Ytrain)) {
    classification <- TRUE
    lev <- levels(Ytrain)
    Ytrain <- transformy(Ytrain)
  } else {
    classification <- FALSE
    lev <- NULL
    Ytrain <- as.matrix(Ytrain)
  }

  ncomp <- as.integer(ncomp)
  if (meth == 1L) {
    cap <- .cap_plssvd_ncomp(ncomp, nrow(Xtrain), ncol(Xtrain), ncol(Ytrain), warn = TRUE)
    ncomp <- cap$ncomp
    model <- .pls_model1_r(
      Xtrain, Ytrain, ncomp, scal, fit,
      svdmeth, rsvd_oversample, rsvd_power, svds_tol, seed
    )
  } else {
    model <- .pls_model2_r(
      Xtrain, Ytrain, ncomp, scal, fit,
      svdmeth, rsvd_oversample, rsvd_power, svds_tol, seed
    )
  }
  model$classification <- classification
  model$lev <- lev

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
                                 rsvd_oversample, rsvd_power, svds_tol, seed + i)
        } else {
          mperm <- .pls_model2_r(Xperm, Ytrain, ncomp, scal, FALSE, svdmeth,
                                 rsvd_oversample, rsvd_power, svds_tol, seed + i)
        }
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
#' @param method One of `"simpls"`, `"plssvd"`, `"simpls_fast"`.
#' @param svd.method One of `"irlba"`, `"arpack"`, `"cpu_rsvd"`, `"cuda_rsvd"` (with `"dc"` kept as a deprecated alias for `"arpack"`).
#' @param rsvd_oversample RSVD oversampling.
#' @param rsvd_power RSVD power iterations.
#' @param svds_tol Tolerance passed to ARPACK `svds()` when `svd.method = "arpack"`.
#'   Larger values can improve speed at the cost of looser convergence.
#' @param seed RSVD seed.
#' @param fast_block `simpls_fast` block refresh size.
#' @param fast_center_t `simpls_fast` score centering toggle.
#' @param fast_reorth_v `simpls_fast` re-orthogonalization toggle.
#' @param fast_incremental `simpls_fast` incremental block-power toggle.
#' @param fast_inc_iters `simpls_fast` incremental power iterations.
#' @param fast_defl_cache `simpls_fast` cached deflation toggle.
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
                 method = c("simpls", "plssvd", "simpls_fast"),
                 svd.method = c("irlba", "arpack", "cpu_rsvd", "cuda_rsvd"),
                 rsvd_oversample = 10L,
                 rsvd_power = 1L,
                 svds_tol = 0,
                 irlba_work = 0L,
                 irlba_maxit = 2000L,
                 irlba_tol = 1e-6,
                 irlba_eps = 1e-9,
                 irlba_svtol = 1e-6,
                 seed = 1L,
                 fast_block = 4L,
                 fast_center_t = FALSE,
                 fast_reorth_v = TRUE,
                 fast_incremental = FALSE,
                 fast_inc_iters = 2L,
                 fast_defl_cache = TRUE,
                 fit = FALSE,
                 proj = FALSE, 
                 perm.test = FALSE, 
                 times = 100) 
{

  scal = pmatch(scaling, c("centering", "autoscaling","none"))[1]
  meth = pmatch(method, c("plssvd", "simpls", "simpls_fast"))[1]
  svd.method <- .normalize_svd_method(svd.method)
  svd.method <- match.arg(svd.method)
  svdmeth <- .svd_method_id(svd.method)
  if (svd.method == "cuda_rsvd" && !has_cuda()) {
    stop("svd.method='cuda_rsvd' requested, but CUDA backend is not available")
  }

  # Tuned CUDA fast profile discovered by cycle benchmarking.
  # Apply only for SIMPLS fast + CUDA when caller did not explicitly set values.
  if (meth == 3L && svd.method == "cuda_rsvd") {
    if (missing(rsvd_oversample)) rsvd_oversample <- 8L
    if (missing(rsvd_power)) rsvd_power <- 0L
    if (missing(fast_block)) fast_block <- 8L
    if (missing(fast_center_t)) fast_center_t <- FALSE
    if (missing(fast_reorth_v)) fast_reorth_v <- FALSE
    if (missing(fast_incremental)) fast_incremental <- TRUE
    if (missing(fast_inc_iters)) fast_inc_iters <- 2L
    if (missing(fast_defl_cache)) fast_defl_cache <- TRUE
  }
  
  Xtrain = as.matrix(Xtrain)
  if (is.factor(Ytrain)){
    classification=TRUE # classification
    lev = levels(Ytrain)
    Ytrain = transformy(Ytrain)
    
  } else{
    classification=FALSE   # regression
    lev=NULL
  }
  if(meth==1){
    cap <- .cap_plssvd_ncomp(ncomp, nrow(Xtrain), ncol(Xtrain), ncol(Ytrain), warn = TRUE)
    ncomp <- cap$ncomp
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
  model$classification=classification
  model$lev=lev

  
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
          
          res_perm=predict(model,Xtest,Ytest)

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
#' @param method One of `"simpls"`, `"plssvd"`, or `"simpls_fast"`.
#' @param fast_block `simpls_fast` block refresh size.
#' @param fast_center_t `simpls_fast` score centering toggle.
#' @param fast_reorth_v `simpls_fast` re-orthogonalization toggle.
#' @param fast_incremental `simpls_fast` incremental block-power toggle.
#' @param fast_inc_iters `simpls_fast` incremental power iterations.
#' @param fast_defl_cache `simpls_fast` cached deflation toggle.
#' @return List with `optim_comp`, `Ypred`, `Q2Y`, `R2Y`, and `fold`.
#' @export
optim.pls.cv =  function (Xdata,
                          Ydata, 
                          ncomp=2, 
                          constrain=NULL,
                          scaling = c("centering", "autoscaling","none"),
                          method = c("simpls", "plssvd", "simpls_fast"),
                          svd.method = c("irlba", "arpack", "cpu_rsvd", "cuda_rsvd"),
                          rsvd_oversample = 10L,
                          rsvd_power = 1L,
                          svds_tol = 0,
                          irlba_work = 0L,
                          irlba_maxit = 2000L,
                          irlba_tol = 1e-6,
                          irlba_eps = 1e-9,
                          irlba_svtol = 1e-6,
                          seed = 1L,
                          fast_block = 4L,
                          fast_center_t = FALSE,
                          fast_reorth_v = TRUE,
                          fast_incremental = FALSE,
                          fast_inc_iters = 2L,
                          fast_defl_cache = TRUE,
                          kfold=10) 
{
  scal = pmatch(scaling, c("centering", "autoscaling","none"))[1]
  meth = pmatch(method, c("plssvd", "simpls", "simpls_fast"))[1]
  
  svd.method <- .normalize_svd_method(svd.method)
  svd.method <- match.arg(svd.method)
  svdmeth <- .svd_method_id(svd.method)
  if (svd.method == "cuda_rsvd" && !has_cuda()) {
    stop("svd.method='cuda_rsvd' requested, but CUDA backend is not available")
  }
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
#' @param method One of `"simpls"`, `"plssvd"`, or `"simpls_fast"`.
#' @param fast_block `simpls_fast` block refresh size.
#' @param fast_center_t `simpls_fast` score centering toggle.
#' @param fast_reorth_v `simpls_fast` re-orthogonalization toggle.
#' @param fast_incremental `simpls_fast` incremental block-power toggle.
#' @param fast_inc_iters `simpls_fast` incremental power iterations.
#' @param fast_defl_cache `simpls_fast` cached deflation toggle.
#' @return List of nested CV outputs and summaries.
#' @export
pls.double.cv = function(Xdata,
                         Ydata,
                         ncomp=2,
                         constrain=1:nrow(Xdata),
                         scaling = c("centering", "autoscaling","none"), 
                         method = c("simpls", "plssvd", "simpls_fast"),
                         svd.method = c("irlba", "arpack", "cpu_rsvd", "cuda_rsvd"),
                         rsvd_oversample = 10L,
                         rsvd_power = 1L,
                         svds_tol = 0,
                         irlba_work = 0L,
                         irlba_maxit = 2000L,
                         irlba_tol = 1e-6,
                         irlba_eps = 1e-9,
                         irlba_svtol = 1e-6,
                         seed = 1L,
                         fast_block = 4L,
                         fast_center_t = FALSE,
                         fast_reorth_v = TRUE,
                         fast_incremental = FALSE,
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
  meth = pmatch(method, c("plssvd", "simpls", "simpls_fast"))[1]
  
  svd.method <- .normalize_svd_method(svd.method)
  svd.method <- match.arg(svd.method)
  svdmeth <- .svd_method_id(svd.method)
  if (svd.method == "cuda_rsvd" && !has_cuda()) {
    stop("svd.method='cuda_rsvd' requested, but CUDA backend is not available")
  }
  
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
  if (u==1) return (Vip(model))
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
