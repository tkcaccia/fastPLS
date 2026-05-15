#!/usr/bin/env Rscript

# Benchmark independent PLS/SIMPLS/NIPALS/kernel PLS/OPLS implementations on
# MetRef at ncomp = 22. Packages that only wrap pls::plsr()/pls::mvr() are not
# included; each runner below calls either fastPLS or a package's own fitting
# routine directly.

options(stringsAsFactors = FALSE)

parse_args <- function(args = commandArgs(trailingOnly = TRUE)) {
  out <- list()
  for (arg in args) {
    if (!startsWith(arg, "--")) next
    kv <- substring(arg, 3L)
    bits <- strsplit(kv, "=", fixed = TRUE)[[1L]]
    key <- gsub("-", "_", bits[[1L]], fixed = TRUE)
    val <- if (length(bits) > 1L) paste(bits[-1L], collapse = "=") else "TRUE"
    out[[key]] <- val
  }
  out
}

args <- parse_args()
arg_value <- function(key, default = NULL) {
  val <- args[[key]]
  if (is.null(val) || !nzchar(val)) default else val
}

ncomp_requested <- as.integer(arg_value("ncomp", Sys.getenv("FASTPLS_NCOMP", "22")))
if (!is.finite(ncomp_requested) || is.na(ncomp_requested) || ncomp_requested < 1L) {
  ncomp_requested <- 22L
}

reps <- as.integer(arg_value("reps", Sys.getenv("FASTPLS_REPS", "3")))
if (!is.finite(reps) || is.na(reps) || reps < 1L) reps <- 3L

split_seed <- as.integer(arg_value("seed", Sys.getenv("FASTPLS_SEED", "123")))
if (!is.finite(split_seed) || is.na(split_seed)) split_seed <- 123L

test_n <- as.integer(arg_value("test_n", Sys.getenv("FASTPLS_TEST_N", "100")))
if (!is.finite(test_n) || is.na(test_n) || test_n < 1L) test_n <- 100L

outfile <- arg_value(
  "out",
  Sys.getenv("FASTPLS_OUTFILE", "metref_pls_opls_speed_accuracy_ncomp22.csv")
)

install_missing <- tolower(Sys.getenv("FASTPLS_INSTALL_MISSING", "false")) %in%
  c("1", "true", "yes", "y")

quiet_require <- function(pkg) {
  suppressPackageStartupMessages(requireNamespace(pkg, quietly = TRUE))
}

maybe_install <- function(pkg, bioc = FALSE) {
  if (quiet_require(pkg) || !isTRUE(install_missing)) return(quiet_require(pkg))
  msg <- sprintf("Package %s is missing; FASTPLS_INSTALL_MISSING=true, trying to install.", pkg)
  message(msg)
  ok <- tryCatch({
    if (isTRUE(bioc)) {
      if (!quiet_require("BiocManager")) {
        utils::install.packages("BiocManager", repos = "https://cloud.r-project.org")
      }
      BiocManager::install(pkg, ask = FALSE, update = FALSE)
    } else {
      utils::install.packages(pkg, repos = "https://cloud.r-project.org")
    }
    quiet_require(pkg)
  }, error = function(e) {
    message("Installation failed for ", pkg, ": ", conditionMessage(e))
    FALSE
  })
  isTRUE(ok)
}

for (pkg in c(
  "fastPLS", "KODAMA", "pls", "peakRAM", "mdatools", "plsdepot", "pcv",
  "plsgenomics", "chemometrics", "spls", "plsRglm", "enpls"
)) {
  maybe_install(pkg, bioc = FALSE)
}
maybe_install("ropls", bioc = TRUE)
maybe_install("mixOmics", bioc = TRUE)

if (!quiet_require("fastPLS")) stop("fastPLS is required for this benchmark.")
if (!quiet_require("pls")) stop("pls is required as the reference implementation.")

safe_factor <- function(x) {
  if (is.factor(x)) return(droplevels(x))
  droplevels(factor(x))
}

make_metref_from_kodama <- function(split_seed, test_n) {
  if (!quiet_require("KODAMA")) {
    stop("KODAMA is required when no MetRef RData task is supplied.")
  }
  data("MetRef", package = "KODAMA")
  X <- MetRef$data
  X <- X[, colSums(X) != 0, drop = FALSE]
  X <- KODAMA::normalization(X)$newXtrain
  y <- safe_factor(MetRef$donor)
  set.seed(split_seed)
  test_n <- min(as.integer(test_n), max(1L, floor(nrow(X) / 3L)))
  test_idx <- sort(sample(seq_len(nrow(X)), test_n))
  train_idx <- setdiff(seq_len(nrow(X)), test_idx)
  list(
    dataset_path = "KODAMA::MetRef",
    Xtrain = as.matrix(X[train_idx, , drop = FALSE]),
    Ytrain = droplevels(y[train_idx]),
    Xtest = as.matrix(X[test_idx, , drop = FALSE]),
    Ytest = factor(y[test_idx], levels = levels(droplevels(y[train_idx]))),
    split_seed = split_seed
  )
}

task_from_loaded_env <- function(e, path, split_seed, test_n) {
  objs <- ls(e)
  if ("out" %in% objs && is.list(e$out) &&
      all(c("Xtrain", "Ytrain", "Xtest", "Ytest") %in% names(e$out))) {
    return(list(
      dataset_path = normalizePath(path, winslash = "/", mustWork = TRUE),
      Xtrain = as.matrix(e$out$Xtrain),
      Ytrain = safe_factor(e$out$Ytrain),
      Xtest = as.matrix(e$out$Xtest),
      Ytest = factor(e$out$Ytest, levels = levels(safe_factor(e$out$Ytrain))),
      split_seed = split_seed
    ))
  }
  if (all(c("Xtrain", "Ytrain", "Xtest", "Ytest") %in% objs)) {
    return(list(
      dataset_path = normalizePath(path, winslash = "/", mustWork = TRUE),
      Xtrain = as.matrix(e$Xtrain),
      Ytrain = safe_factor(e$Ytrain),
      Xtest = as.matrix(e$Xtest),
      Ytest = factor(e$Ytest, levels = levels(safe_factor(e$Ytrain))),
      split_seed = split_seed
    ))
  }
  if ("MetRef" %in% objs && is.list(e$MetRef) && all(c("data", "donor") %in% names(e$MetRef))) {
    X <- e$MetRef$data
    X <- X[, colSums(X) != 0, drop = FALSE]
    if (quiet_require("KODAMA")) X <- KODAMA::normalization(X)$newXtrain
    y <- safe_factor(e$MetRef$donor)
    set.seed(split_seed)
    test_n <- min(as.integer(test_n), max(1L, floor(nrow(X) / 3L)))
    test_idx <- sort(sample(seq_len(nrow(X)), test_n))
    train_idx <- setdiff(seq_len(nrow(X)), test_idx)
    return(list(
      dataset_path = normalizePath(path, winslash = "/", mustWork = TRUE),
      Xtrain = as.matrix(X[train_idx, , drop = FALSE]),
      Ytrain = droplevels(y[train_idx]),
      Xtest = as.matrix(X[test_idx, , drop = FALSE]),
      Ytest = factor(y[test_idx], levels = levels(droplevels(y[train_idx]))),
      split_seed = split_seed
    ))
  }
  NULL
}

find_metref_rdata <- function() {
  candidates <- unique(Filter(nzchar, c(
    Sys.getenv("FASTPLS_METREF_RDATA", ""),
    arg_value("metref_rdata", ""),
    file.path(getwd(), "metref.RData"),
    file.path(getwd(), "benchmark", "metref.RData"),
    "/Users/stefano/Documents/GPUPLS/Data/metref_remote_task.RData",
    "/Users/stefano/Documents/GPUPLS/remote_fastpls_data/metref.RData",
    path.expand("~/Documents/GPUPLS/Data/metref.RData"),
    path.expand("~/GPUPLS/Data/metref.RData")
  )))
  candidates[file.exists(candidates)][1L]
}

load_metref_task <- function(split_seed, test_n) {
  path <- find_metref_rdata()
  if (length(path) && !is.na(path) && file.exists(path)) {
    e <- new.env(parent = emptyenv())
    load(path, envir = e)
    task <- task_from_loaded_env(e, path, split_seed, test_n)
    if (!is.null(task)) return(task)
  }
  make_metref_from_kodama(split_seed, test_n)
}

task <- load_metref_task(split_seed, test_n)
Xtrain <- task$Xtrain
Ytrain <- droplevels(task$Ytrain)
Xtest <- task$Xtest
Ytest <- factor(task$Ytest, levels = levels(Ytrain))

if (anyNA(Ytest)) {
  keep <- !is.na(Ytest)
  Xtest <- Xtest[keep, , drop = FALSE]
  Ytest <- droplevels(Ytest[keep])
}

class_levels <- levels(Ytrain)
Ytrain_dummy <- stats::model.matrix(~ Ytrain - 1)
colnames(Ytrain_dummy) <- class_levels

accuracy <- function(pred) {
  pred <- factor(pred, levels = levels(Ytest))
  mean(as.character(pred) == as.character(Ytest), na.rm = TRUE)
}

decode_scores <- function(scores, levels = class_levels) {
  if (is.null(scores)) stop("No prediction scores returned.")
  scores <- as.matrix(scores)
  if (!nrow(scores)) stop("Prediction score matrix has zero rows.")
  if (ncol(scores) == 1L) {
    return(factor(ifelse(scores[, 1L] >= 0.5, levels[2L], levels[1L]), levels = levels))
  }
  idx <- max.col(scores, ties.method = "first")
  factor(levels[idx], levels = levels)
}

last_component_matrix <- function(x, ncomp_eff = ncomp_requested, n_classes = length(class_levels)) {
  if (is.null(x)) stop("No prediction matrix/array supplied.")
  if (length(dim(x)) == 3L) {
    d <- dim(x)
    dn <- dimnames(x)
    names2 <- tolower(dn[[2L]] %||% character())
    names3 <- tolower(dn[[3L]] %||% character())
    class_names <- tolower(class_levels)
    dim2_is_comp <- length(names2) && any(grepl("^comp", names2))
    dim3_is_comp <- length(names3) && any(grepl("^comp", names3))
    dim2_is_class <- length(names2) && all(names2 %in% class_names)
    dim3_is_class <- length(names3) && all(names3 %in% class_names)
    if (isTRUE(dim2_is_comp) || (d[3L] == n_classes && !isTRUE(dim2_is_class))) {
      ncomp_eff <- as.integer(min(ncomp_eff, d[2L]))
      return(as.matrix(x[, ncomp_eff, , drop = TRUE]))
    }
    if (isTRUE(dim3_is_comp) || isTRUE(dim2_is_class) || d[2L] == n_classes) {
      ncomp_eff <- as.integer(min(ncomp_eff, d[3L]))
      return(as.matrix(x[, , ncomp_eff, drop = TRUE]))
    }
    ncomp_eff <- as.integer(min(ncomp_eff, max(d[2L], d[3L], na.rm = TRUE)))
    if (d[2L] >= ncomp_eff) return(as.matrix(x[, ncomp_eff, , drop = TRUE]))
    if (d[3L] >= ncomp_eff) return(as.matrix(x[, , ncomp_eff, drop = TRUE]))
  }
  as.matrix(x)
}

decode_fastpls <- function(model) {
  yp <- model$Ypred
  if (is.null(yp)) {
    pred <- predict(model, Xtest, Ytest = Ytest)$Ypred
    yp <- pred
  }
  if (is.data.frame(yp)) return(factor(yp[[ncol(yp)]], levels = levels(Ytest)))
  if (is.factor(yp)) return(factor(yp, levels = levels(Ytest)))
  if (is.vector(yp) && !is.list(yp)) return(factor(yp, levels = levels(Ytest)))
  if (length(dim(yp)) == 3L) {
    mat <- yp[, , dim(yp)[3L], drop = TRUE]
    return(decode_scores(mat, levels = model$lev %||% class_levels))
  }
  if (is.matrix(yp)) {
    if (ncol(yp) == length(class_levels)) return(decode_scores(yp, levels = class_levels))
    return(factor(yp[, ncol(yp)], levels = levels(Ytest)))
  }
  stop("Unsupported fastPLS prediction format.")
}

`%||%` <- function(a, b) if (is.null(a)) b else a

predict_from_pls_fit <- function(fit, Xnew, ncomp_eff) {
  ncomp_eff <- min(as.integer(ncomp_eff), dim(fit$coefficients)[3L])
  coef <- fit$coefficients[, , ncomp_eff, drop = TRUE]
  if (is.null(dim(coef))) coef <- matrix(coef, ncol = length(fit$Ymeans))
  Xc <- sweep(as.matrix(Xnew), 2L, fit$Xmeans, "-")
  pred <- Xc %*% coef
  pred <- sweep(pred, 2L, fit$Ymeans, "+")
  decode_scores(pred, class_levels)
}

fastpls_algorithm_label <- function(method_name) {
  switch(method_name,
         plssvd = "PLSSVD",
         simpls = "SIMPLS",
         opls = "OPLS",
         kernelpls = "kernel PLS",
         method_name)
}

run_fastpls_variant <- function(method_name, backend, svd_method, classifier) {
  args <- list(
    Xtrain = Xtrain,
    Ytrain = Ytrain,
    Xtest = Xtest,
    Ytest = Ytest,
    ncomp = ncomp_requested,
    method = method_name,
    scaling = "centering",
    backend = backend,
    classifier = classifier,
    fit = FALSE,
    proj = FALSE,
    return_variance = FALSE
  )
  if (identical(backend, "cpp")) {
    args$svd.method <- svd_method
  }
  do.call(fastPLS::pls, args)
}

make_fastpls_spec <- function(method_name, backend, svd_method, classifier) {
  force(method_name)
  force(backend)
  force(svd_method)
  force(classifier)
  svd_label <- if (identical(backend, "cuda")) "cuda_rsvd" else svd_method
  list(
    id = paste("fastPLS", method_name, backend, svd_label, classifier, sep = "_"),
    package = "fastPLS",
    algorithm = fastpls_algorithm_label(method_name),
    function_name = sprintf(
      "fastPLS::pls(method='%s', backend='%s', svd.method='%s', classifier='%s')",
      method_name, backend, svd_label, classifier
    ),
    independent = TRUE,
    fastpls_method = method_name,
    fastpls_backend = backend,
    fastpls_svd_method = svd_label,
    fastpls_classifier = classifier,
    requires_cuda = identical(backend, "cuda"),
    runner = function() {
      list(
        fit = run_fastpls_variant(method_name, backend, svd_method, classifier),
        pred = NULL
      )
    },
    decoder = decode_fastpls
  )
}

make_fastpls_specs <- function() {
  methods <- c("plssvd", "simpls", "opls", "kernelpls")
  specs <- list()
  k <- 1L
  for (method_name in methods) {
    for (svd_method in c("irlba", "cpu_rsvd")) {
      for (classifier in c("argmax", "lda")) {
        specs[[k]] <- make_fastpls_spec(method_name, "cpp", svd_method, classifier)
        k <- k + 1L
      }
    }
    for (classifier in c("argmax", "lda")) {
      specs[[k]] <- make_fastpls_spec(method_name, "cuda", "cuda_rsvd", classifier)
      k <- k + 1L
    }
  }
  specs
}

run_pls_fit <- function(fit_fun) {
  fit <- fit_fun(Xtrain, Ytrain_dummy, ncomp = ncomp_requested)
  list(fit = fit, pred = predict_from_pls_fit(fit, Xtest, ncomp_requested))
}

extract_prediction_generic <- function(obj, Xnew = Xtest) {
  pred_obj <- tryCatch(stats::predict(obj, Xnew), error = function(e) NULL)
  if (is.null(pred_obj)) pred_obj <- obj
  candidates <- list(
    pred_obj$predclass, pred_obj$Ypred, pred_obj$ypred, pred_obj$y.pred,
    pred_obj$p.pred, pred_obj$c.pred, pred_obj$pred, pred_obj$prediction,
    pred_obj$y.hat, pred_obj$class, pred_obj$classes
  )
  for (cand in candidates) {
    if (is.null(cand)) next
    if (is.list(cand) && !is.data.frame(cand)) {
      for (part in cand) {
        if (is.matrix(part) || is.array(part) || is.data.frame(part) ||
            is.factor(part) || is.character(part)) {
          cand <- part
          break
        }
      }
    }
    if (is.factor(cand) || is.character(cand)) return(factor(cand, levels = levels(Ytest)))
    if (is.data.frame(cand)) {
      if (ncol(cand) == 1L) return(factor(cand[[1L]], levels = levels(Ytest)))
      return(decode_scores(as.matrix(cand), class_levels))
    }
    if (is.matrix(cand) || is.array(cand)) {
      mat <- last_component_matrix(cand)
      return(decode_scores(mat, class_levels))
    }
  }
  if (is.factor(pred_obj) || is.character(pred_obj)) return(factor(pred_obj, levels = levels(Ytest)))
  if (is.matrix(pred_obj) || is.array(pred_obj)) {
    mat <- pred_obj
    if (length(dim(mat)) == 3L) mat <- mat[, , dim(mat)[3L], drop = TRUE]
    return(decode_scores(mat, class_levels))
  }
  stop("Could not decode predictions from object of class: ", paste(class(pred_obj), collapse = ","))
}

runner_mdatools <- function() {
  ns <- asNamespace("mdatools")
  if (exists("plsda", envir = ns, inherits = FALSE)) {
    f <- get("plsda", envir = ns)
    fit <- tryCatch(
      f(Xtrain, Ytrain, ncomp = ncomp_requested, center = TRUE, scale = FALSE, cv = NULL),
      error = function(e) f(Xtrain, Ytrain, ncomp = ncomp_requested, center = TRUE, scale = FALSE)
    )
    pred_obj <- stats::predict(fit, Xtest)
    scores <- if (!is.null(pred_obj$p.pred)) pred_obj$p.pred else pred_obj$c.pred
    return(list(fit = fit, pred = decode_scores(last_component_matrix(scores), class_levels)))
  }
  f <- get("pls", envir = ns)
  fit <- tryCatch(
    f(Xtrain, Ytrain_dummy, ncomp = ncomp_requested, center = TRUE, scale = FALSE, method = "simpls", cv = NULL),
    error = function(e) f(Xtrain, Ytrain_dummy, ncomp = ncomp_requested, center = TRUE, scale = FALSE)
  )
  list(fit = fit, pred = extract_prediction_generic(fit))
}

runner_plsgenomics_regression <- function() {
  ns <- asNamespace("plsgenomics")
  f <- get("pls.regression", envir = ns)
  fit <- f(Xtrain, Ytrain_dummy, Xtest = Xtest, ncomp = ncomp_requested)
  list(fit = fit, pred = decode_scores(last_component_matrix(fit$Ypred), class_levels))
}

runner_plsgenomics_lda <- function() {
  ns <- asNamespace("plsgenomics")
  f <- get("pls.lda", envir = ns)
  fit <- f(Xtrain, Ytrain, Xtest = Xtest, ncomp = ncomp_requested, nruncv = 0)
  list(fit = fit, pred = factor(fit$predclass, levels = levels(Ytest)))
}

runner_chemometrics_pls2_nipals <- function() {
  Xc <- scale(Xtrain, center = TRUE, scale = FALSE)
  Xm <- attr(Xc, "scaled:center")
  Yc <- scale(Ytrain_dummy, center = TRUE, scale = FALSE)
  Ym <- attr(Yc, "scaled:center")
  fit <- chemometrics::pls2_nipals(Xc, Yc, a = ncomp_requested, scale = FALSE)
  pred <- sweep(as.matrix(Xtest), 2L, Xm, "-") %*% fit$B
  pred <- sweep(pred, 2L, Ym, "+")
  list(fit = fit, pred = decode_scores(pred, class_levels))
}

runner_chemometrics_pls_eigen <- function() {
  Xc <- scale(Xtrain, center = TRUE, scale = FALSE)
  Xm <- attr(Xc, "scaled:center")
  Yc <- scale(Ytrain_dummy, center = TRUE, scale = FALSE)
  Ym <- attr(Yc, "scaled:center")
  fit <- chemometrics::pls_eigen(Xc, Yc, a = ncomp_requested)
  coef_t <- solve(crossprod(fit$T), crossprod(fit$T, Yc))
  pred <- (sweep(as.matrix(Xtest), 2L, Xm, "-") %*% fit$P) %*% coef_t
  pred <- sweep(pred, 2L, Ym, "+")
  list(fit = fit, pred = decode_scores(pred, class_levels))
}

runner_mixomics_plsda <- function(sparse = FALSE) {
  if (isTRUE(sparse)) {
    fit <- mixOmics::splsda(
      Xtrain, Ytrain, ncomp = ncomp_requested,
      keepX = rep(ncol(Xtrain), ncomp_requested),
      scale = FALSE
    )
  } else {
    fit <- mixOmics::plsda(Xtrain, Ytrain, ncomp = ncomp_requested, scale = FALSE)
  }
  pred_obj <- stats::predict(fit, Xtest)
  pred <- pred_obj$class$max.dist[, min(ncomp_requested, ncol(pred_obj$class$max.dist))]
  list(fit = fit, pred = factor(pred, levels = levels(Ytest)))
}

runner_mixomics_pls <- function() {
  fit <- mixOmics::pls(
    Xtrain, Ytrain_dummy, ncomp = ncomp_requested,
    scale = FALSE, mode = "regression"
  )
  pred_obj <- stats::predict(fit, Xtest)
  list(fit = fit, pred = decode_scores(last_component_matrix(pred_obj$predict), class_levels))
}

runner_spls <- function(sparse_da = FALSE) {
  if (isTRUE(sparse_da)) {
    fit <- spls::splsda(
      Xtrain, Ytrain, K = ncomp_requested, eta = 0.9,
      classifier = "lda", scale.x = FALSE
    )
    pred <- stats::predict(fit, Xtest)
    return(list(fit = fit, pred = factor(pred, levels = levels(Ytest))))
  }
  fit <- spls::spls(
    Xtrain, Ytrain_dummy, K = ncomp_requested, eta = 0.9,
    scale.x = FALSE, scale.y = FALSE, fit = "simpls"
  )
  pred <- stats::predict(fit, Xtest)
  list(fit = fit, pred = decode_scores(pred, class_levels))
}

runner_simple_named <- function(pkg, fun_name) {
  ns <- asNamespace(pkg)
  f <- get(fun_name, envir = ns)
  attempts <- list(
    function() f(Xtrain, Ytrain_dummy, ncomp = ncomp_requested, center = TRUE, scale = FALSE),
    function() f(Xtrain, Ytrain_dummy, comps = ncomp_requested, center = TRUE, scale = FALSE),
    function() f(x = Xtrain, y = Ytrain_dummy, ncomp = ncomp_requested, center = TRUE, scale = FALSE),
    function() f(X = Xtrain, Y = Ytrain_dummy, ncomp = ncomp_requested, center = TRUE, scale = FALSE),
    function() f(Xtrain, Ytrain_dummy, ncomp = ncomp_requested),
    function() f(Xtrain, Ytrain_dummy, comps = ncomp_requested),
    function() f(Xtrain, Ytrain_dummy, ncomp_requested),
    function() f(x = Xtrain, y = Ytrain_dummy, ncomp = ncomp_requested),
    function() f(X = Xtrain, Y = Ytrain_dummy, ncomp = ncomp_requested)
  )
  last <- NULL
  for (attempt in attempts) {
    fit <- tryCatch(attempt(), error = function(e) {
      last <<- conditionMessage(e)
      NULL
    })
    if (!is.null(fit)) return(list(fit = fit, pred = extract_prediction_generic(fit)))
  }
  stop(last %||% "No compatible call signature found.")
}

runner_pcv_simpls <- function() {
  Xc <- scale(Xtrain, center = TRUE, scale = TRUE)
  Xm <- attr(Xc, "scaled:center")
  Xs <- attr(Xc, "scaled:scale")
  Xs[!is.finite(Xs) | Xs == 0] <- 1

  Yc <- scale(Ytrain_dummy, center = TRUE, scale = TRUE)
  Ym <- attr(Yc, "scaled:center")
  Ys <- attr(Yc, "scaled:scale")
  Ys[!is.finite(Ys) | Ys == 0] <- 1

  fit <- get("simpls", envir = asNamespace("pcv"))(Xc, Yc, ncomp_requested)
  ncomp_eff <- min(ncomp_requested, ncol(fit$R), ncol(fit$C))
  Xnew <- sweep(sweep(as.matrix(Xtest), 2L, Xm, "-"), 2L, Xs, "/")
  pred <- Xnew %*% fit$R[, seq_len(ncomp_eff), drop = FALSE] %*%
    t(fit$C[, seq_len(ncomp_eff), drop = FALSE])
  pred <- sweep(sweep(pred, 2L, Ys, "*"), 2L, Ym, "+")
  list(fit = fit, pred = decode_scores(pred, class_levels))
}

runner_plsdepot_simpls <- function() {
  fit <- plsdepot::simpls(Xtrain, Ytrain_dummy, comps = ncomp_requested)
  ncomp_eff <- min(ncomp_requested, ncol(fit$x.wgs), ncol(fit$y.wgs))
  Xm <- colMeans(Xtrain)
  Xs <- apply(Xtrain, 2L, stats::sd)
  Xs[!is.finite(Xs) | Xs == 0] <- 1
  Ym <- colMeans(Ytrain_dummy)
  Ys <- apply(Ytrain_dummy, 2L, stats::sd)
  Ys[!is.finite(Ys) | Ys == 0] <- 1

  Xnew <- sweep(sweep(as.matrix(Xtest), 2L, Xm, "-"), 2L, Xs, "/")
  pred <- Xnew %*% fit$x.wgs[, seq_len(ncomp_eff), drop = FALSE] %*%
    t(fit$y.wgs[, seq_len(ncomp_eff), drop = FALSE])
  pred <- sweep(sweep(pred, 2L, Ys, "*"), 2L, Ym, "+")
  list(fit = fit, pred = decode_scores(pred, class_levels))
}

runner_ropls <- function(orthoI = 0L) {
  if (orthoI > 0L && nlevels(Ytrain) > 2L) {
    stop("ropls OPLS-DA is only available for binary classification; MetRef has ",
         nlevels(Ytrain), " classes.")
  }
  f <- get("opls", envir = asNamespace("ropls"))
  fit <- f(
    x = Xtrain,
    y = Ytrain,
    predI = ncomp_requested,
    orthoI = orthoI,
    crossvalI = if (orthoI == 0L) 2L else 0L,
    permI = 0,
    scaleC = "center",
    fig.pdfC = "none",
    info.txtC = "none"
  )
  pred_fun <- methods::selectMethod("predict", "opls")
  pred <- pred_fun(fit, Xtest)
  if (is.factor(pred) || is.character(pred)) {
    return(list(fit = fit, pred = factor(pred, levels = levels(Ytest))))
  }
  list(fit = fit, pred = decode_scores(pred, class_levels))
}

method_specs <- c(make_fastpls_specs(), list(
  list(id = "pls_simpls_fit", package = "pls", algorithm = "SIMPLS",
       function_name = "pls::simpls.fit", independent = TRUE,
       runner = function() run_pls_fit(pls::simpls.fit)),
  list(id = "pls_oscorespls_fit", package = "pls", algorithm = "NIPALS/oscores PLS",
       function_name = "pls::oscorespls.fit", independent = TRUE,
       runner = function() run_pls_fit(pls::oscorespls.fit)),
  list(id = "pls_kernelpls_fit", package = "pls", algorithm = "kernel PLS",
       function_name = "pls::kernelpls.fit", independent = TRUE,
       runner = function() run_pls_fit(pls::kernelpls.fit)),
  list(id = "mdatools_plsda_or_pls", package = "mdatools", algorithm = "SIMPLS/PLS-DA",
       function_name = "mdatools::plsda or mdatools::pls", independent = TRUE,
       runner = runner_mdatools),
  list(id = "plsdepot_simpls", package = "plsdepot", algorithm = "SIMPLS",
       function_name = "plsdepot::simpls", independent = TRUE,
       runner = runner_plsdepot_simpls),
  list(id = "pcv_simpls", package = "pcv", algorithm = "SIMPLS",
       function_name = "pcv::simpls", independent = TRUE,
       runner = runner_pcv_simpls),
  list(id = "plsgenomics_pls_regression", package = "plsgenomics", algorithm = "PLS regression",
       function_name = "plsgenomics::pls.regression", independent = TRUE,
       runner = runner_plsgenomics_regression),
  list(id = "plsgenomics_pls_lda", package = "plsgenomics", algorithm = "PLS-LDA",
       function_name = "plsgenomics::pls.lda", independent = TRUE,
       runner = runner_plsgenomics_lda),
  list(id = "mixOmics_plsda", package = "mixOmics", algorithm = "PLS-DA",
       function_name = "mixOmics::plsda", independent = TRUE,
       runner = function() runner_mixomics_plsda(FALSE)),
  list(id = "mixOmics_pls", package = "mixOmics", algorithm = "PLS regression",
       function_name = "mixOmics::pls", independent = TRUE,
       runner = runner_mixomics_pls),
  list(id = "mixOmics_splsda", package = "mixOmics", algorithm = "sPLS-DA",
       function_name = "mixOmics::splsda", independent = TRUE,
       runner = function() runner_mixomics_plsda(TRUE)),
  list(id = "chemometrics_pls_eigen", package = "chemometrics", algorithm = "PLS eigen",
       function_name = "chemometrics::pls_eigen", independent = TRUE,
       runner = runner_chemometrics_pls_eigen),
  list(id = "chemometrics_pls2_nipals", package = "chemometrics", algorithm = "PLS2 NIPALS",
       function_name = "chemometrics::pls2_nipals", independent = TRUE,
       runner = runner_chemometrics_pls2_nipals),
  list(id = "spls_splsda", package = "spls", algorithm = "sPLS-DA",
       function_name = "spls::splsda", independent = TRUE,
       runner = function() runner_spls(TRUE)),
  list(id = "spls_spls", package = "spls", algorithm = "sPLS regression",
       function_name = "spls::spls", independent = TRUE,
       runner = function() runner_spls(FALSE)),
  list(id = "ropls_plsda", package = "ropls", algorithm = "PLS-DA",
       function_name = "ropls::opls(orthoI=0)", independent = TRUE,
       runner = function() runner_ropls(0L)),
  list(id = "ropls_oplsda", package = "ropls", algorithm = "OPLS-DA",
       function_name = "ropls::opls(orthoI=1)", independent = TRUE,
       requires_binary_y = TRUE,
       runner = function() runner_ropls(1L))
))

package_version_chr <- function(pkg) {
  if (!quiet_require(pkg)) return(NA_character_)
  as.character(utils::packageVersion(pkg))
}

function_available <- function(spec) {
  if (!quiet_require(spec$package)) return(FALSE)
  if (identical(spec$id, "mdatools_plsda_or_pls")) {
    ns <- asNamespace("mdatools")
    return(exists("plsda", envir = ns, inherits = FALSE) ||
             exists("pls", envir = ns, inherits = FALSE))
  }
  parts <- strsplit(spec$function_name, "::", fixed = TRUE)[[1L]]
  if (length(parts) < 2L) return(TRUE)
  fun <- sub("\\(.*$", "", parts[[2L]])
  exists(fun, envir = asNamespace(spec$package), inherits = FALSE)
}

measure_once <- function(fun) {
  gc(FALSE)
  err <- NULL
  warn <- character()
  value <- NULL
  elapsed_ms <- NA_real_
  peak_mb <- NA_real_
  eval_fun <- function() {
    withCallingHandlers(
      fun(),
      warning = function(w) {
        warn <<- c(warn, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    )
  }

  if (quiet_require("peakRAM")) {
    pr <- tryCatch(
      peakRAM::peakRAM({
        value <- eval_fun()
      }),
      error = function(e) {
        err <<- conditionMessage(e)
        NULL
      }
    )
    if (!is.null(pr)) {
      elapsed_ms <- as.numeric(pr$Elapsed_Time_sec[1L]) * 1000
      peak_mb <- as.numeric(pr$Peak_RAM_Used_MiB[1L])
      if (is.finite(peak_mb) && peak_mb > 8192) peak_mb <- peak_mb / 1024
    }
  } else {
    t0 <- proc.time()[3L]
    value <- tryCatch(eval_fun(), error = function(e) {
      err <<- conditionMessage(e)
      NULL
    })
    elapsed_ms <- as.numeric(proc.time()[3L] - t0) * 1000
  }

  list(
    value = value,
    elapsed_ms = elapsed_ms,
    peak_mb = peak_mb,
    error = err,
    warning = paste(unique(warn), collapse = " | ")
  )
}

rows <- list()
k <- 1L
message("MetRef source: ", task$dataset_path)
message("Rows train/test: ", nrow(Xtrain), "/", nrow(Xtest),
        "; p=", ncol(Xtrain), "; classes=", nlevels(Ytrain),
        "; ncomp=", ncomp_requested)

for (spec in method_specs) {
  pkg_ok <- quiet_require(spec$package)
  fun_ok <- if (pkg_ok) function_available(spec) else FALSE
  compatible <- !(isTRUE(spec$requires_binary_y) && nlevels(Ytrain) != 2L)
  cuda_ok <- !(isTRUE(spec$requires_cuda) &&
                 (!quiet_require("fastPLS") ||
                    !isTRUE(tryCatch(fastPLS::has_cuda(), error = function(e) FALSE))))
  runnable <- pkg_ok && fun_ok && compatible && cuda_ok
  n_rep <- if (runnable) reps else 1L

  for (rep_id in seq_len(n_rep)) {
    status <- "ok"
    error_message <- ""
    total_runtime_ms <- NA_real_
    peak_memory_mb <- NA_real_
    acc <- NA_real_
    ncomp_effective <- ncomp_requested
    warning_message <- ""

    if (!pkg_ok) {
      status <- "skipped_package_not_installed"
      error_message <- sprintf("Package '%s' is not installed.", spec$package)
    } else if (!fun_ok) {
      status <- "skipped_function_or_method_not_available"
      error_message <- sprintf("Function/method '%s' is not available.", spec$function_name)
    } else if (!compatible) {
      status <- "skipped_incompatible_response"
      error_message <- sprintf(
        "%s requires a binary response; this MetRef split has %d classes.",
        spec$function_name,
        nlevels(Ytrain)
      )
    } else if (!cuda_ok) {
      status <- "skipped_cuda_unavailable"
      error_message <- "This fastPLS build does not report CUDA availability via fastPLS::has_cuda()."
    } else {
      message(sprintf("[%s] rep %d/%d", spec$id, rep_id, reps))
      measured <- measure_once(spec$runner)
      total_runtime_ms <- measured$elapsed_ms
      peak_memory_mb <- measured$peak_mb
      warning_message <- measured$warning
      if (!is.null(measured$error)) {
        status <- "error"
        error_message <- measured$error
      } else {
        pred <- tryCatch({
          if (!is.null(measured$value$pred)) measured$value$pred else spec$decoder(measured$value$fit)
        }, error = function(e) {
          status <<- "prediction_error"
          error_message <<- conditionMessage(e)
          NULL
        })
        if (!is.null(pred)) acc <- accuracy(pred)
      }
    }

    rows[[k]] <- data.frame(
      dataset = "MetRef",
      dataset_path = task$dataset_path,
      split_seed = split_seed,
      n_train = nrow(Xtrain),
      n_test = nrow(Xtest),
      p = ncol(Xtrain),
      n_classes = nlevels(Ytrain),
      ncomp_requested = ncomp_requested,
      ncomp_effective = ncomp_effective,
      replicate = if (runnable) rep_id else NA_integer_,
      method_id = spec$id,
      package = spec$package,
      package_version = package_version_chr(spec$package),
      function_name = spec$function_name,
      algorithm = spec$algorithm,
      fastpls_method = spec$fastpls_method %||% NA_character_,
      fastpls_backend = spec$fastpls_backend %||% NA_character_,
      fastpls_svd_method = spec$fastpls_svd_method %||% NA_character_,
      fastpls_classifier = spec$fastpls_classifier %||% NA_character_,
      independent_implementation = isTRUE(spec$independent),
      delegates_to_pls_package = FALSE,
      total_runtime_ms = total_runtime_ms,
      peak_memory_mb = peak_memory_mb,
      accuracy = acc,
      status = status,
      warning_message = warning_message,
      error_message = error_message,
      stringsAsFactors = FALSE
    )
    k <- k + 1L
  }
}

result <- do.call(rbind, rows)
result <- result[order(result$status != "ok", result$total_runtime_ms, result$method_id), ]

dir.create(dirname(normalizePath(outfile, mustWork = FALSE)), recursive = TRUE, showWarnings = FALSE)
utils::write.csv(result, outfile, row.names = FALSE, quote = TRUE)

message("Wrote: ", normalizePath(outfile, winslash = "/", mustWork = FALSE))
print(result[, c("method_id", "package", "algorithm", "replicate", "total_runtime_ms",
                 "peak_memory_mb", "accuracy", "status", "warning_message", "error_message")],
      row.names = FALSE)
