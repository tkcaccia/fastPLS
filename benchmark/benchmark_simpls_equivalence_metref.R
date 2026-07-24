#!/usr/bin/env Rscript

# Real-data numerical agreement check for the accelerated SIMPLS path.  The
# reference is pls::simpls.fit on the identical, explicitly dummy-coded
# response, preprocessing, split, and requested component path.

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default) {
  key <- paste0("--", name, "=")
  hit <- args[startsWith(args, key)]
  if (!length(hit)) return(default)
  sub(key, "", hit[[1L]], fixed = TRUE)
}

if (!requireNamespace("fastPLS", quietly = TRUE) ||
    !requireNamespace("pls", quietly = TRUE) ||
    !requireNamespace("KODAMA", quietly = TRUE)) {
  stop("fastPLS, pls, and KODAMA are required.", call. = FALSE)
}

suppressPackageStartupMessages({
  library(fastPLS)
  library(pls)
  library(KODAMA)
})

out_dir <- get_arg("out", "benchmark_results/simpls_equivalence_metref")
ncomp <- as.integer(strsplit(get_arg("ncomp", "2,5,10,18"), ",", fixed = TRUE)[[1L]])
seed <- as.integer(get_arg("seed", "123"))
svd_method <- match.arg(get_arg("svd", "irlba"), c("irlba", "rsvd"))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

principal_angle_degrees <- function(A, B) {
  qa <- qr.Q(qr(A))
  qb <- qr.Q(qr(B))
  d <- svd(crossprod(qa, qb), nu = 0L, nv = 0L)$d
  acos(pmin(1, pmax(-1, d))) * 180 / pi
}

cube_slice <- function(x, index) {
  x[, , index, drop = FALSE][, , 1L]
}

prediction_slice <- function(x, index, ncomp_value) {
  if (is.list(x) && !is.data.frame(x)) {
    return(as.matrix(x[[paste0("ncomp=", ncomp_value)]]))
  }
  if (length(dim(x)) == 3L) return(cube_slice(x, index))
  as.matrix(x)
}

data("MetRef", package = "KODAMA")
X <- MetRef$data
X <- X[, colSums(X) != 0, drop = FALSE]
X <- normalization(X)$newXtrain
X <- scaling(X)$newXtrain
y <- factor(MetRef$donor)

set.seed(seed)
test_index <- sample(seq_len(nrow(X)), min(100L, floor(nrow(X) / 5L)))
train_index <- setdiff(seq_len(nrow(X)), test_index)
levels_y <- levels(y)
Y <- matrix(0, nrow = length(y), ncol = length(levels_y),
            dimnames = list(NULL, levels_y))
Y[cbind(seq_along(y), as.integer(y))] <- 1

Xtrain <- as.matrix(X[train_index, , drop = FALSE])
Xtest <- as.matrix(X[test_index, , drop = FALSE])
Ytrain <- Y[train_index, , drop = FALSE]
Ytest <- Y[test_index, , drop = FALSE]
ytest <- y[test_index]

fast_elapsed <- system.time({
  fast_fit <- fastPLS::pls(
    Xtrain, Ytrain, Xtest, Ytest,
    ncomp = ncomp, method = "simpls", backend = "cpu", svd.method = svd_method,
    scaling = "centering", fit = TRUE, return_variance = FALSE, seed = seed
  )
})[["elapsed"]]

reference_elapsed <- system.time({
  reference_fit <- pls::simpls.fit(Xtrain, Ytrain, ncomp = max(ncomp), center = TRUE)
})[["elapsed"]]

rows <- lapply(seq_along(ncomp), function(i) {
  k <- ncomp[[i]]
  fast_score <- prediction_slice(fast_fit$Ypred, i, k)
  reference_b <- cube_slice(reference_fit$coefficients, k)
  # simpls.fit coefficients map centred X into centred Y.
  reference_score <- sweep(Xtest, 2L, reference_fit$Xmeans, "-") %*% reference_b
  reference_score <- sweep(reference_score, 2L, reference_fit$Ymeans, "+")
  fast_b <- cube_slice(fast_fit$B, i)
  fast_label <- max.col(fast_score, ties.method = "first")
  reference_label <- max.col(reference_score, ties.method = "first")
  data.frame(
    dataset = "MetRef",
    n_train = nrow(Xtrain), n_test = nrow(Xtest), p = ncol(Xtrain), q = ncol(Ytrain),
    ncomp = k,
    svd_method = svd_method,
    fastpls_elapsed_sec = fast_elapsed,
    reference_elapsed_sec = reference_elapsed,
    prediction_correlation = cor(as.vector(fast_score), as.vector(reference_score)),
    relative_prediction_error = sqrt(sum((fast_score - reference_score)^2)) /
      max(sqrt(sum(reference_score^2)), .Machine$double.eps),
    coefficient_relative_error = sqrt(sum((fast_b - reference_b)^2)) /
      max(sqrt(sum(reference_b^2)), .Machine$double.eps),
    max_principal_angle_degrees = max(principal_angle_degrees(
      fast_fit$R[, seq_len(k), drop = FALSE],
      reference_fit$projection[, seq_len(k), drop = FALSE]
    )),
    prediction_label_agreement = mean(fast_label == reference_label),
    fastpls_accuracy = mean(fast_label == as.integer(ytest)),
    reference_accuracy = mean(reference_label == as.integer(ytest)),
    stringsAsFactors = FALSE
  )
})

result <- do.call(rbind, rows)
stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
csv <- file.path(out_dir, paste0("metref_simpls_equivalence_", stamp, ".csv"))
utils::write.csv(result, csv, row.names = FALSE)
saveRDS(list(results = result, fastpls = fast_fit, reference = reference_fit,
             split = list(train = train_index, test = test_index),
             session = utils::sessionInfo()), sub("[.]csv$", ".rds", csv))
print(result)
cat("Saved: ", normalizePath(csv, winslash = "/", mustWork = FALSE), "\n", sep = "")
