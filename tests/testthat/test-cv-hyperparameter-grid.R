library(fastPLS)

test_that("pls.single.cv tunes prediction hyperparameters", {
  idx <- c(1:12, 51:62, 101:112)
  X <- as.matrix(iris[idx, 1:4])
  y <- factor(iris[idx, 5])

  opt <- pls.single.cv(
    X,
    y,
    ncomp = 1:2,
    kfold = 3,
    method = "kernelpls",
    backend = "cpu",
    svd.method = "rsvd",
    kernel = c("linear", "rbf"),
    gamma = c(0.1, 1),
    seed = 1
  )

  expect_true(opt$best_ncomp %in% 1:2)
  expect_true(all(c("kernel", "gamma", "ncomp") %in% names(opt$best_parameters)))
  expect_false("method" %in% names(opt$best_parameters))
  expect_true(nrow(opt$tuning_summary) >= 2L)
  expect_true(all(c("kernel", "gamma", "best_metric_value") %in% names(opt$tuning_summary)))
})

test_that("pls.double.cv uses inner selected hyperparameters", {
  idx <- c(1:10, 51:60, 101:110)
  X <- as.matrix(iris[idx, 1:4])
  y <- factor(iris[idx, 5])

  nested <- pls.double.cv(
    X,
    y,
    ncomp = 1:2,
    runn = 1,
    kfold_inner = 2,
    kfold_outer = 2,
    method = "kernelpls",
    backend = "cpu",
    svd.method = "rsvd",
    kernel = c("linear", "rbf"),
    gamma = c(0.1, 1),
    seed = 1
  )

  expect_length(nested$results, 1L)
  expect_true(all(nested$results[[1]]$best_ncomp %in% 1:2))
  expect_true(all(vapply(nested$results[[1]]$best_parameters, function(x) {
    is.list(x) && all(c("kernel", "gamma", "ncomp") %in% names(x)) &&
      !("method" %in% names(x))
  }, logical(1L))))
})

test_that("pls.single.cv reports only optimized best parameters", {
  idx <- c(1:12, 51:62, 101:112)
  X <- as.matrix(iris[idx, 1:4])
  y <- factor(iris[idx, 5])

  opt <- pls.single.cv(
    X,
    y,
    ncomp = 2:4,
    kfold = 3,
    classifier = c("argmax", "lda"),
    seed = 1
  )

  expect_setequal(names(opt$best_parameters), c("ncomp", "classifier"))
  expect_true(opt$best_parameters$ncomp %in% 2:4)
  expect_true(opt$best_parameters$classifier %in% c("argmax", "lda"))
})

test_that("pls.single.cv tuning_config omits irrelevant classifier controls", {
  idx <- c(1:12, 51:62, 101:112)
  X <- as.matrix(iris[idx, 1:4])
  y <- factor(iris[idx, 5])

  argmax <- pls.single.cv(
    X,
    y,
    ncomp = 2,
    kfold = 3,
    classifier = "argmax",
    seed = 2
  )
  expect_true("classifier" %in% names(argmax$tuning_config))
  expect_false(any(c("lda_ridge", "k", "tau", "alpha", "top_m", "cknn_memory") %in% names(argmax$tuning_config)))

  lda <- pls.single.cv(
    X,
    y,
    ncomp = 2,
    kfold = 3,
    classifier = "lda",
    seed = 2
  )
  expect_false(any(c("lda_ridge", "k", "tau", "alpha", "top_m", "cknn_memory") %in% names(lda$tuning_config)))

  cknn <- pls.single.cv(
    X,
    y,
    ncomp = 2,
    kfold = 3,
    classifier = "cknn",
    seed = 2
  )
  expect_true(all(c("k", "tau", "alpha", "top_m", "cknn_memory") %in% names(cknn$tuning_config)))
  expect_false("lda_ridge" %in% names(cknn$tuning_config))
})

test_that("pls refits and predicts from a pls.single.cv result", {
  set.seed(2106)
  test_idx <- sample(seq_len(nrow(iris)), 30)
  Xtrain <- as.matrix(iris[-test_idx, 1:4])
  Ytrain <- factor(iris[-test_idx, 5])
  Xtest <- as.matrix(iris[test_idx, 1:4])
  Ytest <- factor(iris[test_idx, 5], levels = levels(Ytrain))

  opt <- pls.single.cv(
    Xtrain,
    Ytrain,
    ncomp = 1:3,
    kfold = 3,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    classifier = c("argmax", "lda"),
    seed = 2106
  )

  expect_s3_class(opt, "fastPLSCV")
  expect_true(all(c("Xdata", "Ydata") %in% names(attr(opt, "fit_data"))))

  fit_named <- pls(opt, Xtest = Xtest, Ytest = Ytest, return_variance = FALSE)
  fit_positional <- pls(opt, Xtest, Ytest = Ytest, return_variance = FALSE)

  expect_s3_class(fit_named, "fastPLS")
  expect_equal(as.integer(attr(fit_named, "fastPLS_internal")$ncomp), as.integer(opt$best_ncomp))
  expect_equal(fit_named$cv_best_parameters, opt$best_parameters)
  expect_true(is.data.frame(fit_named$Ypred))
  expect_equal(nrow(fit_named$Ypred), nrow(Xtest))
  expect_equal(fit_named$Ypred, fit_positional$Ypred)
})

test_that("pls refits regression models selected by pls.single.cv", {
  set.seed(2107)
  X <- matrix(rnorm(60 * 6), nrow = 60, ncol = 6)
  y <- X[, 1] - 0.5 * X[, 2] + rnorm(60, sd = 0.2)
  test_idx <- seq(1, 60, by = 3)

  opt <- pls.single.cv(
    X[-test_idx, , drop = FALSE],
    y[-test_idx],
    ncomp = 1:3,
    kfold = 3,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    seed = 2107,
    fit = FALSE
  )
  fit <- pls(
    opt,
    Xtest = X[test_idx, , drop = FALSE],
    Ytest = y[test_idx],
    return_variance = FALSE
  )

  expect_s3_class(fit, "fastPLS")
  expect_equal(as.integer(attr(fit, "fastPLS_internal")$ncomp), as.integer(opt$best_ncomp))
  expect_true(is.array(fit$Ypred))
  expect_equal(dim(fit$Ypred)[1], length(test_idx))
  expect_true(any(is.finite(fit$Ypred)))
  expect_true(any(is.finite(fit$Q2Y)))
})
