test_that("Metal PLS backend fits core method families when available", {
  skip_if_not(fastPLS::has_metal(), "Metal backend is not available")

  set.seed(99)
  n <- 70
  p <- 12
  X <- matrix(rnorm(n * p), n, p)
  y <- matrix(X[, 1] - 0.5 * X[, 2] + rnorm(n, sd = 0.2), ncol = 1)
  cls <- factor(ifelse(y[, 1] > median(y[, 1]), "a", "b"))
  test <- seq(1, n, by = 5)

  for (method in c("plssvd", "simpls", "opls", "kernelpls")) {
    ncomp_test <- if (identical(method, "plssvd")) 1L else 1:2
    fit <- fastPLS::pls(
      X[-test, , drop = FALSE],
      y[-test, , drop = FALSE],
      X[test, , drop = FALSE],
      y[test, , drop = FALSE],
      ncomp = ncomp_test,
      method = method,
      backend = "metal",
      kernel = "linear",
      north = 1,
      return_variance = FALSE,
      seed = 99
    )
    expect_true(inherits(fit, "fastPLS"))
    if (identical(method, "plssvd")) {
      expect_identical(fit$predict_backend, "metal")
      expect_identical(fit$backend, "metal")
    } else {
      expect_true(
        identical(fit$backend, "metal") ||
          identical(fit$inner_model$backend, "metal")
      )
    }
    expect_true(all(is.finite(fit$Q2Y)))
  }

  fit_cls <- fastPLS::pls(
    X[-test, , drop = FALSE],
    cls[-test],
    X[test, , drop = FALSE],
    cls[test],
    ncomp = 1:2,
    method = "simpls",
    backend = "metal",
    return_variance = FALSE,
    seed = 100
  )
  expect_true(inherits(fit_cls, "fastPLS"))
  expect_equal(fit_cls$backend, "metal")
  expect_true(is.data.frame(fit_cls$Ypred))
})

test_that("Metal backend is available through public CV helpers", {
  skip_if_not(fastPLS::has_metal(), "Metal backend is not available")

  set.seed(101)
  n <- 36
  p <- 8
  X <- matrix(rnorm(n * p), n, p)
  y_signal <- 0.6 * X[, 1] - 0.4 * X[, 2] + rnorm(n, sd = 0.2)
  y_num <- cbind(y_signal, 0.3 * X[, 3] + 0.2 * X[, 4] + rnorm(n, sd = 0.2))
  y_cls <- factor(ifelse(y_signal > median(y_signal), "hi", "lo"))

  fixed <- fastPLS::pls.single.cv(
    Xdata = X,
    Ydata = y_cls,
    ncomp = 1:2,
    kfold = 2,
    method = "simpls",
    backend = "metal",
    seed = 101
  )
  expect_identical(fixed$backend, "metal")
  expect_identical(fixed$prediction_backend, "metal")
  expect_equal(nrow(fixed$metrics), 2L)
  expect_true(all(is.finite(fixed$metrics$metric_value)))

  opt <- fastPLS::optim.pls.cv(
    Xdata = X,
    Ydata = y_num,
    ncomp = 1:2,
    kfold = 2,
    method = "plssvd",
    backend = "metal",
    seed = 102
  )
  expect_identical(opt$backend, "metal")
  expect_identical(opt$prediction_backend, "metal")
  expect_true(opt$optim_comp %in% 1:2)

  nested <- fastPLS::pls.double.cv(
    Xdata = X,
    Ydata = y_cls,
    ncomp = 1:2,
    runn = 1,
    kfold_inner = 2,
    kfold_outer = 2,
    method = "simpls",
    backend = "metal",
    seed = 103
  )
  expect_identical(nested$backend, "metal")
  expect_true(is.finite(nested$medianQ2Y))
})
