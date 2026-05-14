test_that("Metal PLS backend fits core method families when available", {
  skip_if_not(fastPLS:::has_metal(), "Metal backend is not available")

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
      inner.method = "simpls",
      kernel = "linear",
      north = 1,
      return_variance = FALSE,
      seed = 99
    )
    expect_true(inherits(fit, "fastPLS"))
    if (identical(method, "plssvd")) {
      expect_true(
        identical(fit$predict_backend, "metal") ||
          identical(fit$backend, "metal_adaptive_cpu")
      )
    } else {
      expect_true(
        identical(fit$backend, "metal") ||
          identical(fit$inner_model$backend, "metal") ||
          identical(fit$backend, "metal_adaptive_cpu") ||
          identical(fit$inner_model$backend, "metal_adaptive_cpu")
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
