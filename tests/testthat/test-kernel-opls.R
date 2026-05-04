test_that("kernel PLS C++ wrapper predicts classification labels", {
  set.seed(2201)
  X <- matrix(rnorm(72 * 10), nrow = 72, ncol = 10)
  y <- factor(sample(c("A", "B", "C"), 72, replace = TRUE))
  idx <- seq_len(12)

  fit_cpp <- pls(
    X[-idx, , drop = FALSE],
    y[-idx],
    X[idx, , drop = FALSE],
    y[idx],
    ncomp = 1:2,
    method = "kernelpls",
    backend = "cpp",
    kernel = "rbf",
    inner.method = "simpls",
    svd.method = "cpu_rsvd"
  )

  expect_s3_class(fit_cpp, "fastPLSKernel")
  expect_true(is.data.frame(fit_cpp$Ypred))
  expect_equal(nrow(fit_cpp$Ypred), length(idx))
})

test_that("kernelpls high-level wrapper dispatches to simpls", {
  set.seed(2203)
  X <- matrix(rnorm(60 * 8), nrow = 60, ncol = 8)
  y <- factor(sample(c("low", "high"), 60, replace = TRUE))
  idx <- seq_len(10)

  fit_fast <- pls(
    X[-idx, , drop = FALSE],
    y[-idx],
    X[idx, , drop = FALSE],
    y[idx],
    ncomp = 1:2,
    method = "kernelpls",
    backend = "cpp",
    inner.method = "simpls",
    kernel = "rbf",
    svd.method = "cpu_rsvd"
  )

  expect_s3_class(fit_fast, "fastPLSKernel")
  expect_identical(fit_fast$inner_model$pls_method, "simpls")
})

test_that("OPLS C++ wrapper predicts regression matrices", {
  set.seed(2202)
  X <- matrix(rnorm(70 * 11), nrow = 70, ncol = 11)
  Y <- cbind(rnorm(70), rnorm(70))
  idx <- seq_len(10)

  fit_cpp <- pls(
    X[-idx, , drop = FALSE],
    Y[-idx, , drop = FALSE],
    X[idx, , drop = FALSE],
    Y[idx, , drop = FALSE],
    ncomp = 1:2,
    method = "opls",
    backend = "cpp",
    north = 1L,
    inner.method = "simpls",
    svd.method = "cpu_rsvd"
  )

  expect_s3_class(fit_cpp, "fastPLSOpls")
  expect_true(is.array(fit_cpp$Ypred))
  expect_equal(dim(fit_cpp$Ypred), c(length(idx), ncol(Y), 2L))
})

test_that("opls high-level wrapper dispatches to simpls", {
  set.seed(2204)
  X <- matrix(rnorm(64 * 9), nrow = 64, ncol = 9)
  Y <- matrix(rnorm(64), ncol = 1)
  idx <- seq_len(12)

  fit <- pls(
    X[-idx, , drop = FALSE],
    Y[-idx, , drop = FALSE],
    X[idx, , drop = FALSE],
    Y[idx, , drop = FALSE],
    ncomp = 1:2,
    method = "opls",
    backend = "cpp",
    inner.method = "simpls",
    north = 1L,
    svd.method = "cpu_rsvd"
  )

  expect_s3_class(fit, "fastPLSOpls")
  expect_identical(fit$inner_model$pls_method, "simpls")
  expect_equal(dim(fit$Ypred), c(length(idx), ncol(Y), 2L))
})
