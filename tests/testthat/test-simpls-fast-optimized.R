with_fastpls_fast_optimized <- function(value, code) {
  old <- Sys.getenv("FASTPLS_FAST_OPTIMIZED", unset = NA_character_)
  on.exit({
    if (length(old) != 1L || is.na(old)) {
      Sys.unsetenv("FASTPLS_FAST_OPTIMIZED")
    } else {
      Sys.setenv(FASTPLS_FAST_OPTIMIZED = old)
    }
  }, add = TRUE)
  Sys.setenv(FASTPLS_FAST_OPTIMIZED = as.character(as.integer(value)))
  force(code)
}

test_that("simpls_fast returns valid fit structure", {
  set.seed(20260321)
  X <- matrix(rnorm(96 * 18), nrow = 96, ncol = 18)
  y <- factor(sample(c("A", "B", "C"), 96, replace = TRUE))
  idx <- sample(seq_len(96), 20)

  fit <- with_fastpls_fast_optimized(1, fastPLS::pls(
    X[-idx, , drop = FALSE],
    y[-idx],
    X[idx, , drop = FALSE],
    y[idx],
    ncomp = 1:4,
    method = "simpls_fast",
    svd.method = "cpu_rsvd",
    fit = TRUE,
    seed = 77L
  ))

  expect_s3_class(fit, "fastPLS")
  expect_equal(dim(fit$B), c(ncol(X), nlevels(y), 4L))
  expect_equal(dim(fit$R), c(ncol(X), 4L))
  expect_equal(dim(fit$Q), c(nlevels(y), 4L))
  expect_true(is.data.frame(fit$Ypred))
  expect_equal(ncol(fit$Ypred), 4L)
})

test_that("simpls_fast stays close to legacy baseline on a controlled regression task", {
  set.seed(20260321)
  n <- 120
  z1 <- rnorm(n)
  z2 <- rnorm(n)
  X <- cbind(
    outer(z1, seq_len(12), `*`) + matrix(rnorm(n * 12, sd = 0.08), n, 12),
    outer(z2, seq_len(12), `*`) + matrix(rnorm(n * 12, sd = 0.08), n, 12)
  )
  Y <- cbind(
    1.5 * z1 - 0.4 * z2 + rnorm(n, sd = 0.05),
    -0.7 * z1 + 1.8 * z2 + rnorm(n, sd = 0.05)
  )
  idx <- sample(seq_len(n), 30)

  fit_args <- list(
    Xtrain = X[-idx, , drop = FALSE],
    Ytrain = Y[-idx, , drop = FALSE],
    Xtest = X[idx, , drop = FALSE],
    Ytest = Y[idx, , drop = FALSE],
    ncomp = c(2L, 4L, 6L),
    method = "simpls_fast",
    svd.method = "cpu_rsvd",
    fit = FALSE,
    seed = 123L
  )

  baseline <- with_fastpls_fast_optimized(0, do.call(fastPLS::pls, fit_args))
  optimized <- with_fastpls_fast_optimized(1, do.call(fastPLS::pls, fit_args))

  base_pred <- baseline$Ypred[, , 3]
  opt_pred <- optimized$Ypred[, , 3]
  truth <- Y[idx, , drop = FALSE]
  base_rmse <- sqrt(mean((base_pred - truth)^2))
  opt_rmse <- sqrt(mean((opt_pred - truth)^2))

  expect_equal(dim(baseline$B), dim(optimized$B))
  expect_true(abs(opt_rmse - base_rmse) <= 0.05)
})
