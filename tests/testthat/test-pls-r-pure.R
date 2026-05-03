test_that("pls backend='r' runs both plssvd and simpls with bundled RSVD", {
  set.seed(2026)
  X <- matrix(rnorm(120 * 15), nrow = 120, ncol = 15)
  Y <- matrix(rnorm(120 * 4), nrow = 120, ncol = 4)

  m1 <- pls(X, Y, ncomp = 1:3, fit = TRUE, method = "plssvd", backend = "r", svd.method = "cpu_rsvd")
  m2 <- pls(X, Y, ncomp = 1:3, fit = TRUE, method = "simpls", backend = "r", svd.method = "cpu_rsvd")

  expect_s3_class(m1, "fastPLS")
  expect_s3_class(m2, "fastPLS")
  expect_equal(dim(m1$B), c(ncol(X), ncol(Y), 3))
  expect_equal(dim(m2$B), c(ncol(X), ncol(Y), 3))
})

test_that("pls backend='r' keeps classification output style", {
  set.seed(123)
  X <- matrix(rnorm(90 * 12), nrow = 90, ncol = 12)
  y <- factor(sample(letters[1:3], 90, replace = TRUE))
  idx <- sample(seq_len(90), 20)

  fit <- pls(X[-idx, ], y[-idx], X[idx, ], y[idx], ncomp = 1:2, backend = "r", fit = TRUE)

  expect_true(is.data.frame(fit$Ypred))
  expect_equal(nrow(fit$Ypred), length(idx))
  expect_equal(ncol(fit$Ypred), 2)
})

test_that("pls backend='r' cpu_rsvd is deterministic with fixed seed", {
  set.seed(11)
  X <- matrix(rnorm(100 * 20), nrow = 100, ncol = 20)
  Y <- matrix(rnorm(100 * 5), nrow = 100, ncol = 5)

  a <- pls(X, Y, ncomp = 1:4, method = "plssvd", backend = "r", svd.method = "cpu_rsvd", seed = 7L)
  b <- pls(X, Y, ncomp = 1:4, method = "plssvd", backend = "r", svd.method = "cpu_rsvd", seed = 7L)

  expect_equal(a$B, b$B)
})

test_that("pls backend='r' plssvd handles ncomp above rank by capping internally", {
  set.seed(77)
  X <- matrix(rnorm(160 * 40), nrow = 160, ncol = 40)
  y <- factor(sample(letters[1:8], 160, replace = TRUE))
  idx <- sample(seq_len(160), 30)

  expect_warning({
    fit <- pls(X[-idx, ], y[-idx], X[idx, ], ncomp = 50, method = "plssvd", backend = "r", svd.method = "cpu_rsvd")
    expect_s3_class(fit, "fastPLS")
    expect_true(is.data.frame(fit$Ypred))
  }, "rank is limited")
})
