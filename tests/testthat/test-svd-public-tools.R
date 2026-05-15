test_that("fastsvd returns decomposition outputs from public backends", {
  set.seed(1)
  A <- matrix(rnorm(120 * 20), 120, 20)
  out <- fastsvd(A, ncomp = 5, backend = "cpu", method = "rsvd")
  expect_true(is.list(out))
  expect_true(all(c("u", "d", "v", "backend", "method", "svd.method", "elapsed") %in% names(out)))
  expect_identical(out$backend, "cpu")
  expect_identical(out$method, "rsvd")
  expect_identical(out$svd.method, "cpu_rsvd")
  expect_equal(ncol(out$u), 5)
  expect_equal(length(out$d), 5)
  expect_equal(ncol(out$v), 5)
})

test_that("unsupported fastsvd method labels use standard choices error", {
  set.seed(11)
  A <- matrix(rnorm(80 * 12), 80, 12)
  expect_error(fastsvd(A, ncomp = 4, backend = "cpu", method = "unsupported"), "should be one of")
  expect_error(fastsvd(A, ncomp = 4, backend = "cpu", method = "full"), "should be one of")
})

test_that("fastsvd maps backend and method to the intended internal SVD", {
  set.seed(12)
  A <- matrix(rnorm(70 * 10), 70, 10)

  cpu_irlba <- fastsvd(A, ncomp = 4, backend = "cpu", method = "irlba")
  expect_identical(cpu_irlba$svd.method, "irlba")

  cpu_rsvd <- fastsvd(A, ncomp = 4, backend = "cpu", method = "rsvd")
  expect_identical(cpu_rsvd$svd.method, "cpu_rsvd")

  expect_error(
    fastsvd(A, ncomp = 4, backend = "cuda", method = "irlba"),
    "only available with backend='cpu'"
  )
  expect_error(
    fastsvd(A, ncomp = 4, backend = "metal", method = "irlba"),
    "only available with backend='cpu'"
  )
})

test_that("small SVD inputs use exact fallback for iterative public backends", {
  set.seed(42)
  A <- matrix(rnorm(40 * 5), 40, 5)
  ref <- base::svd(A, nu = 3, nv = 3)

  for (method in c("irlba", "rsvd")) {
    out <- fastsvd(
      A,
      ncomp = 3,
      backend = "cpu",
      method = method,
      oversample = 0L,
      power = 0L,
      seed = 99L
    )
    expect_equal(out$d, ref$d[1:3], tolerance = 1e-8)
    expect_equal(abs(out$u), abs(ref$u[, 1:3, drop = FALSE]), tolerance = 1e-6)
    expect_equal(abs(out$v), abs(ref$v[, 1:3, drop = FALSE]), tolerance = 1e-6)
  }
})

test_that("pca uses public SVD backends and returns plottable scores", {
  set.seed(9)
  X <- matrix(rnorm(60 * 8), 60, 8)
  fit <- pca(X, ncomp = 3, backend = "cpu", method = "rsvd", seed = 12)
  expect_s3_class(fit, "fastPLSPCA")
  expect_equal(dim(fit$scores), c(60L, 3L))
  expect_equal(dim(fit$loadings), c(8L, 3L))
  expect_true(all(is.finite(fit$variance_explained)))
  expect_equal(length(fit$cumulative_variance_explained), 3L)
  expect_true(all(diff(fit$cumulative_variance_explained) >= -1e-12))
  expect_error(pca(X, ncomp = 3, backend = "cuda", method = "irlba"), "only available with backend='cpu'")
  expect_error(pca(X, ncomp = 3, backend = "metal", method = "irlba"), "only available with backend='cpu'")
})

test_that("fastPLS does not mask base svd", {
  expect_identical(svd, base::svd)
})
