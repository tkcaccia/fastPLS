test_that("fastsvd returns decomposition outputs from public backends", {
  set.seed(1)
  A <- matrix(rnorm(120 * 20), 120, 20)
  out <- suppressWarnings(fastsvd(A, ncomp = 5, backend = "cpu"))
  expect_true(is.list(out))
  expect_true(all(c(
    "u", "d", "v", "backend", "method", "svd.method", "elapsed",
    "diagnostics"
  ) %in% names(out)))
  expect_identical(out$backend, "cpu")
  expect_identical(out$method, "rsvd")
  expect_identical(out$svd.method, "cpu_rsvd")
  expect_equal(ncol(out$u), 5)
  expect_equal(length(out$d), 5)
  expect_equal(ncol(out$v), 5)
  expect_true(out$diagnostics$status %in% c(
    "rsvd_triplet_checks_passed",
    "warning_approximation_quality",
    "failed_large_triplet_residual"
  ))
  expect_true(is.finite(out$diagnostics$max_relative_triplet_residual))
  if (identical(out$diagnostics$status, "failed_large_triplet_residual")) {
    expect_gt(
      out$diagnostics$max_relative_triplet_residual,
      out$diagnostics$residual_failure_threshold
    )
  }
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

test_that("fastPLS does not mask base svd", {
  expect_identical(svd, base::svd)
})
