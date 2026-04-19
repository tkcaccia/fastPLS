test_that("svd_methods returns expected methods", {
  x <- svd_methods()
  expect_true(is.data.frame(x))
  expect_true(all(c("method", "enabled") %in% colnames(x)))
  expect_true(all(c("irlba", "arpack", "cpu_rsvd") %in% x$method))
  expect_false("cuda_rsvd" %in% x$method)
})

test_that("svd_run returns decomposition outputs", {
  set.seed(1)
  A <- matrix(rnorm(120 * 20), 120, 20)
  out <- svd_run(A, k = 5, method = "arpack")
  expect_true(is.list(out))
  expect_true(all(c("U", "s", "Vt", "method", "elapsed") %in% names(out)))
  expect_equal(ncol(out$U), 5)
  expect_equal(length(out$s), 5)
  expect_equal(nrow(out$Vt), 5)
})

test_that("deprecated dc label is accepted as alias of arpack", {
  set.seed(11)
  A <- matrix(rnorm(80 * 12), 80, 12)
  expect_warning(
    out <- svd_run(A, k = 4, method = "dc"),
    "deprecated"
  )
  expect_equal(out$method, "arpack")
  expect_equal(ncol(out$U), 4L)
})

test_that("svd_benchmark returns rows per method and rep", {
  set.seed(2)
  A <- matrix(rnorm(100 * 30), 100, 30)
  b <- svd_benchmark(A, k = 4, methods = c("irlba", "arpack"), reps = 2)
  expect_true(is.data.frame(b))
  expect_equal(nrow(b), 4)
  expect_true(all(c("method", "rep", "elapsed", "status") %in% colnames(b)))
  expect_true(all(b$status %in% c("ok", "error", "unavailable")))
})
