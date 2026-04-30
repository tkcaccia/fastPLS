test_that("cpu_rsvd approximates exact SVD on dense matrices", {
  set.seed(1001)
  A <- matrix(rnorm(80 * 30), nrow = 80, ncol = 30)

  exact <- fastPLS:::truncated_svd_debug(
    A = A,
    k = 6,
    svd_method = 3L,
    rsvd_oversample = 10L,
    rsvd_power = 1L,
    svds_tol = 0,
    seed = 7L,
    left_only = FALSE
  )
  rsvd <- fastPLS:::truncated_svd_debug(
    A = A,
    k = 6,
    svd_method = 4L,
    rsvd_oversample = 12L,
    rsvd_power = 2L,
    svds_tol = 0,
    seed = 7L,
    left_only = FALSE
  )

  expect_equal(length(exact$d), 6L)
  expect_equal(length(rsvd$d), 6L)
  expect_equal(rsvd$d, exact$d, tolerance = 1e-2)

  proj_exact <- exact$u %*% t(exact$u)
  proj_rsvd <- rsvd$u %*% t(rsvd$u)
  expect_lt(norm(proj_exact - proj_rsvd, type = "F"), 0.5)
})

test_that("block-Krylov rsvd variant returns a valid truncated SVD", {
  old_variant <- Sys.getenv("FASTPLS_RSVD_VARIANT", unset = NA_character_)
  old_flag <- Sys.getenv("FASTPLS_RSVD_BLOCK_KRYLOV", unset = NA_character_)
  on.exit({
    if (is.na(old_variant)) Sys.unsetenv("FASTPLS_RSVD_VARIANT") else Sys.setenv(FASTPLS_RSVD_VARIANT = old_variant)
    if (is.na(old_flag)) Sys.unsetenv("FASTPLS_RSVD_BLOCK_KRYLOV") else Sys.setenv(FASTPLS_RSVD_BLOCK_KRYLOV = old_flag)
  }, add = TRUE)

  set.seed(20260430)
  A <- matrix(rnorm(120 * 45), nrow = 120, ncol = 45)
  Sys.setenv(FASTPLS_RSVD_VARIANT = "block_krylov")
  out <- fastPLS:::truncated_svd_debug(
    A = A,
    k = 8,
    svd_method = 4L,
    rsvd_oversample = 4L,
    rsvd_power = 2L,
    svds_tol = 0,
    seed = 9L,
    left_only = FALSE
  )

  expect_equal(dim(out$u), c(120L, 8L))
  expect_equal(length(out$d), 8L)
  expect_equal(dim(out$v), c(45L, 8L))
  expect_true(all(is.finite(out$d)))
  expect_true(all(diff(out$d) <= sqrt(.Machine$double.eps)))
})
