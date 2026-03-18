test_that("cpu_rsvd approximates dc on dense matrices", {
  set.seed(1001)
  A <- matrix(rnorm(80 * 30), nrow = 80, ncol = 30)

  exact <- fastPLS:::truncated_svd_debug(
    A = A,
    k = 6,
    svd_method = 3L,
    rsvd_oversample = 10L,
    rsvd_power = 1L,
    seed = 7L,
    left_only = FALSE
  )
  rsvd <- fastPLS:::truncated_svd_debug(
    A = A,
    k = 6,
    svd_method = 4L,
    rsvd_oversample = 12L,
    rsvd_power = 2L,
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
