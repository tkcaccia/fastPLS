test_that("Metal rSVD backend returns a valid decomposition when available", {
  skip_if_not(fastPLS::has_metal(), "Metal backend is not available")

  set.seed(42)
  A <- matrix(rnorm(80 * 25), 80, 25)
  decomp <- fastPLS::fastsvd(
    A,
    ncomp = 5,
    backend = "metal",
    method = "rsvd",
    oversample = 8,
    power = 1,
    seed = 123
  )

  expect_equal(dim(decomp$u), c(80L, 5L))
  expect_equal(dim(decomp$v), c(25L, 5L))
  expect_length(decomp$d, 5L)
  expect_true(all(is.finite(decomp$d)))
  expect_true(all(decomp$d >= 0))

  exact <- svd(A, nu = 5, nv = 5)
  expect_true(cor(decomp$d, exact$d[1:5]) > 0.99)
})
