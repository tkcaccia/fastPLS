test_that("linear regression head preserves regression predictions", {
  set.seed(123)
  X <- matrix(rnorm(90 * 25), 90, 25)
  Y <- matrix(rnorm(90 * 8), 90, 8)
  train <- 1:60
  test <- 61:90

  for (method in c("plssvd", "simpls")) {
    standard <- pls(
      X[train, , drop = FALSE],
      Y[train, , drop = FALSE],
      X[test, , drop = FALSE],
      Y[test, , drop = FALSE],
      ncomp = c(3, 6),
      method = method,
      backend = "cpp",
      svd.method = "cpu_rsvd",
      rsvd_power = 1L,
      seed = 11L,
      fit = FALSE
    )
    linear <- pls(
      X[train, , drop = FALSE],
      Y[train, , drop = FALSE],
      X[test, , drop = FALSE],
      Y[test, , drop = FALSE],
      ncomp = c(3, 6),
      method = method,
      backend = "cpp",
      svd.method = "cpu_rsvd",
      rsvd_power = 1L,
      seed = 11L,
      fit = FALSE,
      regression_head = "linear_cpp"
    )

    expect_equal(linear$regression_head, "linear_cpp")
    expect_equal(dim(linear$Ypred), dim(standard$Ypred))
    expect_equal(linear$Ypred, standard$Ypred, tolerance = 1e-8)
    expect_equal(linear$Q2Y, standard$Q2Y, tolerance = 1e-8)
  }
})
