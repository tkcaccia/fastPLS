test_that("pls accepts float32 regression input without upcasting predictions", {
  skip_if_not_installed("float")
  set.seed(10)
  X <- float::fl(as.matrix(mtcars[, c("disp", "hp", "wt", "qsec")]))
  y <- float::fl(matrix(mtcars$mpg, ncol = 1L))

  fit <- pls(
    X,
    y,
    X[1:6, ],
    y[1:6, ],
    ncomp = 1:2,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    fit = TRUE,
    return_variance = FALSE
  )

  expect_s3_class(fit, "fastPLS")
  expect_false(any(grepl("attr(", capture.output(print(fit)), fixed = TRUE)))
  expect_equal(attr(fit, "fastPLS_internal")$precision, "float32")
  expect_true(inherits(fit$Ypred[[1L]], "float32"))
  expect_named(fit$Q2Y, c("ncomp=1", "ncomp=2"))
  expect_false("predict_backend" %in% names(predict(fit, X[1:2, ])))
})

test_that("float32 detection handles S4 float matrices used in the vignette", {
  skip_if_not_installed("float")
  set.seed(12)
  Xreg <- as.matrix(mtcars[, c("disp", "hp", "wt", "qsec", "drat")])
  Yreg <- matrix(mtcars$mpg, ncol = 1)
  idx <- sample(seq_len(nrow(Xreg)), 8)
  Xreg_train <- Xreg[-idx, , drop = FALSE]
  Xreg_test <- Xreg[idx, , drop = FALSE]
  Ytrain_reg <- Yreg[-idx, , drop = FALSE]
  Ytest_reg <- Yreg[idx, , drop = FALSE]

  Xreg32 <- float::fl(as.matrix(Xreg_train))
  Yreg32 <- float::fl(matrix(Ytrain_reg, ncol = 1))
  expect_true(methods::is(Xreg32, "float32"))
  expect_true(.has_float32_input(Xreg32, Yreg32))

  fit_reg32 <- pls(
    Xreg32,
    Yreg32,
    float::fl(as.matrix(Xreg_test)),
    float::fl(matrix(Ytest_reg, ncol = 1)),
    ncomp = 1:2,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    return_variance = FALSE
  )

  expect_equal(attr(fit_reg32, "fastPLS_internal")$precision, "float32")
  expect_named(fit_reg32$Q2Y, c("ncomp=1", "ncomp=2"))
  expect_true(all(is.finite(fit_reg32$Q2Y)))
})

test_that("pls accepts float32 classification input with argmax", {
  skip_if_not_installed("float")
  set.seed(11)
  X <- float::fl(as.matrix(iris[, 1:4]))
  y <- iris$Species

  fit <- pls(
    X,
    y,
    X[1:12, ],
    y[1:12],
    ncomp = 2,
    method = "plssvd",
    backend = "cpu",
    svd.method = "rsvd",
    classifier = "argmax",
    return_variance = FALSE
  )

  expect_s3_class(fit, "fastPLS")
  expect_false(any(grepl("attr(", capture.output(print(fit)), fixed = TRUE)))
  expect_equal(attr(fit, "fastPLS_internal")$precision, "float32")
  expect_true(is.factor(fit$Ypred[[1L]]))
  expect_named(fit$accuracy, "ncomp=2")
})

test_that("float32 input refuses unsupported non-float routes", {
  skip_if_not_installed("float")
  X <- float::fl(as.matrix(iris[, 1:4]))
  y <- iris$Species

  expect_error(
    pls(X, y, ncomp = 2, backend = "cuda", return_variance = FALSE),
    "backend = 'cpu'"
  )
  expect_error(
    pls(X, y, ncomp = 2, backend = "cpu", svd.method = "irlba", return_variance = FALSE),
    "svd.method = 'rsvd'"
  )
})
