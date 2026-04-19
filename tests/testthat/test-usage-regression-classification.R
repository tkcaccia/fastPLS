test_that("pls and predict support regression workflow", {
  set.seed(1001)
  X <- matrix(rnorm(80 * 10), nrow = 80, ncol = 10)
  Y <- matrix(rnorm(80 * 2), nrow = 80, ncol = 2)
  idx <- sample(seq_len(80), 16)

  fit <- pls(
    X[-idx, , drop = FALSE],
    Y[-idx, , drop = FALSE],
    X[idx, , drop = FALSE],
    Y[idx, , drop = FALSE],
    ncomp = 1:3,
    method = "simpls",
    svd.method = "arpack",
    fit = TRUE
  )

  expect_s3_class(fit, "fastPLS")
  expect_equal(dim(fit$B), c(ncol(X), ncol(Y), 3))
  expect_true(is.array(fit$Ypred))
  expect_equal(dim(fit$Ypred), c(length(idx), ncol(Y), 3))
  expect_length(fit$Q2Y, 3L)

  pr <- predict(fit, X[idx, , drop = FALSE], Ytest = Y[idx, , drop = FALSE], proj = TRUE)
  expect_true(is.array(pr$Ypred))
  expect_length(pr$Q2Y, 3L)
  expect_true(is.matrix(pr$Ttest))
})

test_that("pls and predict support classification workflow", {
  set.seed(1002)
  X <- matrix(rnorm(90 * 12), nrow = 90, ncol = 12)
  y <- factor(sample(c("A", "B", "C"), 90, replace = TRUE))
  idx <- sample(seq_len(90), 20)

  fit <- pls(
    X[-idx, , drop = FALSE],
    y[-idx],
    X[idx, , drop = FALSE],
    y[idx],
    ncomp = 1:2,
    method = "plssvd",
    svd.method = "arpack",
    fit = TRUE
  )

  expect_s3_class(fit, "fastPLS")
  expect_true(is.data.frame(fit$Ypred))
  expect_equal(nrow(fit$Ypred), length(idx))
  expect_equal(ncol(fit$Ypred), 2L)
  expect_true(is.data.frame(fit$Yfit))
  expect_equal(levels(fit$Ypred[[1]]), levels(y))

  pr <- predict(fit, X[idx, , drop = FALSE], Ytest = y[idx], proj = FALSE)
  expect_true(is.data.frame(pr$Ypred))
  expect_length(pr$Q2Y, 2L)
})

test_that("pls_r supports regression and classification workflows", {
  set.seed(1003)
  X <- matrix(rnorm(84 * 9), nrow = 84, ncol = 9)
  Y <- matrix(rnorm(84 * 2), nrow = 84, ncol = 2)
  y <- factor(sample(c("ctrl", "case"), 84, replace = TRUE))
  idx <- sample(seq_len(84), 18)

  reg_fit <- pls_r(
    X[-idx, , drop = FALSE],
    Y[-idx, , drop = FALSE],
    X[idx, , drop = FALSE],
    Y[idx, , drop = FALSE],
    ncomp = 1:2,
    method = "plssvd",
    svd.method = "arpack",
    fit = TRUE
  )
  expect_s3_class(reg_fit, "fastPLS")
  expect_true(is.array(reg_fit$Ypred))
  expect_length(reg_fit$Q2Y, 2L)

  cls_fit <- pls_r(
    X[-idx, , drop = FALSE],
    y[-idx],
    X[idx, , drop = FALSE],
    y[idx],
    ncomp = 1:2,
    method = "plssvd",
    svd.method = "cpu_rsvd",
    seed = 123L,
    fit = TRUE
  )
  expect_s3_class(cls_fit, "fastPLS")
  expect_true(is.data.frame(cls_fit$Ypred))
  expect_equal(levels(cls_fit$Ypred[[1]]), levels(y))
})

test_that("optim.pls.cv and pls.double.cv run in both contexts", {
  set.seed(1004)
  X <- matrix(rnorm(60 * 8), nrow = 60, ncol = 8)
  Yreg <- matrix(rnorm(60), ncol = 1)
  ycls <- factor(sample(c("L", "M", "H"), 60, replace = TRUE))

  cv_reg <- optim.pls.cv(
    Xdata = X,
    Ydata = Yreg,
    ncomp = 1:2,
    kfold = 3,
    method = "simpls",
    svd.method = "arpack"
  )
  expect_true(is.list(cv_reg))
  expect_true("Q2Y" %in% names(cv_reg))

  cv_cls <- optim.pls.cv(
    Xdata = X,
    Ydata = ycls,
    ncomp = 1:2,
    kfold = 3,
    method = "simpls",
    svd.method = "arpack"
  )
  expect_true(is.list(cv_cls))
  expect_true("Q2Y" %in% names(cv_cls))

  dcv_reg <- pls.double.cv(
    Xdata = X,
    Ydata = Yreg,
    ncomp = 1:2,
    runn = 2,
    kfold_inner = 3,
    kfold_outer = 3,
    method = "simpls",
    svd.method = "arpack"
  )
  expect_true(is.list(dcv_reg))
  expect_true("Q2Y" %in% names(dcv_reg))
  expect_true("R2Y" %in% names(dcv_reg))
  expect_length(dcv_reg$Q2Y, 2L)
  expect_length(dcv_reg$R2Y, 2L)

  dcv_cls <- pls.double.cv(
    Xdata = X,
    Ydata = ycls,
    ncomp = 1:2,
    runn = 2,
    kfold_inner = 3,
    kfold_outer = 3,
    method = "simpls",
    svd.method = "arpack"
  )
  expect_true(is.list(dcv_cls))
  expect_true(is.factor(dcv_cls$Ypred))
  expect_true("acc_tot" %in% names(dcv_cls))
})

test_that("SVD utilities and helper functions are usable in practice", {
  set.seed(1005)
  A <- matrix(rnorm(70 * 14), nrow = 70, ncol = 14)

  sm <- svd_methods()
  expect_true(is.data.frame(sm))
  expect_equal(colnames(sm), c("method", "enabled"))
  expect_true(all(c("irlba", "arpack", "cpu_rsvd") %in% sm$method))
  expect_false("cuda_rsvd" %in% sm$method)

  sr <- svd_run(A, k = 4, method = "arpack")
  expect_true(is.list(sr))
  expect_true(all(c("U", "s", "Vt", "method", "elapsed") %in% names(sr)))
  expect_equal(ncol(sr$U), 4L)
  expect_equal(length(sr$s), 4L)

  sb <- svd_benchmark(A, k = 4, methods = c("irlba", "arpack", "cpu_rsvd"), reps = 2L)
  expect_true(is.data.frame(sb))
  expect_equal(nrow(sb), 6L)
  expect_true(all(c("method", "rep", "elapsed", "status") %in% names(sb)))
  expect_true(all(sb$status %in% c("ok", "error", "unavailable")))

  x <- factor(c("a", "b", "a", "c", "b", "a"))
  tx <- transformy(x)
  expect_true(is.matrix(tx))
  expect_equal(nrow(tx), length(x))

  C1 <- fastcor(A, byrow = FALSE, diag = FALSE)
  expect_true(is.matrix(C1))
  expect_equal(dim(C1), c(ncol(A), ncol(A)))

  C2 <- fastcor(A[, 1:5, drop = FALSE], A[, 1:5, drop = FALSE], byrow = FALSE, diag = TRUE)
  expect_true(is.numeric(C2))
  expect_equal(length(C2), 5L)

  model_uni <- pls(A, matrix(rnorm(nrow(A)), ncol = 1), ncomp = 1:3, method = "simpls", svd.method = "arpack")
  vip_uni <- ViP(model_uni)
  expect_true(is.matrix(vip_uni))

  model_multi <- pls(A, matrix(rnorm(nrow(A) * 2), ncol = 2), ncomp = 1:3, method = "simpls", svd.method = "arpack")
  vip_multi <- ViP(model_multi)
  expect_true(is.list(vip_multi))
  expect_equal(length(vip_multi), 2L)
})
