test_that("pls supports all declared method/backend combinations (smoke)", {
  set.seed(20260318)
  X <- matrix(rnorm(72 * 10), nrow = 72, ncol = 10)
  y_reg <- matrix(rnorm(72 * 2), ncol = 2)
  y_cls <- factor(sample(c("A", "B", "C"), 72, replace = TRUE))
  idx <- sample(seq_len(72), 18)

  back <- c("exact", "irlba", "cpu_rsvd")

  for (m in c("plssvd", "simpls")) {
    for (s in back) {
      fit_reg <- pls(
        X[-idx, , drop = FALSE],
        y_reg[-idx, , drop = FALSE],
        X[idx, , drop = FALSE],
        y_reg[idx, , drop = FALSE],
        ncomp = 1:2,
        method = m,
        svd.method = s,
        fit = TRUE
      )
      expect_s3_class(fit_reg, "fastPLS")
      expect_true("Ypred" %in% names(fit_reg))

      fit_cls <- pls(
        X[-idx, , drop = FALSE],
        y_cls[-idx],
        X[idx, , drop = FALSE],
        y_cls[idx],
        ncomp = 1:2,
        method = m,
        svd.method = s,
        fit = TRUE
      )
      expect_s3_class(fit_cls, "fastPLS")
      expect_true(is.data.frame(fit_cls$Ypred))
    }
  }
})

test_that("optim.pls.cv and pls.double.cv support accelerated simpls", {
  set.seed(20260318)
  X <- matrix(rnorm(54 * 8), nrow = 54, ncol = 8)
  y_reg <- matrix(rnorm(54 * 2), ncol = 2)
  y_cls <- factor(sample(c("L", "M", "H"), 54, replace = TRUE))

  back <- c("exact", "irlba", "cpu_rsvd")

  for (s in back) {
    for (m in c("plssvd", "simpls")) {
      cv_reg <- optim.pls.cv(
        Xdata = X,
        Ydata = y_reg,
        ncomp = 1:2,
        kfold = 3,
        method = m,
        svd.method = s
      )
      expect_true(is.list(cv_reg))
      expect_true("Q2Y" %in% names(cv_reg))

      cv_cls <- optim.pls.cv(
        Xdata = X,
        Ydata = y_cls,
        ncomp = 1:2,
        kfold = 3,
        method = m,
        svd.method = s
      )
      expect_true(is.list(cv_cls))
      expect_true("Q2Y" %in% names(cv_cls))

      dcv_reg <- pls.double.cv(
        Xdata = X,
        Ydata = y_reg,
        ncomp = 1:2,
        runn = 1,
        kfold_inner = 3,
        kfold_outer = 3,
        method = m,
        svd.method = s
      )
      expect_true(is.list(dcv_reg))
      expect_true("Q2Y" %in% names(dcv_reg))

      dcv_cls <- pls.double.cv(
        Xdata = X,
        Ydata = y_cls,
        ncomp = 1:2,
        runn = 1,
        kfold_inner = 3,
        kfold_outer = 3,
        method = m,
        svd.method = s
      )
      expect_true(is.list(dcv_cls))
      expect_true("Ypred" %in% names(dcv_cls))
    }
  }
})

test_that("pls backend='r' supports accelerated simpls for regression and classification", {
  set.seed(20260321)
  X <- matrix(rnorm(60 * 9), nrow = 60, ncol = 9)
  y_reg <- matrix(rnorm(60 * 2), ncol = 2)
  y_cls <- factor(sample(c("A", "B", "C"), 60, replace = TRUE))
  idx <- sample(seq_len(60), 15)

  for (s in c("irlba", "cpu_rsvd")) {
    fit_reg <- pls(
      X[-idx, , drop = FALSE],
      y_reg[-idx, , drop = FALSE],
      X[idx, , drop = FALSE],
      y_reg[idx, , drop = FALSE],
      ncomp = 1:3,
      method = "simpls",
      backend = "r",
      svd.method = s,
      fit = TRUE
    )
    expect_s3_class(fit_reg, "fastPLS")
    expect_true("Ypred" %in% names(fit_reg))
    expect_equal(dim(fit_reg$B), c(ncol(X), ncol(y_reg), 3L))

    fit_cls <- pls(
      X[-idx, , drop = FALSE],
      y_cls[-idx],
      X[idx, , drop = FALSE],
      y_cls[idx],
      ncomp = 1:2,
      method = "simpls",
      backend = "r",
      svd.method = s,
      fit = TRUE
    )
    expect_s3_class(fit_cls, "fastPLS")
    expect_true(is.data.frame(fit_cls$Ypred))
    expect_equal(ncol(fit_cls$Ypred), 2L)
  }
})
