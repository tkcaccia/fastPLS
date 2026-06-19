test_that("pls metric paths are named by component count", {
  set.seed(123)
  X <- as.matrix(iris[, 1:4])
  y_reg <- iris$Sepal.Length
  y_cls <- iris$Species
  ncomp <- c(1L, 2L, 3L)

  reg <- pls(
    X,
    y_reg,
    Xtest = X,
    Ytest = y_reg,
    ncomp = ncomp,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    fit = TRUE,
    return_variance = FALSE
  )
  expect_identical(names(reg$R2Y), paste0("ncomp=", ncomp))
  expect_identical(names(reg$Q2Y), paste0("ncomp=", ncomp))

  cls <- pls(
    X,
    y_cls,
    Xtest = X,
    Ytest = y_cls,
    ncomp = ncomp,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    classifier = "argmax",
    fit = TRUE,
    return_variance = FALSE
  )
  expect_identical(names(cls$R2Y), paste0("ncomp=", ncomp))
  expect_identical(names(cls$Q2Y), paste0("ncomp=", ncomp))
  expect_identical(names(cls$accuracy), paste0("ncomp=", ncomp))
})
