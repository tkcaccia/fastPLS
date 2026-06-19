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
  expect_true(is.numeric(reg$R2Y))
  expect_null(dim(reg$R2Y))
  expect_true(is.numeric(reg$Q2Y))
  expect_null(dim(reg$Q2Y))
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
    fit = TRUE
  )
  expect_true(is.numeric(cls$R2Y))
  expect_null(dim(cls$R2Y))
  expect_true(is.numeric(cls$Q2Y))
  expect_null(dim(cls$Q2Y))
  expect_true(is.numeric(cls$accuracy))
  expect_null(dim(cls$accuracy))
  expect_identical(names(cls$R2Y), paste0("ncomp=", ncomp))
  expect_identical(names(cls$Q2Y), paste0("ncomp=", ncomp))
  expect_identical(names(cls$accuracy), paste0("ncomp=", ncomp))
})

test_that("single-CV training R2 path is a named vector", {
  set.seed(123)
  X <- as.matrix(iris[, 1:4])
  y <- iris$Species
  cv <- pls.single.cv(
    X,
    y,
    ncomp = 1:2,
    kfold = 3,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    classifier = "argmax",
    fit = TRUE
  )
  expect_true(is.numeric(cv$R2Y))
  expect_null(dim(cv$R2Y))
  expect_identical(names(cv$R2Y), paste0("ncomp=", cv$ncomp))
})
