test_that("pls accepts Biobase ExpressionSet predictors", {
  data("sample.ExpressionSet", package = "Biobase")
  y <- factor(Biobase::pData(sample.ExpressionSet)$type)
  train <- seq_len(18)
  test <- 19:26

  fit <- pls(
    sample.ExpressionSet[, train],
    y[train],
    sample.ExpressionSet[, test],
    y[test],
    ncomp = 1:2,
    method = "simpls",
    backend = "cpu",
    seed = 42
  )

  expect_s3_class(fit, "fastPLS")
  expect_length(fit$Ypred, 2L)
  expect_equal(length(fit$Ypred[[1L]]), length(test))

  predicted <- predict(fit, sample.ExpressionSet[, test])
  expect_length(predicted$Ypred, 2L)
  expect_equal(length(predicted$Ypred[[1L]]), length(test))
})
