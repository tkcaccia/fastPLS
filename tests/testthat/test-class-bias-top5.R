test_that("top-k classification prediction preserves argmax by default", {
  set.seed(20260511)
  X <- matrix(rnorm(96 * 10), nrow = 96, ncol = 10)
  y <- factor(sample(paste0("C", seq_len(6)), 96, replace = TRUE))
  idx <- sample(seq_len(96), 18)

  fit <- pls(
    X[-idx, , drop = FALSE],
    y[-idx],
    ncomp = 1:3,
    method = "plssvd",
    svd.method = "cpu_rsvd",
    seed = 123L
  )

  fast <- predict(fit, X[idx, , drop = FALSE])
  full <- predict(fit, X[idx, , drop = FALSE], raw_scores = TRUE)
  expect_equal(fast$Ypred, full$Ypred)

  top5 <- predict(fit, X[idx, , drop = FALSE], top5 = TRUE)
  expect_true(is.list(top5$Ypred_top))
  expect_equal(dim(top5$Ypred_top[[1]]), c(length(idx), 5L))
  expect_equal(dim(top5$Ypred_top_score[[1]]), c(length(idx), 5L))
})

test_that("class-bias classifier is fitted and can be used for top-k prediction", {
  set.seed(20260512)
  X <- matrix(rnorm(90 * 9), nrow = 90, ncol = 9)
  y <- factor(sample(c("A", "B", "C"), 90, replace = TRUE))
  idx <- sample(seq_len(90), 20)

  fit <- pls(
    X[-idx, , drop = FALSE],
    y[-idx],
    ncomp = 1:2,
    method = "simpls",
    svd.method = "cpu_rsvd",
    classifier = "class_bias_cpp",
    class_bias_method = "iter_count_ratio",
    class_bias_lambda = 0.025,
    class_bias_iter = 3L,
    class_bias_clip = 0.2,
    class_bias_calibration_fraction = 0.5,
    seed = 123L
  )

  expect_equal(fit$classification_rule, "class_bias_cpp")
  expect_true(is.matrix(fit$class_bias))
  expect_equal(dim(fit$class_bias), c(nlevels(y), 2L))
  expect_equal(fit$class_bias_parameters$method, "iter_count_ratio")
  expect_equal(fit$class_bias_parameters$iter, 3L)
  expect_equal(fit$class_bias_parameters$lambda, 0.025)
  expect_equal(fit$class_bias_parameters$clip, 0.2)
  expect_true(fit$class_bias_parameters$n_calibration < length(y[-idx]))

  pred <- predict(fit, X[idx, , drop = FALSE], top = 3L)
  expect_true(is.data.frame(pred$Ypred))
  expect_equal(dim(pred$Ypred_top[[1]]), c(length(idx), 3L))

  forced <- predict(
    fit,
    X[idx, , drop = FALSE],
    class_bias = NULL,
    raw_scores = TRUE
  )
  expect_true(is.data.frame(forced$Ypred))
})

test_that("count-ratio class-bias calibration records one pass", {
  set.seed(20260513)
  X <- matrix(rnorm(72 * 8), nrow = 72, ncol = 8)
  y <- factor(sample(c("A", "B", "C", "D"), 72, replace = TRUE))

  fit <- pls(
    X,
    y,
    ncomp = 2,
    method = "plssvd",
    svd.method = "cpu_rsvd",
    classifier = "class_bias_cpp",
    class_bias_method = "count_ratio",
    class_bias_lambda = 0.01,
    class_bias_iter = 5L,
    return_variance = FALSE,
    seed = 321L
  )

  expect_equal(fit$class_bias_parameters$method, "count_ratio")
  expect_equal(fit$class_bias_parameters$iter, 1L)
  expect_equal(fit$class_bias_parameters$n_calibration, length(y))
  expect_true(all(is.finite(fit$class_bias)))
})

test_that("label-aware PLSSVD model avoids dense response storage", {
  set.seed(20260514)
  X <- matrix(rnorm(80 * 12), nrow = 80, ncol = 12)
  y <- factor(sample(paste0("K", seq_len(5)), 80, replace = TRUE))
  idx <- sample(seq_len(80), 15)
  fit_fun <- get(".plssvd_label_aware_stream_model", envir = asNamespace("fastPLS"))
  fit <- fit_fun(
    X[-idx, , drop = FALSE],
    y[-idx],
    ncomp = 1:3,
    scaling = 1L,
    backend = "cpp",
    block_size = 13L
  )
  expect_true(isTRUE(fit$classification))
  expect_null(fit[["B"]])
  expect_equal(fit$xprod_mode, "label_aware_stream")

  pred <- predict(fit, X[idx, , drop = FALSE], top5 = TRUE)
  expect_equal(dim(pred$Ypred_top[[1]]), c(length(idx), 5L))
  expect_true(all(as.character(pred$Ypred[[1]]) %in% levels(y)))
})
