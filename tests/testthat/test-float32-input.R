skip_native_float32_on_windows <- function() {
  skip_if(
    .Platform$OS.type == "windows",
    "native float32 kernels are not available on Windows"
  )
}

expected_float32_cpu_backend <- function() {
  if (identical(.Platform$OS.type, "windows")) {
    "float32_windows_float"
  } else {
    "float32_cpp"
  }
}

test_that("Windows float32 argmax uses the portable compiled entry point", {
  skip_if_not_installed("float")
  skip_if_not(.Platform$OS.type == "windows", "Windows-only fallback")
  scores <- float::fl(matrix(c(1, 3, 2, 4), nrow = 2L))
  expect_identical(fastPLS:::float32_argmax_cpp(scores), c(2L, 2L))
})

test_that("portable float32 OPLS and LDA helpers retain single precision", {
  skip_if_not_installed("float")
  set.seed(148)
  X <- float::fl(matrix(rnorm(72L * 8L), 72L, 8L))
  y <- factor(rep(letters[1:3], each = 24L))
  Y <- float::fl(fastPLS:::transformy(y))

  filtered <- fastPLS:::.float32_portable_opls_filter(
    X, Y, north = 1L, scaling = 1L,
    rsvd_oversample = 8L, rsvd_power = 2L, seed = 148L
  )
  reapplied <- fastPLS:::.float32_portable_opls_apply(
    X, filtered$mX, filtered$vX, filtered$W_orth, filtered$P_orth
  )
  expect_true(inherits(filtered$X, "float32"))
  expect_true(inherits(reapplied, "float32"))
  expect_equal(dim(reapplied), dim(X))

  scores <- filtered$X[, 1:3, drop = FALSE]
  lda <- fastPLS:::.float32_portable_lda_train_prefix(
    scores, as.integer(y), n_classes = 3L, ncomp = c(1L, 3L)
  )
  pred <- fastPLS:::.float32_portable_lda_predict(scores, lda[["3"]])
  expect_true(inherits(lda[["3"]]$linear, "float32"))
  expect_true(inherits(pred$scores, "float32"))
  expect_length(pred$pred, nrow(X))
})

test_that("portable Windows float32 CPU fallback preserves float32 PLS data", {
  skip_if_not_installed("float")
  skip_if_not(.Platform$OS.type == "windows", "Windows-only fallback")
  set.seed(149)
  X <- float::fl(as.matrix(mtcars[, c("disp", "hp", "wt", "qsec")]))
  y <- float::fl(matrix(mtcars$mpg, ncol = 1L))
  prep <- fastPLS:::.float32_prepare_response(y)

  fit <- fastPLS:::.float32_windows_cpu_fit(
    Xtrain = X,
    yprep = prep,
    ncomp = 1:2,
    scaling = 1L,
    method = "simpls",
    backend = "cpu",
    svd.method = "cpu_rsvd",
    rsvd_oversample = 5L,
    rsvd_power = 1L,
    seed = 149L,
    fit = TRUE
  )

  expect_identical(fit$precision, "float32")
  expect_identical(fit$predict_backend, "float32_windows_float")
  expect_true(inherits(fit$R, "float32"))
  expect_true(inherits(fit$Q, "float32"))
  expect_true(all(is.finite(fit$R2Y)))
  expect_error(
    fastPLS:::.float32_windows_cpu_fit(
      X, prep, 1L, 1L, "simpls", "cuda", "cpu_rsvd", 5L, 1L, 149L, FALSE
    ),
    "backend = 'cpu'"
  )
})

test_that("Windows public float32 OPLS and nonlinear kernel PLS support LDA", {
  skip_if_not_installed("float")
  skip_if_not(.Platform$OS.type == "windows", "Windows-only fallback")
  set.seed(1491)
  train <- sample(seq_len(nrow(iris)), 105L)
  Xtrain <- float::fl(as.matrix(iris[train, 1:4]))
  Xtest <- float::fl(as.matrix(iris[-train, 1:4]))
  ytrain <- droplevels(iris$Species[train])
  ytest <- factor(iris$Species[-train], levels = levels(ytrain))

  for (method in c("opls", "kernelpls")) {
    fit <- suppressWarnings(pls(
      Xtrain, ytrain, Xtest, ytest,
      ncomp = 2L, method = method, kernel = "rbf", north = 1L,
      backend = "cpu", svd.method = "rsvd", classifier = "lda",
      return_variance = FALSE, seed = 1491L
    ))
    expect_identical(attr(fit, "fastPLS_internal")$precision, "float32")
    expect_true(is.factor(fit$Ypred[[1L]]))
    expect_true(all(is.finite(fit$accuracy)))
  }
})

test_that("portable Windows float32 SVD fallback returns float32 vectors", {
  skip_if_not_installed("float")
  skip_if_not(.Platform$OS.type == "windows", "Windows-only fallback")
  set.seed(150)
  A <- float::fl(matrix(rnorm(48), nrow = 12L))
  out <- fastPLS:::.fastsvd_float32_windows(
    A, k = 3L, backend = "cpu", svd.method = "cpu_rsvd",
    oversample = 4L, power = 1L, seed = 150L
  )

  expect_true(inherits(out$U, "float32"))
  expect_true(inherits(out$Vt, "float32"))
  expect_length(out$s, 3L)
  expect_true(all(is.finite(out$s)))
})

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
  expect_equal(attr(fit, "fastPLS_internal")$predict_backend, expected_float32_cpu_backend())
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
  expect_true(fastPLS:::.has_float32_input(Xreg32, Yreg32))

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
  expect_equal(attr(fit_reg32, "fastPLS_internal")$predict_backend, expected_float32_cpu_backend())
  expect_named(fit_reg32$Q2Y, c("ncomp=1", "ncomp=2"))
  expect_true(all(is.finite(fit_reg32$Q2Y)))

  fit_reg64 <- pls(
    Xreg_train,
    Ytrain_reg,
    Xreg_test,
    Ytest_reg,
    ncomp = 1:2,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    return_variance = FALSE
  )
  expect_equal(unname(fit_reg32$Q2Y), unname(fit_reg64$Q2Y), tolerance = 1e-3)
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
  expect_equal(attr(fit, "fastPLS_internal")$predict_backend, expected_float32_cpu_backend())
  expect_true(is.factor(fit$Ypred[[1L]]))
  expect_named(fit$accuracy, "ncomp=2")
})

test_that("float32 label-aware products match dense one-hot fitting", {
  skip_if_not_installed("float")
  skip_native_float32_on_windows()
  set.seed(151)
  n <- 180L
  X <- float::fl(matrix(rnorm(n * 24L), n, 24L))
  y <- factor(sample(letters[1:5], n, replace = TRUE))
  dense_y <- float::fl(fastPLS:::transformy(as.integer(y)))

  for (method in c(plssvd = 1L, simpls = 3L)) {
    compact <- fastPLS:::pls_float32_labels_cpp(
      X, as.integer(y), nlevels(y), c(2L, 4L), 1L, TRUE,
      method, 0L, 3L, 8L, 2L, 151L
    )
    dense <- fastPLS:::pls_float32_cpu_cpp(
      X, dense_y, c(2L, 4L), 1L, TRUE,
      method, 0L, 3L, 8L, 2L, 151L
    )
    compact_r <- float::dbl(fastPLS:::.float32_from_bits(compact$R))
    dense_r <- float::dbl(fastPLS:::.float32_from_bits(dense$R))

    expect_equal(abs(compact_r), abs(dense_r), tolerance = 2e-3)
    expect_equal(compact$R2Y, dense$R2Y, tolerance = 2e-3)
    expect_identical(compact$xprod_mode, "float32_label_class_sums")
  }
})

test_that("float32 classification avoids fitted and double-score work by default", {
  skip_if_not_installed("float")
  skip_native_float32_on_windows()
  set.seed(152)
  X <- float::fl(matrix(rnorm(150L * 18L), 150L, 18L))
  y <- factor(rep(letters[1:5], each = 30L))
  fit <- suppressWarnings(pls(
    X, y,
    ncomp = c(2L, 4L),
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    fit = FALSE,
    return_variance = FALSE,
    power = 2L,
    seed = 152L
  ))

  expect_null(fit$Yfit)
  expect_identical(fit$xprod_mode, "float32_label_class_sums")
  pred <- predict(fit, X)
  expect_true(is.factor(pred$Ypred[["ncomp=4"]]))

  scores <- float::fl(matrix(c(1, 3, 2, 4, 2, 0), nrow = 2L))
  expect_identical(fastPLS:::float32_argmax_cpp(scores), c(2L, 2L))
})

test_that("float32 input refuses unsupported non-float routes", {
  skip_if_not_installed("float")
  skip_native_float32_on_windows()
  X <- float::fl(as.matrix(iris[, 1:4]))
  y <- iris$Species

  if (has_cuda()) {
    expect_s3_class(
      pls(X, y, ncomp = 2, backend = "cuda", return_variance = FALSE),
      "fastPLS"
    )
  } else {
    expect_error(
      pls(X, y, ncomp = 2, backend = "cuda", return_variance = FALSE),
      "requires a CUDA-enabled fastPLS build"
    )
  }
})

test_that("float32 input supports CPU IRLBA-style SVD", {
  skip_if_not_installed("float")
  skip_native_float32_on_windows()
  set.seed(13)
  Xreg <- as.matrix(mtcars[, c("disp", "hp", "wt", "qsec", "drat")])
  Yreg <- matrix(mtcars$mpg, ncol = 1)
  idx <- sample(seq_len(nrow(Xreg)), 8)
  fit_reg32 <- pls(
    float::fl(Xreg[-idx, , drop = FALSE]),
    float::fl(Yreg[-idx, , drop = FALSE]),
    float::fl(Xreg[idx, , drop = FALSE]),
    float::fl(Yreg[idx, , drop = FALSE]),
    ncomp = 1:2,
    method = "simpls",
    backend = "cpu",
    svd.method = "irlba",
    return_variance = FALSE
  )

  expect_equal(attr(fit_reg32, "fastPLS_internal")$precision, "float32")
  expect_equal(attr(fit_reg32, "fastPLS_internal")$predict_backend, "float32_cpp")
  expect_named(fit_reg32$Q2Y, c("ncomp=1", "ncomp=2"))
  expect_true(all(is.finite(fit_reg32$Q2Y)))

  fit_cls32 <- pls(
    float::fl(as.matrix(iris[, 1:4])),
    iris$Species,
    float::fl(as.matrix(iris[1:15, 1:4])),
    iris$Species[1:15],
    ncomp = 2,
    method = "plssvd",
    backend = "cpu",
    svd.method = "irlba",
    classifier = "argmax",
    return_variance = FALSE
  )
  expect_true(is.factor(fit_cls32$Ypred[[1L]]))
  expect_named(fit_cls32$accuracy, "ncomp=2")
})

test_that("pls.single.cv preserves float32 input instead of entering the double CV kernel", {
  skip_if_not_installed("float")
  skip_native_float32_on_windows()

  set.seed(148)
  X <- float::fl(matrix(rnorm(72 * 8), 72, 8))
  y <- factor(rep(c("a", "b", "c"), each = 24))
  cv <- pls.single.cv(
    X,
    y,
    ncomp = 1:2,
    kfold = 3,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    fit = FALSE,
    seed = 148
  )

  internal <- attr(cv, "fastPLS_internal", exact = TRUE)
  expect_identical(internal$precision, "float32")
  expect_identical(internal$cv_engine, "float32_fold_pls")
  expect_true(all(is.finite(cv$accuracy)))
  expect_true(cv$best_ncomp %in% 1:2)
})

test_that("CUDA float32 rSVD sketch matches CPU float32 arithmetic", {
  skip_if_not_installed("float")
  skip_native_float32_on_windows()
  skip_if_not(has_cuda(), "CUDA backend not available")

  set.seed(123)
  A <- float::fl(matrix(rnorm(30), nrow = 6))
  out <- fastPLS:::cuda_float32_rsvd_sample_cpp(A, l = 3L, power_iters = 1L, seed = 44L)
  Y_cuda <- fastPLS:::.float32_from_bits(out$Y)
  Omega <- fastPLS:::.float32_from_bits(out$Omega)
  Y_cpu <- A %*% Omega
  Y_cpu <- A %*% (crossprod(A, Y_cpu))

  expect_true(inherits(Y_cuda, "float32"))
  expect_equal(as.numeric(Y_cuda), as.numeric(Y_cpu), tolerance = 1e-4)
})

test_that("public CUDA float32 rSVD and PLS routes work when available", {
  skip_if_not_installed("float")
  skip_native_float32_on_windows()
  skip_if_not(has_cuda(), "CUDA backend not available")

  set.seed(1234)
  A <- float::fl(matrix(rnorm(120), nrow = 20))
  sv <- fastsvd(A, ncomp = 3, backend = "cuda", method = "rsvd", seed = 9)
  expect_true(inherits(sv$u, "float32"))
  expect_true(inherits(sv$v, "float32"))
  expect_identical(sv$precision, "float32")
  expect_equal(dim(sv$u), c(20L, 3L))
  expect_error(
    fastsvd(A, ncomp = 3, backend = "cuda", method = "irlba", seed = 9),
    "supports method = 'rsvd' only"
  )

  X <- float::fl(as.matrix(iris[, 1:4]))
  y <- iris$Species
  fit <- pls(
    X,
    y,
    X[1:20, ],
    y[1:20],
    ncomp = 2,
    method = "simpls",
    backend = "cuda",
    svd.method = "rsvd",
    classifier = "argmax",
    return_variance = FALSE
  )
  expect_s3_class(fit, "fastPLS")
  expect_equal(attr(fit, "fastPLS_internal")$precision, "float32")
  expect_equal(attr(fit, "fastPLS_internal")$predict_backend, "float32_cuda")
  expect_true(is.factor(fit$Ypred[[1L]]))
  expect_named(fit$accuracy, "ncomp=2")
})

test_that("Metal float32 matrix multiply stays float32", {
  skip_if_not_installed("float")
  skip_native_float32_on_windows()
  skip_if_not(has_metal(), "Metal backend not available")

  set.seed(124)
  A <- float::fl(matrix(rnorm(24), nrow = 6))
  B <- float::fl(matrix(rnorm(12), nrow = 4))
  out <- fastPLS:::metal_float32_matrix_multiply_cpp(A, B)
  C_metal <- fastPLS:::.float32_from_bits(out$C)
  C_cpu <- A %*% B

  expect_true(inherits(C_metal, "float32"))
  expect_equal(dim(C_metal), dim(C_cpu))
  expect_equal(as.numeric(C_metal), as.numeric(C_cpu), tolerance = 1e-4)

  D <- float::fl(matrix(rnorm(18), nrow = 6))
  out_t <- fastPLS:::metal_float32_matrix_multiply_cpp(A, D, transpose_left = TRUE)
  C_t_metal <- fastPLS:::.float32_from_bits(out_t$C)
  C_t_cpu <- crossprod(A, D)
  expect_true(inherits(C_t_metal, "float32"))
  expect_equal(dim(C_t_metal), dim(C_t_cpu))
  expect_equal(as.numeric(C_t_metal), as.numeric(C_t_cpu), tolerance = 1e-4)
})

test_that("Metal float32 rSVD sketch matches CPU float32 arithmetic", {
  skip_if_not_installed("float")
  skip_native_float32_on_windows()
  skip_if_not(has_metal(), "Metal backend not available")

  set.seed(125)
  A <- float::fl(matrix(rnorm(30), nrow = 6))
  out <- fastPLS:::metal_float32_rsvd_sample_cpp(A, l = 3L, power_iters = 1L, seed = 45L)
  Y_metal <- fastPLS:::.float32_from_bits(out$Y)
  Omega <- fastPLS:::.float32_from_bits(out$Omega)
  Y_cpu <- A %*% Omega
  Y_cpu <- A %*% (crossprod(A, Y_cpu))

  expect_true(inherits(Y_metal, "float32"))
  expect_equal(dim(Y_metal), dim(Y_cpu))
  expect_equal(as.numeric(Y_metal), as.numeric(Y_cpu), tolerance = 1e-3)
})

test_that("Metal float32 IRLBA-style SVD returns a valid approximation", {
  skip_if_not_installed("float")
  skip_native_float32_on_windows()
  skip_if_not(has_metal(), "Metal backend not available")

  set.seed(126)
  A_mat <- matrix(rnorm(80), nrow = 10)
  A <- float::fl(A_mat)
  out <- fastPLS:::metal_float32_irlba_cpp(A, k = 3L, seed = 46L)
  U <- fastPLS:::.float32_from_bits(out$u)
  V <- fastPLS:::.float32_from_bits(out$v)

  expect_true(inherits(U, "float32"))
  expect_true(inherits(V, "float32"))
  expect_equal(dim(U), c(10L, 3L))
  expect_equal(dim(V), c(8L, 3L))
  expect_equal(length(out$d), 3L)
  expect_true(all(is.finite(out$d)))
  expect_true(all(diff(out$d) <= 1e-5))

  exact <- svd(A_mat, nu = 3, nv = 3)$d[1:3]
  expect_equal(as.numeric(out$d), exact, tolerance = 2e-1)
})

test_that("fastsvd supports public float32 CPU routes", {
  skip_if_not_installed("float")
  skip_native_float32_on_windows()
  set.seed(127)
  A <- float::fl(matrix(rnorm(72), nrow = 12))

  out_rsvd <- fastsvd(A, ncomp = 3, backend = "cpu", method = "rsvd", seed = 1)
  expect_true(inherits(out_rsvd$u, "float32"))
  expect_true(inherits(out_rsvd$v, "float32"))
  expect_identical(out_rsvd$precision, "float32")
  expect_equal(dim(out_rsvd$u), c(12L, 3L))
  expect_equal(dim(out_rsvd$v), c(6L, 3L))

  out_irlba <- fastsvd(A, ncomp = 3, backend = "cpu", method = "irlba", seed = 1)
  expect_true(inherits(out_irlba$u, "float32"))
  expect_true(inherits(out_irlba$v, "float32"))
  expect_identical(out_irlba$precision, "float32")
  expect_true(all(is.finite(out_irlba$d)))
})

test_that("fastsvd supports public float32 Metal rSVD route", {
  skip_if_not_installed("float")
  skip_native_float32_on_windows()
  skip_if_not(has_metal(), "Metal backend not available")
  set.seed(128)
  A <- float::fl(matrix(rnorm(72), nrow = 12))
  out <- fastsvd(A, ncomp = 3, backend = "metal", method = "rsvd", seed = 1)
  expect_true(inherits(out$u, "float32"))
  expect_true(inherits(out$v, "float32"))
  expect_identical(out$precision, "float32")
  expect_equal(dim(out$u), c(12L, 3L))
  expect_equal(dim(out$v), c(6L, 3L))

  out_irlba <- fastsvd(A, ncomp = 3, backend = "metal", method = "irlba", seed = 1)
  expect_true(inherits(out_irlba$u, "float32"))
  expect_true(inherits(out_irlba$v, "float32"))
  expect_identical(out_irlba$precision, "float32")
  expect_true(all(is.finite(out_irlba$d)))
})

test_that("pls supports float32 Metal backend when available", {
  skip_if_not_installed("float")
  skip_native_float32_on_windows()
  skip_if_not(has_metal(), "Metal backend not available")

  set.seed(130)
  X <- float::fl(as.matrix(mtcars[, c("disp", "hp", "wt", "qsec", "drat")]))
  y <- float::fl(matrix(mtcars$mpg, ncol = 1))
  fit <- pls(
    X,
    y,
    X[1:6, ],
    y[1:6, ],
    ncomp = 1:2,
    method = "simpls",
    backend = "metal",
    svd.method = "rsvd",
    return_variance = FALSE
  )
  expect_s3_class(fit, "fastPLS")
  expect_equal(attr(fit, "fastPLS_internal")$precision, "float32")
  expect_equal(attr(fit, "fastPLS_internal")$predict_backend, "float32_metal")
  expect_true(inherits(fit$Ypred[[1L]], "float32"))
  expect_named(fit$Q2Y, c("ncomp=1", "ncomp=2"))
})

test_that("pls supports the float32 LDA classifier", {
  skip_if_not_installed("float")
  skip_native_float32_on_windows()
  set.seed(131)
  idx <- sample(seq_len(nrow(iris)), 30)
  Xtrain <- float::fl(as.matrix(iris[-idx, 1:4]))
  Xtest <- float::fl(as.matrix(iris[idx, 1:4]))
  ytrain <- iris$Species[-idx]
  ytest <- iris$Species[idx]

  fit_lda <- pls(
    Xtrain,
    ytrain,
    Xtest,
    ytest,
    ncomp = 2:3,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    classifier = "lda",
    return_variance = FALSE
  )
  expect_s3_class(fit_lda, "fastPLS")
  expect_equal(attr(fit_lda, "fastPLS_internal")$precision, "float32")
  expect_equal(attr(fit_lda, "fastPLS_internal")$classification_rule, "lda_cpp")
  expect_true(all(vapply(fit_lda$Ypred, is.factor, logical(1))))
  expect_named(fit_lda$accuracy, c("ncomp=2", "ncomp=3"))
  expect_true(inherits(fit_lda$Ttrain, "float32"))
  expect_equal(dim(fit_lda$Ttrain), c(nrow(Xtrain), 3L))

  Xscaled <- sweep(float::dbl(Xtrain), 2L, as.numeric(float::dbl(fit_lda$mX)), "-")
  Xscaled <- sweep(Xscaled, 2L, as.numeric(float::dbl(fit_lda$vX)), "/")
  expected_scores <- Xscaled %*% float::dbl(fit_lda$R)
  expect_equal(
    unname(float::dbl(fit_lda$Ttrain)),
    unname(expected_scores),
    tolerance = 2e-4
  )

  pred_lda <- predict(fit_lda, Xtest, ytest, top = 2)
  expect_true("Ypred_top" %in% names(pred_lda))
  expect_named(pred_lda$accuracy, c("ncomp=2", "ncomp=3"))

})

test_that("float32 OPLS supports regression, classification, and independent prediction", {
  skip_if_not_installed("float")
  skip_native_float32_on_windows()
  set.seed(132)
  train <- sample(seq_len(nrow(iris)), 110)
  Xtrain <- as.matrix(iris[train, 1:4])
  Xtest <- as.matrix(iris[-train, 1:4])
  ytrain <- droplevels(iris$Species[train])
  ytest <- factor(iris$Species[-train], levels = levels(ytrain))

  fit32 <- pls(
    float::fl(Xtrain), ytrain, float::fl(Xtest), ytest,
    ncomp = 2, method = "opls", north = 1, backend = "cpu",
    svd.method = "rsvd", classifier = "lda", return_variance = FALSE,
    seed = 12
  )
  fit64 <- pls(
    Xtrain, ytrain, Xtest, ytest,
    ncomp = 2, method = "opls", north = 1, backend = "cpu",
    svd.method = "rsvd", classifier = "lda", return_variance = FALSE,
    seed = 12
  )

  expect_s3_class(fit32, "fastPLSOpls")
  expect_identical(attr(fit32, "fastPLS_internal")$precision, "float32")
  expect_true(inherits(fit32$mX, "float32"))
  expect_true(inherits(fit32$W_orth, "float32"))
  expect_equal(fit32$accuracy, fit64$accuracy, tolerance = 0.05)
  pred <- predict(fit32, float::fl(Xtest[1:5, , drop = FALSE]))
  expect_true(is.factor(pred$Ypred[[1L]]))

  Xreg <- as.matrix(mtcars[, c("disp", "hp", "wt", "qsec")])
  Yreg <- matrix(mtcars$mpg, ncol = 1L)
  reg32 <- pls(
    float::fl(Xreg[1:24, , drop = FALSE]),
    float::fl(Yreg[1:24, , drop = FALSE]),
    float::fl(Xreg[25:32, , drop = FALSE]),
    float::fl(Yreg[25:32, , drop = FALSE]),
    ncomp = 1:2, method = "opls", backend = "cpu",
    svd.method = "rsvd", fit = TRUE, return_variance = FALSE, seed = 13
  )
  expect_true(all(is.finite(reg32$Q2Y)))
  expect_true(all(vapply(reg32$Ypred, inherits, logical(1L), "float32")))
})

test_that("float32 kernel PLS-LDA supports linear, RBF, and polynomial kernels", {
  skip_if_not_installed("float")
  skip_native_float32_on_windows()
  set.seed(133)
  train <- sample(seq_len(nrow(iris)), 110)
  Xtrain <- as.matrix(iris[train, 1:4])
  Xtest <- as.matrix(iris[-train, 1:4])
  ytrain <- droplevels(iris$Species[train])
  ytest <- factor(iris$Species[-train], levels = levels(ytrain))

  for (kernel in c("linear", "rbf", "poly")) {
    fit32 <- pls(
      float::fl(Xtrain), ytrain, float::fl(Xtest), ytest,
      ncomp = 2, method = "kernelpls", kernel = kernel, backend = "cpu",
      svd.method = "rsvd", classifier = "lda", return_variance = FALSE,
      seed = 14
    )
    fit64 <- pls(
      Xtrain, ytrain, Xtest, ytest,
      ncomp = 2, method = "kernelpls", kernel = kernel, backend = "cpu",
      svd.method = "rsvd", classifier = "lda", return_variance = FALSE,
      seed = 14
    )
    expect_identical(attr(fit32, "fastPLS_internal")$precision, "float32")
    expect_equal(fit32$accuracy, fit64$accuracy, tolerance = 0.05)
    pred <- predict(fit32, float::fl(Xtest[1:5, , drop = FALSE]))
    expect_true(is.factor(pred$Ypred[[1L]]))
    if (!identical(kernel, "linear")) {
      expect_s3_class(fit32, "fastPLSKernel")
      expect_true(inherits(fit32$Xref, "float32"))
      expect_true(inherits(fit32$kernel_center$col_means, "float32"))
    }
  }
})

test_that("float32 OPLS and kernel PLS use Metal when available", {
  skip_if_not_installed("float")
  skip_native_float32_on_windows()
  skip_if_not(has_metal(), "Metal backend not available")
  X <- float::fl(as.matrix(iris[, 1:4]))
  y <- iris$Species

  opls_fit <- pls(
    X, y, X[1:12, ], y[1:12], ncomp = 2, method = "opls",
    backend = "metal", svd.method = "rsvd", classifier = "lda",
    return_variance = FALSE,
    seed = 15
  )
  kernel_fit <- pls(
    X, y, X[1:12, ], y[1:12], ncomp = 2, method = "kernelpls",
    kernel = "rbf", backend = "metal", svd.method = "rsvd",
    classifier = "lda", return_variance = FALSE, seed = 15
  )
  expect_identical(attr(opls_fit, "fastPLS_internal")$precision, "float32")
  expect_identical(attr(kernel_fit, "fastPLS_internal")$precision, "float32")
  expect_match(opls_fit$opls_engine, "metal")
  expect_match(kernel_fit$kernel_engine, "metal")
  expect_true(is.factor(opls_fit$Ypred[[1L]]))
  expect_true(is.factor(kernel_fit$Ypred[[1L]]))
})

test_that("float32 OPLS and kernel PLS use CUDA when available", {
  skip_if_not_installed("float")
  skip_native_float32_on_windows()
  skip_if_not(has_cuda(), "CUDA backend not available")
  X <- float::fl(as.matrix(iris[, 1:4]))
  y <- iris$Species

  for (method in c("opls", "kernelpls")) {
    fit <- pls(
      X, y, X[1:12, ], y[1:12], ncomp = 2, method = method,
      kernel = "rbf", backend = "cuda", svd.method = "rsvd",
      classifier = "lda", return_variance = FALSE, seed = 16
    )
    expect_identical(attr(fit, "fastPLS_internal")$precision, "float32")
    expect_true(is.factor(fit$Ypred[[1L]]))
  }
})
