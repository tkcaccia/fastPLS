skip_native_float32_on_windows <- function() {
  skip_if(
    .Platform$OS.type == "windows",
    "native float32 kernels are not available on Windows"
  )
}

test_that("pls accepts float32 regression input without upcasting predictions", {
  skip_if_not_installed("float")
  skip_native_float32_on_windows()
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
  expect_equal(attr(fit, "fastPLS_internal")$predict_backend, "float32_cpp")
  expect_true(inherits(fit$Ypred[[1L]], "float32"))
  expect_named(fit$Q2Y, c("ncomp=1", "ncomp=2"))
  expect_false("predict_backend" %in% names(predict(fit, X[1:2, ])))
})

test_that("float32 detection handles S4 float matrices used in the vignette", {
  skip_if_not_installed("float")
  skip_native_float32_on_windows()
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
  expect_equal(attr(fit_reg32, "fastPLS_internal")$predict_backend, "float32_cpp")
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
  skip_native_float32_on_windows()
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
  expect_equal(attr(fit, "fastPLS_internal")$predict_backend, "float32_cpp")
  expect_true(is.factor(fit$Ypred[[1L]]))
  expect_named(fit$accuracy, "ncomp=2")
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

test_that("CUDA float32 rSVD sketch matches CPU float32 arithmetic", {
  skip_if_not_installed("float")
  skip_native_float32_on_windows()
  skip_if_not(has_cuda(), "CUDA backend not available")

  set.seed(123)
  A <- float::fl(matrix(rnorm(30), nrow = 6))
  out <- cuda_float32_rsvd_sample_cpp(A, l = 3L, power_iters = 1L, seed = 44L)
  Y_cuda <- .float32_from_bits(out$Y)
  Omega <- .float32_from_bits(out$Omega)
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
  out <- metal_float32_matrix_multiply_cpp(A, B)
  C_metal <- .float32_from_bits(out$C)
  C_cpu <- A %*% B

  expect_true(inherits(C_metal, "float32"))
  expect_equal(dim(C_metal), dim(C_cpu))
  expect_equal(as.numeric(C_metal), as.numeric(C_cpu), tolerance = 1e-4)

  D <- float::fl(matrix(rnorm(18), nrow = 6))
  out_t <- metal_float32_matrix_multiply_cpp(A, D, transpose_left = TRUE)
  C_t_metal <- .float32_from_bits(out_t$C)
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
  out <- metal_float32_rsvd_sample_cpp(A, l = 3L, power_iters = 1L, seed = 45L)
  Y_metal <- .float32_from_bits(out$Y)
  Omega <- .float32_from_bits(out$Omega)
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
  out <- metal_float32_irlba_cpp(A, k = 3L, seed = 46L)
  U <- .float32_from_bits(out$u)
  V <- .float32_from_bits(out$v)

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

test_that("pca and predict preserve float32 matrices", {
  skip_if_not_installed("float")
  skip_native_float32_on_windows()
  set.seed(129)
  X <- float::fl(as.matrix(iris[, 1:4]))
  pc <- pca(X, ncomp = 2, backend = "cpu", method = "rsvd", seed = 1)
  expect_s3_class(pc, "fastPLSPCA")
  expect_identical(pc$precision, "float32")
  expect_true(inherits(pc$scores, "float32"))
  expect_true(inherits(pc$loadings, "float32"))

  projected <- predict(pc, float::fl(as.matrix(iris[1:5, 1:4])))
  expect_true(inherits(projected, "float32"))
  expect_equal(dim(projected), c(5L, 2L))
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

test_that("pls supports float32 LDA and cKNN classifiers", {
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

  pred_lda <- predict(fit_lda, Xtest, ytest, top = 2)
  expect_true("Ypred_top" %in% names(pred_lda))
  expect_named(pred_lda$accuracy, c("ncomp=2", "ncomp=3"))

  fit_cknn <- pls(
    Xtrain,
    ytrain,
    Xtest,
    ytest,
    ncomp = 2,
    method = "plssvd",
    backend = "cpu",
    svd.method = "irlba",
    classifier = "cknn",
    k = 3,
    tau = 0.2,
    alpha = 0.75,
    return_variance = FALSE
  )
  expect_s3_class(fit_cknn, "fastPLS")
  expect_equal(attr(fit_cknn, "fastPLS_internal")$precision, "float32")
  expect_equal(attr(fit_cknn, "fastPLS_internal")$classification_rule, "candidate_knn_cpp")
  expect_true(is.factor(fit_cknn$Ypred[[1L]]))
  expect_named(fit_cknn$accuracy, "ncomp=2")
})
