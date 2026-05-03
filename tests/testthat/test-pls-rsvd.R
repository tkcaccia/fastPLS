library(fastPLS)

test_that("default svd.method behavior remains unchanged", {
  set.seed(42)
  X <- matrix(rnorm(70 * 18), nrow = 70, ncol = 18)
  Y <- matrix(rnorm(70 * 4), nrow = 70, ncol = 4)

  m_default <- pls(X, Y, ncomp = 1:3, fit = TRUE)
  m_explicit <- pls(X, Y, ncomp = 1:3, fit = TRUE, svd.method = "irlba")

  align_signs <- function(ref, x) {
    out <- x
    for (j in seq_len(min(ncol(ref), ncol(out)))) {
      s <- sum(ref[, j] * out[, j], na.rm = TRUE)
      if (is.finite(s) && s < 0) {
        out[, j] <- -out[, j]
      }
    }
    out
  }

  expect_equal(m_default$B, m_explicit$B)
  expect_equal(align_signs(m_default$R, m_explicit$R), m_default$R)
  expect_equal(align_signs(m_default$Q, m_explicit$Q), m_default$Q)
})

test_that("cpu_rsvd tracks irlba on PLS outputs", {
  set.seed(99)
  X <- matrix(rnorm(80 * 24), nrow = 80, ncol = 24)
  Y <- matrix(rnorm(80 * 6), nrow = 80, ncol = 6)

  exact <- pls(
    X,
    Y,
    ncomp = 1:4,
    fit = TRUE,
    svd.method = "irlba"
  )

  rsvd <- pls(
    X,
    Y,
    ncomp = 1:4,
    fit = TRUE,
    svd.method = "cpu_rsvd",
    rsvd_oversample = 12L,
    rsvd_power = 2L,
    seed = 123L
  )

  expect_equal(dim(exact$B), dim(rsvd$B))
  expect_equal(dim(exact$R), dim(rsvd$R))
  expect_equal(rsvd$B, exact$B, tolerance = 5e-2)
  expect_true(all(is.finite(rsvd$R)))
  expect_true(all(is.finite(rsvd$Q)))
})

test_that("cpu_rsvd is deterministic with a fixed seed", {
  set.seed(7)
  X <- matrix(rnorm(90 * 25), nrow = 90, ncol = 25)
  Y <- matrix(rnorm(90 * 12), nrow = 90, ncol = 12)

  fit1 <- pls(
    X,
    Y,
    ncomp = 1:5,
    fit = TRUE,
    svd.method = "cpu_rsvd",
    rsvd_oversample = 5L,
    rsvd_power = 1L,
    seed = 777L
  )

  fit2 <- pls(
    X,
    Y,
    ncomp = 1:5,
    fit = TRUE,
    svd.method = "cpu_rsvd",
    rsvd_oversample = 5L,
    rsvd_power = 1L,
    seed = 777L
  )

  fit3 <- pls(
    X,
    Y,
    ncomp = 1:5,
    fit = TRUE,
    svd.method = "cpu_rsvd",
    rsvd_oversample = 5L,
    rsvd_power = 1L,
    seed = 778L
  )

  expect_equal(fit1$B, fit2$B)
  expect_false(isTRUE(all.equal(fit1$B, fit3$B)))
})

test_that("IRLBA xprod default does not trigger for medium-n synthetic reg_q shape", {
  should_use <- get(".should_use_xprod_irlba_default", envir = asNamespace("fastPLS"))

  expect_false(should_use(n = 5000, p = 1000, q = 101, ncomp = 50))
  expect_false(should_use(n = 5000, p = 1000, q = 1000, ncomp = 50))
  expect_false(should_use(n = 5000, p = 1000, q = 10000, ncomp = 50))
  expect_true(should_use(n = 10000, p = 1000, q = 5000, ncomp = 50))
})

test_that("xprod default threshold matches the benchmark rule", {
  should_use_rsvd <- get(".should_use_xprod_default", envir = asNamespace("fastPLS"))
  should_use_irlba <- get(".should_use_xprod_irlba_default", envir = asNamespace("fastPLS"))

  # singlecell-like shape: q is large, but ncomp is not small and X'Y is tiny.
  expect_false(should_use_rsvd(p = 50, q = 133, ncomp = 50))
  expect_false(should_use_irlba(n = 23822, p = 50, q = 133, ncomp = 50))

  # CIFAR-like classification uses xprod for rSVD only at small component counts.
  expect_true(should_use_rsvd(p = 2048, q = 100, ncomp = 10))
  expect_false(should_use_rsvd(p = 2048, q = 100, ncomp = 20))
  expect_false(should_use_irlba(n = 50000, p = 2048, q = 100, ncomp = 10))

  # Large cross-response products use xprod for rSVD, and only large enough
  # n/min(p,q) cases use the IRLBA operator path.
  expect_true(should_use_rsvd(p = 5000, q = 1000, ncomp = 50))
  expect_false(should_use_irlba(n = 5000, p = 5000, q = 1000, ncomp = 50))
  expect_true(should_use_irlba(n = 10000, p = 5000, q = 1000, ncomp = 50))
})

test_that("CPU FlashSVD prediction is the default for compiled and R PLS", {
  set.seed(17)
  X <- matrix(rnorm(70 * 20), nrow = 70, ncol = 20)
  Y <- matrix(rnorm(70 * 5), nrow = 70, ncol = 5)
  idx <- 1:12

  for (impl in c("cpp", "r")) {
    for (method in c("plssvd", "simpls")) {
      ref <- pls(
        X[-idx, ],
        Y[-idx, ],
        ncomp = 1:4,
        method = method,
        backend = impl,
        svd.method = "cpu_rsvd",
        rsvd_oversample = 8L,
        rsvd_power = 1L,
        seed = 17L
      )
      flash <- pls(
        X[-idx, ],
        Y[-idx, ],
        ncomp = 1:4,
        method = method,
        backend = impl,
        svd.method = "cpu_rsvd",
        rsvd_oversample = 8L,
        rsvd_power = 1L,
        seed = 17L
      )
      pred_ref <- predict(ref, X[idx, , drop = FALSE], predict.backend = "cpu")
      pred_flash <- predict(flash, X[idx, , drop = FALSE])

      expect_s3_class(flash, "fastPLS")
      expect_true(isTRUE(flash$flash_svd))
      expect_identical(flash$flash_svd_backend, "cpu")
      expect_identical(flash$predict_backend, "cpu_flash")
      expect_identical(flash$flash_svd_mode, "streamed_low_rank_prediction")
      expect_equal(flash$B, ref$B)
      expect_equal(pred_flash$Ypred, pred_ref$Ypred, tolerance = 1e-10)
    }
  }
})

test_that("has_cuda returns scalar logical and hybrid cuda path is removed from pls", {
  flag <- has_cuda()
  expect_type(flag, "logical")
  expect_length(flag, 1L)

  set.seed(1)
  X <- matrix(rnorm(40 * 10), nrow = 40, ncol = 10)
  Y <- matrix(rnorm(40 * 3), nrow = 40, ncol = 3)
  expect_error(pls(X, Y, ncomp = 1:2, svd.method = "cuda_rsvd"), "removed")
})

test_that("simpls path uses the SVD backend selector", {
  set.seed(1234)
  X <- matrix(rnorm(100 * 25), nrow = 100, ncol = 25)
  Y <- matrix(rnorm(100 * 8), nrow = 100, ncol = 8)

  exact <- pls(
    X,
    Y,
    ncomp = 1:5,
    fit = TRUE,
    method = "simpls",
    svd.method = "irlba"
  )

  rsvd <- pls(
    X,
    Y,
    ncomp = 1:5,
    fit = TRUE,
    method = "simpls",
    svd.method = "cpu_rsvd",
    rsvd_oversample = 10L,
    rsvd_power = 1L,
    seed = 99L
  )

  expect_equal(dim(exact$B), dim(rsvd$B))
  expect_equal(rsvd$B, exact$B, tolerance = 7e-2)
  expect_true(all(is.finite(rsvd$R)))
  expect_true(all(is.finite(rsvd$Q)))

  expect_error(
    pls(
      X,
      Y,
      ncomp = 1:5,
      fit = TRUE,
      method = "simpls",
      svd.method = "cuda_rsvd",
      rsvd_oversample = 10L,
      rsvd_power = 1L,
      seed = 99L
    ),
    "removed"
  )
})

test_that("Rcpp plssvd handles ncomp above rank by capping internally", {
  set.seed(78)
  X <- matrix(rnorm(180 * 45), nrow = 180, ncol = 45)
  y <- factor(sample(letters[1:10], 180, replace = TRUE))
  idx <- sample(seq_len(180), 40)

  expect_warning({
    fit <- pls(X[-idx, ], y[-idx], X[idx, ], ncomp = 60, method = "plssvd", svd.method = "cpu_rsvd")
    expect_s3_class(fit, "fastPLS")
    expect_true(is.data.frame(fit$Ypred))
  }, "rank is limited")
})
