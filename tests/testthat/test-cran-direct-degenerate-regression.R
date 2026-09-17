test_that("direct SIMPLS fits support zero effective directions", {
    set.seed(1L)
    x <- matrix(rnorm(41L * 768L), nrow = 41L, ncol = 768L)

    for (value in c(0, 3)) {
        fit <- pls(
            x,
            rep(value, nrow(x)),
            ncomp = 1:10,
            fit = TRUE,
            return_loadings = TRUE,
            backend = "cpu",
            seed = 20261542L
        )
        prediction <- predict(fit, x[seq_len(5L), , drop = FALSE])

        expect_identical(
            attr(fit, "fastPLS_internal")$ncomp,
            1:10
        )
        expect_identical(fit$effective_ncomp, rep(0L, 10L))
        expect_identical(
            fit$diagnostics$requested_component_path,
            1:10
        )
        expect_identical(
            fit$diagnostics$effective_component_path,
            rep(0L, 10L)
        )
        expect_identical(fit$diagnostics$requested_components, 10L)
        expect_identical(fit$diagnostics$effective_components, 0L)
        expect_true(all(is.na(fit$R2Y)))
        expect_equal(dim(fit$B), c(ncol(x), 1L, 10L))
        expect_true(all(fit$B == 0))
        expect_equal(dim(fit$P), c(ncol(x), 0L))
        expect_equal(dim(fit$Ttrain), c(nrow(x), 0L))
        expect_equal(dim(fit$Yfit), c(nrow(x), 1L, 10L))
        expect_true(all(fit$Yfit == value))
        expect_equal(dim(prediction$Ypred), c(5L, 1L, 10L))
        expect_true(all(is.finite(prediction$Ypred)))
        expect_true(all(prediction$Ypred == value))
    }
})

test_that("constant-response direct fits are deterministic", {
    set.seed(2L)
    x <- matrix(rnorm(36L * 9L), nrow = 36L, ncol = 9L)
    arguments <- list(
        Xtrain = x,
        Ytrain = rep(2.5, nrow(x)),
        ncomp = 1:4,
        scaling = "autoscaling",
        fit = TRUE,
        backend = "cpu",
        seed = 117L
    )

    first <- do.call(pls, arguments)
    second <- do.call(pls, arguments)

    expect_identical(first$effective_ncomp, second$effective_ncomp)
    expect_identical(first$Yfit, second$Yfit)
    expect_identical(first$B, second$B)
})

test_that("unavailable higher directions repeat the last estimable path", {
    signal <- rep(c(-1, 1), 20L)
    x <- cbind(signal, 0)
    fit <- pls(
        x,
        signal,
        ncomp = 1:2,
        scaling = "none",
        fit = TRUE,
        backend = "cpu",
        seed = 1L
    )
    prediction <- predict(fit, x[seq_len(6L), , drop = FALSE])

    expect_identical(fit$effective_ncomp, c(1L, 1L))
    expect_equal(fit$B[, , 2L], fit$B[, , 1L], tolerance = 0)
    expect_equal(fit$Yfit[, , 2L], fit$Yfit[, , 1L], tolerance = 0)
    expect_equal(
        prediction$Ypred[, , 2L],
        prediction$Ypred[, , 1L],
        tolerance = 0
    )
})
