test_that("bundled example data exercise classification and prediction", {
    data("fastpls_example", package = "fastPLS")

    expect_equal(dim(fastpls_example$X_train), c(90L, 40L))
    expect_equal(dim(fastpls_example$X_test), c(30L, 40L))
    expect_false(anyNA(fastpls_example$y_train))
    expect_false(anyNA(fastpls_example$y_test))

    fit <- pls(
        fastpls_example$X_train,
        fastpls_example$y_train,
        fastpls_example$X_test,
        fastpls_example$y_test,
        ncomp = 1:2,
        method = "simpls",
        backend = "cpu",
        seed = 42
    )

    expect_s3_class(fit, "fastPLS")
    expect_length(fit$Ypred, 2L)
    expect_true(all(is.finite(fit$accuracy)))

    predicted <- predict(fit, fastpls_example$X_test)
    expect_length(predicted$Ypred, 2L)
    expect_equal(length(predicted$Ypred[[1L]]), 30L)
})
