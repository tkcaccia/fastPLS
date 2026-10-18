test_that("binary LDA supports many correlated latent directions", {
    set.seed(19L)
    predictors <- matrix(rnorm(40L * 128L), nrow = 40L)
    response <- factor(rep(c(1L, 3L), 20L))
    for (single_precision in c(FALSE, TRUE)) {
        x <- if (single_precision) float::fl(predictors) else predictors
        for (scaling in c("none", "centering", "autoscaling")) {
            arguments <- list(
                Xtrain = x, Ytrain = response, ncomp = 24L,
                method = "simpls", classifier = "lda", scaling = scaling,
                backend = "cpu", seed = 314159L, fit = TRUE
            )
            first <- do.call(pls, arguments)
            second <- do.call(pls, arguments)
            new_data <- x[1:3, , drop = FALSE]
            predicted <- predict(first, new_data, raw_scores = TRUE)
            repeated <- predict(second, new_data, raw_scores = TRUE)
            expect_identical(as.integer(first$effective_ncomp), 24L)
            expect_identical(dim(predicted$LDA_scores), c(3L, 2L, 1L))
            expect_true(all(is.finite(predicted$LDA_scores)))
            expect_false(anyNA(predicted$Ypred))
            expect_equal(predicted$LDA_scores, repeated$LDA_scores,
                         tolerance = 0)
            expect_identical(predicted$Ypred, repeated$Ypred)
        }
    }
})
