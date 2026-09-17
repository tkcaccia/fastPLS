degenerate_regression_data <- function() {
    set.seed(10L)
    list(
        x = matrix(rnorm(54L * 768L), nrow = 54L, ncol = 768L),
        y = c(rep(0, 52L), 0.00549885308390477, 0.0599934866563745)
    )
}

run_degenerate_nested_cv <- function(x, y, ncomp, scaling) {
    pls.double.cv(
        x,
        y,
        ncomp = ncomp,
        kfold_outer = 5L,
        kfold_inner = 5L,
        scaling = scaling,
        backend = "cpu",
        seed = 20260852L,
        perm.test = FALSE
    )
}

test_that("direct constant-response paths cover scaling and matrix shape", {
    configurations <- list(
        wide = c(samples = 41L, predictors = 768L),
        tall = c(samples = 80L, predictors = 12L)
    )
    for (configuration in configurations) {
        set.seed(sum(configuration))
        x <- matrix(
            rnorm(configuration[["samples"]] *
                configuration[["predictors"]]),
            nrow = configuration[["samples"]],
            ncol = configuration[["predictors"]]
        )
        for (value in c(0, 4.25)) {
            for (scaling in c("centering", "autoscaling", "none")) {
                for (components in list(1L, 1:10)) {
                    fit <- pls(
                        x,
                        rep(value, nrow(x)),
                        ncomp = components,
                        scaling = scaling,
                        fit = TRUE,
                        return_loadings = TRUE,
                        backend = "cpu",
                        seed = 20261542L
                    )
                    prediction <- predict(fit, x[seq_len(7L), , drop = FALSE])

                    expect_identical(
                        fit$effective_ncomp,
                        rep(0L, length(components))
                    )
                    expect_true(all(is.finite(prediction$Ypred)))
                    expect_true(all(prediction$Ypred == value))
                    expect_true(all(fit$B == 0))
                    expect_true(all(is.na(fit$R2Y)))
                    expect_equal(
                        dim(fit$Yfit),
                        c(nrow(x), 1L, length(components))
                    )
                }
            }
        }
    }
})

test_that("nested CV retains every prefix when folds lose effective rank", {
    data <- degenerate_regression_data()

    for (scaling in c("centering", "autoscaling", "none")) {
        for (ncomp in list(1L, 1:10)) {
            result <- run_degenerate_nested_cv(
                data$x, data$y, ncomp, scaling
            )
            expect_length(result$Ypred, nrow(data$x))
            expect_true(all(is.finite(result$Ypred)))
            expect_length(result$Q2Y, 1L)
            expect_length(result$RMSD, 1L)
            expect_true(all(result$results[[1L]]$best_ncomp %in% ncomp))
            expect_length(result$results[[1L]]$inner, 5L)
            for (fold in result$results[[1L]]$inner) {
                expect_identical(fold$ncomp, as.integer(ncomp))
                expect_length(fold$metric_value, length(ncomp))
            }
        }
    }
})

test_that("degenerate nested CV is deterministic", {
    data <- degenerate_regression_data()
    first <- run_degenerate_nested_cv(
        data$x, data$y, 1:10, "centering"
    )
    second <- run_degenerate_nested_cv(
        data$x, data$y, 1:10, "centering"
    )

    expect_identical(first$Ypred, second$Ypred)
    expect_identical(first$Q2Y, second$Q2Y)
    expect_identical(first$RMSD, second$RMSD)
    expect_identical(
        first$results[[1L]]$best_ncomp,
        second$results[[1L]]$best_ncomp
    )
})

test_that("a completely constant response uses fold training means", {
    data <- degenerate_regression_data()
    single <- pls.single.cv(
        data$x,
        rep(0, nrow(data$x)),
        ncomp = 1:10,
        kfold = 5L,
        method = "simpls",
        backend = "cpu",
        seed = 20260852L,
        fit = FALSE
    )
    result <- run_degenerate_nested_cv(
        data$x, rep(0, nrow(data$x)), 1:10, "centering"
    )

    expect_equal(dim(single$Ypred), c(nrow(data$x), 1L, 10L))
    expect_true(all(is.finite(single$Ypred)))
    for (prefix in seq_len(dim(single$Ypred)[[3L]])) {
        expect_equal(
            single$Ypred[, , prefix],
            rep(0, nrow(data$x)),
            tolerance = 0
        )
    }
    expect_length(result$Ypred, nrow(data$x))
    expect_true(all(is.finite(result$Ypred)))
    expect_equal(
        result$Ypred,
        matrix(0, nrow = nrow(data$x), ncol = 1L),
        tolerance = 0
    )
})
