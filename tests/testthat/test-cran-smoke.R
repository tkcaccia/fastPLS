small_regression_data <- function(seed = 4101L) {
    set.seed(seed)
    x <- matrix(rnorm(48L * 8L), nrow = 48L)
    y <- cbind(
        0.8 * x[, 1L] - 0.3 * x[, 2L] + rnorm(48L, sd = 0.1),
        -0.5 * x[, 3L] + 0.4 * x[, 4L] + rnorm(48L, sd = 0.1)
    )
    list(x = x, y = y)
}

small_classification_data <- function(seed = 4102L) {
    set.seed(seed)
    x <- matrix(rnorm(60L * 7L), nrow = 60L)
    score <- x[, 1L] - 0.5 * x[, 2L] + 0.25 * x[, 3L]
    y <- factor(cut(
        score,
        breaks = c(-Inf, stats::quantile(score, c(1 / 3, 2 / 3)), Inf),
        labels = c("low", "middle", "high")
    ))
    list(x = x, y = y)
}

expect_finite_numeric <- function(value) {
    expect_true(is.numeric(value))
    expect_true(length(value) > 0L)
    expect_true(all(is.finite(value)))
}

test_that("the compiled numerical library and available details are reported", {
    backend <- fastPLS_blas(details = FALSE)
    information <- fastPLS_blas()

    expect_true(backend %in% c("Accelerate", "OpenBLAS", "R BLAS/LAPACK"))
    expect_identical(information$backend, backend)
    expect_named(
        information,
        c(
            "backend", "version", "configuration", "core", "parallel",
            "threads", "library"
        )
    )
    if (identical(backend, "OpenBLAS")) {
        expect_match(information$version, "^[0-9]+[.][0-9]+[.][0-9]+")
        expect_true(nzchar(information$core))
    }
})

test_that("all PLS families fit and predict small regression tasks", {
    data <- small_regression_data()
    train <- seq_len(38L)
    test <- setdiff(seq_len(nrow(data$x)), train)
    specifications <- list(
        plssvd = list(),
        simpls = list(),
        opls = list(north = 1L),
        kernelpls = list(kernel = "rbf", gamma = 0.2)
    )

    for (method in names(specifications)) {
        arguments <- c(
            list(
                Xtrain = data$x[train, , drop = FALSE],
                Ytrain = data$y[train, , drop = FALSE],
                ncomp = 2L,
                method = method,
                backend = "cpu",
                return_variance = FALSE
            ),
            specifications[[method]]
        )
        fit <- do.call(pls, arguments)
        prediction <- predict(
            fit,
            data$x[test, , drop = FALSE],
            Ytest = data$y[test, , drop = FALSE]
        )

        expect_s3_class(fit, paste0(
            "fastPLS",
            if (method == "opls") "Opls" else if (method == "kernelpls") {
                "Kernel"
            } else {
                ""
            }
        ))
        expect_equal(dim(prediction$Ypred), c(length(test), 2L, 1L))
        expect_finite_numeric(prediction$Ypred)
        expect_identical(prediction$metrics$task, "regression")
    }
})

test_that("argmax and LDA classification produce valid labels", {
    data <- small_classification_data()
    train <- seq_len(48L)
    test <- setdiff(seq_len(nrow(data$x)), train)

    for (classifier in c("argmax", "lda")) {
        fit <- pls(
            data$x[train, , drop = FALSE],
            data$y[train],
            ncomp = 2L,
            method = "simpls",
            classifier = classifier,
            backend = "cpu",
            return_variance = FALSE
        )
        prediction <- predict(
            fit,
            data$x[test, , drop = FALSE],
            Ytest = data$y[test]
        )

        expect_s3_class(fit, "fastPLS")
        expect_true(is.data.frame(prediction$Ypred))
        expect_equal(nrow(prediction$Ypred), length(test))
        expect_setequal(levels(prediction$Ypred[[1L]]), levels(data$y))
        expect_true(all(prediction$Ypred[[1L]] %in% levels(data$y)))
        expect_true(is.finite(prediction$metrics$metrics$accuracy[[1L]]))
    }
})

test_that("float32 input follows the public fitting and prediction path", {
    data <- small_regression_data(4103L)
    x <- float::fl(data$x)
    y <- float::fl(data$y)
    fit_call <- function() {
        pls(
            x,
            y,
            ncomp = 2L,
            method = "plssvd",
            backend = "cpu",
            return_variance = FALSE
        )
    }
    if (.Platform$OS.type == "windows") {
        expect_warning(
            fit <- fit_call(),
            "Windows uses portable float-package CPU routes"
        )
    } else {
        fit <- fit_call()
    }
    prediction <- predict(fit, x)
    predicted <- float::dbl(prediction$Ypred[[1L]])

    expect_s3_class(fit, "fastPLS")
    expect_length(prediction$Ypred, 1L)
    expect_equal(dim(predicted), c(nrow(data$x), ncol(data$y)))
    expect_finite_numeric(predicted)
})

test_that("single and nested cross-validation return finite predictions", {
    data <- small_classification_data(4104L)
    groups <- rep(seq_len(20L), each = 3L)

    single <- pls.single.cv(
        data$x,
        data$y,
        ncomp = 1:2,
        constrain = groups,
        kfold = 2L,
        method = "simpls",
        classifier = "lda",
        selection = "balanced_accuracy",
        backend = "cpu",
        seed = 19L
    )
    nested <- pls.double.cv(
        data$x,
        data$y,
        ncomp = 1:2,
        constrain = groups,
        runn = 1L,
        kfold_inner = 2L,
        kfold_outer = 2L,
        method = "simpls",
        classifier = "lda",
        selection = "balanced_accuracy",
        backend = "cpu",
        seed = 19L
    )

    expect_true(single$best_ncomp %in% 1:2)
    expect_true(is.finite(single$best_metric_value))
    expect_true(is.factor(nested$Ypred))
    expect_equal(length(nested$Ypred), nrow(data$x))
    expect_true(all(!is.na(nested$Ypred)))
})

test_that("unavailable accelerators fail without a CPU fallback", {
    data <- small_classification_data(4105L)
    unavailable <- c(
        if (!has_cuda()) "cuda",
        if (!has_metal()) "metal"
    )
    skip_if(length(unavailable) == 0L, "Both accelerators are available")

    for (backend in unavailable) {
        expect_error(
            pls(
                data$x,
                data$y,
                ncomp = 1L,
                method = "simpls",
                backend = backend,
                return_variance = FALSE
            ),
            regexp = "not available|unavailable|not compiled|requires",
            ignore.case = TRUE
        )
    }
})
