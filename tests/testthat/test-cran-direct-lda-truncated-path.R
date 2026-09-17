direct_lda_rank_example <- function(float32 = FALSE) {
    set.seed(42L)
    x <- matrix(rnorm(9L * 20L), nrow = 9L, ncol = 20L)
    if (isTRUE(float32)) {
        x <- float::fl(x)
    }
    list(
        x = x,
        y = factor(
            c(0L, 0L, 0L, 0L, 0L, 1L, 1L, 1L, 1L),
            levels = c(0L, 1L)
        )
    )
}

expect_complete_lda_path <- function(fit, prediction, requested, effective) {
    expect_identical(fit$requested_ncomp, requested)
    expect_identical(as.integer(fit$effective_ncomp), effective)
    expect_identical(
        fit$diagnostics$requested_component_path,
        requested
    )
    expect_identical(
        fit$diagnostics$effective_component_path,
        effective
    )
    expect_identical(
        dim(prediction$LDA_scores),
        c(3L, 2L, length(requested))
    )
    expect_identical(dim(prediction$Ypred), c(3L, length(requested)))
    expect_identical(dim(fit$Yfit), c(nrow(fit$Ttrain), length(requested)))
    if (!is.null(fit$B) && length(dim(fit$B)) == 3L) {
        expect_identical(dim(fit$B)[[3L]], length(requested))
    }
    expect_true(all(is.finite(prediction$LDA_scores)))
    expect_false(anyNA(prediction$Ypred))
}

test_that("direct LDA retains rank-limited requested component paths", {
    data <- direct_lda_rank_example()
    for (components in list(1L, 1:8, 1:10)) {
        fit <- suppressWarnings(pls(
            data$x,
            data$y,
            ncomp = components,
            classifier = "lda",
            fit = TRUE,
            backend = "cpu",
            seed = 77L
        ))
        prediction <- predict(
            fit,
            data$x[1:3, , drop = FALSE],
            raw_scores = TRUE
        )
        effective <- pmin(components, 8L)
        expect_complete_lda_path(
            fit, prediction, as.integer(components), effective
        )
        if (max(components) > 8L) {
            expect_equal(
                prediction$LDA_scores[, , 9L],
                prediction$LDA_scores[, , 8L],
                tolerance = 0
            )
            expect_equal(
                prediction$LDA_scores[, , 10L],
                prediction$LDA_scores[, , 8L],
                tolerance = 0
            )
            expect_identical(
                prediction$Ypred[, 9L],
                prediction$Ypred[, 8L]
            )
            expect_identical(
                prediction$Ypred[, 10L],
                prediction$Ypred[, 8L]
            )
            expect_identical(fit$Yfit[, 9L], fit$Yfit[, 8L])
            expect_identical(fit$Yfit[, 10L], fit$Yfit[, 8L])
            if (!is.null(fit$B) && length(dim(fit$B)) == 3L) {
                expect_equal(fit$B[, , 9L], fit$B[, , 8L], tolerance = 0)
                expect_equal(fit$B[, , 10L], fit$B[, , 8L], tolerance = 0)
            }
        }
    }
})

test_that("direct LDA capped paths are deterministic across preprocessing", {
    data <- direct_lda_rank_example()
    for (scaling in c("centering", "autoscaling", "none")) {
        arguments <- list(
            Xtrain = data$x,
            Ytrain = data$y,
            ncomp = 1:10,
            classifier = "lda",
            scaling = scaling,
            fit = TRUE,
            backend = "cpu",
            seed = 77L
        )
        first <- suppressWarnings(do.call(pls, arguments))
        second <- suppressWarnings(do.call(pls, arguments))
        first_prediction <- predict(
            first, data$x[1:3, , drop = FALSE], raw_scores = TRUE
        )
        second_prediction <- predict(
            second, data$x[1:3, , drop = FALSE], raw_scores = TRUE
        )
        expect_identical(first$effective_ncomp, second$effective_ncomp)
        expect_identical(first_prediction$Ypred, second_prediction$Ypred)
        expect_equal(
            first_prediction$LDA_scores,
            second_prediction$LDA_scores,
            tolerance = 0
        )
    }
})

test_that("empirical-prior direct LDA paths remain finite", {
    data <- direct_lda_rank_example()
    set.seed(43L)
    cases <- list(
        empirical = data,
        equal = list(
            x = matrix(rnorm(10L * 20L), nrow = 10L, ncol = 20L),
            y = factor(rep(c(0L, 1L), each = 5L))
        )
    )
    for (case in cases) {
        fit <- suppressWarnings(pls(
            case$x,
            case$y,
            ncomp = 1:11,
            classifier = "lda",
            fit = TRUE,
            backend = "cpu",
            seed = 77L
        ))
        prediction <- predict(
            fit, case$x[1:3, , drop = FALSE], raw_scores = TRUE
        )
        expect_true(all(is.finite(prediction$LDA_scores)))
        expect_false(anyNA(prediction$Ypred))
    }
})

test_that("float32 and available accelerators retain direct LDA paths", {
    data <- direct_lda_rank_example(float32 = TRUE)
    backends <- c("cpu")
    if (isTRUE(has_metal())) {
        backends <- c(backends, "metal")
    }
    if (isTRUE(has_cuda())) {
        backends <- c(backends, "cuda")
    }
    for (backend in backends) {
        fit <- suppressWarnings(pls(
            data$x,
            data$y,
            ncomp = 1:10,
            classifier = "lda",
            fit = TRUE,
            backend = backend,
            seed = 77L
        ))
        prediction <- predict(
            fit,
            data$x[1:3, , drop = FALSE],
            raw_scores = TRUE,
            backend = backend
        )
        expect_complete_lda_path(fit, prediction, 1:10, c(1:8, 8L, 8L))
        expect_equal(
            prediction$LDA_scores[, , 10L],
            prediction$LDA_scores[, , 8L],
            tolerance = 0
        )
    }
})

test_that("direct LDA zero-direction and single-class behavior is explicit", {
    x <- matrix(1, nrow = 20L, ncol = 8L)
    y <- factor(rep(c("a", "b"), 10L))
    fit <- pls(
        x,
        y,
        ncomp = 1:3,
        classifier = "lda",
        fit = TRUE,
        backend = "cpu",
        seed = 11L
    )
    prediction <- predict(fit, x[1:3, , drop = FALSE], raw_scores = TRUE)
    expect_identical(fit$effective_ncomp, rep(0L, 3L))
    expect_true(all(is.finite(prediction$LDA_scores)))
    expect_identical(prediction$Ypred[, 1L], prediction$Ypred[, 3L])
    expect_equal(
        prediction$LDA_scores[, , 1L],
        prediction$LDA_scores[, , 3L],
        tolerance = 0
    )
    expect_error(
        pls(x, factor(rep("a", nrow(x))), ncomp = 1:3,
            classifier = "lda"),
        "at least two observed classes"
    )
})
