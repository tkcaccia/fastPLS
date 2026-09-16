grouped_lda_example <- function() {
    set.seed(1L)
    list(
        x = matrix(rnorm(53L * 768L), nrow = 53L, ncol = 768L),
        y = factor(
            c(rep(0L, 30L), rep(1L, 18L), 1L, 0L, 0L, 0L, 1L),
            levels = c(0L, 1L)
        ),
        group = c(rep("OR", 48L), "OU", "PA", rep("PK", 3L))
    )
}

run_grouped_lda_single <- function(data, components, scaling = "centering") {
    pls.single.cv(
        data$x,
        data$y,
        ncomp = components,
        constrain = data$group,
        kfold = 5L,
        classifier = "lda",
        selection = "balanced_accuracy",
        scaling = scaling,
        backend = "cpu",
        oversample = 10L,
        power = 2L,
        seed = 20262901L,
        fit = FALSE
    )
}

test_that("grouped LDA CV retains paths beyond fold-effective rank", {
    data <- grouped_lda_example()
    for (components in list(1L, 1:2, 1:10)) {
        result <- run_grouped_lda_single(data, components)
        expect_equal(result$ncomp, components)
        expect_equal(ncol(result$class_pred), length(components))
        expect_equal(ncol(result$effective_ncomp), length(components))
        expect_true(all(is.finite(result$lda_scores)))
        expect_false(anyNA(result$class_pred))
    }

    result <- run_grouped_lda_single(data, 1:10)
    limited_fold <- which(result$effective_ncomp[, 10L] < 10L)[[1L]]
    effective <- result$effective_ncomp[limited_fold, 10L]
    held_out <- which(result$fold == limited_fold - 1L)
    expect_gt(effective, 0L)
    for (prefix in seq.int(effective + 1L, 10L)) {
        expect_identical(
            result$class_pred[held_out, prefix],
            result$class_pred[held_out, effective]
        )
        expect_equal(
            result$lda_scores[held_out, , prefix],
            result$lda_scores[held_out, , effective],
            tolerance = 0
        )
    }
})

test_that("grouped LDA CV is deterministic across scaling choices", {
    data <- grouped_lda_example()
    for (scaling in c("centering", "autoscaling", "none")) {
        first <- run_grouped_lda_single(data, 1:10, scaling)
        second <- run_grouped_lda_single(data, 1:10, scaling)
        expect_identical(first$class_pred, second$class_pred)
        expect_equal(first$lda_scores, second$lda_scores, tolerance = 0)
        expect_identical(first$effective_ncomp, second$effective_ncomp)
        expect_true(all(is.finite(first$accuracy)))
        expect_true(all(is.finite(first$balanced_accuracy)))
    }
})

test_that("nested grouped LDA CV retains requested inner paths", {
    data <- grouped_lda_example()
    run <- function(components) {
        pls.double.cv(
            data$x,
            data$y,
            ncomp = components,
            constrain = data$group,
            kfold_outer = 5L,
            kfold_inner = 5L,
            classifier = "lda",
            selection = "balanced_accuracy",
            scaling = "centering",
            backend = "cpu",
            oversample = 10L,
            power = 2L,
            seed = 20262901L,
            perm.test = FALSE
        )
    }
    for (components in list(1L, 1:2, 1:10)) {
        result <- run(components)
        expect_false(anyNA(result$Ypred))
        expect_true(all(vapply(
            result$results[[1L]]$inner,
            function(inner) identical(inner$ncomp, components),
            logical(1L)
        )))
        expect_true(all(vapply(
            result$results[[1L]]$inner,
            function(inner) ncol(inner$effective_ncomp) == length(components),
            logical(1L)
        )))
    }
    expect_identical(run(1:10)$Ypred, run(1:10)$Ypred)
})

test_that("zero-direction LDA folds use empirical class priors", {
    x <- matrix(1, nrow = 20L, ncol = 8L)
    y <- factor(rep(c("a", "b"), 10L))
    for (scaling in c("centering", "autoscaling", "none")) {
        result <- pls.single.cv(
            x,
            y,
            ncomp = 1:3,
            kfold = 5L,
            classifier = "lda",
            selection = "balanced_accuracy",
            scaling = scaling,
            backend = "cpu",
            oversample = 4L,
            power = 2L,
            seed = 11L,
            fit = FALSE
        )
        expect_true(all(result$effective_ncomp == 0L))
        expect_true(all(result$status == 5L))
        expect_identical(result$best_ncomp, 1L)
        expect_false(anyNA(result$class_pred))
        expect_true(all(is.finite(result$lda_scores)))
        expect_equal(result$class_pred[, 1L], result$class_pred[, 3L])
        expect_equal(result$lda_scores[, , 1L], result$lda_scores[, , 3L])
    }

    nested <- pls.double.cv(
        x,
        y,
        ncomp = 1:3,
        kfold_outer = 2L,
        kfold_inner = 2L,
        classifier = "lda",
        selection = "balanced_accuracy",
        backend = "cpu",
        oversample = 10L,
        power = 2L,
        seed = 11L,
        perm.test = FALSE
    )
    expect_true(all(is.finite(nested$Ypred)))
    expect_true(all(vapply(
        nested$results[[1L]]$inner,
        function(inner) all(inner$effective_ncomp == 0L),
        logical(1L)
    )))
})

test_that("single-class grouped folds use an explicit finite fallback", {
    set.seed(2L)
    x <- matrix(rnorm(20L * 6L), nrow = 20L, ncol = 6L)
    y <- factor(rep(c("a", "b"), each = 10L))
    group <- rep(c("g1", "g2"), each = 10L)
    result <- suppressWarnings(pls.single.cv(
        x,
        y,
        ncomp = 1:3,
        constrain = group,
        kfold = 2L,
        classifier = "lda",
        selection = "balanced_accuracy",
        backend = "cpu",
        oversample = 20L,
        power = 2L,
        seed = 11L,
        fit = FALSE
    ))
    expect_true(all(result$status == 4L))
    expect_true(all(result$effective_ncomp == 0L))
    expect_false(anyNA(result$class_pred))
    expect_true(all(is.finite(result$lda_scores)))
})

test_that("PLS-SVD LDA caps prefixes when a fold loses a class", {
    set.seed(3L)
    x <- matrix(rnorm(30L * 12L), nrow = 30L, ncol = 12L)
    y <- factor(rep(c("a", "b", "c"), each = 10L))
    group <- c(rep("a_only", 10L), rep("mixed", 20L))
    result <- suppressWarnings(pls.single.cv(
        x,
        y,
        ncomp = 1:2,
        constrain = group,
        kfold = 2L,
        method = "plssvd",
        classifier = "lda",
        selection = "balanced_accuracy",
        backend = "cpu",
        oversample = 20L,
        power = 2L,
        seed = 13L,
        fit = FALSE
    ))
    expect_identical(result$ncomp, 1:2)
    expect_true(any(result$effective_ncomp[, 2L] == 1L))
    expect_false(anyNA(result$class_pred))
    expect_true(all(is.finite(result$lda_scores)))
})

test_that("classification CV preserves paths beyond the full-data rank", {
    set.seed(4L)
    x <- matrix(rnorm(5L * 12L), nrow = 5L, ncol = 12L)
    y <- factor(c("a", "a", "a", "b", "b"))
    result <- pls.single.cv(
        x,
        y,
        ncomp = 1:10,
        kfold = 2L,
        classifier = "lda",
        selection = "balanced_accuracy",
        backend = "cpu",
        oversample = 10L,
        power = 2L,
        seed = 17L,
        fit = FALSE
    )
    expect_identical(result$ncomp, 1:10)
    expect_identical(ncol(result$effective_ncomp), 10L)
    expect_true(all(result$effective_ncomp <= 3L))
    expect_false(anyNA(result$class_pred))
    expect_true(all(is.finite(result$lda_scores)))
})
