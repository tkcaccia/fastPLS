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

test_that("AUROC pools finite scores from regular and one-class folds", {
    set.seed(5L)
    x <- matrix(rnorm(20L * 6L), nrow = 20L, ncol = 6L)
    y <- factor(c(rep("negative", 10L), rep("positive", 10L)),
        levels = c("negative", "positive"))
    group <- c(rep("negative_group", 10L), rep("positive_a", 5L),
        rep("positive_b", 5L))
    run <- function() {
        suppressWarnings(pls.single.cv(
            x,
            y,
            ncomp = 1:3,
            constrain = group,
            kfold = 3L,
            classifier = "lda",
            selection = "AUROC",
            backend = "cpu",
            seed = 19L,
            fit = FALSE,
            return_splits = TRUE
        ))
    }
    first <- run()
    second <- run()

    expect_identical(first$selection_metric, "AUROC")
    expect_true(all(is.finite(first$selection_values)))
    pooled_scores <- vapply(seq_len(3L), function(index) {
        fastPLS:::.cv_binary_auroc(
            y,
            first$lda_scores[, 2L, index] -
                first$lda_scores[, 1L, index],
            levels(y)
        )
    }, numeric(1L))
    expect_equal(unname(first$selection_values), pooled_scores, tolerance = 0)
    expect_identical(first$selection_values, second$selection_values)
    balanced <- suppressWarnings(pls.single.cv(
        x,
        y,
        ncomp = 1:3,
        constrain = group,
        kfold = 3L,
        classifier = "lda",
        selection = "balanced_accuracy",
        backend = "cpu",
        seed = 19L,
        fit = FALSE,
        return_splits = TRUE
    ))
    expect_identical(first$split_index, balanced$split_index)
    expect_length(first$degenerate_inner_folds, 1L)
    expect_true(first$constant_classifier_fallback)
    expect_identical(first$minimum_negative_training_count, 0L)
    expect_identical(first$minimum_positive_training_count, 5L)
    expect_true(first$component_selection_informative)
    expect_true(all(is.finite(first$lda_scores)))
    expect_identical(first$best_ncomp, 1L)

    argmax <- suppressWarnings(pls.single.cv(
        x,
        y,
        ncomp = 1:2,
        constrain = group,
        kfold = 3L,
        classifier = "argmax",
        selection = "AUROC",
        backend = "cpu",
        seed = 19L,
        fit = FALSE
    ))
    expect_true(all(is.finite(argmax$selection_values)))
    expect_true(argmax$constant_classifier_fallback)
})

test_that("nested AUROC flags a non-estimable outer discrimination fold", {
    set.seed(6L)
    x <- matrix(rnorm(20L * 6L), nrow = 20L, ncol = 6L)
    y <- factor(c(rep("negative", 10L), rep("positive", 10L)),
        levels = c("negative", "positive"))
    group <- c(rep("negative_group", 10L), rep("positive_a", 5L),
        rep("positive_b", 5L))
    run <- function() {
        suppressWarnings(pls.double.cv(
            x,
            y,
            ncomp = 1:2,
            constrain = group,
            kfold_outer = 3L,
            kfold_inner = 2L,
            classifier = "lda",
            selection = "AUROC",
            backend = "cpu",
            seed = 23L,
            perm.test = FALSE
        ))
    }
    first <- run()
    second <- run()

    expect_identical(first$Ypred, second$Ypred)
    expect_false(anyNA(first$Ypred))
    expect_true(any(!first$outer_discrimination_estimable))
    expect_gt(nrow(first$non_estimable_outer_folds), 0L)
    expect_true(first$constant_classifier_fallback)
    expect_false(first$component_selection_informative)
    expect_identical(first$minimum_negative_training_count, 0L)
    expect_identical(first$minimum_positive_training_count, 0L)
    expect_true(is.finite(first$AUROC[[1L]]))
    expect_identical(first$results[[1L]]$metric_name, "AUROC")
})

test_that("AUROC selection rejects multiclass responses before fitting", {
    expect_error(
        pls.single.cv(
            as.matrix(iris[, 1:4]),
            iris$Species,
            ncomp = 1:2,
            selection = "AUROC",
            backend = "cpu"
        ),
        "exactly two response classes"
    )
})
