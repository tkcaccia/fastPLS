test_that("cached-crossproduct SIMPLS materializes scores in one equivalent batch", {
    skip_if_not_installed("float")

    set.seed(731)
    predictors_double <- matrix(rnorm(800L * 40L), 800L, 40L)
    predictors <- float::fl(predictors_double)
    labels <- factor(rep(letters[1:8], each = 100L))
    model <- pls(
        predictors, labels,
        ncomp = 20L,
        scaling = "centering",
        method = "simpls",
        classifier = "lda",
        backend = "cpu",
        fit = TRUE,
        seed = 731L
    )

    center <- as.vector(float::dbl(model$mX))
    scale <- as.vector(float::dbl(model$vX))
    weights <- as.matrix(float::dbl(model$R))
    scores <- as.matrix(float::dbl(model$Ttrain))
    expected <- sweep(
        sweep(predictors_double, 2L, center, "-"), 2L, scale, "/"
    ) %*% weights

    expect_equal(scores, expected, tolerance = 2e-5)
    expect_true(all(is.finite(scores)))
})

test_that("classification families avoid duplicate LDA scores when fit is false", {
    skip_if_not_installed("float")

    set.seed(732)
    predictors64 <- matrix(rnorm(90L * 18L), 90L, 18L)
    labels <- factor(rep(letters[1:3], each = 30L))
    families <- c("plssvd", "simpls", "opls", "kernelpls")

    for (precision in c("float64", "float32")) {
        predictors <- if (precision == "float32") {
            float::fl(predictors64)
        } else {
            predictors64
        }
        for (family in families) {
            compact <- pls(
                predictors, labels,
                ncomp = 2L,
                method = family,
                classifier = "lda",
                backend = "cpu",
                fit = FALSE,
                return_variance = FALSE,
                seed = 732L
            )
            retained <- pls(
                predictors, labels,
                ncomp = 2L,
                method = family,
                classifier = "lda",
                backend = "cpu",
                fit = TRUE,
                return_variance = FALSE,
                seed = 732L
            )
            compact_inner <- if (is.null(compact$inner_model)) {
                compact
            } else {
                compact$inner_model
            }
            retained_inner <- if (is.null(retained$inner_model)) {
                retained
            } else {
                retained$inner_model
            }
            compact_pred <- predict(compact, predictors)$Ypred[[1L]]
            retained_pred <- predict(retained, predictors)$Ypred[[1L]]

            expect_null(
                compact_inner$Ttrain,
                info = paste(precision, family)
            )
            expect_false(
                is.null(retained_inner$Ttrain),
                info = paste(precision, family)
            )
            expect_identical(
                compact_pred,
                retained_pred,
                info = paste(precision, family)
            )
        }
    }
})

test_that("compact float32 OPLS LDA matches explicit-score training", {
    skip_if_not_installed("float")

    set.seed(733)
    predictors <- float::fl(matrix(rnorm(150L * 24L), 150L, 24L))
    labels <- factor(rep(letters[1:3], each = 50L))
    reference <- pls(
        predictors, labels,
        ncomp = 8L,
        method = "opls",
        classifier = "argmax",
        backend = "cpu",
        fit = TRUE,
        return_variance = FALSE,
        seed = 733L
    )
    inner <- reference$inner_model
    encoded <- as.integer(factor(labels, levels = inner$lev))
    lda_models <- fastPLS:::lda_train_prefix_float32_cpp(
        inner$Ttrain,
        encoded,
        length(inner$lev),
        as.integer(inner$ncomp)
    )
    names(lda_models) <- as.character(inner$ncomp)
    inner$lda <- list(
        ncomp = inner$ncomp,
        models = lda_models,
        ridge = vapply(lda_models, `[[`, numeric(1L), "ridge"),
        train_backend = "explicit_score_reference"
    )
    inner$classification_rule <- "lda_cpp"
    reference$inner_model <- inner
    compact <- pls(
        predictors, labels,
        ncomp = 8L,
        method = "opls",
        classifier = "lda",
        backend = "cpu",
        fit = FALSE,
        return_variance = FALSE,
        seed = 733L
    )

    expect_identical(
        predict(compact, predictors)$Ypred[[1L]],
        predict(reference, predictors)$Ypred[[1L]]
    )
})
