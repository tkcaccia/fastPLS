test_that("compact float32 PLS-SVD reuses an equivalent score Gram", {
    skip_if_not_installed("float")

    set.seed(421)
    predictors <- float::fl(matrix(rnorm(240L * 80L), 240L, 80L))
    labels <- factor(rep(letters[1:4], each = 60L))
    fit_model <- function(store_scores) {
        fastPLS:::.fit_float32_pls(
            predictors, labels, 3L, 1L, "plssvd", "cpu", "cpu_rsvd",
            20L, 2L, 421L, FALSE, store_scores
        )
    }

    retained <- fit_model(TRUE)
    compact <- fit_model(FALSE)
    as_double <- function(value) as.matrix(float::dbl(value))

    expect_equal(as_double(compact$R), as_double(retained$R), tolerance = 0)
    expect_equal(as_double(compact$Q), as_double(retained$Q), tolerance = 0)
    expect_equal(
        as_double(compact$W_latent[[1L]]),
        as_double(retained$W_latent[[1L]]),
        tolerance = 0
    )
    expect_null(compact$Ttrain)
    expect_true(is.matrix(attr(compact, "fastPLS_score_gram")))
})

test_that("temporary float32 classifier moments are not public output", {
    skip_if_not_installed("float")

    set.seed(422)
    predictors <- float::fl(matrix(rnorm(120L * 24L), 120L, 24L))
    labels <- factor(rep(letters[1:3], each = 40L))
    model <- pls(
        predictors, labels,
        ncomp = 2L,
        method = "plssvd",
        classifier = "lda",
        backend = "cpu",
        fit = FALSE,
        seed = 422L
    )

    expect_null(attr(model, "fastPLS_score_gram"))
    expect_null(attr(model, "fastPLS_class_predictor_sums"))
})
