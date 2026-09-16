test_that("nested CV handles a constant-response inner fold", {
    set.seed(10L)
    x <- matrix(rnorm(54L * 768L), nrow = 54L, ncol = 768L)
    y <- c(rep(0, 52L), 0.00549885308390477, 0.0599934866563745)

    result <- pls.double.cv(
        x,
        y,
        ncomp = 1L,
        kfold_outer = 5L,
        kfold_inner = 5L,
        scaling = "centering",
        backend = "cpu",
        oversample = 10L,
        power = 2L,
        seed = 20260852L,
        perm.test = FALSE
    )

    expect_length(result$Ypred, nrow(x))
    expect_true(all(is.finite(result$Ypred)))
    expect_length(result$Q2Y, 1L)
    expect_length(result$RMSD, 1L)
})
