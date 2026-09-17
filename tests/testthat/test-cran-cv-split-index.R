test_that("single CV optionally returns sample-by-fold membership", {
    set.seed(5101L)
    x <- matrix(rnorm(30L * 6L), 30L, 6L)
    y <- factor(rep(letters[1:3], each = 10L))

    ordinary <- pls.single.cv(
        x, y, ncomp = 1L, kfold = 3L, fit = FALSE, seed = 11L
    )
    indexed <- pls.single.cv(
        x, y, ncomp = 1L, kfold = 3L, fit = FALSE,
        return_splits = TRUE, seed = 11L
    )

    expect_null(ordinary$split_index)
    expect_identical(dim(indexed$split_index), c(30L, 3L))
    expect_identical(rownames(indexed$split_index), as.character(1:30))
    expect_true(all(indexed$split_index %in% c("training", "test")))
    expect_true(all(rowSums(indexed$split_index == "test") == 1L))
    for (fold in seq_len(3L)) {
        expect_identical(
            unname(indexed$split_index[, fold] == "test"),
            indexed$fold == fold - 1L
        )
    }
})

test_that("nested CV optionally returns outer and inner membership", {
    set.seed(5102L)
    x <- matrix(rnorm(30L * 5L), 30L, 5L)
    y <- factor(rep(letters[1:3], each = 10L))

    ordinary <- pls.double.cv(
        x, y, ncomp = 1L, runn = 2L,
        kfold_inner = 2L, kfold_outer = 3L, seed = 12L
    )
    indexed <- pls.double.cv(
        x, y, ncomp = 1L, runn = 2L,
        kfold_inner = 2L, kfold_outer = 3L,
        return_splits = TRUE, seed = 12L
    )

    expect_null(ordinary$split_index)
    expect_identical(dim(indexed$split_index), c(30L, 18L))
    expect_identical(rownames(indexed$split_index), as.character(1:30))
    expect_true(all(indexed$split_index %in%
        c("training", "test", "outer_test")))

    for (run in seq_len(2L)) {
        outer_names <- sprintf("run_%d_outer_%d", run, seq_len(3L))
        outer <- indexed$split_index[, outer_names, drop = FALSE]
        expect_true(all(rowSums(outer == "test") == 1L))
        expect_identical(
            max.col(outer == "test", ties.method = "first"),
            indexed$results[[run]]$fold
        )
        for (outer_fold in seq_len(3L)) {
            inner_names <- sprintf(
                "run_%d_outer_%d_inner_%d",
                run,
                outer_fold,
                seq_len(2L)
            )
            inner <- indexed$split_index[, inner_names, drop = FALSE]
            held_out <- outer[, outer_fold] == "test"
            expect_true(all(inner[held_out, ] == "outer_test"))
            expect_true(all(rowSums(inner[!held_out, , drop = FALSE] ==
                "test") == 1L))
        }
    }
})

test_that("CV split output flag is validated", {
    x <- matrix(seq_len(48), 12L, 4L)
    y <- rep(c(0, 1), each = 6L)

    expect_identical(formals(pls.single.cv)$return_splits, FALSE)
    expect_identical(formals(pls.double.cv)$return_splits, FALSE)

    expect_error(
        pls.single.cv(x, y, ncomp = 1L, kfold = 2L, return_splits = 1),
        "return_splits"
    )
    expect_error(
        pls.double.cv(
            x, y, ncomp = 1L, kfold_inner = 2L, kfold_outer = 2L,
            return_splits = NA
        ),
        "return_splits"
    )
})
