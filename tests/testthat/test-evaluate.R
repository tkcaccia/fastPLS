test_that("evaluate computes classification metrics", {
  observed <- factor(c("a", "a", "b", "b", "c"))
  predicted <- factor(c("a", "b", "b", "b", "c"), levels = levels(observed))

  res <- evaluate(observed, predicted)

  expect_identical(res$task, "classification")
  expect_equal(res$metrics$accuracy, 4 / 5)
  expect_equal(res$metrics$no_information_rate, 2 / 5)
  expect_equal(res$metrics$lift_accuracy, 2)
  expect_true("macro_f1" %in% names(res$metrics))
  expect_equal(sum(res$confusion), 5)
  expect_false("notes" %in% names(res))
})

test_that("evaluate computes top-k accuracy from score matrices", {
  observed <- factor(c("a", "b", "c"))
  scores <- matrix(
    c(
      0.8, 0.1, 0.1,
      0.4, 0.5, 0.1,
      0.6, 0.3, 0.1
    ),
    nrow = 3,
    byrow = TRUE,
    dimnames = list(NULL, c("a", "b", "c"))
  )

  res <- evaluate(observed, scores, top_k = c(1L, 3L))

  expect_equal(res$metrics$accuracy, 2 / 3)
  expect_equal(res$topk$accuracy[res$topk$k == 3L], 1)
})

test_that("evaluate computes regression and spectral metrics", {
  observed <- matrix(c(1, 2, 3, 2, 4, 6), nrow = 3, ncol = 2)
  predicted <- observed + 0.1
  train <- observed + 0.5

  res <- evaluate(observed, predicted, ytrain = train)

  expect_identical(res$task, "regression")
  expect_true(all(c("R2", "Q2", "RMSD", "MRE_percent", "RPD") %in% names(res$metrics)))
  expect_equal(nrow(res$per_response), 2)
  expect_false(isTRUE(all.equal(res$metrics$R2, res$metrics$Q2)))
  expect_false("notes" %in% names(res))
})

test_that("evaluate can omit response-wise regression metrics", {
  observed <- matrix(c(1, 2, 3, 2, 4, 6), nrow = 3, ncol = 2)
  predicted <- observed + 0.1

  res <- evaluate(observed, predicted, bycol = FALSE)

  expect_null(res$per_response)
  expect_true(all(c("R2", "Q2", "RMSD", "MAE") %in% names(res$metrics)))
  expect_true(is.na(res$metrics$Q2))
})

test_that("evaluate only reports notes when they are informative", {
  observed <- c(1, 2, 3)
  predicted <- observed + 0.1

  res <- evaluate(observed, predicted)

  expect_true("notes" %in% names(res))
  expect_match(res$notes[[1]], "Q2 was not computed")
  expect_true(is.na(res$metrics$Q2))
  expect_match(res$metric_definitions$Q2, "requires ytrain")
})
