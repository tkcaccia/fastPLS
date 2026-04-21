helper_path <- testthat::test_path("../../benchmark/helpers_simulated_fastpls.R")
if (!file.exists(helper_path)) {
  testthat::skip("benchmark helper excluded from source package")
}
source(helper_path, local = TRUE)

test_that("spectral regimes are generated and ordered sensibly", {
  fast <- simfast_spectrum(8, "fast_decay")
  sharp <- simfast_spectrum(8, "sharp_decay")
  slow <- simfast_spectrum(8, "slow_decay")
  cluster <- simfast_spectrum(12, "clustered_top")

  expect_length(fast, 8L)
  expect_length(sharp, 8L)
  expect_length(slow, 8L)
  expect_true(all(diff(fast) < 0))
  expect_true(all(diff(slow) < 0))
  expect_true(cluster[1] >= cluster[2])
  expect_true(abs(cluster[1] - cluster[min(10L, length(cluster))]) < 0.3)
})

test_that("regression generator returns finite centered train/test matrices", {
  cfg <- utils::modifyList(simfast_family_catalog()[["sim_reg_base"]], list(
    n = 120L,
    p = 40L,
    q = 6L,
    r_true = 4L
  ))
  ds <- simfast_generate_dataset(cfg, seed_data = 11L, seed_split = 12L)

  expect_true(all(is.finite(ds$Xtrain)))
  expect_true(all(is.finite(ds$Xtest)))
  expect_true(all(is.finite(ds$Ytrain)))
  expect_true(all(is.finite(ds$Ytest)))
  expect_equal(ncol(ds$Xtrain), 40L)
  expect_equal(ncol(ds$Ytrain), 6L)
  expect_equal(ds$meta$r_true, 4L)
  expect_equal(ds$task_type, "regression")
})

test_that("classification generator is non-degenerate and preserves all classes", {
  cfg <- utils::modifyList(simfast_family_catalog()[["sim_cls_base"]], list(
    n = 300L,
    p = 60L,
    n_classes = 4L,
    q = 4L,
    r_true = 5L,
    class_bias = c(0, 0.4, -0.2, 0.1)
  ))
  ds <- simfast_generate_dataset(cfg, seed_data = 21L, seed_split = 22L)

  expect_true(is.factor(ds$Ytrain))
  expect_true(is.factor(ds$Ytest))
  expect_equal(length(unique(ds$Ytrain)), 4L)
  expect_equal(length(unique(ds$Ytest)), 4L)
  expect_equal(ncol(ds$Ytrain_dummy), 4L)
  expect_true(is.finite(ds$meta$class_margin))
})

test_that("dropout family increases sparsity", {
  base_cfg <- utils::modifyList(simfast_family_catalog()[["sim_reg_base"]], list(
    n = 100L,
    p = 50L,
    q = 5L,
    r_true = 4L,
    dropout_rate = 0
  ))
  sparse_cfg <- utils::modifyList(simfast_family_catalog()[["sim_sparse_singlecell_like"]], list(
    n = 200L,
    p = 80L,
    n_classes = 5L,
    q = 5L,
    r_true = 4L,
    dropout_rate = 0.8
  ))

  dense_ds <- simfast_generate_dataset(base_cfg, seed_data = 31L, seed_split = 32L)
  sparse_ds <- simfast_generate_dataset(sparse_cfg, seed_data = 31L, seed_split = 32L)

  dense_zero_rate <- dense_ds$meta$observed_zero_rate_X
  sparse_zero_rate <- sparse_ds$meta$observed_zero_rate_X
  expect_gt(sparse_zero_rate, dense_zero_rate)
})

test_that("analysis transforms preserve finite data and shrink dimensions", {
  cfg <- utils::modifyList(simfast_family_catalog()[["sim_reg_base"]], list(
    n = 120L,
    p = 40L,
    q = 10L,
    r_true = 4L
  ))
  ds <- simfast_generate_dataset(cfg, seed_data = 41L, seed_split = 42L)

  ds_sample <- simfast_prepare_analysis_dataset(ds, "sample_fraction", 0.5, seed_analysis = 43L)$dataset
  ds_x <- simfast_prepare_analysis_dataset(ds, "xvar_fraction", 0.25, seed_analysis = 44L)$dataset
  ds_y <- simfast_prepare_analysis_dataset(ds, "yvar_fraction", 0.30, seed_analysis = 45L)$dataset

  expect_true(nrow(ds_sample$Xtrain) < nrow(ds$Xtrain))
  expect_true(ncol(ds_x$Xtrain) < ncol(ds$Xtrain))
  expect_true(ncol(ds_y$Ytrain) < ncol(ds$Ytrain))
  expect_true(all(is.finite(ds_sample$Xtrain)))
  expect_true(all(is.finite(ds_x$Xtrain)))
  expect_true(all(is.finite(ds_y$Ytrain)))
})
