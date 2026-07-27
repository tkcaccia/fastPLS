test_that("public classifier names route through backend", {
  expect_equal(fastPLS:::.resolve_classifier_for_backend("argmax", "cpu"), "argmax")
  expect_equal(fastPLS:::.resolve_classifier_for_backend("lda", "cpu"), "lda_cpp")
  expect_equal(fastPLS:::.resolve_classifier_for_backend("lda", "cuda"), "lda_cuda")
  expect_equal(fastPLS:::.resolve_classifier_for_backend("lda", "metal"), "lda_metal")
  expect_error(fastPLS:::.resolve_classifier_for_backend("cknn", "cpu"))
  expect_error(fastPLS:::.resolve_classifier_for_backend("candidate_knn", "cpu"))
  expect_error(fastPLS:::.normalize_classifier_public("lda_cuda"))
})
