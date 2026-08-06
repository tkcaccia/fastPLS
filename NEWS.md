# fastPLS 0.99.8

* `pls.single.cv()` and `pls.double.cv()` can now select classification models
  with `selection_metric = "balanced_accuracy"`. Nested permutation tests use
  the same selected endpoint, preventing classification analyses tuned and
  reported by balanced accuracy from being tested against dummy-response Q2.
* macOS installation now detects the system Metal frameworks even when
  `xcrun --show-sdk-path` is unavailable, and configure output reports CUDA and
  Metal status independently.
* Float32 capability reporting now distinguishes validated, experimental,
  hybrid, unavailable, and measured failed routes. The public `pls()` interface
  emits route-specific warnings or errors before allocation, and benchmark
  summaries separate input storage, baseline and incremental host RSS, sampled
  GPU use, runtime, and predictive differences from float64.
* Added public PLS, PCA, SVD, prediction, evaluation and cross-validation
  interfaces for CPU, optional CUDA and optional Apple Metal backends.
* Added optional classification heads for PLS-DA, including argmax, latent-space
  LDA and candidate-kNN.
* Added package datasets, examples, benchmark scripts and a single user
  vignette.
* CUDA and Metal builds are optional; CPU-only installation remains the default.
* Float32 SIMPLS now retains the latent scores already produced by the compiled
  recurrence. LDA reuses these scores instead of centering, scaling and
  projecting the full training matrix a second time, reducing peak memory and
  runtime without changing predictions.
* Float32 fitting now reports shape-based warnings for precision-sensitive
  classification, extreme multivariate responses, and nonlinear kernel routes.
  Float64 remains the numerical reference.
