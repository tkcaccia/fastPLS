# fastPLS 0.99.6

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
