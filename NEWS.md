# fastPLS 0.99.18

* Standardized backend selection on the package-specific API used across the
  KODAMA ecosystem: explicit `backend`, `options(fastPLS.backend = ...)`,
  `FASTPLS_BACKEND`, then CPU. Unrelated global backend settings are ignored.

# fastPLS 0.99.17

* Added a session-wide backend selector through `fastPLS_backend()`,
  `options(backend = ...)`, and `BACKEND`. Explicit function
  arguments retain precedence and CPU remains the default. Legacy
  fastPLS-specific selectors remain compatibility fallbacks.

# fastPLS 0.99.16

* Corrected CUDA cross-validation smoke tests to request the supported rSVD
  backend explicitly instead of inheriting an incompatible IRLBA setting.

# fastPLS 0.99.15

* Corrected portable Windows float32 classification prediction so argmax
  decoding no longer calls an unavailable native single-precision kernel.

* Made float32 capability-policy tests platform-independent by explicitly
  testing Unix accelerator policies separately from Windows availability.

* Simplified the `fastcor()` example to use ten numeric rows from `iris`.

# fastPLS 0.99.14

* Incremented the Bioconductor development version to trigger refreshed
  multi-platform validation of the architecture-independent randomized-SVD
  diagnostic test.

# fastPLS 0.99.13

* Made the randomized-SVD diagnostic test architecture-independent. The test
  now accepts and verifies the documented large-residual failure state instead
  of assuming that a stochastic approximation must meet the quality threshold
  on every BLAS and CPU architecture. Runtime diagnostics remain unchanged.

# fastPLS 0.99.12

* Corrected Windows test scoping for native CPU float32 LDA. Tests that require
  unavailable single-precision BLAS/LAPACK kernels are now skipped on Windows;
  the documented runtime error and portable supported float32 routes are
  unchanged.

# fastPLS 0.99.11

* Removed the retired candidate-kNN and class-bias native ABI completely,
  including CPU and CUDA kernels, generated Rcpp wrappers, registrations, and
  unreachable compiled cross-validation branches. Classification remains
  limited to the documented argmax and latent-space LDA heads.

* Preserved compact top-k argmax prediction through a bias-free CPU/CUDA
  implementation, including optional top-5 output.

* Removed retired classifier variants and tuning controls from active benchmark
  generators. Archived benchmark results remain unchanged for provenance.

# fastPLS 0.99.10

* Fixed compilation of CPU-only Windows builds. The unavailable native-float32
  argmax fallback now raises the documented platform error through a
  type-correct integer-vector entry point instead of returning a list.

* Removed the public PCA API and its S3 methods. Principal component analysis
  remains available through dedicated R packages; `fastsvd()` remains the
  package's public standalone decomposition interface.

* Synchronized the public API and documentation around the two supported
  classification heads, argmax and latent-space LDA. Historical candidate-kNN
  wording was removed.

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
* Added public PLS, SVD, prediction, evaluation and cross-validation
  interfaces for CPU, optional CUDA and optional Apple Metal backends.
* Added optional classification heads for PLS-DA using argmax decoding or
  latent-space LDA.
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
