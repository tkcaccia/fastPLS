# fastPLS 0.99.29

* Resolved the BiocCheck source-formatting notes by limiting package-facing
  lines to 80 characters and refactoring functions to at most 50 coding lines.

* Fixed float32 single-cross-validation argument forwarding and completed the
  resident Metal SIMPLS model-assembly helpers.

* Preserved the public PLS, cross-validation, metric, and permutation
  contracts. The complete package test suite passes.

# fastPLS 0.99.28

* Added direct support for `Biobase::ExpressionSet` predictor inputs in
  `pls()` and `predict()`. Assay rows are interpreted as variables and assay
  columns as samples, and are transposed internally to the sample-by-variable
  layout used by fastPLS.

* Added a Bioconductor-native classification example to the package vignette
  and expanded automated tests for fitting and predicting from
  `ExpressionSet` objects.

* Reformatted R and vignette sources to improve Bioconductor style compliance.

# fastPLS 0.99.25

* Synchronized the exported API, vignette, manual, README, and manuscript
  capability descriptions. Removed stale PCA references and deprecated the
  ignored `lda_ridge` compatibility argument; supplying it now warns, and it is
  no longer included in cross-validation tuning records.

* Standardized response-variance metrics across the public API. Training
  `R2Y`, independent-test `Q2Y`, fold-training-mean single-CV `Q2Y`, and
  outer-fold `Q2Y` now use explicit, documented denominators; dummy-response
  PLS-DA values are labelled separately from classification accuracy.
  `evaluate()` returns `NA` for Q2 when no training response is supplied rather
  than silently reproducing R2.

* Corrected finite Monte Carlo permutation inference to use
  `(b + 1) / (B + 1)`, preventing zero p-values. Grouped nested validation now
  permutes complete constraint blocks within equal-size exchangeability strata,
  holds folds and randomized-solver seeds fixed, and records failed null fits.

* Clarified that returning the sequential SIMPLS component path is standard
  behavior also provided by `pls::simpls.fit` and is not claimed as a fastPLS
  novelty.

* Defined the fastPLS contribution as compiled, shape-dependent execution and
  storage: cached deflation and cross-products, incremental coefficient and
  fitted-value updates, compact latent prediction, and implicit
  cross-covariance products.

* Added a minimally optimized compiled SIMPLS baseline and explicit asymptotic
  time/storage expressions to the implementation mapping and vignette, while
  documenting that the retained optimizations are not uniformly faster.

# fastPLS 0.99.24

* Standardized the public SIMPLS direction-refresh rule across CPU, CUDA, and
  Metal. Every component now receives a fresh rank-one IRLBA solve or
  oversampled rSVD sketch from the current deflated cross-covariance; no
  candidate block or preceding latent direction is reused.

* Removed the abandoned warm-start, block-refresh, and adaptive-refresh
  controls from the backend-control registry and removed the residual native
  Metal warm start.

* Added fitted-model diagnostics and unit tests that identify the active
  `fresh_per_component` rule, retained execution optimizations, and rejected
  development prototypes.

# fastPLS 0.99.23

* Changed the randomized-SVD default to `(oversample = 20, power = 2)` on CPU
  and CUDA. This stronger setting met all 585 CPU and 40 CUDA component-level checks across
  five prespecified random seeds in the release-candidate audits; the prior
  `(10, 2)` setting failed five of 255 screening checks and is not treated as
  qualified.

* Added an explicit warning and fitted-model diagnostic status whenever a
  user requests randomized controls that were not qualified on the
  prespecified backend validation panel. Metal randomized SVD remains marked
  as unqualified pending a dedicated multi-seed audit.

* Aligned non-exported C++ bridge defaults with the qualified CPU controls and
  expanded release tests for backend-specific dispatch and diagnostics.

# fastPLS 0.99.22

* Made the numerically qualified randomized-SVD configuration the package-wide
  default: oversampling is 10 and the number of power iterations is 2.

* Removed undocumented, matrix-shape-dependent SIMPLS overrides that could
  silently reduce randomized-SVD oversampling or power iterations. Explicit
  controls supplied through `...` still take precedence and are recorded in
  fitted-model diagnostics.

* Added release tests that verify the effective randomized-SVD defaults used by
  `fastsvd()`, `pls()`, and cross-validation.

* Strengthened benchmark provenance records with Git worktree support, source
  tree and tag identifiers, and SHA-256 checksums for both the benchmark script
  and frozen package archive.

# fastPLS 0.99.21

* Made randomized SVD the effective default throughout the public PLS and
  cross-validation APIs, including the internal cross-validation tuning grid
  and the refit path used by stored CV configurations.

* Corrected single-split permutation p-values for multi-component models so
  each permuted Q2 distribution is compared with the corresponding observed
  component rather than a recycled full Q2 vector.

* Expanded the permanent input-grid tests across all four PLS families,
  regression and classification, argmax and LDA, IRLBA and rSVD, and both
  single and double cross-validation. CPU, CUDA, Metal, and float32 grids were
  also exercised during release validation.

# fastPLS 0.99.20

* Fixed nested permutation testing with latent-space LDA. Recursive
  `pls.double.cv()` calls now receive the public `"lda"` classifier name and
  resolve it for the selected backend internally, instead of leaking the
  CPU-specific internal identifier `"lda_cpp"` through the public API.

# fastPLS 0.99.19

* Incremented the Bioconductor development version after synchronizing the
  package-specific backend configuration across both GitHub repositories.

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
