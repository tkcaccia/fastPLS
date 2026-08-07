# fastPLS 0.99.10 code audit

Audit date: 2026-08-08

## Scope

The audit covered the exported R API, generated documentation, CPU/CUDA/Metal
routing, float32 capability checks, PLS estimator selection, LDA fitting,
prediction output, single and nested cross-validation, permutation testing,
tests, and claims made in the CMPB manuscript and supplement.

## Verified behavior

- The public modelling families are PLS-SVD, SIMPLS, OPLS, and kernel PLS.
- The public classification heads are argmax and latent-space LDA. Former
  candidate-kNN names are rejected by the public classifier resolver.
- Requested SIMPLS is not silently replaced by PLS-SVD. Unsafe dense CUDA
  classification responses stop with an explicit error.
- CPU IRLBA is the deterministic reference route. rSVD is explicitly labelled
  approximate and records oversampling, power iterations, seed, effective rank,
  and structural diagnostics.
- OPLS and nonlinear kernel PLS retain host orchestration on accelerator routes.
  Metal LDA is hybrid: Metal projects scores and the compiled CPU solver fits
  the discriminant model.
- Float32 capability checks run before fitting and distinguish validated,
  experimental, hybrid, measured-risk, and unavailable combinations.
- Windows CPU-only installation supports a portable float-package rSVD route
  for PLS-SVD, SIMPLS, and linear-kernel SIMPLS with argmax. Native compiled
  float32 OPLS, nonlinear kernel PLS, and LDA remain unavailable on Windows.
- `pls.single.cv()` can select by accuracy, balanced accuracy, R2, Q2, or RMSD.
  `pls.double.cv()` forwards the same endpoint to inner selection and its
  permutation test. For a loss metric, the permutation tail is reversed
  correctly.
- Training R2, held-out Q2, and RMSD are calculated and returned separately.
- Model output settings are stored in the private `fastPLS_internal` attribute
  and are not printed as public list fields.

## Corrections made

- Removed the public `pca()` function, `fastPLSPCA` prediction/plot methods,
  help pages, vignette sections, README entries, namespace registrations, and
  PCA-specific tests. `fastsvd()` remains the standalone decomposition API.
- Removed obsolete public documentation claiming candidate-kNN availability.
  Argmax and LDA are now the only documented classification heads.
- Corrected Windows float32 documentation: the portable CPU fallback is
  available for a restricted rSVD/argmax scope; native compiled single-
  precision support is not claimed.
- Corrected `has_cuda()` and `has_metal()` help-page links after PCA removal.
- Added a Windows regression test for the type-correct unavailable native
  float32 argmax entry point.

## Residual internal debt

Historical candidate-kNN and class-bias kernels remain as unreachable internal
compiled symbols used by neither the exported R API nor current benchmark
scripts. They do not affect estimator routing or manuscript results, but a
future ABI-focused cleanup should remove them together from C++, CUDA,
registration code, and generated wrappers in one coordinated change.

## Validation

- Targeted API, CV, classifier-routing, SVD, and float32 tests: 253 passed,
  zero failed.
- Complete testthat suite after public PCA removal: 622 passed, zero failed;
  three expected float32 capability warnings and 25 hardware/platform skips.
- Source archive: `fastPLS_0.99.10.tar.gz`.
- SHA-256: `163ac7bd5c0c241f3817fac989e219f71b3956b388f6fcefa2f3420c45051b25`.
- CRAN-style checking proceeds successfully through compilation, code, tests,
  examples, vignette rebuilding, and manuals. The only expected incoming NOTE
  is the development-version jump relative to the old CRAN 0.2 release.

## Manuscript interpretation

Quantitative benchmark values remain tied to the archived 0.99.6 source used
to produce them. Version 0.99.10 is the audited current interface and contains
documentation, installation, balanced-accuracy selection, Windows fallback,
and API-cleanup changes. No benchmark value is relabelled as a 0.99.10 rerun.
