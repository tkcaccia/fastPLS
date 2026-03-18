# fastPLS API/Documentation Audit (2026-03-18)

## What Was Inconsistent

1. Method-level documentation used mixed terminology about backend behavior,
   especially for `irlba` and `dc` across high-level modeling vs. SVD utility
   wrappers.
2. Output descriptions in several man pages were generic and did not always
   reflect method-specific returned objects (for example, `simpls_fast` returns
   empty `P` and `Ttrain` in current C++ implementation).
3. Utility docs did not clearly state that `svd_run(method = "irlba")` currently
   follows exact CPU dispatch instead of invoking the legacy IRLB branch used in
   `pls_model1/2`.

## What Was Fixed

- Rewrote `man/pls.Rd` at algorithm/backend level and tied each option to actual
  code paths in `R/main.R`, `src/fastPLS.cpp`, and `src/svd_*` files.
- Rewrote `man/pls_r.Rd` to explicitly document that pure-R `irlba`/`dc` both map
  to exact base `svd()` while `cpu_rsvd` is randomized.
- Rewrote `man/svd_methods.Rd` and `man/svd_tools.Rd` to document dispatch truthfully,
  including partial GPU residency of `cuda_rsvd` and current `irlba` behavior in
  utility wrappers.
- Rewrote `man/predict.fastPLS.Rd` and `man/ViP.Rd` to match actual returned values
  and computation semantics.
- Expanded `README.md` with a code-faithful methods/backends section.

## What Remains Intentionally Internal

- `truncated_svd_debug()` is intentionally not exported as a user-facing function.
- Low-level C++ wrappers (`pls_model1`, `pls_model2`, `pls_model2_fast`) remain
  internal implementation details exposed through generated Rcpp bindings but not
  part of the stable high-level API contract.
- Backend enum value `SVD_METHOD_CPU_EXACT` exists in C++ for compatibility and
  internal routing, but is not exposed as a public `svd.method` choice in `pls()`.
