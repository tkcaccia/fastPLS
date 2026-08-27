# Timing-study revision

This revision addresses the timing-methodology concerns without changing the
public API or SIMPLS estimator.

## Short-task repetition and timing modes

`scripts/run_external_simpls_timing.sh` now uses an adaptive cold-process
policy: 50 repetitions when the pilot fit plus prediction takes no longer than
0.5 s, 30 up to 2 s, 15 up to 10 s, and five otherwise. Raw times are retained
at full precision. A separately labelled warm-batch measurement performs one
untimed complete fit and prediction before 10--50 measured iterations. Cold
and warm results are never pooled.

## Output and phase decomposition

The minimum-common-output and ordinary-public-workflow comparisons remain
separate. A third, fastPLS-only `phase_decomposition` scope uses opt-in internal
timers to separate preprocessing and cross-covariance construction, estimator
and deflation work, coefficient-path construction, fitted-value construction,
C++ model assembly, R-wrapper overhead, and held-out prediction. Primary
timings run with these timers disabled. Ordinary `pls()` output is unchanged.

Classification comparisons report both requested and effective component
counts. Both implementations use the same effective count after the
centered-one-hot response-rank limit is applied. This prevents an unsupported
null response direction from being timed as if it were an identifiable PLS
component.

## Joint shape validation

The controlled-scaling publication profile retains the prespecified
one-factor sweeps and adds 24 deterministic Latin-hypercube development cases
plus 16 independently generated holdout cases. These vary sample count,
predictor and response dimensions, effective rank, retained components,
requested prefixes, and class count jointly. Automatic, explicit, and
matrix-free routes are compared on each case against the same deterministic
reference. Route-choice accuracy and automatic runtime relative to the faster
qualified route are reported separately for development and holdout cases.

## CPU and accelerator baselines

Every CPU row records the loaded BLAS, requested threads, and threads reported
by `RhpcBLASctl`. The accelerator workstation's original R installation uses a
single-thread reference BLAS. OpenBLAS 0.3.20 was therefore installed in an
isolated user directory and is preloaded only for benchmark workers.
`reference_1`, `optimized_1`, and `optimized_4` profiles are accepted only when
the loaded library and active thread count are verified. CUDA controlled-scaling
times include public-call setup, host-device transfer, synchronization, and
result transfer; warm execution is labelled separately where measured.

## Verification status

The benchmark-only phase timer has a unit test confirming that it is opt-in,
finite, nonnegative, and absent from normal public output. The complete package
test suite passes. `R CMD check --as-cran --no-manual` completes with no errors
or warnings and one local macOS check-directory detritus note.

The corrected MetRef smoke study used five cold processes and 50 warm
iterations per successful short-task profile. fastPLS and `pls::simpls.fit`
both obtained accuracy 0.75 at the shared effective count of 21 components.
The full nine-dataset adaptive comparison and joint-shape CPU/CUDA study are
run from the same isolated source snapshot; their raw rows and summaries must
replace the smoke values in the manuscript when complete.
