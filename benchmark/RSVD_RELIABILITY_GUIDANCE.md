# rSVD reliability and validation

Randomized SVD is an approximate, seed-dependent direction solver. A completed
fit with finite latent factors is not evidence that its predictions, latent
subspace, or coefficients agree with a deterministic SIMPLS fit.

## Failure criteria

For validation against a deterministic CPU IRLBA fit on the same data, split,
preprocessing, component count, and prediction head, fastPLS classifies an
rSVD result as failed when any applicable condition is met:

- relative prediction error is greater than 0.05;
- prediction correlation is less than 0.99;
- a score, projection, or loading subspace angle is greater than 10 degrees;
- classification-label agreement is less than 0.99; or
- the absolute predictive-metric difference is greater than 0.01.

These tolerances are validation criteria, not mathematical error bounds.
Near-boundary scientific conclusions should be confirmed with stricter,
application-specific tolerances.

## Corrected SIMPLS implementation

Each SIMPLS deflation now obtains its leading direction from a fresh,
oversampled randomized range finder applied to the current deflated
cross-covariance operator. The rejected one-vector warm-start and adaptive
refresh shortcut is no longer used by the public CPU, CUDA, or Metal rSVD
paths. Each solve requests exactly one accepted direction; only its workspace,
not a candidate direction, may persist within an accelerator implementation.

In the prespecified CPU validation, oversampling by 10 directions and one power
iteration passed 101 of 117 component-level comparisons. The 16 failures were
concentrated in high-rank-response simulation and MetRef. With the same
oversampling and two power iterations, all 117 comparisons passed. The worst
observed relative prediction error was 0.0332, the minimum prediction
correlation was 0.99939, the maximum score-subspace angle was 4.93 degrees, and
minimum classification-label agreement was 0.99.

A smaller CUDA audit showed that oversampling by 10 directions and four power
iterations, or oversampling by 20 directions with one or more power iterations,
passed all eight audited high-rank regression and MetRef component points.
These results guide settings but are not a universal guarantee.

The release-candidate multi-seed audit subsequently found that oversampling 10
with two power iterations failed 5 of 255 component-level checks across seeds
1, 7, 19, 43, and 123. CPU therefore starts from oversampling 20 with two power
iterations and applies a case-specific audit, adaptive retries, and
deterministic recovery. The wider controlled-shape study showed that CUDA 20/2
was not sufficiently reliable; CUDA now enforces a safety floor of oversampling
48 with four power iterations, which met all 174 CUDA runs across three seeds
in the controlled shape panel. Metal rSVD is
also identified as unqualified until a dedicated audit is available. The
release qualification repeats randomized fits across multiple seeds; a single
fixed seed is insufficient for a general reliability claim. Subsequent wider
shape sweeps showed that panel-level controls do not certify a new matrix.

After adding CPU case auditing, independent-sketch consensus, and deterministic
recovery, all 174 CPU and all 174 CUDA controlled-shape runs met the
prespecified tolerances. The automatic-route subset comprised 138 CPU and 138
CUDA runs, resolving all 73 failures in the former 276-run automatic panel.
CPU recovery was used in 30 of 174 runs and is reported explicitly; CUDA's
174/174 result is panel evidence rather than a per-fit certificate.

## Per-fit safeguards

CPU float64 rSVD now orthonormalizes both sides of every subspace iteration and
audits every requested decomposition. The audit checks normalized left and
right singular-triplet residuals and searches the orthogonal complement for a
missed direction. An omitted-to-retained boundary ratio above 0.95 is treated
as weak separation: even an accurate randomized subspace may then choose a
seed-dependent direction that differs from deterministic sequential SIMPLS.
The solver automatically retries with at least 32 oversampling directions and
three power iterations, then 48 and four. For a weak boundary, two independent
strengthened sketches must agree on the retained subspace and singular values.
If neither the direct audit nor this consensus criterion is met, CPU execution
replaces only that direction with IRLBA.
Matrix-free SIMPLS applies the same rule. The recovery and effective controls
are recorded in `diagnostics$rsvd$case_audit`; they are not silent.

Routes that do not yet expose this case-specific certificate are marked
unverified and emit a warning. A successful historical panel is reported as
validation evidence only, never as a universal guarantee.

## Choosing IRLBA

Use deterministic CPU IRLBA for confirmatory inference, coefficient or loading
interpretation, ill-conditioned or rank-deficient matrices, slowly decaying
singular spectra, unstable results across rSVD seeds, or any rSVD result that
fails the criteria above. Use rSVD when speed or accelerator execution is
important and its approximation has been audited on the target task.

`pls()` returns `diagnostics`. For rSVD,
`basic_checks_passed_approximation_not_audited` means only that structural
checks passed; it must not be interpreted as agreement with IRLBA. `fastsvd()`
also reports singular-triplet residual and orthogonality diagnostics for
double-precision inputs.
