# rSVD release qualification

This folder records the prespecified numerical qualification used to choose
the fastPLS 0.99.23 randomized-SVD defaults. The candidate default is
`oversample = 20`, `power = 2`; randomized seeds were fixed in advance to
1, 7, 19, 43, and 123.

The CPU panel covered regression and classification, low- and high-rank
responses, `p < n`, `p > n`, ill-conditioned and rank-deficient synthetic
matrices, Breast, Colon, MetRef, and an NMR spectral subset. Deterministic
IRLBA met the estimator-preservation tolerances in 117 of 117 component-level
comparisons with `pls::simpls.fit`. CPU rSVD met the separate approximation
tolerances in 585 of 585 checks across the five randomized seeds.

The CUDA panel covered a high-rank multivariate regression task and MetRef.
CUDA rSVD with the candidate default met the approximation tolerances in 40 of
40 checks across the same five seeds. In contrast, CUDA `(10, 1)` failed all
40 checks and `(10, 2)` failed 13 of 40. A CPU screening run also found five
failures among 255 checks for `(10, 2)`. These rejected settings are therefore
not public defaults.

Passing a panel qualifies the numerical controls, not an arbitrary fitted
model. Fitted objects report the effective controls and qualification panel.
Explicit unaudited controls remain available for research but generate a
warning. Metal rSVD remains explicitly unqualified pending a dedicated
multi-seed panel.

The machine-readable aggregate is `qualified_default_summary.csv`; complete
component-level records and session information are retained under `cpu/` and
`cuda/`.
