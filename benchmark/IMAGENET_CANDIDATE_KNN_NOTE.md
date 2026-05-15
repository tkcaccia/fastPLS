# ImageNet DINOv2 PLS cKNN Benchmark

This benchmark describes the simplified ImageNet/DINOv2 PLS-only classifier
used by `fastPLS::pls(classifier = "cknn")`. cKNN is the short public name for
the candidate-kNN classifier. The implementation is
selected from `backend`, so `backend = "cpu"` uses the compiled C++ path,
`backend = "cuda"` uses the CUDA path, and `backend = "metal"` uses the Metal
fit/projection path where available.

## Classifier

The classifier operates entirely in supervised PLS score space:

1. Fit/project DINOv2 features to PLS scores.
2. Score each test sample against class prototypes in PLS score space.
3. Select the top candidate classes from the prototype scores.
4. Rerank every sample with candidate kNN inside the PLS score space.

The older gated/class-count-calibrated class-bias variants have been removed.
The remaining tuning parameters are:

- `candidate_knn_k`
- `candidate_tau`
- `candidate_alpha`
- `candidate_top_m`

## Current Reduced Tuning Grid

For ImageNet/DINOv2 tuning on Chiamaka, test:

- `ncomp = c(300, 500)`
- `candidate_knn_k = c(3, 5, 10)`
- `candidate_tau = c(0.05, 0.1, 0.2)`
- `candidate_alpha = c(0.5, 0.75, 1)`

For MetRef tuning, use the same grid except `ncomp = 50`.

## Outputs

The reduced tuning scripts should write ranked CSV files with one row per
configuration and at least:

- `dataset`
- `ncomp`
- `knn_k`
- `tau`
- `alpha`
- `top1_accuracy`
- `top5_accuracy` when available
- `score_time_sec`
- `status`
