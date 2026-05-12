# ImageNet DINOv2 PLS Candidate Class-Bias Benchmark

This benchmark reproduces the strongest ImageNet/DINOv2 PLS-only classifier found in the exploration runs.

It is intentionally separate from the public `fastPLS::pls()` prediction API because it needs precomputed PLS score files and standalone CUDA scoring executables.

## Classifier

The classifier operates in the supervised PLS score space:

1. Fit/project DINOv2 features to PLS scores.
2. Score each test sample against CUDA prototype ensembles in PLS score space.
3. Select the top candidate classes from the prototype scores.
4. Rerank uncertain samples with CUDA candidate kNN inside the PLS score space.
5. Apply a small iterative class-count calibration to candidate scores.

This is different from `classifier = "class_bias_cuda"` in `fastPLS::pls()`, which only adds per-class offsets to the direct PLS response scores.

## Reproduced ImageNet Result

Using 1,000,000 ImageNet/DINOv2 training samples, 50,000 test samples, and 300 PLS components:

| classifier | top-1 | top-5 | prototype time | candidate-kNN time | total scoring time |
| --- | ---: | ---: | ---: | ---: | ---: |
| gated candidate kNN + iterative class-count calibration | 0.85962 | 0.96978 | 11.110 s | 6.250 s | 17.360 s |

The grouped L2-normalized training-score cache is now built in row blocks. In the smoke test it stayed near 123 MB RSS, replacing the earlier full-matrix grouping path that reached about 8 GB RSS.

In the full 1,000,000-training-sample run, the scoring script reported a peak RSS of 1.70 GB while reusing the grouped cache.

## Main Script

```sh
scripts/run_imagenet_candidate_class_bias.sh
```

The launcher expects:

```sh
FASTPLS_IMAGENET_PLS_SCORE_DIR=/path/to/results/knn_scores_raw
FASTPLS_IMAGENET_PROTOTYPE_RUN=/path/to/prototype_run
FASTPLS_CUDA_PROTOTYPE_SCORER=/path/to/cuda_prototype_scores_variable_ensemble
FASTPLS_CUDA_CANDIDATE_KNN=/path/to/cuda_pls_candidate_knn_scores
```

Optional:

```sh
FASTPLS_IMAGENET_GROUPED_CACHE_DIR=/path/to/reusable/grouped_l2_cache
FASTPLS_IMAGENET_TRAIN_N=1000000
FASTPLS_IMAGENET_TEST_N=50000
FASTPLS_IMAGENET_NCOMP=300
FASTPLS_IMAGENET_TOP_M=20
FASTPLS_IMAGENET_KNN_K=3
FASTPLS_IMAGENET_TAU=0.1
FASTPLS_IMAGENET_ALPHA=0.75
FASTPLS_IMAGENET_GATE_FRACS=0.6
FASTPLS_IMAGENET_ETAS=0.0025
FASTPLS_IMAGENET_ITERS=5
FASTPLS_IMAGENET_CLIPS=0.05
```

## Chiamaka Stable Cache

The reusable grouped training-score cache for the reproduced 300-component, 1M-training run is:

```text
/home/chiamaka/fastPLS_imagenet_grouped_l2_cache/ncomp300_train1000000
```

It contains:

```text
Ttrain_l2_grouped_f32.bin
class_counts_i32.bin
class_offsets_i32.bin
```

## Outputs

Each run writes:

```text
imagenet_pls_candidate_class_bias.csv
imagenet_pls_candidate_class_bias_best.csv
imagenet_candidate_class_bias_time_accuracy.png
parameters.txt
sessionInfo.txt
```

The result table includes both current RSS and peak RSS columns.
