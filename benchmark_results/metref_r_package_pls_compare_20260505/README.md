# MetRef R package PLS comparison, ncomp = 22

This benchmark compares fastPLS 0.3 against independent PLS implementations
available from R packages on the MetRef classification task. The fastPLS block
now expands the public `fastPLS::pls()` options across methods, compiled CPU
SVD backends, and classification decision rules:

```text
methods: plssvd, simpls, opls, kernelpls
C++ SVD: irlba, cpu_rsvd
C++ classifiers: argmax, lda_cpp
CUDA SVD: cuda_rsvd
CUDA classifiers: argmax, lda_cuda
```

Dataset:

```text
/Users/stefano/Documents/GPUPLS/Data/metref_remote_task.RData
train/test: 773/100
p: 375
classes: 22
ncomp: 22
replicates: 3
```

The fastPLS calls use `return_variance = FALSE` so the timing excludes optional
predictor-space variance metadata that other packages do not compute.

Files:

```text
metref_pls_opls_speed_accuracy_ncomp22.csv
metref_pls_opls_speed_accuracy_ncomp22_summary.csv
```

The package-specific runners call each package's own implementation directly.
For packages without a standard prediction method, predictions are reconstructed
from the fitted SIMPLS matrices instead of using a generic decoder.

Fastest successful median runtimes:

```text
fastPLS simpls cpp cpu_rsvd argmax:       9 ms, accuracy 0.80
fastPLS kernelpls cpp cpu_rsvd argmax:    9 ms, accuracy 0.80
fastPLS kernelpls cpp irlba argmax:       9 ms, accuracy 0.80
fastPLS simpls cpp irlba argmax:          9 ms, accuracy 0.80
fastPLS kernelpls cpp cpu_rsvd lda_cpp:  11 ms, accuracy 0.92
fastPLS simpls cpp cpu_rsvd lda_cpp:     11 ms, accuracy 0.92
fastPLS plssvd cpp cpu_rsvd argmax:      11 ms, accuracy 0.83
pls::kernelpls.fit:                      29 ms, accuracy 0.75
pls::simpls.fit:                         34 ms, accuracy 0.77
plsgenomics PLS-LDA:                     36 ms, accuracy 0.88
```

Highest accuracy:

```text
fastPLS kernelpls cpp cpu_rsvd lda_cpp: 0.92 median accuracy, 11 ms median runtime
fastPLS simpls cpp cpu_rsvd lda_cpp: 0.92 median accuracy, 11 ms median runtime
fastPLS opls cpp cpu_rsvd lda_cpp: 0.92 median accuracy, 20 ms median runtime
pcv::simpls: 0.89 median accuracy, 86 ms median runtime
```

Methods with non-OK rows:

```text
fastPLS CUDA rows: skipped locally because fastPLS::has_cuda() is FALSE
ropls::opls(orthoI=1): skipped because ropls OPLS-DA requires binary response
```

CUDA validation on chiamaka:

```text
remote host: chiamaka@137.158.224.178
GPU: NVIDIA GeForce RTX 5060 Ti
fastPLS::has_cuda(): TRUE
result files:
  chiamaka_cuda/metref_pls_opls_speed_accuracy_ncomp22_chiamaka_cuda.csv
  chiamaka_cuda/metref_pls_opls_speed_accuracy_ncomp22_chiamaka_cuda_summary.csv
```

Fastest fastPLS rows on the CUDA-enabled run:

```text
fastPLS plssvd cuda cuda_rsvd argmax:       6 ms, accuracy 0.80
fastPLS plssvd cuda cuda_rsvd lda_cuda:     7 ms, accuracy 0.91
fastPLS plssvd cpp cpu_rsvd argmax:        10 ms, accuracy 0.82
fastPLS kernelpls cuda cuda_rsvd argmax:   11 ms, accuracy 0.75
fastPLS simpls cuda cuda_rsvd argmax:      12 ms, accuracy 0.75
fastPLS simpls cuda cuda_rsvd lda_cuda:    13 ms, accuracy 0.88
fastPLS opls cuda cuda_rsvd lda_cuda:      18 ms, accuracy 0.93
```

The external package rows were mostly skipped on chiamaka because those
comparison packages were not installed there; the full external-package
comparison remains the local CSV above.
