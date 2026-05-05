# MetRef R package PLS comparison, ncomp = 22

This benchmark compares fastPLS 0.3 against independent PLS implementations
available from R packages on the MetRef classification task.

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
fastPLS::pls(method="kernelpls"):  8 ms, accuracy 0.80
fastPLS::pls(method="simpls"):     9 ms, accuracy 0.80
fastPLS::pls(method="plssvd"):    10 ms, accuracy 0.81
fastPLS::pls(method="opls"):      16 ms, accuracy 0.79
pls::kernelpls.fit:               27 ms, accuracy 0.75
plsgenomics PLS:                  29 ms, accuracy 0.77
pls::simpls.fit:                  30 ms, accuracy 0.77
plsgenomics PLS-LDA:              34 ms, accuracy 0.88
pcv::simpls:                      86 ms, accuracy 0.89
```

Highest accuracy:

```text
pcv::simpls: 0.89 median accuracy, 86 ms median runtime
plsgenomics::pls.lda: 0.88 median accuracy, 34 ms median runtime
plsdepot::simpls: 0.87 median accuracy, 1000 ms median runtime
```

Methods with non-OK rows:

```text
ropls::opls(orthoI=1): skipped because ropls OPLS-DA requires binary response
```
