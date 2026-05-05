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

Package installation was enabled for missing packages. `plsdof` was installed
successfully, but its available PLS fit functions are univariate and did not
accept the 22-column MetRef dummy response directly. `plsVarSel` could not be
loaded because its dependency `praznik` is unavailable for this local R 4.3
setup.

Fastest successful median runtimes:

```text
fastPLS::pls(method="kernelpls"): 10 ms, accuracy 0.80
fastPLS::pls(method="plssvd"):    10 ms, accuracy 0.81
fastPLS::pls(method="simpls"):    10 ms, accuracy 0.80
fastPLS::pls(method="opls"):      21 ms, accuracy 0.79
plsgenomics PLS:                  31 ms, accuracy 0.77
pls::kernelpls.fit:               35 ms, accuracy 0.75
pls::simpls.fit:                  35 ms, accuracy 0.77
plsgenomics PLS-LDA:              37 ms, accuracy 0.88
```

Highest accuracy:

```text
plsgenomics::pls.lda: 0.88 median accuracy, 36 ms median runtime
```

Methods with non-OK rows:

```text
pcv::simpls: fitted but prediction decoder could not extract class scores
plsdepot::simpls: fitted but prediction decoder could not extract class scores
ropls::opls(orthoI=1): OPLS-DA only supports binary classification
ropls::opls(orthoI=0): prediction extraction failed on this multiclass task
```
