# Development test suite

This directory contains the exhaustive fastPLS validation suite. It is kept
outside source archives by `.Rbuildignore`, so routine `R CMD check` runs only
the compact deterministic suite in `tests/testthat`.

After installing the package from the current source tree, run the complete
suite explicitly with:

```r
source("tests/development/run.R")
```

The development suite includes repeated randomized fits, backend grids,
accelerator tests, permutation studies, and extensive cross-validation parity
checks. Continuous integration should run it separately from CRAN checks.
