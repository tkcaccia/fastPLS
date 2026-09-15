## Submission

This is an update of fastPLS from CRAN version 0.2 to version 0.3.

The release replaces the former Rcpp/RcppArmadillo implementation with the
package's compiled C++17 core and expands the public methods, validation,
float32 support, and optional accelerator routes. The release has no
Bioconductor package dependency; its examples and vignettes use base R data
and a package-owned synthetic dataset.

The package license has changed from GPL-3 to MIT. The package authors own the
new implementation and have agreed to distribute it under the MIT license.

The maintainer address has changed from stefano.cacciatore@icgeb.org to
tkcaccia@gmail.com. I can confirm this change from the former address if
requested.

## Optional native libraries

The CPU implementation always builds. On Linux and Windows, configuration uses
OpenBLAS when a compatible development installation is found and otherwise
uses the BLAS/LAPACK libraries supplied by R. macOS uses Apple Accelerate by
default.

CUDA is optional on Linux and Windows x86-64 and is compiled only when a
compatible CUDA Toolkit is available. Apple Metal is optional on macOS and is
compiled only when the required system frameworks and a Metal device are
available. An unavailable accelerator is never selected silently at runtime.

We would appreciate CRAN guidance on whether any additional configuration is
recommended so that CRAN-produced Linux or Windows binaries can use OpenBLAS,
and whether CRAN supports any distribution route for optional CUDA-enabled
binaries. Neither OpenBLAS nor CUDA is required for installation or ordinary
CPU use.

## Checks

The source archive was built with R 4.6.0 on macOS arm64.

`R CMD check --as-cran` completed with:

* 0 errors
* 0 warnings
* 1 note

The note reports the maintainer-address change disclosed above.
