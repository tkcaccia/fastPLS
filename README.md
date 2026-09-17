# fastPLS

`fastPLS` provides compiled partial least-squares methods for regression and
classification. This page covers installation from CRAN and GitHub on macOS,
Windows, Ubuntu, and Fedora. The package vignette documents models and usage.

The public `method = "simpls"` name covers the fastPLS SIMPLS-family
estimator. Its one-direction route applies the classical sequential
orthogonalization and deflation structure. When an eligible workload uses a
bounded candidate block from one deflated state, the resulting estimator is an
approximate SIMPLS-family variant and is not described as classical de Jong
SIMPLS.

After installation, open the complete platform and accelerator guide with
`vignette("installation", package = "fastPLS")`.

## Install from CRAN

Install the released package with:

```r
install.packages("fastPLS")
```

CRAN binary packages contain the capabilities available on the corresponding
build service. Compile from source on the target computer when a local CUDA
Toolkit, Apple Metal, or a specific OpenBLAS installation must be enabled.

## Install from GitHub

Install `remotes` once if it is not already available:

```r
install.packages("remotes")
```

After installing the operating-system requirements below, install the
development version in a fresh R session:

```r
remotes::install_github(
    "tkcaccia/fastPLS",
    upgrade = "never",
    force = TRUE,
    build_vignettes = TRUE
)
```

## macOS

Install Apple's command-line developer tools:

```sh
xcode-select --install
```

The normal macOS build uses Apple Accelerate for CPU matrix operations and
enables Metal automatically when the required system frameworks are available.
No separate OpenBLAS installation is required or recommended on macOS.

To test OpenBLAS instead of Accelerate, install it with Homebrew:

```sh
brew install openblas pkg-config
```

Then install from a fresh R session:

```r
Sys.setenv(
    FASTPLS_USE_OPENBLAS = "1",
    OPENBLAS_ROOT = system("brew --prefix openblas", intern = TRUE)
)
remotes::install_github(
    "tkcaccia/fastPLS",
    upgrade = "never",
    force = TRUE,
    build_vignettes = TRUE
)
```

## Ubuntu

Install the compiler toolchain and OpenBLAS development files:

```sh
sudo apt update
sudo apt install build-essential gfortran pkg-config libopenblas-dev
```

Require OpenBLAS during installation so a missing library cannot silently use
the BLAS supplied by R:

```r
Sys.setenv(FASTPLS_USE_OPENBLAS = "1")
remotes::install_github(
    "tkcaccia/fastPLS",
    upgrade = "never",
    force = TRUE,
    build_vignettes = TRUE
)
```

Use a current OpenBLAS build compiled for the target processor. The OpenBLAS
release and the CPU kernel selected at runtime can materially affect large
matrix products even when both installations are reported simply as
`"OpenBLAS"`. Distribution packages that predate the processor may select a
generic or older kernel and should not be used for performance measurements
without verification.

## Fedora

Install the compiler toolchain and OpenBLAS development files:

```sh
sudo dnf install gcc gcc-c++ gcc-gfortran make pkgconf-pkg-config openblas-devel
```

Then require OpenBLAS when installing:

```r
Sys.setenv(FASTPLS_USE_OPENBLAS = "1")
remotes::install_github(
    "tkcaccia/fastPLS",
    upgrade = "never",
    force = TRUE,
    build_vignettes = TRUE
)
```

## Windows

Install the version of Rtools matching the installed R version. A standard
CPU-only installation can then be performed from a fresh R session with the R
command shown under **R package installer**.

OpenBLAS is optional. For x86-64 Windows, install MSYS2 and run the following
command in its **UCRT64** terminal:

```sh
pacman -S --needed mingw-w64-ucrt-x86_64-openblas
```

The usual MSYS2 location is `C:/msys64/ucrt64`. Point fastPLS to that static
OpenBLAS installation:

```r
Sys.setenv(
    FASTPLS_USE_OPENBLAS = "1",
    OPENBLAS_ROOT = "C:/msys64/ucrt64"
)
remotes::install_github(
    "tkcaccia/fastPLS",
    upgrade = "never",
    force = TRUE,
    build_vignettes = TRUE
)
```

Restart R before reinstalling an existing Windows build because Windows cannot
replace a package DLL while it is loaded.

Windows ARM64 builds must use libraries compiled for ARM64. The configuration
rejects x86-64 OpenBLAS and CUDA libraries instead of attempting to link them.
When a matching ARM64 OpenBLAS installation is unavailable, the default
`FASTPLS_USE_OPENBLAS=auto` setting uses the BLAS/LAPACK supplied by R.

## Verify the installation

```r
library(fastPLS)
fastPLS_blas()
has_cuda()
has_metal()
```

`fastPLS_blas()` returns a named report containing the backend, library version,
configuration, selected CPU core, parallel runtime, active thread count, and
resolved library where available. Linux and Windows performance runs should
verify that `fastPLS_blas()$backend` is `"OpenBLAS"` and inspect its `version`
and `core`. Use `fastPLS_blas(details = FALSE)` when only the former scalar
backend name is needed. If OpenBLAS is not installed, fastPLS remains
installable and uses the BLAS/LAPACK supplied by R unless
`FASTPLS_USE_OPENBLAS=1` was set.

Reproducible benchmarks must record the resolved OpenBLAS library, its version,
and the value returned by OpenBLAS for the active core. The publication scripts in
[`fastPLS-extra`](https://github.com/tkcaccia/fastPLS-extra) perform this check
before any fastPLS timing stage. Timings obtained with a different or
unverified OpenBLAS build must not be pooled with the verified benchmark.

CUDA is optional on Linux and Windows. A CUDA build additionally requires the
NVIDIA CUDA Toolkit and `CUDA_ROOT`; Metal is available only on macOS. Requests
for an unavailable accelerator return an error and never silently use the CPU.
