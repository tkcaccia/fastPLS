# Strict CUDA validation

This directory contains development-only infrastructure for validating a real
CUDA build. It is excluded from the CRAN source archive's normal tests.

`debian13-r46.def` creates an immutable Debian 13 image with R 4.6.0,
OpenBLAS/LAPACK, compilers, vignette tools, and declared R dependencies. It does
not contain or manage an NVIDIA driver. The compatible host driver is exposed
only with Singularity `--nv`; a separately installed CUDA Toolkit can be bound
read-only at `/usr/local/cuda`.

Example execution:

```sh
singularity build --fakeroot fastpls-debian13-r46.sif debian13-r46.def
singularity exec --nv \
  --bind /usr/local/cuda-13.0:/usr/local/cuda:ro \
  --bind /path/to/fastPLS:/work/fastPLS \
  --bind /path/to/evidence:/work/evidence \
  fastpls-debian13-r46.sif \
  /work/fastPLS/tests/development/cuda/run-strict-cuda.sh \
  /work/fastPLS /work/evidence
```

The runner enables `FASTPLS_REQUIRE_CUDA=1`, builds a new source archive,
installs that archive, checks `cuda_info()`, executes a real GPU matrix product
against a CPU reference, runs a CUDA PLS fit, times the compact testthat suite,
and executes `R CMD check --as-cran`. Any missing CUDA component or CPU fallback
causes failure.
