\name{svd_run}
\alias{svd_run}
\alias{svd_benchmark}
\title{Run and Benchmark Truncated SVD Backends}
\description{
Utilities to run a single truncated SVD call (\code{svd_run}) or benchmark
multiple SVD backends (\code{svd_benchmark}) through the same internal
fastPLS SVD dispatch layer used by PLS.

For \code{"cuda_rsvd"}, randomized range sampling is computed on GPU
(cuBLAS GEMM path). The reduced QR/SVD finalization step currently runs on CPU.
}
\usage{
svd_run(
  A,
  k,
  method = c("dc", "cpu_rsvd", "irlba", "cuda_rsvd"),
  rsvd_oversample = 10L,
  rsvd_power = 1L,
  seed = 1L,
  left_only = FALSE
)

svd_benchmark(
  A,
  k,
  methods = c("irlba", "dc", "cpu_rsvd", "cuda_rsvd"),
  reps = 3L,
  rsvd_oversample = 10L,
  rsvd_power = 1L,
  seed = 1L,
  left_only = FALSE
)
}
\arguments{
\item{A}{Numeric matrix to decompose.}
\item{k}{Requested truncated rank.}
\item{method}{Single method name used by \code{svd_run}.}
\item{methods}{Vector of method names used by \code{svd_benchmark}.}
\item{reps}{Number of repetitions per method in \code{svd_benchmark}.}
\item{rsvd_oversample}{extra random directions used by RSVD beyond the requested rank \code{k} (effective subspace size is approximately \code{k + rsvd_oversample}). Larger values usually improve accuracy but increase compute time and memory. Typical range: 5--30; default \code{10}.}
\item{rsvd_power}{number of power-iteration refinement steps in RSVD. Higher values improve separation of close singular values (better accuracy) but increase runtime; \code{0} is fastest, \code{1} is a common tradeoff (default), and \code{2} is for harder problems.}
\item{seed}{Random seed used for randomized methods.}
\item{left_only}{Whether to compute left singular vectors only.}
}
\value{
\code{svd_run} returns a list containing \code{U}, \code{s}, \code{Vt},
\code{method}, and \code{elapsed}.

\code{svd_benchmark} returns a data frame with one row per method and
repetition containing \code{method}, \code{rep}, \code{elapsed}, and
\code{status}.
}
\examples{
set.seed(1)
A <- matrix(rnorm(200 * 40), 200, 40)

svd_run(A, k = 5, method = "dc")
svd_benchmark(A, k = 5, methods = c("irlba", "dc", "cpu_rsvd"), reps = 2)
}
