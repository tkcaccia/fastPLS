\name{svd_run}
\alias{svd_run}
\alias{svd_benchmark}
\title{Direct SVD Utility Layer Used by fastPLS}

\description{
Run or benchmark truncated SVD backends through the same low-level dispatch
infrastructure used by the package.
}

\usage{
svd_run(
  A,
  k,
  method = c("arpack", "cpu_rsvd", "irlba", "cuda_rsvd"),
  rsvd_oversample = 10L,
  rsvd_power = 1L,
  seed = 1L,
  left_only = FALSE
)

svd_benchmark(
  A,
  k,
  methods = c("irlba", "arpack", "cpu_rsvd", "cuda_rsvd"),
  reps = 3L,
  rsvd_oversample = 10L,
  rsvd_power = 1L,
  seed = 1L,
  left_only = FALSE
)
}

\arguments{
\item{A}{Numeric matrix.}
\item{k}{Requested truncated rank.}
\item{method}{One method for \code{svd_run}.}
\item{methods}{Method vector for \code{svd_benchmark}.}
\item{reps}{Benchmark repetitions per method.}
\item{rsvd_oversample}{Oversampling for RSVD methods.}
\item{rsvd_power}{Power iterations for RSVD methods.}
\item{seed}{Seed for randomized methods.}
\item{left_only}{If \code{TRUE}, return only left singular vectors and singular values.}
}

\details{
\strong{Backend behavior in current implementation}
\itemize{
\item \code{arpack}: ARPACK-truncated CPU SVD (Armadillo \code{svds} path).
\item \code{cpu_rsvd}: randomized SVD on CPU (Gaussian sketch + optional power iterations +
small exact SVD).
\item \code{cuda_rsvd}: GPU-accelerated randomized sampling (cuBLAS GEMM) followed by CPU QR
and reduced exact SVD finalization.
\item \code{irlba}: accepted label for API consistency, but in these utility wrappers it
currently follows exact CPU dispatch (legacy IRLB branch is not used here).
}

These helpers call \code{truncated_svd_debug()} and are intended for backend
inspection/benchmarking rather than full PLS model fitting.
}

\value{
\code{svd_run} returns a list:
\itemize{
\item \code{U}: left singular vectors (truncated).
\item \code{s}: singular values.
\item \code{Vt}: right singular vectors transposed (empty if \code{left_only = TRUE}).
\item \code{method}: method label used.
\item \code{elapsed}: elapsed wall time in seconds.
}

\code{svd_benchmark} returns a data frame with columns
\code{method}, \code{rep}, \code{elapsed}, \code{status}.
}

\references{
Halko, N., Martinsson, P.-G., and Tropp, J. A. (2011).
\emph{SIAM Review}, 53(2), 217--288.\cr
Musco, C. and Musco, C. (2015). \emph{arXiv:1504.05477}.
}

\examples{
set.seed(1)
A <- matrix(rnorm(200 * 40), 200, 40)

svd_run(A, k = 5, method = "arpack")
svd_benchmark(A, k = 5, methods = c("arpack", "cpu_rsvd"), reps = 2)
}
