\name{pls_r}
\alias{pls_r}
\title{Pure-R Reference Implementation of fastPLS Algorithms}

\description{
Pure-R implementation of \code{plssvd} and optimized \code{simpls} paths used
for validation and CPU-only reference runs. \code{simpls_fast} remains accepted
as a legacy alias for \code{simpls}. This function still excludes CUDA backends.
}

\usage{
pls_r(
  Xtrain,
  Ytrain,
  Xtest = NULL,
  Ytest = NULL,
  ncomp = 2,
  scaling = c("centering", "autoscaling", "none"),
  method = c("simpls", "plssvd", "simpls_fast"),
  svd.method = c("irlba", "cpu_rsvd"),
  rsvd_oversample = 10L,
  rsvd_power = 1L,
  svds_tol = 0,
  irlba_work = 0L,
  irlba_maxit = 1000L,
  irlba_tol = 1e-05,
  irlba_eps = 1e-09,
  irlba_svtol = 1e-05,
  seed = 1L,
  fast_block = 1L,
  fit = FALSE,
  proj = FALSE,
  perm.test = FALSE,
  times = 100
)
}

\arguments{
\item{Xtrain, Ytrain, Xtest, Ytest, ncomp, scaling, method, fit, proj, perm.test, times}{
Same semantics as in \code{\link{pls}}, restricted to the pure-R code path.}

\item{svd.method}{SVD option in pure R:
\itemize{
\item \code{"cpu_rsvd"}: randomized SVD implemented in R (Gaussian range finder,
optional power iterations, reduced exact SVD).
\item \code{"irlba"}: bundled internal IRLBA wrapper.
}
}

\item{rsvd_oversample}{RSVD oversampling parameter.}
\item{rsvd_power}{RSVD power iterations.}
\item{svds_tol}{Reserved backend tolerance placeholder.}
\item{irlba_work, irlba_maxit, irlba_tol, irlba_eps, irlba_svtol}{Controls for the
bundled IRLBA wrapper used by the pure-R \code{"irlba"} path.}
\item{seed}{RSVD seed.}
\item{fast_block}{Refresh block size for the optimized SIMPLS path.}
}

\details{
\strong{Implementation note}

\code{pls_r} reproduces the same high-level model structure returned by
\code{pls()} but uses R matrix operations and a pure-R truncated SVD helper.
It is suitable for correctness checks, method comparison, and environments
where the C++ path is not desired. The \code{simpls} pure-R path now mirrors
the fixed optimized incremental-deflation profile formerly exposed as
\code{simpls_fast}.
}

\value{
A \code{fastPLS}-class object with components analogous to \code{\link{pls}}.
}

\references{
de Jong, S. (1993). SIMPLS: an alternative approach to partial least squares regression.
\emph{Chemometrics and Intelligent Laboratory Systems}, 18(3), 251--263.\cr
Halko, N., Martinsson, P.-G., and Tropp, J. A. (2011).
\emph{SIAM Review}, 53(2), 217--288.
}

\seealso{\code{\link{pls}}}
