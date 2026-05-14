\name{fastsvd}
\alias{fastsvd}
\title{Singular value decomposition through fastPLS backends}
\usage{
fastsvd(
  x,
  nu = NULL,
  nv = NULL,
  ncomp = NULL,
  method = c("irlba", "cpu_rsvd", "cuda_rsvd", "metal_rsvd"),
  rsvd_oversample = 10L,
  rsvd_power = 1L,
  svds_tol = 0,
  seed = 1L
)
}
\arguments{
\item{x}{Numeric matrix.}
\item{nu}{Number of left singular vectors to return.}
\item{nv}{Number of right singular vectors to return.}
\item{ncomp}{Optional truncated rank. When supplied, it overrides the decomposition rank implied by \code{nu} and \code{nv}.}
\item{method}{SVD backend: \code{"irlba"}, \code{"cpu_rsvd"}, \code{"cuda_rsvd"}, or \code{"metal_rsvd"}.}
\item{rsvd_oversample}{Randomized SVD oversampling.}
\item{rsvd_power}{Randomized SVD power iterations.}
\item{svds_tol}{Iterative backend tolerance.}
\item{seed}{Random seed used by randomized backends.}
}
\value{
A list compatible with \code{base::svd()} containing \code{d}, \code{u}, and
\code{v}, plus backend metadata \code{method}, \code{elapsed}, and \code{ncomp}.
}
\description{
User-facing SVD wrapper for the fastPLS bundled IRLBA-style, CPU randomized
SVD, CUDA randomized SVD, and Apple Metal randomized SVD backends. It is
intentionally named \code{fastsvd()} so loading fastPLS does not mask
\code{base::svd()}.
}
