\name{pls_r}
\alias{pls_r}
\title{Pure R PLS (No GPU)}
\usage{
pls_r(Xtrain, Ytrain, Xtest = NULL, Ytest = NULL,
  ncomp = 2,
  scaling = c("centering", "autoscaling", "none"),
  method = c("simpls", "plssvd"),
  svd.method = c("irlba", "dc", "cpu_rsvd"),
  rsvd_oversample = 10L, rsvd_power = 1L, seed = 1L,
  fit = FALSE, proj = FALSE, perm.test = FALSE, times = 100)
}
\description{
Pure R implementation of fastPLS algorithms (\code{plssvd} and \code{simpls})
with the same user-facing input style as \code{pls()}, excluding GPU backends.
}
\arguments{
  \item{Xtrain}{Training predictor matrix.}
  \item{Ytrain}{Training response vector/factor/matrix.}
  \item{Xtest}{Optional test predictor matrix.}
  \item{Ytest}{Optional test response for Q2Y evaluation.}
  \item{ncomp}{Number of components (single value or vector).}
  \item{scaling}{Scaling mode: \code{"centering"}, \code{"autoscaling"}, \code{"none"}.}
  \item{method}{PLS algorithm: \code{"plssvd"} or \code{"simpls"}.}
  \item{svd.method}{SVD backend in pure R.
  \code{"cpu_rsvd"} uses randomized SVD in R with \code{rsvd_oversample}, \code{rsvd_power}, and \code{seed}.
  \code{"irlba"} and \code{"dc"} map to exact base \code{svd()} in this pure-R implementation.}
  \item{rsvd_oversample}{extra random directions used by RSVD beyond the requested rank \code{k} (effective subspace size is approximately \code{k + rsvd_oversample}). Larger values usually improve accuracy but increase compute time and memory. Typical range: 5--30; default \code{10}.}
  \item{rsvd_power}{number of power-iteration refinement steps in RSVD. Higher values improve separation of close singular values (better accuracy) but increase runtime; \code{0} is fastest, \code{1} is a common tradeoff (default), and \code{2} is for harder problems.}
  \item{seed}{Random seed for RSVD and permutation testing.}
  \item{fit}{Compute fitted values on training set.}
  \item{proj}{Return projected test scores.}
  \item{perm.test}{Run permutation test.}
  \item{times}{Number of permutations.}
}
\value{
A \code{fastPLS}-class object with the same output structure used by \code{pls()}.
}
