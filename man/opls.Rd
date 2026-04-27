\name{opls_r}
\alias{opls_r}
\alias{opls_cpp}
\alias{opls_cuda}
\alias{opls_fast_r}
\alias{opls_fast_cpp}
\alias{opls_fast_cuda}
\alias{predict.fastPLSOpls}
\title{Experimental orthogonal PLS}
\usage{
opls_r(Xtrain, Ytrain, Xtest = NULL, Ytest = NULL, ncomp = 2,
  north = 1L, scaling = c("centering", "autoscaling", "none"),
  method = c("simpls", "plssvd", "simpls_fast"),
  svd.method = c("irlba", "cpu_rsvd"), rsvd_oversample = 10L,
  rsvd_power = 1L, svds_tol = 0, irlba_work = 0L,
  irlba_maxit = 1000L, irlba_tol = 1e-5, irlba_eps = 1e-9,
  irlba_svtol = 1e-5, seed = 1L, fit = FALSE, proj = FALSE)

opls_cpp(Xtrain, Ytrain, Xtest = NULL, Ytest = NULL, ncomp = 2,
  north = 1L, scaling = c("centering", "autoscaling", "none"),
  method = c("simpls", "plssvd", "simpls_fast"),
  svd.method = c("irlba", "cpu_rsvd"), rsvd_oversample = 10L,
  rsvd_power = 1L, svds_tol = 0, irlba_work = 0L,
  irlba_maxit = 1000L, irlba_tol = 1e-5, irlba_eps = 1e-9,
  irlba_svtol = 1e-5, seed = 1L, fit = FALSE, proj = FALSE)

opls_cuda(Xtrain, Ytrain, Xtest = NULL, Ytest = NULL, ncomp = 2,
  north = 1L, scaling = c("centering", "autoscaling", "none"),
  method = c("plssvd", "simpls_fast"), rsvd_oversample = 10L,
  rsvd_power = 1L, svds_tol = 0, seed = 1L, fit = FALSE,
  proj = FALSE, ...)

opls_fast_r(...)

opls_fast_cpp(...)

opls_fast_cuda(...)
}
\arguments{
\item{Xtrain}{Numeric training predictor matrix.}
\item{Ytrain}{Training response, either numeric or a factor.}
\item{Xtest}{Optional numeric test predictor matrix.}
\item{Ytest}{Optional observed test response.}
\item{ncomp}{Number of predictive PLS components.}
\item{north}{Number of supervised orthogonal components to remove from \code{Xtrain}.}
\item{scaling}{Predictor preprocessing before OPLS filtering.}
\item{method}{PLS core to fit after OPLS filtering.}
\item{svd.method}{CPU SVD backend for the R and C++ variants.}
\item{rsvd_oversample}{RSVD oversampling.}
\item{rsvd_power}{RSVD power iterations.}
\item{svds_tol}{Reserved backend tolerance placeholder.}
\item{irlba_work}{IRLBA working subspace size.}
\item{irlba_maxit}{IRLBA maximum iterations.}
\item{irlba_tol}{IRLBA tolerance.}
\item{irlba_eps}{IRLBA epsilon.}
\item{irlba_svtol}{IRLBA singular-value tolerance.}
\item{seed}{Random seed.}
\item{fast_block}{Refresh block size for SIMPLS-fast variants.}
\item{fit}{Return fitted values when \code{TRUE}.}
\item{proj}{Return projected test scores when \code{TRUE}.}
\item{...}{Additional arguments passed to the selected PLS core.}
}
\description{
Removes supervised orthogonal variation from the predictor matrix, then fits the
requested PLS core on the filtered matrix. The CUDA variant performs OPLS
filtering on the host and dispatches the filtered matrix to the CUDA-native
PLSSVD or SIMPLS-fast core.
\code{opls_fast_*()} fixes \code{method = "simpls_fast"} for the fast OPLS
variant.
}
\value{
A \code{fastPLSOpls} object. Prediction applies the stored OPLS filter to new
data before dispatching to \code{predict.fastPLS()}.
}
\keyword{models}
