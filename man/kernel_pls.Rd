\name{kernel_pls_r}
\alias{kernel_pls_r}
\alias{kernel_pls_cpp}
\alias{kernel_pls_cuda}
\alias{kernel_pls_fast_r}
\alias{kernel_pls_fast_cpp}
\alias{kernel_pls_fast_cuda}
\alias{predict.fastPLSKernel}
\title{Experimental kernel PLS}
\usage{
kernel_pls_r(Xtrain, Ytrain, Xtest = NULL, Ytest = NULL, ncomp = 2,
  scaling = c("centering", "autoscaling", "none"),
  kernel = c("linear", "rbf", "poly"), gamma = NULL, degree = 3L,
  coef0 = 1, method = c("simpls", "simpls_fast"),
  svd.method = c("irlba", "cpu_rsvd"), rsvd_oversample = 10L,
  rsvd_power = 1L, svds_tol = 0, irlba_work = 0L,
  irlba_maxit = 1000L, irlba_tol = 1e-5, irlba_eps = 1e-9,
  irlba_svtol = 1e-5, seed = 1L, fit = FALSE, proj = FALSE)

kernel_pls_cpp(Xtrain, Ytrain, Xtest = NULL, Ytest = NULL, ncomp = 2,
  scaling = c("centering", "autoscaling", "none"),
  kernel = c("linear", "rbf", "poly"), gamma = NULL, degree = 3L,
  coef0 = 1, method = c("simpls", "simpls_fast"),
  svd.method = c("irlba", "cpu_rsvd"), rsvd_oversample = 10L,
  rsvd_power = 1L, svds_tol = 0, irlba_work = 0L,
  irlba_maxit = 1000L, irlba_tol = 1e-5, irlba_eps = 1e-9,
  irlba_svtol = 1e-5, seed = 1L, fit = FALSE, proj = FALSE)

kernel_pls_cuda(Xtrain, Ytrain, Xtest = NULL, Ytest = NULL, ncomp = 2,
  scaling = c("centering", "autoscaling", "none"),
  kernel = c("linear", "rbf", "poly"), gamma = NULL, degree = 3L,
  coef0 = 1, method = c("simpls_fast"),
  rsvd_oversample = 10L, rsvd_power = 1L, svds_tol = 0,
  seed = 1L, fit = FALSE, proj = FALSE, ...)

kernel_pls_fast_r(...)

kernel_pls_fast_cpp(...)

kernel_pls_fast_cuda(...)
}
\arguments{
\item{Xtrain}{Numeric training predictor matrix.}
\item{Ytrain}{Training response, either numeric or a factor.}
\item{Xtest}{Optional numeric test predictor matrix.}
\item{Ytest}{Optional observed test response.}
\item{ncomp}{Number of PLS components.}
\item{scaling}{Predictor preprocessing before kernel construction.}
\item{kernel}{Kernel type: \code{"linear"}, \code{"rbf"}, or \code{"poly"}.}
\item{gamma}{Kernel scale. Defaults to \code{1 / ncol(Xtrain)}.}
\item{degree}{Polynomial kernel degree.}
\item{coef0}{Polynomial kernel offset.}
\item{method}{PLS core to fit on the centered kernel matrix.}
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
Fits PLS on a centered training kernel matrix. The R and C++ variants differ in
kernel construction and PLS core. The CUDA variant builds and centers the kernel
on the host, then calls the CUDA-native SIMPLS-fast PLS core.
\code{kernel_pls_fast_*()} fixes \code{method = "simpls_fast"} for the
SIMPLS-fast kernel PLS core.
}
\value{
A \code{fastPLSKernel} object. Prediction uses the stored training kernel
centering statistics before dispatching to \code{predict.fastPLS()}.
}
\keyword{models}
