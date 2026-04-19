\name{plssvd_gpu}
\alias{plssvd_gpu}
\title{Experimental GPU-native PLSSVD fit}
\description{
Fit the PLSSVD model with a dedicated CUDA path that keeps the cross-covariance
SVD and downstream latent linear algebra on the GPU, returning the standard
\code{fastPLS} object structure.
}
\usage{
plssvd_gpu(
  Xtrain,
  Ytrain,
  Xtest = NULL,
  Ytest = NULL,
  ncomp = 2,
  scaling = c("centering", "autoscaling", "none"),
  rsvd_oversample = 10L,
  rsvd_power = 1L,
  svds_tol = 0,
  seed = 1L,
  fit = FALSE,
  proj = FALSE
)
}
\arguments{
\item{Xtrain}{Numeric training predictor matrix.}

\item{Ytrain}{Training response (numeric matrix/vector or factor).}

\item{Xtest}{Optional test predictor matrix.}

\item{Ytest}{Optional observed response used to compute \code{Q2Y}.}

\item{ncomp}{Number of components (scalar or vector).}

\item{scaling}{One of \code{"centering"}, \code{"autoscaling"}, \code{"none"}.}

\item{rsvd_oversample}{RSVD oversampling used inside the GPU PLSSVD backend.}

\item{rsvd_power}{RSVD power iterations used inside the GPU PLSSVD backend.}

\item{svds_tol}{Tolerance placeholder passed through to the backend.}

\item{seed}{Random seed.}

\item{fit}{Return fitted values and \code{R2Y} when \code{TRUE}.}

\item{proj}{Return projected \code{Ttest} when \code{TRUE}.}
}
\value{
A \code{fastPLS} object.
}
\details{
\code{plssvd_gpu()} is a GPU-native alternative to
\code{pls(..., method = "plssvd", svd.method = "cpu_rsvd")}. Unlike the former
hybrid CPU/GPU route through \code{svd.method = "cuda_rsvd"}, this function
keeps the PLSSVD fit on the dedicated GPU path and only returns host-side
outputs once the fit is complete.

This function requires a CUDA-enabled \code{fastPLS} build and an available CUDA
device; otherwise it errors.
}
\seealso{
\code{\link{pls}}, \code{\link{pls_gpu}}, \code{\link{predict.fastPLS}},
\code{\link{has_cuda}}
}
\examples{
\dontrun{
fit <- plssvd_gpu(Xtrain, Ytrain, Xtest, Ytest, ncomp = 1:5)
}
}
