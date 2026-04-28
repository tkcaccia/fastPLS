\name{simpls_gpu}
\alias{simpls_gpu}
\alias{pls_gpu}
\title{Experimental GPU-Native SIMPLS-fast Fit}

\description{
Fit the experimental CUDA-oriented \code{simpls_fast} engine. This path keeps
training matrices and the evolving cross-covariance resident on device for most
of the fit while preserving the standard \code{fastPLS} output structure.
}

\usage{
simpls_gpu(
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
  proj = FALSE,
  gpu_device_state = TRUE,
  gpu_qr = FALSE,
  gpu_eig = FALSE,
  gpu_qless_qr = TRUE,
  gpu_finalize_threshold = 32L,
  fast_block = 1L,
  fast_center_t = FALSE,
  fast_reorth_v = FALSE,
  fast_incremental = TRUE,
  fast_inc_iters = 2L,
  fast_defl_cache = TRUE
)
}

\arguments{
\item{Xtrain}{Numeric predictor matrix.}

\item{Ytrain}{Numeric response or factor response. Factor responses are
one-hot encoded internally, like in \code{\link{pls}}.}

\item{Xtest}{Optional test predictor matrix.}

\item{Ytest}{Optional observed test response used to compute \code{Q2Y}.}

\item{ncomp}{Requested number of components (scalar or vector).}

\item{scaling}{Preprocessing of \code{Xtrain}: \code{"centering"},
\code{"autoscaling"}, or \code{"none"}.}

\item{rsvd_oversample}{RSVD oversampling parameter used inside the GPU refresh
logic.}

\item{rsvd_power}{RSVD power iterations used inside the GPU refresh logic.}

\item{svds_tol}{Placeholder tolerance passed through to the backend.}

\item{seed}{Random seed used by the GPU refresh logic.}

\item{fit}{If \code{TRUE}, return in-sample fitted values and \code{R2Y}.}

\item{proj}{If \code{TRUE} and \code{Xtest} is provided, return projected test
scores \code{Ttest}.}

\item{gpu_device_state}{Keep the SIMPLS block-refresh and deflation state
resident on the GPU. This is enabled by default for CUDA SIMPLS.}

\item{gpu_qr}{Use GPU-native QR where supported.}

\item{gpu_eig}{Use GPU-native eigendecomposition where supported.}

\item{gpu_qless_qr}{Use the q-less QR path where supported.}

\item{gpu_finalize_threshold}{Threshold controlling when GPU-side
finalization is attempted.}

\item{fast_block}{Refresh block size for SIMPLS-fast variants.}

\item{fast_center_t}{Deprecated and ignored.}

\item{fast_reorth_v}{Deprecated and ignored.}

\item{fast_incremental}{Deprecated and ignored.}

\item{fast_inc_iters}{Deprecated and ignored.}

\item{fast_defl_cache}{Deprecated and ignored.}
}

\details{
\code{simpls_gpu()} is separate from \code{\link{pls}} and does not change the
existing behavior of \code{method = "simpls_fast"}. It is intended for
benchmarking and experimental GPU work.

Compared with the removed hybrid CUDA-assisted path, this function keeps more
of the SIMPLS-fast loop on device:
\itemize{
\item training \code{X} and \code{Y},
\item the evolving deflated cross-covariance,
\item block refresh and GPU-native QR,
\item GPU-side component-stat and fit updates.
}

The implementation remains experimental and may evolve more rapidly than the
standard \code{pls()} API.

The default configuration uses double-precision training buffers and automatic
workspace reset after each fit.
}

\value{
An object of class \code{fastPLS} with the same main structure returned by
\code{\link{pls}}:
\itemize{
\item \code{B}: coefficient tensor,
\item \code{Q}, \code{R}: latent loading/direction matrices,
\item \code{mX}, \code{vX}, \code{mY}: preprocessing statistics,
\item optional \code{Yfit}, \code{R2Y}, \code{Ypred}, \code{Q2Y}, \code{Ttest}.
}
}

\seealso{\code{\link{pls}}, \code{\link{predict.fastPLS}}, \code{\link{has_cuda}}}

\examples{
\dontrun{
fit <- simpls_gpu(
  Xtrain,
  Ytrain,
  Xtest = Xtest,
  Ytest = Ytest,
  ncomp = 50,
  fit = FALSE
)
}
}
