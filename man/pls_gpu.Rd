\name{pls_gpu}
\alias{pls_gpu}
\title{Experimental GPU-Native PLS Fit}

\description{
Fit the experimental CUDA-oriented \code{simpls_fast} engine. This path keeps
training matrices and the evolving cross-covariance resident on device for most
of the fit while preserving the standard \code{fastPLS} output structure.
}

\usage{
pls_gpu(
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
}

\details{
\code{pls_gpu()} is separate from \code{\link{pls}} and does not change the
existing behavior of \code{method = "simpls_fast"}. It is intended for
benchmarking and experimental GPU work.

Compared with the hybrid CUDA-assisted path, this function keeps more of the
SIMPLS-fast loop on device:
\itemize{
\item training \code{X} and \code{Y},
\item the evolving deflated cross-covariance,
\item block refresh and GPU-native QR,
\item GPU-side component-stat and fit updates.
}

The implementation remains experimental and may evolve more rapidly than the
standard \code{pls()} API.
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
fit <- pls_gpu(
  Xtrain,
  Ytrain,
  Xtest = Xtest,
  Ytest = Ytest,
  ncomp = 50,
  fit = FALSE
)
}
}
