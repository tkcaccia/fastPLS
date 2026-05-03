\name{pls.single.cv}
\alias{pls.single.cv}
\title{Single grouped cross-validation for PLS}
\usage{
pls.single.cv(
  Xdata,
  Ydata,
  constrain = NULL,
  ncomp = 2L,
  kfold = 10L,
  scaling = c("centering", "autoscaling", "none"),
  method = c("simpls", "plssvd", "opls", "kernelpls"),
  backend = c("cpp", "r", "cuda"),
  inner.method = c("simpls", "plssvd"),
  svd.method = c("irlba", "cpu_rsvd", "exact"),
  rsvd_oversample = 10L,
  rsvd_power = 1L,
  svds_tol = 0,
  seed = 1L,
  north = 1L,
  kernel = c("linear", "rbf", "poly"),
  gamma = NULL,
  degree = 3L,
  coef0 = 1,
  gaussian_y = FALSE,
  gaussian_y_dim = NULL,
  gaussian_y_seed = seed,
  return_scores = FALSE,
  xprod = NULL,
  ...
)
}
\arguments{
\item{Xdata}{Numeric predictor matrix.}
\item{Ydata}{Response vector, response matrix, or factor.}
\item{constrain}{Optional grouping vector; samples with the same value remain in the same fold.}
\item{ncomp}{Number of components.}
\item{kfold}{Number of folds.}
\item{scaling}{One of \code{"centering"}, \code{"autoscaling"}, or \code{"none"}.}
\item{method}{Model family: \code{"simpls"}, \code{"plssvd"}, \code{"opls"}, or \code{"kernelpls"}.}
\item{backend}{Implementation backend: \code{"cpp"}, \code{"r"}, or \code{"cuda"}.}
\item{inner.method}{Inner PLS core used by OPLS and kernel PLS.}
\item{svd.method}{CPU SVD backend: \code{"irlba"}, \code{"cpu_rsvd"}, or \code{"exact"}.}
\item{rsvd_oversample, rsvd_power, svds_tol, seed}{SVD backend controls.}
\item{north}{Number of orthogonal components for OPLS.}
\item{kernel, gamma, degree, coef0}{Kernel PLS controls.}
\item{gaussian_y, gaussian_y_dim, gaussian_y_seed}{Optional Gaussian response compression controls.}
\item{return_scores}{For classification, keep score predictions when supported.}
\item{xprod}{Use the matrix-free cross-product route where available. \code{NULL} applies fastPLS defaults.}
\item{...}{Additional backend tuning arguments.}
}
\value{
A list with fold assignments, predictions, metrics, status, and backend metadata.
}
\description{
Runs one fixed-component grouped cross-validation through the same high-level
algorithm router used by \code{\link{pls}}.
}
