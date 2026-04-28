\name{plssvd_cv_cpp}
\alias{plssvd_cv_cpp}
\alias{simpls_cv_cpp}
\alias{simpls_fast_cv_cpp}
\alias{plssvd_cv_cuda}
\alias{simpls_fast_cv_cuda}
\title{Fast grouped PLS cross-validation for compiled backends}
\usage{
plssvd_cv_cpp(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
  scaling = c("centering", "autoscaling", "none"),
  svd.method = c("cpu_rsvd", "irlba"), xprod = TRUE, ...)

simpls_cv_cpp(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
  scaling = c("centering", "autoscaling", "none"),
  svd.method = c("cpu_rsvd", "irlba"), ...)

simpls_fast_cv_cpp(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
  scaling = c("centering", "autoscaling", "none"),
  svd.method = c("cpu_rsvd", "irlba"), xprod = TRUE, ...)

plssvd_cv_cuda(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
  scaling = c("centering", "autoscaling", "none"), xprod = TRUE, ...)

simpls_fast_cv_cuda(Xdata, Ydata, constrain = NULL, ncomp = 2L, kfold = 10L,
  scaling = c("centering", "autoscaling", "none"), xprod = TRUE, ...)
}
\arguments{
\item{Xdata}{Numeric predictor matrix.}
\item{Ydata}{Factor response for classification, or numeric vector/matrix for regression.}
\item{constrain}{Optional grouping vector. Samples with the same value stay in the same fold.}
\item{ncomp}{Number of PLS components.}
\item{kfold}{Number of cross-validation folds.}
\item{scaling}{One of \code{"centering"}, \code{"autoscaling"}, or \code{"none"}.}
\item{svd.method}{CPU SVD backend for Cpp functions.}
\item{xprod}{Use matrix-free cross-product backends where available. Ignored for
\code{svd.method = "irlba"}.}
\item{...}{Additional backend tuning arguments.}
}
\value{
A list with numeric cross-validated \code{Ypred}, decoded \code{pred},
\code{metrics}, \code{fold}, fold \code{status}, effective \code{ncomp},
\code{method}, \code{backend}, and \code{xprod}.
}
\description{
Fixed-component grouped k-fold cross-validation helpers for compiled fastPLS
implementations. They use Cpp or CUDA PLS fits and the optimized compiled
prediction path. Classification returns accuracy; univariate regression returns
Q2; multivariate regression returns RMSD.
}
