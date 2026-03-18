\name{optim.pls.cv}
\alias{optim.pls.cv}
\title{Component Selection by k-Fold Cross-Validation}

\description{
Selects the number of latent components by cross-validation for
\code{method = "simpls"}, \code{method = "plssvd"}, or
\code{method = "simpls_fast"}.
}

\usage{
optim.pls.cv(
  Xdata,
  Ydata,
  ncomp = 2,
  constrain = NULL,
  scaling = c("centering", "autoscaling", "none"),
  method = c("simpls", "plssvd", "simpls_fast"),
  svd.method = c("irlba", "arpack", "cpu_rsvd", "cuda_rsvd"),
  rsvd_oversample = 10L,
  rsvd_power = 1L,
  seed = 1L,
  fast_block = 4L,
  fast_center_t = FALSE,
  fast_reorth_v = TRUE,
  fast_incremental = FALSE,
  fast_inc_iters = 2L,
  fast_defl_cache = TRUE,
  kfold = 10
)
}

\arguments{
\item{Xdata}{Predictor matrix.}
\item{Ydata}{Response (numeric for regression, factor for classification).}
\item{ncomp}{Candidate component counts. For \code{plssvd}, values are internally
capped to \code{min(nrow(Xdata), ncol(Xdata), ncol(Y_transformed))}.}
\item{constrain}{Optional grouping vector for constrained fold assignment.}
\item{scaling}{One of \code{"centering"}, \code{"autoscaling"}, \code{"none"}.}
\item{method}{One of \code{"simpls"}, \code{"plssvd"}, or \code{"simpls_fast"}.}
\item{svd.method}{Inner SVD backend used in each fitted fold model.}
\item{rsvd_oversample}{RSVD oversampling parameter for randomized backends.}
\item{rsvd_power}{RSVD power iterations for randomized backends.}
\item{seed}{Seed for randomized backends.}
\item{fast_block}{\code{simpls_fast} block refresh size.}
\item{fast_center_t}{\code{simpls_fast} score-centering toggle.}
\item{fast_reorth_v}{\code{simpls_fast} re-orthogonalization toggle.}
\item{fast_incremental}{\code{simpls_fast} incremental block-power toggle.}
\item{fast_inc_iters}{\code{simpls_fast} incremental power iterations.}
\item{fast_defl_cache}{\code{simpls_fast} cached-deflation toggle.}
\item{kfold}{Number of folds.}
}

\details{
If \code{Ydata} is a factor, it is converted to one-hot columns internally.
The function delegates fold fitting to C++ cross-validation code and returns
Q2-based component selection statistics.
}

\value{
A list with:
\itemize{
\item \code{optim_comp}: selected component count (Q2-maximizing).
\item \code{Ypred}: fold-wise cross-validated predictions.
\item \code{Q2Y}: Q2 values for tested component counts.
\item \code{R2Y}: in-sample R2 values from full-data fits per component.
\item \code{fold}: assigned fold index per sample.
}
}

\seealso{\code{\link{pls}}, \code{\link{pls.double.cv}}}

\examples{
\donttest{
set.seed(1)
data(iris)
X <- as.matrix(iris[, -5])
y <- iris[, 5]
res <- optim.pls.cv(X, y, ncomp = 2:4)
res$optim_comp
}
}
