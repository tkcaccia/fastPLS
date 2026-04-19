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
  svd.method = c("irlba", "arpack", "cpu_rsvd"),
  rsvd_oversample = 10L,
  rsvd_power = 1L,
  svds_tol = 0,
  irlba_work = 0L,
  irlba_maxit = 2000L,
  irlba_tol = 1e-06,
  irlba_eps = 1e-09,
  irlba_svtol = 1e-06,
  seed = 1L,
  fast_block = 8L,
  fast_center_t = FALSE,
  fast_reorth_v = FALSE,
  fast_incremental = TRUE,
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
\item{svd.method}{Inner SVD backend used in each fitted fold model. The former
hybrid CUDA route via \code{svd.method = "cuda_rsvd"} has been removed from
\code{optim.pls.cv()}; use \code{\link{pls_gpu}} for experimental GPU-native
fits outside this CV helper.}
\item{rsvd_oversample}{RSVD oversampling parameter for randomized backends.}
\item{rsvd_power}{RSVD power iterations for randomized backends.}
\item{svds_tol}{Tolerance passed to ARPACK \code{svds()} when
\code{svd.method = "arpack"}.}
\item{irlba_work}{IRLBA working subspace size. \code{0} keeps internal default.}
\item{irlba_maxit}{IRLBA maximum iterations.}
\item{irlba_tol}{IRLBA convergence tolerance.}
\item{irlba_eps}{IRLBA invariant subspace detection tolerance.}
\item{irlba_svtol}{IRLBA singular-value convergence tolerance.}
\item{seed}{Seed for randomized backends.}
\item{fast_block}{Deprecated and ignored. \code{simpls_fast} now permanently uses
the optimized former incremental-deflation profile.}
\item{fast_center_t}{Deprecated and ignored. \code{simpls_fast} now permanently uses
the optimized former incremental-deflation profile.}
\item{fast_reorth_v}{Deprecated and ignored. \code{simpls_fast} now permanently uses
the optimized former incremental-deflation profile.}
\item{fast_incremental}{Deprecated and ignored. \code{simpls_fast} now permanently uses
the optimized former incremental-deflation profile.}
\item{fast_inc_iters}{Deprecated and ignored. \code{simpls_fast} now permanently uses
the optimized former incremental-deflation profile.}
\item{fast_defl_cache}{Deprecated and ignored. \code{simpls_fast} now permanently uses
the optimized former incremental-deflation profile.}
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
