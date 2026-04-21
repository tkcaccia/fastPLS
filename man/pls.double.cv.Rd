\name{pls.double.cv}
\alias{pls.double.cv}
\title{Nested Cross-Validation for PLS}

\description{
Runs nested cross-validation (outer assessment + inner component selection)
for \code{method = "simpls"}, \code{method = "plssvd"}, or
\code{method = "simpls_fast"}.
}

\usage{
pls.double.cv(
  Xdata,
  Ydata,
  ncomp = 2,
  constrain = 1:nrow(Xdata),
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
  perm.test = FALSE,
  times = 100,
  runn = 10,
  kfold_inner = 10,
  kfold_outer = 10
)
}

\arguments{
\item{Xdata}{Predictor matrix.}
\item{Ydata}{Response (numeric or factor).}
\item{ncomp}{Candidate component counts. For \code{plssvd}, values are capped
internally by rank constraints.}
\item{constrain}{Grouping vector for constrained splitting.}
\item{scaling}{One of \code{"centering"}, \code{"autoscaling"}, \code{"none"}.}
\item{method}{One of \code{"simpls"}, \code{"plssvd"}, or \code{"simpls_fast"}.}
\item{svd.method}{Inner SVD backend used in fold models. The former hybrid CUDA
route via \code{svd.method = "cuda_rsvd"} has been removed from
\code{pls.double.cv()}; use \code{\link{simpls_gpu}} for experimental GPU-native
fits outside this nested CV helper.}
\item{rsvd_oversample}{RSVD oversampling parameter.}
\item{rsvd_power}{RSVD power iterations.}
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
\item{perm.test}{Run permutation test on median Q2 if \code{TRUE}.}
\item{times}{Permutation count.}
\item{runn}{Number of repeated nested-CV runs.}
\item{kfold_inner}{Inner-fold count.}
\item{kfold_outer}{Outer-fold count.}
}

\details{
For factor responses, labels are transformed to one-hot columns internally and
converted back to class labels by argmax of averaged prediction scores.
}

\value{
A list containing nested-CV outputs, including:
\itemize{
\item \code{results}: per-run raw outputs.
\item \code{Q2Y}, \code{R2Y}: per-run vectors.
\item \code{medianQ2Y}, \code{CI95Q2Y}, \code{medianR2Y}, \code{CI95R2Y}.
\item \code{bcomp}: most frequent selected component.
\item \code{Ypred}: aggregated predictions (class labels for classification).
\item classification summaries when applicable (for example \code{conf}, \code{acc_tot}).
\item permutation outputs when requested (\code{Q2Ysampled}, \code{p.value}).
}
}

\seealso{\code{\link{optim.pls.cv}}, \code{\link{pls}}}

\examples{
\donttest{
set.seed(1)
data(iris)
X <- as.matrix(iris[, -5])
y <- iris[, 5]
res <- pls.double.cv(X, y, ncomp = 2:4)
}
}
