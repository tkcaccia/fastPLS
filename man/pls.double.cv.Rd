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
  svd.method = c("irlba", "dc", "cpu_rsvd", "cuda_rsvd"),
  rsvd_oversample = 10L,
  rsvd_power = 1L,
  seed = 1L,
  fast_block = 4L,
  fast_center_t = FALSE,
  fast_reorth_v = TRUE,
  fast_incremental = FALSE,
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
\item{svd.method}{Inner SVD backend used in fold models.}
\item{rsvd_oversample}{RSVD oversampling parameter.}
\item{rsvd_power}{RSVD power iterations.}
\item{seed}{Seed for randomized backends.}
\item{fast_block}{\code{simpls_fast} block refresh size.}
\item{fast_center_t}{\code{simpls_fast} score-centering toggle.}
\item{fast_reorth_v}{\code{simpls_fast} re-orthogonalization toggle.}
\item{fast_incremental}{\code{simpls_fast} incremental block-power toggle.}
\item{fast_inc_iters}{\code{simpls_fast} incremental power iterations.}
\item{fast_defl_cache}{\code{simpls_fast} cached-deflation toggle.}
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
