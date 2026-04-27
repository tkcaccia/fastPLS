\name{pls}
\alias{pls}
\title{Partial Least Squares With Explicit Algorithm and SVD Backend Control}

\description{
Fit PLS models for regression or classification using C++ implementations of
PLS-SVD (\code{method = "plssvd"}) or the optimized SIMPLS implementation
(\code{method = "simpls"}). \code{method = "simpls_fast"} is kept as a legacy
alias for the same optimized SIMPLS code path.
}

\usage{
pls(
  Xtrain,
  Ytrain,
  Xtest = NULL,
  Ytest = NULL,
  ncomp = 2,
  scaling = c("centering", "autoscaling", "none"),
  method = c("simpls", "plssvd", "simpls_fast"),
  svd.method = c("irlba", "cpu_rsvd"),
  rsvd_oversample = 10L,
  rsvd_power = 1L,
  svds_tol = 0,
  irlba_work = 0L,
  irlba_maxit = 1000L,
  irlba_tol = 1e-05,
  irlba_eps = 1e-09,
  irlba_svtol = 1e-05,
  seed = 1L,
  fast_block = 1L,
  fast_center_t = FALSE,
  fast_reorth_v = FALSE,
  fast_incremental = TRUE,
  fast_inc_iters = 2L,
  fast_defl_cache = TRUE,
  fit = FALSE,
  proj = FALSE,
  perm.test = FALSE,
  times = 100
)
}

\arguments{
\item{Xtrain}{Numeric predictor matrix (rows: samples, columns: variables).}

\item{Ytrain}{Response. Numeric vector/matrix for regression, or factor for classification.
For factor responses, the code converts labels to one-hot columns via
\code{transformy()}.}

\item{Xtest}{Optional test predictor matrix. If provided, predictions are returned.}

\item{Ytest}{Optional test response used only to compute \code{Q2Y} in prediction output.}

\item{ncomp}{Requested number of components (scalar or vector). For
\code{method = "plssvd"}, effective rank is capped at
\code{min(nrow(Xtrain), ncol(Xtrain), ncol(Ytrain_transformed))}.}

\item{scaling}{Preprocessing of \code{Xtrain}:
\code{"centering"} (subtract column means),
\code{"autoscaling"} (center and divide by sample SD), or
\code{"none"}.}

\item{method}{PLS algorithm:
\itemize{
\item \code{"plssvd"}: single truncated SVD of cross-covariance
\eqn{S = X^T Y}, then component-wise reconstruction of regression coefficients.
\item \code{"simpls"}: optimized SIMPLS-like deflation using the fixed
incremental-deflation profile formerly exposed as \code{"simpls_fast"}.
\item \code{"simpls_fast"}: legacy alias for \code{"simpls"}.
}
}

\item{svd.method}{Backend for truncated SVD calls in the CPU path. The former
hybrid CUDA route via \code{svd.method = "cuda_rsvd"} has been removed from
\code{pls()}; use \code{\link{simpls_gpu}} for the experimental GPU-native fit.
\itemize{
\item \code{"irlba"}: in \code{pls_model1/2}, uses legacy IRLB path only when the
cross-covariance matrix is larger than \code{5 x 5}; otherwise falls back to exact CPU SVD.
\item \code{"cpu_rsvd"}: approximate randomized SVD on CPU.
}}

\item{rsvd_oversample}{Oversampling parameter for randomized SVD.
Effective sampled subspace size is approximately \code{k + rsvd_oversample}.}

\item{rsvd_power}{Number of randomized power iterations for RSVD backends.}

\item{svds_tol}{Reserved backend tolerance placeholder.}

\item{irlba_work}{IRLBA working subspace size. Used when legacy \code{"irlba"}
path is active in C++ PLS loops. \code{0} keeps the internal default.}

\item{irlba_maxit}{IRLBA maximum iterations.}

\item{irlba_tol}{IRLBA convergence tolerance.}

\item{irlba_eps}{IRLBA invariant subspace detection tolerance.}

\item{irlba_svtol}{IRLBA singular-value convergence tolerance.}

\item{seed}{Random seed used by randomized SVD backends.}

\item{fast_block}{Refresh block size for the optimized \code{simpls} path.}

\item{fast_center_t}{Deprecated and ignored. \code{simpls} now permanently uses
the optimized former incremental-deflation profile.}

\item{fast_reorth_v}{Deprecated and ignored. \code{simpls} now permanently uses
the optimized former incremental-deflation profile.}

\item{fast_incremental}{Deprecated and ignored. \code{simpls} now permanently uses
the optimized former incremental-deflation profile.}

\item{fast_inc_iters}{Deprecated and ignored. \code{simpls} now permanently uses
the optimized former incremental-deflation profile.}

\item{fast_defl_cache}{Deprecated and ignored. \code{simpls} now permanently uses
the optimized former incremental-deflation profile.}

\item{fit}{If \code{TRUE}, return in-sample fitted values and \code{R2Y}.}

\item{proj}{If \code{TRUE} and \code{Xtest} is provided, return projected test scores
\code{Ttest}.}

\item{perm.test}{If \code{TRUE}, run permutation testing.}

\item{times}{Number of permutations when \code{perm.test = TRUE}.}
}

\details{
\strong{Algorithmic structure in this implementation}

\strong{PLS-SVD (\code{plssvd}):}
\itemize{
\item Forms centered/scaled \eqn{X} and centered \eqn{Y}, then computes truncated SVD of
\eqn{S = X^T Y}.
\item Uses leading singular subspaces to build \eqn{R}, \eqn{Q}, scores \eqn{T = XR}, and
component-wise coefficient tensors \eqn{B(:,:,a)}.
\item One-shot SVD step plus coefficient reconstruction (not iterative deflation per component).
}

\strong{SIMPLS (\code{simpls}):}
\itemize{
\item Preserves SIMPLS-like deflation and cumulative \eqn{B} updates.
\item Uses the fixed optimized incremental-deflation profile that was previously
benchmarked as the incdefl variant.
\item \code{simpls_fast} remains accepted only as a compatibility alias.
}

\strong{Exact vs approximate backends}
\itemize{
\item \code{cpu_rsvd}: approximate RSVD path (seeded and reproducible for fixed
seed/parameters).
\item GPU execution is now exposed separately through \code{\link{simpls_gpu}}
rather than through \code{svd.method}.
\item \code{irlba}: legacy IRLB branch used only in specific PLS code paths and matrix sizes
(as described above).
}

\strong{Classification handling}
\itemize{
\item Factor \code{Ytrain} is converted to one-hot columns.
\item Prediction returns class labels by \code{argmax} over predicted response columns,
with levels stored in \code{lev}.
}
}

\value{
An object of class \code{fastPLS}.
Important components include:
\itemize{
\item \code{B}: coefficient tensor \eqn{p \times m \times length(ncomp)}.
\item \code{Q}, \code{R}: latent Y-loadings and X weight/direction matrices.
\item \code{P}: X-loadings (available for \code{simpls}; empty for \code{plssvd} and
\code{simpls_fast} in current C++ implementation).
\item \code{Ttrain}: training scores (available for \code{plssvd}/\code{simpls}; empty for
\code{simpls_fast} in current C++ implementation).
\item \code{mX}, \code{vX}, \code{mY}: preprocessing statistics.
\item \code{Yfit}, \code{R2Y}: present when \code{fit = TRUE}.
\item \code{Ypred}, \code{Q2Y}, \code{Ttest}: present when \code{Xtest} is provided.
\item \code{classification}, \code{lev}: classification metadata.
}
}

\references{
de Jong, S. (1993). SIMPLS: an alternative approach to partial least squares regression.
\emph{Chemometrics and Intelligent Laboratory Systems}, 18(3), 251--263.\cr

Halko, N., Martinsson, P.-G., and Tropp, J. A. (2011).
Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions.
\emph{SIAM Review}, 53(2), 217--288.\cr

Musco, C. and Musco, C. (2015).
Randomized block Krylov methods for stronger and faster approximate singular value decomposition.
\emph{arXiv:1504.05477}.\cr

Baglama, J. and Reichel, L. (2005).
Augmented implicitly restarted Lanczos bidiagonalization methods.
\emph{SIAM Journal on Scientific Computing}, 27(1), 19--42.\cr

Wegelin, J. A. (2000).
A survey of Partial Least Squares (PLS) methods, with emphasis on the two-block case.
Technical report, University of Washington.
}

\seealso{\code{\link{simpls_gpu}}, \code{\link{pls_r}}, \code{\link{predict.fastPLS}}, \code{\link{svd_run}}, \code{\link{svd_methods}}}

\examples{
set.seed(1)
data(iris)
X <- as.matrix(iris[, -5])
y <- iris[, 5]
ss <- sample(seq_len(nrow(X)), 30)

fit <- pls(X[-ss, ], y[-ss], ncomp = c(2, 3), method = "simpls", svd.method = "irlba")
pred <- predict(fit, X[ss, ], Ytest = y[ss])
}
