\name{pls}
\alias{pls}
\title{Partial Least Squares With Explicit Algorithm and SVD Backend Control}

\description{
Fit PLS models for regression or classification using C++ implementations of
PLS-SVD (\code{method = "plssvd"}), SIMPLS-style deflation
(\code{method = "simpls"}), or an experimental SIMPLS-accelerated variant
(\code{method = "simpls_fast"}).
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
  svd.method = c("irlba", "arpack", "cpu_rsvd", "cuda_rsvd"),
  rsvd_oversample = 10L,
  rsvd_power = 1L,
  svds_tol = 0,
  seed = 1L,
  fast_block = 4L,
  fast_center_t = FALSE,
  fast_reorth_v = TRUE,
  fast_incremental = FALSE,
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
\item \code{"simpls"}: iterative SIMPLS-like deflation of \eqn{S} with one
leading direction extracted per component.
\item \code{"simpls_fast"}: experimental accelerated SIMPLS-like deflation with
block direction refresh and optional incremental/block-power updates.
}
}

\item{svd.method}{Backend for truncated SVD calls:
\itemize{
\item \code{"irlba"}: in \code{pls_model1/2}, uses legacy IRLB path only when the
cross-covariance matrix is larger than \code{5 x 5}; otherwise falls back to exact CPU SVD.
\item \code{"arpack"}: ARPACK-truncated SVD path (Armadillo \code{svds}).
\item \code{"cpu_rsvd"}: approximate randomized SVD on CPU.
\item \code{"cuda_rsvd"}: approximate randomized SVD with GPU GEMM sampling
(available only if \code{\link{has_cuda}()} is TRUE).
}
}

\item{rsvd_oversample}{Oversampling parameter for randomized SVD.
Effective sampled subspace size is approximately \code{k + rsvd_oversample}.}

\item{rsvd_power}{Number of randomized power iterations for RSVD backends.}

\item{svds_tol}{Tolerance passed to ARPACK \code{svds()} when
\code{svd.method = "arpack"}. Larger values can reduce runtime but may loosen
convergence precision. Ignored by non-ARPACK backends.}

\item{seed}{Random seed used by randomized SVD backends.}

\item{fast_block}{\code{simpls_fast} only. Block size for direction refresh cycles.
Mapped to environment variable \code{FASTPLS_FAST_BLOCK}.}

\item{fast_center_t}{\code{simpls_fast} only. Whether to mean-center component scores
before normalization. Mapped to \code{FASTPLS_FAST_CENTER_T}.}

\item{fast_reorth_v}{\code{simpls_fast} only. Whether to apply extra re-orthogonalization
for deflation vectors. Mapped to \code{FASTPLS_FAST_REORTH_V}.}

\item{fast_incremental}{\code{simpls_fast} only. Enable incremental block-power update
mode (warm-started randomized block). Mapped to \code{FASTPLS_FAST_INCREMENTAL}.}

\item{fast_inc_iters}{\code{simpls_fast} only. Iterations for incremental block-power mode.
Mapped to \code{FASTPLS_FAST_INC_ITERS}.}

\item{fast_defl_cache}{\code{simpls_fast} only. Use cached deflation update
\eqn{v^T S} before rank-1 subtraction. Mapped to \code{FASTPLS_FAST_DEFLCACHE}.}

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
\item Iteratively extracts one leading direction from current \eqn{S}.
\item Computes \eqn{t, p, q, v}, orthogonalizes \eqn{v}, and deflates
\eqn{S \leftarrow S - v(v^T S)}.
\item Builds cumulative \eqn{B} as \eqn{RQ^T} at requested components.
}

\strong{SIMPLS fast (\code{simpls_fast}):}
\itemize{
\item Preserves SIMPLS-like deflation and cumulative \eqn{B} updates.
\item Refreshes directions in blocks (\code{fast_block}).
\item Optional incremental randomized block-power update for direction proposals.
\item Experimental; numerical trajectories can differ from \code{simpls} while keeping
similar structure and outputs.
}

\strong{Exact vs approximate backends}
\itemize{
\item \code{arpack}: ARPACK-truncated SVD path.
\item \code{cpu_rsvd}, \code{cuda_rsvd}: approximate RSVD paths (seeded and reproducible
for fixed seed/parameters).
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

\seealso{\code{\link{pls_r}}, \code{\link{predict.fastPLS}}, \code{\link{svd_run}}, \code{\link{svd_methods}}}

\examples{
set.seed(1)
data(iris)
X <- as.matrix(iris[, -5])
y <- iris[, 5]
ss <- sample(seq_len(nrow(X)), 30)

fit <- pls(X[-ss, ], y[-ss], ncomp = c(2, 3), method = "simpls", svd.method = "arpack")
pred <- predict(fit, X[ss, ], Ytest = y[ss])
}
