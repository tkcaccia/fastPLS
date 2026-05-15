\name{fastsvd}
\alias{fastsvd}
\title{Singular value decomposition through fastPLS backends}
\usage{
fastsvd(
  x,
  nu = NULL,
  nv = NULL,
  ncomp = NULL,
  backend = c("cpu", "cuda", "metal"),
  method = c("irlba", "rsvd"),
  oversample = 10L,
  power = 1L,
  svds_tol = 0,
  work = 0L,
  maxit = 1000L,
  tol = 1e-05,
  eps = 1e-09,
  svtol = 1e-05,
  seed = 1L
)
}
\arguments{
\item{x}{Numeric matrix to decompose, with observations or rows in rows and
variables or columns in columns. Sparse matrices should be converted by the
caller; \code{fastsvd()} currently works on a dense numeric matrix.}

\item{nu}{Number of left singular vectors to return. If \code{NULL}, the function
uses the largest feasible rank implied by the matrix dimensions. When
\code{ncomp} is supplied, \code{ncomp} controls the decomposition rank and
\code{nu} controls only how many left vectors are kept in the returned object.}

\item{nv}{Number of right singular vectors to return. If \code{NULL}, the
function uses the largest feasible rank implied by the matrix dimensions.
When \code{ncomp} is supplied, \code{ncomp} controls the decomposition rank and
\code{nv} controls only how many right vectors are kept in the returned object.}

\item{ncomp}{Optional truncated rank. When supplied, it overrides the rank
implied by \code{nu} and \code{nv}; the final rank is always capped at
\code{min(nrow(x), ncol(x))}.}

\item{backend}{Compute backend. \code{cpu} runs on the host CPU. \code{cuda}
dispatches randomized SVD to the CUDA-native backend and requires
\code{has_cuda()} to be \code{TRUE}. \code{metal} dispatches randomized SVD to
the Apple Metal backend and requires \code{has_metal()} to be \code{TRUE}.}

\item{method}{SVD algorithm family. \code{irlba} uses the bundled iterative
IRLBA-style CPU backend and is valid only with \code{backend = cpu}.
\code{rsvd} uses randomized SVD on the selected backend.}

\item{oversample}{Non-negative oversampling dimension used by randomized SVD.
The sketch dimension is approximately \code{ncomp + oversample}, capped by the
matrix rank. Larger values can improve approximation accuracy at the cost of
extra time and memory.}

\item{power}{Number of randomized-SVD power iterations. Larger values improve
accuracy when singular values decay slowly, but each iteration adds
additional matrix multiplications.}

\item{svds_tol}{Tolerance forwarded to iterative SVD backends. A value of
\code{0} keeps the backend default.}

\item{work}{IRLBA working subspace size. A value of \code{0} lets the bundled
IRLBA backend choose its default workspace.}

\item{maxit}{Maximum number of IRLBA iterations before the CPU IRLBA backend
stops.}

\item{tol}{IRLBA residual convergence tolerance. Smaller values can
improve numerical convergence but may increase runtime.}

\item{eps}{IRLBA orthogonality threshold used internally by the bundled
implementation.}

\item{svtol}{IRLBA singular-value convergence tolerance.}

\item{seed}{Random seed used by randomized backends to generate the Gaussian
sketch. It affects \code{rsvd} results and is ignored by deterministic
backends.}
}
\value{
A list compatible with \code{base::svd()} containing \code{d}, \code{u}, and
\code{v}, plus backend metadata \code{backend}, \code{method},
\code{svd.method}, \code{elapsed}, and \code{ncomp}.
}
\description{
Public interface to the truncated singular value decomposition engines used
internally by fastPLS. The function returns a base-SVD-compatible object from
the same CPU and accelerator implementations used by the PCA and PLS
routines, making it useful for direct low-rank decompositions, backend
checks, and numerical validation without masking \code{base::svd()}.
}
\examples{
set.seed(1)
x <- matrix(rnorm(12 * 5), 12, 5)
s <- fastsvd(x, ncomp = 2, backend = "cpu", method = "rsvd", seed = 1)
s$d

s_irlba <- fastsvd(x, ncomp = 2, backend = "cpu", method = "irlba")
s_irlba$svd.method
}
