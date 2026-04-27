\name{svd_methods}
\alias{svd_methods}
\title{List SVD Backends Exposed by fastPLS}

\description{
Report the SVD backend names accepted by high-level fastPLS APIs and whether
they are currently usable in the running session.
}

\usage{
svd_methods()
}

\details{
The returned table reflects \code{R/main.R} dispatch identifiers:
\itemize{
\item \code{irlba}: legacy label. In PLS model code paths this can trigger the IRLB branch
for sufficiently large cross-covariance matrices; in generic SVD utility wrappers it maps to
exact CPU SVD dispatch.
\item \code{cpu_rsvd}: CPU randomized SVD.
}

GPU-native fits are exposed separately through \code{\link{simpls_gpu}} and
\code{\link{plssvd_gpu}}. The former mixed CPU/GPU route through
\code{svd.method = "cuda_rsvd"} is no longer part of the public CPU-side SVD API.
}

\value{
A data frame with columns:
\itemize{
\item \code{method}: backend name.
\item \code{enabled}: logical availability flag.
}
}

\seealso{\code{\link{svd_run}}, \code{\link{has_cuda}}, \code{\link{simpls_gpu}},
\code{\link{plssvd_gpu}}, \code{\link{pls}}}

\examples{
svd_methods()
}
