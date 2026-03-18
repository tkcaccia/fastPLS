\name{svd_methods}
\alias{svd_methods}
\title{List Available SVD Implementations}
\description{
Returns the SVD implementations wired into \code{fastPLS} and whether each
method is currently enabled.
}
\details{
Method names returned by \code{svd_methods()}:
\itemize{
\item \code{irlba}: legacy truncated IRLBA path used by fastPLS.
\item \code{dc}: deterministic exact CPU SVD via Armadillo \code{svd_econ()}.
\item \code{cpu_rsvd}: CPU randomized SVD.
\item \code{cuda_rsvd}: CUDA randomized SVD (enabled only when CUDA build/runtime are available).
}
}
\usage{
svd_methods()
}
\value{
A data frame with columns:
\itemize{
\item \code{method}: method name.
\item \code{enabled}: logical flag indicating availability in the current session.
}
}
\examples{
svd_methods()
}
