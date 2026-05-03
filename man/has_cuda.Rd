\name{has_cuda}
\alias{has_cuda}
\title{Check CUDA Runtime Availability for fastPLS}

\description{
Report whether CUDA-native fastPLS execution can be used in the current session.
}

\usage{
has_cuda()
}

\details{
\code{has_cuda()} returns \code{TRUE} only when both conditions hold:
\itemize{
\item the package was built with CUDA support (compile-time macro
\code{FASTPLS_HAS_CUDA});
\item CUDA runtime reports at least one available device.
}

\code{has_cuda()} is mainly relevant for \code{\link{pls}} with
\code{backend = "cuda"}, and for \code{\link{fastsvd}} or \code{\link{pca}} calls
that request CUDA SVD. The former hybrid CUDA route through
\code{pls(..., svd.method = "cuda_rsvd")} has been removed from the high-level
PLS/CV APIs.
}

\value{A length-1 logical.}

\seealso{\code{\link{pls}}, \code{\link{fastsvd}}, \code{\link{pca}}}

\examples{
has_cuda()
}
