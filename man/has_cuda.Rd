\name{has_cuda}
\alias{has_cuda}
\title{Check CUDA Runtime Availability for fastPLS}

\description{
Report whether CUDA RSVD can be used in the current session.
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

If \code{FALSE}, requesting \code{svd.method = "cuda_rsvd"} in high-level APIs
raises an error.
}

\value{A length-1 logical.}

\seealso{\code{\link{svd_methods}}, \code{\link{pls}}}

\examples{
has_cuda()
}
