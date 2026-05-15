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

If either condition is not met, the function returns \code{FALSE}. The result
describes backend availability for the current R session; it does not select a
device, allocate GPU memory, or run a test computation.
}

\value{A length-1 logical.}

\seealso{\code{\link{pls}}, \code{\link{fastsvd}}, \code{\link{pca}}}

\examples{
has_cuda()
}
