\name{has_metal}
\alias{has_metal}
\title{Check Apple Metal Backend Availability for fastPLS}

\description{
Report whether Apple Metal-native fastPLS execution can be used in the current
session.
}

\usage{
has_metal()
}

\details{
\code{has_metal()} returns \code{TRUE} only when both conditions hold:
\itemize{
\item the package was built with Apple Metal support;
\item the Metal backend is available in the current session.
}

If either condition is not met, the function returns \code{FALSE}. This includes
non-macOS builds and portable builds that use the Metal stub backend. The result
describes backend availability for the current R session; it does not select a
device, allocate GPU memory, or run a test computation.
}

\value{A length-1 logical.}

\seealso{\code{\link{has_cuda}}, \code{\link{fastsvd}}, \code{\link{pca}}, \code{\link{pls}}}

\examples{
has_metal()
}
