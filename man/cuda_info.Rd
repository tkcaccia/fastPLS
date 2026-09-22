\name{cuda_info}
\alias{cuda_info}
\title{Report CUDA Build and Runtime Capability}
\description{
Distinguish a functional CUDA build from a package built without CUDA and from
an explicit diagnostic-only build.
}
\usage{
cuda_info()
}
\details{
A functional result requires both CUDA code compiled into fastPLS and at least
one device reported by the CUDA runtime. fastPLS never substitutes the CPU
backend for an explicit CUDA request. CUDA version integers use the CUDA Runtime
API representation; for example, 13000 denotes CUDA 13.0.
}
\value{
A named list containing \code{status}, \code{compiled}, \code{available},
\code{diagnostic_only}, \code{device_count}, \code{runtime_version},
\code{driver_version}, and \code{no_cpu_fallback}.
}
\seealso{\code{\link{has_cuda}}, \code{\link{pls}}}
\examples{
cuda_info()
}
