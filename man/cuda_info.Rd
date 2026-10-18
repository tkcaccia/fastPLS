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

Hardware and shared-library diagnostics come from \pkg{gpuinfo}.
\code{hardware_available} reports its independent CUDA driver probe and does
not override the runtime result from the libraries linked into fastPLS.
The \code{libraries} table includes \code{required_by_fastPLS} to identify
the CUDA driver, runtime, cuBLAS, cuSOLVER, and cuRAND. cuFFT and NCCL are not
required. A missing separately discoverable library does not disable a
working fastPLS binary, which can use linked or bundled libraries.
}
\value{
A named list containing \code{status}, \code{compiled}, \code{available},
\code{diagnostic_only}, \code{device_count}, \code{runtime_version},
\code{driver_version}, \code{no_cpu_fallback}, and
\code{hardware_available}, and a \code{libraries} diagnostic table.
\code{available} and \code{status} describe this fastPLS build and its own
CUDA runtime probe.
}
\seealso{\code{\link[gpuinfo:cuda_info]{gpuinfo::cuda_info()}},
\code{\link[gpuinfo:cuda_libraries]{gpuinfo::cuda_libraries()}},
\code{\link{pls}}}
\examples{
fastPLS::cuda_info()
}
