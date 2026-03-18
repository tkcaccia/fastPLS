#include "svd_cuda_rsvd.h"

#ifdef FASTPLS_HAS_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace fastpls_svd {
namespace {

void throw_cuda_error(cudaError_t code, const char* where) {
  throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(code));
}

void check_cuda(cudaError_t code, const char* where) {
  if (code != cudaSuccess) {
    throw_cuda_error(code, where);
  }
}

void check_cublas(cublasStatus_t code, const char* where) {
  if (code != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(where) + ": cublas call failed");
  }
}

} // namespace

bool cuda_runtime_available() {
  int n_devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&n_devices);
  return (status == cudaSuccess && n_devices > 0);
}

void cuda_rsvd_sample_y(
  const double* hA,
  int m,
  int n,
  const double* hOmega,
  int l,
  int power_iters,
  double* hY
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }

  double* dA = nullptr;
  double* dOmega = nullptr;
  double* dY = nullptr;
  double* dZ = nullptr;
  cublasHandle_t handle = nullptr;

  check_cublas(cublasCreate(&handle), "cublasCreate");

  try {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&dA), sizeof(double) * static_cast<size_t>(m) * static_cast<size_t>(n)), "cudaMalloc(dA)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&dOmega), sizeof(double) * static_cast<size_t>(n) * static_cast<size_t>(l)), "cudaMalloc(dOmega)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&dY), sizeof(double) * static_cast<size_t>(m) * static_cast<size_t>(l)), "cudaMalloc(dY)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&dZ), sizeof(double) * static_cast<size_t>(n) * static_cast<size_t>(l)), "cudaMalloc(dZ)");

    check_cuda(cudaMemcpy(dA, hA, sizeof(double) * static_cast<size_t>(m) * static_cast<size_t>(n), cudaMemcpyHostToDevice), "cudaMemcpy(A)");
    check_cuda(cudaMemcpy(dOmega, hOmega, sizeof(double) * static_cast<size_t>(n) * static_cast<size_t>(l), cudaMemcpyHostToDevice), "cudaMemcpy(Omega)");

    const double alpha = 1.0;
    const double beta = 0.0;

    check_cublas(
      cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, l, n, &alpha, dA, m, dOmega, n, &beta, dY, m),
      "cublasDgemm(A*Omega)"
    );

    for (int i = 0; i < power_iters; ++i) {
      check_cublas(
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, l, m, &alpha, dA, m, dY, m, &beta, dZ, n),
        "cublasDgemm(A^T*Y)"
      );
      check_cublas(
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, l, n, &alpha, dA, m, dZ, n, &beta, dY, m),
        "cublasDgemm(A*Z)"
      );
    }

    check_cuda(cudaMemcpy(hY, dY, sizeof(double) * static_cast<size_t>(m) * static_cast<size_t>(l), cudaMemcpyDeviceToHost), "cudaMemcpy(Y)");

    cudaFree(dA);
    cudaFree(dOmega);
    cudaFree(dY);
    cudaFree(dZ);
    cublasDestroy(handle);
  } catch (...) {
    if (dA) cudaFree(dA);
    if (dOmega) cudaFree(dOmega);
    if (dY) cudaFree(dY);
    if (dZ) cudaFree(dZ);
    if (handle) cublasDestroy(handle);
    throw;
  }
}

} // namespace fastpls_svd

#endif
