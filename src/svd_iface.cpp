#include "svd_iface.h"

#include <algorithm>
#include <stdexcept>

#ifdef FASTPLS_HAS_CUDA
#include "svd_cuda_rsvd.h"
#endif

namespace fastpls_svd {

SVDOptions options_from_method_id(
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  unsigned int seed,
  bool left_only,
  bool use_full_svd
) {
  SVDOptions opt;
  opt.oversample = std::max(rsvd_oversample, 0);
  opt.power_iters = std::max(rsvd_power, 0);
  opt.svds_tol = std::max(svds_tol, 0.0);
  opt.seed = seed;
  opt.left_only = left_only;
  opt.use_full_svd = use_full_svd;

  switch (svd_method) {
    case SVD_METHOD_IRLBA:
      opt.method = Method::IRLBA;
      break;
    case SVD_METHOD_CPU_RSVD:
    case SVD_METHOD_CUDA_RSVD:
      opt.method = Method::RSVD;
      break;
    default:
      opt.method = Method::EXACT;
      break;
  }

  return opt;
}

Backend backend_from_method_id(int svd_method) {
  switch (svd_method) {
    case SVD_METHOD_CUDA_RSVD:
      return Backend::CUDA;
    default:
      return Backend::CPU;
  }
}

bool method_is_legacy_irlba(int svd_method) {
  return (svd_method == SVD_METHOD_IRLBA);
}

SVDResult truncated_svd(const Mat& A, int k, const SVDOptions& opt, Backend backend) {
  if (k < 1) {
    throw std::runtime_error("truncated_svd: k must be >= 1");
  }

  if (backend == Backend::CUDA) {
#ifdef FASTPLS_HAS_CUDA
    if (opt.method == Method::IRLBA) {
      return truncated_svd_cpu_irlba(A, k, opt);
    }
    if (opt.method != Method::RSVD) {
      return truncated_svd_cpu_exact(A, k, opt);
    }
    return truncated_svd_cuda_rsvd(A, k, opt);
#else
    throw std::runtime_error("CUDA backend requested but fastPLS was built without CUDA support");
#endif
  }

#ifdef FASTPLS_HAS_BANDICOOT
  if (backend == Backend::BANDICOOT) {
    // Placeholder for optional Bandicoot backend wiring.
    if (opt.method == Method::IRLBA) {
      return truncated_svd_cpu_irlba(A, k, opt);
    }
    if (opt.method == Method::RSVD) {
      return truncated_svd_cpu_rsvd(A, k, opt);
    }
    return truncated_svd_cpu_exact(A, k, opt);
  }
#endif

  if (opt.method == Method::IRLBA) {
    return truncated_svd_cpu_irlba(A, k, opt);
  }

  if (opt.method == Method::RSVD) {
    return truncated_svd_cpu_rsvd(A, k, opt);
  }

  return truncated_svd_cpu_exact(A, k, opt);
}

bool has_cuda_build() {
#ifdef FASTPLS_HAS_CUDA
  return true;
#else
  return false;
#endif
}

bool has_cuda_backend() {
#ifdef FASTPLS_HAS_CUDA
  return cuda_runtime_available();
#else
  return false;
#endif
}

} // namespace fastpls_svd
