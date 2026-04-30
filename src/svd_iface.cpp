#include "svd_iface.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>

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

bool rsvd_block_krylov_enabled() {
  const char* flag = std::getenv("FASTPLS_RSVD_BLOCK_KRYLOV");
  if (flag != nullptr) {
    std::string value(flag);
    for (char& c : value) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (value == "1" || value == "true" || value == "yes" || value == "y") return true;
    if (value == "0" || value == "false" || value == "no" || value == "n") return false;
  }

  const char* raw = std::getenv("FASTPLS_RSVD_VARIANT");
  if (raw == nullptr) return false;
  std::string value(raw);
  for (char& c : value) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return (
    value == "block_krylov" ||
    value == "krylov" ||
    value == "krylov_aware"
  );
}

int rsvd_block_krylov_blocks(int row_dim, int max_rank, int block_cols, int power_iters) {
  if (!rsvd_block_krylov_enabled() || power_iters <= 0 || row_dim <= 0 || max_rank <= 0 || block_cols <= 0) {
    return 1;
  }
  const int requested = std::max(power_iters, 0) + 1;
  const int by_rows = std::max(1, row_dim / block_cols);
  const int by_rank = std::max(1, max_rank / block_cols);
  return std::max(1, std::min(requested, std::min(by_rows, by_rank)));
}

SVDResult truncated_svd(const Mat& A, int k, const SVDOptions& opt, Backend backend) {
  if (k < 1) {
    throw std::runtime_error("truncated_svd: k must be >= 1");
  }

  const arma::uword min_dim = std::min(A.n_rows, A.n_cols);
  if (opt.use_full_svd || min_dim < 6) {
    SVDOptions full_opt = opt;
    full_opt.method = Method::EXACT;
    full_opt.use_full_svd = true;
    return truncated_svd_cpu_exact(A, k, full_opt);
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
