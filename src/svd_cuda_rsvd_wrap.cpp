#include "svd_cuda_rsvd.h"

#ifdef FASTPLS_HAS_CUDA

#include <algorithm>
#include <random>
#include <stdexcept>

namespace fastpls_svd {
namespace {

Mat gaussian_matrix(arma::uword n_rows, arma::uword n_cols, unsigned int seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> norm(0.0, 1.0);
  Mat out(n_rows, n_cols);
  for (arma::uword j = 0; j < n_cols; ++j) {
    for (arma::uword i = 0; i < n_rows; ++i) {
      out(i, j) = norm(rng);
    }
  }
  return out;
}

} // namespace

SVDResult truncated_svd_cuda_rsvd(const Mat& A, int k, const SVDOptions& opt) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }

  const arma::uword max_rank = std::min(A.n_rows, A.n_cols);
  const arma::uword target = std::min<arma::uword>(
    max_rank,
    static_cast<arma::uword>(std::max(k, 1))
  );
  const arma::uword l = std::min<arma::uword>(
    max_rank,
    target + static_cast<arma::uword>(std::max(opt.oversample, 0))
  );

  if (l >= max_rank) {
    SVDOptions exact_opt = opt;
    exact_opt.method = Method::EXACT;
    return truncated_svd_cpu_exact(A, static_cast<int>(target), exact_opt);
  }

  Mat Omega = gaussian_matrix(A.n_cols, l, opt.seed);
  Mat Y(A.n_rows, l);

  cuda_rsvd_sample_y(
    A.memptr(),
    static_cast<int>(A.n_rows),
    static_cast<int>(A.n_cols),
    Omega.memptr(),
    static_cast<int>(l),
    std::max(opt.power_iters, 0),
    Y.memptr()
  );

  return finalize_rsvd_from_sample(A, Y, static_cast<int>(target), opt.left_only);
}

} // namespace fastpls_svd

#endif
