#include "svd_iface.h"

#include <algorithm>
#include <random>

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

SVDResult finalize_rsvd_from_sample(const Mat& A, const Mat& Y, int k, bool left_only) {
  SVDResult out;

  const arma::uword max_rank = std::min(A.n_rows, A.n_cols);
  const arma::uword rank = std::min<arma::uword>(max_rank, static_cast<arma::uword>(std::max(k, 1)));

  arma::mat Q;
  arma::mat R;
  arma::qr_econ(Q, R, Y);

  arma::mat B = Q.t() * A;

  arma::mat Uhat;
  arma::vec s;
  arma::mat V;

  if (left_only) {
    arma::svd_econ(Uhat, s, V, B, "left");
    arma::mat U = Q * Uhat;
    out.U = U.cols(0, rank - 1);
    out.s = s.subvec(0, rank - 1);
    out.Vt.reset();
    return out;
  }

  arma::svd_econ(Uhat, s, V, B, "both");
  arma::mat U = Q * Uhat;

  out.U = U.cols(0, rank - 1);
  out.s = s.subvec(0, rank - 1);
  out.Vt = V.cols(0, rank - 1).t();
  return out;
}

SVDResult truncated_svd_cpu_rsvd(const Mat& A, int k, const SVDOptions& opt) {
  const arma::uword max_rank = std::min(A.n_rows, A.n_cols);
  const arma::uword target = std::min<arma::uword>(max_rank, static_cast<arma::uword>(std::max(k, 1)));
  const arma::uword l = std::min<arma::uword>(max_rank, target + static_cast<arma::uword>(std::max(opt.oversample, 0)));

  if (l >= max_rank) {
    SVDOptions exact_opt = opt;
    exact_opt.method = Method::EXACT;
    return truncated_svd_cpu_exact(A, static_cast<int>(target), exact_opt);
  }

  Mat Omega = gaussian_matrix(A.n_cols, l, opt.seed);
  Mat Y = A * Omega;

  for (int i = 0; i < std::max(opt.power_iters, 0); ++i) {
    Mat Z = A.t() * Y;
    Mat Qz;
    Mat Rz;
    arma::qr_econ(Qz, Rz, Z);
    Y = A * Qz;
  }

  return finalize_rsvd_from_sample(A, Y, static_cast<int>(target), opt.left_only);
}

} // namespace fastpls_svd
