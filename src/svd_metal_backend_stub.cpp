#include "svd_metal_backend.h"

#include <stdexcept>

namespace fastpls_svd {

bool has_metal_backend() {
  return false;
}

arma::mat metal_matrix_multiply(const arma::mat& A, const arma::mat& B) {
  (void) A;
  (void) B;
  throw std::runtime_error("Metal backend is only available on macOS builds with Apple Metal support");
}

arma::mat metal_matrix_multiply(const arma::mat& A,
                                const arma::mat& B,
                                bool transpose_left,
                                bool transpose_right) {
  (void) A;
  (void) B;
  (void) transpose_left;
  (void) transpose_right;
  throw std::runtime_error("Metal backend is only available on macOS builds with Apple Metal support");
}

arma::mat metal_crossprod(const arma::mat& A, const arma::mat& B) {
  (void) A;
  (void) B;
  throw std::runtime_error("Metal backend is only available on macOS builds with Apple Metal support");
}

Rcpp::List metal_simpls_resident(const arma::mat& X,
                                 const arma::mat& Y,
                                 int ncomp,
                                 int power_iters,
                                 int seed) {
  (void) X;
  (void) Y;
  (void) ncomp;
  (void) power_iters;
  (void) seed;
  throw std::runtime_error("Metal backend is only available on macOS builds with Apple Metal support");
}

} // namespace fastpls_svd
