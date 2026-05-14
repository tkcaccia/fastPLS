#ifndef FASTPLS_SVD_METAL_BACKEND_H
#define FASTPLS_SVD_METAL_BACKEND_H

#include <RcppArmadillo.h>

namespace fastpls_svd {

bool has_metal_backend();
arma::mat metal_matrix_multiply(const arma::mat& A, const arma::mat& B);
arma::mat metal_matrix_multiply(const arma::mat& A,
                                const arma::mat& B,
                                bool transpose_left,
                                bool transpose_right);
arma::mat metal_crossprod(const arma::mat& A, const arma::mat& B);
Rcpp::List metal_simpls_resident(const arma::mat& X,
                                 const arma::mat& Y,
                                 int ncomp,
                                 int power_iters,
                                 int seed);

} // namespace fastpls_svd

#endif
