#include "svd_metal_backend.h"

#include <RcppArmadillo.h>

// [[Rcpp::export]]
bool has_metal() {
  return fastpls_svd::has_metal_backend();
}

// [[Rcpp::export]]
arma::mat metal_matrix_multiply_cpp(const arma::mat& A, const arma::mat& B) {
  return fastpls_svd::metal_matrix_multiply(A, B);
}

// [[Rcpp::export]]
arma::mat metal_crossprod_cpp(const arma::mat& A, const arma::mat& B) {
  return fastpls_svd::metal_crossprod(A, B);
}

// [[Rcpp::export]]
Rcpp::List metal_simpls_resident_cpp(const arma::mat& X,
                                     const arma::mat& Y,
                                     int ncomp,
                                     int power_iters = 2,
                                     int seed = 1) {
  return fastpls_svd::metal_simpls_resident(X, Y, ncomp, power_iters, seed);
}
