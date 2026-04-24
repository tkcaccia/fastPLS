#include "svd_iface.h"

#include <algorithm>

namespace fastpls_svd {

SVDResult truncated_svd_cpu_exact(const Mat& A, int k, const SVDOptions& opt) {
  SVDResult out;

  const arma::uword max_rank = std::min(A.n_rows, A.n_cols);
  const arma::uword rank = std::min<arma::uword>(max_rank, static_cast<arma::uword>(std::max(k, 1)));

  if (rank == 0) {
    out.U.reset();
    out.s.reset();
    out.Vt.reset();
    return out;
  }

  arma::mat U;
  arma::vec s;
  arma::mat V;

  // Truncated SVD path via ARPACK-based svds when rank is strict-truncated.
  // Armadillo svds requires rank < min(n_rows, n_cols); fall back otherwise.
  // When use_full_svd is set, callers are explicitly asking for the
  // deterministic full/exact LAPACK route, so do not try ARPACK first.
  bool ok = false;
#if defined(ARMA_USE_ARPACK)
  if (!opt.use_full_svd && rank < max_rank) {
    ok = arma::svds(U, s, V, A, rank, opt.svds_tol);
  }
#endif
  if (!ok) {
    if (opt.left_only) {
      arma::svd_econ(U, s, V, A, "left");
    } else {
      arma::svd_econ(U, s, V, A, "both");
    }
  }

  out.U = U.cols(0, rank - 1);
  out.s = s.subvec(0, rank - 1);
  if (opt.left_only) {
    out.Vt.reset();
  } else {
    out.Vt = V.cols(0, rank - 1).t();
  }
  return out;
}

} // namespace fastpls_svd
