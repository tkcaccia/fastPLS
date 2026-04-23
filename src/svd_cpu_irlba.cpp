#include "svd_iface.h"
#include "fastPLS.h"

#include <algorithm>
#include <cstdlib>
#include <limits>

namespace fastpls_svd {

namespace {

int env_int_or_local(const char* key, int fallback, int lo, int hi) {
  const char* raw = std::getenv(key);
  if (raw == nullptr) return fallback;
  char* endptr = nullptr;
  long v = std::strtol(raw, &endptr, 10);
  if (endptr == raw) return fallback;
  if (v < lo) v = lo;
  if (v > hi) v = hi;
  return static_cast<int>(v);
}

double env_double_or_local(const char* key, double fallback, double lo, double hi) {
  const char* raw = std::getenv(key);
  if (raw == nullptr) return fallback;
  char* endptr = nullptr;
  double v = std::strtod(raw, &endptr);
  if (endptr == raw || !std::isfinite(v)) return fallback;
  if (v < lo) v = lo;
  if (v > hi) v = hi;
  return v;
}

} // namespace

SVDResult truncated_svd_cpu_irlba(const Mat& A, int k, const SVDOptions& opt) {
  SVDResult out;

  const arma::uword max_rank = std::min(A.n_rows, A.n_cols);
  const int rank = std::min<int>(std::max(k, 1), static_cast<int>(max_rank));
  if (rank < 1) {
    return out;
  }

  int work = env_int_or_local("FASTPLS_IRLBA_WORK", 0, 0, static_cast<int>(max_rank));
  if (work <= rank) {
    work = std::max(rank + 7, 8);
  }
  if (work > static_cast<int>(max_rank)) {
    work = static_cast<int>(max_rank);
  }

  const int maxit = env_int_or_local("FASTPLS_IRLBA_MAXIT", 1000, 1, 10000000);
  const double tol = env_double_or_local("FASTPLS_IRLBA_TOL", 1e-5, 0.0, 1.0);
  const double eps = env_double_or_local("FASTPLS_IRLBA_EPS", 1e-9, 0.0, 1.0);
  const double svtol = env_double_or_local("FASTPLS_IRLBA_SVTOL", 1e-5, 0.0, 1.0);

  Rcpp::List res = IRLB(A, rank, work, maxit, tol, eps, svtol);

  out.U = Rcpp::as<arma::mat>(res["u"]);
  out.s = Rcpp::as<arma::vec>(res["d"]);

  if (!opt.left_only) {
    arma::mat V = Rcpp::as<arma::mat>(res["v"]);
    out.Vt = V.t();
  }

  return out;
}

} // namespace fastpls_svd
