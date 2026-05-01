

#include <RcppArmadillo.h>
#include <R_ext/Rdynload.h>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <limits>
#include <random>

#include "fastPLS.h"
#include "svd_iface.h"
#include "svd_cuda_rsvd.h"

extern "C" {
#include "irlba.h"
}

// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

namespace {

fastpls_svd::SVDResult compute_truncated_svd_dispatch(
  const arma::mat& S,
  int k,
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  unsigned int seed,
  bool left_only,
  bool use_full_svd
);

int env_int_or(const char* key, int fallback, int lo, int hi) {
  const char* raw = std::getenv(key);
  if (raw == nullptr) return fallback;
  char* endptr = nullptr;
  long v = std::strtol(raw, &endptr, 10);
  if (endptr == raw) return fallback;
  if (v < lo) v = lo;
  if (v > hi) v = hi;
  return static_cast<int>(v);
}

double env_double_or(const char* key, double fallback, double lo, double hi) {
  const char* raw = std::getenv(key);
  if (raw == nullptr) return fallback;
  char* endptr = nullptr;
  double v = std::strtod(raw, &endptr);
  if (endptr == raw || !std::isfinite(v)) return fallback;
  if (v < lo) v = lo;
  if (v > hi) v = hi;
  return v;
}

bool should_store_coefficients(
  const int p,
  const int m,
  const int n_slices,
  const bool compact_prediction_available
) {
  const char* mode = std::getenv("FASTPLS_STORE_B");
  if (mode != nullptr) {
    std::string value(mode);
    for (char& c : value) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (value == "always" || value == "1" || value == "true" || value == "yes") return true;
    if (value == "never" || value == "0" || value == "false" || value == "no") return false;
  }
  if (!compact_prediction_available) return true;
  const int max_mb = env_int_or("FASTPLS_STORE_B_MAX_MB", 256, 0, 1048576);
  const double b_mb =
    static_cast<double>(p) *
    static_cast<double>(m) *
    static_cast<double>(std::max(n_slices, 1)) *
    static_cast<double>(sizeof(double)) /
    (1024.0 * 1024.0);
  return b_mb <= static_cast<double>(max_mb);
}

void annotate_coefficient_storage(Rcpp::List& out, const bool store_B) {
  out["B_stored"] = store_B;
  out["compact_prediction"] = !store_B;
}

bool is_rsvd_backend_method(const int svd_method) {
  return (
    svd_method == fastpls_svd::SVD_METHOD_CPU_RSVD ||
    svd_method == fastpls_svd::SVD_METHOD_CUDA_RSVD
  );
}

arma::mat gaussian_matrix_local(
  const arma::uword n_rows,
  const arma::uword n_cols,
  const unsigned int seed
) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> norm(0.0, 1.0);

  arma::mat out(n_rows, n_cols);
  double* ptr = out.memptr();
  const arma::uword n_elem = out.n_elem;
  for (arma::uword i = 0; i < n_elem; ++i) {
    ptr[i] = norm(rng);
  }

  return out;
}

arma::vec top1_rsvd_left_vector(
  const arma::mat& S,
  const arma::vec* warm_start,
  const int oversample,
  const int power_iters,
  const unsigned int seed
) {
  if (S.n_rows < 1 || S.n_cols < 1) {
    return arma::vec();
  }

  arma::vec u;
  if (warm_start != nullptr && warm_start->n_elem == S.n_rows) {
    u = *warm_start;
  } else {
    arma::mat omega = gaussian_matrix_local(S.n_cols, 1, seed + static_cast<unsigned int>(std::max(oversample, 0)));
    u = S * omega.col(0);
  }

  double unorm = arma::norm(u, 2);
  if (!std::isfinite(unorm) || unorm <= 0.0) {
    return arma::vec();
  }
  u /= unorm;

  arma::vec v(S.n_cols, arma::fill::zeros);
  for (int it = 0; it < std::max(power_iters, 0); ++it) {
    v = S.t() * u;
    const double vnorm = arma::norm(v, 2);
    if (!std::isfinite(vnorm) || vnorm <= 0.0) {
      return arma::vec();
    }
    v /= vnorm;
    u = S * v;
    unorm = arma::norm(u, 2);
    if (!std::isfinite(unorm) || unorm <= 0.0) {
      return arma::vec();
    }
    u /= unorm;
  }

  return u;
}

arma::vec leading_left_vec_dispatch(
  const arma::mat& S,
  const int svd_method,
  const int rsvd_oversample,
  const int rsvd_power,
  const double svds_tol,
  const unsigned int seed,
  const arma::vec* warm_start
) {
  if (S.n_rows < 1 || S.n_cols < 1) {
    return arma::vec();
  }

  if (svd_method == fastpls_svd::SVD_METHOD_CUDA_RSVD && fastpls_svd::has_cuda_backend()) {
    arma::mat Y0(S.n_rows, 1, arma::fill::zeros);
    if (warm_start != nullptr && warm_start->n_elem == S.n_rows) {
      Y0.col(0) = *warm_start;
    } else {
      Y0 = gaussian_matrix_local(S.n_rows, 1, seed);
    }
    const double init_norm = arma::norm(Y0.col(0), 2);
    if (!std::isfinite(init_norm) || init_norm <= 0.0) {
      return arma::vec();
    }
    Y0.col(0) /= init_norm;

    arma::mat Y(S.n_rows, 1, arma::fill::zeros);
    fastpls_svd::cuda_rsvd_refresh_left_block(
      S.memptr(),
      static_cast<int>(S.n_rows),
      static_cast<int>(S.n_cols),
      Y0.memptr(),
      1,
      std::max(rsvd_power, 0),
      Y.memptr()
    );
    arma::vec u = Y.col(0);
    const double unorm = arma::norm(u, 2);
    if (!std::isfinite(unorm) || unorm <= 0.0) {
      return arma::vec();
    }
    return u / unorm;
  }

  if (is_rsvd_backend_method(svd_method)) {
    const int max_iters = env_int_or("FASTPLS_LEADING_LEFT_MAX_ITERS", std::max(rsvd_power + 2, 4), 1, 64);
    const double tol = std::max(svds_tol, 1e-8);
    arma::vec u;
    if (warm_start != nullptr && warm_start->n_elem == S.n_rows) {
      u = *warm_start;
    } else {
      arma::mat omega = gaussian_matrix_local(S.n_cols, 1, seed + static_cast<unsigned int>(std::max(rsvd_oversample, 0)));
      u = S * omega.col(0);
    }

    double unorm = arma::norm(u, 2);
    if (!std::isfinite(unorm) || unorm <= 0.0) {
      return arma::vec();
    }
    u /= unorm;

    arma::vec v(S.n_cols, arma::fill::zeros);
    for (int it = 0; it < max_iters; ++it) {
      v = S.t() * u;
      const double vnorm = arma::norm(v, 2);
      if (!std::isfinite(vnorm) || vnorm <= 0.0) {
        return arma::vec();
      }
      v /= vnorm;

      arma::vec u_next = S * v;
      const double u_next_norm = arma::norm(u_next, 2);
      if (!std::isfinite(u_next_norm) || u_next_norm <= 0.0) {
        return arma::vec();
      }
      u_next /= u_next_norm;

      const double resid = arma::norm(u_next - u, 2);
      u = std::move(u_next);
      if (resid <= tol) {
        break;
      }
    }
    return u;
  }

  fastpls_svd::SVDResult svd_res = compute_truncated_svd_dispatch(
    S,
    1,
    svd_method,
    rsvd_oversample,
    rsvd_power,
    svds_tol,
    seed,
    true,
    false
  );
  if (svd_res.U.n_cols < 1) {
    return arma::vec();
  }
  return svd_res.U.col(0);
}

bool finalize_left_block_from_bsmall(
  const arma::mat& Bsmall,
  arma::mat& Uhat,
  arma::vec& shat,
  arma::mat& Vhat
) {
  if (Bsmall.n_rows < 1 || Bsmall.n_cols < 1) {
    return false;
  }

  const int eig_threshold = env_int_or("FASTPLS_GPU_FINALIZE_THRESHOLD", 4, 1, 256);
  if (static_cast<int>(Bsmall.n_rows) >= eig_threshold) {
    arma::mat gram = Bsmall * Bsmall.t();
    arma::vec evals;
    arma::mat evecs;
    const bool ok = arma::eig_sym(evals, evecs, gram);
    if (ok && evals.n_elem > 0) {
      arma::uvec ord = arma::sort_index(evals, "descend");
      Uhat = evecs.cols(ord);
      shat = arma::sqrt(arma::clamp(evals(ord), 0.0, std::numeric_limits<double>::infinity()));
      Vhat.reset();
      return true;
    }
  }

  arma::svd_econ(Uhat, shat, Vhat, Bsmall, "left");
  return Uhat.n_cols > 0;
}

fastpls_svd::SVDResult compute_truncated_svd_dispatch(
  const arma::mat& S,
  int k,
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  unsigned int seed,
  bool left_only,
  bool use_full_svd
) {
  fastpls_svd::SVDOptions opt = fastpls_svd::options_from_method_id(
    svd_method,
    rsvd_oversample,
    rsvd_power,
    svds_tol,
    seed,
    left_only,
    use_full_svd
  );

  const fastpls_svd::Backend backend = fastpls_svd::backend_from_method_id(svd_method);
  return fastpls_svd::truncated_svd(S, k, opt, backend);
}

bool plssvd_use_small_exact_svd(const int max_rank) {
  const int threshold = env_int_or("FASTPLS_PLSSVD_SMALL_EXACT_MAX_RANK", 32, 5, 512);
  return max_rank <= threshold;
}

arma::mat numeric_matrix_view(SEXP x, const char* name) {
  if (!Rf_isReal(x) || !Rf_isMatrix(x)) {
    Rcpp::stop("%s must be a numeric matrix", name);
  }
  Rcpp::NumericMatrix rx(x);
  return arma::mat(
    REAL(rx),
    static_cast<arma::uword>(rx.nrow()),
    static_cast<arma::uword>(rx.ncol()),
    false,
    true
  );
}

arma::rowvec variance_nocopy(const arma::mat& x) {
  const int nrow = static_cast<int>(x.n_rows);
  const int ncol = static_cast<int>(x.n_cols);
  arma::rowvec out(ncol);
  for (int j = 0; j < ncol; ++j) {
    double mean_j = 0.0;
    double m2 = 0.0;
    for (int i = 0; i < nrow; ++i) {
      const double xx = x(i, j);
      const double delta = xx - mean_j;
      mean_j += delta / static_cast<double>(i + 1);
      m2 += delta * (xx - mean_j);
    }
    out(j) = std::sqrt(m2 / static_cast<double>(std::max(nrow - 1, 1)));
  }
  return out;
}

struct CenterScaleMatrixView {
  const arma::mat& X;
  arma::rowvec center;
  arma::rowvec scale;

  arma::mat times(const arma::mat& M) const {
    arma::mat Mscaled = M;
    Mscaled.each_col() /= scale.t();
    arma::mat out = X * Mscaled;
    arma::rowvec offset = (center / scale) * M;
    out.each_row() -= offset;
    return out;
  }

  arma::vec times(const arma::vec& v) const {
    arma::vec vscaled = v / scale.t();
    arma::vec out = X * vscaled;
    const double offset = arma::as_scalar((center / scale) * v);
    out -= offset;
    return out;
  }

  arma::mat t_times(const arma::mat& M) const {
    arma::mat out = X.t() * M;
    arma::rowvec sums = arma::sum(M, 0);
    out -= center.t() * sums;
    out.each_col() /= scale.t();
    return out;
  }

  arma::vec t_times(const arma::vec& v) const {
    arma::vec out = X.t() * v;
    out -= center.t() * arma::sum(v);
    out /= scale.t();
    return out;
  }
};

struct CenterOnlyMatrixView {
  const arma::mat& Y;
  arma::rowvec center;

  arma::mat times(const arma::mat& M) const {
    arma::mat out = Y * M;
    arma::rowvec offset = center * M;
    out.each_row() -= offset;
    return out;
  }

  arma::vec times(const arma::vec& v) const {
    arma::vec out = Y * v;
    const double offset = arma::as_scalar(center * v);
    out -= offset;
    return out;
  }

  arma::mat t_times(const arma::mat& M) const {
    arma::mat out = Y.t() * M;
    arma::rowvec sums = arma::sum(M, 0);
    out -= center.t() * sums;
    return out;
  }

  arma::vec t_times(const arma::vec& v) const {
    arma::vec out = Y.t() * v;
    out -= center.t() * arma::sum(v);
    return out;
  }

  arma::mat centered_copy() const {
    arma::mat out = Y;
    out.each_row() -= center;
    return out;
  }
};

fastpls_svd::SVDResult finalize_rsvd_from_q_b_double(
  const arma::mat& Q,
  const arma::mat& B,
  const int k,
  const bool left_only
) {
  fastpls_svd::SVDResult out;
  const arma::uword max_rank = std::min(B.n_rows, B.n_cols);
  const arma::uword rank = std::min<arma::uword>(
    max_rank,
    static_cast<arma::uword>(std::max(k, 1))
  );
  if (rank == 0 || Q.n_cols < 1 || B.n_rows < 1 || B.n_cols < 1) {
    return out;
  }

  arma::mat Uhat;
  arma::vec s;
  arma::mat V;
  if (left_only) {
    arma::svd_econ(Uhat, s, V, B, "left");
  } else {
    arma::svd_econ(Uhat, s, V, B, "both");
  }

  arma::uword actual = std::min<arma::uword>(rank, std::min<arma::uword>(Uhat.n_cols, s.n_elem));
  if (!left_only) {
    actual = std::min<arma::uword>(actual, V.n_cols);
  }
  if (actual == 0) {
    return out;
  }

  out.U = Q * Uhat.cols(0, actual - 1);
  out.s = s.subvec(0, actual - 1);
  if (!left_only) {
    out.Vt = V.cols(0, actual - 1).t();
  }
  return out;
}

fastpls_svd::SVDResult truncated_rsvd_crossprod_double_view(
  const CenterScaleMatrixView& Xop,
  const CenterOnlyMatrixView& Yop,
  const int k,
  const int rsvd_oversample,
  const int rsvd_power,
  const unsigned int seed,
  const bool left_only,
  const bool use_full_svd
) {
  const arma::uword p = Xop.X.n_cols;
  const arma::uword m = Yop.Y.n_cols;
  const arma::uword max_rank = std::min(p, m);
  const arma::uword target = std::min<arma::uword>(
    max_rank,
    static_cast<arma::uword>(std::max(k, 1))
  );
  const arma::uword l = std::min<arma::uword>(
    max_rank,
    target + static_cast<arma::uword>(std::max(rsvd_oversample, 0))
  );

  if (target == 0) {
    return fastpls_svd::SVDResult();
  }

  if (use_full_svd || l >= max_rank) {
    arma::mat S = Xop.t_times(Yop.centered_copy());
    return compute_truncated_svd_dispatch(
      S,
      static_cast<int>(target),
      fastpls_svd::SVD_METHOD_CPU_RSVD,
      rsvd_oversample,
      rsvd_power,
      0.0,
      seed,
      left_only,
      true
    );
  }

  auto a_times = [&](const arma::mat& M) -> arma::mat {
    return Xop.t_times(Yop.times(M));
  };
  auto at_times = [&](const arma::mat& M) -> arma::mat {
    return Yop.t_times(Xop.times(M));
  };

  arma::mat Omega = gaussian_matrix_local(m, l, seed);
  arma::mat Ysample = a_times(Omega);

  const int power_iters = std::max(rsvd_power, 0);
  if (power_iters == 1) {
    Ysample = a_times(at_times(Ysample));
  } else {
    for (int i = 0; i < power_iters; ++i) {
      arma::mat Z = at_times(Ysample);
      arma::mat Qz;
      arma::mat Rz;
      arma::qr_econ(Qz, Rz, Z);
      Ysample = a_times(Qz);
    }
  }

  arma::mat Q;
  arma::mat R;
  arma::qr_econ(Q, R, Ysample);
  if (Q.n_cols < 1) {
    return fastpls_svd::SVDResult();
  }

  arma::mat B = Yop.t_times(Xop.times(Q)).t();
  return finalize_rsvd_from_q_b_double(Q, B, static_cast<int>(target), left_only);
}

fastpls_svd::SVDResult truncated_rsvd_crossprod_double(
  const arma::mat& X,
  const arma::mat& Ymat,
  const int k,
  const int rsvd_oversample,
  const int rsvd_power,
  const unsigned int seed,
  const bool left_only,
  const bool use_full_svd
) {
  const arma::uword p = X.n_cols;
  const arma::uword m = Ymat.n_cols;
  const arma::uword max_rank = std::min(p, m);
  const arma::uword target = std::min<arma::uword>(
    max_rank,
    static_cast<arma::uword>(std::max(k, 1))
  );
  const arma::uword l = std::min<arma::uword>(
    max_rank,
    target + static_cast<arma::uword>(std::max(rsvd_oversample, 0))
  );

  if (target == 0) {
    return fastpls_svd::SVDResult();
  }

  if (use_full_svd || l >= max_rank) {
    arma::mat S = X.t() * Ymat;
    return compute_truncated_svd_dispatch(
      S,
      static_cast<int>(target),
      fastpls_svd::SVD_METHOD_CPU_RSVD,
      rsvd_oversample,
      rsvd_power,
      0.0,
      seed,
      left_only,
      true
    );
  }

  auto a_times = [&](const arma::mat& M) -> arma::mat {
    return X.t() * (Ymat * M);
  };
  auto at_times = [&](const arma::mat& M) -> arma::mat {
    return Ymat.t() * (X * M);
  };

  arma::mat Omega = gaussian_matrix_local(m, l, seed);
  arma::mat Ysample = a_times(Omega);

  const int power_iters = std::max(rsvd_power, 0);
  if (power_iters == 1) {
    Ysample = a_times(at_times(Ysample));
  } else {
    for (int i = 0; i < power_iters; ++i) {
      arma::mat Z = at_times(Ysample);
      arma::mat Qz;
      arma::mat Rz;
      arma::qr_econ(Qz, Rz, Z);
      Ysample = a_times(Qz);
    }
  }

  arma::mat Q;
  arma::mat R;
  arma::qr_econ(Q, R, Ysample);
  if (Q.n_cols < 1) {
    return fastpls_svd::SVDResult();
  }

  arma::mat B = (X * Q).t() * Ymat;
  return finalize_rsvd_from_q_b_double(Q, B, static_cast<int>(target), left_only);
}

arma::mat project_deflated_left_double(
  arma::mat M,
  const arma::mat& V,
  const int n_prev
) {
  if (n_prev > 0) {
    const arma::uword cols = std::min<arma::uword>(
      static_cast<arma::uword>(n_prev),
      V.n_cols
    );
    if (cols > 0) {
      arma::mat Vprev = V.cols(0, cols - 1);
      M -= Vprev * (Vprev.t() * M);
    }
  }
  return M;
}

void project_deflated_left_inplace(
  arma::vec& x,
  const arma::mat& V,
  const int n_prev
) {
  if (n_prev <= 0 || V.n_cols < 1 || x.n_elem != V.n_rows) {
    return;
  }
  const arma::uword cols = std::min<arma::uword>(
    static_cast<arma::uword>(n_prev),
    V.n_cols
  );
  if (cols > 0) {
    arma::mat Vprev = V.cols(0, cols - 1);
    x -= Vprev * (Vprev.t() * x);
  }
}

struct CrossprodIrlbaOperatorData {
  const arma::mat* X = nullptr;
  const arma::mat* Y = nullptr;
  const arma::mat* V = nullptr;
  int n_prev = 0;
  arma::vec sample_tmp;
  arma::vec left_tmp;
};

void crossprod_irlba_mult(char transpose, int m, int n, void* data, double* b, double* c) {
  CrossprodIrlbaOperatorData* op = static_cast<CrossprodIrlbaOperatorData*>(data);
  if (op == nullptr || op->X == nullptr || op->Y == nullptr) {
    return;
  }
  const arma::mat& X = *(op->X);
  const arma::mat& Ymat = *(op->Y);
  if (transpose == 't' || transpose == 'T') {
    arma::vec lhs(b, static_cast<arma::uword>(m), false, true);
    arma::vec out(c, static_cast<arma::uword>(n), false, true);
    if (op->V != nullptr && op->n_prev > 0) {
      op->left_tmp = lhs;
      project_deflated_left_inplace(op->left_tmp, *(op->V), op->n_prev);
      op->sample_tmp = X * op->left_tmp;
    } else {
      op->sample_tmp = X * lhs;
    }
    out = Ymat.t() * op->sample_tmp;
  } else {
    arma::vec rhs(b, static_cast<arma::uword>(n), false, true);
    arma::vec out(c, static_cast<arma::uword>(m), false, true);
    op->sample_tmp = Ymat * rhs;
    out = X.t() * op->sample_tmp;
    if (op->V != nullptr && op->n_prev > 0) {
      project_deflated_left_inplace(out, *(op->V), op->n_prev);
    }
  }
}

fastpls_svd::SVDResult truncated_irlba_crossprod_double(
  const arma::mat& X,
  const arma::mat& Ymat,
  const int k,
  const bool left_only,
  const bool use_full_svd
) {
  fastpls_svd::SVDResult out;

  const arma::uword p = X.n_cols;
  const arma::uword m = Ymat.n_cols;
  const arma::uword max_rank = std::min(p, m);
  const int rank = std::min<int>(std::max(k, 1), static_cast<int>(max_rank));
  if (rank < 1) {
    return out;
  }

  if (use_full_svd || max_rank < 6) {
    arma::mat S = X.t() * Ymat;
    return compute_truncated_svd_dispatch(
      S,
      rank,
      fastpls_svd::SVD_METHOD_IRLBA,
      0,
      0,
      0.0,
      1U,
      left_only,
      true
    );
  }

  int work = env_int_or("FASTPLS_IRLBA_WORK", 0, 0, static_cast<int>(max_rank));
  if (work <= rank) {
    work = std::max(rank + 7, 8);
  }
  if (work > static_cast<int>(max_rank)) {
    work = static_cast<int>(max_rank);
  }

  const int maxit = env_int_or("FASTPLS_IRLBA_MAXIT", 1000, 1, 10000000);
  const double tol = env_double_or("FASTPLS_IRLBA_TOL", 1e-5, 0.0, 1.0);
  const double eps = env_double_or("FASTPLS_IRLBA_EPS", 1e-9, 0.0, 1.0);
  const double svtol = env_double_or("FASTPLS_IRLBA_SVTOL", 1e-5, 0.0, 1.0);

  int iter = 0;
  int mprod = 0;
  int lwork = 7 * work * (1 + work);

  arma::vec s = arma::randn<arma::vec>(rank);
  arma::mat U = arma::randn<arma::mat>(p, work);
  arma::mat Vright = arma::randn<arma::mat>(m, work);
  arma::mat V1 = arma::zeros<arma::mat>(m, work);
  arma::mat U1 = arma::zeros<arma::mat>(p, work);
  arma::mat W = arma::zeros<arma::mat>(p, work);
  arma::vec F = arma::zeros<arma::vec>(m);
  arma::mat B = arma::zeros<arma::mat>(work, work);
  arma::mat BU = arma::zeros<arma::mat>(work, work);
  arma::mat BV = arma::mat(work, work);
  arma::vec BS = arma::zeros<arma::vec>(work);
  arma::vec BW = arma::zeros<arma::vec>(lwork);
  arma::vec res = arma::zeros<arma::vec>(work);
  arma::vec T = arma::zeros<arma::vec>(lwork);
  arma::vec svratio = arma::zeros<arma::vec>(work);

  CrossprodIrlbaOperatorData data;
  data.X = &X;
  data.Y = &Ymat;
  fastpls_irlba_operator op;
  op.mult = &crossprod_irlba_mult;
  op.data = &data;

  irlb(
    nullptr,
    &op,
    2,
    static_cast<int>(p),
    static_cast<int>(m),
    rank,
    work,
    maxit,
    0,
    tol,
    nullptr,
    nullptr,
    nullptr,
    s.memptr(),
    U.memptr(),
    Vright.memptr(),
    &iter,
    &mprod,
    eps,
    lwork,
    V1.memptr(),
    U1.memptr(),
    W.memptr(),
    F.memptr(),
    B.memptr(),
    BU.memptr(),
    BV.memptr(),
    BS.memptr(),
    BW.memptr(),
    res.memptr(),
    T.memptr(),
    svtol,
    svratio.memptr()
  );

  out.U = U.cols(0, static_cast<arma::uword>(rank - 1));
  out.s = s.subvec(0, static_cast<arma::uword>(rank - 1));
  if (!left_only) {
    out.Vt = Vright.cols(0, static_cast<arma::uword>(rank - 1)).t();
  }
  return out;
}

bool refresh_deflated_crossprod_left_irlba_double(
  const arma::mat& X,
  const arma::mat& Ymat,
  const arma::mat& V,
  const int n_prev,
  const int k_block,
  arma::mat& Ublock,
  arma::vec& shat
) {
  const arma::uword p = X.n_cols;
  const arma::uword m = Ymat.n_cols;
  if (p < 1 || m < 1 || k_block < 1) {
    return false;
  }

  const arma::uword max_rank = std::min(p, m);
  const int rank = std::min<int>(std::max(k_block, 1), static_cast<int>(max_rank));
  if (rank < 1) {
    return false;
  }
  if (max_rank < 6) {
    arma::mat S = project_deflated_left_double(X.t() * Ymat, V, n_prev);
    fastpls_svd::SVDResult res = compute_truncated_svd_dispatch(
      S,
      rank,
      fastpls_svd::SVD_METHOD_IRLBA,
      0,
      0,
      0.0,
      1U,
      true,
      true
    );
    Ublock = res.U;
    shat = res.s;
    return Ublock.n_cols > 0;
  }

  int work = env_int_or("FASTPLS_IRLBA_WORK", 0, 0, static_cast<int>(max_rank));
  if (work <= rank) {
    work = std::max(rank + 7, 8);
  }
  if (work > static_cast<int>(max_rank)) {
    work = static_cast<int>(max_rank);
  }

  const int maxit = env_int_or("FASTPLS_IRLBA_MAXIT", 1000, 1, 10000000);
  const double tol = env_double_or("FASTPLS_IRLBA_TOL", 1e-5, 0.0, 1.0);
  const double eps = env_double_or("FASTPLS_IRLBA_EPS", 1e-9, 0.0, 1.0);
  const double svtol = env_double_or("FASTPLS_IRLBA_SVTOL", 1e-5, 0.0, 1.0);

  int iter = 0;
  int mprod = 0;
  int lwork = 7 * work * (1 + work);

  arma::vec s = arma::randn<arma::vec>(rank);
  arma::mat U = arma::randn<arma::mat>(p, work);
  arma::mat Vright = arma::randn<arma::mat>(m, work);
  arma::mat V1 = arma::zeros<arma::mat>(m, work);
  arma::mat U1 = arma::zeros<arma::mat>(p, work);
  arma::mat W = arma::zeros<arma::mat>(p, work);
  arma::vec F = arma::zeros<arma::vec>(m);
  arma::mat B = arma::zeros<arma::mat>(work, work);
  arma::mat BU = arma::zeros<arma::mat>(work, work);
  arma::mat BV = arma::mat(work, work);
  arma::vec BS = arma::zeros<arma::vec>(work);
  arma::vec BW = arma::zeros<arma::vec>(lwork);
  arma::vec res = arma::zeros<arma::vec>(work);
  arma::vec T = arma::zeros<arma::vec>(lwork);
  arma::vec svratio = arma::zeros<arma::vec>(work);

  CrossprodIrlbaOperatorData data;
  data.X = &X;
  data.Y = &Ymat;
  data.V = &V;
  data.n_prev = n_prev;
  fastpls_irlba_operator op;
  op.mult = &crossprod_irlba_mult;
  op.data = &data;

  irlb(
    nullptr,
    &op,
    2,
    static_cast<int>(p),
    static_cast<int>(m),
    rank,
    work,
    maxit,
    0,
    tol,
    nullptr,
    nullptr,
    nullptr,
    s.memptr(),
    U.memptr(),
    Vright.memptr(),
    &iter,
    &mprod,
    eps,
    lwork,
    V1.memptr(),
    U1.memptr(),
    W.memptr(),
    F.memptr(),
    B.memptr(),
    BU.memptr(),
    BV.memptr(),
    BS.memptr(),
    BW.memptr(),
    res.memptr(),
    T.memptr(),
    svtol,
    svratio.memptr()
  );

  Ublock = project_deflated_left_double(U.cols(0, static_cast<arma::uword>(rank - 1)), V, n_prev);
  shat = s.subvec(0, static_cast<arma::uword>(rank - 1));
  return Ublock.n_cols > 0;
}

bool refresh_deflated_crossprod_left_double(
  const arma::mat& X,
  const arma::mat& Ymat,
  const arma::mat& V,
  const int n_prev,
  const arma::vec* warm_start,
  const int k_block,
  const int power_iters,
  const unsigned int seed,
  arma::mat& Ublock,
  arma::vec& shat
) {
  const arma::uword p = X.n_cols;
  const arma::uword m = Ymat.n_cols;
  if (p < 1 || m < 1 || k_block < 1) {
    return false;
  }

  auto a_times = [&](const arma::mat& M) -> arma::mat {
    return project_deflated_left_double(X.t() * (Ymat * M), V, n_prev);
  };
  auto at_times = [&](const arma::mat& M) -> arma::mat {
    arma::mat Mp = project_deflated_left_double(M, V, n_prev);
    return Ymat.t() * (X * Mp);
  };

  arma::mat Ysample = gaussian_matrix_local(
    p,
    static_cast<arma::uword>(k_block),
    seed
  );
  if (warm_start != nullptr && warm_start->n_elem == p) {
    Ysample.col(0) = *warm_start;
  }
  Ysample = project_deflated_left_double(Ysample, V, n_prev);

  for (int it = 0; it < std::max(power_iters, 0); ++it) {
    Ysample = a_times(at_times(Ysample));
  }

  arma::mat Q;
  arma::mat R;
  arma::qr_econ(Q, R, Ysample);
  if (Q.n_cols < 1) {
    return false;
  }
  Q = project_deflated_left_double(Q, V, n_prev);
  arma::qr_econ(Q, R, Q);
  if (Q.n_cols < 1) {
    return false;
  }

  arma::mat Bsmall = (X * Q).t() * Ymat;
  arma::mat Uhat;
  arma::mat Vhat;
  if (!finalize_left_block_from_bsmall(Bsmall, Uhat, shat, Vhat) || Uhat.n_cols < 1) {
    return false;
  }

  Ublock = project_deflated_left_double(Q * Uhat, V, n_prev);
  if (Ublock.n_cols > static_cast<arma::uword>(k_block)) {
    Ublock = Ublock.cols(0, static_cast<arma::uword>(k_block - 1));
  }
  return (Ublock.n_cols > 0);
}

bool refresh_deflated_crossprod_left_double_view(
  const CenterScaleMatrixView& Xop,
  const CenterOnlyMatrixView& Yop,
  const arma::mat& V,
  const int n_prev,
  const arma::vec* warm_start,
  const int k_block,
  const int power_iters,
  const unsigned int seed,
  arma::mat& Ublock,
  arma::vec& shat
) {
  const arma::uword p = Xop.X.n_cols;
  const arma::uword m = Yop.Y.n_cols;
  if (p < 1 || m < 1 || k_block < 1) {
    return false;
  }

  auto a_times = [&](const arma::mat& M) -> arma::mat {
    return project_deflated_left_double(Xop.t_times(Yop.times(M)), V, n_prev);
  };
  auto at_times = [&](const arma::mat& M) -> arma::mat {
    arma::mat Mp = project_deflated_left_double(M, V, n_prev);
    return Yop.t_times(Xop.times(Mp));
  };

  arma::mat Ysample = gaussian_matrix_local(
    p,
    static_cast<arma::uword>(k_block),
    seed
  );
  if (warm_start != nullptr && warm_start->n_elem == p) {
    Ysample.col(0) = *warm_start;
  }
  Ysample = project_deflated_left_double(Ysample, V, n_prev);

  for (int it = 0; it < std::max(power_iters, 0); ++it) {
    Ysample = a_times(at_times(Ysample));
  }

  arma::mat Q;
  arma::mat R;
  arma::qr_econ(Q, R, Ysample);
  if (Q.n_cols < 1) {
    return false;
  }
  Q = project_deflated_left_double(Q, V, n_prev);
  arma::qr_econ(Q, R, Q);
  if (Q.n_cols < 1) {
    return false;
  }

  arma::mat Bsmall = Yop.t_times(Xop.times(Q)).t();
  arma::mat Uhat;
  arma::mat Vhat;
  if (!finalize_left_block_from_bsmall(Bsmall, Uhat, shat, Vhat) || Uhat.n_cols < 1) {
    return false;
  }

  Ublock = project_deflated_left_double(Q * Uhat, V, n_prev);
  if (Ublock.n_cols > static_cast<arma::uword>(k_block)) {
    Ublock = Ublock.cols(0, static_cast<arma::uword>(k_block - 1));
  }
  return (Ublock.n_cols > 0);
}

struct SimplsFastRefreshWorkspace {
  arma::mat Omega;
  arma::mat Y;
  arma::mat Z;
  arma::mat Q;
  arma::mat R;
  arma::mat Bsmall;
  arma::mat Uhat;
  arma::vec shat;
  arma::mat Vhat;
  bool gpu_refresh_enabled = false;

  void prepare_gpu_refresh(
    const int s_rows,
    const int s_cols,
    const arma::vec* warm_start,
    const int k_block,
    const int power_iters,
    const unsigned int seed
  ) {
    const bool has_warm_start =
      (warm_start != nullptr && warm_start->n_elem == static_cast<arma::uword>(s_rows));
    if (has_warm_start) {
      Omega.set_size(static_cast<arma::uword>(s_rows), static_cast<arma::uword>(k_block));
      Omega = gaussian_matrix_local(
        static_cast<arma::uword>(s_rows),
        static_cast<arma::uword>(k_block),
        seed
      );
      Omega.col(0) = *warm_start;
    } else {
      Omega.reset();
    }
    Uhat.reset();
    Y.set_size(static_cast<arma::uword>(s_rows), static_cast<arma::uword>(k_block));
    shat.set_size(static_cast<arma::uword>(k_block));
    fastpls_svd::cuda_rsvd_refresh_left_block_u_resident(
      s_rows,
      s_cols,
      has_warm_start ? Omega.memptr() : nullptr,
      k_block,
      k_block,
      seed,
      std::max(power_iters, 0),
      Y.memptr(),
      shat.memptr()
    );
  }

  void prepare_cpu_refresh(
    const arma::mat& S,
    const arma::vec* warm_start,
    const int k_block,
    const int power_iters,
    const unsigned int seed
  ) {
    Omega = gaussian_matrix_local(
      S.n_rows,
      static_cast<arma::uword>(k_block),
      seed
    );
    if (warm_start != nullptr && warm_start->n_elem == S.n_rows) {
      Omega.col(0) = *warm_start;
    }

    Y = Omega;
    for (int it = 0; it < power_iters; ++it) {
      Z = S.t() * Y;
      Y = S * Z;
    }
  }

  bool refresh(
    const arma::mat& S,
    const arma::vec* warm_start,
    const int k_block,
    const int power_iters,
    const unsigned int seed,
    arma::mat& Ublock
  ) {
    if (S.n_rows < 1 || S.n_cols < 1 || k_block < 1) {
      return false;
    }

    if (gpu_refresh_enabled) {
      prepare_gpu_refresh(
        static_cast<int>(S.n_rows),
        static_cast<int>(S.n_cols),
        warm_start,
        k_block,
        power_iters,
        seed
      );
      Ublock = Y;
      if (Ublock.n_cols > static_cast<arma::uword>(k_block)) {
        Ublock = Ublock.cols(0, static_cast<arma::uword>(k_block - 1));
      }
      return (Ublock.n_cols > 0);
    } else {
      prepare_cpu_refresh(S, warm_start, k_block, power_iters, seed);
    }

    arma::qr_econ(Q, R, Y);
    if (Q.n_cols < 1) {
      return false;
    }

    Bsmall = Q.t() * S;
    if (Bsmall.n_rows < 1 || Bsmall.n_cols < 1) {
      return false;
    }

    if (!finalize_left_block_from_bsmall(Bsmall, Uhat, shat, Vhat) || Uhat.n_cols < 1) {
      return false;
    }

    Ublock = Q * Uhat;
    if (Ublock.n_cols > static_cast<arma::uword>(k_block)) {
      Ublock = Ublock.cols(0, static_cast<arma::uword>(k_block - 1));
    }
    return (Ublock.n_cols > 0);
  }
};

struct AdaptiveRefreshPolicy {
  bool enabled = false;
  int base_block = 8;
  int base_power = 2;
  int min_block = 2;
  int max_block = 16;
  int min_power = 1;
  int max_power = 4;
  double flat_ratio = 0.55;
  double steep_ratio = 0.12;
  int current_block = 8;
  int current_power = 2;

  static AdaptiveRefreshPolicy from_env(int base_block_in, int base_power_in) {
    AdaptiveRefreshPolicy out;
    out.enabled = (env_int_or("FASTPLS_FAST_ADAPTIVE_RSVD", 0, 0, 1) == 1);
    out.base_block = base_block_in;
    out.base_power = base_power_in;
    out.min_block = env_int_or("FASTPLS_FAST_ADAPTIVE_MIN_BLOCK", std::min(4, std::max(1, base_block_in)), 1, 64);
    out.max_block = env_int_or("FASTPLS_FAST_ADAPTIVE_MAX_BLOCK", std::max(base_block_in, 16), 1, 128);
    out.min_power = env_int_or("FASTPLS_FAST_ADAPTIVE_MIN_POWER", std::min(base_power_in, 1), 0, 8);
    out.max_power = env_int_or("FASTPLS_FAST_ADAPTIVE_MAX_POWER", std::max(base_power_in, 4), 0, 12);
    out.flat_ratio = std::max(0.0, std::min(0.99, std::atof(std::getenv("FASTPLS_FAST_ADAPTIVE_FLAT_RATIO") ? std::getenv("FASTPLS_FAST_ADAPTIVE_FLAT_RATIO") : "0.55")));
    out.steep_ratio = std::max(0.0, std::min(out.flat_ratio, std::atof(std::getenv("FASTPLS_FAST_ADAPTIVE_STEEP_RATIO") ? std::getenv("FASTPLS_FAST_ADAPTIVE_STEEP_RATIO") : "0.12")));
    out.current_block = std::min(std::max(out.base_block, out.min_block), out.max_block);
    out.current_power = std::min(std::max(out.base_power, out.min_power), out.max_power);
    return out;
  }

  std::pair<int,int> current(int remaining) const {
    return std::make_pair(
      std::max(1, std::min(current_block, remaining)),
      std::max(0, current_power)
    );
  }

  void update_from_spectrum(const arma::vec& shat_in, int remaining_after) {
    if (!enabled || shat_in.n_elem < 2) {
      return;
    }
    const double head = std::max(shat_in(0), std::numeric_limits<double>::epsilon());
    const arma::uword tail_idx = std::min<arma::uword>(shat_in.n_elem - 1, 1);
    const double tail = std::max(shat_in(tail_idx), 0.0);
    const double ratio = tail / head;

    if (ratio >= flat_ratio) {
      current_block = std::min(max_block, std::max(current_block + 2, current_block * 2));
      current_power = std::min(max_power, current_power + 1);
    } else if (ratio <= steep_ratio) {
      current_block = std::max(min_block, current_block / 2);
      current_power = std::max(min_power, current_power - 1);
    }

    current_block = std::max(1, std::min(current_block, remaining_after));
  }
};

} // namespace


// [[Rcpp::export]]
arma::mat ORTHOG(arma::mat& X, arma::mat& Y, arma::mat& T, int xm, int xn, int yn) {


  // Copy preserve R's data
  arma::mat Ycopy = arma::mat(Y.memptr(), Y.n_rows, Y.n_cols);
  orthog(X.memptr(), Ycopy.memptr(), T.memptr(), xm, xn, yn);
  return Ycopy;
}

// [[Rcpp::export]]
double RQ(arma::mat yData,arma::mat yPred){

  double TSS=0,PRESS=0;
  for(unsigned int i=0;i<yData.n_cols;i++){
    double my=mean(yData.col(i));
    for(unsigned int j=0;j<yData.n_rows;j++){
      double b1=yPred(j,i);
      double c1=yData(j,i);
      double d1=c1-my;
      double arg_TR=(c1-b1);
      PRESS+=arg_TR*arg_TR;
      TSS+=d1*d1;  
    }
  }
  
  double R2Y=1-PRESS/TSS;
  return R2Y;
}



/* irlb C++ implementation wrapper for Armadillo
* X double precision input matrix
* NU integer number of singular values/vectors to compute must be > 3
* INIT double precision starting vector length(INIT) must equal ncol(X)
* WORK integer working subspace dimension must be > NU
* MAXIT integer maximum number of iterations
* TOL double tolerance
* EPS double invariant subspace detection tolerance
* MULT integer 0 X is a dense matrix (dgemm), 1 sparse (cholmod)
* RESTART integer 0 no or > 0 indicates restart of dimension n
* RV, RW, RS optional restart V W and S values of dimension RESTART
*    (only used when RESTART > 0)
* SCALE either NULL (no scaling) or a vector of length ncol(X)
* SHIFT either NULL (no shift) or a single double-precision number
* CENTER either NULL (no centering) or a vector of length ncol(X)
* SVTOL double tolerance max allowed per cent change in each estimated singular value */
// [[Rcpp::export]]
List IRLB(const arma::mat& X,
                 int nu,
                 int work,
                 int maxit,
                 double tol,
                 double eps,
                 double svtol)
{

  int m = X.n_rows;
  int n = X.n_cols;
  int iter, mprod;
  int lwork = 7 * work * (1 + work);

  arma::vec s = arma::randn<arma::vec>(nu);
  arma::mat U = arma::randn<arma::mat>(m, work);
  arma::mat V = arma::randn<arma::mat>(n, work);

  arma::mat V1 = arma::zeros<arma::mat>(n, work); // n x work
  arma::mat U1 = arma::zeros<arma::mat>(m, work); // m x work
  arma::mat  W = arma::zeros<arma::mat>(m, work);  // m x work  input when restart > 0
  arma::vec F  = arma::zeros<arma::vec>(n);     // n
  arma::mat B  = arma::zeros<arma::mat>(work, work);  // work x work  input when restart > 0
  arma::mat BU = arma::zeros<arma::mat>(work, work);  // work x work
  arma::mat BV = arma::mat(work, work);  // work x work
  arma::vec BS = arma::zeros<arma::vec>(work);  // work
  arma::vec BW = arma::zeros<arma::vec>(lwork); // lwork
  arma::vec res = arma::zeros<arma::vec>(work); // work
  arma::vec T = arma::zeros<arma::vec>(lwork);  // lwork
  arma::vec svratio = arma::zeros<arma::vec>(work); // work


  irlb (const_cast<double*>(X.memptr()), NULL, 0, m, n, nu, work, maxit, 0,
          tol, NULL, NULL, NULL,
          s.memptr(), U.memptr(), V.memptr(), &iter, &mprod,
          eps, lwork, V1.memptr(), U1.memptr(), W.memptr(),
          F.memptr(), B.memptr(), BU.memptr(), BV.memptr(),
          BS.memptr(), BW.memptr(), res.memptr(), T.memptr(),
          svtol, svratio.memptr());
  return List::create(Rcpp::Named("d")=s,
                            Rcpp::Named("u")=U.cols(0, nu-1),
                            Rcpp::Named("v")=V.cols(0,nu-1),
                            Rcpp::Named("iter")=iter,
                            Rcpp::Named("mprod")=mprod);
                            // Rcpp::Named("converged")=conv);
}



arma::mat variance(arma::mat x) {
  int nrow = x.n_rows, ncol = x.n_cols;
  arma::mat out(1,ncol);
  
  for (int j = 0; j < ncol; j++) {
    double mean = 0;
    double M2 = 0;
    int n=0;
    double delta, xx;
    for (int i = 0; i < nrow; i++) {
      n = i+1;
      xx = x(i,j);
      delta = xx - mean;
      mean += delta/n;
      M2 = M2 + delta*(xx-mean);
    }
    out(0,j) = sqrt(M2/(n-1));
  }
  return out;
}


// [[Rcpp::export]]
arma::mat transformy(arma::ivec y){
  int n=y.size();
  int nc=max(y);
  arma::mat yy(n,nc);
  yy.zeros();
  for(int i=0;i<nc;i++){
    for(int j=0;j<n;j++){
      yy(j,i)=((i+1)==y(j));
    }
  }
  return yy;
}

// [[Rcpp::export]]
bool has_cuda() {
  return fastpls_svd::has_cuda_backend();
}

// [[Rcpp::export]]
void cuda_reset_workspace() {
  fastpls_svd::cuda_reset_workspace();
}

// [[Rcpp::export]]
arma::mat cuda_matrix_multiply(const arma::mat& A, const arma::mat& B) {
  return fastpls_svd::cuda_matrix_multiply(A, B);
}

// [[Rcpp::export]]
arma::mat cuda_thin_qr(const arma::mat& A) {
  return fastpls_svd::cuda_thin_qr(A);
}

// [[Rcpp::export]]
Rcpp::List truncated_svd_debug(
  const arma::mat& A,
  int k,
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed,
  bool left_only
) {
  fastpls_svd::SVDResult res = compute_truncated_svd_dispatch(
    A,
    k,
    svd_method,
    rsvd_oversample,
    rsvd_power,
    svds_tol,
    static_cast<unsigned int>(seed),
    left_only,
    false
  );

  return Rcpp::List::create(
    Rcpp::Named("u") = res.U,
    Rcpp::Named("d") = res.s,
    Rcpp::Named("vt") = res.Vt
  );
}

// [[Rcpp::export]]
List pls_model2(
  arma::mat Xtrain,
  arma::mat Ytrain,
  arma::ivec ncomp,
  int scaling,
  bool fit,
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed
) {

  // n <-dim(Xtrain)[1]
  int n = Xtrain.n_rows;
  
  // p <-dim(Xtrain)[2]
  int p = Xtrain.n_cols;
  
  // m <- dim(Y)[2]
  int m = Ytrain.n_cols;
  
  int max_ncomp=max(ncomp);
  int length_ncomp=ncomp.n_elem;

  // X <- scale(Xtrain,center=TRUE,scale=FALSE)
  // Xtest <-scale(Xtest,center=mX)
  arma::mat mX(1,p); 
  mX.zeros();
  if(scaling<3){
    mX=mean(Xtrain,0);
    Xtrain.each_row()-=mX;
  } 
  
  arma::mat vX(1,p); 
  vX.ones();
  if(scaling==2){
    vX=variance(Xtrain); 
    Xtrain.each_row()/=vX;
  }
  
  //X=Xtrain
  arma::mat X=Xtrain;
  
  //Y=Ytrain
  arma::mat Y=Ytrain;
  
  // Y <- scale(Ytrain,center=TRUE,scale=FALSE)
  arma::mat mY=mean(Ytrain,0);
  Y.each_row()-=mY;
  
  // S <- crossprod(X,Y)
  arma::mat S=trans(X)*Y;
  
  //  RR<-matrix(0,ncol=ncomp,nrow=p)
  arma::mat RR(p,max_ncomp);
  RR.zeros();
  
  //  PP<-matrix(0,ncol=ncomp,nrow=p)
  arma::mat PP(p,max_ncomp);
  PP.zeros();
  
  //  QQ<-matrix(0,ncol=ncomp,nrow=m)
  arma::mat QQ(m,max_ncomp);
  QQ.zeros();
  
  //  TT<-matrix(0,ncol=ncomp,nrow=n)
  arma::mat TT(n,max_ncomp);
  TT.zeros();
  
  //  VV<-matrix(0,ncol=ncomp,nrow=p)
  arma::mat VV(p,max_ncomp);
  VV.zeros();
  
  const bool store_B = should_store_coefficients(p, m, length_ncomp, true);
  arma::cube B;
  if (store_B) {
    B.set_size(p, m, length_ncomp);
    B.zeros();
  }
  
  // Yfit <- matrix(0,ncol=m,nrow=n)
  arma::cube Yfit;
  arma::vec R2Y(length_ncomp);
  if(fit){
    Yfit.resize(n,m,length_ncomp);
//    Yfit.zeros();  
  }
  
  arma::mat qq;
  arma::mat pp;
  arma::mat rr;
  arma::mat tt;
  arma::mat vv;
  
  int i_out=0; //position of the saving output
  
  // for(a in 1:ncomp){
  for (int a=0; a<max_ncomp; a++) {
    //qq<-svd(S)$v[,1]
    //rr <- S%*%qq
//    if(S.n_rows<=16 || S.n_cols<=16){
    rr = leading_left_vec_dispatch(
      S,
      svd_method,
      rsvd_oversample,
      rsvd_power,
      svds_tol,
      static_cast<unsigned int>(seed + a),
      nullptr
    );
    if (rr.n_elem != static_cast<arma::uword>(S.n_rows)) {
      break;
    }
  
    // tt<-scale(X%*%rr,scale=FALSE)
    tt=X*rr; 
    arma::mat mtt=mean(tt,0);
    tt.each_row()-=mtt;
    
    //tnorm<-sqrt(sum(tt*tt))
    double tnorm=sqrt(sum(sum(tt%tt)));
    
    //tt<-tt/tnorm
    tt/=tnorm;
    
    //rr<-rr/tnorm
    rr/=tnorm;
    
    // pp <- crossprod(X,tt)
    pp=trans(X)*tt;
    
    // qq <- crossprod(Y,tt)
    qq=trans(Y)*tt;
    
    //vv<-pp
    vv=pp;
    
    if(a>0){
      //vv<-vv-VV%*%crossprod(VV,pp)
      vv-=VV*(trans(VV)*pp);
    }
    
    //vv <- vv/sqrt(sum(vv*vv))
    vv/=sqrt(sum(sum(vv%vv)));
    
    //S <- S-vv%*%crossprod(vv,S)
    S-=vv*(trans(vv)*S);
    
    //RR[,a]=rr
    RR.col(a)=rr;
    TT.col(a)=tt;
    PP.col(a)=pp;
    QQ.col(a)=qq;
    VV.col(a)=vv;
    
    if(a==(ncomp(i_out)-1)){
      arma::mat R_a = RR.cols(0, a);
      arma::mat Q_a = QQ.cols(0, a);
      if (store_B) {
        B.slice(i_out) = R_a * trans(Q_a);
      }
      if(fit){
        arma::mat temp1 = TT.cols(0, a) * trans(Q_a);
        temp1.each_row()+=mY;
        Yfit.slice(i_out)=temp1;
        R2Y(i_out)=RQ(Ytrain,temp1);
        
      }
      i_out++;
    }
  } 
  List out = List::create(
    Named("P")       = PP,
    Named("Q")       = QQ,
    Named("Ttrain")  = TT,
    Named("R")       = RR,
    Named("mX")      = mX,
    Named("vX")      = vX,
    Named("mY")      = mY,
    Named("p")       = p,
    Named("m")       = m,
    Named("ncomp")   = ncomp,
    Named("Yfit")    = Yfit,
    Named("R2Y")     = R2Y
  );
  if (store_B) {
    out["B"] = B;
  }
  annotate_coefficient_storage(out, store_B);
  return out;
}

// [[Rcpp::export]]
List pls_model2_fast(
  arma::mat Xtrain,
  arma::mat Ytrain,
  arma::ivec ncomp,
  int scaling,
  bool fit,
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed
) {
  const int n = Xtrain.n_rows;
  const int p = Xtrain.n_cols;
  const int m = Ytrain.n_cols;

  if (ncomp.n_elem < 1) {
    stop("ncomp must contain at least one value");
  }
  for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
    if (ncomp(i) < 1) {
      ncomp(i) = 1;
    }
  }

  const int max_ncomp = max(ncomp);
  const int length_ncomp = ncomp.n_elem;

  arma::mat mX(1, p, fill::zeros);
  if (scaling < 3) {
    mX = mean(Xtrain, 0);
    Xtrain.each_row() -= mX;
  }

  arma::mat vX(1, p, fill::ones);
  if (scaling == 2) {
    vX = variance(Xtrain);
    Xtrain.each_row() /= vX;
  }

  arma::mat mY = mean(Ytrain, 0);
  Ytrain.each_row() -= mY;

  const arma::mat Xt = Xtrain.t();
  const arma::mat Yt = Ytrain.t();
  arma::mat S = Xt * Ytrain;
  arma::mat XtX_cache;
  arma::mat Sxy_cache;

  arma::mat RR(p, max_ncomp, fill::zeros);
  arma::mat QQ(m, max_ncomp, fill::zeros);
  arma::mat VV(p, max_ncomp, fill::zeros);
  const bool store_B = should_store_coefficients(p, m, length_ncomp, true);
  arma::cube B;
  if (store_B) {
    B.zeros(p, m, length_ncomp);
  }

  arma::cube Yfit;
  arma::vec R2Y(length_ncomp, fill::zeros);
  arma::mat Yfit_cur;
  if (fit) {
    Yfit.set_size(n, m, length_ncomp);
    Yfit_cur.zeros(n, m);
  }

  arma::mat Bcur;
  if (store_B) {
    Bcur.zeros(p, m);
  }
  int i_out = 0;

  // Inspired by block-Krylov randomized SVD literature (e.g. arXiv:1504.05477):
  // refresh a small block of singular vectors to reduce per-component SVD overhead.
  const int refresh_block = env_int_or("FASTPLS_FAST_BLOCK", 1, 1, 16);
  const int center_t = env_int_or("FASTPLS_FAST_CENTER_T", 0, 0, 1);
  const int reorth_v = env_int_or("FASTPLS_FAST_REORTH_V", 0, 0, 1);
  const int incremental_svd = 1;
  const int inc_power_iters = env_int_or("FASTPLS_FAST_INC_ITERS", 2, 1, 6);
  const int defl_cache = env_int_or("FASTPLS_FAST_DEFLCACHE", 1, 0, 1);
  const int fast_optimized = env_int_or("FASTPLS_FAST_OPTIMIZED", 1, 0, 1);
  const int fast_top1_rsvd = env_int_or("FASTPLS_FAST_RSVD_TOP1", 0, 0, 1);
  const int fast_crossprod_min_ncomp = env_int_or("FASTPLS_FAST_CROSSPROD_MIN_NCOMP", 20, 1, 1024);
  const int fast_crossprod_max_p = env_int_or("FASTPLS_FAST_CROSSPROD_MAX_P", 512, 16, 65536);
  const int fast_crossprod_min_n_to_p_ratio = env_int_or("FASTPLS_FAST_CROSSPROD_MIN_N_TO_P_RATIO", 8, 1, 1024);
  const int top1_rsvd_oversample = env_int_or(
    "FASTPLS_FAST_RSVD_TOP1_OVERSAMPLE",
    std::min(std::max(rsvd_oversample, 0), 2),
    0,
    8
  );
  const int top1_rsvd_power = env_int_or(
    "FASTPLS_FAST_RSVD_TOP1_POWER",
    std::min(std::max(rsvd_power, 0) + 1, 2),
    0,
    4
  );
  arma::vec rr_prev;
  bool has_rr_prev = false;
  const bool use_crossprod_cache =
    (fast_optimized == 1) &&
    (center_t == 0) &&
    (max_ncomp >= fast_crossprod_min_ncomp) &&
    (p <= n) &&
    (n >= p * fast_crossprod_min_n_to_p_ratio) &&
    (p <= fast_crossprod_max_p);
  if (use_crossprod_cache) {
    XtX_cache = Xt * Xtrain;
    Sxy_cache = S;
  }
  bool gpu_deflation_enabled = false;
  auto append_component = [&](arma::vec rr, const int a_idx) -> bool {
    arma::vec pp;
    arma::vec qq;
    arma::vec tt;
    if (use_crossprod_cache) {
      pp = XtX_cache * rr;
      const double tnorm_sq = arma::dot(rr, pp);
      if (!std::isfinite(tnorm_sq) || tnorm_sq <= 0.0) {
        return false;
      }
      const double tnorm = std::sqrt(tnorm_sq);
      rr /= tnorm;
      pp /= tnorm;
      qq = Sxy_cache.t() * rr;
      if (fit) {
        tt = Xtrain * rr;
      }
    } else {
      tt = Xtrain * rr;
      if (center_t == 1) {
        tt -= arma::mean(tt);
      }
      const double tnorm = arma::norm(tt, 2);
      if (!std::isfinite(tnorm) || tnorm <= 0.0) {
        return false;
      }
      tt /= tnorm;
      rr /= tnorm;
      pp = Xt * tt;
      qq = Yt * tt;
    }
    rr_prev = rr;
    has_rr_prev = true;

    arma::vec vv = pp;
    if (a_idx > 0) {
      auto Vprev = VV.cols(0, a_idx - 1);
      vv -= Vprev * (Vprev.t() * pp);
      if (reorth_v == 1) {
        vv -= Vprev * (Vprev.t() * vv);
      }
    }
    const double vnorm = arma::norm(vv, 2);
    if (!std::isfinite(vnorm) || vnorm <= 0.0) {
      return false;
    }
    vv /= vnorm;

    if (gpu_deflation_enabled) {
      arma::vec vS(S.n_cols, arma::fill::zeros);
      fastpls_svd::cuda_rsvd_project_left_row(
        vv.memptr(),
        static_cast<int>(S.n_rows),
        static_cast<int>(S.n_cols),
        vS.memptr()
      );
      fastpls_svd::cuda_rsvd_deflate_left_rank1(
        vv.memptr(),
        vS.memptr(),
        static_cast<int>(S.n_rows),
        static_cast<int>(S.n_cols)
      );
    } else if (defl_cache == 1) {
      arma::rowvec vS = vv.t() * S;
      S -= vv * vS;
    } else {
      S -= vv * (vv.t() * S);
    }

    RR.col(a_idx) = rr;
    QQ.col(a_idx) = qq;
    VV.col(a_idx) = vv;
    if (store_B) {
      Bcur += rr * qq.t();
    }

    while (i_out < length_ncomp && a_idx == (ncomp(i_out) - 1)) {
      if (store_B) {
        B.slice(i_out) = Bcur;
      }
      if (fit) {
        Yfit_cur += tt * qq.t();
        R2Y(i_out) = RQ(Ytrain, Yfit_cur);
        arma::mat yf = Yfit_cur;
        yf.each_row() += mY;
        Yfit.slice(i_out) = yf;
      }
      ++i_out;
    }
    return true;
  };

  const bool can_incremental = (incremental_svd == 1) && (S.n_rows > 5) && (S.n_cols > 1);
  const bool use_optimized_top1_rsvd =
    (fast_optimized == 1) &&
    (fast_top1_rsvd == 1) &&
    can_incremental &&
    is_rsvd_backend_method(svd_method);
  const bool use_gpu_refresh =
    (fast_optimized == 1) &&
    can_incremental &&
    (svd_method == fastpls_svd::SVD_METHOD_CUDA_RSVD) &&
    fastpls_svd::cuda_rsvd_prefer_block_gpu(
      static_cast<int>(S.n_rows),
      static_cast<int>(S.n_cols),
      std::min(refresh_block, max_ncomp),
      inc_power_iters
    );
  SimplsFastRefreshWorkspace refresh_ws;
  refresh_ws.gpu_refresh_enabled = use_gpu_refresh;
  AdaptiveRefreshPolicy adaptive_policy = AdaptiveRefreshPolicy::from_env(refresh_block, inc_power_iters);
  gpu_deflation_enabled = use_gpu_refresh;
  if (use_gpu_refresh) {
    fastpls_svd::cuda_rsvd_set_resident_matrix(
      S.memptr(),
      static_cast<int>(S.n_rows),
      static_cast<int>(S.n_cols)
    );
  }

  if (use_optimized_top1_rsvd) {
    for (int a = 0; a < max_ncomp; ++a) {
      arma::vec rr = top1_rsvd_left_vector(
        S,
        has_rr_prev ? &rr_prev : nullptr,
        top1_rsvd_oversample,
        top1_rsvd_power,
        static_cast<unsigned int>(seed + a)
      );
      if (rr.n_elem != S.n_rows || !append_component(rr, a)) {
        break;
      }
    }
  } else {
    int a = 0;
    while (a < max_ncomp) {
      const int remaining = max_ncomp - a;
      const std::pair<int,int> refresh_cfg = adaptive_policy.current(remaining);
      const int k_block = std::min(refresh_cfg.first, remaining);
      const int power_iters_block = refresh_cfg.second;
      arma::mat Ublock;
      if (can_incremental) {
        const arma::vec* warm_start = has_rr_prev ? &rr_prev : nullptr;
        if (!refresh_ws.refresh(
              S,
              warm_start,
              k_block,
              power_iters_block,
              static_cast<unsigned int>(seed + a),
              Ublock
            )) {
          break;
        }
        adaptive_policy.update_from_spectrum(refresh_ws.shat, max_ncomp - (a + k_block));
      } else {
        fastpls_svd::SVDResult svd_res = compute_truncated_svd_dispatch(
          S,
          k_block,
          svd_method,
          rsvd_oversample,
          rsvd_power,
          svds_tol,
          static_cast<unsigned int>(seed + a),
          true,
          false
        );
        Ublock = svd_res.U;
      }
      if (Ublock.n_cols < 1) {
        break;
      }

      const int use_cols = std::min(static_cast<int>(Ublock.n_cols), k_block);
      bool stop_now = false;
      for (int j = 0; j < use_cols && a < max_ncomp; ++j, ++a) {
        if (!append_component(Ublock.col(j), a)) {
          stop_now = true;
          break;
        }
      }
      if (stop_now) {
        break;
      }
    }
  }

  List out = List::create(
    Named("P")       = arma::mat(),
    Named("Q")       = QQ,
    Named("Ttrain")  = arma::mat(),
    Named("R")       = RR,
    Named("mX")      = mX,
    Named("vX")      = vX,
    Named("mY")      = mY,
    Named("p")       = p,
    Named("m")       = m,
    Named("ncomp")   = ncomp,
    Named("Yfit")    = Yfit,
    Named("R2Y")     = R2Y
  );
  if (store_B) {
    out["B"] = B;
  }
  annotate_coefficient_storage(out, store_B);
  return out;
}

// [[Rcpp::export]]
List pls_model2_fast_gpu(
  arma::mat Xtrain,
  arma::mat Ytrain,
  arma::ivec ncomp,
  int scaling,
  bool fit,
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed
) {
  if (svd_method != fastpls_svd::SVD_METHOD_CUDA_RSVD || !fastpls_svd::has_cuda_backend()) {
    stop("pls_model2_fast_gpu requires svd.method='cuda_rsvd' with CUDA available");
  }

  const int n = Xtrain.n_rows;
  const int p = Xtrain.n_cols;
  const int m = Ytrain.n_cols;

  if (ncomp.n_elem < 1) {
    stop("ncomp must contain at least one value");
  }
  for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
    if (ncomp(i) < 1) {
      ncomp(i) = 1;
    }
  }

  const int max_ncomp = max(ncomp);
  const int length_ncomp = ncomp.n_elem;

  arma::mat mX(1, p, fill::zeros);
  if (scaling < 3) {
    mX = mean(Xtrain, 0);
    Xtrain.each_row() -= mX;
  }

  arma::mat vX(1, p, fill::ones);
  if (scaling == 2) {
    vX = variance(Xtrain);
    Xtrain.each_row() /= vX;
  }

  arma::mat mY = mean(Ytrain, 0);
  Ytrain.each_row() -= mY;

  const bool use_implicit_xprod =
    (env_int_or("FASTPLS_GPU_SIMPLS_XPROD", 0, 0, 1) == 1);
  const bool use_device_state =
    (env_int_or("FASTPLS_GPU_DEVICE_STATE", 0, 0, 1) == 1);
  arma::mat Xt;
  arma::mat Yt;
  if (!use_implicit_xprod && !use_device_state) {
    Xt = Xtrain.t();
    Yt = Ytrain.t();
  }

  arma::mat RR(p, max_ncomp, fill::zeros);
  arma::mat QQ(m, max_ncomp, fill::zeros);
  arma::mat VV(p, max_ncomp, fill::zeros);
  const bool store_B = should_store_coefficients(p, m, length_ncomp, true);
  arma::cube B;
  if (store_B) {
    B.zeros(p, m, length_ncomp);
  }

  arma::cube Yfit;
  arma::vec R2Y(length_ncomp, fill::zeros);
  arma::mat Yfit_cur;
  if (fit) {
    Yfit.set_size(n, m, length_ncomp);
    Yfit_cur.zeros(n, m);
  }

  arma::mat Bcur;
  if (store_B) {
    Bcur.zeros(p, m);
  }
  int i_out = 0;

  const int refresh_block = env_int_or("FASTPLS_FAST_BLOCK", 1, 1, 16);
  const int center_t = env_int_or("FASTPLS_FAST_CENTER_T", 0, 0, 1);
  const int reorth_v = env_int_or("FASTPLS_FAST_REORTH_V", 0, 0, 1);
  const int inc_power_iters = env_int_or("FASTPLS_FAST_INC_ITERS", 2, 1, 6);
  const int defl_cache = env_int_or("FASTPLS_FAST_DEFLCACHE", 1, 0, 1);
  (void)defl_cache;
  AdaptiveRefreshPolicy adaptive_policy = AdaptiveRefreshPolicy::from_env(refresh_block, inc_power_iters);
  if (center_t == 1) {
    stop("pls_model2_fast_gpu does not support FASTPLS_FAST_CENTER_T=1");
  }

  fastpls_svd::cuda_simpls_fast_set_training_matrices(
    Xtrain.memptr(),
    n,
    p,
    Ytrain.memptr(),
    m,
    fit,
    !use_implicit_xprod
  );
  if (use_device_state) {
    fastpls_svd::cuda_simpls_fast_begin_device_loop(n, p, m, max_ncomp, fit);
    bool has_rr_prev = false;
    int a = 0;
    while (a < max_ncomp) {
      const int remaining = max_ncomp - a;
      const std::pair<int,int> refresh_cfg = adaptive_policy.current(remaining);
      const int k_block = std::min(refresh_cfg.first, remaining);
      const int power_iters_block = refresh_cfg.second;
      arma::vec shat_block(k_block, arma::fill::zeros);
      if (use_implicit_xprod) {
        fastpls_svd::cuda_simpls_fast_refresh_block_implicit_resident(
          n,
          p,
          m,
          k_block,
          k_block,
          a,
          has_rr_prev,
          static_cast<unsigned int>(seed + a),
          power_iters_block,
          shat_block.memptr()
        );
      } else {
        fastpls_svd::cuda_simpls_fast_refresh_block_resident(
          p,
          m,
          k_block,
          k_block,
          has_rr_prev,
          static_cast<unsigned int>(seed + a),
          power_iters_block,
          shat_block.memptr()
        );
      }
      adaptive_policy.update_from_spectrum(shat_block, max_ncomp - (a + k_block));

      bool stop_now = false;
      for (int j = 0; j < k_block && a < max_ncomp;) {
        bool used_retry_refresh = false;
        bool appended = fastpls_svd::cuda_simpls_fast_append_component_from_block(
              n,
              p,
              m,
              a,
              j,
              a,
              (reorth_v == 1),
              fit,
              !use_implicit_xprod
            );
        if (!appended) {
          // A warm-started randomized refresh can occasionally land in a
          // direction removed by SIMPLS deflation. Retry entirely on-device
          // with fresh random starts instead of terminating the coefficient path.
          const int max_gpu_refresh_retries = 8;
          for (int retry = 0; retry < max_gpu_refresh_retries && !appended; ++retry) {
            const int retry_l = std::min(
              std::min(p, m),
              std::max(2, std::min(32, k_block * (retry + 2)))
            );
            arma::vec retry_shat(1, arma::fill::zeros);
            const unsigned int retry_seed =
              static_cast<unsigned int>(seed + a + 7919 * (retry + 1));
            const int retry_power_iters = std::min(power_iters_block + retry + 1, 8);
            if (use_implicit_xprod) {
              fastpls_svd::cuda_simpls_fast_refresh_block_implicit_resident(
                n,
                p,
                m,
                retry_l,
                1,
                a,
                false,
                retry_seed,
                retry_power_iters,
                retry_shat.memptr()
              );
            } else {
              fastpls_svd::cuda_simpls_fast_refresh_block_resident(
                p,
                m,
                retry_l,
                1,
                false,
                retry_seed,
                retry_power_iters,
                retry_shat.memptr()
              );
            }
            appended = fastpls_svd::cuda_simpls_fast_append_component_from_block(
              n,
              p,
              m,
              a,
              0,
              a,
              (reorth_v == 1),
              fit,
              !use_implicit_xprod
            );
            used_retry_refresh = appended;
          }
        }
        if (!appended) {
          stop_now = true;
          break;
        }
        has_rr_prev = true;

        while (i_out < length_ncomp && a == (ncomp(i_out) - 1)) {
          if (store_B) {
            fastpls_svd::cuda_simpls_fast_copy_bcur(B.slice(i_out).memptr(), p, m);
          }
          if (fit) {
            fastpls_svd::cuda_simpls_fast_copy_yfit(Yfit_cur.memptr(), n, m);
            R2Y(i_out) = RQ(Ytrain, Yfit_cur);
            arma::mat yf = Yfit_cur;
            yf.each_row() += mY;
            Yfit.slice(i_out) = yf;
          }
          ++i_out;
        }
        ++a;
        if (used_retry_refresh) {
          break;
        }
        ++j;
      }
      if (stop_now) {
        break;
      }
    }

    while (i_out < length_ncomp) {
      if (store_B) {
        fastpls_svd::cuda_simpls_fast_copy_bcur(B.slice(i_out).memptr(), p, m);
      }
      if (fit) {
        fastpls_svd::cuda_simpls_fast_copy_yfit(Yfit_cur.memptr(), n, m);
        R2Y(i_out) = RQ(Ytrain, Yfit_cur);
        arma::mat yf = Yfit_cur;
        yf.each_row() += mY;
        Yfit.slice(i_out) = yf;
      }
      ++i_out;
    }

    fastpls_svd::cuda_simpls_fast_copy_rr(RR.memptr(), p, max_ncomp);
    fastpls_svd::cuda_simpls_fast_copy_qq(QQ.memptr(), m, max_ncomp);
  } else {
    arma::mat S_shape;
    if (!use_implicit_xprod) {
      S_shape = Xt * Ytrain;
    }
    SimplsFastRefreshWorkspace refresh_ws;
    refresh_ws.gpu_refresh_enabled = false;
    arma::vec rr_prev;
    bool has_rr_prev = false;

    auto append_component = [&](arma::vec rr, const int a_idx) -> bool {
      arma::vec tt(n, arma::fill::zeros);
      arma::vec pp(p, arma::fill::zeros);
      arma::vec qq(m, arma::fill::zeros);
      double tnorm = 0.0;
      bool gpu_stats_ok = true;
      try {
        fastpls_svd::cuda_simpls_fast_component_stats(
          rr.memptr(),
          n,
          p,
          m,
          tt.memptr(),
          pp.memptr(),
          qq.memptr(),
          &tnorm
        );
      } catch (const std::exception&) {
        gpu_stats_ok = false;
      }

      if (!gpu_stats_ok || !std::isfinite(tnorm) || tnorm <= 0.0) {
        tt = Xtrain * rr;
        const double host_tnorm = arma::norm(tt, 2);
        if (!std::isfinite(host_tnorm) || host_tnorm <= 0.0) {
          return false;
        }
        tt /= host_tnorm;
        rr /= host_tnorm;
        pp = Xtrain.t() * tt;
        qq = Ytrain.t() * tt;
      } else {
        rr /= tnorm;
      }

      rr_prev = rr;
      has_rr_prev = true;

      arma::vec vv = pp;
      if (a_idx > 0) {
        auto Vprev = VV.cols(0, a_idx - 1);
        vv -= Vprev * (Vprev.t() * pp);
        if (reorth_v == 1) {
          vv -= Vprev * (Vprev.t() * vv);
        }
      }
      const double vnorm = arma::norm(vv, 2);
      if (!std::isfinite(vnorm) || vnorm <= 0.0) {
        return false;
      }
      vv /= vnorm;

      if (!use_implicit_xprod) {
        arma::rowvec vS = vv.t() * S_shape;
        S_shape -= vv * vS;
      }

      RR.col(a_idx) = rr;
      QQ.col(a_idx) = qq;
      VV.col(a_idx) = vv;
      if (store_B) {
        Bcur += rr * qq.t();
      }

      while (i_out < length_ncomp && a_idx == (ncomp(i_out) - 1)) {
        if (store_B) {
          B.slice(i_out) = Bcur;
        }
        if (fit) {
          fastpls_svd::cuda_simpls_fast_rank1_fit_update(
            tt.memptr(),
            n,
            qq.memptr(),
            m,
            Yfit_cur.memptr()
          );
          R2Y(i_out) = RQ(Ytrain, Yfit_cur);
          arma::mat yf = Yfit_cur;
          yf.each_row() += mY;
          Yfit.slice(i_out) = yf;
        }
        ++i_out;
      }
      return true;
    };

    int a = 0;
    while (a < max_ncomp) {
      const int remaining = max_ncomp - a;
      const std::pair<int,int> refresh_cfg = adaptive_policy.current(remaining);
      const int k_block = std::min(refresh_cfg.first, remaining);
      const int power_iters_block = refresh_cfg.second;
      arma::mat Ublock;
      const arma::vec* warm_start = has_rr_prev ? &rr_prev : nullptr;
      if (use_implicit_xprod) {
        arma::vec shat_block;
        if (!refresh_deflated_crossprod_left_double(
              Xtrain,
              Ytrain,
              VV,
              a,
              warm_start,
              k_block,
              power_iters_block,
              static_cast<unsigned int>(seed + a),
              Ublock,
              shat_block
            )) {
          break;
        }
        adaptive_policy.update_from_spectrum(shat_block, max_ncomp - (a + k_block));
      } else {
        if (!refresh_ws.refresh(
              S_shape,
              warm_start,
              k_block,
              power_iters_block,
              static_cast<unsigned int>(seed + a),
              Ublock
            )) {
          break;
        }
        adaptive_policy.update_from_spectrum(refresh_ws.shat, max_ncomp - (a + k_block));
      }
      if (Ublock.n_cols < 1) {
        break;
      }

      const int use_cols = std::min(static_cast<int>(Ublock.n_cols), k_block);
      bool stop_now = false;
      for (int j = 0; j < use_cols && a < max_ncomp; ++j, ++a) {
        if (!append_component(Ublock.col(j), a)) {
          stop_now = true;
          break;
        }
      }
      if (stop_now) {
        break;
      }
    }
  }

  List out = List::create(
    Named("P")       = arma::mat(),
    Named("Q")       = QQ,
    Named("Ttrain")  = arma::mat(),
    Named("R")       = RR,
    Named("mX")      = mX,
    Named("vX")      = vX,
    Named("mY")      = mY,
    Named("p")       = p,
    Named("m")       = m,
    Named("ncomp")   = ncomp,
    Named("Yfit")    = Yfit,
    Named("R2Y")     = R2Y,
    Named("xprod_mode") = use_implicit_xprod ?
      (use_device_state ? "implicit_resident" : "implicit") :
      (use_device_state ? "materialized_resident" : "materialized"),
    Named("gpu_resident") = use_device_state
  );
  if (store_B) {
    out["B"] = B;
  }
  annotate_coefficient_storage(out, store_B);
  return out;
}


// [[Rcpp::export]]
List pls_predict(List& model, arma::mat Xtest, bool proj) {

  // columns of Ytrain
  const int m = Rcpp::as<int>(model["m"]);
  
  // w <-dim(Xtest)[1]
  const int w = Xtest.n_rows;
  
  arma::ivec ncomp = Rcpp::as<arma::ivec>(model["ncomp"]);
  const arma::uword length_ncomp = static_cast<arma::uword>(ncomp.n_elem);
  
  //scaling factors
  Rcpp::NumericVector mX_vec = model["mX"];
  arma::rowvec mX(mX_vec.begin(), mX_vec.size(), false, true);
  Xtest.each_row()-=mX;
  Rcpp::NumericVector vX_vec = model["vX"];
  arma::rowvec vX(vX_vec.begin(), vX_vec.size(), false, true);
  Xtest.each_row()/=vX;
  Rcpp::NumericVector mY_vec = model["mY"];
  arma::rowvec mY(mY_vec.begin(), mY_vec.size(), false, true);

  arma::cube Ypred(w, m, length_ncomp, arma::fill::none);
  bool used_latent_predict = false;

  std::string pls_method;
  if (model.containsElementNamed("pls_method")) {
    pls_method = Rcpp::as<std::string>(model["pls_method"]);
  }
  bool latent_predict_enabled = false;
  if (model.containsElementNamed("predict_latent_ok")) {
    latent_predict_enabled = Rcpp::as<bool>(model["predict_latent_ok"]);
  }

  const int latent_min_b_mb = env_int_or("FASTPLS_PREDICT_LATENT_MIN_B_MB", 256, 0, 1048576);
  const double coefficient_matrix_mb =
    static_cast<double>(Xtest.n_cols) * static_cast<double>(m) * sizeof(double) /
    (1024.0 * 1024.0);
  const bool prefer_latent_predict =
    (latent_min_b_mb == 0) || (coefficient_matrix_mb >= static_cast<double>(latent_min_b_mb));
  const bool has_B = model.containsElementNamed("B");
  const bool use_latent_predict = prefer_latent_predict || !has_B;

  if (latent_predict_enabled &&
      use_latent_predict &&
      (pls_method == "simpls" || pls_method == "simpls_fast")) {
    Rcpp::NumericVector R_vec = model["R"];
    Rcpp::NumericVector Q_vec = model["Q"];
    Rcpp::IntegerVector R_dim = R_vec.attr("dim");
    Rcpp::IntegerVector Q_dim = Q_vec.attr("dim");
    if (R_dim.size() == 2L && Q_dim.size() == 2L &&
        R_dim[0] == Xtest.n_cols && Q_dim[0] == m &&
        R_dim[1] > 0 && Q_dim[1] > 0) {
      const arma::mat RR(
        R_vec.begin(),
        static_cast<arma::uword>(R_dim[0]),
        static_cast<arma::uword>(R_dim[1]),
        false,
        true
      );
      const arma::mat QQ(
        Q_vec.begin(),
        static_cast<arma::uword>(Q_dim[0]),
        static_cast<arma::uword>(Q_dim[1]),
        false,
        true
      );
      bool latent_ok = true;
      for (arma::uword a = 0; a < length_ncomp; ++a) {
        const int mc = ncomp(a);
        if (mc < 1 ||
            mc > static_cast<int>(RR.n_cols) ||
            mc > static_cast<int>(QQ.n_cols)) {
          latent_ok = false;
          break;
        }
        arma::mat scores = Xtest * RR.cols(0, static_cast<arma::uword>(mc - 1));
        Ypred.slice(a) = scores * QQ.cols(0, static_cast<arma::uword>(mc - 1)).t();
        Ypred.slice(a).each_row() += mY;
      }
      used_latent_predict = latent_ok;
    }
  }

  if (!used_latent_predict &&
      use_latent_predict &&
      pls_method == "plssvd" &&
      !model.containsElementNamed("W_latent") &&
      model.containsElementNamed("C_latent")) {
    Rcpp::NumericVector R_vec = model["R"];
    Rcpp::NumericVector Q_vec = model["Q"];
    Rcpp::NumericVector C_vec = model["C_latent"];
    Rcpp::IntegerVector R_dim = R_vec.attr("dim");
    Rcpp::IntegerVector Q_dim = Q_vec.attr("dim");
    Rcpp::IntegerVector C_dim = C_vec.attr("dim");
    if (R_dim.size() == 2L && Q_dim.size() == 2L && C_dim.size() == 3L &&
        R_dim[0] == Xtest.n_cols && Q_dim[0] == m &&
        C_dim[0] == R_dim[1] && C_dim[1] == R_dim[1] &&
        C_dim[2] >= static_cast<int>(length_ncomp) &&
        R_dim[1] > 0 && Q_dim[1] == R_dim[1]) {
      const arma::mat RR(
        R_vec.begin(),
        static_cast<arma::uword>(R_dim[0]),
        static_cast<arma::uword>(R_dim[1]),
        false,
        true
      );
      const arma::mat QQ(
        Q_vec.begin(),
        static_cast<arma::uword>(Q_dim[0]),
        static_cast<arma::uword>(Q_dim[1]),
        false,
        true
      );
      const arma::cube CC(
        C_vec.begin(),
        static_cast<arma::uword>(C_dim[0]),
        static_cast<arma::uword>(C_dim[1]),
        static_cast<arma::uword>(C_dim[2]),
        false,
        true
      );
      bool latent_ok = true;
      for (arma::uword a = 0; a < length_ncomp; ++a) {
        const int mc = ncomp(a);
        if (mc < 1 ||
            mc > static_cast<int>(RR.n_cols) ||
            mc > static_cast<int>(QQ.n_cols) ||
            a >= CC.n_slices) {
          latent_ok = false;
          break;
        }
        arma::mat scores = Xtest * RR.cols(0, static_cast<arma::uword>(mc - 1));
        arma::mat coeff = CC.slice(a).submat(0, 0, mc - 1, mc - 1);
        Ypred.slice(a) = scores * coeff * QQ.cols(0, static_cast<arma::uword>(mc - 1)).t();
        Ypred.slice(a).each_row() += mY;
      }
      used_latent_predict = latent_ok;
    }
  }

  if (!used_latent_predict &&
      use_latent_predict &&
      pls_method == "plssvd" &&
      model.containsElementNamed("W_latent")) {
    Rcpp::NumericVector R_vec = model["R"];
    Rcpp::NumericVector W_vec = model["W_latent"];
    Rcpp::IntegerVector R_dim = R_vec.attr("dim");
    Rcpp::IntegerVector W_dim = W_vec.attr("dim");
    if (R_dim.size() == 2L && W_dim.size() == 3L &&
        R_dim[0] == Xtest.n_cols &&
        W_dim[0] == R_dim[1] && W_dim[1] == m &&
        W_dim[2] >= static_cast<int>(length_ncomp) &&
        R_dim[1] > 0) {
      const arma::mat RR(
        R_vec.begin(),
        static_cast<arma::uword>(R_dim[0]),
        static_cast<arma::uword>(R_dim[1]),
        false,
        true
      );
      const arma::cube WW(
        W_vec.begin(),
        static_cast<arma::uword>(W_dim[0]),
        static_cast<arma::uword>(W_dim[1]),
        static_cast<arma::uword>(W_dim[2]),
        false,
        true
      );
      bool latent_ok = true;
      for (arma::uword a = 0; a < length_ncomp; ++a) {
        const int mc = ncomp(a);
        if (mc < 1 ||
            mc > static_cast<int>(RR.n_cols) ||
            mc > static_cast<int>(WW.n_rows) ||
            a >= WW.n_slices) {
          latent_ok = false;
          break;
        }
        arma::mat scores = Xtest * RR.cols(0, static_cast<arma::uword>(mc - 1));
        Ypred.slice(a) = scores * WW.slice(a).rows(0, static_cast<arma::uword>(mc - 1));
        Ypred.slice(a).each_row() += mY;
      }
      used_latent_predict = latent_ok;
    }
  }

  if (!used_latent_predict) {
    if (!has_B) {
      Rcpp::stop("Model does not store `B`, and compact latent prediction was not available");
    }
    Rcpp::NumericVector B_vec = model["B"];
    Rcpp::IntegerVector B_dim = B_vec.attr("dim");
    if (B_dim.size() != 3L) {
      Rcpp::stop("Model coefficient array `B` must have 3 dimensions");
    }
    const arma::cube B(
      B_vec.begin(),
      static_cast<arma::uword>(B_dim[0]),
      static_cast<arma::uword>(B_dim[1]),
      static_cast<arma::uword>(B_dim[2]),
      false,
      true
    );
    if (B.n_slices < length_ncomp) {
      Rcpp::stop("Model coefficient array `B` has fewer slices than `ncomp`");
    }
    for (arma::uword a = 0; a < length_ncomp; ++a) {
      Ypred.slice(a) = Xtest * B.slice(a);
      Ypred.slice(a).each_row() += mY;
    }
  }

  arma::mat T_Xtest;
  if(proj){
    Rcpp::NumericVector RR_vec = model["R"];
    Rcpp::IntegerVector RR_dim = RR_vec.attr("dim");
    if (RR_dim.size() == 2L && RR_dim[0] > 0 && RR_dim[1] > 0) {
      const arma::mat RR(
        RR_vec.begin(),
        static_cast<arma::uword>(RR_dim[0]),
        static_cast<arma::uword>(RR_dim[1]),
        false,
        true
      );
      T_Xtest = Xtest*RR;
    } else {
      T_Xtest.set_size(w, 0);
    }
  }

  return List::create(
    Named("Ypred")  = Ypred,
    Named("Ttest")   = T_Xtest
  );
}

// [[Rcpp::export]]
List pls_predict_flash_cuda(List& model, arma::mat Xtest, bool proj) {
  if (!fastpls_svd::has_cuda_backend()) {
    Rcpp::stop("pls_predict_flash_cuda requires CUDA support");
  }

  const int m = Rcpp::as<int>(model["m"]);
  arma::ivec ncomp = Rcpp::as<arma::ivec>(model["ncomp"]);
  const arma::uword length_ncomp = static_cast<arma::uword>(ncomp.n_elem);

  Rcpp::NumericVector mX_vec = model["mX"];
  arma::rowvec mX(mX_vec.begin(), mX_vec.size(), false, true);
  Xtest.each_row() -= mX;
  Rcpp::NumericVector vX_vec = model["vX"];
  arma::rowvec vX(vX_vec.begin(), vX_vec.size(), false, true);
  Xtest.each_row() /= vX;
  Rcpp::NumericVector mY_vec = model["mY"];
  arma::rowvec mY(mY_vec.begin(), mY_vec.size(), false, true);

  std::string pls_method;
  if (model.containsElementNamed("pls_method")) {
    pls_method = Rcpp::as<std::string>(model["pls_method"]);
  }

  Rcpp::NumericVector R_vec = model["R"];
  Rcpp::IntegerVector R_dim = R_vec.attr("dim");
  if (R_dim.size() != 2L || R_dim[0] != Xtest.n_cols || R_dim[1] < 1) {
    Rcpp::stop("Model `R` is not compatible with CUDA flash prediction");
  }
  const arma::mat RR(
    R_vec.begin(),
    static_cast<arma::uword>(R_dim[0]),
    static_cast<arma::uword>(R_dim[1]),
    false,
    true
  );
  const int kmax = static_cast<int>(RR.n_cols);

  arma::cube Wflash;
  if ((pls_method == "simpls" || pls_method == "simpls_fast") &&
      model.containsElementNamed("Q")) {
    Rcpp::NumericVector Q_vec = model["Q"];
    Rcpp::IntegerVector Q_dim = Q_vec.attr("dim");
    if (Q_dim.size() != 2L || Q_dim[0] != m || Q_dim[1] < 1) {
      Rcpp::stop("Model `Q` is not compatible with CUDA flash prediction");
    }
    const arma::mat QQ(
      Q_vec.begin(),
      static_cast<arma::uword>(Q_dim[0]),
      static_cast<arma::uword>(Q_dim[1]),
      false,
      true
    );
    Wflash.zeros(kmax, m, length_ncomp);
    for (arma::uword a = 0; a < length_ncomp; ++a) {
      int mc = ncomp(a);
      if (mc < 1 || mc > kmax || mc > static_cast<int>(QQ.n_cols)) {
        Rcpp::stop("ncomp exceeds latent rank for CUDA flash prediction");
      }
      Wflash.slice(a).rows(0, static_cast<arma::uword>(mc - 1)) =
        QQ.cols(0, static_cast<arma::uword>(mc - 1)).t();
    }
  } else if (pls_method == "plssvd" && model.containsElementNamed("W_latent")) {
    Rcpp::NumericVector W_vec = model["W_latent"];
    Rcpp::IntegerVector W_dim = W_vec.attr("dim");
    if (W_dim.size() != 3L || W_dim[0] != kmax || W_dim[1] != m ||
        W_dim[2] < static_cast<int>(length_ncomp)) {
      Rcpp::stop("Model `W_latent` is not compatible with CUDA flash prediction");
    }
    const arma::cube WW(
      W_vec.begin(),
      static_cast<arma::uword>(W_dim[0]),
      static_cast<arma::uword>(W_dim[1]),
      static_cast<arma::uword>(W_dim[2]),
      false,
      true
    );
    Wflash = WW.slices(0, length_ncomp - 1);
  } else if (pls_method == "plssvd" &&
             model.containsElementNamed("C_latent") &&
             model.containsElementNamed("Q")) {
    Rcpp::NumericVector Q_vec = model["Q"];
    Rcpp::NumericVector C_vec = model["C_latent"];
    Rcpp::IntegerVector Q_dim = Q_vec.attr("dim");
    Rcpp::IntegerVector C_dim = C_vec.attr("dim");
    if (Q_dim.size() != 2L || C_dim.size() != 3L ||
        Q_dim[0] != m || Q_dim[1] != kmax ||
        C_dim[0] != kmax || C_dim[1] != kmax ||
        C_dim[2] < static_cast<int>(length_ncomp)) {
      Rcpp::stop("Model latent PLSSVD factors are not compatible with CUDA flash prediction");
    }
    const arma::mat QQ(
      Q_vec.begin(),
      static_cast<arma::uword>(Q_dim[0]),
      static_cast<arma::uword>(Q_dim[1]),
      false,
      true
    );
    const arma::cube CC(
      C_vec.begin(),
      static_cast<arma::uword>(C_dim[0]),
      static_cast<arma::uword>(C_dim[1]),
      static_cast<arma::uword>(C_dim[2]),
      false,
      true
    );
    Wflash.zeros(kmax, m, length_ncomp);
    for (arma::uword a = 0; a < length_ncomp; ++a) {
      int mc = ncomp(a);
      if (mc < 1 || mc > kmax) {
        Rcpp::stop("ncomp exceeds latent rank for CUDA flash prediction");
      }
      arma::mat Cmc = CC.slice(a).submat(0, 0, mc - 1, mc - 1);
      Wflash.slice(a).rows(0, static_cast<arma::uword>(mc - 1)) =
        Cmc * QQ.cols(0, static_cast<arma::uword>(mc - 1)).t();
    }
  } else {
    Rcpp::stop("CUDA flash prediction requires compact low-rank factors");
  }

  arma::cube Ypred = fastpls_svd::cuda_flash_lowrank_predict(
    Xtest,
    RR,
    Wflash,
    mY,
    ncomp
  );

  arma::mat T_Xtest;
  if (proj) {
    T_Xtest = Xtest * RR;
  }

  return List::create(
    Named("Ypred") = Ypred,
    Named("Ttest") = T_Xtest,
    Named("predict_backend") = "cuda_flash"
  );
}

// [[Rcpp::export]]
List pls_predict_flash_cpu(List& model, arma::mat Xtest, bool proj, int block_size) {
  const int m = Rcpp::as<int>(model["m"]);
  arma::ivec ncomp = Rcpp::as<arma::ivec>(model["ncomp"]);
  const arma::uword length_ncomp = static_cast<arma::uword>(ncomp.n_elem);

  Rcpp::NumericVector mX_vec = model["mX"];
  arma::rowvec mX(mX_vec.begin(), mX_vec.size(), false, true);
  Xtest.each_row() -= mX;
  Rcpp::NumericVector vX_vec = model["vX"];
  arma::rowvec vX(vX_vec.begin(), vX_vec.size(), false, true);
  Xtest.each_row() /= vX;
  Rcpp::NumericVector mY_vec = model["mY"];
  arma::rowvec mY(mY_vec.begin(), mY_vec.size(), false, true);

  std::string pls_method;
  if (model.containsElementNamed("pls_method")) {
    pls_method = Rcpp::as<std::string>(model["pls_method"]);
  }

  Rcpp::NumericVector R_vec = model["R"];
  Rcpp::IntegerVector R_dim = R_vec.attr("dim");
  if (R_dim.size() != 2L || R_dim[0] != Xtest.n_cols || R_dim[1] < 1) {
    Rcpp::stop("Model `R` is not compatible with CPU flash prediction");
  }
  const arma::mat RR(
    R_vec.begin(),
    static_cast<arma::uword>(R_dim[0]),
    static_cast<arma::uword>(R_dim[1]),
    false,
    true
  );
  const int kmax = static_cast<int>(RR.n_cols);

  arma::cube Wflash(kmax, static_cast<arma::uword>(m), length_ncomp, arma::fill::zeros);
  if ((pls_method == "simpls" || pls_method == "simpls_fast") &&
      model.containsElementNamed("Q")) {
    Rcpp::NumericVector Q_vec = model["Q"];
    Rcpp::IntegerVector Q_dim = Q_vec.attr("dim");
    if (Q_dim.size() != 2L || Q_dim[0] != m || Q_dim[1] < 1) {
      Rcpp::stop("Model `Q` is not compatible with CPU flash prediction");
    }
    const arma::mat QQ(
      Q_vec.begin(),
      static_cast<arma::uword>(Q_dim[0]),
      static_cast<arma::uword>(Q_dim[1]),
      false,
      true
    );
    for (arma::uword a = 0; a < length_ncomp; ++a) {
      const int mc = ncomp(a);
      if (mc < 1 || mc > kmax || mc > static_cast<int>(QQ.n_cols)) {
        Rcpp::stop("ncomp exceeds latent rank for CPU flash prediction");
      }
      Wflash.slice(a).rows(0, static_cast<arma::uword>(mc - 1)) =
        QQ.cols(0, static_cast<arma::uword>(mc - 1)).t();
    }
  } else if (pls_method == "plssvd" && model.containsElementNamed("W_latent")) {
    Rcpp::NumericVector W_vec = model["W_latent"];
    Rcpp::IntegerVector W_dim = W_vec.attr("dim");
    if (W_dim.size() != 3L || W_dim[0] != kmax || W_dim[1] != m ||
        W_dim[2] < static_cast<int>(length_ncomp)) {
      Rcpp::stop("Model `W_latent` is not compatible with CPU flash prediction");
    }
    const arma::cube WW(
      W_vec.begin(),
      static_cast<arma::uword>(W_dim[0]),
      static_cast<arma::uword>(W_dim[1]),
      static_cast<arma::uword>(W_dim[2]),
      false,
      true
    );
    Wflash = WW.slices(0, length_ncomp - 1);
  } else if (pls_method == "plssvd" &&
             model.containsElementNamed("C_latent") &&
             model.containsElementNamed("Q")) {
    Rcpp::NumericVector Q_vec = model["Q"];
    Rcpp::NumericVector C_vec = model["C_latent"];
    Rcpp::IntegerVector Q_dim = Q_vec.attr("dim");
    Rcpp::IntegerVector C_dim = C_vec.attr("dim");
    if (Q_dim.size() != 2L || C_dim.size() != 3L ||
        Q_dim[0] != m || Q_dim[1] != kmax ||
        C_dim[0] != kmax || C_dim[1] != kmax ||
        C_dim[2] < static_cast<int>(length_ncomp)) {
      Rcpp::stop("Model latent PLSSVD factors are not compatible with CPU flash prediction");
    }
    const arma::mat QQ(
      Q_vec.begin(),
      static_cast<arma::uword>(Q_dim[0]),
      static_cast<arma::uword>(Q_dim[1]),
      false,
      true
    );
    const arma::cube CC(
      C_vec.begin(),
      static_cast<arma::uword>(C_dim[0]),
      static_cast<arma::uword>(C_dim[1]),
      static_cast<arma::uword>(C_dim[2]),
      false,
      true
    );
    for (arma::uword a = 0; a < length_ncomp; ++a) {
      const int mc = ncomp(a);
      if (mc < 1 || mc > kmax) {
        Rcpp::stop("ncomp exceeds latent rank for CPU flash prediction");
      }
      arma::mat Cmc = CC.slice(a).submat(0, 0, mc - 1, mc - 1);
      Wflash.slice(a).rows(0, static_cast<arma::uword>(mc - 1)) =
        Cmc * QQ.cols(0, static_cast<arma::uword>(mc - 1)).t();
    }
  } else {
    Rcpp::stop("CPU flash prediction requires compact low-rank factors");
  }

  const arma::uword ntest = Xtest.n_rows;
  arma::cube Ypred(ntest, static_cast<arma::uword>(m), length_ncomp, arma::fill::none);
  arma::mat T_Xtest;
  if (proj) {
    T_Xtest.set_size(ntest, static_cast<arma::uword>(kmax));
  }

  arma::uword bs = static_cast<arma::uword>(block_size > 0 ? block_size : 4096);
  if (bs == 0 || bs > ntest) {
    bs = ntest;
  }

  for (arma::uword start = 0; start < ntest; start += bs) {
    const arma::uword stop = std::min(start + bs - 1, ntest - 1);
    const arma::mat Xblock = Xtest.rows(start, stop);
    const arma::mat scores = Xblock * RR;
    if (proj) {
      T_Xtest.rows(start, stop) = scores;
    }
    for (arma::uword a = 0; a < length_ncomp; ++a) {
      const int mc = ncomp(a);
      arma::mat Yblock =
        scores.cols(0, static_cast<arma::uword>(mc - 1)) *
        Wflash.slice(a).rows(0, static_cast<arma::uword>(mc - 1));
      Yblock.each_row() += mY;
      Ypred.slice(a).rows(start, stop) = Yblock;
    }
  }

  return List::create(
    Named("Ypred") = Ypred,
    Named("Ttest") = T_Xtest,
    Named("predict_backend") = "cpu_flash"
  );
}

// [[Rcpp::export]]
arma::mat kernel_matrix_cpp(
  const arma::mat& X1,
  const arma::mat& X2,
  const int kernel,
  const double gamma,
  const int degree,
  const double coef0
) {
  if (X1.n_cols != X2.n_cols) {
    Rcpp::stop("X1 and X2 must have the same number of columns");
  }
  if (kernel == 1) {
    return X1 * X2.t();
  }
  arma::mat dots = X1 * X2.t();
  if (kernel == 3) {
    return arma::pow(gamma * dots + coef0, degree);
  }
  if (kernel != 2) {
    Rcpp::stop("Unknown kernel id");
  }

  arma::vec n1 = arma::sum(arma::square(X1), 1);
  arma::rowvec n2 = arma::sum(arma::square(X2), 1).t();
  arma::mat dist2 = arma::repmat(n1, 1, X2.n_rows) + arma::repmat(n2, X1.n_rows, 1) - 2.0 * dots;
  dist2.transform([](double v) { return v < 0.0 && v > -1e-10 ? 0.0 : v; });
  return arma::exp(-gamma * dist2);
}

// [[Rcpp::export]]
Rcpp::List center_kernel_train_cpp(const arma::mat& K) {
  arma::rowvec col_means = arma::mean(K, 0);
  arma::vec row_means = arma::mean(K, 1);
  const double grand_mean = arma::mean(col_means);
  arma::mat Kc = K;
  Kc.each_row() -= col_means;
  Kc.each_col() -= row_means;
  Kc += grand_mean;
  return Rcpp::List::create(
    Rcpp::Named("K") = Kc,
    Rcpp::Named("col_means") = col_means,
    Rcpp::Named("grand_mean") = grand_mean
  );
}

// [[Rcpp::export]]
arma::mat center_kernel_test_cpp(
  const arma::mat& Ktest,
  const arma::rowvec& train_col_means,
  const double train_grand_mean
) {
  if (Ktest.n_cols != train_col_means.n_cols) {
    Rcpp::stop("Ktest columns must match the training kernel size");
  }
  arma::vec row_means = arma::mean(Ktest, 1);
  arma::mat Kc = Ktest;
  Kc.each_row() -= train_col_means;
  Kc.each_col() -= row_means;
  Kc += train_grand_mean;
  return Kc;
}

// [[Rcpp::export]]
Rcpp::List opls_filter_cpp(arma::mat X, arma::mat Y, const int north, const int scaling) {
  if (X.n_rows != Y.n_rows) {
    Rcpp::stop("X and Y must have the same number of rows");
  }
  if (north < 0) {
    Rcpp::stop("north must be >= 0");
  }

  arma::rowvec mX(X.n_cols, arma::fill::zeros);
  if (scaling < 3) {
    mX = arma::mean(X, 0);
    X.each_row() -= mX;
  }
  arma::rowvec vX(X.n_cols, arma::fill::ones);
  if (scaling == 2) {
    vX = arma::stddev(X, 0, 0);
    for (arma::uword j = 0; j < vX.n_elem; ++j) {
      if (!std::isfinite(vX[j]) || vX[j] == 0.0) {
        vX[j] = 1.0;
      }
    }
    X.each_row() /= vX;
  }

  arma::rowvec mY = arma::mean(Y, 0);
  Y.each_row() -= mY;

  arma::mat W_orth(X.n_cols, static_cast<arma::uword>(north), arma::fill::zeros);
  arma::mat P_orth(X.n_cols, static_cast<arma::uword>(north), arma::fill::zeros);
  int used = 0;

  for (int a = 0; a < north; ++a) {
    arma::mat S = X.t() * Y;
    arma::mat U;
    arma::vec d;
    arma::mat V;
    const bool ok = arma::svd_econ(U, d, V, S);
    if (!ok || U.n_cols == 0) break;

    arma::vec w = U.col(0);
    const double w_norm = arma::norm(w, 2);
    if (!std::isfinite(w_norm) || w_norm <= 0.0) break;
    w /= w_norm;

    arma::vec t = X * w;
    const double t_ss = arma::dot(t, t);
    if (!std::isfinite(t_ss) || t_ss <= 0.0) break;

    arma::vec p = X.t() * t / t_ss;
    const double ww = arma::dot(w, w);
    arma::vec w_orth = p - w * (arma::dot(w, p) / ww);
    const double wo_norm = arma::norm(w_orth, 2);
    if (!std::isfinite(wo_norm) || wo_norm <= 0.0) break;
    w_orth /= wo_norm;

    arma::vec t_orth = X * w_orth;
    const double to_ss = arma::dot(t_orth, t_orth);
    if (!std::isfinite(to_ss) || to_ss <= 0.0) break;
    arma::vec p_orth = X.t() * t_orth / to_ss;

    X -= t_orth * p_orth.t();
    W_orth.col(static_cast<arma::uword>(used)) = w_orth;
    P_orth.col(static_cast<arma::uword>(used)) = p_orth;
    ++used;
  }

  if (used < north) {
    W_orth = W_orth.cols(0, used > 0 ? static_cast<arma::uword>(used - 1) : 0);
    P_orth = P_orth.cols(0, used > 0 ? static_cast<arma::uword>(used - 1) : 0);
    if (used == 0) {
      W_orth.set_size(X.n_cols, 0);
      P_orth.set_size(X.n_cols, 0);
    }
  }

  return Rcpp::List::create(
    Rcpp::Named("X") = X,
    Rcpp::Named("mX") = mX,
    Rcpp::Named("vX") = vX,
    Rcpp::Named("W_orth") = W_orth,
    Rcpp::Named("P_orth") = P_orth,
    Rcpp::Named("north") = used
  );
}

// [[Rcpp::export]]
arma::mat opls_apply_filter_cpp(
  arma::mat X,
  const arma::rowvec& mX,
  const arma::rowvec& vX,
  const arma::mat& W_orth,
  const arma::mat& P_orth
) {
  if (X.n_cols != mX.n_cols || X.n_cols != vX.n_cols) {
    Rcpp::stop("X columns must match stored OPLS preprocessing");
  }
  X.each_row() -= mX;
  X.each_row() /= vX;
  if (W_orth.n_cols != P_orth.n_cols || W_orth.n_rows != X.n_cols || P_orth.n_rows != X.n_cols) {
    Rcpp::stop("Invalid OPLS orthogonal filter dimensions");
  }
  for (arma::uword a = 0; a < W_orth.n_cols; ++a) {
    arma::vec t_orth = X * W_orth.col(a);
    X -= t_orth * P_orth.col(a).t();
  }
  return X;
}


// [[Rcpp::export]]
int unic(arma::mat x){
  int x_size=x.size();
  for(int i=0;i<x_size;i++){
    if(x(i)!=x(0))
      return 2;
  }
  return 1;
}


// This function performs a random selection of the elements of a vector "yy".
// The number of elements to select is defined by the variable "size".

IntegerVector samplewithoutreplace(IntegerVector yy,int size){
  IntegerVector xx(size);
  int rest=yy.size();
  int it;
  for(int ii=0;ii<size;ii++){
    it=unif_rand()*rest;
    xx[ii]=yy[it];
    yy.erase(it);
    rest--;
  }
  return xx;
}



// [[Rcpp::export]]
List optim_pls_cv(
  arma::mat Xdata,
  arma::mat Ydata,
  arma::ivec constrain,
  arma::ivec ncomp,
  int scaling,
  int kfold,
  int method,
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed
) {
  if (method == 1) {
    const int max_plssvd_rank = std::min(static_cast<int>(Xdata.n_rows),
      std::min(static_cast<int>(Xdata.n_cols), static_cast<int>(Ydata.n_cols)));
    for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
      if (ncomp(i) > max_plssvd_rank) {
        ncomp(i) = max_plssvd_rank;
      }
      if (ncomp(i) < 1) {
        ncomp(i) = 1;
      }
    }
  }
  
  int length_ncomp=ncomp.n_elem;
  
  int nsamples=Xdata.n_rows;
  
  int ncolY=Ydata.n_cols;
  arma::cube Ypred(nsamples,ncolY,length_ncomp); 
  //int xsa_t = max(constrain);

  arma::ivec indices = unique(constrain);

  arma::ivec constrain2=constrain;
  

  for (arma::uword j = 0; j < indices.size(); ++j) {
    arma::uvec ind = arma::find(constrain == indices(j));
    
    constrain2.elem(ind).fill(j + 1);
  }
  
  int xsa_t = indices.size();
  
  
  IntegerVector frame = seq_len(xsa_t);
  IntegerVector v=samplewithoutreplace(frame,xsa_t);
  int mm=constrain2.size();
  
  arma::ivec fold(mm);
  for (int i=0; i<mm; i++) 
    fold[i]=v[constrain2(i)-1]%kfold;
  
  for (int i=0; i<kfold; i++) {
    
    arma::uvec w1,w9;
    arma::ivec temp;
    arma::mat Xtrain,Xtest;
    arma::mat Ytrain;
    w1=find(fold==i);
    w9=find(fold!=i);
    int w1_size=w1.size();
    
    Xtrain=Xdata.rows(w9);
    
    Xtest=Xdata.rows(w1);
    Ytrain=Ydata.rows(w9);
    List model;
    if(method==1){
      model=pls_model1(Xtrain,Ytrain,ncomp,scaling,FALSE,svd_method,rsvd_oversample,rsvd_power,svds_tol,seed);
    }
    if(method==2){
      model=pls_model2(Xtrain,Ytrain,ncomp,scaling,FALSE,svd_method,rsvd_oversample,rsvd_power,svds_tol,seed);
    }
    if(method==3){
      model=pls_model2_fast(Xtrain,Ytrain,ncomp,scaling,FALSE,svd_method,rsvd_oversample,rsvd_power,svds_tol,seed);
    }
    List pls=pls_predict(model,Xtest,FALSE);
    arma::cube temp1=pls("Ypred");
    for(int ii=0;ii<w1_size;ii++)  for(int jj=0;jj<length_ncomp;jj++)  for(int kk=0;kk<ncolY;kk++)  Ypred(w1[ii],kk,jj)=temp1(ii,kk,jj);  
    
  }  
  List model_all;
  if(method==1){
    model_all=pls_model1(Xdata,Ydata,ncomp,scaling,TRUE,svd_method,rsvd_oversample,rsvd_power,svds_tol,seed);
  }
  if(method==2){
    model_all=pls_model2(Xdata,Ydata,ncomp,scaling,TRUE, svd_method,rsvd_oversample,rsvd_power,svds_tol,seed);
  }
  if(method==3){
    model_all=pls_model2_fast(Xdata,Ydata,ncomp,scaling,TRUE, svd_method,rsvd_oversample,rsvd_power,svds_tol,seed);
  }
  arma::vec R2Y=model_all("R2Y");
  arma::vec Q2Y(length_ncomp);
  
  int j=0;
  for(int i=0;i<length_ncomp;i++){
    arma::mat Ypred_i=Ypred.slice(i);
    Q2Y(i)=RQ(Ydata,Ypred_i);
    if(Q2Y(i)>Q2Y(j)) j=i;
  }
  arma::vec optim_c(1);
  optim_c(0)=ncomp(j);
  return List::create(
    Named("optim_comp") = optim_c,
    Named("Ypred")      = Ypred,
    Named("Q2Y")        = Q2Y,
    Named("R2Y")        = R2Y,
    Named("fold")       = fold
  );
  
}



// [[Rcpp::export]]
List double_pls_cv(
  arma::mat Xdata,
  arma::mat Ydata,
  arma::ivec ncomp,
  arma::ivec constrain,
  int scaling,
  int kfold_inner,
  int kfold_outer,
  int method,
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed
) {
  
  int nsamples=Xdata.n_rows;
  
  int ncolY=Ydata.n_cols;
  arma::mat Ypred(nsamples,ncolY); 
  
  arma::ivec indices = unique(constrain);
  
  arma::ivec constrain2=constrain;
  

  for (arma::uword j = 0; j < indices.size(); ++j) {
    arma::uvec ind = arma::find(constrain == indices(j));
    
    constrain2.elem(ind).fill(j + 1);
  }
  
  int xsa_t = indices.size();
  
  
  IntegerVector frame = seq_len(xsa_t);
  IntegerVector v=samplewithoutreplace(frame,xsa_t);
  int mm=constrain2.size();
  
  arma::ivec fold(mm);
  for (int i=0; i<mm; i++) 
    fold[i]=v[constrain2(i)-1]%kfold_outer;

  
  
  
  // We have different R2Y for each cycle of cross-validation
  // because it could change the optimized value of components
  arma::vec R2Y(kfold_outer);
  arma::ivec best_comp(kfold_outer);
  for (int i=0; i<kfold_outer; i++) {
    
    arma::uvec w1,w9;
    arma::ivec temp;
    arma::mat Xtrain,Xtest;
    arma::mat Ytrain;
    w1=find(fold==i);
    w9=find(fold!=i);
    int w1_size=w1.size();
    
    Xtrain=Xdata.rows(w9);
    Xtest=Xdata.rows(w1);
    Ytrain=Ydata.rows(w9);
    arma::ivec constrain_train=constrain.elem(w9);
    
    List opt=optim_pls_cv(
      Xtrain,
      Ytrain,
      constrain_train,
      ncomp,
      scaling,
      kfold_inner,
      method,
      svd_method,
      rsvd_oversample,
      rsvd_power,
      svds_tol,
      seed
    );
      
    List model;
    if(method==1){
      model=pls_model1(Xtrain,Ytrain,opt("optim_comp"),scaling,FALSE, svd_method,rsvd_oversample,rsvd_power,svds_tol,seed);
    }
    if(method==2){
      model=pls_model2(Xtrain,Ytrain,opt("optim_comp"),scaling,FALSE, svd_method,rsvd_oversample,rsvd_power,svds_tol,seed);
    }
    if(method==3){
      model=pls_model2_fast(Xtrain,Ytrain,opt("optim_comp"),scaling,FALSE, svd_method,rsvd_oversample,rsvd_power,svds_tol,seed);
    }
      
    List pls=pls_predict(model,Xtest,FALSE);
    arma::cube temp1=pls("Ypred");
    for(int ii=0;ii<w1_size;ii++)  for(int kk=0;kk<ncolY;kk++)  Ypred(w1[ii],kk)=temp1(ii,kk,0);  
    
    // Calculation of R2Y
    List model_all;
    if(method==1){
      model_all=pls_model1(Xtrain,Ytrain,opt("optim_comp"),scaling,TRUE, svd_method,rsvd_oversample,rsvd_power,svds_tol,seed);
    }
    if(method==2){
      model_all=pls_model2(Xtrain,Ytrain,opt("optim_comp"),scaling,TRUE, svd_method,rsvd_oversample,rsvd_power,svds_tol,seed);
    }
    if(method==3){
      model_all=pls_model2_fast(Xtrain,Ytrain,opt("optim_comp"),scaling,TRUE, svd_method,rsvd_oversample,rsvd_power,svds_tol,seed);
    }
    
    
    R2Y(i)=model_all("R2Y");
    best_comp(i)=opt("optim_comp");
  }  
  

  double Q2Y;
  
  
  Q2Y=RQ(Ydata,Ypred);
  
  return List::create(
    Named("Ypred")      = Ypred,
    Named("Q2Y")        = Q2Y,
    Named("R2Y")        = R2Y,
    Named("optim_comp") = best_comp
  );
  
}



// [[Rcpp::export]]
List pls_model1(
  arma::mat Xtrain,
  arma::mat Ytrain,
  arma::ivec ncomp,
  int scaling,
  bool fit,
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed
) {
  
  // n <-dim(Xtrain)[1]
  int n = Xtrain.n_rows;
  
  // p <-dim(Xtrain)[2]
  int p = Xtrain.n_cols;
  
  // m <- dim(Y)[2]
  int m = Ytrain.n_cols;
  int max_plssvd_rank = std::min(n, std::min(p, m));
  int length_ncomp=ncomp.n_elem;
  for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
    if (ncomp(i) > max_plssvd_rank) {
      ncomp(i) = max_plssvd_rank;
    }
    if (ncomp(i) < 1) {
      ncomp(i) = 1;
    }
  }
  int max_ncomp=max(ncomp);
  int max_ncomp_eff = std::min(max_ncomp, max_plssvd_rank);
  if (max_ncomp_eff < 1) {
    stop("plssvd effective rank is < 1");
  }
  
  // Xtrain <- scale(Xtrain,center=TRUE,scale=FALSE)
  // Xtest <-scale(Xtest,center=mX)
  arma::mat mX(1,p); 
  mX.zeros();
  if(scaling<3){
    mX=mean(Xtrain,0);
    Xtrain.each_row()-=mX;
  } 
  arma::mat vX(1,p); 
  vX.ones();
  if(scaling==2){
    vX=variance(Xtrain); 
    Xtrain.each_row()/=vX;
  }
  
  // Y <- scale(Ytrain,center=TRUE,scale=FALSE)
  arma::mat mY=mean(Ytrain,0);
  Ytrain.each_row()-=mY;
  
  // S <- crossprod(X,Y)
  arma::mat S=trans(Xtrain)*Ytrain;
  
  arma::mat svd_u;
  arma::vec svd_s;
  arma::mat svd_v;
  
  fastpls_svd::SVDResult svd_res = compute_truncated_svd_dispatch(
    S,
    max_ncomp_eff,
    svd_method,
    rsvd_oversample,
    rsvd_power,
    svds_tol,
    static_cast<unsigned int>(seed),
    false,
    plssvd_use_small_exact_svd(max_plssvd_rank)
  );
  svd_u = svd_res.U;
  svd_s = svd_res.s;
  svd_v = svd_res.Vt.t();

  const bool store_B = should_store_coefficients(p, m, length_ncomp, true);
  arma::cube B;
  if (store_B) {
    B.set_size(p, m, length_ncomp);
    B.zeros();
  }
  arma::cube C_latent(max_ncomp_eff, max_ncomp_eff, length_ncomp, arma::fill::zeros);
  arma::cube W_latent(max_ncomp_eff, m, length_ncomp, arma::fill::zeros);
  arma::cube Yfit;
  if(fit){
    Yfit.resize(n,m,length_ncomp);
  }

  max_ncomp_eff = std::min(max_ncomp_eff, static_cast<int>(svd_u.n_cols));
  if(svd_v.n_cols > 0){
    max_ncomp_eff = std::min(max_ncomp_eff, static_cast<int>(svd_v.n_cols));
  }
  if (max_ncomp_eff < 1) {
    stop("plssvd effective rank is < 1 after SVD");
  }
  svd_u = svd_u.cols(0,max_ncomp_eff-1);
  if (svd_v.n_cols > static_cast<arma::uword>(max_ncomp_eff)) {
    svd_v = svd_v.cols(0,max_ncomp_eff-1);
  }
  arma::mat svd_u_eff = svd_u;
  arma::mat svd_v_eff = svd_v;
  arma::mat T_eff = Xtrain*svd_u_eff;
  arma::mat T = T_eff;

  arma::vec R2Y(length_ncomp);
  const int plssvd_optimized = env_int_or("FASTPLS_PLSSVD_OPTIMIZED", 1, 0, 1);
  arma::mat G_full;
  if (plssvd_optimized == 1) {
    G_full = T_eff.t() * T_eff;
  }

  for (int a=0; a<length_ncomp; a++) {
    int mc=ncomp(a);
    int mc_eff = std::min(mc, max_ncomp_eff);
    arma::mat svd_u_mc = svd_u_eff.cols(0,mc_eff-1);
    arma::mat svd_v_mc = svd_v_eff.cols(0,mc_eff-1);
    arma::mat T_a = T_eff.cols(0,mc_eff-1);

    if (plssvd_optimized == 1) {
      arma::mat G_a = G_full.submat(0, 0, mc_eff - 1, mc_eff - 1);
      arma::mat D_a(mc_eff, mc_eff, fill::zeros);
      D_a.diag() = svd_s.subvec(0, mc_eff - 1);

      arma::mat coeff_latent;
      bool solved = arma::solve(coeff_latent, G_a, D_a, arma::solve_opts::likely_sympd);
      if (!solved) {
        solved = arma::solve(coeff_latent, G_a, D_a);
      }
      if (!solved) {
        stop("plssvd latent solve failed");
      }

      C_latent.slice(a).submat(0, 0, mc_eff - 1, mc_eff - 1) = coeff_latent;
      arma::mat W_a = coeff_latent * svd_v_mc.t();
      W_latent.slice(a).submat(0, 0, mc_eff - 1, m - 1) = W_a;
      if (store_B) {
        B.slice(a) = svd_u_mc * W_a;
      }
      if(fit){
        arma::mat temp1 = T_a * W_a;
        R2Y(a)=RQ(Ytrain,temp1);
        temp1.each_row()+=mY;
        Yfit.slice(a)=temp1;
      }
    } else {
      arma::mat U = Ytrain * svd_v_mc;
      arma::mat T_at = T_a.t();
      arma::mat gram = T_at * T_a;
      arma::mat rhs = T_at * U;
      arma::mat coeff_latent;
      bool solved = arma::solve(coeff_latent, gram, rhs, arma::solve_opts::likely_sympd);
      if (!solved) {
        solved = arma::solve(coeff_latent, gram, rhs);
      }
      if (!solved) {
        stop("plssvd legacy latent solve failed");
      }
      arma::mat D_a(mc_eff, mc_eff, fill::zeros);
      D_a.diag() = svd_s.subvec(0, mc_eff - 1);
      arma::mat coeff_for_predict;
      bool predict_solved = arma::solve(coeff_for_predict, gram, D_a, arma::solve_opts::likely_sympd);
      if (!predict_solved) {
        predict_solved = arma::solve(coeff_for_predict, gram, D_a);
      }
      if (predict_solved) {
        C_latent.slice(a).submat(0, 0, mc_eff - 1, mc_eff - 1) = coeff_for_predict;
      }
      arma::mat W_a = coeff_latent * svd_v_mc.t();
      W_latent.slice(a).submat(0, 0, mc_eff - 1, m - 1) = W_a;
      if (store_B) {
        B.slice(a)= svd_u_mc * W_a;
      }
      if(fit){
        arma::mat temp1=T_a * W_a;
        R2Y(a)=RQ(Ytrain,temp1);
        temp1.each_row()+=mY;
        Yfit.slice(a)=temp1;
      }
    }
  }



  List out = List::create(
    Named("C_latent") = C_latent,
    Named("W_latent") = W_latent,
    Named("Q")       = svd_v_eff,
    Named("Ttrain")  = T,
    Named("R")       = svd_u_eff,
    Named("mX")      = mX,
    Named("vX")      = vX,
    Named("mY")      = mY,
    Named("p")       = p,
    Named("m")       = m,
    Named("ncomp")   = ncomp,
    Named("Yfit")    = Yfit,
    Named("R2Y")     = R2Y
  );
  if (store_B) {
    out["B"] = B;
  }
  annotate_coefficient_storage(out, store_B);
  return out;
}

List pls_model1_rsvd_xprod_precision_view_impl(
  SEXP XtrainSEXP,
  SEXP YtrainSEXP,
  arma::ivec ncomp,
  int scaling,
  bool fit,
  int rsvd_oversample,
  int rsvd_power,
  int seed
) {
  const arma::mat Xview = numeric_matrix_view(XtrainSEXP, "Xtrain");
  const arma::mat Yview = numeric_matrix_view(YtrainSEXP, "Ytrain");
  const int n = Xview.n_rows;
  const int p = Xview.n_cols;
  const int m = Yview.n_cols;
  if (Yview.n_rows != static_cast<arma::uword>(n)) {
    stop("Xtrain and Ytrain must have the same number of rows");
  }

  const int max_plssvd_rank = std::min(n, std::min(p, m));
  const int length_ncomp = ncomp.n_elem;
  for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
    if (ncomp(i) > max_plssvd_rank) ncomp(i) = max_plssvd_rank;
    if (ncomp(i) < 1) ncomp(i) = 1;
  }

  const int max_ncomp = max(ncomp);
  int max_ncomp_eff = std::min(max_ncomp, max_plssvd_rank);
  if (max_ncomp_eff < 1) {
    stop("plssvd effective rank is < 1");
  }

  arma::rowvec mX_row(p, fill::zeros);
  if (scaling < 3) {
    mX_row = mean(Xview, 0);
  }
  arma::rowvec vX_row(p, fill::ones);
  if (scaling == 2) {
    vX_row = variance_nocopy(Xview);
  }
  arma::rowvec mY_row = mean(Yview, 0);

  CenterScaleMatrixView Xop{Xview, mX_row, vX_row};
  CenterOnlyMatrixView Yop{Yview, mY_row};

  fastpls_svd::SVDResult svd_res = truncated_rsvd_crossprod_double_view(
    Xop,
    Yop,
    max_ncomp_eff,
    rsvd_oversample,
    rsvd_power,
    static_cast<unsigned int>(seed),
    false,
    plssvd_use_small_exact_svd(max_plssvd_rank)
  );

  arma::mat svd_u = svd_res.U;
  arma::vec svd_s = svd_res.s;
  arma::mat svd_v = svd_res.Vt.t();

  const bool store_B = should_store_coefficients(p, m, length_ncomp, true);
  arma::cube B;
  if (store_B) {
    B.zeros(p, m, length_ncomp);
  }
  arma::cube Yfit;
  if (fit) {
    Yfit.set_size(n, m, length_ncomp);
  }

  max_ncomp_eff = std::min(max_ncomp_eff, static_cast<int>(svd_u.n_cols));
  if (svd_v.n_cols > 0) {
    max_ncomp_eff = std::min(max_ncomp_eff, static_cast<int>(svd_v.n_cols));
  }
  if (max_ncomp_eff < 1) {
    stop("plssvd effective rank is < 1 after SVD");
  }

  svd_u = svd_u.cols(0, max_ncomp_eff - 1);
  if (svd_v.n_cols > static_cast<arma::uword>(max_ncomp_eff)) {
    svd_v = svd_v.cols(0, max_ncomp_eff - 1);
  }

  arma::mat T_eff = Xop.times(svd_u);
  arma::mat G_full = T_eff.t() * T_eff;
  arma::cube C_latent(max_ncomp_eff, max_ncomp_eff, length_ncomp, arma::fill::zeros);
  arma::cube W_latent(max_ncomp_eff, m, length_ncomp, arma::fill::zeros);
  arma::vec R2Y(length_ncomp, fill::zeros);
  arma::mat Ycentered;
  if (fit) {
    Ycentered = Yop.centered_copy();
  }

  for (int a = 0; a < length_ncomp; ++a) {
    const int mc_eff = std::min(static_cast<int>(ncomp(a)), max_ncomp_eff);
    arma::mat svd_u_mc = svd_u.cols(0, mc_eff - 1);
    arma::mat svd_v_mc = svd_v.cols(0, mc_eff - 1);
    arma::mat T_a = T_eff.cols(0, mc_eff - 1);
    arma::mat G_a = G_full.submat(0, 0, mc_eff - 1, mc_eff - 1);
    arma::mat D_a(mc_eff, mc_eff, fill::zeros);
    D_a.diag() = svd_s.subvec(0, mc_eff - 1);

    arma::mat coeff_latent;
    bool solved = arma::solve(coeff_latent, G_a, D_a, arma::solve_opts::likely_sympd);
    if (!solved) solved = arma::solve(coeff_latent, G_a, D_a);
    if (!solved) stop("plssvd latent solve failed");

    C_latent.slice(a).submat(0, 0, mc_eff - 1, mc_eff - 1) = coeff_latent;
    arma::mat W_a = coeff_latent * svd_v_mc.t();
    W_latent.slice(a).submat(0, 0, mc_eff - 1, m - 1) = W_a;
    if (store_B) {
      B.slice(a) = svd_u_mc * W_a;
    }
    if (fit) {
      arma::mat temp1 = T_a * W_a;
      R2Y(a) = RQ(Ycentered, temp1);
      temp1.each_row() += mY_row;
      Yfit.slice(a) = temp1;
    }
  }

  arma::mat mX(1, p); mX.row(0) = mX_row;
  arma::mat vX(1, p); vX.row(0) = vX_row;
  arma::mat mY(1, m); mY.row(0) = mY_row;
  List out = List::create(
    Named("C_latent") = C_latent,
    Named("W_latent") = W_latent,
    Named("Q")       = svd_v,
    Named("Ttrain")  = T_eff,
    Named("R")       = svd_u,
    Named("mX")      = mX,
    Named("vX")      = vX,
    Named("mY")      = mY,
    Named("p")       = p,
    Named("m")       = m,
    Named("ncomp")   = ncomp,
    Named("Yfit")    = Yfit,
    Named("R2Y")     = R2Y,
    Named("xprod_precision") = 3,
    Named("xprod_mode") = "implicit"
  );
  if (store_B) {
    out["B"] = B;
  }
  annotate_coefficient_storage(out, store_B);
  return out;
}

List pls_model2_fast_rsvd_xprod_precision_view_impl(
  SEXP XtrainSEXP,
  SEXP YtrainSEXP,
  arma::ivec ncomp,
  int scaling,
  bool fit,
  int rsvd_power,
  int seed
) {
  const arma::mat Xview = numeric_matrix_view(XtrainSEXP, "Xtrain");
  const arma::mat Yview = numeric_matrix_view(YtrainSEXP, "Ytrain");
  const int n = Xview.n_rows;
  const int p = Xview.n_cols;
  const int m = Yview.n_cols;
  if (Yview.n_rows != static_cast<arma::uword>(n)) {
    stop("Xtrain and Ytrain must have the same number of rows");
  }

  if (ncomp.n_elem < 1) {
    stop("ncomp must contain at least one value");
  }
  for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
    if (ncomp(i) < 1) ncomp(i) = 1;
  }

  const int max_ncomp = max(ncomp);
  const int length_ncomp = ncomp.n_elem;

  arma::rowvec mX_row(p, fill::zeros);
  if (scaling < 3) {
    mX_row = mean(Xview, 0);
  }
  arma::rowvec vX_row(p, fill::ones);
  if (scaling == 2) {
    vX_row = variance_nocopy(Xview);
  }
  arma::rowvec mY_row = mean(Yview, 0);

  CenterScaleMatrixView Xop{Xview, mX_row, vX_row};
  CenterOnlyMatrixView Yop{Yview, mY_row};

  arma::mat RR(p, max_ncomp, fill::zeros);
  arma::mat QQ(m, max_ncomp, fill::zeros);
  arma::mat VV(p, max_ncomp, fill::zeros);
  const bool store_B = should_store_coefficients(p, m, length_ncomp, true);
  arma::cube B;
  if (store_B) {
    B.zeros(p, m, length_ncomp);
  }

  arma::cube Yfit;
  arma::vec R2Y(length_ncomp, fill::zeros);
  arma::mat Yfit_cur;
  arma::mat Ycentered;
  if (fit) {
    Yfit.set_size(n, m, length_ncomp);
    Yfit_cur.zeros(n, m);
    Ycentered = Yop.centered_copy();
  }

  arma::mat Bcur;
  if (store_B) {
    Bcur.zeros(p, m);
  }
  int i_out = 0;

  const int refresh_block = env_int_or("FASTPLS_FAST_BLOCK", 1, 1, 16);
  const int center_t = env_int_or("FASTPLS_FAST_CENTER_T", 0, 0, 1);
  const int reorth_v = env_int_or("FASTPLS_FAST_REORTH_V", 0, 0, 1);
  const int inc_power_iters = env_int_or("FASTPLS_FAST_INC_ITERS", 2, 1, 6);
  AdaptiveRefreshPolicy adaptive_policy = AdaptiveRefreshPolicy::from_env(refresh_block, inc_power_iters);

  arma::vec rr_prev;
  bool has_rr_prev = false;
  auto append_component = [&](arma::vec rr, const int a_idx) -> bool {
    arma::vec tt = Xop.times(rr);
    if (center_t == 1) {
      tt -= arma::mean(tt);
    }
    const double tnorm = arma::norm(tt, 2);
    if (!std::isfinite(tnorm) || tnorm <= 0.0) return false;
    tt /= tnorm;
    rr /= tnorm;
    arma::vec pp = Xop.t_times(tt);
    arma::vec qq = Yop.t_times(tt);

    rr_prev = rr;
    has_rr_prev = true;

    arma::vec vv = pp;
    if (a_idx > 0) {
      auto Vprev = VV.cols(0, a_idx - 1);
      vv -= Vprev * (Vprev.t() * pp);
      if (reorth_v == 1) {
        vv -= Vprev * (Vprev.t() * vv);
      }
    }
    const double vnorm = arma::norm(vv, 2);
    if (!std::isfinite(vnorm) || vnorm <= 0.0) return false;
    vv /= vnorm;

    RR.col(a_idx) = rr;
    QQ.col(a_idx) = qq;
    VV.col(a_idx) = vv;
    if (store_B) {
      Bcur += rr * qq.t();
    }

    while (i_out < length_ncomp && a_idx == (ncomp(i_out) - 1)) {
      if (store_B) {
        B.slice(i_out) = Bcur;
      }
      if (fit) {
        Yfit_cur += tt * qq.t();
        R2Y(i_out) = RQ(Ycentered, Yfit_cur);
        arma::mat yf = Yfit_cur;
        yf.each_row() += mY_row;
        Yfit.slice(i_out) = yf;
      }
      ++i_out;
    }
    return true;
  };

  int a = 0;
  while (a < max_ncomp) {
    const int remaining = max_ncomp - a;
    const std::pair<int,int> refresh_cfg = adaptive_policy.current(remaining);
    const int k_block = std::min(refresh_cfg.first, remaining);
    const int power_iters_block = refresh_cfg.second;
    arma::mat Ublock;
    arma::vec shat_block;
    const arma::vec* warm_start = has_rr_prev ? &rr_prev : nullptr;
    if (!refresh_deflated_crossprod_left_double_view(
          Xop,
          Yop,
          VV,
          a,
          warm_start,
          k_block,
          power_iters_block,
          static_cast<unsigned int>(seed + a),
          Ublock,
          shat_block
        )) {
      break;
    }
    adaptive_policy.update_from_spectrum(shat_block, max_ncomp - (a + k_block));
    if (Ublock.n_cols < 1) break;

    const int use_cols = std::min(static_cast<int>(Ublock.n_cols), k_block);
    bool stop_now = false;
    for (int j = 0; j < use_cols && a < max_ncomp; ++j, ++a) {
      if (!append_component(Ublock.col(j), a)) {
        stop_now = true;
        break;
      }
    }
    if (stop_now) break;
  }

  arma::mat mX(1, p); mX.row(0) = mX_row;
  arma::mat vX(1, p); vX.row(0) = vX_row;
  arma::mat mY(1, m); mY.row(0) = mY_row;
  List out = List::create(
    Named("P")       = arma::mat(),
    Named("Q")       = QQ,
    Named("Ttrain")  = arma::mat(),
    Named("R")       = RR,
    Named("mX")      = mX,
    Named("vX")      = vX,
    Named("mY")      = mY,
    Named("p")       = p,
    Named("m")       = m,
    Named("ncomp")   = ncomp,
    Named("Yfit")    = Yfit,
    Named("R2Y")     = R2Y,
    Named("xprod_precision") = 3,
    Named("xprod_mode") = "implicit"
  );
  if (store_B) {
    out["B"] = B;
  }
  annotate_coefficient_storage(out, store_B);
  return out;
}

// [[Rcpp::export]]
List pls_model1_rsvd_xprod_precision(
  SEXP XtrainSEXP,
  SEXP YtrainSEXP,
  arma::ivec ncomp,
  int scaling,
  bool fit,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed,
  int xprod_precision
) {
  if (xprod_precision == 3) {
    return pls_model1_rsvd_xprod_precision_view_impl(
      XtrainSEXP,
      YtrainSEXP,
      ncomp,
      scaling,
      fit,
      rsvd_oversample,
      rsvd_power,
      seed
    );
  }
  if (xprod_precision == 5) {
    // IRLBA xprod keeps the bundled C IRLBA operator path, but is selected
    // only by the stricter R-side threshold to avoid poor shapes.
  }

  arma::mat Xtrain = Rcpp::as<arma::mat>(XtrainSEXP);
  arma::mat Ytrain = Rcpp::as<arma::mat>(YtrainSEXP);
  const int n = Xtrain.n_rows;
  const int p = Xtrain.n_cols;
  const int m = Ytrain.n_cols;
  const int max_plssvd_rank = std::min(n, std::min(p, m));
  const int length_ncomp = ncomp.n_elem;

  for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
    if (ncomp(i) > max_plssvd_rank) ncomp(i) = max_plssvd_rank;
    if (ncomp(i) < 1) ncomp(i) = 1;
  }

  const int max_ncomp = max(ncomp);
  int max_ncomp_eff = std::min(max_ncomp, max_plssvd_rank);
  if (max_ncomp_eff < 1) {
    stop("plssvd effective rank is < 1");
  }

  arma::mat mX(1, p, fill::zeros);
  if (scaling < 3) {
    mX = mean(Xtrain, 0);
    Xtrain.each_row() -= mX;
  }

  arma::mat vX(1, p, fill::ones);
  if (scaling == 2) {
    vX = variance(Xtrain);
    Xtrain.each_row() /= vX;
  }

  arma::mat mY = mean(Ytrain, 0);
  Ytrain.each_row() -= mY;

  if (xprod_precision == 1 || xprod_precision == 2 || xprod_precision == 4) {
    Rcpp::stop("xprod_precision values 1, 2, and 4 have been removed from fastPLS.");
  }

  fastpls_svd::SVDResult svd_res;
  if (xprod_precision == 3) {
    // Matrix-free 64-bit RSVD for A = X'Y: avoid materializing the huge
    // p-by-q crossproduct while preserving double-precision arithmetic.
    svd_res = truncated_rsvd_crossprod_double(
      Xtrain,
      Ytrain,
      max_ncomp_eff,
      rsvd_oversample,
      rsvd_power,
      static_cast<unsigned int>(seed),
      false,
      plssvd_use_small_exact_svd(max_plssvd_rank)
    );
  } else if (xprod_precision == 5) {
    // Matrix-free IRLBA for A = X'Y using the bundled C IRLBA operator API.
    svd_res = truncated_irlba_crossprod_double(
      Xtrain,
      Ytrain,
      max_ncomp_eff,
      false,
      plssvd_use_small_exact_svd(max_plssvd_rank)
    );
  } else {
    arma::mat S = Xtrain.t() * Ytrain;
    svd_res = compute_truncated_svd_dispatch(
      S,
      max_ncomp_eff,
      fastpls_svd::SVD_METHOD_CPU_RSVD,
      rsvd_oversample,
      rsvd_power,
      svds_tol,
      static_cast<unsigned int>(seed),
      false,
      plssvd_use_small_exact_svd(max_plssvd_rank)
    );
  }

  arma::mat svd_u = svd_res.U;
  arma::vec svd_s = svd_res.s;
  arma::mat svd_v = svd_res.Vt.t();

  const bool store_B = should_store_coefficients(p, m, length_ncomp, true);
  arma::cube B;
  if (store_B) {
    B.zeros(p, m, length_ncomp);
  }
  arma::cube Yfit;
  if (fit) {
    Yfit.set_size(n, m, length_ncomp);
  }

  max_ncomp_eff = std::min(max_ncomp_eff, static_cast<int>(svd_u.n_cols));
  if (svd_v.n_cols > 0) {
    max_ncomp_eff = std::min(max_ncomp_eff, static_cast<int>(svd_v.n_cols));
  }
  if (max_ncomp_eff < 1) {
    stop("plssvd effective rank is < 1 after SVD");
  }

  svd_u = svd_u.cols(0, max_ncomp_eff - 1);
  if (svd_v.n_cols > static_cast<arma::uword>(max_ncomp_eff)) {
    svd_v = svd_v.cols(0, max_ncomp_eff - 1);
  }

  arma::mat T_eff = Xtrain * svd_u;
  arma::mat G_full = T_eff.t() * T_eff;
  arma::cube C_latent(max_ncomp_eff, max_ncomp_eff, length_ncomp, arma::fill::zeros);
  arma::cube W_latent(max_ncomp_eff, m, length_ncomp, arma::fill::zeros);
  arma::vec R2Y(length_ncomp, fill::zeros);

  for (int a = 0; a < length_ncomp; ++a) {
    const int mc_eff = std::min(static_cast<int>(ncomp(a)), max_ncomp_eff);
    arma::mat svd_u_mc = svd_u.cols(0, mc_eff - 1);
    arma::mat svd_v_mc = svd_v.cols(0, mc_eff - 1);
    arma::mat T_a = T_eff.cols(0, mc_eff - 1);
    arma::mat G_a = G_full.submat(0, 0, mc_eff - 1, mc_eff - 1);
    arma::mat D_a(mc_eff, mc_eff, fill::zeros);
    D_a.diag() = svd_s.subvec(0, mc_eff - 1);

    arma::mat coeff_latent;
    bool solved = arma::solve(coeff_latent, G_a, D_a, arma::solve_opts::likely_sympd);
    if (!solved) solved = arma::solve(coeff_latent, G_a, D_a);
    if (!solved) stop("plssvd latent solve failed");

    C_latent.slice(a).submat(0, 0, mc_eff - 1, mc_eff - 1) = coeff_latent;
    arma::mat W_a = coeff_latent * svd_v_mc.t();
    W_latent.slice(a).submat(0, 0, mc_eff - 1, m - 1) = W_a;
    if (store_B) {
      B.slice(a) = svd_u_mc * W_a;
    }
    if (fit) {
      arma::mat temp1 = T_a * W_a;
      R2Y(a) = RQ(Ytrain, temp1);
      temp1.each_row() += mY;
      Yfit.slice(a) = temp1;
    }
  }

  List out = List::create(
    Named("C_latent") = C_latent,
    Named("W_latent") = W_latent,
    Named("Q")       = svd_v,
    Named("Ttrain")  = T_eff,
    Named("R")       = svd_u,
    Named("mX")      = mX,
    Named("vX")      = vX,
    Named("mY")      = mY,
    Named("p")       = p,
    Named("m")       = m,
    Named("ncomp")   = ncomp,
    Named("Yfit")    = Yfit,
    Named("R2Y")     = R2Y,
    Named("xprod_precision") = xprod_precision,
    Named("xprod_mode") = (xprod_precision == 5 ? "implicit_irlba" : (xprod_precision == 3 ? "implicit" : "materialized"))
  );
  if (store_B) {
    out["B"] = B;
  }
  annotate_coefficient_storage(out, store_B);
  return out;
}

// [[Rcpp::export]]
List pls_model2_fast_rsvd_xprod_precision(
  SEXP XtrainSEXP,
  SEXP YtrainSEXP,
  arma::ivec ncomp,
  int scaling,
  bool fit,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed,
  int xprod_precision
) {
  if (xprod_precision == 3) {
    return pls_model2_fast_rsvd_xprod_precision_view_impl(
      XtrainSEXP,
      YtrainSEXP,
      ncomp,
      scaling,
      fit,
      rsvd_power,
      seed
    );
  }

  arma::mat Xtrain = Rcpp::as<arma::mat>(XtrainSEXP);
  arma::mat Ytrain = Rcpp::as<arma::mat>(YtrainSEXP);
  const int n = Xtrain.n_rows;
  const int p = Xtrain.n_cols;
  const int m = Ytrain.n_cols;

  if (ncomp.n_elem < 1) {
    stop("ncomp must contain at least one value");
  }
  for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
    if (ncomp(i) < 1) ncomp(i) = 1;
  }

  const int max_ncomp = max(ncomp);
  const int length_ncomp = ncomp.n_elem;

  arma::mat mX(1, p, fill::zeros);
  if (scaling < 3) {
    mX = mean(Xtrain, 0);
    Xtrain.each_row() -= mX;
  }

  arma::mat vX(1, p, fill::ones);
  if (scaling == 2) {
    vX = variance(Xtrain);
    Xtrain.each_row() /= vX;
  }

  arma::mat mY = mean(Ytrain, 0);
  Ytrain.each_row() -= mY;

  if (xprod_precision == 1 || xprod_precision == 2 || xprod_precision == 4) {
    Rcpp::stop("xprod_precision values 1, 2, and 4 have been removed from fastPLS.");
  }

  const bool use_implicit_double_xprod = (xprod_precision == 3);
  const bool use_implicit_irlba_xprod = (xprod_precision == 5);
  const bool use_implicit_xprod = use_implicit_double_xprod || use_implicit_irlba_xprod;

  arma::mat Xt;
  arma::mat Yt;
  if (!use_implicit_xprod) {
    Xt = Xtrain.t();
    Yt = Ytrain.t();
  }
  arma::mat S;
  if (!use_implicit_xprod) {
    S = Xt * Ytrain;
  }

  arma::mat XtX_cache;
  arma::mat Sxy_cache;
  arma::mat RR(p, max_ncomp, fill::zeros);
  arma::mat QQ(m, max_ncomp, fill::zeros);
  arma::mat VV(p, max_ncomp, fill::zeros);
  const bool store_B = should_store_coefficients(p, m, length_ncomp, true);
  arma::cube B;
  if (store_B) {
    B.zeros(p, m, length_ncomp);
  }

  arma::cube Yfit;
  arma::vec R2Y(length_ncomp, fill::zeros);
  arma::mat Yfit_cur;
  if (fit) {
    Yfit.set_size(n, m, length_ncomp);
    Yfit_cur.zeros(n, m);
  }

  arma::mat Bcur;
  if (store_B) {
    Bcur.zeros(p, m);
  }
  int i_out = 0;

  const int refresh_block = env_int_or("FASTPLS_FAST_BLOCK", 1, 1, 16);
  const int center_t = env_int_or("FASTPLS_FAST_CENTER_T", 0, 0, 1);
  const int reorth_v = env_int_or("FASTPLS_FAST_REORTH_V", 0, 0, 1);
  const int inc_power_iters = env_int_or("FASTPLS_FAST_INC_ITERS", 2, 1, 6);
  const int defl_cache = env_int_or("FASTPLS_FAST_DEFLCACHE", 1, 0, 1);
  const int fast_optimized = env_int_or("FASTPLS_FAST_OPTIMIZED", 1, 0, 1);
  const int fast_crossprod_min_ncomp = env_int_or("FASTPLS_FAST_CROSSPROD_MIN_NCOMP", 20, 1, 1024);
  const int fast_crossprod_max_p = env_int_or("FASTPLS_FAST_CROSSPROD_MAX_P", 512, 16, 65536);
  const int fast_crossprod_min_n_to_p_ratio = env_int_or("FASTPLS_FAST_CROSSPROD_MIN_N_TO_P_RATIO", 8, 1, 1024);
  const bool use_crossprod_cache =
    (!use_implicit_xprod) &&
    (fast_optimized == 1) &&
    (center_t == 0) &&
    (max_ncomp >= fast_crossprod_min_ncomp) &&
    (p <= n) &&
    (n >= p * fast_crossprod_min_n_to_p_ratio) &&
    (p <= fast_crossprod_max_p);

  if (use_crossprod_cache) {
    XtX_cache = Xt * Xtrain;
    Sxy_cache = S;
  }

  arma::vec rr_prev;
  bool has_rr_prev = false;
  auto append_component = [&](arma::vec rr, const int a_idx) -> bool {
    arma::vec pp;
    arma::vec qq;
    arma::vec tt;

    if (use_crossprod_cache) {
      pp = XtX_cache * rr;
      const double tnorm_sq = arma::dot(rr, pp);
      if (!std::isfinite(tnorm_sq) || tnorm_sq <= 0.0) return false;
      const double tnorm = std::sqrt(tnorm_sq);
      rr /= tnorm;
      pp /= tnorm;
      qq = Sxy_cache.t() * rr;
      if (fit) tt = Xtrain * rr;
    } else if (use_implicit_xprod) {
      tt = Xtrain * rr;
      if (center_t == 1) {
        tt -= arma::mean(tt);
      }
      const double tnorm = arma::norm(tt, 2);
      if (!std::isfinite(tnorm) || tnorm <= 0.0) return false;
      tt /= tnorm;
      rr /= tnorm;
      pp = Xtrain.t() * tt;
      qq = Ytrain.t() * tt;
    } else {
      tt = Xtrain * rr;
      if (center_t == 1) {
        tt -= arma::mean(tt);
      }
      const double tnorm = arma::norm(tt, 2);
      if (!std::isfinite(tnorm) || tnorm <= 0.0) return false;
      tt /= tnorm;
      rr /= tnorm;
      pp = Xt * tt;
      qq = Yt * tt;
    }

    rr_prev = rr;
    has_rr_prev = true;

    arma::vec vv = pp;
    if (a_idx > 0) {
      auto Vprev = VV.cols(0, a_idx - 1);
      vv -= Vprev * (Vprev.t() * pp);
      if (reorth_v == 1) {
        vv -= Vprev * (Vprev.t() * vv);
      }
    }
    const double vnorm = arma::norm(vv, 2);
    if (!std::isfinite(vnorm) || vnorm <= 0.0) return false;
    vv /= vnorm;

    if (use_implicit_xprod) {
      // No persistent S exists in the implicit paths. Future refreshes apply
      // the VV projector directly to X'Y.
    } else if (defl_cache == 1) {
      arma::rowvec vS = vv.t() * S;
      S -= vv * vS;
    } else {
      S -= vv * (vv.t() * S);
    }

    RR.col(a_idx) = rr;
    QQ.col(a_idx) = qq;
    VV.col(a_idx) = vv;
    if (store_B) {
      Bcur += rr * qq.t();
    }

    while (i_out < length_ncomp && a_idx == (ncomp(i_out) - 1)) {
      if (store_B) {
        B.slice(i_out) = Bcur;
      }
      if (fit) {
        Yfit_cur += tt * qq.t();
        R2Y(i_out) = RQ(Ytrain, Yfit_cur);
        arma::mat yf = Yfit_cur;
        yf.each_row() += mY;
        Yfit.slice(i_out) = yf;
      }
      ++i_out;
    }
    return true;
  };

  SimplsFastRefreshWorkspace refresh_ws;
  AdaptiveRefreshPolicy adaptive_policy = AdaptiveRefreshPolicy::from_env(refresh_block, inc_power_iters);
  int a = 0;
  while (a < max_ncomp) {
    const int remaining = max_ncomp - a;
    const std::pair<int,int> refresh_cfg = adaptive_policy.current(remaining);
    const int k_block = std::min(refresh_cfg.first, remaining);
    const int power_iters_block = refresh_cfg.second;
    arma::mat Ublock;
    const arma::vec* warm_start = has_rr_prev ? &rr_prev : nullptr;
    if (use_implicit_irlba_xprod) {
      if (!refresh_deflated_crossprod_left_irlba_double(
            Xtrain,
            Ytrain,
            VV,
            a,
            k_block,
            Ublock,
            refresh_ws.shat
          )) {
        break;
      }
      adaptive_policy.update_from_spectrum(refresh_ws.shat, max_ncomp - (a + k_block));
    } else if (use_implicit_double_xprod) {
      arma::vec shat_block;
      if (!refresh_deflated_crossprod_left_double(
            Xtrain,
            Ytrain,
            VV,
            a,
            warm_start,
            k_block,
            power_iters_block,
            static_cast<unsigned int>(seed + a),
            Ublock,
            shat_block
          )) {
        break;
      }
      adaptive_policy.update_from_spectrum(shat_block, max_ncomp - (a + k_block));
    } else {
      if (!refresh_ws.refresh(
            S,
            warm_start,
            k_block,
            power_iters_block,
            static_cast<unsigned int>(seed + a),
            Ublock
          )) {
        break;
      }
      adaptive_policy.update_from_spectrum(refresh_ws.shat, max_ncomp - (a + k_block));
    }
    if (Ublock.n_cols < 1) break;

    const int use_cols = std::min(static_cast<int>(Ublock.n_cols), k_block);
    bool stop_now = false;
    for (int j = 0; j < use_cols && a < max_ncomp; ++j, ++a) {
      if (!append_component(Ublock.col(j), a)) {
        stop_now = true;
        break;
      }
    }
    if (stop_now) break;
  }

  List out = List::create(
    Named("P")       = arma::mat(),
    Named("Q")       = QQ,
    Named("Ttrain")  = arma::mat(),
    Named("R")       = RR,
    Named("mX")      = mX,
    Named("vX")      = vX,
    Named("mY")      = mY,
    Named("p")       = p,
    Named("m")       = m,
    Named("ncomp")   = ncomp,
    Named("Yfit")    = Yfit,
    Named("R2Y")     = R2Y,
    Named("xprod_precision") = xprod_precision,
    Named("xprod_mode") = use_implicit_irlba_xprod ? "implicit_irlba" : (use_implicit_double_xprod ? "implicit" : "materialized")
  );
  if (store_B) {
    out["B"] = B;
  }
  annotate_coefficient_storage(out, store_B);
  return out;
}

// [[Rcpp::export]]
List pls_model1_gpu(
  arma::mat Xtrain,
  arma::mat Ytrain,
  arma::ivec ncomp,
  int scaling,
  bool fit,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed
) {
  if (!fastpls_svd::has_cuda_backend()) {
    stop("pls_model1_gpu requires CUDA support");
  }

  const int n = Xtrain.n_rows;
  const int p = Xtrain.n_cols;
  const int m = Ytrain.n_cols;
  const int max_plssvd_rank = std::min(n, std::min(p, m));
  for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
    if (ncomp(i) > max_plssvd_rank) {
      ncomp(i) = max_plssvd_rank;
    }
    if (ncomp(i) < 1) {
      ncomp(i) = 1;
    }
  }
  const int max_ncomp = max(ncomp);
  const int max_ncomp_eff = std::min(max_ncomp, max_plssvd_rank);
  if (max_ncomp_eff < 1) {
    stop("plssvd effective rank is < 1");
  }

  arma::mat mX(1, p, fill::zeros);
  if (scaling < 3) {
    mX = mean(Xtrain, 0);
    Xtrain.each_row() -= mX;
  }

  arma::mat vX(1, p, fill::ones);
  if (scaling == 2) {
    vX = variance(Xtrain);
    Xtrain.each_row() /= vX;
  }

  arma::mat mY = mean(Ytrain, 0);
  Ytrain.each_row() -= mY;

  fastpls_svd::SVDOptions opt;
  opt.method = fastpls_svd::Method::RSVD;
  opt.oversample = std::max(rsvd_oversample, 0);
  opt.power_iters = std::max(rsvd_power, 0);
  opt.svds_tol = std::max(svds_tol, 0.0);
  opt.seed = static_cast<unsigned int>(seed);
  opt.left_only = false;
  opt.use_full_svd = false;

  fastpls_svd::PLSSVDGPUResult gpu = fastpls_svd::cuda_plssvd_fit(
    Xtrain,
    Ytrain,
    ncomp,
    fit,
    opt
  );

  arma::cube Yfit = gpu.Yfit;
  if (fit && Yfit.n_elem > 0) {
    for (arma::uword i = 0; i < Yfit.n_slices; ++i) {
      Yfit.slice(i).each_row() += mY;
    }
  }

  const bool store_B = should_store_coefficients(p, m, ncomp.n_elem, true);
  List out = List::create(
    Named("C_latent") = gpu.C_latent,
    Named("W_latent") = gpu.W_latent,
    Named("Q")       = gpu.Q,
    Named("Ttrain")  = gpu.Ttrain,
    Named("R")       = gpu.R,
    Named("mX")      = mX,
    Named("vX")      = vX,
    Named("mY")      = mY,
    Named("p")       = p,
    Named("m")       = m,
    Named("ncomp")   = ncomp,
    Named("Yfit")    = Yfit,
    Named("R2Y")     = gpu.R2Y
  );
  if (store_B) {
    out["B"] = gpu.B;
  }
  annotate_coefficient_storage(out, store_B);
  return out;
}

// [[Rcpp::export]]
List pls_model1_gpu_implicit_xprod(
  arma::mat Xtrain,
  arma::mat Ytrain,
  arma::ivec ncomp,
  int scaling,
  bool fit,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed
) {
  if (!fastpls_svd::has_cuda_backend()) {
    stop("pls_model1_gpu_implicit_xprod requires CUDA support");
  }

  const int n = Xtrain.n_rows;
  const int p = Xtrain.n_cols;
  const int m = Ytrain.n_cols;
  const int max_plssvd_rank = std::min(n, std::min(p, m));
  for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
    if (ncomp(i) > max_plssvd_rank) {
      ncomp(i) = max_plssvd_rank;
    }
    if (ncomp(i) < 1) {
      ncomp(i) = 1;
    }
  }
  const int max_ncomp = max(ncomp);
  const int max_ncomp_eff = std::min(max_ncomp, max_plssvd_rank);
  if (max_ncomp_eff < 1) {
    stop("plssvd effective rank is < 1");
  }

  arma::mat mX(1, p, fill::zeros);
  if (scaling < 3) {
    mX = mean(Xtrain, 0);
    Xtrain.each_row() -= mX;
  }

  arma::mat vX(1, p, fill::ones);
  if (scaling == 2) {
    vX = variance(Xtrain);
    Xtrain.each_row() /= vX;
  }

  arma::mat mY = mean(Ytrain, 0);
  Ytrain.each_row() -= mY;

  fastpls_svd::SVDOptions opt;
  opt.method = fastpls_svd::Method::RSVD;
  opt.oversample = std::max(rsvd_oversample, 0);
  opt.power_iters = std::max(rsvd_power, 0);
  opt.svds_tol = std::max(svds_tol, 0.0);
  opt.seed = static_cast<unsigned int>(seed);
  opt.left_only = false;
  opt.use_full_svd = false;

  fastpls_svd::PLSSVDGPUResult gpu = fastpls_svd::cuda_plssvd_fit_implicit_xprod(
    Xtrain,
    Ytrain,
    ncomp,
    fit,
    opt
  );

  arma::cube Yfit = gpu.Yfit;
  if (fit && Yfit.n_elem > 0) {
    for (arma::uword i = 0; i < Yfit.n_slices; ++i) {
      Yfit.slice(i).each_row() += mY;
    }
  }

  const bool store_B = should_store_coefficients(p, m, ncomp.n_elem, true);
  List out = List::create(
    Named("C_latent") = gpu.C_latent,
    Named("W_latent") = gpu.W_latent,
    Named("Q")       = gpu.Q,
    Named("Ttrain")  = gpu.Ttrain,
    Named("R")       = gpu.R,
    Named("mX")      = mX,
    Named("vX")      = vX,
    Named("mY")      = mY,
    Named("p")       = p,
    Named("m")       = m,
    Named("ncomp")   = ncomp,
    Named("Yfit")    = Yfit,
    Named("R2Y")     = gpu.R2Y,
    Named("xprod_mode") = "implicit"
  );
  if (store_B) {
    out["B"] = gpu.B;
  }
  annotate_coefficient_storage(out, store_B);
  return out;
}

// [[Rcpp::export]]
List pls_cv_predict_compiled(
  arma::mat Xdata,
  arma::mat Ydata,
  arma::ivec constrain,
  arma::ivec ncomp,
  int scaling,
  int kfold,
  int method,
  int backend,
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed,
  bool classification,
  bool xprod
) {
  const int nsamples = Xdata.n_rows;
  const int ncolY = Ydata.n_cols;
  if (nsamples < 2) stop("Xdata must contain at least two samples");
  if (Ydata.n_rows != static_cast<arma::uword>(nsamples)) {
    stop("Ydata must have the same number of rows as Xdata");
  }
  if (constrain.n_elem != static_cast<arma::uword>(nsamples)) {
    stop("constrain must have one value for each sample");
  }
  if (ncomp.n_elem < 1) stop("ncomp must contain at least one value");
  if (kfold < 2) kfold = 2;
  if (method < 1 || method > 3) stop("method must be 1=plssvd, 2=simpls, or 3=simpls_fast");
  if (backend < 0 || backend > 1) stop("backend must be 0=cpp or 1=cuda");
  if (backend == 1 && method == 2) {
    stop("CUDA classic SIMPLS is not implemented; use simpls_fast CUDA instead");
  }
  if (backend == 1 && !fastpls_svd::has_cuda_backend()) {
    stop("CUDA CV requires a CUDA-enabled fastPLS build");
  }
  if (method == 1) {
    const int max_plssvd_rank = std::min(
      nsamples,
      std::min(static_cast<int>(Xdata.n_cols), ncolY)
    );
    for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
      if (ncomp(i) > max_plssvd_rank) ncomp(i) = max_plssvd_rank;
      if (ncomp(i) < 1) ncomp(i) = 1;
    }
  } else {
    for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
      if (ncomp(i) < 1) ncomp(i) = 1;
    }
  }

  arma::ivec unique_groups = arma::unique(constrain);
  arma::ivec constrain2 = constrain;
  for (arma::uword j = 0; j < unique_groups.n_elem; ++j) {
    arma::uvec ind = arma::find(constrain == unique_groups(j));
    constrain2.elem(ind).fill(static_cast<int>(j) + 1);
  }

  const int ngroups = unique_groups.n_elem;
  IntegerVector frame = seq_len(ngroups);
  IntegerVector perm = samplewithoutreplace(frame, ngroups);
  arma::ivec fold(nsamples);
  for (int i = 0; i < nsamples; ++i) {
    const int group_idx = constrain2(i) - 1;
    fold(i) = (perm[group_idx] - 1) % kfold;
  }

  const int length_ncomp = ncomp.n_elem;
  arma::cube Ypred(nsamples, ncolY, length_ncomp, arma::fill::zeros);
  arma::ivec status(kfold, arma::fill::zeros);

  const std::string method_name =
    (method == 1) ? "plssvd" : ((method == 2) ? "simpls" : "simpls_fast");

  for (int f = 0; f < kfold; ++f) {
    Rcpp::checkUserInterrupt();
    arma::uvec test_idx = arma::find(fold == f);
    arma::uvec train_idx = arma::find(fold != f);
    if (test_idx.n_elem == 0) {
      status(f) = 2; // empty fold
      continue;
    }
    if (train_idx.n_elem == 0) {
      for (int s = 0; s < length_ncomp; ++s) {
        Ypred.slice(s).rows(test_idx) = Ydata.rows(test_idx);
      }
      status(f) = 3; // no training data
      continue;
    }

    arma::mat Xtrain = Xdata.rows(train_idx);
    arma::mat Xtest = Xdata.rows(test_idx);
    arma::mat Ytrain = Ydata.rows(train_idx);

    if (classification) {
      arma::rowvec class_counts = arma::sum(Ytrain, 0);
      arma::uvec active = arma::find(class_counts > 0.5);
      if (active.n_elem <= 1) {
        arma::rowvec fallback(ncolY, arma::fill::zeros);
        if (active.n_elem == 1) {
          fallback(active(0)) = 1.0;
        } else {
          fallback = arma::mean(Ydata, 0);
        }
        for (int s = 0; s < length_ncomp; ++s) {
          for (arma::uword ii = 0; ii < test_idx.n_elem; ++ii) {
            Ypred.slice(s).row(test_idx(ii)) = fallback;
          }
        }
        status(f) = 4; // degenerate classification fold
        continue;
      }
    }

    List model;
    if (backend == 1) {
      if (method == 1) {
        if (xprod) {
          model = pls_model1_gpu_implicit_xprod(
            Xtrain, Ytrain, ncomp, scaling, false,
            rsvd_oversample, rsvd_power, svds_tol, seed + f
          );
        } else {
          model = pls_model1_gpu(
            Xtrain, Ytrain, ncomp, scaling, false,
            rsvd_oversample, rsvd_power, svds_tol, seed + f
          );
        }
      } else {
        model = pls_model2_fast_gpu(
          Xtrain, Ytrain, ncomp, scaling, false,
          fastpls_svd::SVD_METHOD_CUDA_RSVD,
          rsvd_oversample, rsvd_power, svds_tol, seed + f
        );
      }
    } else {
      if (method == 1) {
        if (xprod) {
          const int xprod_precision = (svd_method == fastpls_svd::SVD_METHOD_IRLBA) ? 5 : 3;
          Rcpp::NumericMatrix Xtrain_r = Rcpp::wrap(Xtrain);
          Rcpp::NumericMatrix Ytrain_r = Rcpp::wrap(Ytrain);
          model = pls_model1_rsvd_xprod_precision(
            Xtrain_r, Ytrain_r, ncomp, scaling, false,
            rsvd_oversample, rsvd_power, svds_tol, seed + f, xprod_precision
          );
        } else {
          model = pls_model1(
            Xtrain, Ytrain, ncomp, scaling, false, svd_method,
            rsvd_oversample, rsvd_power, svds_tol, seed + f
          );
        }
      } else if (method == 2) {
        model = pls_model2(
          Xtrain, Ytrain, ncomp, scaling, false, svd_method,
          rsvd_oversample, rsvd_power, svds_tol, seed + f
        );
      } else {
        if (xprod) {
          const int xprod_precision = (svd_method == fastpls_svd::SVD_METHOD_IRLBA) ? 5 : 3;
          Rcpp::NumericMatrix Xtrain_r = Rcpp::wrap(Xtrain);
          Rcpp::NumericMatrix Ytrain_r = Rcpp::wrap(Ytrain);
          model = pls_model2_fast_rsvd_xprod_precision(
            Xtrain_r, Ytrain_r, ncomp, scaling, false,
            rsvd_oversample, rsvd_power, svds_tol, seed + f, xprod_precision
          );
        } else {
          model = pls_model2_fast(
            Xtrain, Ytrain, ncomp, scaling, false, svd_method,
            rsvd_oversample, rsvd_power, svds_tol, seed + f
          );
        }
      }
    }

    model["pls_method"] = method_name;
    model["predict_latent_ok"] = true;

    List pred = pls_predict(model, Xtest, false);
    arma::cube fold_pred = pred["Ypred"];
    const int ncopy = std::min(length_ncomp, static_cast<int>(fold_pred.n_slices));
    for (int s = 0; s < ncopy; ++s) {
      Ypred.slice(s).rows(test_idx) = fold_pred.slice(s);
    }
    status(f) = 1; // ok
  }

  return List::create(
    Named("Ypred") = Ypred,
    Named("fold") = fold + 1,
    Named("status") = status,
    Named("ncomp") = ncomp,
    Named("method") = method_name,
    Named("backend") = (backend == 1 ? "cuda" : "cpp"),
    Named("xprod") = xprod
  );
}
