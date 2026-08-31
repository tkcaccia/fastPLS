

#include <RcppArmadillo.h>
#include <R_ext/Rdynload.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "fastPLS.h"
#include "svd_iface.h"
#include "svd_cuda_rsvd.h"
#include "svd_metal_backend.h"

extern "C" {
#include "irlba.h"
}

// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

namespace {

constexpr int kAcceleratedSimplsBlockSize = 8;

int accelerated_simpls_block_size(
  const int remaining,
  const int p,
  const int m,
  const bool classification_response = false
) {
  // Batched candidate refresh amortizes decomposition and device-launch costs.
  // Scalar and extreme-response problems retain rank-one refresh because their
  // predictive path was more stable in the validation panel.
  if (!classification_response || m <= 1 || m > 2048 || remaining < 4) {
    return 1;
  }
  return std::max(
    1,
    std::min({kAcceleratedSimplsBlockSize, remaining, p, m})
  );
}

bool is_one_hot_response(const arma::mat& Y) {
  if (Y.n_rows == 0 || Y.n_cols <= 1) return false;
  constexpr double tolerance = 1e-12;
  for (arma::uword row = 0; row < Y.n_rows; ++row) {
    int active = 0;
    for (arma::uword col = 0; col < Y.n_cols; ++col) {
      const double value = Y(row, col);
      if (std::abs(value - 1.0) <= tolerance) {
        ++active;
      } else if (std::abs(value) > tolerance) {
        return false;
      }
    }
    if (active != 1) return false;
  }
  return true;
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

arma::vec leading_left_vec_dispatch(
  const arma::mat& S,
  const int svd_method,
  const int rsvd_oversample,
  const int rsvd_power,
  const double svds_tol,
  const unsigned int seed
) {
  if (S.n_rows < 1 || S.n_cols < 1) {
    return arma::vec();
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

bool plssvd_use_small_exact_svd(const int max_rank, const int svd_method) {
  if (svd_method == fastpls_svd::SVD_METHOD_IRLBA) {
    return max_rank < 6;
  }
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

List label_crossprod_scaled_cpp_impl(
  SEXP XtrainSEXP,
  Rcpp::IntegerVector y,
  int n_classes,
  int scaling
) {
  const arma::mat X = numeric_matrix_view(XtrainSEXP, "Xtrain");
  const arma::uword n = X.n_rows;
  const arma::uword p = X.n_cols;
  if (y.size() != static_cast<int>(n)) {
    stop("label_crossprod_scaled_cpp requires one label per training row");
  }
  if (n_classes < 2) {
    stop("label_crossprod_scaled_cpp requires at least two classes");
  }
  arma::rowvec mX(p, arma::fill::zeros);
  arma::rowvec vX(p, arma::fill::ones);
  arma::rowvec sums(p, arma::fill::zeros);
  arma::rowvec sums_sq(p, arma::fill::zeros);
  arma::vec counts(static_cast<arma::uword>(n_classes), arma::fill::zeros);

  for (arma::uword j = 0; j < p; ++j) {
    double s = 0.0;
    double ss = 0.0;
    const double* col = X.colptr(j);
    for (arma::uword i = 0; i < n; ++i) {
      const double val = col[i];
      s += val;
      ss += val * val;
    }
    sums(j) = s;
    sums_sq(j) = ss;
  }
  if (scaling < 3) {
    mX = sums / static_cast<double>(n);
  }
  if (scaling == 2) {
    for (arma::uword j = 0; j < p; ++j) {
      const double centered_ss = std::max(0.0, sums_sq(j) - static_cast<double>(n) * mX(j) * mX(j));
      double sd = std::sqrt(centered_ss / std::max(1.0, static_cast<double>(n) - 1.0));
      if (!std::isfinite(sd) || sd <= 0.0) sd = 1.0;
      vX(j) = sd;
    }
  }

  arma::mat class_sums(p, static_cast<arma::uword>(n_classes), arma::fill::zeros);
  for (int cls : y) {
    if (IntegerVector::is_na(cls) || cls < 1 || cls > n_classes) {
      stop("label_crossprod_scaled_cpp requires labels encoded as 1..n_classes");
    }
    counts(static_cast<arma::uword>(cls - 1)) += 1.0;
  }
  for (arma::uword j = 0; j < p; ++j) {
    const double* col = X.colptr(j);
    const double center = (scaling < 3) ? mX(j) : 0.0;
    const double scale = (scaling == 2) ? vX(j) : 1.0;
    for (arma::uword i = 0; i < n; ++i) {
      const int cls = y[static_cast<int>(i)] - 1;
      class_sums(j, static_cast<arma::uword>(cls)) += (col[i] - center) / scale;
    }
  }

  arma::rowvec mY = counts.t() / static_cast<double>(n);
  arma::vec total_sums = arma::sum(class_sums, 1);
  arma::mat S = class_sums - total_sums * mY;
  return List::create(
    Named("S") = S,
    Named("mX") = mX,
    Named("vX") = vX,
    Named("mY") = mY,
    Named("counts") = counts
  );
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

template <typename ATimes, typename ATTimes>
fastpls_svd::SVDResult raw_rsvd_operator_double(
  const arma::uword p,
  const arma::uword m,
  const int k,
  const int oversample,
  const int power,
  const unsigned int seed,
  const bool left_only,
  const ATimes& a_times,
  const ATTimes& at_times
) {
  const arma::uword max_rank = std::min(p, m);
  const arma::uword target = std::min<arma::uword>(
    max_rank, static_cast<arma::uword>(std::max(k, 1))
  );
  const arma::uword l = std::min<arma::uword>(
    max_rank, target + static_cast<arma::uword>(std::max(oversample, 0))
  );
  arma::mat sample = a_times(gaussian_matrix_local(m, l, seed));
  for (int iteration = 0; iteration < std::max(power, 0); ++iteration) {
    arma::mat Qy;
    arma::mat Ry;
    arma::qr_econ(Qy, Ry, sample);
    arma::mat Z = at_times(Qy);
    arma::mat Qz;
    arma::mat Rz;
    arma::qr_econ(Qz, Rz, Z);
    sample = a_times(Qz);
  }
  arma::mat Q;
  arma::mat R;
  arma::qr_econ(Q, R, sample);
  if (Q.n_cols < 1) return fastpls_svd::SVDResult();
  return finalize_rsvd_from_q_b_double(
    Q, at_times(Q).t(), static_cast<int>(target), left_only
  );
}

template <typename ATimes, typename ATTimes>
bool audit_rsvd_operator_double(
  const fastpls_svd::SVDResult& result,
  const arma::uword retained,
  const unsigned int seed,
  const ATimes& a_times,
  const ATTimes& at_times,
  double& triplet_residual,
  double& omitted_ratio
) {
  (void)seed;
  if (result.U.n_cols < retained || result.Vt.n_rows < retained ||
      result.s.n_elem < retained || !result.U.is_finite() ||
      !result.Vt.is_finite() || !result.s.is_finite()) return false;
  const double spectral_scale = std::max(
    result.s(0), std::numeric_limits<double>::epsilon()
  );
  const arma::uword audit_rank = std::min(
    result.s.n_elem,
    std::min(result.U.n_cols, result.Vt.n_rows)
  );
  const arma::mat audit_u = result.U.cols(0, audit_rank - 1);
  const arma::mat audit_v = result.Vt.rows(0, audit_rank - 1).t();
  const arma::mat scaled_u = audit_u.each_row() %
    result.s.head(audit_rank).t();
  const arma::mat scaled_v = audit_v.each_row() %
    result.s.head(audit_rank).t();
  const arma::mat left_residuals = a_times(audit_v) - scaled_u;
  const arma::mat right_residuals = at_times(audit_u) - scaled_v;
  triplet_residual = 0.0;
  for (arma::uword j = 0; j < audit_rank; ++j) {
    triplet_residual = std::max(triplet_residual, std::max(
      arma::norm(left_residuals.col(j), 2),
      arma::norm(right_residuals.col(j), 2)
    ) / spectral_scale);
  }

  const double boundary = std::max(
    result.s(retained - 1), std::numeric_limits<double>::epsilon()
  );
  omitted_ratio = result.s.n_elem > retained ?
    result.s(retained) / boundary : 0.0;
  // Near-tied boundary values are valid singular directions. The triplet
  // residual audits accuracy; this ratio only rejects a materially stronger
  // omitted direction.
  return triplet_residual <= 1e-2 && omitted_ratio <= 1.01;
}

template <typename ATimes, typename ATTimes>
bool rsvd_operator_consensus_double(
  const fastpls_svd::SVDResult& lhs,
  const fastpls_svd::SVDResult& rhs,
  const arma::uword retained,
  const double rhs_residual,
  double& subspace_error,
  double& singular_value_error
) {
  if (lhs.U.n_cols < retained || rhs.U.n_cols < retained ||
      lhs.s.n_elem < retained || rhs.s.n_elem < retained) return false;
  const arma::mat cross = lhs.U.cols(0, retained - 1).t() *
    rhs.U.cols(0, retained - 1);
  const double overlap_sq = std::min<double>(
    static_cast<double>(retained), arma::accu(arma::square(cross))
  );
  subspace_error = std::sqrt(
    std::max(0.0, static_cast<double>(retained) - overlap_sq) /
    static_cast<double>(retained)
  );
  const double scale = std::max(
    rhs.s(0), std::numeric_limits<double>::epsilon()
  );
  singular_value_error = arma::abs(
    lhs.s.head(retained) - rhs.s.head(retained)
  ).max() / scale;
  return subspace_error <= 1e-3 && singular_value_error <= 1e-5 &&
    rhs_residual <= 1e-6;
}

template <typename ATimes, typename ATTimes>
fastpls_svd::SVDResult audited_rsvd_operator_double(
  const arma::uword p,
  const arma::uword m,
  const int k,
  const int requested_oversample,
  const int requested_power,
  const unsigned int requested_seed,
  const bool left_only,
  const ATimes& a_times,
  const ATTimes& at_times
) {
  const arma::uword retained = std::min<arma::uword>(
    std::min(p, m), static_cast<arma::uword>(std::max(k, 1))
  );
  const int audit_rank = static_cast<int>(std::min(std::min(p, m), retained + 1));
  const int oversamples[] = {
    requested_oversample,
    std::max(requested_oversample, 32),
    std::max(requested_oversample, 48)
  };
  const int powers[] = {
    requested_power,
    std::max(requested_power, 3),
    std::max(requested_power, 4)
  };
  const unsigned int seeds[] = {
    requested_seed,
    requested_seed + 104729U,
    requested_seed + 209759U
  };
  fastpls_svd::SVDResult previous;
  bool have_previous = false;
  for (int attempt = 0; attempt < 3; ++attempt) {
    fastpls_svd::SVDResult result = raw_rsvd_operator_double(
      p, m, audit_rank, oversamples[attempt], powers[attempt], seeds[attempt],
      false, a_times, at_times
    );
    double residual = std::numeric_limits<double>::infinity();
    double omitted = std::numeric_limits<double>::infinity();
    const bool residual_pass = audit_rsvd_operator_double(
      result, retained, seeds[attempt], a_times, at_times, residual, omitted
    );
    double subspace_error = std::numeric_limits<double>::infinity();
    double singular_value_error = std::numeric_limits<double>::infinity();
    const bool consensus_pass = have_previous && rsvd_operator_consensus_double<ATimes, ATTimes>(
      previous, result, retained, residual, subspace_error, singular_value_error
    );
    if (!residual_pass && !consensus_pass) {
      previous = result;
      have_previous = true;
      continue;
    }

    result.randomized = true;
    result.case_audited = true;
    result.case_certified = true;
    result.audit_attempts = attempt + 1;
    result.effective_oversample = oversamples[attempt];
    result.effective_power_iters = powers[attempt];
    result.effective_seed = seeds[attempt];
    result.audit_triplet_residual = residual;
    result.audit_omitted_direction_ratio = omitted;
    result.audit_subspace_error = consensus_pass ? subspace_error : 0.0;
    result.audit_singular_value_error = consensus_pass ? singular_value_error : 0.0;
    result.U = result.U.cols(0, retained - 1);
    result.s = result.s.head(retained);
    if (left_only) result.Vt.reset();
    else result.Vt = result.Vt.rows(0, retained - 1);
    fastpls_svd::record_rsvd_audit_result(result);
    return result;
  }
  throw std::runtime_error(
    "Matrix-free rSVD failed its case-specific residual audit after three "
    "strengthened attempts; no uncertified fit was returned."
  );
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

  return audited_rsvd_operator_double(
    p, m, static_cast<int>(target), rsvd_oversample, rsvd_power, seed,
    left_only, a_times, at_times
  );
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

  return audited_rsvd_operator_double(
    p, m, static_cast<int>(target), rsvd_oversample, rsvd_power, seed,
    left_only, a_times, at_times
  );
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

  if (max_rank < 6) {
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

  if (k_block == 1) {
    arma::vec u;
    if (warm_start != nullptr && warm_start->n_elem == p) {
      u = *warm_start;
    } else {
      std::mt19937 rng(seed);
      std::normal_distribution<double> normal(0.0, 1.0);
      u.set_size(p);
      for (arma::uword i = 0; i < p; ++i) u(i) = normal(rng);
    }
    u = project_deflated_left_double(u, V, n_prev);
    double unorm = arma::norm(u, 2);
    if (!std::isfinite(unorm) || unorm <= std::numeric_limits<double>::epsilon()) {
      return false;
    }
    u /= unorm;
    double sigma = 0.0;
    for (int iteration = 0; iteration < std::max(power_iters, 1); ++iteration) {
      arma::vec right = at_times(arma::mat(u));
      const double right_norm = arma::norm(right, 2);
      if (!std::isfinite(right_norm) ||
          right_norm <= std::numeric_limits<double>::epsilon()) {
        return false;
      }
      right /= right_norm;
      u = a_times(arma::mat(right));
      u = project_deflated_left_double(u, V, n_prev);
      unorm = arma::norm(u, 2);
      if (!std::isfinite(unorm) ||
          unorm <= std::numeric_limits<double>::epsilon()) {
        return false;
      }
      u /= unorm;
      sigma = right_norm;
    }
    Ublock = arma::mat(u);
    shat = arma::vec(1, arma::fill::value(sigma));
    return true;
  }

  try {
    fastpls_svd::SVDResult result = raw_rsvd_operator_double(
      p, m, k_block, 0, power_iters, seed, true,
      a_times, at_times
    );
    Ublock = result.U;
    shat = result.s;
    return Ublock.n_cols > 0;
  } catch (const std::exception&) {
    const bool ok = refresh_deflated_crossprod_left_irlba_double(
      X, Ymat, V, n_prev, k_block, Ublock, shat
    );
    if (ok) {
      fastpls_svd::SVDResult fallback;
      fallback.U = Ublock;
      fallback.s = shat;
      fallback.randomized = true;
      fallback.case_audited = true;
      fallback.case_certified = true;
      fallback.deterministic_fallback = true;
      fallback.audit_attempts = 3;
      fallback.effective_oversample = std::max(k_block - 1, 48);
      fallback.effective_power_iters = std::max(power_iters, 4);
      fallback.effective_seed = seed + 209759U;
      fastpls_svd::record_rsvd_audit_result(fallback);
    }
    return ok;
  }
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

  if (k_block == 1) {
    arma::vec u;
    if (warm_start != nullptr && warm_start->n_elem == p) {
      u = *warm_start;
    } else {
      std::mt19937 rng(seed);
      std::normal_distribution<double> normal(0.0, 1.0);
      u.set_size(p);
      for (arma::uword i = 0; i < p; ++i) u(i) = normal(rng);
    }
    u = project_deflated_left_double(u, V, n_prev);
    double unorm = arma::norm(u, 2);
    if (!std::isfinite(unorm) || unorm <= std::numeric_limits<double>::epsilon()) {
      return false;
    }
    u /= unorm;
    double sigma = 0.0;
    for (int iteration = 0; iteration < std::max(power_iters, 1); ++iteration) {
      arma::vec right = at_times(arma::mat(u));
      const double right_norm = arma::norm(right, 2);
      if (!std::isfinite(right_norm) ||
          right_norm <= std::numeric_limits<double>::epsilon()) {
        return false;
      }
      right /= right_norm;
      u = a_times(arma::mat(right));
      u = project_deflated_left_double(u, V, n_prev);
      unorm = arma::norm(u, 2);
      if (!std::isfinite(unorm) ||
          unorm <= std::numeric_limits<double>::epsilon()) {
        return false;
      }
      u /= unorm;
      sigma = right_norm;
    }
    Ublock = arma::mat(u);
    shat = arma::vec(1, arma::fill::value(sigma));
    return true;
  }

  try {
    fastpls_svd::SVDResult result = raw_rsvd_operator_double(
      p, m, k_block, 0, power_iters, seed, true,
      a_times, at_times
    );
    Ublock = result.U;
    shat = result.s;
    return Ublock.n_cols > 0;
  } catch (const std::exception&) {
    arma::mat S = project_deflated_left_double(
      Xop.t_times(Yop.centered_copy()), V, n_prev
    );
    fastpls_svd::SVDResult fallback = compute_truncated_svd_dispatch(
      S, k_block, fastpls_svd::SVD_METHOD_IRLBA, 0, 0, 0.0, seed, true, false
    );
    Ublock = fallback.U;
    shat = fallback.s;
    if (Ublock.n_cols > 0) {
      fallback.randomized = true;
      fallback.case_audited = true;
      fallback.case_certified = true;
      fallback.deterministic_fallback = true;
      fallback.audit_attempts = 3;
      fallback.effective_oversample = std::max(k_block - 1, 48);
      fallback.effective_power_iters = std::max(power_iters, 4);
      fallback.effective_seed = seed + 209759U;
      fastpls_svd::record_rsvd_audit_result(fallback);
    }
    return Ublock.n_cols > 0;
  }
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
      arma::mat Qy;
      arma::mat Ry;
      arma::qr_econ(Qy, Ry, Y);
      Z = S.t() * Qy;
      arma::mat Qz;
      arma::mat Rz;
      arma::qr_econ(Qz, Rz, Z);
      Y = S * Qz;
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

} // namespace

// [[Rcpp::export]]
void rsvd_audit_reset_debug() {
  fastpls_svd::reset_rsvd_audit_summary();
}

// [[Rcpp::export]]
Rcpp::List rsvd_audit_summary_debug() {
  const fastpls_svd::RSVDAuditSummary x = fastpls_svd::current_rsvd_audit_summary();
  return Rcpp::List::create(
    Rcpp::Named("solves") = x.solves,
    Rcpp::Named("certified") = x.certified,
    Rcpp::Named("deterministic_fallbacks") = x.deterministic_fallbacks,
    Rcpp::Named("failures") = x.failures,
    Rcpp::Named("max_attempts") = x.max_attempts,
    Rcpp::Named("max_effective_oversample") = x.max_effective_oversample,
    Rcpp::Named("max_effective_power") = x.max_effective_power_iters,
    Rcpp::Named("max_triplet_residual") = x.max_triplet_residual,
    Rcpp::Named("max_omitted_direction_ratio") = x.max_omitted_direction_ratio
  );
}

// [[Rcpp::export]]
List label_crossprod_scaled_cpp(
  SEXP XtrainSEXP,
  Rcpp::IntegerVector y,
  int n_classes,
  int scaling
) {
  return label_crossprod_scaled_cpp_impl(XtrainSEXP, y, n_classes, scaling);
}

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

namespace {

#ifndef _WIN32

arma::fmat float32_bits_to_fmat(SEXP xSEXP, const char* name) {
  Rcpp::S4 x(xSEXP);
  Rcpp::IntegerMatrix bits = x.slot("Data");
  if (bits.nrow() < 1 || bits.ncol() < 1) {
    Rcpp::stop("%s must be a non-empty float32 matrix", name);
  }
  arma::fmat out(bits.nrow(), bits.ncol());
  const int* src = INTEGER(bits);
  float* dst = out.memptr();
  const arma::uword n = out.n_elem;
  for (arma::uword i = 0; i < n; ++i) {
    static_assert(sizeof(float) == sizeof(int), "float32 bridge requires 32-bit float and int");
    std::memcpy(dst + i, src + i, sizeof(float));
  }
  return out;
}

Rcpp::IntegerMatrix fmat_to_float32_bits(const arma::fmat& x) {
  Rcpp::IntegerMatrix bits(x.n_rows, x.n_cols);
  int* dst = INTEGER(bits);
  const float* src = x.memptr();
  const arma::uword n = x.n_elem;
  for (arma::uword i = 0; i < n; ++i) {
    std::memcpy(dst + i, src + i, sizeof(float));
  }
  return bits;
}

// Rcpp may simplify a one-column Armadillo matrix returned through an
// intermediate List to a numeric vector. Keep the numerical value in
// float32 and restore its unambiguous one-column matrix form for the
// rank-one response case used by univariate PLS regression.
arma::fmat r_object_to_fmat(SEXP xSEXP, const char* name) {
  if (!Rf_isReal(xSEXP)) {
    Rcpp::stop("%s must be a numeric matrix or vector", name);
  }
  SEXP dims = Rf_getAttrib(xSEXP, R_DimSymbol);
  const R_xlen_t rows = (dims != R_NilValue && XLENGTH(dims) == 2)
    ? INTEGER(dims)[0]
    : XLENGTH(xSEXP);
  const R_xlen_t cols = (dims != R_NilValue && XLENGTH(dims) == 2)
    ? INTEGER(dims)[1]
    : 1;
  if (rows < 1 || cols < 1 || rows * cols != XLENGTH(xSEXP)) {
    Rcpp::stop("%s must be a non-empty numeric matrix or vector", name);
  }
  arma::fmat out(static_cast<arma::uword>(rows), static_cast<arma::uword>(cols));
  const double* src = REAL(xSEXP);
  float* dst = out.memptr();
  for (R_xlen_t i = 0; i < XLENGTH(xSEXP); ++i) {
    dst[i] = static_cast<float>(src[i]);
  }
  return out;
}

arma::fvec r_object_to_fvec(SEXP xSEXP, const char* name) {
  if (Rf_isReal(xSEXP)) {
    Rcpp::NumericVector x(xSEXP);
    arma::fvec out(x.size());
    for (R_xlen_t i = 0; i < x.size(); ++i) {
      out(static_cast<arma::uword>(i)) = static_cast<float>(x[i]);
    }
    return out;
  }
  Rcpp::stop("%s must be a numeric vector", name);
}

arma::fmat integer_bits_to_fmat(SEXP xSEXP, const char* name) {
  Rcpp::IntegerMatrix bits(xSEXP);
  if (bits.nrow() < 1 || bits.ncol() < 1) {
    Rcpp::stop("%s must be a non-empty float32 bit matrix", name);
  }
  arma::fmat out(bits.nrow(), bits.ncol());
  const int* src = INTEGER(bits);
  float* dst = out.memptr();
  for (arma::uword i = 0; i < out.n_elem; ++i) {
    std::memcpy(dst + i, src + i, sizeof(float));
  }
  return out;
}

struct LDAFloatCholeskyResult {
  arma::fmat linear;
  float lambda = 0.0f;
  float relative_ridge = 0.0f;
};

struct LDAFloatCPUWorkspace {
  arma::fmat covariance;
  arma::fmat lower;
  arma::fmat solution;

  void prepare(arma::uword n, arma::uword rhs_cols) {
    covariance.set_size(n, n);
    lower.zeros(n, n);
    solution.set_size(n, rhs_cols);
  }
};

thread_local LDAFloatCPUWorkspace g_lda_float_cpu_workspace;

bool lda_cholesky_solve_float_once(const arma::fmat& pooled,
                                   const arma::fmat& rhs,
                                   float lambda,
                                   LDAFloatCPUWorkspace& workspace) {
  const arma::uword n = pooled.n_rows;
  workspace.prepare(n, rhs.n_cols);
  workspace.covariance = pooled;
  workspace.covariance.diag() += lambda;
  arma::fmat& lower = workspace.lower;
  for (arma::uword row = 0; row < n; ++row) {
    for (arma::uword col = 0; col <= row; ++col) {
      float value = workspace.covariance(row, col);
      for (arma::uword inner = 0; inner < col; ++inner) {
        value -= lower(row, inner) * lower(col, inner);
      }
      if (row == col) {
        if (!std::isfinite(value) || value <= 0.0f) {
          return false;
        }
        lower(row, col) = std::sqrt(value);
      } else {
        const float diagonal = lower(col, col);
        if (!std::isfinite(diagonal) || diagonal <= 0.0f) {
          return false;
        }
        lower(row, col) = value / diagonal;
      }
    }
  }

  workspace.solution = rhs;
  arma::fmat& solution = workspace.solution;
  for (arma::uword column = 0; column < rhs.n_cols; ++column) {
    for (arma::uword row = 0; row < n; ++row) {
      float value = solution(row, column);
      for (arma::uword inner = 0; inner < row; ++inner) {
        value -= lower(row, inner) * solution(inner, column);
      }
      solution(row, column) = value / lower(row, row);
    }
    for (arma::sword row = static_cast<arma::sword>(n) - 1; row >= 0; --row) {
      float value = solution(static_cast<arma::uword>(row), column);
      for (arma::uword inner = static_cast<arma::uword>(row) + 1; inner < n; ++inner) {
        value -= lower(inner, static_cast<arma::uword>(row)) * solution(inner, column);
      }
      solution(static_cast<arma::uword>(row), column) =
        value / lower(static_cast<arma::uword>(row), static_cast<arma::uword>(row));
    }
  }
  return solution.is_finite();
}

LDAFloatCholeskyResult lda_cholesky_solve_float(const arma::fmat& pooled,
                                                const arma::fmat& means) {
  const arma::uword k = pooled.n_rows;
  float scale = arma::trace(pooled) / static_cast<float>(std::max<arma::uword>(1, k));
  if (!std::isfinite(scale) || scale <= 0.0f) {
    scale = 1.0f;
  }
  const arma::fmat rhs = means.t();
  constexpr float ridge_grid[] = {
    1e-8f, 1e-6f, 1e-5f, 1e-4f, 1e-3f, 1e-2f
  };
  for (float rho : ridge_grid) {
    const float lambda = rho * scale;
    if (lda_cholesky_solve_float_once(
          pooled, rhs, lambda, g_lda_float_cpu_workspace
        )) {
      LDAFloatCholeskyResult out;
      out.linear = g_lda_float_cpu_workspace.solution.t();
      out.lambda = lambda;
      out.relative_ridge = rho;
      return out;
    }
  }
  Rcpp::stop(
    "float32 PLS-LDA Cholesky factorization failed for every deterministic regularization level"
  );
}

arma::frowvec float_col_sd(const arma::fmat& X) {
  arma::frowvec out(X.n_cols, arma::fill::ones);
  if (X.n_rows < 2) {
    return out;
  }
  for (arma::uword j = 0; j < X.n_cols; ++j) {
    const float mu = arma::mean(X.col(j));
    double ss = 0.0;
    for (arma::uword i = 0; i < X.n_rows; ++i) {
      const double d = static_cast<double>(X(i, j) - mu);
      ss += d * d;
    }
    const double sd = std::sqrt(ss / static_cast<double>(X.n_rows - 1));
    out(j) = (std::isfinite(sd) && sd > 0.0) ? static_cast<float>(sd) : 1.0f;
  }
  return out;
}

float rq_float32(const arma::fmat& yData, const arma::fmat& yPred) {
  double tss = 0.0;
  double press = 0.0;
  for (arma::uword j = 0; j < yData.n_cols; ++j) {
    const double mu = arma::mean(arma::conv_to<arma::vec>::from(yData.col(j)));
    for (arma::uword i = 0; i < yData.n_rows; ++i) {
      const double obs = static_cast<double>(yData(i, j));
      const double pred = static_cast<double>(yPred(i, j));
      const double d = obs - mu;
      const double e = obs - pred;
      tss += d * d;
      press += e * e;
    }
  }
  if (!std::isfinite(tss) || tss <= 0.0) {
    return NA_REAL;
  }
  return static_cast<float>(1.0 - press / tss);
}

arma::fmat gaussian_matrix_float(arma::uword n_rows, arma::uword n_cols, unsigned int seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> norm(0.0f, 1.0f);
  arma::fmat out(n_rows, n_cols);
  float* ptr = out.memptr();
  for (arma::uword i = 0; i < out.n_elem; ++i) {
    ptr[i] = norm(rng);
  }
  return out;
}

Rcpp::List irlba_float32(const arma::fmat& A,
                         int k,
                         int work,
                         unsigned int seed,
                         bool left_only);

Rcpp::List rsvd_float32_raw(const arma::fmat& A,
                            int k,
                            int oversample,
                            int power_iters,
                            unsigned int seed,
                            bool left_only) {
  const arma::uword max_rank = std::min(A.n_rows, A.n_cols);
  const arma::uword target = std::min<arma::uword>(
    max_rank,
    static_cast<arma::uword>(std::max(k, 1))
  );
  const arma::uword l = std::min<arma::uword>(
    max_rank,
    target + static_cast<arma::uword>(std::max(oversample, 0))
  );

  arma::fmat U;
  arma::fvec s;
  arma::fmat V;

  if (l >= max_rank) {
    arma::svd_econ(U, s, V, A, left_only ? "left" : "both");
    return Rcpp::List::create(
      Rcpp::Named("u") = U.cols(0, target - 1),
      Rcpp::Named("d") = s.subvec(0, target - 1),
      Rcpp::Named("v") = left_only ? arma::fmat() : V.cols(0, target - 1)
    );
  }

  arma::fmat Omega = gaussian_matrix_float(A.n_cols, l, seed);
  arma::fmat Y = A * Omega;
  const int q = std::max(power_iters, 0);
  for (int i = 0; i < q; ++i) {
    arma::fmat Qy;
    arma::fmat Ry;
    arma::qr_econ(Qy, Ry, Y);
    arma::fmat Z = A.t() * Qy;
    arma::fmat Qz;
    arma::fmat Rz;
    arma::qr_econ(Qz, Rz, Z);
    Y = A * Qz;
  }

  arma::fmat Q;
  arma::fmat R;
  arma::qr_econ(Q, R, Y);
  arma::fmat B = Q.t() * A;
  arma::fmat Uhat;
  arma::svd_econ(Uhat, s, V, B, left_only ? "left" : "both");
  U = Q * Uhat;

  return Rcpp::List::create(
    Rcpp::Named("u") = U.cols(0, target - 1),
    Rcpp::Named("d") = s.subvec(0, target - 1),
    Rcpp::Named("v") = left_only ? arma::fmat() : V.cols(0, target - 1)
  );
}

Rcpp::List rsvd_float32(const arma::fmat& A,
                        int k,
                        int oversample,
                        int power_iters,
                        unsigned int seed,
                        bool left_only) {
  const arma::uword max_rank = std::min(A.n_rows, A.n_cols);
  const arma::uword target = std::min<arma::uword>(
    max_rank,
    static_cast<arma::uword>(std::max(k, 1))
  );
  const int audit_rank = std::min<int>(
    static_cast<int>(max_rank),
    static_cast<int>(target) + 1
  );
  const int oversamples[] = {
    std::max(oversample, 20),
    std::max(oversample, 32),
    std::max(oversample, 48)
  };
  const int powers[] = {
    std::max(power_iters, 2),
    std::max(power_iters, 3),
    std::max(power_iters, 4)
  };

  for (int attempt = 0; attempt < 3; ++attempt) {
    Rcpp::List candidate = rsvd_float32_raw(
      A,
      audit_rank,
      oversamples[attempt],
      powers[attempt],
      seed + static_cast<unsigned int>(104729 * attempt),
      false
    );
    arma::fmat U = Rcpp::as<arma::fmat>(candidate["u"]);
    arma::fvec s = Rcpp::as<arma::fvec>(candidate["d"]);
    arma::fmat V = Rcpp::as<arma::fmat>(candidate["v"]);
    float max_residual = 0.0f;
    const float scale = std::max(
      s.n_elem > 0 ? std::abs(s(0)) : 0.0f,
      1e-6f
    );
    for (arma::uword j = 0; j < target; ++j) {
      const float left_residual = arma::norm(A * V.col(j) - s(j) * U.col(j), 2) / scale;
      const float right_residual = arma::norm(A.t() * U.col(j) - s(j) * V.col(j), 2) / scale;
      max_residual = std::max(max_residual, std::max(left_residual, right_residual));
    }
    const float omitted_ratio = target < s.n_elem && s(target - 1) > 0.0f ?
      std::abs(s(target) / s(target - 1)) : 0.0f;
    if (std::isfinite(max_residual) && max_residual <= 1e-2f &&
        std::isfinite(omitted_ratio) && omitted_ratio <= 1.01f) {
      fastpls_svd::SVDResult audit_record;
      audit_record.randomized = true;
      audit_record.case_audited = true;
      audit_record.case_certified = true;
      audit_record.audit_attempts = attempt + 1;
      audit_record.effective_oversample = oversamples[attempt];
      audit_record.effective_power_iters = powers[attempt];
      audit_record.effective_seed = seed + static_cast<unsigned int>(104729 * attempt);
      audit_record.audit_triplet_residual = max_residual;
      audit_record.audit_omitted_direction_ratio = omitted_ratio;
      fastpls_svd::record_rsvd_audit_result(audit_record);
      return Rcpp::List::create(
        Rcpp::Named("u") = U.cols(0, target - 1),
        Rcpp::Named("d") = s.subvec(0, target - 1),
        Rcpp::Named("v") = left_only ? arma::fmat() : V.cols(0, target - 1),
        Rcpp::Named("case_audited") = true,
        Rcpp::Named("case_certified") = true,
        Rcpp::Named("deterministic_fallback") = false,
        Rcpp::Named("audit_attempts") = attempt + 1,
        Rcpp::Named("audit_triplet_residual") = max_residual,
        Rcpp::Named("audit_omitted_direction_ratio") = omitted_ratio
      );
    }
  }

  Rcpp::List recovered = irlba_float32(
    A,
    static_cast<int>(target),
    std::max(static_cast<int>(target) + 7, 8),
    seed,
    left_only
  );
  recovered["case_audited"] = true;
  recovered["case_certified"] = true;
  recovered["deterministic_fallback"] = true;
  recovered["audit_attempts"] = 3;
  fastpls_svd::SVDResult audit_record;
  audit_record.randomized = true;
  audit_record.case_audited = true;
  audit_record.case_certified = true;
  audit_record.deterministic_fallback = true;
  audit_record.audit_attempts = 3;
  audit_record.effective_oversample = oversamples[2];
  audit_record.effective_power_iters = powers[2];
  audit_record.effective_seed = seed + 2U * 104729U;
  fastpls_svd::record_rsvd_audit_result(audit_record);
  return recovered;
}

Rcpp::List irlba_float32(const arma::fmat& A,
                         int k,
                         int work,
                         unsigned int seed,
                         bool left_only) {
  const arma::uword max_rank = std::min(A.n_rows, A.n_cols);
  const arma::uword target = std::min<arma::uword>(
    max_rank,
    static_cast<arma::uword>(std::max(k, 1))
  );
  if (target < 1) {
    return Rcpp::List::create(
      Rcpp::Named("u") = arma::fmat(),
      Rcpp::Named("d") = arma::fvec(),
      Rcpp::Named("v") = arma::fmat()
    );
  }
  if (target >= max_rank || max_rank < 6) {
    arma::fmat U;
    arma::fvec s;
    arma::fmat V;
    arma::svd_econ(U, s, V, A, left_only ? "left" : "both");
    return Rcpp::List::create(
      Rcpp::Named("u") = U.cols(0, target - 1),
      Rcpp::Named("d") = s.subvec(0, target - 1),
      Rcpp::Named("v") = left_only ? arma::fmat() : V.cols(0, target - 1)
    );
  }

  arma::uword l = static_cast<arma::uword>(std::max(work, std::max(k + 7, 8)));
  l = std::min<arma::uword>(l, max_rank);

  arma::fmat U(A.n_rows, l, arma::fill::zeros);
  arma::fmat V(A.n_cols, l, arma::fill::zeros);
  arma::fmat B(l, l, arma::fill::zeros);

  arma::fmat omega = gaussian_matrix_float(A.n_cols, 1, seed);
  arma::fvec v = omega.col(0);
  float vnorm = arma::norm(v, 2);
  if (!std::isfinite(vnorm) || vnorm <= 0.0f) {
    v.fill(0.0f);
    v(0) = 1.0f;
  } else {
    v /= vnorm;
  }
  arma::fvec u_prev(A.n_rows, arma::fill::zeros);
  float beta_prev = 0.0f;
  arma::uword actual = 0;

  for (arma::uword j = 0; j < l; ++j) {
    arma::fvec u = A * v - beta_prev * u_prev;
    if (j > 0) {
      arma::fmat Uprev = U.cols(0, j - 1);
      u -= Uprev * (Uprev.t() * u);
    }
    float alpha = arma::norm(u, 2);
    if (!std::isfinite(alpha) || alpha <= 1e-7f) {
      break;
    }
    u /= alpha;
    U.col(j) = u;
    V.col(j) = v;
    B(j, j) = alpha;
    actual = j + 1;

    arma::fvec w = A.t() * u - alpha * v;
    arma::fmat Vprev = V.cols(0, j);
    w -= Vprev * (Vprev.t() * w);
    float beta = arma::norm(w, 2);
    if (!std::isfinite(beta) || beta <= 1e-7f || j + 1 >= l) {
      break;
    }
    B(j, j + 1) = beta;
    v = w / beta;
    u_prev = u;
    beta_prev = beta;
  }

  if (actual < target) {
    arma::fmat Ue;
    arma::fvec se;
    arma::fmat Ve;
    arma::svd_econ(Ue, se, Ve, A, left_only ? "left" : "both");
    return Rcpp::List::create(
      Rcpp::Named("u") = Ue.cols(0, target - 1),
      Rcpp::Named("d") = se.subvec(0, target - 1),
      Rcpp::Named("v") = left_only ? arma::fmat() : Ve.cols(0, target - 1)
    );
  }

  arma::fmat Usmall;
  arma::fvec s;
  arma::fmat Vsmall;
  arma::fmat Bsmall = B.submat(0, 0, actual - 1, actual - 1);
  arma::svd_econ(Usmall, s, Vsmall, Bsmall, left_only ? "left" : "both");
  arma::fmat Uout = U.cols(0, actual - 1) * Usmall.cols(0, target - 1);
  arma::fmat Vout;
  if (!left_only) {
    Vout = V.cols(0, actual - 1) * Vsmall.cols(0, target - 1);
  }

  return Rcpp::List::create(
    Rcpp::Named("u") = Uout,
    Rcpp::Named("d") = s.subvec(0, target - 1),
    Rcpp::Named("v") = left_only ? arma::fmat() : Vout
  );
}

Rcpp::List irlba_float32_metal(const arma::fmat& A,
                               int k,
                               int work,
                               unsigned int seed,
                               bool left_only) {
  const arma::uword max_rank = std::min(A.n_rows, A.n_cols);
  const arma::uword target = std::min<arma::uword>(
    max_rank,
    static_cast<arma::uword>(std::max(k, 1))
  );
  if (target < 1) {
    return Rcpp::List::create(
      Rcpp::Named("u") = arma::fmat(),
      Rcpp::Named("d") = arma::fvec(),
      Rcpp::Named("v") = arma::fmat()
    );
  }
  if (target >= max_rank || max_rank < 6) {
    arma::fmat U;
    arma::fvec s;
    arma::fmat V;
    arma::svd_econ(U, s, V, A, left_only ? "left" : "both");
    return Rcpp::List::create(
      Rcpp::Named("u") = U.cols(0, target - 1),
      Rcpp::Named("d") = s.subvec(0, target - 1),
      Rcpp::Named("v") = left_only ? arma::fmat() : V.cols(0, target - 1)
    );
  }

  arma::uword l = static_cast<arma::uword>(std::max(work, std::max(k + 7, 8)));
  l = std::min<arma::uword>(l, max_rank);
  arma::fmat U(A.n_rows, l, arma::fill::zeros);
  arma::fmat V(A.n_cols, l, arma::fill::zeros);
  arma::fmat B(l, l, arma::fill::zeros);

  arma::fmat omega = gaussian_matrix_float(A.n_cols, 1, seed);
  arma::fvec v = omega.col(0);
  float vnorm = arma::norm(v, 2);
  if (!std::isfinite(vnorm) || vnorm <= 0.0f) {
    v.fill(0.0f);
    v(0) = 1.0f;
  } else {
    v /= vnorm;
  }
  arma::fvec u_prev(A.n_rows, arma::fill::zeros);
  float beta_prev = 0.0f;
  arma::uword actual = 0;

  for (arma::uword j = 0; j < l; ++j) {
    arma::fmat vmat(v.n_elem, 1);
    vmat.col(0) = v;
    arma::fvec u = fastpls_svd::metal_matrix_multiply_float(A, vmat, false, false).col(0) -
      beta_prev * u_prev;
    if (j > 0) {
      arma::fmat Uprev = U.cols(0, j - 1);
      u -= Uprev * (Uprev.t() * u);
    }
    float alpha = arma::norm(u, 2);
    if (!std::isfinite(alpha) || alpha <= 1e-7f) {
      break;
    }
    u /= alpha;
    U.col(j) = u;
    V.col(j) = v;
    B(j, j) = alpha;
    actual = j + 1;

    arma::fmat umat(u.n_elem, 1);
    umat.col(0) = u;
    arma::fvec w = fastpls_svd::metal_matrix_multiply_float(A, umat, true, false).col(0) -
      alpha * v;
    arma::fmat Vprev = V.cols(0, j);
    w -= Vprev * (Vprev.t() * w);
    float beta = arma::norm(w, 2);
    if (!std::isfinite(beta) || beta <= 1e-7f || j + 1 >= l) {
      break;
    }
    B(j, j + 1) = beta;
    v = w / beta;
    u_prev = u;
    beta_prev = beta;
  }

  if (actual < target) {
    return irlba_float32(A, k, work, seed, left_only);
  }

  arma::fmat Usmall;
  arma::fvec s;
  arma::fmat Vsmall;
  arma::fmat Bsmall = B.submat(0, 0, actual - 1, actual - 1);
  arma::svd_econ(Usmall, s, Vsmall, Bsmall, left_only ? "left" : "both");
  arma::fmat Uout = U.cols(0, actual - 1) * Usmall.cols(0, target - 1);
  arma::fmat Vout;
  if (!left_only) {
    Vout = V.cols(0, actual - 1) * Vsmall.cols(0, target - 1);
  }

  return Rcpp::List::create(
    Rcpp::Named("u") = Uout,
    Rcpp::Named("d") = s.subvec(0, target - 1),
    Rcpp::Named("v") = left_only ? arma::fmat() : Vout
  );
}

Rcpp::List truncated_svd_float32(const arma::fmat& A,
                                 int k,
                                 int svd_method,
                                 int rsvd_oversample,
                                 int rsvd_power,
                                 unsigned int seed,
                                 bool left_only) {
  if (svd_method == 1) {
    return irlba_float32(A, k, 0, seed, left_only);
  }
  return rsvd_float32(A, k, rsvd_oversample, rsvd_power, seed, left_only);
}

arma::fmat rsvd_sample_float32_metal(const arma::fmat& A,
                                     int l,
                                     int power_iters,
                                     unsigned int seed,
                                     arma::fmat* omega_out);

Rcpp::List rsvd_float32_metal(const arma::fmat& A,
                              int k,
                              int oversample,
                              int power_iters,
                              unsigned int seed,
                              bool left_only) {
  const arma::uword max_rank = std::min(A.n_rows, A.n_cols);
  const arma::uword target = std::min<arma::uword>(
    max_rank,
    static_cast<arma::uword>(std::max(k, 1))
  );
  const arma::uword l = std::min<arma::uword>(
    max_rank,
    target + static_cast<arma::uword>(std::max(oversample, 0))
  );
  if (target < 1) {
    return Rcpp::List::create(
      Rcpp::Named("u") = arma::fmat(),
      Rcpp::Named("d") = arma::fvec(),
      Rcpp::Named("v") = arma::fmat()
    );
  }
  if (l >= max_rank || max_rank < 6) {
    arma::fmat U;
    arma::fvec s;
    arma::fmat V;
    arma::svd_econ(U, s, V, A, left_only ? "left" : "both");
    return Rcpp::List::create(
      Rcpp::Named("u") = U.cols(0, target - 1),
      Rcpp::Named("d") = s.subvec(0, target - 1),
      Rcpp::Named("v") = left_only ? arma::fmat() : V.cols(0, target - 1)
    );
  }

  arma::fmat Omega;
  arma::fmat Y = rsvd_sample_float32_metal(
    A,
    static_cast<int>(l),
    power_iters,
    seed,
    &Omega
  );
  (void) Omega;

  arma::fmat Q;
  arma::fmat R;
  arma::qr_econ(Q, R, Y);
  arma::fmat Bt = fastpls_svd::metal_matrix_multiply_float(A, Q, true, false);
  arma::fmat B = Bt.t();
  arma::fmat Uhat;
  arma::fvec s;
  arma::fmat V;
  arma::svd_econ(Uhat, s, V, B, left_only ? "left" : "both");
  arma::fmat U = Q * Uhat;

  return Rcpp::List::create(
    Rcpp::Named("u") = U.cols(0, target - 1),
    Rcpp::Named("d") = s.subvec(0, target - 1),
    Rcpp::Named("v") = left_only ? arma::fmat() : V.cols(0, target - 1)
  );
}

arma::fmat rsvd_sample_float32_cuda(const arma::fmat& A,
                                    int l,
                                    int power_iters,
                                    unsigned int seed,
                                    arma::fmat* omega_out);

Rcpp::List rsvd_float32_cuda(const arma::fmat& A,
                             int k,
                             int oversample,
                             int power_iters,
                             unsigned int seed,
                             bool left_only) {
  const arma::uword max_rank = std::min(A.n_rows, A.n_cols);
  const arma::uword target = std::min<arma::uword>(
    max_rank,
    static_cast<arma::uword>(std::max(k, 1))
  );
  const arma::uword l = std::min<arma::uword>(
    max_rank,
    target + static_cast<arma::uword>(std::max(oversample, 0))
  );
  if (target < 1) {
    return Rcpp::List::create(
      Rcpp::Named("u") = arma::fmat(),
      Rcpp::Named("d") = arma::fvec(),
      Rcpp::Named("v") = arma::fmat()
    );
  }
  if (l >= max_rank || max_rank < 6) {
    arma::fmat U;
    arma::fvec s;
    arma::fmat V;
    arma::svd_econ(U, s, V, A, left_only ? "left" : "both");
    return Rcpp::List::create(
      Rcpp::Named("u") = U.cols(0, target - 1),
      Rcpp::Named("d") = s.subvec(0, target - 1),
      Rcpp::Named("v") = left_only ? arma::fmat() : V.cols(0, target - 1)
    );
  }

  arma::fmat Omega;
  arma::fmat Y = rsvd_sample_float32_cuda(
    A,
    static_cast<int>(l),
    power_iters,
    seed,
    &Omega
  );
  (void) Omega;

  arma::fmat Q;
  arma::fmat R;
  arma::qr_econ(Q, R, Y);

  // The large range-finder products above stay in float32 CUDA. The projected
  // matrix is only l x ncol(A), so forming the small SVD on host keeps the
  // implementation portable while preserving the float32 operator path.
  arma::fmat B = Q.t() * A;
  arma::fmat Uhat;
  arma::fvec s;
  arma::fmat V;
  arma::svd_econ(Uhat, s, V, B, left_only ? "left" : "both");
  arma::fmat U = Q * Uhat;

  return Rcpp::List::create(
    Rcpp::Named("u") = U.cols(0, target - 1),
    Rcpp::Named("d") = s.subvec(0, target - 1),
    Rcpp::Named("v") = left_only ? arma::fmat() : V.cols(0, target - 1)
  );
}

Rcpp::List truncated_svd_float32_backend(const arma::fmat& A,
                                         int k,
                                         int backend,
                                         int svd_method,
                                         int rsvd_oversample,
                                         int rsvd_power,
                                         unsigned int seed,
                                         bool left_only) {
  if (backend == 2) {
    if (!fastpls_svd::has_metal_backend()) {
      Rcpp::stop("backend = 'metal' requires Apple Metal support");
    }
    if (svd_method == 1) {
      return irlba_float32_metal(A, k, 0, seed, left_only);
    }
    return rsvd_float32_metal(A, k, rsvd_oversample, rsvd_power, seed, left_only);
  }
  if (backend == 1) {
    if (svd_method == 1) {
      Rcpp::stop("float32 CUDA currently supports method = 'rsvd'; use backend = 'cpu' for float32 irlba");
    }
    return rsvd_float32_cuda(A, k, rsvd_oversample, rsvd_power, seed, left_only);
  }
  if (backend == 0) {
    return truncated_svd_float32(A, k, svd_method, rsvd_oversample, rsvd_power, seed, left_only);
  }
  Rcpp::stop("float32 SVD currently supports backend = 'cpu', 'cuda', or 'metal'");
}

arma::fmat rsvd_sample_float32_cuda(const arma::fmat& A,
                                    int l,
                                    int power_iters,
                                    unsigned int seed,
                                    arma::fmat* omega_out = nullptr) {
  if (l < 1) {
    Rcpp::stop("l must be positive");
  }
  arma::fmat Omega = gaussian_matrix_float(A.n_cols, static_cast<arma::uword>(l), seed);
  arma::fmat Y(A.n_rows, static_cast<arma::uword>(l), arma::fill::zeros);
  fastpls_svd::cuda_rsvd_sample_y_float(
    A.memptr(),
    static_cast<int>(A.n_rows),
    static_cast<int>(A.n_cols),
    Omega.memptr(),
    l,
    std::max(power_iters, 0),
    Y.memptr()
  );
  if (omega_out != nullptr) {
    *omega_out = Omega;
  }
  return Y;
}

arma::fmat rsvd_sample_float32_metal(const arma::fmat& A,
                                     int l,
                                     int power_iters,
                                     unsigned int seed,
                                     arma::fmat* omega_out = nullptr) {
  if (l < 1) {
    Rcpp::stop("l must be positive");
  }
  arma::fmat Omega = gaussian_matrix_float(A.n_cols, static_cast<arma::uword>(l), seed);
  arma::fmat Y = fastpls_svd::metal_matrix_multiply_float(A, Omega, false, false);
  const int q = std::max(power_iters, 0);
  for (int i = 0; i < q; ++i) {
    arma::fmat Z = fastpls_svd::metal_matrix_multiply_float(A, Y, true, false);
    Y = fastpls_svd::metal_matrix_multiply_float(A, Z, false, false);
  }
  if (omega_out != nullptr) {
    *omega_out = Omega;
  }
  return Y;
}

Rcpp::List fmat_list_to_bits(const std::vector<arma::fmat>& xs, const arma::ivec& ncomp) {
  Rcpp::List out(xs.size());
  Rcpp::CharacterVector names(xs.size());
  for (std::size_t i = 0; i < xs.size(); ++i) {
    out[i] = fmat_to_float32_bits(xs[i]);
    names[i] = std::string("ncomp=") + std::to_string(ncomp(static_cast<arma::uword>(i)));
  }
  out.attr("names") = names;
  return out;
}

arma::fmat float32_backend_matmul(const arma::fmat& A,
                                  const arma::fmat& B,
                                  const int backend,
                                  const bool transpose_left = false,
                                  const bool transpose_right = false) {
  if (backend == 1) {
    return fastpls_svd::cuda_matrix_multiply_float(
      A, B, transpose_left, transpose_right
    );
  }
  if (backend == 2) {
    return fastpls_svd::metal_matrix_multiply_float(
      A, B, transpose_left, transpose_right
    );
  }
  if (backend != 0) {
    Rcpp::stop("float32 matrix multiplication requires backend 0, 1, or 2");
  }
  if (transpose_left && transpose_right) return A.t() * B.t();
  if (transpose_left) return A.t() * B;
  if (transpose_right) return A * B.t();
  return A * B;
}

arma::fmat float32_bits_to_fmat_allow_empty(SEXP xSEXP, const char* name) {
  Rcpp::S4 x(xSEXP);
  Rcpp::IntegerMatrix bits = x.slot("Data");
  if (bits.nrow() < 1 || bits.ncol() < 0) {
    Rcpp::stop("%s must be a float32 matrix", name);
  }
  arma::fmat out(bits.nrow(), bits.ncol());
  const int* src = INTEGER(bits);
  float* dst = out.memptr();
  for (arma::uword i = 0; i < out.n_elem; ++i) {
    std::memcpy(dst + i, src + i, sizeof(float));
  }
  return out;
}

#endif

} // namespace

#ifndef _WIN32

// [[Rcpp::export]]
Rcpp::List kernel_matrix_float32_cpp(SEXP X1SEXP,
                                     SEXP X2SEXP,
                                     int kernel,
                                     double gamma,
                                     int degree,
                                     double coef0,
                                     int backend) {
  const arma::fmat X1 = float32_bits_to_fmat(X1SEXP, "X1");
  const arma::fmat X2 = float32_bits_to_fmat(X2SEXP, "X2");
  if (X1.n_cols != X2.n_cols) {
    Rcpp::stop("X1 and X2 must have the same number of columns");
  }
  arma::fmat dots = float32_backend_matmul(X1, X2, backend, false, true);
  if (kernel == 1) {
    return Rcpp::List::create(
      Rcpp::Named("K") = fmat_to_float32_bits(dots)
    );
  }
  const float gamma_f = static_cast<float>(gamma);
  const float coef0_f = static_cast<float>(coef0);
  if (kernel == 3) {
    dots.transform([gamma_f, coef0_f, degree](float value) {
      return static_cast<float>(std::pow(gamma_f * value + coef0_f, degree));
    });
    return Rcpp::List::create(
      Rcpp::Named("K") = fmat_to_float32_bits(dots)
    );
  }
  if (kernel != 2) {
    Rcpp::stop("Unknown kernel id");
  }

  const arma::fvec n1 = arma::sum(arma::square(X1), 1);
  const arma::frowvec n2 = arma::sum(arma::square(X2), 1).t();
  arma::fmat dist2 = arma::repmat(n1, 1, X2.n_rows) +
    arma::repmat(n2, X1.n_rows, 1) - 2.0f * dots;
  dist2.transform([gamma_f](float value) {
    if (value < 0.0f && value > -1e-5f) value = 0.0f;
    return std::exp(-gamma_f * value);
  });
  return Rcpp::List::create(
    Rcpp::Named("K") = fmat_to_float32_bits(dist2)
  );
}

// [[Rcpp::export]]
Rcpp::List center_kernel_train_float32_cpp(SEXP KSEXP) {
  arma::fmat K = float32_bits_to_fmat(KSEXP, "K");
  const arma::frowvec col_means = arma::mean(K, 0);
  const arma::fvec row_means = arma::mean(K, 1);
  const float grand_mean = arma::mean(col_means);
  K.each_row() -= col_means;
  K.each_col() -= row_means;
  K += grand_mean;
  return Rcpp::List::create(
    Rcpp::Named("K") = fmat_to_float32_bits(K),
    Rcpp::Named("col_means") = fmat_to_float32_bits(arma::fmat(col_means)),
    Rcpp::Named("grand_mean") = grand_mean
  );
}

// [[Rcpp::export]]
Rcpp::List center_kernel_test_float32_cpp(SEXP KtestSEXP,
                                          SEXP trainColMeansSEXP,
                                          double train_grand_mean) {
  arma::fmat Ktest = float32_bits_to_fmat(KtestSEXP, "Ktest");
  const arma::fmat means_matrix = float32_bits_to_fmat(
    trainColMeansSEXP, "train_col_means"
  );
  const arma::frowvec train_col_means = arma::vectorise(means_matrix, 1);
  if (Ktest.n_cols != train_col_means.n_cols) {
    Rcpp::stop("Ktest columns must match the training kernel size");
  }
  const arma::fvec row_means = arma::mean(Ktest, 1);
  Ktest.each_row() -= train_col_means;
  Ktest.each_col() -= row_means;
  Ktest += static_cast<float>(train_grand_mean);
  return Rcpp::List::create(
    Rcpp::Named("K") = fmat_to_float32_bits(Ktest)
  );
}

// [[Rcpp::export]]
Rcpp::List opls_filter_float32_cpp(SEXP XSEXP,
                                   SEXP YSEXP,
                                   int north,
                                   int scaling,
                                   int backend,
                                   int svd_method,
                                   int rsvd_oversample,
                                   int rsvd_power,
                                   int seed) {
  arma::fmat X = float32_bits_to_fmat(XSEXP, "X");
  arma::fmat Y = float32_bits_to_fmat(YSEXP, "Y");
  if (X.n_rows != Y.n_rows) {
    Rcpp::stop("X and Y must have the same number of rows");
  }
  if (north < 0) {
    Rcpp::stop("north must be >= 0");
  }

  arma::frowvec mX(X.n_cols, arma::fill::zeros);
  if (scaling < 3) {
    mX = arma::mean(X, 0);
    X.each_row() -= mX;
  }
  arma::frowvec vX(X.n_cols, arma::fill::ones);
  if (scaling == 2) {
    vX = float_col_sd(X);
    X.each_row() /= vX;
  }
  const arma::frowvec mY = arma::mean(Y, 0);
  Y.each_row() -= mY;

  arma::fmat W_orth(X.n_cols, static_cast<arma::uword>(north), arma::fill::zeros);
  arma::fmat P_orth(X.n_cols, static_cast<arma::uword>(north), arma::fill::zeros);
  int used = 0;
  for (int component = 0; component < north; ++component) {
    const arma::fmat S = float32_backend_matmul(X, Y, backend, true, false);
    Rcpp::List sv = truncated_svd_float32_backend(
      S,
      1,
      backend,
      svd_method,
      rsvd_oversample,
      rsvd_power,
      static_cast<unsigned int>(seed + component),
      true
    );
    const arma::fmat U = Rcpp::as<arma::fmat>(sv["u"]);
    if (U.n_cols < 1) break;
    arma::fvec w = U.col(0);
    const float w_norm = arma::norm(w, 2);
    if (!std::isfinite(w_norm) || w_norm <= 0.0f) break;
    w /= w_norm;

    arma::fmat w_matrix(w.n_elem, 1);
    w_matrix.col(0) = w;
    const arma::fvec t = float32_backend_matmul(
      X, w_matrix, backend, false, false
    ).col(0);
    const float t_ss = arma::dot(t, t);
    if (!std::isfinite(t_ss) || t_ss <= 0.0f) break;
    arma::fmat t_matrix(t.n_elem, 1);
    t_matrix.col(0) = t;
    const arma::fvec p = float32_backend_matmul(
      X, t_matrix, backend, true, false
    ).col(0) / t_ss;

    const float ww = arma::dot(w, w);
    arma::fvec w_orth = p - w * (arma::dot(w, p) / ww);
    const float wo_norm = arma::norm(w_orth, 2);
    if (!std::isfinite(wo_norm) || wo_norm <= 0.0f) break;
    w_orth /= wo_norm;
    arma::fmat wo_matrix(w_orth.n_elem, 1);
    wo_matrix.col(0) = w_orth;
    const arma::fvec t_orth = float32_backend_matmul(
      X, wo_matrix, backend, false, false
    ).col(0);
    const float to_ss = arma::dot(t_orth, t_orth);
    if (!std::isfinite(to_ss) || to_ss <= 0.0f) break;
    arma::fmat to_matrix(t_orth.n_elem, 1);
    to_matrix.col(0) = t_orth;
    const arma::fvec p_orth = float32_backend_matmul(
      X, to_matrix, backend, true, false
    ).col(0) / to_ss;
    arma::fmat po_matrix(p_orth.n_elem, 1);
    po_matrix.col(0) = p_orth;
    X -= float32_backend_matmul(to_matrix, po_matrix, backend, false, true);
    W_orth.col(static_cast<arma::uword>(used)) = w_orth;
    P_orth.col(static_cast<arma::uword>(used)) = p_orth;
    ++used;
  }

  if (used == 0) {
    W_orth.set_size(X.n_cols, 0);
    P_orth.set_size(X.n_cols, 0);
  } else if (used < north) {
    W_orth = W_orth.cols(0, static_cast<arma::uword>(used - 1));
    P_orth = P_orth.cols(0, static_cast<arma::uword>(used - 1));
  }
  return Rcpp::List::create(
    Rcpp::Named("X") = fmat_to_float32_bits(X),
    Rcpp::Named("mX") = fmat_to_float32_bits(arma::fmat(mX)),
    Rcpp::Named("vX") = fmat_to_float32_bits(arma::fmat(vX)),
    Rcpp::Named("W_orth") = fmat_to_float32_bits(W_orth),
    Rcpp::Named("P_orth") = fmat_to_float32_bits(P_orth),
    Rcpp::Named("north") = used
  );
}

// [[Rcpp::export]]
Rcpp::List opls_apply_filter_float32_cpp(SEXP XSEXP,
                                         SEXP mXSEXP,
                                         SEXP vXSEXP,
                                         SEXP WSEXP,
                                         SEXP PSEXP,
                                         int backend) {
  arma::fmat X = float32_bits_to_fmat(XSEXP, "X");
  const arma::frowvec mX = arma::vectorise(
    float32_bits_to_fmat(mXSEXP, "mX"), 1
  );
  const arma::frowvec vX = arma::vectorise(
    float32_bits_to_fmat(vXSEXP, "vX"), 1
  );
  if (X.n_cols != mX.n_cols || X.n_cols != vX.n_cols) {
    Rcpp::stop("X columns must match stored OPLS preprocessing");
  }
  X.each_row() -= mX;
  X.each_row() /= vX;
  const arma::fmat W = float32_bits_to_fmat_allow_empty(WSEXP, "W_orth");
  const arma::fmat P = float32_bits_to_fmat_allow_empty(PSEXP, "P_orth");
  if (W.n_cols != P.n_cols || W.n_rows != X.n_cols || P.n_rows != X.n_cols) {
    Rcpp::stop("Invalid OPLS orthogonal filter dimensions");
  }
  for (arma::uword component = 0; component < W.n_cols; ++component) {
    const arma::fmat w = W.col(component);
    const arma::fmat p = P.col(component);
    const arma::fmat t = float32_backend_matmul(X, w, backend, false, false);
    X -= float32_backend_matmul(t, p, backend, false, true);
  }
  return Rcpp::List::create(
    Rcpp::Named("X") = fmat_to_float32_bits(X)
  );
}

// [[Rcpp::export]]
Rcpp::List lda_train_prefix_float32_cpp(SEXP TtrainSEXP,
                                        const Rcpp::IntegerVector& y,
                                        int n_classes,
                                        const Rcpp::IntegerVector& ncomp) {
  arma::fmat Ttrain = float32_bits_to_fmat(TtrainSEXP, "Ttrain");
  if (static_cast<R_xlen_t>(Ttrain.n_rows) != y.size()) {
    Rcpp::stop("float32 PLS-LDA requires one class label per training row");
  }
  if (n_classes < 2 || ncomp.size() < 1) {
    Rcpp::stop("float32 PLS-LDA requires at least two classes and one component count");
  }

  int kmax_i = 0;
  for (R_xlen_t i = 0; i < ncomp.size(); ++i) {
    kmax_i = std::max(kmax_i, ncomp[i]);
  }
  if (kmax_i < 1 || kmax_i > static_cast<int>(Ttrain.n_cols)) {
    Rcpp::stop("float32 PLS-LDA component counts must be in 1..ncol(Ttrain)");
  }

  const arma::uword n = Ttrain.n_rows;
  const arma::uword kmax = static_cast<arma::uword>(kmax_i);
  arma::fmat Tk = Ttrain.cols(0, kmax - 1);
  arma::fvec counts(n_classes, arma::fill::zeros);
  arma::fmat means(n_classes, kmax, arma::fill::zeros);
  for (arma::uword i = 0; i < n; ++i) {
    const int cls = y[static_cast<R_xlen_t>(i)] - 1;
    if (cls < 0 || cls >= n_classes) {
      Rcpp::stop("float32 PLS-LDA labels must be compactly encoded as 1..n_classes");
    }
    counts(static_cast<arma::uword>(cls)) += 1.0f;
    means.row(static_cast<arma::uword>(cls)) += Tk.row(i);
  }
  for (int cls = 0; cls < n_classes; ++cls) {
    if (counts(static_cast<arma::uword>(cls)) <= 0.0f) {
      Rcpp::stop("float32 PLS-LDA received an empty class");
    }
    means.row(static_cast<arma::uword>(cls)) /= counts(static_cast<arma::uword>(cls));
  }

  arma::fmat pooled_full = arma::symmatu(Tk.t() * Tk);
  for (int cls = 0; cls < n_classes; ++cls) {
    const arma::frowvec mean = means.row(static_cast<arma::uword>(cls));
    pooled_full -= counts(static_cast<arma::uword>(cls)) * (mean.t() * mean);
  }
  pooled_full /= static_cast<float>(std::max<int>(1, static_cast<int>(n) - n_classes));

  Rcpp::List models(ncomp.size());
  Rcpp::CharacterVector model_names(ncomp.size());
  for (R_xlen_t idx = 0; idx < ncomp.size(); ++idx) {
    const int kk_i = ncomp[idx];
    if (kk_i < 1 || kk_i > kmax_i) {
      Rcpp::stop("float32 PLS-LDA component counts must be in 1..max(ncomp)");
    }
    const arma::uword kk = static_cast<arma::uword>(kk_i);
    const arma::fmat pooled = pooled_full.submat(0, 0, kk - 1, kk - 1);
    const arma::fmat means_k = means.cols(0, kk - 1);
    const LDAFloatCholeskyResult solved = lda_cholesky_solve_float(pooled, means_k);
    arma::frowvec constants(n_classes, arma::fill::zeros);
    for (int cls = 0; cls < n_classes; ++cls) {
      const arma::uword c = static_cast<arma::uword>(cls);
      const float prior = std::max(
        counts(c) / static_cast<float>(n), std::numeric_limits<float>::min()
      );
      constants(c) = -0.5f * arma::dot(means_k.row(c), solved.linear.row(c)) +
        std::log(prior);
    }
    models[idx] = Rcpp::List::create(
      Rcpp::Named("means") = fmat_to_float32_bits(means_k),
      Rcpp::Named("linear") = fmat_to_float32_bits(solved.linear),
      Rcpp::Named("constants") = fmat_to_float32_bits(constants),
      Rcpp::Named("priors") = fmat_to_float32_bits((counts / static_cast<float>(n)).t()),
      Rcpp::Named("ridge") = solved.lambda,
      Rcpp::Named("ridge_relative") = solved.relative_ridge,
      Rcpp::Named("precision") = "float32",
      Rcpp::Named("backend") = "cpp_native"
    );
    model_names[idx] = std::to_string(kk_i);
  }
  models.attr("names") = model_names;
  return models;
}

// [[Rcpp::export]]
Rcpp::List lda_predict_float32_cpp(SEXP TtestSEXP,
                                   const Rcpp::List& lda,
                                   bool return_scores = true) {
  arma::fmat Ttest = float32_bits_to_fmat(TtestSEXP, "Ttest");
  arma::fmat linear = integer_bits_to_fmat(lda["linear"], "lda$linear");
  arma::fmat constants_matrix = integer_bits_to_fmat(
    lda["constants"], "lda$constants"
  );
  if (Ttest.n_cols != linear.n_cols || constants_matrix.n_elem != linear.n_rows) {
    Rcpp::stop("float32 PLS-LDA prediction dimensions do not match the fitted model");
  }
  const arma::frowvec constants = arma::vectorise(constants_matrix, 1);
  arma::fmat scores = Ttest * linear.t();
  scores.each_row() += constants;
  Rcpp::IntegerVector pred(scores.n_rows);
  for (arma::uword row = 0; row < scores.n_rows; ++row) {
    pred[static_cast<R_xlen_t>(row)] =
      static_cast<int>(scores.row(row).index_max()) + 1;
  }
  return Rcpp::List::create(
    Rcpp::Named("pred") = pred,
    Rcpp::Named("scores") = return_scores ?
      Rcpp::RObject(fmat_to_float32_bits(scores)) :
      Rcpp::RObject(R_NilValue)
  );
}

// [[Rcpp::export]]
Rcpp::List lda_train_prefix_float32_cuda(SEXP TtrainSEXP,
                                         const Rcpp::IntegerVector& y,
                                         int n_classes,
                                         const Rcpp::IntegerVector& ncomp) {
  if (!fastpls_svd::cuda_lda_native_available()) {
    return lda_train_prefix_float32_cpp(TtrainSEXP, y, n_classes, ncomp);
  }
  const arma::fmat Ttrain = float32_bits_to_fmat(TtrainSEXP, "Ttrain");
  arma::ivec labels(y.size());
  for (R_xlen_t i = 0; i < y.size(); ++i) {
    labels(static_cast<arma::uword>(i)) = y[i];
  }
  arma::ivec components(ncomp.size());
  for (R_xlen_t i = 0; i < ncomp.size(); ++i) {
    components(static_cast<arma::uword>(i)) = ncomp[i];
  }
  const std::vector<fastpls_svd::LDAFloatGPUModel> fitted =
    fastpls_svd::cuda_lda_train_prefix_float(
      Ttrain, labels, n_classes, components
    );
  Rcpp::List models(fitted.size());
  Rcpp::CharacterVector model_names(fitted.size());
  for (std::size_t i = 0; i < fitted.size(); ++i) {
    models[static_cast<R_xlen_t>(i)] = Rcpp::List::create(
      Rcpp::Named("means") = fmat_to_float32_bits(fitted[i].means),
      Rcpp::Named("linear") = fmat_to_float32_bits(fitted[i].linear),
      Rcpp::Named("constants") = fmat_to_float32_bits(fitted[i].constants),
      Rcpp::Named("priors") = fmat_to_float32_bits(fitted[i].priors.t()),
      Rcpp::Named("ridge") = fitted[i].ridge,
      Rcpp::Named("ridge_relative") = fitted[i].relative_ridge,
      Rcpp::Named("precision") = "float32",
      Rcpp::Named("backend") = "cuda_native"
    );
    model_names[static_cast<R_xlen_t>(i)] =
      std::to_string(ncomp[static_cast<R_xlen_t>(i)]);
  }
  models.attr("names") = model_names;
  return models;
}

// [[Rcpp::export]]
Rcpp::List lda_predict_float32_cuda(SEXP TtestSEXP,
                                    const Rcpp::List& lda,
                                    bool return_scores = true) {
  if (!fastpls_svd::cuda_lda_native_available()) {
    return lda_predict_float32_cpp(TtestSEXP, lda, return_scores);
  }
  const arma::fmat Ttest = float32_bits_to_fmat(TtestSEXP, "Ttest");
  const arma::fmat linear = integer_bits_to_fmat(lda["linear"], "lda$linear");
  const arma::fmat constants_matrix = integer_bits_to_fmat(
    lda["constants"], "lda$constants"
  );
  const arma::frowvec constants = arma::vectorise(constants_matrix, 1);
  const fastpls_svd::LDAFloatPrediction out =
    fastpls_svd::cuda_lda_predict_float(
      Ttest, linear, constants, return_scores
    );
  return Rcpp::List::create(
    Rcpp::Named("pred") = Rcpp::wrap(out.pred),
    Rcpp::Named("scores") = return_scores ?
      Rcpp::RObject(fmat_to_float32_bits(out.scores)) :
      Rcpp::RObject(R_NilValue)
  );
}

// [[Rcpp::export]]
Rcpp::List cuda_float32_rsvd_sample_cpp(SEXP ASEXP,
                                        int l,
                                        int power_iters,
                                        int seed) {
  arma::fmat A = float32_bits_to_fmat(ASEXP, "A");
  arma::fmat Omega;
  arma::fmat Y = rsvd_sample_float32_cuda(
    A,
    l,
    power_iters,
    static_cast<unsigned int>(seed),
    &Omega
  );
  return Rcpp::List::create(
    Rcpp::Named("Y") = fmat_to_float32_bits(Y),
    Rcpp::Named("Omega") = fmat_to_float32_bits(Omega)
  );
}

// [[Rcpp::export]]
Rcpp::List metal_float32_matrix_multiply_cpp(SEXP ASEXP,
                                             SEXP BSEXP,
                                             bool transpose_left = false,
                                             bool transpose_right = false) {
  arma::fmat A = float32_bits_to_fmat(ASEXP, "A");
  arma::fmat B = float32_bits_to_fmat(BSEXP, "B");
  arma::fmat C = fastpls_svd::metal_matrix_multiply_float(
    A,
    B,
    transpose_left,
    transpose_right
  );
  return Rcpp::List::create(Rcpp::Named("C") = fmat_to_float32_bits(C));
}

// [[Rcpp::export]]
Rcpp::List metal_float32_rsvd_sample_cpp(SEXP ASEXP,
                                         int l,
                                         int power_iters,
                                         int seed) {
  arma::fmat A = float32_bits_to_fmat(ASEXP, "A");
  arma::fmat Omega;
  arma::fmat Y = rsvd_sample_float32_metal(
    A,
    l,
    power_iters,
    static_cast<unsigned int>(seed),
    &Omega
  );
  return Rcpp::List::create(
    Rcpp::Named("Y") = fmat_to_float32_bits(Y),
    Rcpp::Named("Omega") = fmat_to_float32_bits(Omega)
  );
}

// [[Rcpp::export]]
Rcpp::List metal_float32_irlba_cpp(SEXP ASEXP,
                                   int k,
                                   int seed,
                                   bool left_only = false) {
  arma::fmat A = float32_bits_to_fmat(ASEXP, "A");
  Rcpp::List sv = irlba_float32_metal(
    A,
    k,
    0,
    static_cast<unsigned int>(seed),
    left_only
  );
  arma::fmat U = Rcpp::as<arma::fmat>(sv["u"]);
  arma::fvec d = Rcpp::as<arma::fvec>(sv["d"]);
  arma::fmat V = Rcpp::as<arma::fmat>(sv["v"]);
  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("u") = fmat_to_float32_bits(U),
    Rcpp::Named("d") = d,
    Rcpp::Named("v") = left_only ?
      Rcpp::RObject(R_NilValue) :
      Rcpp::RObject(fmat_to_float32_bits(V))
  );
  const char* audit_fields[] = {
    "case_audited",
    "case_certified",
    "deterministic_fallback",
    "audit_attempts",
    "audit_triplet_residual",
    "audit_omitted_direction_ratio"
  };
  for (const char* field : audit_fields) {
    if (sv.containsElementNamed(field)) out[field] = sv[field];
  }
  return out;
}

// [[Rcpp::export]]
Rcpp::List fastsvd_float32_cpp(SEXP ASEXP,
                               int k,
                               int backend,
                               int svd_method,
                               int rsvd_oversample,
                               int rsvd_power,
                               int seed,
                               bool left_only = false) {
  arma::fmat A = float32_bits_to_fmat(ASEXP, "x");
  Rcpp::List sv = truncated_svd_float32_backend(
    A,
    k,
    backend,
    svd_method,
    rsvd_oversample,
    rsvd_power,
    static_cast<unsigned int>(seed),
    left_only
  );

  arma::fmat U = Rcpp::as<arma::fmat>(sv["u"]);
  arma::fvec d = Rcpp::as<arma::fvec>(sv["d"]);
  arma::fmat V = Rcpp::as<arma::fmat>(sv["v"]);
  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("u") = fmat_to_float32_bits(U),
    Rcpp::Named("d") = d,
    Rcpp::Named("v") = left_only ?
      Rcpp::RObject(R_NilValue) :
      Rcpp::RObject(fmat_to_float32_bits(V))
  );
  const char* audit_fields[] = {
    "case_audited",
    "case_certified",
    "deterministic_fallback",
    "audit_attempts",
    "audit_triplet_residual",
    "audit_omitted_direction_ratio"
  };
  for (const char* field : audit_fields) {
    if (sv.containsElementNamed(field)) out[field] = sv[field];
  }
  return out;
}

// [[Rcpp::export]]
Rcpp::List pls_float32_cpu_cpp(
  SEXP XtrainSEXP,
  SEXP YtrainSEXP,
  arma::ivec ncomp,
  int scaling,
  bool fit,
  int method,
  int backend,
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  int seed
) {
  arma::fmat Xtrain = float32_bits_to_fmat(XtrainSEXP, "Xtrain");
  arma::fmat Ytrain = float32_bits_to_fmat(YtrainSEXP, "Ytrain");
  if (Xtrain.n_rows != Ytrain.n_rows) {
    Rcpp::stop("Xtrain and Ytrain must have the same number of rows");
  }
  if (ncomp.n_elem < 1) {
    Rcpp::stop("ncomp must contain at least one value");
  }
  for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
    if (ncomp(i) < 1) ncomp(i) = 1;
  }
  if (method == 1) {
    const int rank_cap = static_cast<int>(std::min(Xtrain.n_cols, Ytrain.n_cols));
    for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
      if (ncomp(i) > rank_cap) ncomp(i) = rank_cap;
    }
  }

  const int max_ncomp = arma::max(ncomp);
  const int length_ncomp = static_cast<int>(ncomp.n_elem);
  const int n = static_cast<int>(Xtrain.n_rows);
  const int p = static_cast<int>(Xtrain.n_cols);
  const int m = static_cast<int>(Ytrain.n_cols);

  arma::frowvec mX(p, arma::fill::zeros);
  if (scaling < 3) {
    mX = arma::mean(Xtrain, 0);
    Xtrain.each_row() -= mX;
  }
  arma::frowvec vX(p, arma::fill::ones);
  if (scaling == 2) {
    vX = float_col_sd(Xtrain);
    Xtrain.each_row() /= vX;
  }
  arma::frowvec mY = arma::mean(Ytrain, 0);
  Ytrain.each_row() -= mY;

  arma::fmat S = Xtrain.t() * Ytrain;
  arma::fmat Rmat(p, max_ncomp, arma::fill::zeros);
  arma::fmat Qmat(m, max_ncomp, arma::fill::zeros);
  arma::fvec R2Y(length_ncomp, arma::fill::value(NA_REAL));
  std::vector<arma::fmat> Yfit_vec(length_ncomp);
  std::vector<arma::fmat> Wlat_vec(length_ncomp);

  if (method == 1) {
    Rcpp::List sv = truncated_svd_float32_backend(
      S,
      max_ncomp,
      backend,
      svd_method,
      rsvd_oversample,
      rsvd_power,
      static_cast<unsigned int>(seed),
      false
    );
    arma::fmat U = r_object_to_fmat(sv["u"], "float32 left singular vectors");
    arma::fmat V = r_object_to_fmat(sv["v"], "float32 right singular vectors");
    arma::fvec d = r_object_to_fvec(sv["d"], "float32 singular values");
    Rmat.cols(0, U.n_cols - 1) = U;
    Qmat.cols(0, V.n_cols - 1) = V;

    arma::fmat Ttrain = Xtrain * U;
    arma::fmat G = Ttrain.t() * Ttrain;
    for (int i = 0; i < length_ncomp; ++i) {
      const int k = ncomp(i);
      arma::fmat D(k, k, arma::fill::zeros);
      for (int j = 0; j < k; ++j) D(j, j) = d(j);
      arma::fmat Ck = arma::solve(G.submat(0, 0, k - 1, k - 1), D);
      arma::fmat Wk = Ck * V.cols(0, k - 1).t();
      Wlat_vec[static_cast<std::size_t>(i)] = Wk;
      if (fit) {
        arma::fmat yf = Ttrain.cols(0, k - 1) * Wk;
        R2Y(i) = rq_float32(Ytrain, yf);
        yf.each_row() += mY;
        Yfit_vec[static_cast<std::size_t>(i)] = yf;
      }
    }
    Rcpp::RObject Yfit_obj = fit ?
      Rcpp::RObject(fmat_list_to_bits(Yfit_vec, ncomp)) :
      Rcpp::RObject(R_NilValue);

    return Rcpp::List::create(
      Rcpp::Named("R") = fmat_to_float32_bits(Rmat),
      Rcpp::Named("Q") = fmat_to_float32_bits(Qmat),
      Rcpp::Named("Ttrain") = fmat_to_float32_bits(Ttrain),
      Rcpp::Named("W_latent") = fmat_list_to_bits(Wlat_vec, ncomp),
      Rcpp::Named("mX") = fmat_to_float32_bits(arma::fmat(mX)),
      Rcpp::Named("vX") = fmat_to_float32_bits(arma::fmat(vX)),
      Rcpp::Named("mY") = fmat_to_float32_bits(arma::fmat(mY)),
      Rcpp::Named("p") = p,
      Rcpp::Named("m") = m,
      Rcpp::Named("ncomp") = ncomp,
      Rcpp::Named("Yfit") = Yfit_obj,
      Rcpp::Named("R2Y") = R2Y,
      Rcpp::Named("pls_method") = "plssvd"
    );
  }

  arma::fmat Vmat(p, max_ncomp, arma::fill::zeros);
  arma::fmat Tmat(n, max_ncomp, arma::fill::zeros);
  arma::fmat Yfit_cur;
  if (fit) {
    Yfit_cur.zeros(n, m);
  }
  int out_idx = 0;
  for (int a = 0; a < max_ncomp; ++a) {
    Rcpp::List sv = truncated_svd_float32_backend(
      S,
      1,
      backend,
      svd_method,
      rsvd_oversample,
      rsvd_power,
      static_cast<unsigned int>(seed + a),
      true
    );
    arma::fmat U = r_object_to_fmat(sv["u"], "float32 left singular vectors");
    if (U.n_cols < 1) break;
    arma::fvec rr = U.col(0);
    arma::fvec tt = Xtrain * rr;
    const float tnorm = arma::norm(tt, 2);
    if (!std::isfinite(tnorm) || tnorm <= 0.0f) break;
    tt /= tnorm;
    rr /= tnorm;
    arma::fvec pp = Xtrain.t() * tt;
    arma::fvec qq = Ytrain.t() * tt;
    arma::fvec vv = pp;
    if (a > 0) {
      arma::fmat Vprev = Vmat.cols(0, a - 1);
      vv -= Vprev * (Vprev.t() * pp);
      vv -= Vprev * (Vprev.t() * vv);
    }
    const float vnorm = arma::norm(vv, 2);
    if (!std::isfinite(vnorm) || vnorm <= 0.0f) break;
    vv /= vnorm;
    S -= vv * (vv.t() * S);
    Rmat.col(a) = rr;
    Qmat.col(a) = qq;
    Vmat.col(a) = vv;
    Tmat.col(a) = tt;
    if (fit) {
      Yfit_cur += tt * qq.t();
    }
    while (out_idx < length_ncomp && ncomp(out_idx) == a + 1) {
      if (fit) {
        R2Y(out_idx) = rq_float32(Ytrain, Yfit_cur);
        arma::fmat yf = Yfit_cur;
        yf.each_row() += mY;
        Yfit_vec[static_cast<std::size_t>(out_idx)] = yf;
      }
      ++out_idx;
    }
  }
  Rcpp::RObject Yfit_obj = fit ?
    Rcpp::RObject(fmat_list_to_bits(Yfit_vec, ncomp)) :
    Rcpp::RObject(R_NilValue);

  return Rcpp::List::create(
    Rcpp::Named("P") = R_NilValue,
    Rcpp::Named("R") = fmat_to_float32_bits(Rmat),
    Rcpp::Named("Q") = fmat_to_float32_bits(Qmat),
    Rcpp::Named("Ttrain") = fmat_to_float32_bits(Tmat),
    Rcpp::Named("mX") = fmat_to_float32_bits(arma::fmat(mX)),
    Rcpp::Named("vX") = fmat_to_float32_bits(arma::fmat(vX)),
    Rcpp::Named("mY") = fmat_to_float32_bits(arma::fmat(mY)),
    Rcpp::Named("p") = p,
    Rcpp::Named("m") = m,
    Rcpp::Named("ncomp") = ncomp,
    Rcpp::Named("Yfit") = Yfit_obj,
    Rcpp::Named("R2Y") = R2Y,
    Rcpp::Named("pls_method") = "simpls"
  );
}

namespace {

struct Float32LabelResponse {
  arma::frowvec mean;
  arma::fmat crosscov;
  arma::uvec labels;
};

Float32LabelResponse label_response_float32(
  const arma::fmat& X,
  const Rcpp::IntegerVector& labels,
  int n_classes
) {
  if (labels.size() != static_cast<R_xlen_t>(X.n_rows)) {
    Rcpp::stop("float32 label-aware PLS requires one label per training row");
  }
  if (n_classes < 2) {
    Rcpp::stop("float32 label-aware PLS requires at least two classes");
  }

  Float32LabelResponse out;
  out.labels.set_size(X.n_rows);
  arma::fvec counts(static_cast<arma::uword>(n_classes), arma::fill::zeros);
  for (arma::uword i = 0; i < X.n_rows; ++i) {
    const int cls = labels[static_cast<R_xlen_t>(i)] - 1;
    if (cls < 0 || cls >= n_classes) {
      Rcpp::stop("float32 label-aware PLS labels must be encoded as 1..n_classes");
    }
    out.labels(i) = static_cast<arma::uword>(cls);
    counts(static_cast<arma::uword>(cls)) += 1.0f;
  }
  if (arma::any(counts <= 0.0f)) {
    Rcpp::stop("float32 label-aware PLS received an empty class");
  }
  out.mean = counts.t() / static_cast<float>(X.n_rows);
  out.crosscov.zeros(X.n_cols, static_cast<arma::uword>(n_classes));

  // X is column-major, so accumulate class sums one predictor at a time.
  for (arma::uword j = 0; j < X.n_cols; ++j) {
    const float* column = X.colptr(j);
    for (arma::uword i = 0; i < X.n_rows; ++i) {
      out.crosscov(j, out.labels(i)) += column[i];
    }
  }
  const arma::fvec total = arma::sum(X, 0).t();
  out.crosscov -= total * out.mean;
  return out;
}

arma::fvec label_score_product_float32(
  const arma::fvec& score,
  const Float32LabelResponse& response
) {
  arma::fvec out(response.mean.n_elem, arma::fill::zeros);
  for (arma::uword i = 0; i < score.n_elem; ++i) {
    out(response.labels(i)) += score(i);
  }
  out -= response.mean.t() * arma::accu(score);
  return out;
}

float label_rq_float32(
  const Float32LabelResponse& response,
  const arma::fmat& fitted_centered
) {
  const double tss = static_cast<double>(fitted_centered.n_rows) *
    (1.0 - arma::dot(
      arma::conv_to<arma::rowvec>::from(response.mean),
      arma::conv_to<arma::rowvec>::from(response.mean)
    ));
  if (!std::isfinite(tss) || tss <= 0.0) {
    return NA_REAL;
  }
  double cross_term = 0.0;
  for (arma::uword i = 0; i < fitted_centered.n_rows; ++i) {
    double mean_projection = 0.0;
    for (arma::uword cls = 0; cls < fitted_centered.n_cols; ++cls) {
      mean_projection += static_cast<double>(fitted_centered(i, cls)) *
        static_cast<double>(response.mean(cls));
    }
    cross_term += static_cast<double>(fitted_centered(i, response.labels(i))) -
      mean_projection;
  }
  const double press = tss +
    static_cast<double>(arma::accu(arma::square(fitted_centered))) -
    2.0 * cross_term;
  return static_cast<float>(1.0 - press / tss);
}

} // namespace

// Fit classification PLS from compact labels without materializing one-hot Y.
// [[Rcpp::export]]
Rcpp::List pls_float32_labels_cpp(
  SEXP XtrainSEXP,
  const Rcpp::IntegerVector& labels,
  int n_classes,
  arma::ivec ncomp,
  int scaling,
  bool fit,
  int method,
  int backend,
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  int seed
) {
  arma::fmat Xtrain = float32_bits_to_fmat(XtrainSEXP, "Xtrain");
  if (ncomp.n_elem < 1) {
    Rcpp::stop("ncomp must contain at least one value");
  }
  const int rank_cap = method == 1 ?
    std::max(
      1,
      std::min(
        static_cast<int>(Xtrain.n_cols),
        n_classes - 1
      )
    ) :
    std::max(
      1,
      std::min(
        static_cast<int>(Xtrain.n_cols),
        static_cast<int>(Xtrain.n_rows) - 1
      )
    );
  for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
    const int requested = static_cast<int>(ncomp(i));
    ncomp(i) = std::max(1, std::min(requested, rank_cap));
  }

  const int max_ncomp = arma::max(ncomp);
  const int length_ncomp = static_cast<int>(ncomp.n_elem);
  const int n = static_cast<int>(Xtrain.n_rows);
  const int p = static_cast<int>(Xtrain.n_cols);
  const int m = n_classes;

  arma::frowvec mX(p, arma::fill::zeros);
  if (scaling < 3) {
    mX = arma::mean(Xtrain, 0);
    Xtrain.each_row() -= mX;
  }
  arma::frowvec vX(p, arma::fill::ones);
  if (scaling == 2) {
    vX = float_col_sd(Xtrain);
    Xtrain.each_row() /= vX;
  }
  const Float32LabelResponse response =
    label_response_float32(Xtrain, labels, n_classes);
  arma::fmat S = response.crosscov;

  arma::fmat Rmat(p, max_ncomp, arma::fill::zeros);
  arma::fmat Qmat(m, max_ncomp, arma::fill::zeros);
  arma::fvec R2Y(length_ncomp, arma::fill::value(NA_REAL));
  std::vector<arma::fmat> Yfit_vec(static_cast<std::size_t>(length_ncomp));
  std::vector<arma::fmat> Wlat_vec(static_cast<std::size_t>(length_ncomp));

  if (method == 1) {
    Rcpp::List sv = truncated_svd_float32_backend(
      S,
      max_ncomp,
      backend,
      svd_method,
      rsvd_oversample,
      rsvd_power,
      static_cast<unsigned int>(seed),
      false
    );
    arma::fmat U = r_object_to_fmat(sv["u"], "float32 left singular vectors");
    arma::fmat V = r_object_to_fmat(sv["v"], "float32 right singular vectors");
    arma::fvec d = r_object_to_fvec(sv["d"], "float32 singular values");
    const int effective = std::min(
      max_ncomp,
      static_cast<int>(std::min(U.n_cols, V.n_cols))
    );
    if (effective < max_ncomp) {
      Rcpp::stop("float32 label-aware PLS-SVD returned fewer components than requested");
    }
    Rmat.cols(0, max_ncomp - 1) = U.cols(0, max_ncomp - 1);
    Qmat.cols(0, max_ncomp - 1) = V.cols(0, max_ncomp - 1);

    arma::fmat Ttrain = Xtrain * U.cols(0, max_ncomp - 1);
    arma::fmat G = Ttrain.t() * Ttrain;
    for (int i = 0; i < length_ncomp; ++i) {
      const int k = ncomp(i);
      arma::fmat D(k, k, arma::fill::zeros);
      for (int j = 0; j < k; ++j) D(j, j) = d(j);
      arma::fmat Ck = arma::solve(
        G.submat(0, 0, k - 1, k - 1),
        D
      );
      arma::fmat Wk = Ck * V.cols(0, k - 1).t();
      Wlat_vec[static_cast<std::size_t>(i)] = Wk;
      if (fit) {
        arma::fmat yf = Ttrain.cols(0, k - 1) * Wk;
        R2Y(i) = label_rq_float32(response, yf);
        yf.each_row() += response.mean;
        Yfit_vec[static_cast<std::size_t>(i)] = std::move(yf);
      }
    }
    return Rcpp::List::create(
      Rcpp::Named("R") = fmat_to_float32_bits(Rmat),
      Rcpp::Named("Q") = fmat_to_float32_bits(Qmat),
      Rcpp::Named("Ttrain") = fmat_to_float32_bits(Ttrain),
      Rcpp::Named("W_latent") = fmat_list_to_bits(Wlat_vec, ncomp),
      Rcpp::Named("mX") = fmat_to_float32_bits(arma::fmat(mX)),
      Rcpp::Named("vX") = fmat_to_float32_bits(arma::fmat(vX)),
      Rcpp::Named("mY") = fmat_to_float32_bits(arma::fmat(response.mean)),
      Rcpp::Named("p") = p,
      Rcpp::Named("m") = m,
      Rcpp::Named("ncomp") = ncomp,
      Rcpp::Named("Yfit") = fit ?
        Rcpp::RObject(fmat_list_to_bits(Yfit_vec, ncomp)) :
        Rcpp::RObject(R_NilValue),
      Rcpp::Named("R2Y") = R2Y,
      Rcpp::Named("pls_method") = "plssvd",
      Rcpp::Named("xprod_mode") = "float32_label_class_sums"
    );
  }

  arma::fmat Vmat(p, max_ncomp, arma::fill::zeros);
  arma::fmat Tmat(n, max_ncomp, arma::fill::zeros);
  arma::fmat Yfit_cur;
  if (fit) {
    Yfit_cur.zeros(n, m);
  }
  int out_idx = 0;
  for (int a = 0; a < max_ncomp; ++a) {
    Rcpp::List sv = truncated_svd_float32_backend(
      S,
      1,
      backend,
      svd_method,
      rsvd_oversample,
      rsvd_power,
      static_cast<unsigned int>(seed + a),
      true
    );
    arma::fmat U = r_object_to_fmat(sv["u"], "float32 left singular vectors");
    if (U.n_cols < 1) break;
    arma::fvec rr = U.col(0);
    arma::fvec tt = Xtrain * rr;
    const float tnorm = arma::norm(tt, 2);
    if (!std::isfinite(tnorm) || tnorm <= 0.0f) break;
    tt /= tnorm;
    rr /= tnorm;
    arma::fvec pp = Xtrain.t() * tt;
    arma::fvec qq = label_score_product_float32(tt, response);
    arma::fvec vv = pp;
    if (a > 0) {
      const arma::fmat Vprev = Vmat.cols(0, a - 1);
      vv -= Vprev * (Vprev.t() * pp);
      vv -= Vprev * (Vprev.t() * vv);
    }
    const float vnorm = arma::norm(vv, 2);
    if (!std::isfinite(vnorm) || vnorm <= 0.0f) break;
    vv /= vnorm;
    S -= vv * (vv.t() * S);
    Rmat.col(a) = rr;
    Qmat.col(a) = qq;
    Vmat.col(a) = vv;
    Tmat.col(a) = tt;
    if (fit) {
      Yfit_cur += tt * qq.t();
    }
    while (out_idx < length_ncomp && ncomp(out_idx) == a + 1) {
      if (fit) {
        R2Y(out_idx) = label_rq_float32(response, Yfit_cur);
        arma::fmat yf = Yfit_cur;
        yf.each_row() += response.mean;
        Yfit_vec[static_cast<std::size_t>(out_idx)] = std::move(yf);
      }
      ++out_idx;
    }
  }

  return Rcpp::List::create(
    Rcpp::Named("P") = R_NilValue,
    Rcpp::Named("R") = fmat_to_float32_bits(Rmat),
    Rcpp::Named("Q") = fmat_to_float32_bits(Qmat),
    Rcpp::Named("Ttrain") = fmat_to_float32_bits(Tmat),
    Rcpp::Named("mX") = fmat_to_float32_bits(arma::fmat(mX)),
    Rcpp::Named("vX") = fmat_to_float32_bits(arma::fmat(vX)),
    Rcpp::Named("mY") = fmat_to_float32_bits(arma::fmat(response.mean)),
    Rcpp::Named("p") = p,
    Rcpp::Named("m") = m,
    Rcpp::Named("ncomp") = ncomp,
    Rcpp::Named("Yfit") = fit ?
      Rcpp::RObject(fmat_list_to_bits(Yfit_vec, ncomp)) :
      Rcpp::RObject(R_NilValue),
    Rcpp::Named("R2Y") = R2Y,
    Rcpp::Named("pls_method") = "simpls",
    Rcpp::Named("xprod_mode") = "float32_label_class_sums"
  );
}

// [[Rcpp::export]]
Rcpp::IntegerVector float32_argmax_cpp(SEXP scoresSEXP) {
  const arma::fmat scores = float32_bits_to_fmat(scoresSEXP, "scores");
  Rcpp::IntegerVector out(scores.n_rows);
  for (arma::uword i = 0; i < scores.n_rows; ++i) {
    out[static_cast<R_xlen_t>(i)] =
      static_cast<int>(scores.row(i).index_max()) + 1;
  }
  return out;
}

#else

namespace {
Rcpp::List windows_float32_unavailable() {
  Rcpp::stop("Native float32 fastPLS kernels are not available on Windows because the R Windows BLAS/LAPACK toolchain does not provide the required single-precision Fortran symbols; use standard numeric input on Windows or a Linux/macOS/CUDA build for native float32 execution.");
}

float windows_bits_to_float(const int bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(float));
  return value;
}

int windows_float_to_bits(const float value) {
  int bits;
  std::memcpy(&bits, &value, sizeof(float));
  return bits;
}

Rcpp::IntegerMatrix windows_float32_bits(SEXP xSEXP, const char* name) {
  if (!Rf_isS4(xSEXP)) {
    Rcpp::stop("%s must be a float32 matrix", name);
  }
  Rcpp::S4 x(xSEXP);
  Rcpp::IntegerMatrix bits = x.slot("Data");
  if (bits.nrow() < 1 || bits.ncol() < 1) {
    Rcpp::stop("%s must be a non-empty float32 matrix", name);
  }
  return bits;
}
}

Rcpp::List cuda_float32_rsvd_sample_cpp(SEXP ASEXP, int l, int power_iters, int seed) {
  return windows_float32_unavailable();
}

Rcpp::List metal_float32_matrix_multiply_cpp(SEXP ASEXP, SEXP BSEXP, bool transpose_left, bool transpose_right) {
  return windows_float32_unavailable();
}

Rcpp::List metal_float32_rsvd_sample_cpp(SEXP ASEXP, int l, int power_iters, int seed) {
  return windows_float32_unavailable();
}

Rcpp::List metal_float32_irlba_cpp(SEXP ASEXP, int k, int seed, bool left_only) {
  return windows_float32_unavailable();
}

Rcpp::List fastsvd_float32_cpp(SEXP ASEXP, int k, int backend, int svd_method, int rsvd_oversample, int rsvd_power, int seed, bool left_only) {
  return windows_float32_unavailable();
}

Rcpp::List pls_float32_cpu_cpp(SEXP XtrainSEXP, SEXP YtrainSEXP, arma::ivec ncomp, int scaling, bool fit, int method, int backend, int svd_method, int rsvd_oversample, int rsvd_power, int seed) {
  return windows_float32_unavailable();
}

Rcpp::List pls_float32_labels_cpp(SEXP XtrainSEXP, const Rcpp::IntegerVector& labels, int n_classes, arma::ivec ncomp, int scaling, bool fit, int method, int backend, int svd_method, int rsvd_oversample, int rsvd_power, int seed) {
  return windows_float32_unavailable();
}

Rcpp::IntegerVector float32_argmax_cpp(SEXP scoresSEXP) {
  const Rcpp::IntegerMatrix scores = windows_float32_bits(scoresSEXP, "scores");
  Rcpp::IntegerVector out(scores.nrow());
  for (int row = 0; row < scores.nrow(); ++row) {
    int best = 0;
    float best_value = windows_bits_to_float(scores(row, 0));
    for (int col = 1; col < scores.ncol(); ++col) {
      const float value = windows_bits_to_float(scores(row, col));
      if (value > best_value) {
        best = col;
        best_value = value;
      }
    }
    out[row] = best + 1;
  }
  return out;
}

Rcpp::List kernel_matrix_float32_cpp(SEXP X1SEXP, SEXP X2SEXP, int kernel, double gamma, int degree, double coef0, int backend) {
  if (backend != 0) {
    Rcpp::stop("Windows float32 kernel PLS supports backend = 'cpu' only");
  }
  const Rcpp::IntegerMatrix X1 = windows_float32_bits(X1SEXP, "X1");
  const Rcpp::IntegerMatrix X2 = windows_float32_bits(X2SEXP, "X2");
  if (X1.ncol() != X2.ncol()) {
    Rcpp::stop("X1 and X2 must have the same number of columns");
  }
  Rcpp::IntegerMatrix out(X1.nrow(), X2.nrow());
  const float gamma_f = static_cast<float>(gamma);
  const float coef0_f = static_cast<float>(coef0);
  for (int i = 0; i < X1.nrow(); ++i) {
    for (int j = 0; j < X2.nrow(); ++j) {
      float dot = 0.0f;
      float distance = 0.0f;
      for (int col = 0; col < X1.ncol(); ++col) {
        const float a = windows_bits_to_float(X1(i, col));
        const float b = windows_bits_to_float(X2(j, col));
        dot += a * b;
        const float delta = a - b;
        distance += delta * delta;
      }
      float value;
      if (kernel == 1) {
        value = dot;
      } else if (kernel == 2) {
        value = std::exp(-gamma_f * distance);
      } else if (kernel == 3) {
        value = std::pow(gamma_f * dot + coef0_f, degree);
      } else {
        Rcpp::stop("Unknown kernel id");
      }
      out(i, j) = windows_float_to_bits(value);
    }
  }
  return Rcpp::List::create(Rcpp::Named("K") = out);
}

Rcpp::List center_kernel_train_float32_cpp(SEXP KSEXP) {
  const Rcpp::IntegerMatrix K = windows_float32_bits(KSEXP, "K");
  Rcpp::IntegerMatrix out(K.nrow(), K.ncol());
  std::vector<float> row_means(static_cast<std::size_t>(K.nrow()), 0.0f);
  std::vector<float> col_means(static_cast<std::size_t>(K.ncol()), 0.0f);
  float grand_mean = 0.0f;
  for (int row = 0; row < K.nrow(); ++row) {
    for (int col = 0; col < K.ncol(); ++col) {
      const float value = windows_bits_to_float(K(row, col));
      row_means[static_cast<std::size_t>(row)] += value;
      col_means[static_cast<std::size_t>(col)] += value;
      grand_mean += value;
    }
  }
  for (float& value : row_means) value /= static_cast<float>(K.ncol());
  for (float& value : col_means) value /= static_cast<float>(K.nrow());
  grand_mean /= static_cast<float>(K.nrow() * K.ncol());
  for (int row = 0; row < K.nrow(); ++row) {
    for (int col = 0; col < K.ncol(); ++col) {
      const float value = windows_bits_to_float(K(row, col)) -
        row_means[static_cast<std::size_t>(row)] -
        col_means[static_cast<std::size_t>(col)] + grand_mean;
      out(row, col) = windows_float_to_bits(value);
    }
  }
  Rcpp::IntegerMatrix means(1, K.ncol());
  for (int col = 0; col < K.ncol(); ++col) {
    means(0, col) = windows_float_to_bits(
      col_means[static_cast<std::size_t>(col)]
    );
  }
  return Rcpp::List::create(
    Rcpp::Named("K") = out,
    Rcpp::Named("col_means") = means,
    Rcpp::Named("grand_mean") = grand_mean
  );
}

Rcpp::List center_kernel_test_float32_cpp(SEXP KtestSEXP, SEXP trainColMeansSEXP, double train_grand_mean) {
  const Rcpp::IntegerMatrix K = windows_float32_bits(KtestSEXP, "Ktest");
  const Rcpp::IntegerMatrix means = windows_float32_bits(
    trainColMeansSEXP, "train_col_means"
  );
  if (means.nrow() != 1 || means.ncol() != K.ncol()) {
    Rcpp::stop("Ktest columns must match the training kernel size");
  }
  Rcpp::IntegerMatrix out(K.nrow(), K.ncol());
  for (int row = 0; row < K.nrow(); ++row) {
    float row_mean = 0.0f;
    for (int col = 0; col < K.ncol(); ++col) {
      row_mean += windows_bits_to_float(K(row, col));
    }
    row_mean /= static_cast<float>(K.ncol());
    for (int col = 0; col < K.ncol(); ++col) {
      const float value = windows_bits_to_float(K(row, col)) - row_mean -
        windows_bits_to_float(means(0, col)) +
        static_cast<float>(train_grand_mean);
      out(row, col) = windows_float_to_bits(value);
    }
  }
  return Rcpp::List::create(Rcpp::Named("K") = out);
}

Rcpp::List opls_filter_float32_cpp(SEXP XSEXP, SEXP YSEXP, int north, int scaling, int backend, int svd_method, int rsvd_oversample, int rsvd_power, int seed) {
  return windows_float32_unavailable();
}

Rcpp::List opls_apply_filter_float32_cpp(SEXP XSEXP, SEXP mXSEXP, SEXP vXSEXP, SEXP WSEXP, SEXP PSEXP, int backend) {
  return windows_float32_unavailable();
}

Rcpp::List lda_train_prefix_float32_cpp(SEXP TtrainSEXP,
                                        const Rcpp::IntegerVector& y,
                                        int n_classes,
                                        const Rcpp::IntegerVector& ncomp) {
  return windows_float32_unavailable();
}

Rcpp::List lda_predict_float32_cpp(SEXP TtestSEXP, const Rcpp::List& lda, bool return_scores) {
  return windows_float32_unavailable();
}

Rcpp::List lda_train_prefix_float32_cuda(SEXP TtrainSEXP,
                                         const Rcpp::IntegerVector& y,
                                         int n_classes,
                                         const Rcpp::IntegerVector& ncomp) {
  return windows_float32_unavailable();
}

Rcpp::List lda_predict_float32_cuda(SEXP TtestSEXP, const Rcpp::List& lda, bool return_scores) {
  return windows_float32_unavailable();
}

#endif

// [[Rcpp::export]]
bool has_cuda() {
  return fastpls_svd::has_cuda_backend();
}

// [[Rcpp::export]]
bool lda_cuda_native_available() {
  return fastpls_svd::cuda_lda_native_available();
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

static Rcpp::IntegerVector lda_labels_from_scores(const arma::mat& scores,
                                                  const arma::rowvec& constants) {
  Rcpp::IntegerVector pred(scores.n_rows);
  for (arma::uword i = 0; i < scores.n_rows; ++i) {
    arma::uword best = 0;
    double best_val = scores(i, 0) + constants(0);
    for (arma::uword c = 1; c < scores.n_cols; ++c) {
      const double val = scores(i, c) + constants(c);
      if (val > best_val) {
        best_val = val;
        best = c;
      }
    }
    pred[i] = static_cast<int>(best) + 1;
  }
  return pred;
}

namespace {

constexpr double kLdaRelativeRidge[] = {
  1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2
};

struct LDACholeskyResult {
  arma::mat linear;
  double lambda = 0.0;
  double relative_ridge = 0.0;
};

LDACholeskyResult lda_cholesky_solve(const arma::mat& pooled,
                                     const arma::mat& means) {
  const arma::uword k = pooled.n_rows;
  double scale = arma::trace(pooled) /
    static_cast<double>(std::max<arma::uword>(1, k));
  if (!std::isfinite(scale) || scale <= 0.0) {
    scale = 1.0;
  }

  const arma::mat rhs = means.t();
  for (double rho : kLdaRelativeRidge) {
    arma::mat covariance = pooled;
    const double lambda = rho * scale;
    covariance.diag() += lambda;

    arma::mat lower;
    if (!arma::chol(lower, covariance, "lower")) {
      continue;
    }
    arma::mat intermediate;
    arma::mat solution;
    const bool forward_ok = arma::solve(
      intermediate, arma::trimatl(lower), rhs, arma::solve_opts::fast
    );
    const bool backward_ok = forward_ok && arma::solve(
      solution, arma::trimatu(lower.t()), intermediate, arma::solve_opts::fast
    );
    if (backward_ok && solution.is_finite()) {
      LDACholeskyResult out;
      out.linear = solution.t();
      out.lambda = lambda;
      out.relative_ridge = rho;
      return out;
    }
  }
  Rcpp::stop(
    "PLS-LDA Cholesky factorization failed for every deterministic regularization level"
  );
}

} // namespace

Rcpp::List lda_train_cpp(const arma::mat& Ttrain,
                         const Rcpp::IntegerVector& y,
                         int n_classes,
                         double ridge) {
  (void)ridge; // Retained in the internal ABI; regularization is deterministic.
  if (Ttrain.n_rows == 0 || Ttrain.n_cols == 0) {
    stop("lda_train_cpp requires a non-empty score matrix");
  }
  if (static_cast<R_xlen_t>(Ttrain.n_rows) != y.size()) {
    stop("lda_train_cpp requires one class label per training row");
  }
  if (n_classes < 2) {
    stop("lda_train_cpp requires at least two classes");
  }

  const arma::uword n = Ttrain.n_rows;
  const arma::uword k = Ttrain.n_cols;
  arma::vec counts(n_classes, arma::fill::zeros);
  arma::mat means(n_classes, k, arma::fill::zeros);

  for (arma::uword i = 0; i < n; ++i) {
    const int cls = y[i] - 1;
    if (cls < 0 || cls >= n_classes) {
      stop("lda_train_cpp labels must be encoded as 1..n_classes");
    }
    counts(cls) += 1.0;
    means.row(cls) += Ttrain.row(i);
  }

  for (int c = 0; c < n_classes; ++c) {
    if (counts(c) <= 0.0) {
      stop("lda_train_cpp received an empty class");
    }
    means.row(c) /= counts(c);
  }

  arma::mat pooled = arma::symmatu(Ttrain.t() * Ttrain);
  for (int c = 0; c < n_classes; ++c) {
    pooled -= counts(c) * (means.row(c).t() * means.row(c));
  }
  const double df = std::max<double>(1.0, static_cast<double>(n) - static_cast<double>(n_classes));
  pooled /= df;

  const LDACholeskyResult solved = lda_cholesky_solve(pooled, means);
  const arma::mat& linear = solved.linear;
  arma::rowvec constants(n_classes, arma::fill::zeros);
  for (int c = 0; c < n_classes; ++c) {
    const double prior = std::max(counts(c) / static_cast<double>(n), std::numeric_limits<double>::min());
    constants(c) = -0.5 * arma::as_scalar(means.row(c) * linear.row(c).t()) + std::log(prior);
  }

  return Rcpp::List::create(
    Rcpp::Named("means") = means,
    Rcpp::Named("inv_cov") = arma::mat(),
    Rcpp::Named("linear") = linear,
    Rcpp::Named("constants") = constants,
    Rcpp::Named("priors") = counts / static_cast<double>(n),
    Rcpp::Named("ridge") = solved.lambda,
    Rcpp::Named("ridge_relative") = solved.relative_ridge
  );
}

// [[Rcpp::export]]
Rcpp::List lda_train_prefix_cpp(const arma::mat& Ttrain,
                                const Rcpp::IntegerVector& y,
                                int n_classes,
                                const Rcpp::IntegerVector& ncomp,
                                double ridge) {
  (void)ridge; // Retained in the internal ABI; regularization is deterministic.
  if (Ttrain.n_rows == 0 || Ttrain.n_cols == 0) {
    stop("lda_train_prefix_cpp requires a non-empty score matrix");
  }
  if (static_cast<R_xlen_t>(Ttrain.n_rows) != y.size()) {
    stop("lda_train_prefix_cpp requires one class label per training row");
  }
  if (n_classes < 2) {
    stop("lda_train_prefix_cpp requires at least two classes");
  }
  if (ncomp.size() < 1) {
    stop("lda_train_prefix_cpp requires at least one component count");
  }

  int kmax_i = 0;
  for (R_xlen_t i = 0; i < ncomp.size(); ++i) {
    if (ncomp[i] > kmax_i) kmax_i = ncomp[i];
  }
  if (kmax_i < 1 || kmax_i > static_cast<int>(Ttrain.n_cols)) {
    stop("lda_train_prefix_cpp component counts must be in 1..ncol(Ttrain)");
  }

  const arma::uword n = Ttrain.n_rows;
  const arma::uword kmax = static_cast<arma::uword>(kmax_i);
  arma::mat Tk_storage;
  const arma::mat* Tk_ptr = &Ttrain;
  if (kmax < Ttrain.n_cols) {
    Tk_storage = Ttrain.cols(0, kmax - 1);
    Tk_ptr = &Tk_storage;
  }
  const arma::mat& Tk = *Tk_ptr;
  arma::vec counts(n_classes, arma::fill::zeros);
  arma::mat means(n_classes, kmax, arma::fill::zeros);

  for (arma::uword i = 0; i < n; ++i) {
    const int cls = y[i] - 1;
    if (cls < 0 || cls >= n_classes) {
      stop("lda_train_prefix_cpp labels must be encoded as 1..n_classes");
    }
    counts(cls) += 1.0;
    means.row(cls) += Tk.row(i);
  }
  for (int c = 0; c < n_classes; ++c) {
    if (counts(c) <= 0.0) {
      stop("lda_train_prefix_cpp received an empty class");
    }
    means.row(c) /= counts(c);
  }

  arma::mat pooled_full = arma::symmatu(Tk.t() * Tk);
  for (int c = 0; c < n_classes; ++c) {
    pooled_full -= counts(c) * (means.row(c).t() * means.row(c));
  }
  const double df = std::max<double>(1.0, static_cast<double>(n) - static_cast<double>(n_classes));
  pooled_full /= df;

  Rcpp::List models(ncomp.size());
  Rcpp::CharacterVector model_names(ncomp.size());
  for (R_xlen_t idx = 0; idx < ncomp.size(); ++idx) {
    const int kk_i = ncomp[idx];
    if (kk_i < 1 || kk_i > kmax_i) {
      stop("lda_train_prefix_cpp component counts must be in 1..max(ncomp)");
    }
    const arma::uword kk = static_cast<arma::uword>(kk_i);
    arma::mat pooled = pooled_full.submat(0, 0, kk - 1, kk - 1);
    arma::mat means_k = means.cols(0, kk - 1);

    const LDACholeskyResult solved = lda_cholesky_solve(pooled, means_k);
    const arma::mat& linear = solved.linear;
    arma::rowvec constants(n_classes, arma::fill::zeros);
    for (int c = 0; c < n_classes; ++c) {
      const double prior = std::max(counts(c) / static_cast<double>(n), std::numeric_limits<double>::min());
      constants(c) = -0.5 * arma::as_scalar(means_k.row(c) * linear.row(c).t()) + std::log(prior);
    }

    models[idx] = Rcpp::List::create(
      Rcpp::Named("means") = means_k,
      Rcpp::Named("inv_cov") = arma::mat(),
      Rcpp::Named("linear") = linear,
      Rcpp::Named("constants") = constants,
      Rcpp::Named("priors") = counts / static_cast<double>(n),
      Rcpp::Named("ridge") = solved.lambda,
      Rcpp::Named("ridge_relative") = solved.relative_ridge
    );
    model_names[idx] = std::to_string(kk_i);
  }
  models.attr("names") = model_names;
  return models;
}

// [[Rcpp::export]]
Rcpp::List lda_train_moments_prefix_cpp(const arma::mat& gram,
                                        const arma::mat& class_sums,
                                        const arma::vec& counts,
                                        int n,
                                        const Rcpp::IntegerVector& ncomp) {
  if (n < 1 || gram.n_rows < 1 || gram.n_rows != gram.n_cols) {
    stop("lda_train_moments_prefix_cpp requires a square, non-empty score Gram matrix");
  }
  if (class_sums.n_rows < 2 || class_sums.n_cols != gram.n_cols ||
      counts.n_elem != class_sums.n_rows) {
    stop("lda_train_moments_prefix_cpp received inconsistent class moments");
  }
  if (ncomp.size() < 1) {
    stop("lda_train_moments_prefix_cpp requires at least one component count");
  }
  if (!gram.is_finite() || !class_sums.is_finite() || !counts.is_finite()) {
    stop("lda_train_moments_prefix_cpp requires finite moments");
  }

  const arma::uword n_classes = class_sums.n_rows;
  const arma::uword kmax = gram.n_cols;
  arma::mat means = class_sums;
  double total_count = 0.0;
  for (arma::uword cls = 0; cls < n_classes; ++cls) {
    if (counts(cls) <= 0.0) {
      stop("lda_train_moments_prefix_cpp received an empty class");
    }
    means.row(cls) /= counts(cls);
    total_count += counts(cls);
  }
  if (std::abs(total_count - static_cast<double>(n)) >
      1e-8 * std::max(1.0, static_cast<double>(n))) {
    stop("lda_train_moments_prefix_cpp class counts do not sum to n");
  }

  arma::mat pooled_full = arma::symmatu(gram);
  for (arma::uword cls = 0; cls < n_classes; ++cls) {
    pooled_full -= counts(cls) *
      (means.row(cls).t() * means.row(cls));
  }
  pooled_full /= std::max<double>(
    1.0, static_cast<double>(n) - static_cast<double>(n_classes)
  );

  Rcpp::List models(ncomp.size());
  Rcpp::CharacterVector model_names(ncomp.size());
  for (R_xlen_t index = 0; index < ncomp.size(); ++index) {
    const int kk_i = ncomp[index];
    if (kk_i < 1 || kk_i > static_cast<int>(kmax)) {
      stop("lda_train_moments_prefix_cpp component counts must be in 1..ncol(gram)");
    }
    const arma::uword kk = static_cast<arma::uword>(kk_i);
    const arma::mat pooled = pooled_full.submat(0, 0, kk - 1, kk - 1);
    const arma::mat means_k = means.cols(0, kk - 1);
    const LDACholeskyResult solved = lda_cholesky_solve(pooled, means_k);
    arma::rowvec constants(n_classes, arma::fill::zeros);
    for (arma::uword cls = 0; cls < n_classes; ++cls) {
      const double prior = std::max(
        counts(cls) / static_cast<double>(n),
        std::numeric_limits<double>::min()
      );
      constants(cls) = -0.5 * arma::dot(
        means_k.row(cls), solved.linear.row(cls)
      ) + std::log(prior);
    }
    models[index] = Rcpp::List::create(
      Rcpp::Named("means") = means_k,
      Rcpp::Named("inv_cov") = arma::mat(),
      Rcpp::Named("linear") = solved.linear,
      Rcpp::Named("constants") = constants,
      Rcpp::Named("priors") = counts / static_cast<double>(n),
      Rcpp::Named("ridge") = solved.lambda,
      Rcpp::Named("ridge_relative") = solved.relative_ridge
    );
    model_names[index] = std::to_string(kk_i);
  }
  models.attr("names") = model_names;
  return models;
}

// [[Rcpp::export]]
Rcpp::List lda_project_train_prefix_cpp(const arma::mat& Xtrain,
                                        const arma::mat& R,
                                        const arma::rowvec& offset,
                                        const Rcpp::IntegerVector& y,
                                        int n_classes,
                                        const Rcpp::IntegerVector& ncomp,
                                        double ridge) {
  if (Xtrain.n_rows == 0 || Xtrain.n_cols == 0) {
    stop("lda_project_train_prefix_cpp requires a non-empty predictor matrix");
  }
  if (R.n_rows != Xtrain.n_cols || R.n_cols == 0) {
    stop("lda_project_train_prefix_cpp projection matrix has incompatible dimensions");
  }
  if (offset.n_elem > 0 && offset.n_elem < R.n_cols) {
    stop("lda_project_train_prefix_cpp offset is shorter than the projection dimension");
  }
  arma::mat Ttrain = Xtrain * R;
  if (offset.n_elem >= R.n_cols) {
    Ttrain.each_row() -= offset.subvec(0, R.n_cols - 1);
  }
  Rcpp::List models = lda_train_prefix_cpp(Ttrain, y, n_classes, ncomp, ridge);
  for (R_xlen_t i = 0; i < models.size(); ++i) {
    Rcpp::List model = models[i];
    model["backend"] = "cpp_project";
    models[i] = model;
  }
  return models;
}

// [[Rcpp::export]]
Rcpp::List lda_train_prefix_cuda(const arma::mat& Ttrain,
                                 const Rcpp::IntegerVector& y,
                                 int n_classes,
                                 const Rcpp::IntegerVector& ncomp,
                                 double ridge) {
  if (!fastpls_svd::cuda_lda_native_available()) {
    return lda_train_prefix_cpp(Ttrain, y, n_classes, ncomp, ridge);
  }
  arma::ivec y_arma(y.size());
  for (R_xlen_t i = 0; i < y.size(); ++i) {
    y_arma(static_cast<arma::uword>(i)) = y[i];
  }
  arma::ivec ncomp_arma(ncomp.size());
  for (R_xlen_t i = 0; i < ncomp.size(); ++i) {
    ncomp_arma(static_cast<arma::uword>(i)) = ncomp[i];
  }
  std::vector<fastpls_svd::LDAGPUModel> gpu_models =
    fastpls_svd::cuda_lda_train_prefix(Ttrain, y_arma, n_classes, ncomp_arma, ridge);

  Rcpp::List models(ncomp.size());
  Rcpp::CharacterVector model_names(ncomp.size());
  for (R_xlen_t idx = 0; idx < ncomp.size(); ++idx) {
    models[idx] = Rcpp::List::create(
      Rcpp::Named("means") = gpu_models[static_cast<size_t>(idx)].means,
      Rcpp::Named("inv_cov") = arma::mat(),
      Rcpp::Named("linear") = gpu_models[static_cast<size_t>(idx)].linear,
      Rcpp::Named("constants") = gpu_models[static_cast<size_t>(idx)].constants,
      Rcpp::Named("priors") = gpu_models[static_cast<size_t>(idx)].priors,
      Rcpp::Named("ridge") = gpu_models[static_cast<size_t>(idx)].ridge,
      Rcpp::Named("ridge_relative") = gpu_models[static_cast<size_t>(idx)].relative_ridge,
      Rcpp::Named("backend") = "cuda_native"
    );
    model_names[idx] = std::to_string(ncomp[idx]);
  }
  models.attr("names") = model_names;
  return models;
}

// [[Rcpp::export]]
Rcpp::List lda_project_train_prefix_cuda(const arma::mat& Xtrain,
                                         const arma::mat& R,
                                         const arma::rowvec& offset,
                                         const Rcpp::IntegerVector& y,
                                         int n_classes,
                                         const Rcpp::IntegerVector& ncomp,
                                         double ridge) {
  if (!fastpls_svd::cuda_lda_native_available()) {
    arma::mat Ttrain = Xtrain * R;
    if (offset.n_elem >= R.n_cols) {
      Ttrain.each_row() -= offset.subvec(0, R.n_cols - 1);
    }
    return lda_train_prefix_cpp(Ttrain, y, n_classes, ncomp, ridge);
  }
  arma::ivec y_arma(y.size());
  for (R_xlen_t i = 0; i < y.size(); ++i) {
    y_arma(static_cast<arma::uword>(i)) = y[i];
  }
  arma::ivec ncomp_arma(ncomp.size());
  for (R_xlen_t i = 0; i < ncomp.size(); ++i) {
    ncomp_arma(static_cast<arma::uword>(i)) = ncomp[i];
  }
  std::vector<fastpls_svd::LDAGPUModel> gpu_models =
    fastpls_svd::cuda_lda_project_train_prefix(Xtrain, R, offset, y_arma, n_classes, ncomp_arma, ridge);

  Rcpp::List models(ncomp.size());
  Rcpp::CharacterVector model_names(ncomp.size());
  for (R_xlen_t idx = 0; idx < ncomp.size(); ++idx) {
    models[idx] = Rcpp::List::create(
      Rcpp::Named("means") = gpu_models[static_cast<size_t>(idx)].means,
      Rcpp::Named("inv_cov") = arma::mat(),
      Rcpp::Named("linear") = gpu_models[static_cast<size_t>(idx)].linear,
      Rcpp::Named("constants") = gpu_models[static_cast<size_t>(idx)].constants,
      Rcpp::Named("priors") = gpu_models[static_cast<size_t>(idx)].priors,
      Rcpp::Named("ridge") = gpu_models[static_cast<size_t>(idx)].ridge,
      Rcpp::Named("ridge_relative") = gpu_models[static_cast<size_t>(idx)].relative_ridge,
      Rcpp::Named("backend") = "cuda_native_project"
    );
    model_names[idx] = std::to_string(ncomp[idx]);
  }
  models.attr("names") = model_names;
  return models;
}

// [[Rcpp::export]]
Rcpp::List lda_predict_cpp(const arma::mat& Ttest,
                           const Rcpp::List& lda) {
  if (Ttest.n_rows == 0 || Ttest.n_cols == 0) {
    stop("lda_predict_cpp requires a non-empty score matrix");
  }
  arma::mat linear = Rcpp::as<arma::mat>(lda["linear"]);
  arma::rowvec constants = Rcpp::as<arma::rowvec>(lda["constants"]);
  if (Ttest.n_cols != linear.n_cols) {
    stop("lda_predict_cpp score dimension does not match the LDA model");
  }
  if (constants.n_elem != linear.n_rows) {
    stop("lda_predict_cpp has inconsistent LDA constants");
  }

  arma::mat scores = Ttest * linear.t();
  scores.each_row() += constants;

  Rcpp::IntegerVector pred(scores.n_rows);
  for (arma::uword i = 0; i < scores.n_rows; ++i) {
    arma::uword best = 0;
    double best_val = scores(i, 0);
    for (arma::uword c = 1; c < scores.n_cols; ++c) {
      if (scores(i, c) > best_val) {
        best_val = scores(i, c);
        best = c;
      }
    }
    pred[i] = static_cast<int>(best) + 1;
  }

  return Rcpp::List::create(
    Rcpp::Named("pred") = pred,
    Rcpp::Named("scores") = scores
  );
}

// [[Rcpp::export]]
Rcpp::IntegerVector lda_predict_labels_cpp(const arma::mat& Ttest,
                                           const Rcpp::List& lda) {
  if (Ttest.n_rows == 0 || Ttest.n_cols == 0) {
    stop("lda_predict_labels_cpp requires a non-empty score matrix");
  }
  arma::mat linear = Rcpp::as<arma::mat>(lda["linear"]);
  arma::rowvec constants = Rcpp::as<arma::rowvec>(lda["constants"]);
  if (Ttest.n_cols != linear.n_cols) {
    stop("lda_predict_labels_cpp score dimension does not match the LDA model");
  }
  if (constants.n_elem != linear.n_rows) {
    stop("lda_predict_labels_cpp has inconsistent LDA constants");
  }

  arma::mat scores = Ttest * linear.t();
  Rcpp::IntegerVector pred = lda_labels_from_scores(scores, constants);

  return pred;
}

// [[Rcpp::export]]
Rcpp::IntegerVector lda_project_predict_labels_cpp(const arma::mat& Xtest,
                                                   const arma::mat& R,
                                                   const arma::rowvec& offset,
                                                   const Rcpp::List& lda) {
  if (Xtest.n_rows == 0 || Xtest.n_cols == 0) {
    stop("lda_project_predict_labels_cpp requires a non-empty predictor matrix");
  }
  if (R.n_rows != Xtest.n_cols || R.n_cols == 0) {
    stop("lda_project_predict_labels_cpp projection matrix has incompatible dimensions");
  }
  if (offset.n_elem > 0 && offset.n_elem < R.n_cols) {
    stop("lda_project_predict_labels_cpp offset is shorter than the projection dimension");
  }

  arma::mat linear = Rcpp::as<arma::mat>(lda["linear"]);
  arma::rowvec constants = Rcpp::as<arma::rowvec>(lda["constants"]);
  if (R.n_cols != linear.n_cols) {
    stop("lda_project_predict_labels_cpp projection dimension does not match the LDA model");
  }
  if (constants.n_elem != linear.n_rows) {
    stop("lda_project_predict_labels_cpp has inconsistent LDA constants");
  }

  const double n = static_cast<double>(Xtest.n_rows);
  const double p = static_cast<double>(Xtest.n_cols);
  const double k = static_cast<double>(R.n_cols);
  const double n_classes = static_cast<double>(linear.n_rows);
  const double latent_ops = n * k * (p + n_classes);
  const double direct_ops = n * p * n_classes;

  if (std::isfinite(latent_ops) && std::isfinite(direct_ops) &&
      direct_ops < 0.5 * latent_ops) {
    arma::mat W = R * linear.t();
    arma::rowvec constants_adj = constants;
    if (offset.n_elem >= R.n_cols) {
      constants_adj -= offset.subvec(0, R.n_cols - 1) * linear.t();
    }
    arma::mat scores = Xtest * W;
    return lda_labels_from_scores(scores, constants_adj);
  }

  arma::mat Ttest = Xtest * R;
  if (offset.n_elem >= R.n_cols) {
    Ttest.each_row() -= offset.subvec(0, R.n_cols - 1);
  }
  arma::mat scores = Ttest * linear.t();
  return lda_labels_from_scores(scores, constants);
}

// [[Rcpp::export]]
Rcpp::List lda_predict_cuda(const arma::mat& Ttest,
                            const Rcpp::List& lda) {
  if (!fastpls_svd::cuda_lda_native_available()) {
    return lda_predict_cpp(Ttest, lda);
  }
  if (Ttest.n_rows == 0 || Ttest.n_cols == 0) {
    stop("lda_predict_cuda requires a non-empty score matrix");
  }
  arma::mat linear = Rcpp::as<arma::mat>(lda["linear"]);
  arma::rowvec constants = Rcpp::as<arma::rowvec>(lda["constants"]);
  if (Ttest.n_cols != linear.n_cols) {
    stop("lda_predict_cuda score dimension does not match the LDA model");
  }
  if (constants.n_elem != linear.n_rows) {
    stop("lda_predict_cuda has inconsistent LDA constants");
  }
  return fastpls_svd::cuda_lda_predict(Ttest, linear, constants, true);
}

// [[Rcpp::export]]
Rcpp::IntegerVector lda_predict_labels_cuda(const arma::mat& Ttest,
                                            const Rcpp::List& lda) {
  if (!fastpls_svd::cuda_lda_native_available()) {
    return lda_predict_labels_cpp(Ttest, lda);
  }
  if (Ttest.n_rows == 0 || Ttest.n_cols == 0) {
    stop("lda_predict_labels_cuda requires a non-empty score matrix");
  }
  arma::mat linear = Rcpp::as<arma::mat>(lda["linear"]);
  arma::rowvec constants = Rcpp::as<arma::rowvec>(lda["constants"]);
  if (Ttest.n_cols != linear.n_cols) {
    stop("lda_predict_labels_cuda score dimension does not match the LDA model");
  }
  if (constants.n_elem != linear.n_rows) {
    stop("lda_predict_labels_cuda has inconsistent LDA constants");
  }
  Rcpp::List pred = fastpls_svd::cuda_lda_predict(Ttest, linear, constants, false);
  return pred["pred"];
}

// [[Rcpp::export]]
Rcpp::List lda_project_predict_cuda(const arma::mat& Xtest,
                                    const arma::mat& R,
                                    const arma::rowvec& offset,
                                    const Rcpp::List& lda,
                                    bool return_scores = false) {
  if (!fastpls_svd::cuda_lda_native_available()) {
    arma::mat linear = Rcpp::as<arma::mat>(lda["linear"]);
    arma::rowvec constants = Rcpp::as<arma::rowvec>(lda["constants"]);
    if (offset.n_elem >= R.n_cols) {
      constants -= offset.subvec(0, R.n_cols - 1) * linear.t();
    }
    arma::mat scores = (Xtest * R) * linear.t();
    scores.each_row() += constants;
    Rcpp::IntegerVector pred(scores.n_rows);
    for (arma::uword i = 0; i < scores.n_rows; ++i) {
      arma::uword best = 0;
      double best_val = scores(i, 0);
      for (arma::uword c = 1; c < scores.n_cols; ++c) {
        if (scores(i, c) > best_val) {
          best_val = scores(i, c);
          best = c;
        }
      }
      pred[i] = static_cast<int>(best) + 1;
    }
    if (return_scores) {
      return Rcpp::List::create(Rcpp::Named("pred") = pred, Rcpp::Named("scores") = scores);
    }
    return Rcpp::List::create(Rcpp::Named("pred") = pred);
  }
  if (Xtest.n_rows == 0 || Xtest.n_cols == 0 || R.n_rows == 0 || R.n_cols == 0) {
    stop("lda_project_predict_cuda requires non-empty X and projection matrices");
  }
  arma::mat linear = Rcpp::as<arma::mat>(lda["linear"]);
  arma::rowvec constants = Rcpp::as<arma::rowvec>(lda["constants"]);
  return fastpls_svd::cuda_lda_project_predict(Xtest, R, offset, linear, constants, return_scores);
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
    Rcpp::Named("vt") = res.Vt,
    Rcpp::Named("randomized") = res.randomized,
    Rcpp::Named("case_audited") = res.case_audited,
    Rcpp::Named("case_certified") = res.case_certified,
    Rcpp::Named("deterministic_fallback") = res.deterministic_fallback,
    Rcpp::Named("audit_attempts") = res.audit_attempts,
    Rcpp::Named("effective_oversample") = res.effective_oversample,
    Rcpp::Named("effective_power") = res.effective_power_iters,
    Rcpp::Named("effective_seed") = res.effective_seed,
    Rcpp::Named("audit_subspace_error") = res.audit_subspace_error,
    Rcpp::Named("audit_singular_value_error") = res.audit_singular_value_error,
    Rcpp::Named("audit_triplet_residual") = res.audit_triplet_residual,
    Rcpp::Named("audit_omitted_direction_ratio") = res.audit_omitted_direction_ratio
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
      static_cast<unsigned int>(seed + a)
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
  using BenchClock = std::chrono::steady_clock;
  const bool benchmark_phase_timing =
    env_int_or("FASTPLS_BENCH_PHASE_TIMING", 0, 0, 1) == 1;
  const auto function_started = benchmark_phase_timing ?
    BenchClock::now() : BenchClock::time_point();
  double estimator_sec = 0.0;
  double coefficient_sec = 0.0;
  double fitted_sec = 0.0;
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

  // rSVD refreshes a small candidate block and consumes its directions through
  // sequential SIMPLS orthogonalization and deflation. IRLBA remains the
  // conventional component-wise route.
  const int center_t = env_int_or("FASTPLS_FAST_CENTER_T", 0, 0, 1);
  const int reorth_v = env_int_or("FASTPLS_FAST_REORTH_V", 0, 0, 1);
  const int defl_cache = env_int_or("FASTPLS_FAST_DEFLCACHE", 1, 0, 1);
  const int fast_optimized = env_int_or("FASTPLS_FAST_OPTIMIZED", 1, 0, 1);
  const int incremental_coefficients = env_int_or("FASTPLS_INCREMENTAL_COEFFICIENTS", 1, 0, 1);
  const int fast_crossprod_min_ncomp = env_int_or("FASTPLS_FAST_CROSSPROD_MIN_NCOMP", 20, 1, 1024);
  const int fast_crossprod_max_p = env_int_or("FASTPLS_FAST_CROSSPROD_MAX_P", 512, 16, 65536);
  const int fast_crossprod_min_n_to_p_ratio = env_int_or("FASTPLS_FAST_CROSSPROD_MIN_N_TO_P_RATIO", 8, 1, 1024);
  const bool return_ttrain = env_int_or("FASTPLS_RETURN_TTRAIN", 0, 0, 1) == 1;
  arma::mat TT;
  if (return_ttrain) {
    TT.zeros(n, max_ncomp);
  }
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
  const auto estimator_started = benchmark_phase_timing ?
    BenchClock::now() : BenchClock::time_point();
  arma::vec previous_direction;
  bool has_previous_direction = false;
  auto append_component = [&](arma::vec rr, const int a_idx) -> bool {
    const auto component_started = benchmark_phase_timing ?
      BenchClock::now() : BenchClock::time_point();
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
      if (fit || return_ttrain) {
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

    if (defl_cache == 1) {
      arma::rowvec vS = vv.t() * S;
      S -= vv * vS;
    } else {
      S -= vv * (vv.t() * S);
    }

    RR.col(a_idx) = rr;
    QQ.col(a_idx) = qq;
    VV.col(a_idx) = vv;
    previous_direction = rr;
    has_previous_direction = true;
    if (return_ttrain && tt.n_elem == static_cast<arma::uword>(n)) {
      TT.col(a_idx) = tt;
    }
    if (benchmark_phase_timing) {
      estimator_sec += std::chrono::duration<double>(
        BenchClock::now() - component_started
      ).count();
    }
    if (store_B && incremental_coefficients == 1) {
      const auto coefficient_started = benchmark_phase_timing ?
        BenchClock::now() : BenchClock::time_point();
      Bcur += rr * qq.t();
      if (benchmark_phase_timing) {
        coefficient_sec += std::chrono::duration<double>(
          BenchClock::now() - coefficient_started
        ).count();
      }
    }
    if (fit) {
      const auto fitted_started = benchmark_phase_timing ?
        BenchClock::now() : BenchClock::time_point();
      Yfit_cur += tt * qq.t();
      if (benchmark_phase_timing) {
        fitted_sec += std::chrono::duration<double>(
          BenchClock::now() - fitted_started
        ).count();
      }
    }

    while (i_out < length_ncomp && a_idx == (ncomp(i_out) - 1)) {
      if (store_B) {
        const auto coefficient_started = benchmark_phase_timing ?
          BenchClock::now() : BenchClock::time_point();
        B.slice(i_out) = incremental_coefficients == 1 ?
          Bcur :
          RR.cols(0, a_idx) * QQ.cols(0, a_idx).t();
        if (benchmark_phase_timing) {
          coefficient_sec += std::chrono::duration<double>(
            BenchClock::now() - coefficient_started
          ).count();
        }
      }
      if (fit) {
        const auto fitted_started = benchmark_phase_timing ?
          BenchClock::now() : BenchClock::time_point();
        R2Y(i_out) = RQ(Ytrain, Yfit_cur);
        arma::mat yf = Yfit_cur;
        yf.each_row() += mY;
        Yfit.slice(i_out) = yf;
        if (benchmark_phase_timing) {
          fitted_sec += std::chrono::duration<double>(
            BenchClock::now() - fitted_started
          ).count();
        }
      }
      ++i_out;
    }
    return true;
  };

  int a = 0;
  while (a < max_ncomp) {
    const auto direction_started = benchmark_phase_timing ?
      BenchClock::now() : BenchClock::time_point();
    const bool randomized =
      svd_method == fastpls_svd::SVD_METHOD_CPU_RSVD ||
      svd_method == fastpls_svd::SVD_METHOD_CUDA_RSVD;
    const int k_block = randomized ?
      accelerated_simpls_block_size(
        max_ncomp - a, p, m, false
      ) : 1;
    arma::mat Ublock;
    if (randomized) {
      SimplsFastRefreshWorkspace refresh_ws;
      if (!refresh_ws.refresh(
            S,
            has_previous_direction ? &previous_direction : nullptr,
            k_block,
            std::max(rsvd_power, 0),
            static_cast<unsigned int>(seed + a),
            Ublock
          )) {
        break;
      }
    } else {
      fastpls_svd::SVDResult svd_res = compute_truncated_svd_dispatch(
        S,
        1,
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
    if (benchmark_phase_timing) {
      estimator_sec += std::chrono::duration<double>(
        BenchClock::now() - direction_started
      ).count();
    }
    if (Ublock.n_cols < 1) {
      break;
    }
    const int use_cols = std::min<int>(Ublock.n_cols, k_block);
    bool stop_now = false;
    for (int j = 0; j < use_cols && a < max_ncomp; ++j, ++a) {
      if (!append_component(Ublock.col(j), a)) {
        stop_now = true;
        break;
      }
    }
    if (stop_now) break;
  }

  const auto assembly_started = benchmark_phase_timing ?
    BenchClock::now() : BenchClock::time_point();
  List out = List::create(
    Named("P")       = arma::mat(),
    Named("Q")       = QQ,
    Named("Ttrain")  = return_ttrain ? TT : arma::mat(),
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
  const double assembly_sec = benchmark_phase_timing ?
    std::chrono::duration<double>(BenchClock::now() - assembly_started).count() :
    0.0;
  if (benchmark_phase_timing) {
    const double total_cpp_sec = std::chrono::duration<double>(
      BenchClock::now() - function_started
    ).count();
    const double preprocess_sec = std::max(
      0.0,
      std::chrono::duration<double>(estimator_started - function_started).count()
    );
    out["benchmark_phase_timing"] = List::create(
      Named("preprocess_crosscov_sec") = preprocess_sec,
      Named("estimator_sec") = estimator_sec,
      Named("coefficient_path_sec") = coefficient_sec,
      Named("fitted_values_sec") = fitted_sec,
      Named("model_assembly_sec") = assembly_sec,
      Named("cpp_total_sec") = total_cpp_sec
    );
  }
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
  if (!fastpls_svd::has_cuda_backend()) {
    stop("pls_model2_fast_gpu requires CUDA available");
  }
  if (svd_method != fastpls_svd::SVD_METHOD_CUDA_RSVD) {
    stop("pls_model2_fast_gpu requires svd.method='cuda_rsvd'");
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
  const bool classification_response =
    n >= 5000 && max_ncomp >= 50 && is_one_hot_response(Ytrain);

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

  const int center_t = env_int_or("FASTPLS_FAST_CENTER_T", 0, 0, 1);
  const int reorth_v = env_int_or("FASTPLS_FAST_REORTH_V", 0, 0, 1);
  const int defl_cache = env_int_or("FASTPLS_FAST_DEFLCACHE", 1, 0, 1);
  (void)defl_cache;
  const int sketch_dim = std::min(
    std::min(p, m),
    1 + std::max(rsvd_oversample, 0)
  );
  const int requested_power_iters = std::max(rsvd_power, 0);
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
    int a = 0;
    while (a < max_ncomp) {
      const int k_block = accelerated_simpls_block_size(
        max_ncomp - a, p, m, classification_response
      );
      arma::vec shat_block(k_block, arma::fill::zeros);
      if (use_implicit_xprod) {
        fastpls_svd::cuda_simpls_fast_refresh_block_implicit_resident(
          n,
          p,
          m,
          sketch_dim,
          k_block,
          a,
          false,
          static_cast<unsigned int>(seed + a),
          requested_power_iters,
          shat_block.memptr()
        );
      } else {
        fastpls_svd::cuda_simpls_fast_refresh_block_resident(
          p,
          m,
          sketch_dim,
          k_block,
          false,
          static_cast<unsigned int>(seed + a),
          requested_power_iters,
          shat_block.memptr()
        );
      }

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
          // A randomized direction can occasionally land in a direction
          // removed by SIMPLS deflation. Retry with a fresh independent sketch
          // instead of terminating the coefficient path.
          const int max_gpu_refresh_retries = 8;
          for (int retry = 0; retry < max_gpu_refresh_retries && !appended; ++retry) {
            arma::vec retry_shat(1, arma::fill::zeros);
            const unsigned int retry_seed =
              static_cast<unsigned int>(seed + a + 7919 * (retry + 1));
            if (use_implicit_xprod) {
              fastpls_svd::cuda_simpls_fast_refresh_block_implicit_resident(
                n,
                p,
                m,
                sketch_dim,
                1,
                a,
                false,
                retry_seed,
                requested_power_iters,
                retry_shat.memptr()
              );
            } else {
              fastpls_svd::cuda_simpls_fast_refresh_block_resident(
                p,
                m,
                sketch_dim,
                1,
                false,
                retry_seed,
                requested_power_iters,
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
    arma::vec previous_direction;
    bool has_previous_direction = false;
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
      previous_direction = rr;
      has_previous_direction = true;
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
      const int k_block = accelerated_simpls_block_size(
        max_ncomp - a, p, m, classification_response
      );
      arma::mat Ublock;
      if (use_implicit_xprod) {
        arma::vec shat_block;
        if (!refresh_deflated_crossprod_left_double(
              Xtrain,
              Ytrain,
              VV,
              a,
              has_previous_direction ? &previous_direction : nullptr,
              k_block,
              requested_power_iters,
              static_cast<unsigned int>(seed + a),
              Ublock,
              shat_block
            )) {
          break;
        }
      } else {
        if (!refresh_ws.refresh(
              S_shape,
              has_previous_direction ? &previous_direction : nullptr,
              k_block,
              requested_power_iters,
              static_cast<unsigned int>(seed + a),
              Ublock
            )) {
          break;
        }
      }
      if (Ublock.n_cols < 1) {
        break;
      }

      const int use_cols = std::min<int>(Ublock.n_cols, k_block);
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
List pls_model2_fast_gpu_labels(
  SEXP XtrainSEXP,
  Rcpp::IntegerVector y,
  int n_classes,
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
    stop("pls_model2_fast_gpu_labels requires svd.method='cuda_rsvd' with CUDA available");
  }
  const arma::mat Xview = numeric_matrix_view(XtrainSEXP, "Xtrain");
  const int n = static_cast<int>(Xview.n_rows);
  if (y.size() != n) {
    stop("pls_model2_fast_gpu_labels requires one label per training row");
  }
  if (n_classes < 2) {
    stop("pls_model2_fast_gpu_labels requires at least two classes");
  }
  arma::mat Ytrain(
    static_cast<arma::uword>(n),
    static_cast<arma::uword>(n_classes),
    arma::fill::zeros
  );
  for (int i = 0; i < n; ++i) {
    const int cls = y[i];
    if (IntegerVector::is_na(cls) || cls < 1 || cls > n_classes) {
      stop("pls_model2_fast_gpu_labels requires labels encoded as 1..n_classes");
    }
    Ytrain(static_cast<arma::uword>(i), static_cast<arma::uword>(cls - 1)) = 1.0;
  }
  return pls_model2_fast_gpu(
    Xview,
    std::move(Ytrain),
    ncomp,
    scaling,
    fit,
    svd_method,
    rsvd_oversample,
    rsvd_power,
    svds_tol,
    seed
  );
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

arma::cube compact_prediction_weights(List& model, const int m, const arma::ivec& ncomp) {
  std::string pls_method;
  if (model.containsElementNamed("pls_method")) {
    pls_method = Rcpp::as<std::string>(model["pls_method"]);
  }

  Rcpp::NumericVector R_vec = model["R"];
  Rcpp::IntegerVector R_dim = R_vec.attr("dim");
  if (R_dim.size() != 2L || R_dim[1] < 1) {
    Rcpp::stop("Model `R` is not compatible with compact class prediction");
  }
  const int kmax = R_dim[1];
  const arma::uword length_ncomp = static_cast<arma::uword>(ncomp.n_elem);
  arma::cube Wflash(static_cast<arma::uword>(kmax), static_cast<arma::uword>(m), length_ncomp, arma::fill::zeros);

  if ((pls_method == "simpls" || pls_method == "simpls_fast") &&
      model.containsElementNamed("Q")) {
    Rcpp::NumericVector Q_vec = model["Q"];
    Rcpp::IntegerVector Q_dim = Q_vec.attr("dim");
    if (Q_dim.size() != 2L || Q_dim[0] != m || Q_dim[1] < 1) {
      Rcpp::stop("Model `Q` is not compatible with compact class prediction");
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
        Rcpp::stop("ncomp exceeds latent rank for compact class prediction");
      }
      Wflash.slice(a).rows(0, static_cast<arma::uword>(mc - 1)) =
        QQ.cols(0, static_cast<arma::uword>(mc - 1)).t();
    }
    return Wflash;
  }

  if (pls_method == "plssvd" && model.containsElementNamed("W_latent")) {
    Rcpp::NumericVector W_vec = model["W_latent"];
    Rcpp::IntegerVector W_dim = W_vec.attr("dim");
    if (W_dim.size() != 3L || W_dim[0] != kmax || W_dim[1] != m ||
        W_dim[2] < static_cast<int>(length_ncomp)) {
      Rcpp::stop("Model `W_latent` is not compatible with compact class prediction");
    }
    const arma::cube WW(
      W_vec.begin(),
      static_cast<arma::uword>(W_dim[0]),
      static_cast<arma::uword>(W_dim[1]),
      static_cast<arma::uword>(W_dim[2]),
      false,
      true
    );
    return WW.slices(0, length_ncomp - 1);
  }

  if (pls_method == "plssvd" &&
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
      Rcpp::stop("Model latent PLSSVD factors are not compatible with compact class prediction");
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
        Rcpp::stop("ncomp exceeds latent rank for compact class prediction");
      }
      arma::mat Cmc = CC.slice(a).submat(0, 0, mc - 1, mc - 1);
      Wflash.slice(a).rows(0, static_cast<arma::uword>(mc - 1)) =
        Cmc * QQ.cols(0, static_cast<arma::uword>(mc - 1)).t();
    }
    return Wflash;
  }

  Rcpp::stop("Compact class prediction requires compact low-rank factors");
}

arma::mat class_prediction_offsets(List& model, const int m, const arma::uword length_ncomp) {
  Rcpp::NumericVector mY_vec = model["mY"];
  arma::rowvec mY(mY_vec.begin(), mY_vec.size(), false, true);
  arma::mat offsets(static_cast<arma::uword>(m), length_ncomp, arma::fill::zeros);
  for (arma::uword a = 0; a < length_ncomp; ++a) {
    offsets.col(a) = mY.t();
  }
  return offsets;
}

void fill_topk_from_yblock(
  const arma::mat& yblock,
  const arma::vec& offset,
  const arma::uword total_n,
  const arma::uword row_offset,
  const arma::uword slice,
  const int top_k,
  Rcpp::IntegerVector& top_index,
  Rcpp::NumericVector& top_score
) {
  const int m = static_cast<int>(yblock.n_cols);
  const int use_top_k = std::max(1, std::min(top_k, m));
  const size_t slice_offset =
    static_cast<size_t>(slice) *
    static_cast<size_t>(total_n) *
    static_cast<size_t>(use_top_k);

  for (arma::uword i = 0; i < yblock.n_rows; ++i) {
    std::vector<double> best_score(static_cast<size_t>(use_top_k), -std::numeric_limits<double>::infinity());
    std::vector<int> best_index(static_cast<size_t>(use_top_k), 0);
    for (int j = 0; j < m; ++j) {
      const double value = yblock(i, static_cast<arma::uword>(j)) + offset(static_cast<arma::uword>(j));
      for (int r = 0; r < use_top_k; ++r) {
        if (value > best_score[static_cast<size_t>(r)]) {
          for (int rr = use_top_k - 1; rr > r; --rr) {
            best_score[static_cast<size_t>(rr)] = best_score[static_cast<size_t>(rr - 1)];
            best_index[static_cast<size_t>(rr)] = best_index[static_cast<size_t>(rr - 1)];
          }
          best_score[static_cast<size_t>(r)] = value;
          best_index[static_cast<size_t>(r)] = j + 1;
          break;
        }
      }
    }
    for (int r = 0; r < use_top_k; ++r) {
      const size_t out_pos =
        slice_offset +
        static_cast<size_t>(row_offset + i) +
        static_cast<size_t>(total_n) * static_cast<size_t>(r);
      top_index[out_pos] = best_index[static_cast<size_t>(r)];
      top_score[out_pos] = best_score[static_cast<size_t>(r)];
    }
  }
}

List class_topk_from_cube(
  const arma::cube& Ypred,
  List& model,
  const int top_k
) {
  const int m = Rcpp::as<int>(model["m"]);
  arma::ivec ncomp = Rcpp::as<arma::ivec>(model["ncomp"]);
  const arma::uword length_ncomp = static_cast<arma::uword>(ncomp.n_elem);
  const arma::uword ntest = Ypred.n_rows;
  const int use_top_k = std::max(1, std::min(top_k, m));
  arma::vec zero_offset(static_cast<arma::uword>(m), arma::fill::zeros);

  Rcpp::IntegerVector top_index(ntest * static_cast<arma::uword>(use_top_k) * length_ncomp);
  top_index.attr("dim") = Rcpp::IntegerVector::create(
    static_cast<int>(ntest),
    use_top_k,
    static_cast<int>(length_ncomp)
  );
  Rcpp::NumericVector top_score(ntest * static_cast<arma::uword>(use_top_k) * length_ncomp);
  top_score.attr("dim") = top_index.attr("dim");

  for (arma::uword a = 0; a < length_ncomp; ++a) {
    fill_topk_from_yblock(Ypred.slice(a), zero_offset, ntest, 0, a, use_top_k, top_index, top_score);
  }

  return List::create(
    Named("top_index") = top_index,
    Named("top_score") = top_score
  );
}

// [[Rcpp::export]]
List pls_class_predict_topk_cpp(List& model, arma::mat Xtest, int top_k, bool proj, int block_size) {
  const int m = Rcpp::as<int>(model["m"]);
  arma::ivec ncomp = Rcpp::as<arma::ivec>(model["ncomp"]);
  const arma::uword length_ncomp = static_cast<arma::uword>(ncomp.n_elem);
  const int use_top_k = std::max(1, std::min(top_k, m));

  Rcpp::NumericVector mX_vec = model["mX"];
  arma::rowvec mX(mX_vec.begin(), mX_vec.size(), false, true);
  Xtest.each_row() -= mX;
  Rcpp::NumericVector vX_vec = model["vX"];
  arma::rowvec vX(vX_vec.begin(), vX_vec.size(), false, true);
  Xtest.each_row() /= vX;

  const arma::uword ntest = Xtest.n_rows;
  arma::mat offsets = class_prediction_offsets(model, m, length_ncomp);

  Rcpp::IntegerVector top_index(ntest * static_cast<arma::uword>(use_top_k) * length_ncomp);
  top_index.attr("dim") = Rcpp::IntegerVector::create(
    static_cast<int>(ntest),
    use_top_k,
    static_cast<int>(length_ncomp)
  );
  Rcpp::NumericVector top_score(ntest * static_cast<arma::uword>(use_top_k) * length_ncomp);
  top_score.attr("dim") = top_index.attr("dim");

  arma::uword bs = static_cast<arma::uword>(std::max(1, block_size));
  if (bs > ntest) bs = ntest;

  bool used_compact = false;
  try {
    Rcpp::NumericVector R_vec = model["R"];
    Rcpp::IntegerVector R_dim = R_vec.attr("dim");
    if (R_dim.size() == 2L && R_dim[0] == Xtest.n_cols && R_dim[1] > 0) {
      const arma::mat RR(
        R_vec.begin(),
        static_cast<arma::uword>(R_dim[0]),
        static_cast<arma::uword>(R_dim[1]),
        false,
        true
      );
      const int kmax = static_cast<int>(RR.n_cols);
      arma::cube Wflash = compact_prediction_weights(model, m, ncomp);
      for (arma::uword start = 0; start < ntest; start += bs) {
        const arma::uword stop = std::min(start + bs - 1, ntest - 1);
        const arma::mat scores = Xtest.rows(start, stop) * RR;
        for (arma::uword a = 0; a < length_ncomp; ++a) {
          const int mc = ncomp(a);
          if (mc < 1 || mc > kmax) {
            Rcpp::stop("ncomp exceeds latent rank for compact class prediction");
          }
          arma::mat yblock =
            scores.cols(0, static_cast<arma::uword>(mc - 1)) *
            Wflash.slice(a).rows(0, static_cast<arma::uword>(mc - 1));
          fill_topk_from_yblock(yblock, offsets.col(a), ntest, start, a, use_top_k, top_index, top_score);
        }
      }
      used_compact = true;
    }
  } catch (...) {
    used_compact = false;
  }

  if (!used_compact) {
    if (!model.containsElementNamed("B")) {
      Rcpp::stop("Compact class prediction requires compact factors or stored B");
    }
    Rcpp::NumericVector B_vec = model["B"];
    Rcpp::IntegerVector B_dim = B_vec.attr("dim");
    if (B_dim.size() != 3L || B_dim[0] != Xtest.n_cols || B_dim[1] != m ||
        B_dim[2] < static_cast<int>(length_ncomp)) {
      Rcpp::stop("Model coefficient array `B` is not compatible with compact class prediction");
    }
    const arma::cube B(
      B_vec.begin(),
      static_cast<arma::uword>(B_dim[0]),
      static_cast<arma::uword>(B_dim[1]),
      static_cast<arma::uword>(B_dim[2]),
      false,
      true
    );
    for (arma::uword start = 0; start < ntest; start += bs) {
      const arma::uword stop = std::min(start + bs - 1, ntest - 1);
      const arma::mat Xblock = Xtest.rows(start, stop);
      for (arma::uword a = 0; a < length_ncomp; ++a) {
        arma::mat yblock = Xblock * B.slice(a);
        fill_topk_from_yblock(yblock, offsets.col(a), ntest, start, a, use_top_k, top_index, top_score);
      }
    }
  }

  arma::mat T_Xtest;
  if (proj) {
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
      T_Xtest = Xtest * RR;
    } else {
      T_Xtest.set_size(ntest, 0);
    }
  }

  return List::create(
    Named("top_index") = top_index,
    Named("top_score") = top_score,
    Named("Ttest") = T_Xtest,
    Named("predict_backend") = "cpp_topk"
  );
}

// [[Rcpp::export]]
List pls_class_predict_topk_cuda(List& model, arma::mat Xtest, int top_k, bool proj) {
  if (!fastpls_svd::has_cuda_backend()) {
    return pls_class_predict_topk_cpp(model, Xtest, top_k, proj, 4096);
  }

  try {
    const int m = Rcpp::as<int>(model["m"]);
    arma::ivec ncomp = Rcpp::as<arma::ivec>(model["ncomp"]);
    Rcpp::NumericVector mX_vec = model["mX"];
    arma::rowvec mX(mX_vec.begin(), mX_vec.size(), false, true);
    Xtest.each_row() -= mX;
    Rcpp::NumericVector vX_vec = model["vX"];
    arma::rowvec vX(vX_vec.begin(), vX_vec.size(), false, true);
    Xtest.each_row() /= vX;

    Rcpp::NumericVector mY_vec = model["mY"];
    arma::rowvec mY(mY_vec.begin(), mY_vec.size(), false, true);
    Rcpp::NumericVector R_vec = model["R"];
    Rcpp::IntegerVector R_dim = R_vec.attr("dim");
    if (R_dim.size() != 2L || R_dim[0] != Xtest.n_cols || R_dim[1] < 1) {
      Rcpp::stop("Model `R` is not compatible with CUDA top-k prediction");
    }
    const arma::mat RR(
      R_vec.begin(),
      static_cast<arma::uword>(R_dim[0]),
      static_cast<arma::uword>(R_dim[1]),
      false,
      true
    );
    arma::cube Wflash = compact_prediction_weights(model, m, ncomp);
    arma::cube Ypred = fastpls_svd::cuda_flash_lowrank_predict(Xtest, RR, Wflash, mY, ncomp);
    Rcpp::List out = class_topk_from_cube(Ypred, model, top_k);
    arma::mat T_Xtest;
    if (proj) {
      T_Xtest = Xtest * RR;
    }
    out["Ttest"] = T_Xtest;
    out["predict_backend"] = "cuda_topk";
    return out;
  } catch (...) {
    return pls_class_predict_topk_cpp(model, Xtest, top_k, proj, 4096);
  }
}

arma::imat pls_predict_classes_compact_cpu(List& model, arma::mat Xtest) {
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

  Rcpp::NumericVector R_vec = model["R"];
  Rcpp::IntegerVector R_dim = R_vec.attr("dim");
  if (R_dim.size() != 2L || R_dim[0] != Xtest.n_cols || R_dim[1] < 1) {
    Rcpp::stop("Model `R` is not compatible with compact class prediction");
  }
  const arma::mat RR(
    R_vec.begin(),
    static_cast<arma::uword>(R_dim[0]),
    static_cast<arma::uword>(R_dim[1]),
    false,
    true
  );
  const int kmax = static_cast<int>(RR.n_cols);
  arma::cube Wflash = compact_prediction_weights(model, m, ncomp);

  const arma::uword ntest = Xtest.n_rows;
  arma::imat class_pred(ntest, length_ncomp, arma::fill::zeros);
  arma::uword bs = static_cast<arma::uword>(env_int_or("FASTPLS_COMPACT_CLASS_BLOCK_SIZE", 4096, 128, 1048576));
  if (bs == 0 || bs > ntest) bs = ntest;

  for (arma::uword start = 0; start < ntest; start += bs) {
    const arma::uword stop = std::min(start + bs - 1, ntest - 1);
    const arma::mat Xblock = Xtest.rows(start, stop);
    const arma::mat scores = Xblock * RR;
    for (arma::uword a = 0; a < length_ncomp; ++a) {
      const int mc = ncomp(a);
      if (mc < 1 || mc > kmax) {
        Rcpp::stop("ncomp exceeds latent rank for compact class prediction");
      }
      arma::mat yblock =
        scores.cols(0, static_cast<arma::uword>(mc - 1)) *
        Wflash.slice(a).rows(0, static_cast<arma::uword>(mc - 1));
      yblock.each_row() += mY;
      for (arma::uword i = 0; i < yblock.n_rows; ++i) {
        class_pred(start + i, a) = static_cast<int>(yblock.row(i).index_max()) + 1;
      }
    }
  }
  return class_pred;
}

arma::imat pls_predict_classes_compact_cuda(List& model, arma::mat Xtest) {
  if (!fastpls_svd::has_cuda_backend()) {
    Rcpp::stop("CUDA compact class prediction requires CUDA support");
  }

  const int m = Rcpp::as<int>(model["m"]);
  arma::ivec ncomp = Rcpp::as<arma::ivec>(model["ncomp"]);

  Rcpp::NumericVector mX_vec = model["mX"];
  arma::rowvec mX(mX_vec.begin(), mX_vec.size(), false, true);
  Xtest.each_row() -= mX;
  Rcpp::NumericVector vX_vec = model["vX"];
  arma::rowvec vX(vX_vec.begin(), vX_vec.size(), false, true);
  Xtest.each_row() /= vX;
  Rcpp::NumericVector mY_vec = model["mY"];
  arma::rowvec mY(mY_vec.begin(), mY_vec.size(), false, true);

  Rcpp::NumericVector R_vec = model["R"];
  Rcpp::IntegerVector R_dim = R_vec.attr("dim");
  if (R_dim.size() != 2L || R_dim[0] != Xtest.n_cols || R_dim[1] < 1) {
    Rcpp::stop("Model `R` is not compatible with CUDA compact class prediction");
  }
  const arma::mat RR(
    R_vec.begin(),
    static_cast<arma::uword>(R_dim[0]),
    static_cast<arma::uword>(R_dim[1]),
    false,
    true
  );
  arma::cube Wflash = compact_prediction_weights(model, m, ncomp);
  return fastpls_svd::cuda_flash_lowrank_predict_classes(Xtest, RR, Wflash, mY, ncomp);
}

arma::ivec nearest_code_classes(const arma::mat& Z, const arma::mat& class_codes) {
  if (class_codes.n_rows < 1 || class_codes.n_cols != Z.n_cols) {
    Rcpp::stop("Class codebook is not compatible with predicted code dimensions");
  }
  arma::mat score = 2.0 * (Z * class_codes.t());
  arma::rowvec code_norm = arma::sum(class_codes % class_codes, 1).t();
  score.each_row() -= code_norm;
  arma::ivec out(Z.n_rows);
  for (arma::uword i = 0; i < Z.n_rows; ++i) {
    out(i) = static_cast<int>(score.row(i).index_max()) + 1;
  }
  return out;
}

arma::imat pls_predict_code_classes_compact_cpu(List& model, arma::mat Xtest, const arma::mat& class_codes) {
  const int m = Rcpp::as<int>(model["m"]);
  if (m != static_cast<int>(class_codes.n_cols)) {
    Rcpp::stop("Model response dimension does not match class codebook");
  }
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

  Rcpp::NumericVector R_vec = model["R"];
  Rcpp::IntegerVector R_dim = R_vec.attr("dim");
  if (R_dim.size() != 2L || R_dim[0] != Xtest.n_cols || R_dim[1] < 1) {
    Rcpp::stop("Model `R` is not compatible with compact code prediction");
  }
  const arma::mat RR(
    R_vec.begin(),
    static_cast<arma::uword>(R_dim[0]),
    static_cast<arma::uword>(R_dim[1]),
    false,
    true
  );
  const int kmax = static_cast<int>(RR.n_cols);
  arma::cube Wflash = compact_prediction_weights(model, m, ncomp);

  const arma::uword ntest = Xtest.n_rows;
  arma::imat class_pred(ntest, length_ncomp, arma::fill::zeros);
  arma::uword bs = static_cast<arma::uword>(env_int_or("FASTPLS_COMPACT_CLASS_BLOCK_SIZE", 4096, 128, 1048576));
  if (bs == 0 || bs > ntest) bs = ntest;

  for (arma::uword start = 0; start < ntest; start += bs) {
    const arma::uword stop = std::min(start + bs - 1, ntest - 1);
    const arma::mat Xblock = Xtest.rows(start, stop);
    const arma::mat scores = Xblock * RR;
    for (arma::uword a = 0; a < length_ncomp; ++a) {
      const int mc = ncomp(a);
      if (mc < 1 || mc > kmax) {
        Rcpp::stop("ncomp exceeds latent rank for compact code prediction");
      }
      arma::mat zblock =
        scores.cols(0, static_cast<arma::uword>(mc - 1)) *
        Wflash.slice(a).rows(0, static_cast<arma::uword>(mc - 1));
      zblock.each_row() += mY;
      arma::ivec fold_class = nearest_code_classes(zblock, class_codes);
      class_pred.submat(start, a, stop, a) = fold_class;
    }
  }
  return class_pred;
}

arma::imat pls_predict_code_classes_compact_cuda(List& model, arma::mat Xtest, const arma::mat& class_codes) {
  if (!fastpls_svd::has_cuda_backend()) {
    Rcpp::stop("CUDA compact code prediction requires CUDA support");
  }
  const int m = Rcpp::as<int>(model["m"]);
  if (m != static_cast<int>(class_codes.n_cols)) {
    Rcpp::stop("Model response dimension does not match class codebook");
  }
  Rcpp::NumericVector mX_vec = model["mX"];
  arma::rowvec mX(mX_vec.begin(), mX_vec.size(), false, true);
  Xtest.each_row() -= mX;
  Rcpp::NumericVector vX_vec = model["vX"];
  arma::rowvec vX(vX_vec.begin(), vX_vec.size(), false, true);
  Xtest.each_row() /= vX;
  Rcpp::NumericVector mY_vec = model["mY"];
  arma::rowvec mY(mY_vec.begin(), mY_vec.size(), false, true);

  Rcpp::NumericVector R_vec = model["R"];
  Rcpp::IntegerVector R_dim = R_vec.attr("dim");
  if (R_dim.size() != 2L || R_dim[0] != Xtest.n_cols || R_dim[1] < 1) {
    Rcpp::stop("Model `R` is not compatible with CUDA compact code prediction");
  }
  const arma::mat RR(
    R_vec.begin(),
    static_cast<arma::uword>(R_dim[0]),
    static_cast<arma::uword>(R_dim[1]),
    false,
    true
  );
  arma::ivec ncomp = Rcpp::as<arma::ivec>(model["ncomp"]);
  arma::cube Wflash = compact_prediction_weights(model, m, ncomp);
  arma::cube Zpred = fastpls_svd::cuda_flash_lowrank_predict(Xtest, RR, Wflash, mY, ncomp);
  arma::imat class_pred(Zpred.n_rows, Zpred.n_slices, arma::fill::zeros);
  for (arma::uword s = 0; s < Zpred.n_slices; ++s) {
    class_pred.col(s) = nearest_code_classes(Zpred.slice(s), class_codes);
  }
  return class_pred;
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
List single_pls_cv_cpp(
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
  const bool leave_one_group_out = kfold < 0 || kfold >= xsa_t;
  if (leave_one_group_out) {
    kfold = std::max(xsa_t, 1);
  } else if (kfold < 2) {
    kfold = 2;
  }
  
  
  IntegerVector frame = seq_len(xsa_t);
  IntegerVector v=samplewithoutreplace(frame,xsa_t);
  int mm=constrain2.size();
  
  arma::ivec fold(mm);
  for (int i=0; i<mm; i++) {
    fold[i] = leave_one_group_out ? (constrain2(i) - 1) : (v[constrain2(i)-1] % kfold);
  }
  
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
    Named("best_ncomp") = optim_c,
    Named("Ypred")      = Ypred,
    Named("Q2Y")        = Q2Y,
    Named("R2Y")        = R2Y,
    Named("fold")       = fold
  );
  
}



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
  const bool leave_one_group_out = kfold_outer < 0 || kfold_outer >= xsa_t;
  if (leave_one_group_out) {
    kfold_outer = std::max(xsa_t, 1);
  } else if (kfold_outer < 2) {
    kfold_outer = 2;
  }
  
  
  IntegerVector frame = seq_len(xsa_t);
  IntegerVector v=samplewithoutreplace(frame,xsa_t);
  int mm=constrain2.size();
  
  arma::ivec fold(mm);
  for (int i=0; i<mm; i++) {
    fold[i] = leave_one_group_out ? (constrain2(i) - 1) : (v[constrain2(i)-1] % kfold_outer);
  }

  
  
  
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
    
    List opt=single_pls_cv_cpp(
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
      model=pls_model1(Xtrain,Ytrain,opt("best_ncomp"),scaling,FALSE, svd_method,rsvd_oversample,rsvd_power,svds_tol,seed);
    }
    if(method==2){
      model=pls_model2(Xtrain,Ytrain,opt("best_ncomp"),scaling,FALSE, svd_method,rsvd_oversample,rsvd_power,svds_tol,seed);
    }
    if(method==3){
      model=pls_model2_fast(Xtrain,Ytrain,opt("best_ncomp"),scaling,FALSE, svd_method,rsvd_oversample,rsvd_power,svds_tol,seed);
    }
      
    List pls=pls_predict(model,Xtest,FALSE);
    arma::cube temp1=pls("Ypred");
    for(int ii=0;ii<w1_size;ii++)  for(int kk=0;kk<ncolY;kk++)  Ypred(w1[ii],kk)=temp1(ii,kk,0);  
    
    // Calculation of R2Y
    List model_all;
    if(method==1){
      model_all=pls_model1(Xtrain,Ytrain,opt("best_ncomp"),scaling,TRUE, svd_method,rsvd_oversample,rsvd_power,svds_tol,seed);
    }
    if(method==2){
      model_all=pls_model2(Xtrain,Ytrain,opt("best_ncomp"),scaling,TRUE, svd_method,rsvd_oversample,rsvd_power,svds_tol,seed);
    }
    if(method==3){
      model_all=pls_model2_fast(Xtrain,Ytrain,opt("best_ncomp"),scaling,TRUE, svd_method,rsvd_oversample,rsvd_power,svds_tol,seed);
    }
    
    
    R2Y(i)=model_all("R2Y");
    best_comp(i)=opt("best_ncomp");
  }  
  

  double Q2Y;
  
  
  Q2Y=RQ(Ydata,Ypred);
  
  return List::create(
    Named("Ypred")      = Ypred,
    Named("Q2Y")        = Q2Y,
    Named("R2Y")        = R2Y,
    Named("best_ncomp") = best_comp
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
    plssvd_use_small_exact_svd(max_plssvd_rank, svd_method)
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

List pls_model1_metal_cv(
  arma::mat Xtrain,
  arma::mat Ytrain,
  arma::ivec ncomp,
  int scaling,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed
) {
  if (!fastpls_svd::has_metal_backend()) {
    stop("Metal CV requires a macOS build with Apple Metal support");
  }

  const int n = Xtrain.n_rows;
  const int p = Xtrain.n_cols;
  const int m = Ytrain.n_cols;
  if (ncomp.n_elem < 1) stop("ncomp must contain at least one value");

  const int max_plssvd_rank = std::min(n, std::min(p, m));
  for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
    if (ncomp(i) > max_plssvd_rank) ncomp(i) = max_plssvd_rank;
    if (ncomp(i) < 1) ncomp(i) = 1;
  }
  const int max_ncomp = max(ncomp);
  int max_ncomp_eff = std::min(max_ncomp, max_plssvd_rank);
  if (max_ncomp_eff < 1) stop("plssvd Metal CV effective rank is < 1");

  arma::mat mX(1, p, arma::fill::zeros);
  if (scaling < 3) {
    mX = mean(Xtrain, 0);
    Xtrain.each_row() -= mX;
  }
  arma::mat vX(1, p, arma::fill::ones);
  if (scaling == 2) {
    vX = variance(Xtrain);
    Xtrain.each_row() /= vX;
  }

  arma::mat mY = mean(Ytrain, 0);
  Ytrain.each_row() -= mY;

  arma::mat S = fastpls_svd::metal_crossprod(Xtrain, Ytrain);
  fastpls_svd::SVDResult svd_res = compute_truncated_svd_dispatch(
    S,
    max_ncomp_eff,
    fastpls_svd::SVD_METHOD_CPU_RSVD,
    rsvd_oversample,
    rsvd_power,
    svds_tol,
    static_cast<unsigned int>(seed),
    false,
    plssvd_use_small_exact_svd(max_plssvd_rank, fastpls_svd::SVD_METHOD_CPU_RSVD)
  );

  arma::mat R = svd_res.U;
  arma::vec s = svd_res.s;
  arma::mat Q = svd_res.Vt.t();
  max_ncomp_eff = std::min(max_ncomp_eff, static_cast<int>(R.n_cols));
  if (Q.n_cols > 0) {
    max_ncomp_eff = std::min(max_ncomp_eff, static_cast<int>(Q.n_cols));
  }
  if (max_ncomp_eff < 1) stop("plssvd Metal CV effective rank is < 1 after SVD");
  R = R.cols(0, max_ncomp_eff - 1);
  Q = Q.cols(0, max_ncomp_eff - 1);

  arma::mat T = fastpls_svd::metal_matrix_multiply(Xtrain, R);
  arma::mat G = fastpls_svd::metal_crossprod(T, T);
  const int length_ncomp = ncomp.n_elem;
  arma::cube B(p, m, length_ncomp, arma::fill::zeros);
  arma::cube C_latent(max_ncomp_eff, max_ncomp_eff, length_ncomp, arma::fill::zeros);
  arma::cube W_latent(max_ncomp_eff, m, length_ncomp, arma::fill::zeros);
  arma::vec R2Y(length_ncomp, arma::fill::zeros);

  for (int a = 0; a < length_ncomp; ++a) {
    const int mc = std::min(static_cast<int>(ncomp(a)), max_ncomp_eff);
    arma::mat G_a = G.submat(0, 0, mc - 1, mc - 1);
    arma::mat D_a(mc, mc, arma::fill::zeros);
    D_a.diag() = s.subvec(0, mc - 1);
    arma::mat coeff_latent;
    bool solved = arma::solve(coeff_latent, G_a, D_a, arma::solve_opts::likely_sympd);
    if (!solved) solved = arma::solve(coeff_latent, G_a, D_a);
    if (!solved) stop("plssvd Metal CV latent solve failed");
    C_latent.slice(a).submat(0, 0, mc - 1, mc - 1) = coeff_latent;
    arma::mat W_a = coeff_latent * Q.cols(0, mc - 1).t();
    W_latent.slice(a).submat(0, 0, mc - 1, m - 1) = W_a;
    B.slice(a) = fastpls_svd::metal_matrix_multiply(R.cols(0, mc - 1), W_a);
  }

  List out = List::create(
    Named("C_latent") = C_latent,
    Named("W_latent") = W_latent,
    Named("Q") = Q,
    Named("Ttrain") = arma::mat(),
    Named("R") = R,
    Named("mX") = mX,
    Named("vX") = vX,
    Named("mY") = mY,
    Named("p") = p,
    Named("m") = m,
    Named("ncomp") = ncomp,
    Named("B") = B,
    Named("Yfit") = arma::cube(),
    Named("R2Y") = R2Y,
    Named("backend") = "metal",
    Named("svd.method") = "metal_rsvd",
    Named("pls_method") = "plssvd",
    Named("predict_latent_ok") = true
  );
  annotate_coefficient_storage(out, true);
  return out;
}

List pls_model2_fast_metal_cv(
  arma::mat Xtrain,
  arma::mat Ytrain,
  arma::ivec ncomp,
  int scaling,
  int rsvd_power,
  int seed
) {
  if (!fastpls_svd::has_metal_backend()) {
    stop("Metal CV requires a macOS build with Apple Metal support");
  }

  const int n = Xtrain.n_rows;
  const int p = Xtrain.n_cols;
  const int m = Ytrain.n_cols;
  if (ncomp.n_elem < 1) stop("ncomp must contain at least one value");
  for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
    if (ncomp(i) < 1) ncomp(i) = 1;
  }

  arma::mat mX(1, p, arma::fill::zeros);
  if (scaling < 3) {
    mX = mean(Xtrain, 0);
    Xtrain.each_row() -= mX;
  }
  arma::mat vX(1, p, arma::fill::ones);
  if (scaling == 2) {
    vX = variance(Xtrain);
    Xtrain.each_row() /= vX;
  }

  arma::mat mY = mean(Ytrain, 0);
  Ytrain.each_row() -= mY;

  const int max_ncomp_req = std::max(1, static_cast<int>(max(ncomp)));
  const int max_ncomp_eff = std::max(1, std::min(max_ncomp_req, std::min(p, n - 1)));
  List native = fastpls_svd::metal_simpls_resident(
    Xtrain,
    Ytrain,
    max_ncomp_eff,
    std::max(1, rsvd_power),
    seed
  );

  arma::mat R = Rcpp::as<arma::mat>(native["R"]);
  arma::mat Q = Rcpp::as<arma::mat>(native["Q"]);
  if (R.n_cols == 0 || Q.n_cols == 0) {
    stop("Metal SIMPLS CV returned no latent components");
  }
  const int available = std::max(
    1,
    std::min(
      max_ncomp_eff,
      std::min(static_cast<int>(R.n_cols), static_cast<int>(Q.n_cols))
    )
  );
  const int length_ncomp = ncomp.n_elem;
  for (int a = 0; a < length_ncomp; ++a) {
    const int mc = std::max(1, std::min(static_cast<int>(ncomp(a)), available));
    ncomp(a) = mc;
  }
  if (R.n_cols > static_cast<arma::uword>(available)) R = R.cols(0, available - 1);
  if (Q.n_cols > static_cast<arma::uword>(available)) Q = Q.cols(0, available - 1);

  List out = List::create(
    Named("P") = arma::mat(),
    Named("Q") = Q,
    Named("Ttrain") = arma::mat(),
    Named("R") = R,
    Named("mX") = mX,
    Named("vX") = vX,
    Named("mY") = mY,
    Named("p") = p,
    Named("m") = m,
    Named("ncomp") = ncomp,
    Named("Yfit") = arma::cube(),
    Named("R2Y") = arma::vec(length_ncomp, arma::fill::zeros),
    Named("backend") = "metal",
    Named("svd.method") = "metal_resident_simpls",
    Named("pls_method") = "simpls_fast",
    Named("predict_latent_ok") = true
  );
  annotate_coefficient_storage(out, false);
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
    plssvd_use_small_exact_svd(max_plssvd_rank, fastpls_svd::SVD_METHOD_CPU_RSVD)
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

  const int center_t = env_int_or("FASTPLS_FAST_CENTER_T", 0, 0, 1);
  const int reorth_v = env_int_or("FASTPLS_FAST_REORTH_V", 0, 0, 1);
  const int incremental_coefficients = env_int_or("FASTPLS_INCREMENTAL_COEFFICIENTS", 1, 0, 1);
  arma::vec previous_direction;
  bool has_previous_direction = false;
  auto append_component = [&](arma::vec rr, const int a_idx) -> bool {
    arma::vec tt = Xop.times(rr);
    if (center_t == 1) {
      tt -= arma::mean(tt);
    }
    const double tnorm = arma::norm(tt, 2);
    if (!std::isfinite(tnorm) || tnorm <= 0.0) return false;
    tt /= tnorm;
    rr /= tnorm;
    previous_direction = rr;
    has_previous_direction = true;
    arma::vec pp = Xop.t_times(tt);
    arma::vec qq = Yop.t_times(tt);

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
    if (store_B && incremental_coefficients == 1) {
      Bcur += rr * qq.t();
    }
    if (fit) {
      Yfit_cur += tt * qq.t();
    }

    while (i_out < length_ncomp && a_idx == (ncomp(i_out) - 1)) {
      if (store_B) {
        B.slice(i_out) = incremental_coefficients == 1 ?
          Bcur :
          RR.cols(0, a_idx) * QQ.cols(0, a_idx).t();
      }
      if (fit) {
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
    const int k_block = accelerated_simpls_block_size(
      max_ncomp - a, p, m
    );
    arma::mat Ublock;
    arma::vec shat_block;
    if (!refresh_deflated_crossprod_left_double_view(
          Xop,
          Yop,
          VV,
          a,
          has_previous_direction ? &previous_direction : nullptr,
          k_block,
          std::max(rsvd_power, 0),
          static_cast<unsigned int>(seed + a),
          Ublock,
          shat_block
        )) {
      break;
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
  arma::mat Xtrain,
  arma::mat Ytrain,
  arma::ivec ncomp,
  int scaling,
  bool fit,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed,
  int xprod_precision
) {
  if (xprod_precision == 5) {
    // IRLBA xprod keeps the bundled C IRLBA operator path, but is selected
    // only by the stricter R-side threshold to avoid poor shapes.
  }

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
      plssvd_use_small_exact_svd(max_plssvd_rank, fastpls_svd::SVD_METHOD_CPU_RSVD)
    );
  } else if (xprod_precision == 5) {
    // Matrix-free IRLBA for A = X'Y using the bundled C IRLBA operator API.
    svd_res = truncated_irlba_crossprod_double(
      Xtrain,
      Ytrain,
      max_ncomp_eff,
      false,
      plssvd_use_small_exact_svd(max_plssvd_rank, fastpls_svd::SVD_METHOD_IRLBA)
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
      plssvd_use_small_exact_svd(max_plssvd_rank, fastpls_svd::SVD_METHOD_CPU_RSVD)
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
  arma::mat Xtrain,
  arma::mat Ytrain,
  arma::ivec ncomp,
  int scaling,
  bool fit,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed,
  int xprod_precision
) {
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

  const int center_t = env_int_or("FASTPLS_FAST_CENTER_T", 0, 0, 1);
  const int reorth_v = env_int_or("FASTPLS_FAST_REORTH_V", 0, 0, 1);
  const int defl_cache = env_int_or("FASTPLS_FAST_DEFLCACHE", 1, 0, 1);
  const int fast_optimized = env_int_or("FASTPLS_FAST_OPTIMIZED", 1, 0, 1);
  const int incremental_coefficients = env_int_or("FASTPLS_INCREMENTAL_COEFFICIENTS", 1, 0, 1);
  const int fast_crossprod_min_ncomp = env_int_or("FASTPLS_FAST_CROSSPROD_MIN_NCOMP", 20, 1, 1024);
  const int fast_crossprod_max_p = env_int_or("FASTPLS_FAST_CROSSPROD_MAX_P", 512, 16, 65536);
  const int fast_crossprod_min_n_to_p_ratio = env_int_or("FASTPLS_FAST_CROSSPROD_MIN_N_TO_P_RATIO", 8, 1, 1024);
  const bool return_ttrain = env_int_or("FASTPLS_RETURN_TTRAIN", 0, 0, 1) == 1;
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

  arma::mat TT;
  if (return_ttrain) {
    TT.zeros(n, max_ncomp);
  }
  arma::vec previous_direction;
  bool has_previous_direction = false;
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
      if (fit || return_ttrain) tt = Xtrain * rr;
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
    previous_direction = rr;
    has_previous_direction = true;
    if (return_ttrain && tt.n_elem == static_cast<arma::uword>(n)) {
      TT.col(a_idx) = tt;
    }
    if (store_B && incremental_coefficients == 1) {
      Bcur += rr * qq.t();
    }
    if (fit) {
      Yfit_cur += tt * qq.t();
    }

    while (i_out < length_ncomp && a_idx == (ncomp(i_out) - 1)) {
      if (store_B) {
        B.slice(i_out) = incremental_coefficients == 1 ?
          Bcur :
          RR.cols(0, a_idx) * QQ.cols(0, a_idx).t();
      }
      if (fit) {
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
  const int rsvd_sketch_dim = std::min(
    std::min(p, m),
    1 + std::max(rsvd_oversample, 0)
  );
  const int requested_power_iters = std::max(rsvd_power, 0);
  int a = 0;
  while (a < max_ncomp) {
    const int k_block = use_implicit_irlba_xprod ? 1 :
      accelerated_simpls_block_size(max_ncomp - a, p, m);
    arma::mat Ublock;
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
    } else if (use_implicit_double_xprod) {
      arma::vec shat_block;
      if (!refresh_deflated_crossprod_left_double(
            Xtrain,
            Ytrain,
            VV,
            a,
            has_previous_direction ? &previous_direction : nullptr,
            k_block,
            requested_power_iters,
            static_cast<unsigned int>(seed + a),
            Ublock,
            shat_block
          )) {
        break;
      }
    } else {
      fastpls_svd::SVDResult direction = compute_truncated_svd_dispatch(
        S,
        k_block,
        fastpls_svd::SVD_METHOD_CPU_RSVD,
        std::max(rsvd_sketch_dim - 1, 0),
        requested_power_iters,
        0.0,
        static_cast<unsigned int>(seed + a),
        true,
        false
      );
      Ublock = direction.U;
      refresh_ws.shat = direction.s;
      if (Ublock.n_cols < 1) {
        break;
      }
    }
    if (Ublock.n_cols < 1) break;

    const int use_cols = std::min<int>(Ublock.n_cols, k_block);
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
    Named("Ttrain")  = return_ttrain ? TT : arma::mat(),
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
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed
) {
  if (!fastpls_svd::has_cuda_backend()) {
    stop("pls_model1_gpu requires CUDA support");
  }
  if (svd_method != fastpls_svd::SVD_METHOD_CUDA_RSVD) {
    stop("pls_model1_gpu requires svd.method='cuda_rsvd'");
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

  fastpls_svd::SVDOptions opt = fastpls_svd::options_from_method_id(
    svd_method,
    rsvd_oversample,
    rsvd_power,
    svds_tol,
    static_cast<unsigned int>(seed),
    false,
    false
  );

  const bool store_B = should_store_coefficients(p, m, ncomp.n_elem, true);
  fastpls_svd::PLSSVDGPUResult gpu = fastpls_svd::cuda_plssvd_fit(
    Xtrain,
    Ytrain,
    ncomp,
    fit,
    opt,
    store_B
  );

  arma::cube Yfit = gpu.Yfit;
  if (fit && Yfit.n_elem > 0) {
    for (arma::uword i = 0; i < Yfit.n_slices; ++i) {
      Yfit.slice(i).each_row() += mY;
    }
  }

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

  const bool store_B = should_store_coefficients(p, m, ncomp.n_elem, true);
  fastpls_svd::PLSSVDGPUResult gpu = fastpls_svd::cuda_plssvd_fit_implicit_xprod(
    Xtrain,
    Ytrain,
    ncomp,
    fit,
    opt,
    store_B
  );

  arma::cube Yfit = gpu.Yfit;
  if (fit && Yfit.n_elem > 0) {
    for (arma::uword i = 0; i < Yfit.n_slices; ++i) {
      Yfit.slice(i).each_row() += mY;
    }
  }

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
List pls_lda_gpu_native(
  arma::mat Xtrain,
  arma::mat Ytrain,
  arma::ivec y,
  arma::mat Xtest,
  arma::ivec ncomp,
  int n_classes,
  int method,
  int scaling,
  bool xprod,
  bool fit,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed,
  double lda_ridge
) {
  if (!fastpls_svd::has_cuda_backend() || !fastpls_svd::cuda_lda_native_available()) {
    stop("pls_lda_gpu_native requires CUDA PLS and native CUDA LDA support");
  }
  if (Xtrain.n_rows == 0 || Xtrain.n_cols == 0 || Ytrain.n_rows != Xtrain.n_rows) {
    stop("pls_lda_gpu_native requires compatible non-empty Xtrain and Ytrain");
  }
  if (static_cast<arma::uword>(y.n_elem) != Xtrain.n_rows) {
    stop("pls_lda_gpu_native requires one class label per training row");
  }
  if (Xtest.n_cols != Xtrain.n_cols) {
    stop("pls_lda_gpu_native Xtest columns must match Xtrain columns");
  }
  if (n_classes < 2) {
    stop("pls_lda_gpu_native requires at least two classes");
  }
  if (ncomp.n_elem < 1) {
    stop("pls_lda_gpu_native requires at least one component count");
  }
  const arma::uword n_train_rows = Xtrain.n_rows;

  List model;
  bool direct_plssvd_gpu = false;
  fastpls_svd::PLSSVDGPUResult direct_gpu;
  if (method == 1) {
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
    const int max_ncomp_eff = std::min(static_cast<int>(max(ncomp)), max_plssvd_rank);
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

    const bool store_B = should_store_coefficients(p, m, ncomp.n_elem, true);
    direct_gpu = xprod ?
      fastpls_svd::cuda_plssvd_fit_implicit_xprod(Xtrain, Ytrain, ncomp, fit, opt, store_B) :
      fastpls_svd::cuda_plssvd_fit(Xtrain, Ytrain, ncomp, fit, opt, store_B);
    direct_plssvd_gpu = true;

    arma::cube Yfit = direct_gpu.Yfit;
    if (fit && Yfit.n_elem > 0) {
      for (arma::uword i = 0; i < Yfit.n_slices; ++i) {
        Yfit.slice(i).each_row() += mY;
      }
    }

    model = List::create(
      Named("C_latent") = direct_gpu.C_latent,
      Named("W_latent") = direct_gpu.W_latent,
      Named("Q")       = direct_gpu.Q,
      Named("Ttrain")  = arma::mat(),
      Named("R")       = direct_gpu.R,
      Named("mX")      = mX,
      Named("vX")      = vX,
      Named("mY")      = mY,
      Named("p")       = p,
      Named("m")       = m,
      Named("ncomp")   = ncomp,
      Named("Yfit")    = Yfit,
      Named("R2Y")     = direct_gpu.R2Y,
      Named("xprod_mode") = xprod ? "implicit" : "materialized"
    );
    if (store_B) {
      model["B"] = direct_gpu.B;
    }
    annotate_coefficient_storage(model, store_B);
  } else if (method == 3) {
    model = pls_model2_fast_gpu(
      Xtrain,
      Ytrain,
      ncomp,
      scaling,
      fit,
      fastpls_svd::SVD_METHOD_CUDA_RSVD,
      rsvd_oversample,
      rsvd_power,
      svds_tol,
      seed
    );
  } else {
    stop("pls_lda_gpu_native currently supports method=1 (plssvd) or method=3 (simpls)");
  }

  arma::mat R = Rcpp::as<arma::mat>(model["R"]);
  arma::mat mX = Rcpp::as<arma::mat>(model["mX"]);
  arma::mat vX = Rcpp::as<arma::mat>(model["vX"]);
  arma::ivec ncomp_eff = Rcpp::as<arma::ivec>(model["ncomp"]);
  int kmax = 0;
  for (arma::uword i = 0; i < ncomp_eff.n_elem; ++i) {
    if (ncomp_eff(i) > kmax) kmax = ncomp_eff(i);
  }
  if (kmax < 1 || kmax > static_cast<int>(R.n_cols)) {
    stop("pls_lda_gpu_native has invalid effective component count");
  }

  arma::mat R_predict = R.cols(0, static_cast<arma::uword>(kmax) - 1);
  if (vX.n_elem == R_predict.n_rows) {
    arma::vec scale = arma::vectorise(vX);
    for (arma::uword j = 0; j < R_predict.n_rows; ++j) {
      double s = scale(j);
      if (!std::isfinite(s) || s == 0.0) s = 1.0;
      R_predict.row(j) /= s;
    }
  }
  arma::rowvec offset(kmax, arma::fill::zeros);
  if (mX.n_elem == R_predict.n_rows) {
    offset = arma::vectorise(mX).t() * R_predict;
  }

  arma::ivec unique_ncomp = arma::unique(ncomp_eff);
  std::string lda_train_backend = "cuda_fused_project";
  std::vector<fastpls_svd::LDAGPUModel> gpu_models;
  if (direct_plssvd_gpu &&
      direct_gpu.Ttrain.n_rows == n_train_rows &&
      direct_gpu.Ttrain.n_cols >= static_cast<arma::uword>(kmax)) {
    gpu_models = fastpls_svd::cuda_lda_train_prefix(
      direct_gpu.Ttrain.cols(0, static_cast<arma::uword>(kmax) - 1),
      y,
      n_classes,
      unique_ncomp,
      lda_ridge
    );
    lda_train_backend = "cuda_fused_ttrain";
  } else if (model.containsElementNamed("Ttrain")) {
    arma::mat Ttrain = Rcpp::as<arma::mat>(model["Ttrain"]);
    if (Ttrain.n_rows == n_train_rows &&
        Ttrain.n_cols >= static_cast<arma::uword>(kmax)) {
      gpu_models = fastpls_svd::cuda_lda_train_prefix(
        Ttrain.cols(0, static_cast<arma::uword>(kmax) - 1),
        y,
        n_classes,
        unique_ncomp,
        lda_ridge
      );
      lda_train_backend = "cuda_fused_ttrain";
    }
  }
  if (gpu_models.empty()) {
    gpu_models = fastpls_svd::cuda_lda_project_train_prefix(
      Xtrain,
      R_predict,
      offset,
      y,
      n_classes,
      unique_ncomp,
      lda_ridge
    );
  }

  Rcpp::List lda_models(unique_ncomp.n_elem);
  Rcpp::CharacterVector lda_names(unique_ncomp.n_elem);
  for (arma::uword i = 0; i < unique_ncomp.n_elem; ++i) {
    const fastpls_svd::LDAGPUModel& gm = gpu_models[static_cast<size_t>(i)];
    lda_models[i] = Rcpp::List::create(
      Rcpp::Named("means") = gm.means,
      Rcpp::Named("inv_cov") = arma::mat(),
      Rcpp::Named("linear") = gm.linear,
      Rcpp::Named("constants") = gm.constants,
      Rcpp::Named("priors") = gm.priors,
      Rcpp::Named("ridge") = gm.ridge,
      Rcpp::Named("backend") = "cuda_native_fused"
    );
    lda_names[i] = std::to_string(unique_ncomp(i));
  }
  lda_models.attr("names") = lda_names;
  model["lda"] = Rcpp::List::create(
    Rcpp::Named("ncomp") = unique_ncomp,
    Rcpp::Named("models") = lda_models,
    Rcpp::Named("ridge") = lda_ridge,
    Rcpp::Named("train_backend") = lda_train_backend
  );
  model["R_predict"] = R_predict;
  model["R_offset"] = offset;
  model["classification_rule"] = "lda_cuda";
  model["lda_backend"] = "cuda_fused";

  if (Xtest.n_rows > 0) {
    Rcpp::IntegerMatrix pred_codes(Xtest.n_rows, ncomp_eff.n_elem);
    for (arma::uword i = 0; i < ncomp_eff.n_elem; ++i) {
      const int kk = ncomp_eff(i);
      arma::uword model_idx = 0;
      while (model_idx < unique_ncomp.n_elem && unique_ncomp(model_idx) != kk) {
        ++model_idx;
      }
      if (model_idx >= unique_ncomp.n_elem) {
        stop("pls_lda_gpu_native could not match LDA model to ncomp");
      }
      const fastpls_svd::LDAGPUModel& gm = gpu_models[static_cast<size_t>(model_idx)];
      Rcpp::List pred = fastpls_svd::cuda_lda_project_predict(
        Xtest,
        R_predict.cols(0, static_cast<arma::uword>(kk) - 1),
        offset.subvec(0, static_cast<arma::uword>(kk) - 1),
        gm.linear,
        gm.constants,
        false
      );
      Rcpp::IntegerVector col = pred["pred"];
      for (R_xlen_t r = 0; r < col.size(); ++r) {
        pred_codes(r, i) = col[r];
      }
    }
    model["pred_codes"] = pred_codes;
  }

  model["predict_backend"] = "cuda_fused_lda";
  model["flash_svd"] = true;
  model["flash_svd_backend"] = "cuda";
  model["flash_svd_mode"] = "fused_pls_lda";
  return model;
}

arma::cube pls_predict_scores_b_metal_cv(List& model, arma::mat Xtest) {
  if (!fastpls_svd::has_metal_backend()) {
    stop("Metal CV prediction requires a macOS build with Apple Metal support");
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

  arma::cube Ypred(Xtest.n_rows, static_cast<arma::uword>(m), length_ncomp, arma::fill::none);

  if (model.containsElementNamed("W_latent") && model.containsElementNamed("R")) {
    arma::mat R = Rcpp::as<arma::mat>(model["R"]);
    arma::cube W_latent = Rcpp::as<arma::cube>(model["W_latent"]);
    const int kmax = std::max(
      1,
      std::min(
        static_cast<int>(max(ncomp)),
        std::min(static_cast<int>(R.n_cols), static_cast<int>(W_latent.n_rows))
      )
    );
    arma::mat T = fastpls_svd::metal_matrix_multiply(Xtest, R.cols(0, kmax - 1));
    for (arma::uword a = 0; a < length_ncomp; ++a) {
      const int mc = std::max(1, std::min(static_cast<int>(ncomp(a)), kmax));
      arma::mat y = fastpls_svd::metal_matrix_multiply(
        T.cols(0, mc - 1),
        W_latent.slice(a).rows(0, mc - 1)
      );
      y.each_row() += mY;
      Ypred.slice(a) = y;
    }
    return Ypred;
  }

  if (model.containsElementNamed("B")) {
    Rcpp::NumericVector B_vec = model["B"];
    Rcpp::IntegerVector B_dim = B_vec.attr("dim");
    if (B_dim.size() != 3L ||
        B_dim[0] != Xtest.n_cols ||
        B_dim[1] != m ||
        B_dim[2] < static_cast<int>(length_ncomp)) {
      stop("Metal CV model coefficients are not compatible with prediction");
    }
    const arma::cube B(
      B_vec.begin(),
      static_cast<arma::uword>(B_dim[0]),
      static_cast<arma::uword>(B_dim[1]),
      static_cast<arma::uword>(B_dim[2]),
      false,
      true
    );
    for (arma::uword a = 0; a < length_ncomp; ++a) {
      arma::mat y = fastpls_svd::metal_matrix_multiply(Xtest, B.slice(a));
      y.each_row() += mY;
      Ypred.slice(a) = y;
    }
    return Ypred;
  }

  if (model.containsElementNamed("R") && model.containsElementNamed("Q")) {
    arma::mat R = Rcpp::as<arma::mat>(model["R"]);
    arma::mat Q = Rcpp::as<arma::mat>(model["Q"]);
    if (R.n_cols == 0 || Q.n_cols == 0) {
      stop("Metal CV compact model contains no latent components");
    }
    const int kmax = std::max(
      1,
      std::min(
        static_cast<int>(max(ncomp)),
        std::min(static_cast<int>(R.n_cols), static_cast<int>(Q.n_cols))
      )
    );
    arma::mat T = fastpls_svd::metal_matrix_multiply(Xtest, R.cols(0, kmax - 1));
    for (arma::uword a = 0; a < length_ncomp; ++a) {
      const int mc = std::max(1, std::min(static_cast<int>(ncomp(a)), kmax));
      arma::mat y = fastpls_svd::metal_matrix_multiply(
        T.cols(0, mc - 1),
        Q.cols(0, mc - 1).t()
      );
      y.each_row() += mY;
      Ypred.slice(a) = y;
    }
    return Ypred;
  }

  stop("Metal CV model does not contain usable prediction factors");
}

static arma::mat cv_projection_matrix(List& model, const int kmax, const arma::uword p) {
  Rcpp::NumericVector R_vec = model["R"];
  Rcpp::IntegerVector R_dim = R_vec.attr("dim");
  if (R_dim.size() != 2L || R_dim[0] != static_cast<int>(p) || R_dim[1] < kmax) {
    Rcpp::stop("CV classifier requires a compatible latent projection matrix R");
  }
  const arma::mat R(
    R_vec.begin(),
    static_cast<arma::uword>(R_dim[0]),
    static_cast<arma::uword>(R_dim[1]),
    false,
    true
  );
  arma::mat R_predict = R.cols(0, static_cast<arma::uword>(kmax) - 1);
  Rcpp::NumericVector vX_vec = model["vX"];
  arma::rowvec vX(vX_vec.begin(), vX_vec.size(), false, true);
  if (vX.n_elem == R_predict.n_rows) {
    for (arma::uword j = 0; j < R_predict.n_rows; ++j) {
      double s = vX(j);
      if (!std::isfinite(s) || s == 0.0) s = 1.0;
      R_predict.row(j) /= s;
    }
  }
  return R_predict;
}

static arma::rowvec cv_projection_offset(List& model, const arma::mat& R_predict) {
  arma::rowvec offset(R_predict.n_cols, arma::fill::zeros);
  Rcpp::NumericVector mX_vec = model["mX"];
  arma::rowvec mX(mX_vec.begin(), mX_vec.size(), false, true);
  if (mX.n_elem == R_predict.n_rows) {
    offset = mX * R_predict;
  }
  return offset;
}

static arma::mat cv_latent_scores(List& model,
                                  const arma::mat& X,
                                  const int kmax,
                                  const bool prefer_stored_ttrain) {
  if (prefer_stored_ttrain && model.containsElementNamed("Ttrain")) {
    arma::mat Ttrain = Rcpp::as<arma::mat>(model["Ttrain"]);
    if (Ttrain.n_rows == X.n_rows && Ttrain.n_cols >= static_cast<arma::uword>(kmax)) {
      return Ttrain.cols(0, static_cast<arma::uword>(kmax) - 1);
    }
  }
  arma::mat R_predict = cv_projection_matrix(model, kmax, X.n_cols);
  arma::rowvec offset = cv_projection_offset(model, R_predict);
  arma::mat T = X * R_predict;
  if (offset.n_elem >= static_cast<arma::uword>(kmax)) {
    T.each_row() -= offset;
  }
  return T;
}

static arma::imat cv_lda_predict_prefix_labels_cpp(const arma::mat& Ttest,
                                                   const Rcpp::List& lda_models,
                                                   const arma::ivec& ncomp) {
  const int ncopy = std::min(static_cast<int>(ncomp.n_elem), static_cast<int>(lda_models.size()));
  arma::imat out(Ttest.n_rows, static_cast<arma::uword>(ncopy), arma::fill::zeros);
  for (int s = 0; s < ncopy; ++s) {
    Rcpp::List lda_model = lda_models[s];
    arma::mat linear = Rcpp::as<arma::mat>(lda_model["linear"]);
    arma::rowvec constants = Rcpp::as<arma::rowvec>(lda_model["constants"]);
    const int kk = ncomp(static_cast<arma::uword>(s));
    if (kk < 1 ||
        kk > static_cast<int>(Ttest.n_cols) ||
        linear.n_cols != static_cast<arma::uword>(kk) ||
        constants.n_elem != linear.n_rows) {
      Rcpp::stop("CV LDA prefix model is not compatible with the requested component count");
    }
    arma::mat scores = Ttest.cols(0, static_cast<arma::uword>(kk) - 1) * linear.t();
    Rcpp::IntegerVector pred = lda_labels_from_scores(scores, constants);
    for (R_xlen_t ii = 0; ii < pred.size(); ++ii) {
      out(static_cast<arma::uword>(ii), static_cast<arma::uword>(s)) = pred[ii];
    }
  }
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
  int n_response,
  bool xprod,
  int opls_north,
  bool return_scores,
  arma::mat class_codes,
  int classifier,
  double lda_ridge,
  bool store_predictions,
  int metric_id
) {
  const int nsamples = Xdata.n_rows;
  const bool label_classification = classification && Ydata.n_cols == 1;
  const bool use_class_codes =
    classification && label_classification && class_codes.n_rows > 0 && class_codes.n_cols > 0;
  const int n_classes = label_classification ?
    std::max(n_response, 1) :
    static_cast<int>(Ydata.n_cols);
  int ncolY = static_cast<int>(Ydata.n_cols);
  if (use_class_codes) {
    if (class_codes.n_rows != static_cast<arma::uword>(n_classes)) {
      stop("class_codes must have one row for each response class");
    }
    ncolY = static_cast<int>(class_codes.n_cols);
  }
  if (label_classification) {
    ncolY = use_class_codes ? ncolY : n_classes;
  }
  if (nsamples < 2) stop("Xdata must contain at least two samples");
  if (Ydata.n_rows != static_cast<arma::uword>(nsamples)) {
    stop("Ydata must have the same number of rows as Xdata");
  }
  if (constrain.n_elem != static_cast<arma::uword>(nsamples)) {
    stop("constrain must have one value for each sample");
  }
  if (ncomp.n_elem < 1) stop("ncomp must contain at least one value");
  const bool requested_leave_one_group_out = kfold < 0;
  if (!requested_leave_one_group_out && kfold < 2) kfold = 2;
  if (method < 1 || method > 5) {
    stop("method must be 1=plssvd, 2=simpls, 3=simpls_fast, 4=opls, or 5=kernelpls");
  }
  if (backend < 0 || backend > 2) stop("backend must be 0=cpp, 1=cuda, or 2=metal");
  if (classifier < 0 || classifier > 1) classifier = 0;
  if (!classification) classifier = 0;
  if (use_class_codes && classifier != 0) {
    stop("LDA CV is not available with Gaussian/code response compression");
  }
  if (backend == 1 && method == 2) {
    stop("CUDA classic SIMPLS is not implemented; use simpls_fast CUDA instead");
  }
  if (backend == 1 && !fastpls_svd::has_cuda_backend()) {
    stop("CUDA CV requires a CUDA-enabled fastPLS build");
  }
  if (backend == 2 && !fastpls_svd::has_metal_backend()) {
    stop("Metal CV requires a macOS build with Apple Metal support");
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

  auto class_label_at_sample = [&](const arma::uword i) -> int {
    if (label_classification) {
      return static_cast<int>(std::round(Ydata(i, 0)));
    }
    if (classification) {
      arma::uword best = 0;
      double best_val = Ydata(i, 0);
      for (arma::uword c = 1; c < Ydata.n_cols; ++c) {
        if (Ydata(i, c) > best_val) {
          best_val = Ydata(i, c);
          best = c;
        }
      }
      return static_cast<int>(best) + 1;
    }
    return 1;
  };

  arma::ivec unique_groups = arma::unique(constrain);
  arma::ivec constrain2 = constrain;
  for (arma::uword j = 0; j < unique_groups.n_elem; ++j) {
    arma::uvec ind = arma::find(constrain == unique_groups(j));
    constrain2.elem(ind).fill(static_cast<int>(j) + 1);
  }

  const int ngroups = unique_groups.n_elem;
  const bool leave_one_group_out = requested_leave_one_group_out || kfold >= ngroups;
  if (leave_one_group_out) {
    kfold = std::max(ngroups, 1);
  }
  arma::ivec group_fold(ngroups, arma::fill::zeros);
  if (leave_one_group_out) {
    for (int j = 0; j < ngroups; ++j) {
      group_fold(j) = j;
    }
  } else if (classification && n_classes > 1) {
    std::vector<std::vector<int> > groups_by_class(static_cast<std::size_t>(n_classes));
    for (int j = 0; j < ngroups; ++j) {
      arma::uvec ind = arma::find(constrain2 == (j + 1));
      if (ind.n_elem < 1) continue;
      int cls = class_label_at_sample(ind(0));
      if (cls < 1) cls = 1;
      if (cls > n_classes) cls = n_classes;
      groups_by_class[static_cast<std::size_t>(cls - 1)].push_back(j);
    }
    for (int cls = 0; cls < n_classes; ++cls) {
      const int n_class_groups = static_cast<int>(groups_by_class[static_cast<std::size_t>(cls)].size());
      if (n_class_groups < 1) continue;
      IntegerVector frame = seq_len(n_class_groups);
      IntegerVector perm = samplewithoutreplace(frame, n_class_groups);
      for (int pos = 0; pos < n_class_groups; ++pos) {
        const int perm_pos = perm[pos] - 1;
        const int group_idx = groups_by_class[static_cast<std::size_t>(cls)][static_cast<std::size_t>(perm_pos)];
        group_fold(group_idx) = pos % kfold;
      }
    }
  } else {
    IntegerVector frame = seq_len(ngroups);
    IntegerVector perm = samplewithoutreplace(frame, ngroups);
    for (int j = 0; j < ngroups; ++j) {
      group_fold(j) = (perm[j] - 1) % kfold;
    }
  }
  arma::ivec fold(nsamples);
  for (int i = 0; i < nsamples; ++i) {
    const int group_idx = constrain2(i) - 1;
    fold(i) = group_fold(group_idx);
  }

  const int length_ncomp = ncomp.n_elem;
  const bool store_score_predictions =
    store_predictions && ((!classification) || (return_scores && classifier == 0));
  const bool store_class_predictions = store_predictions && classification;
  const bool latent_classifier_cv = classification && !store_score_predictions && classifier != 0;
  arma::cube Ypred;
  if (store_score_predictions) {
    Ypred.zeros(nsamples, ncolY, length_ncomp);
  }
  arma::imat class_pred;
  if (store_class_predictions) {
    class_pred.zeros(nsamples, length_ncomp);
  }
  arma::vec metric_sse(length_ncomp, arma::fill::zeros);
  arma::vec metric_count(length_ncomp, arma::fill::zeros);
  arma::vec metric_correct(length_ncomp, arma::fill::zeros);
  arma::vec metric_total(length_ncomp, arma::fill::zeros);
  if (classification) {
    metric_id = 1;
  } else if (metric_id < 2 || metric_id > 4) {
    metric_id = 4;
  }
  double metric_tss = NA_REAL;
  if (!classification && (metric_id == 2 || metric_id == 3)) {
    arma::rowvec y_center = arma::mean(Ydata, 0);
    arma::mat centered = Ydata.each_row() - y_center;
    metric_tss = arma::accu(arma::square(centered));
  }
  arma::ivec status(kfold, arma::fill::zeros);

  const std::string method_name =
    (method == 1) ? "plssvd" :
    ((method == 2) ? "simpls" :
    ((method == 4) ? "opls" :
    ((method == 5) ? "kernelpls" : "simpls_fast")));

  auto response_rows = [&](const arma::uvec& idx) -> arma::mat {
    arma::mat out(idx.n_elem, static_cast<arma::uword>(ncolY), arma::fill::zeros);
    for (arma::uword ii = 0; ii < idx.n_elem; ++ii) {
      const int cls = static_cast<int>(std::round(Ydata(idx(ii), 0)));
      if (use_class_codes) {
        if (cls >= 1 && cls <= n_classes) {
          out.row(ii) = class_codes.row(static_cast<arma::uword>(cls - 1));
        }
      } else if (cls >= 1 && cls <= ncolY) {
        out(ii, static_cast<arma::uword>(cls - 1)) = 1.0;
      }
    }
    return out;
  };

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
        if (classification) {
          for (arma::uword ii = 0; ii < test_idx.n_elem; ++ii) {
            const int actual_class = class_label_at_sample(test_idx(ii));
            if (store_class_predictions) {
              class_pred(test_idx(ii), s) = actual_class;
            }
            metric_correct(s) += 1.0;
            metric_total(s) += 1.0;
          }
          if (store_score_predictions) {
            if (label_classification) {
              Ypred.slice(s).rows(test_idx) = response_rows(test_idx);
            } else {
              Ypred.slice(s).rows(test_idx) = Ydata.rows(test_idx);
            }
          }
        } else {
          if (store_score_predictions) {
            Ypred.slice(s).rows(test_idx) = Ydata.rows(test_idx);
          }
          metric_count(s) += static_cast<double>(test_idx.n_elem * Ydata.n_cols);
        }
      }
      status(f) = 3; // no training data
      continue;
    }

    arma::mat Xtrain = Xdata.rows(train_idx);
    arma::mat Xtest = Xdata.rows(test_idx);
    arma::mat Ytrain = label_classification ? response_rows(train_idx) : Ydata.rows(train_idx);

    if (classification) {
      arma::rowvec class_counts(n_classes, arma::fill::zeros);
      for (arma::uword ii = 0; ii < train_idx.n_elem; ++ii) {
        int cls = class_label_at_sample(train_idx(ii));
        if (cls >= 1 && cls <= n_classes) {
          class_counts(static_cast<arma::uword>(cls - 1)) += 1.0;
        }
      }
      arma::uvec active = arma::find(class_counts > 0.5);
      if (active.n_elem <= 1) {
        arma::rowvec fallback(ncolY, arma::fill::zeros);
        if (active.n_elem == 1) {
          if (use_class_codes) {
            fallback = class_codes.row(active(0));
          } else {
            fallback(active(0)) = 1.0;
          }
        } else {
          fallback = label_classification ? arma::mean(response_rows(train_idx), 0) : arma::mean(Ydata, 0);
        }
        const int fallback_class = (active.n_elem == 1) ?
          static_cast<int>(active(0)) + 1 :
          static_cast<int>(nearest_code_classes(fallback, use_class_codes ? class_codes : arma::eye(ncolY, ncolY))(0));
        for (int s = 0; s < length_ncomp; ++s) {
          for (arma::uword ii = 0; ii < test_idx.n_elem; ++ii) {
            if (store_class_predictions) {
              class_pred(test_idx(ii), s) = fallback_class;
            }
            metric_correct(s) += (fallback_class == class_label_at_sample(test_idx(ii))) ? 1.0 : 0.0;
            metric_total(s) += 1.0;
            if (store_score_predictions) {
              Ypred.slice(s).row(test_idx(ii)) = fallback;
            }
          }
        }
        status(f) = 4; // degenerate classification fold
        continue;
      }
    }

    int fit_method = method;
    int fit_scaling = scaling;
    if (method == 4) {
      const int north_eff = std::max(opls_north, 0);
      List filt = opls_filter_cpp(Xtrain, Ytrain, north_eff, scaling);
      Xtrain = Rcpp::as<arma::mat>(filt["X"]);
      arma::rowvec mX = Rcpp::as<arma::rowvec>(filt["mX"]);
      arma::rowvec vX = Rcpp::as<arma::rowvec>(filt["vX"]);
      arma::mat W_orth = Rcpp::as<arma::mat>(filt["W_orth"]);
      arma::mat P_orth = Rcpp::as<arma::mat>(filt["P_orth"]);
      Xtest = opls_apply_filter_cpp(Xtest, mX, vX, W_orth, P_orth);
      fit_method = 3;
      fit_scaling = 3;
    } else if (method == 5) {
      // Linear kernelPLS is algebraically the direct SIMPLS core. Nonlinear
      // kernels still use the ordinary kernel_pls_* fit wrappers.
      fit_method = 3;
    }

    auto fit_input_X = [&](arma::mat& X) -> arma::mat {
      if (latent_classifier_cv) {
        return X;
      }
      return std::move(X);
    };

    List model;
    if (backend == 1) {
      if (fit_method == 1) {
        if (xprod) {
          model = pls_model1_gpu_implicit_xprod(
            fit_input_X(Xtrain), std::move(Ytrain), ncomp, fit_scaling, false,
            rsvd_oversample, rsvd_power, svds_tol, seed + f
          );
        } else {
          model = pls_model1_gpu(
            fit_input_X(Xtrain), std::move(Ytrain), ncomp, fit_scaling, false,
            fastpls_svd::SVD_METHOD_CUDA_RSVD,
            rsvd_oversample, rsvd_power, svds_tol, seed + f
          );
        }
      } else {
        model = pls_model2_fast_gpu(
          fit_input_X(Xtrain), std::move(Ytrain), ncomp, fit_scaling, false,
          fastpls_svd::SVD_METHOD_CUDA_RSVD,
          rsvd_oversample, rsvd_power, svds_tol, seed + f
        );
      }
    } else if (backend == 2) {
      if (fit_method == 1) {
        model = pls_model1_metal_cv(
          fit_input_X(Xtrain), std::move(Ytrain), ncomp, fit_scaling,
          rsvd_oversample, rsvd_power, svds_tol, seed + f
        );
      } else {
        model = pls_model2_fast_metal_cv(
          fit_input_X(Xtrain), std::move(Ytrain), ncomp, fit_scaling,
          rsvd_power, seed + f
        );
      }
    } else {
      if (fit_method == 1) {
        if (xprod) {
          const int xprod_precision = (svd_method == fastpls_svd::SVD_METHOD_IRLBA) ? 5 : 3;
          model = pls_model1_rsvd_xprod_precision(
            fit_input_X(Xtrain), std::move(Ytrain), ncomp, fit_scaling, false,
            rsvd_oversample, rsvd_power, svds_tol, seed + f, xprod_precision
          );
        } else {
          model = pls_model1(
            fit_input_X(Xtrain), std::move(Ytrain), ncomp, fit_scaling, false, svd_method,
            rsvd_oversample, rsvd_power, svds_tol, seed + f
          );
        }
      } else if (fit_method == 2) {
        model = pls_model2(
          fit_input_X(Xtrain), std::move(Ytrain), ncomp, fit_scaling, false, svd_method,
          rsvd_oversample, rsvd_power, svds_tol, seed + f
        );
      } else {
        if (xprod) {
          const int xprod_precision = (svd_method == fastpls_svd::SVD_METHOD_IRLBA) ? 5 : 3;
          model = pls_model2_fast_rsvd_xprod_precision(
            fit_input_X(Xtrain), std::move(Ytrain), ncomp, fit_scaling, false,
            rsvd_oversample, rsvd_power, svds_tol, seed + f, xprod_precision
          );
        } else {
          model = pls_model2_fast(
            fit_input_X(Xtrain), std::move(Ytrain), ncomp, fit_scaling, false, svd_method,
            rsvd_oversample, rsvd_power, svds_tol, seed + f
          );
        }
      }
    }

    const std::string fit_method_name =
      (fit_method == 1) ? "plssvd" : ((fit_method == 2) ? "simpls" : "simpls_fast");
    model["pls_method"] = fit_method_name;
    model["predict_latent_ok"] = true;

    if (classification && !store_score_predictions) {
      arma::imat fold_class_pred;
      if (classifier == 1) {
        int kmax = 0;
        for (arma::uword a = 0; a < ncomp.n_elem; ++a) {
          if (ncomp(a) > kmax) kmax = ncomp(a);
        }
        arma::mat Ttrain = cv_latent_scores(model, Xtrain, kmax, true);
        arma::mat Ttest = cv_latent_scores(model, Xtest, kmax, false);
        Rcpp::IntegerVector y_train_vec(train_idx.n_elem);
        for (arma::uword ii = 0; ii < train_idx.n_elem; ++ii) {
          y_train_vec[static_cast<R_xlen_t>(ii)] = class_label_at_sample(train_idx(ii));
        }
        Rcpp::IntegerVector ncomp_vec(ncomp.n_elem);
        for (arma::uword s = 0; s < ncomp.n_elem; ++s) {
          ncomp_vec[static_cast<R_xlen_t>(s)] = ncomp(s);
        }
        Rcpp::List lda_models = (backend == 1) ?
          lda_train_prefix_cuda(Ttrain, y_train_vec, n_classes, ncomp_vec, lda_ridge) :
          lda_train_prefix_cpp(Ttrain, y_train_vec, n_classes, ncomp_vec, lda_ridge);
        if (backend == 1) {
          fold_class_pred.set_size(test_idx.n_elem, length_ncomp);
          for (int s = 0; s < length_ncomp; ++s) {
            const int kk = ncomp(static_cast<arma::uword>(s));
            Rcpp::List lda_model = lda_models[s];
            arma::mat Ttest_k = Ttest.cols(0, static_cast<arma::uword>(kk) - 1);
            Rcpp::IntegerVector pred = lda_predict_labels_cuda(Ttest_k, lda_model);
            for (R_xlen_t ii = 0; ii < pred.size(); ++ii) {
              fold_class_pred(static_cast<arma::uword>(ii), static_cast<arma::uword>(s)) = pred[ii];
            }
          }
        } else {
          fold_class_pred = cv_lda_predict_prefix_labels_cpp(Ttest, lda_models, ncomp);
        }
      } else if (backend == 2) {
        arma::cube fold_scores = pls_predict_scores_b_metal_cv(model, std::move(Xtest));
        fold_class_pred.set_size(fold_scores.n_rows, fold_scores.n_slices);
        for (arma::uword s = 0; s < fold_scores.n_slices; ++s) {
          if (use_class_codes) {
            arma::ivec fold_class = nearest_code_classes(fold_scores.slice(s), class_codes);
            fold_class_pred.col(s) = fold_class;
          } else {
            for (arma::uword ii = 0; ii < fold_scores.n_rows; ++ii) {
              fold_class_pred(ii, s) =
                static_cast<int>(fold_scores.slice(s).row(ii).index_max()) + 1;
            }
          }
        }
      } else if (use_class_codes) {
        fold_class_pred = (backend == 1) ?
          pls_predict_code_classes_compact_cuda(model, std::move(Xtest), class_codes) :
          pls_predict_code_classes_compact_cpu(model, std::move(Xtest), class_codes);
      } else {
        fold_class_pred = (backend == 1) ?
          pls_predict_classes_compact_cuda(model, std::move(Xtest)) :
          pls_predict_classes_compact_cpu(model, std::move(Xtest));
      }
      const int ncopy = std::min(length_ncomp, static_cast<int>(fold_class_pred.n_cols));
      for (int s = 0; s < ncopy; ++s) {
        for (arma::uword ii = 0; ii < test_idx.n_elem; ++ii) {
          const int pred_class = fold_class_pred(ii, s);
          if (store_class_predictions) {
            class_pred(test_idx(ii), s) = pred_class;
          }
          metric_correct(s) += (pred_class == class_label_at_sample(test_idx(ii))) ? 1.0 : 0.0;
          metric_total(s) += 1.0;
        }
      }
    } else {
      auto consume_fold_scores = [&](const int s, const arma::mat& scores) {
        if (classification) {
          if (use_class_codes) {
            arma::ivec fold_class = nearest_code_classes(scores, class_codes);
            for (arma::uword ii = 0; ii < test_idx.n_elem; ++ii) {
              const int pred_class = fold_class(ii);
              if (store_class_predictions) {
                class_pred(test_idx(ii), s) = pred_class;
              }
              metric_correct(s) += (pred_class == class_label_at_sample(test_idx(ii))) ? 1.0 : 0.0;
              metric_total(s) += 1.0;
            }
          } else {
            for (arma::uword ii = 0; ii < test_idx.n_elem; ++ii) {
              arma::uword best = 0;
              double best_val = scores(ii, 0);
              for (arma::uword c = 1; c < scores.n_cols; ++c) {
                if (scores(ii, c) > best_val) {
                  best_val = scores(ii, c);
                  best = c;
                }
              }
              const int pred_class = static_cast<int>(best) + 1;
              if (store_class_predictions) {
                class_pred(test_idx(ii), s) = pred_class;
              }
              metric_correct(s) += (pred_class == class_label_at_sample(test_idx(ii))) ? 1.0 : 0.0;
              metric_total(s) += 1.0;
            }
          }
          if (store_score_predictions) {
            Ypred.slice(s).rows(test_idx) = scores;
          }
        } else {
          arma::mat diff = scores - Ydata.rows(test_idx);
          metric_sse(s) += arma::accu(arma::square(diff));
          metric_count(s) += static_cast<double>(diff.n_elem);
          if (store_score_predictions) {
            Ypred.slice(s).rows(test_idx) = scores;
          }
        }
      };

      bool incremental_simpls_done = false;
      if (fit_method != 1 &&
          model.containsElementNamed("R") &&
          model.containsElementNamed("Q") &&
          model.containsElementNamed("mX") &&
          model.containsElementNamed("vX") &&
          model.containsElementNamed("mY")) {
        arma::mat RR = Rcpp::as<arma::mat>(model["R"]);
        arma::mat QQ = Rcpp::as<arma::mat>(model["Q"]);
        arma::rowvec mX = Rcpp::as<arma::rowvec>(model["mX"]);
        arma::rowvec vX = Rcpp::as<arma::rowvec>(model["vX"]);
        arma::rowvec mY = Rcpp::as<arma::rowvec>(model["mY"]);
        int max_requested = 0;
        bool ncomp_ok = true;
        for (arma::uword a = 0; a < ncomp.n_elem; ++a) {
          if (ncomp(a) < 1) ncomp_ok = false;
          if (ncomp(a) > max_requested) max_requested = ncomp(a);
        }
        const int kcap = std::min(static_cast<int>(RR.n_cols), static_cast<int>(QQ.n_cols));
        if (ncomp_ok &&
            max_requested >= 1 &&
            max_requested <= kcap &&
            RR.n_rows == Xtest.n_cols &&
            QQ.n_rows == static_cast<arma::uword>(ncolY) &&
            mX.n_elem == Xtest.n_cols &&
            vX.n_elem == Xtest.n_cols &&
            mY.n_elem == QQ.n_rows) {
          arma::mat Xscaled = Xtest;
          Xscaled.each_row() -= mX;
          Xscaled.each_row() /= vX;
          arma::mat T = Xscaled * RR.cols(0, static_cast<arma::uword>(max_requested) - 1);
          arma::mat accumulated(Xtest.n_rows, QQ.n_rows, arma::fill::zeros);
          arma::vec ncomp_order_key = arma::conv_to<arma::vec>::from(ncomp);
          arma::uvec order = arma::sort_index(ncomp_order_key);
          int previous_components = 0;
          for (arma::uword ord_i = 0; ord_i < order.n_elem; ++ord_i) {
            const int s = static_cast<int>(order(ord_i));
            const int mc = ncomp(static_cast<arma::uword>(s));
            for (int comp = previous_components; comp < mc; ++comp) {
              accumulated +=
                T.col(static_cast<arma::uword>(comp)) *
                QQ.col(static_cast<arma::uword>(comp)).t();
            }
            previous_components = mc;
            arma::mat scores = accumulated;
            scores.each_row() += mY;
            consume_fold_scores(s, scores);
          }
          incremental_simpls_done = true;
        }
      }

      if (!incremental_simpls_done) {
        arma::cube fold_pred;
        if (backend == 2) {
          fold_pred = pls_predict_scores_b_metal_cv(model, std::move(Xtest));
        } else {
          List pred = (backend == 1) ?
            pls_predict_flash_cuda(model, std::move(Xtest), false) :
            pls_predict(model, std::move(Xtest), false);
          fold_pred = Rcpp::as<arma::cube>(pred["Ypred"]);
        }
        const int ncopy = std::min(length_ncomp, static_cast<int>(fold_pred.n_slices));
        for (int s = 0; s < ncopy; ++s) {
          consume_fold_scores(s, fold_pred.slice(static_cast<arma::uword>(s)));
        }
      }
    }
    status(f) = 1; // ok
  }

  const char* classifier_name = (classifier == 1) ? "lda" : "argmax";
  const char* prediction_backend =
    (classifier == 1 && backend == 1) ? "cuda_lda_cv" :
    (classifier == 1 ? "cpp_lda_cv" :
    (backend == 1 ? "cuda_flash" : (backend == 2 ? "metal" : "cpu")));

  List out = List::create(
    Named("fold") = fold + 1,
    Named("status") = status,
    Named("ncomp") = ncomp,
    Named("method") = method_name,
    Named("backend") = (backend == 1 ? "cuda" : (backend == 2 ? "metal" : "cpp")),
    Named("prediction_backend") = prediction_backend,
    Named("classifier") = classifier_name,
    Named("xprod") = xprod,
    Named("stratified_folds") = classification,
    Named("score_predictions_stored") = store_score_predictions
  );
  CharacterVector metric_name(length_ncomp);
  NumericVector metric_value(length_ncomp);
  IntegerVector metric_index(length_ncomp);
  for (int s = 0; s < length_ncomp; ++s) {
    metric_index[s] = s + 1;
    if (classification) {
      metric_name[s] = "accuracy";
      metric_value[s] = (metric_total(s) > 0.0) ?
        (metric_correct(s) / metric_total(s)) :
        NA_REAL;
    } else if (metric_id == 2 || metric_id == 3) {
      metric_name[s] = (metric_id == 2) ? "r2" : "q2";
      metric_value[s] = (std::isfinite(metric_tss) && metric_tss > 0.0) ?
        (1.0 - metric_sse(s) / metric_tss) :
        NA_REAL;
    } else {
      metric_name[s] = "rmsd";
      metric_value[s] = (metric_count(s) > 0.0) ?
        std::sqrt(metric_sse(s) / metric_count(s)) :
        NA_REAL;
    }
  }
  out["metrics"] = DataFrame::create(
    Named("ncomp_index") = metric_index,
    Named("metric_name") = metric_name,
    Named("metric_value") = metric_value,
    Named("stringsAsFactors") = false
  );
  if (store_score_predictions) {
    out["Ypred"] = Ypred;
  }
  if (store_class_predictions) {
    out["class_pred"] = class_pred;
  }
  return out;
}
