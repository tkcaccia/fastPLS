#ifndef FASTPLS_CPP_FASTPLS_HPP
#define FASTPLS_CPP_FASTPLS_HPP

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fastpls {

template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

enum class Kernel {
  Linear,
  Rbf,
  Polynomial
};

struct RsvdOptions {
  int oversample = 32;
  int power_iters = 5;
  std::uint32_t seed = 1;
};

template <typename T>
struct Preprocess {
  Vector<T> x_mean;
  Vector<T> x_scale;
  Vector<T> y_mean;
};

template <typename T>
struct SvdResult {
  Matrix<T> U;
  Vector<T> s;
  Matrix<T> Vt;
};

template <typename T>
struct PlsModel {
  std::string method;
  int ncomp = 0;
  Matrix<T> R;
  Matrix<T> P;
  Matrix<T> Q;
  Matrix<T> B;
  Matrix<T> scores;
  Preprocess<T> prep;
};

template <typename T>
struct OplsModel {
  PlsModel<T> inner;
  Matrix<T> W_orth;
  Matrix<T> P_orth;
  int north = 0;
};

template <typename T>
struct KernelPlsModel {
  PlsModel<T> inner;
  Kernel kernel = Kernel::Linear;
  T gamma = T(1);
  int degree = 3;
  T coef0 = T(1);
  Matrix<T> X_train_centered;
  Vector<T> x_mean;
  Vector<T> kernel_mean_cols;
  T kernel_mean_all = T(0);
};

namespace detail {

template <typename T>
inline void check_matrix(const Matrix<T>& X, const char* name) {
  if (X.rows() < 1 || X.cols() < 1) {
    throw std::invalid_argument(std::string(name) + " must be non-empty");
  }
  if (!X.allFinite()) {
    throw std::invalid_argument(std::string(name) + " contains non-finite values");
  }
}

template <typename T>
inline Matrix<T> center_columns(const Matrix<T>& X, Vector<T>* mean_out) {
  Matrix<T> out = X;
  *mean_out = out.colwise().mean().transpose();
  out.rowwise() -= mean_out->transpose();
  return out;
}

template <typename T>
inline Vector<T> ones_scale(int p) {
  return Vector<T>::Ones(p);
}

template <typename T>
inline Matrix<T> gaussian_matrix(int rows, int cols, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<T> normal(T(0), T(1));
  Matrix<T> out(rows, cols);
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      out(i, j) = normal(rng);
    }
  }
  return out;
}

template <typename T>
inline Matrix<T> orthonormalize(const Matrix<T>& A) {
  Eigen::HouseholderQR<Matrix<T>> qr(A);
  const int cols = A.cols();
  Matrix<T> Q = qr.householderQ() * Matrix<T>::Identity(A.rows(), cols);
  return Q.leftCols(cols);
}

template <typename T>
inline int capped_rank(const Matrix<T>& A, int k) {
  return std::max(1, std::min<int>(k, std::min(A.rows(), A.cols())));
}

template <typename T>
inline Matrix<T> subcols_or_empty(const Matrix<T>& X, int k) {
  if (k <= 0) return Matrix<T>(X.rows(), 0);
  return X.leftCols(std::min(k, static_cast<int>(X.cols())));
}

template <typename T>
inline T safe_norm2(const Vector<T>& x) {
  const T n = x.norm();
  return n > std::numeric_limits<T>::epsilon() ? n : T(1);
}

} // namespace detail

template <typename T>
inline SvdResult<T> exact_svd(const Matrix<T>& A, int k) {
  detail::check_matrix(A, "A");
  k = detail::capped_rank(A, k);
  Eigen::BDCSVD<Matrix<T>> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  SvdResult<T> out;
  out.U = svd.matrixU().leftCols(k);
  out.s = svd.singularValues().head(k);
  out.Vt = svd.matrixV().leftCols(k).transpose();
  return out;
}

template <typename T>
inline SvdResult<T> rsvd(const Matrix<T>& A, int k, RsvdOptions opt = {}) {
  detail::check_matrix(A, "A");
  k = detail::capped_rank(A, k);
  const int min_dim = std::min(A.rows(), A.cols());
  const int sketch_dim = std::min(min_dim, k + std::max(0, opt.oversample));
  if (min_dim < 6 || sketch_dim >= min_dim) {
    return exact_svd(A, k);
  }

  Matrix<T> Omega = detail::gaussian_matrix<T>(A.cols(), sketch_dim, opt.seed);
  Matrix<T> Y = A * Omega;
  for (int i = 0; i < std::max(0, opt.power_iters); ++i) {
    Matrix<T> Z = A.transpose() * detail::orthonormalize(Y);
    Y = A * detail::orthonormalize(Z);
  }

  Matrix<T> Q = detail::orthonormalize(Y);
  Matrix<T> B = Q.transpose() * A;
  SvdResult<T> small = exact_svd(B, k);

  SvdResult<T> out;
  out.U = Q * small.U;
  out.s = small.s;
  out.Vt = small.Vt;
  return out;
}

template <typename T>
inline PlsModel<T> plssvd(const Matrix<T>& X, const Matrix<T>& Y, int ncomp,
                          RsvdOptions opt = {}) {
  detail::check_matrix(X, "X");
  detail::check_matrix(Y, "Y");
  if (X.rows() != Y.rows()) {
    throw std::invalid_argument("X and Y must have the same number of rows");
  }

  PlsModel<T> model;
  model.method = "plssvd";
  model.ncomp = detail::capped_rank(Matrix<T>(X.cols(), Y.cols()), ncomp);
  Matrix<T> Xc = detail::center_columns(X, &model.prep.x_mean);
  Matrix<T> Yc = detail::center_columns(Y, &model.prep.y_mean);
  model.prep.x_scale = detail::ones_scale<T>(X.cols());

  Matrix<T> S = Xc.transpose() * Yc;
  SvdResult<T> sv = rsvd(S, model.ncomp, opt);
  model.R = sv.U;
  model.Q = Matrix<T>(Y.cols(), model.ncomp);
  model.P = Matrix<T>(X.cols(), model.ncomp);
  model.scores = Xc * model.R;
  for (int a = 0; a < model.ncomp; ++a) {
    Vector<T> t = model.scores.col(a);
    const T denom = std::max<T>(t.squaredNorm(), std::numeric_limits<T>::epsilon());
    model.P.col(a) = (Xc.transpose() * t) / denom;
    model.Q.col(a) = (Yc.transpose() * t) / denom;
  }
  model.B = model.R * model.Q.transpose();
  return model;
}

template <typename T>
inline PlsModel<T> simpls(const Matrix<T>& X, const Matrix<T>& Y, int ncomp,
                          RsvdOptions opt = {}) {
  detail::check_matrix(X, "X");
  detail::check_matrix(Y, "Y");
  if (X.rows() != Y.rows()) {
    throw std::invalid_argument("X and Y must have the same number of rows");
  }

  PlsModel<T> model;
  model.method = "simpls";
  model.ncomp = std::max(1, std::min<int>(ncomp, std::min(X.cols(), X.rows() - 1)));
  Matrix<T> Xc = detail::center_columns(X, &model.prep.x_mean);
  Matrix<T> Yc = detail::center_columns(Y, &model.prep.y_mean);
  model.prep.x_scale = detail::ones_scale<T>(X.cols());

  Matrix<T> S = Xc.transpose() * Yc;
  Matrix<T> V(X.cols(), model.ncomp);
  model.R.resize(X.cols(), model.ncomp);
  model.P.resize(X.cols(), model.ncomp);
  model.Q.resize(Y.cols(), model.ncomp);
  model.scores.resize(X.rows(), model.ncomp);

  for (int a = 0; a < model.ncomp; ++a) {
    SvdResult<T> sv = rsvd(S, 1, RsvdOptions{opt.oversample, opt.power_iters, opt.seed + static_cast<std::uint32_t>(a)});
    Vector<T> r = sv.U.col(0);
    Vector<T> t = Xc * r;
    const T tnorm = detail::safe_norm2(t);
    t /= tnorm;
    r /= tnorm;

    Vector<T> p = Xc.transpose() * t;
    Vector<T> q = Yc.transpose() * t;
    Vector<T> v = p;
    for (int j = 0; j < a; ++j) {
      v -= V.col(j) * (V.col(j).dot(p));
    }
    const T vnorm = detail::safe_norm2(v);
    v /= vnorm;
    S -= v * (v.transpose() * S);

    V.col(a) = v;
    model.R.col(a) = r;
    model.P.col(a) = p;
    model.Q.col(a) = q;
    model.scores.col(a) = t;
  }

  model.B = model.R * model.Q.transpose();
  return model;
}

template <typename T>
inline Matrix<T> predict(const PlsModel<T>& model, const Matrix<T>& X) {
  detail::check_matrix(X, "X");
  if (X.cols() != model.prep.x_mean.size()) {
    throw std::invalid_argument("X has incompatible number of columns");
  }
  Matrix<T> Xc = X;
  Xc.rowwise() -= model.prep.x_mean.transpose();
  Matrix<T> Yhat = Xc * model.B;
  Yhat.rowwise() += model.prep.y_mean.transpose();
  return Yhat;
}

template <typename T>
inline OplsModel<T> opls(const Matrix<T>& X, const Matrix<T>& Y, int ncomp,
                         int north = 1, RsvdOptions opt = {}) {
  detail::check_matrix(X, "X");
  detail::check_matrix(Y, "Y");
  if (north < 0) {
    throw std::invalid_argument("north must be non-negative");
  }

  Matrix<T> X_work = X;
  OplsModel<T> out;
  out.north = north;
  out.W_orth.resize(X.cols(), north);
  out.P_orth.resize(X.cols(), north);

  for (int h = 0; h < north; ++h) {
    PlsModel<T> pred = simpls(X_work, Y, 1, opt);
    Matrix<T> Xc = X_work;
    Xc.rowwise() -= pred.prep.x_mean.transpose();
    Vector<T> t_pred = Xc * pred.R.col(0);
    Vector<T> w_orth = Xc.transpose() * t_pred;
    w_orth -= pred.R.col(0) * pred.R.col(0).dot(w_orth);
    w_orth /= detail::safe_norm2(w_orth);
    Vector<T> t_orth = Xc * w_orth;
    const T denom = std::max<T>(t_orth.squaredNorm(), std::numeric_limits<T>::epsilon());
    Vector<T> p_orth = (Xc.transpose() * t_orth) / denom;
    X_work = Xc - t_orth * p_orth.transpose();
    out.W_orth.col(h) = w_orth;
    out.P_orth.col(h) = p_orth;
  }

  out.inner = simpls(X_work, Y, ncomp, opt);
  out.inner.method = "opls";
  return out;
}

template <typename T>
inline Matrix<T> linear_kernel(const Matrix<T>& A, const Matrix<T>& B) {
  return A * B.transpose();
}

template <typename T>
inline Matrix<T> rbf_kernel(const Matrix<T>& A, const Matrix<T>& B, T gamma) {
  Matrix<T> K(A.rows(), B.rows());
  for (int i = 0; i < A.rows(); ++i) {
    for (int j = 0; j < B.rows(); ++j) {
      K(i, j) = std::exp(-gamma * (A.row(i) - B.row(j)).squaredNorm());
    }
  }
  return K;
}

template <typename T>
inline Matrix<T> polynomial_kernel(const Matrix<T>& A, const Matrix<T>& B, T gamma, int degree, T coef0) {
  return ((gamma * (A * B.transpose())).array() + coef0).pow(degree).matrix();
}

template <typename T>
inline Matrix<T> kernel_matrix(const Matrix<T>& A, const Matrix<T>& B, Kernel kernel,
                               T gamma, int degree, T coef0) {
  switch (kernel) {
    case Kernel::Linear:
      return linear_kernel(A, B);
    case Kernel::Rbf:
      return rbf_kernel(A, B, gamma);
    case Kernel::Polynomial:
      return polynomial_kernel(A, B, gamma, degree, coef0);
  }
  throw std::invalid_argument("Unsupported kernel");
}

template <typename T>
inline KernelPlsModel<T> kernelpls(const Matrix<T>& X, const Matrix<T>& Y, int ncomp,
                                   Kernel kernel = Kernel::Linear,
                                   T gamma = T(-1),
                                   int degree = 3,
                                   T coef0 = T(1),
                                   RsvdOptions opt = {}) {
  detail::check_matrix(X, "X");
  detail::check_matrix(Y, "Y");
  KernelPlsModel<T> model;
  model.kernel = kernel;
  model.gamma = gamma > T(0) ? gamma : T(1) / static_cast<T>(X.cols());
  model.degree = degree;
  model.coef0 = coef0;
  model.X_train_centered = detail::center_columns(X, &model.x_mean);
  Matrix<T> K = kernel_matrix(model.X_train_centered, model.X_train_centered, kernel, model.gamma, degree, coef0);
  model.kernel_mean_cols = K.colwise().mean().transpose();
  model.kernel_mean_all = K.mean();
  K.rowwise() -= model.kernel_mean_cols.transpose();
  K.colwise() -= model.kernel_mean_cols;
  K.array() += model.kernel_mean_all;
  model.inner = simpls(K, Y, ncomp, opt);
  model.inner.method = "kernelpls";
  return model;
}

template <typename T>
inline Matrix<T> predict(const KernelPlsModel<T>& model, const Matrix<T>& X) {
  Matrix<T> Xc = X;
  Xc.rowwise() -= model.x_mean.transpose();
  Matrix<T> K = kernel_matrix(Xc, model.X_train_centered, model.kernel, model.gamma, model.degree, model.coef0);
  K.rowwise() -= model.kernel_mean_cols.transpose();
  Vector<T> row_means = K.rowwise().mean();
  K.colwise() -= row_means;
  K.array() += model.kernel_mean_all;
  return predict(model.inner, K);
}

template <typename T>
inline T rmse(const Matrix<T>& observed, const Matrix<T>& predicted) {
  if (observed.rows() != predicted.rows() || observed.cols() != predicted.cols()) {
    throw std::invalid_argument("observed and predicted dimensions differ");
  }
  return std::sqrt((observed - predicted).array().square().mean());
}

} // namespace fastpls

#endif
