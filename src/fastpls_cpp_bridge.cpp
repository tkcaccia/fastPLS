// [[Rcpp::depends(RcppEigen)]]
#include <Rcpp.h>
#include <fastpls/fastpls.hpp>

namespace {

template <typename T>
fastpls::Matrix<T> r_to_fastpls_matrix(SEXP x, const char* name) {
  if (!Rf_isMatrix(x)) {
    Rcpp::stop("%s must be a matrix", name);
  }
  Rcpp::NumericMatrix xr(x);
  fastpls::Matrix<T> out(xr.nrow(), xr.ncol());
  for (int j = 0; j < xr.ncol(); ++j) {
    for (int i = 0; i < xr.nrow(); ++i) {
      out(i, j) = static_cast<T>(xr(i, j));
    }
  }
  return out;
}

template <typename T>
Rcpp::NumericMatrix fastpls_to_r_matrix(const fastpls::Matrix<T>& x) {
  Rcpp::NumericMatrix out(x.rows(), x.cols());
  for (int j = 0; j < x.cols(); ++j) {
    for (int i = 0; i < x.rows(); ++i) {
      out(i, j) = static_cast<double>(x(i, j));
    }
  }
  return out;
}

template <typename T>
Rcpp::NumericVector fastpls_to_r_vector(const fastpls::Vector<T>& x) {
  Rcpp::NumericVector out(x.size());
  for (int i = 0; i < x.size(); ++i) {
    out[i] = static_cast<double>(x(i));
  }
  return out;
}

fastpls::RsvdOptions bridge_options(int oversample, int power, int seed) {
  fastpls::RsvdOptions opt;
  opt.oversample = std::max(0, oversample);
  opt.power_iters = std::max(0, power);
  opt.seed = static_cast<std::uint32_t>(std::max(0, seed));
  return opt;
}

template <typename T>
Rcpp::List model_to_list(const fastpls::PlsModel<T>& model,
                         const fastpls::Matrix<T>& prediction) {
  return Rcpp::List::create(
    Rcpp::Named("method") = model.method,
    Rcpp::Named("ncomp") = model.ncomp,
    Rcpp::Named("R") = fastpls_to_r_matrix(model.R),
    Rcpp::Named("P") = fastpls_to_r_matrix(model.P),
    Rcpp::Named("Q") = fastpls_to_r_matrix(model.Q),
    Rcpp::Named("B") = fastpls_to_r_matrix(model.B),
    Rcpp::Named("scores") = fastpls_to_r_matrix(model.scores),
    Rcpp::Named("Ypred_train") = fastpls_to_r_matrix(prediction),
    Rcpp::Named("x_mean") = fastpls_to_r_vector(model.prep.x_mean),
    Rcpp::Named("y_mean") = fastpls_to_r_vector(model.prep.y_mean)
  );
}

} // namespace

// [[Rcpp::export]]
Rcpp::List fastpls_cpp_core_rsvd(SEXP A,
                                 int k,
                                 int oversample = 20,
                                 int power = 2,
                                 int seed = 1,
                                 bool use_float = false) {
  if (use_float) {
    fastpls::Matrix<float> Af = r_to_fastpls_matrix<float>(A, "A");
    auto sv = fastpls::rsvd<float>(Af, k, bridge_options(oversample, power, seed));
    fastpls::Matrix<float> V = sv.Vt.transpose();
    return Rcpp::List::create(
      Rcpp::Named("u") = fastpls_to_r_matrix(sv.U),
      Rcpp::Named("d") = fastpls_to_r_vector(sv.s),
      Rcpp::Named("v") = fastpls_to_r_matrix(V),
      Rcpp::Named("precision") = "float32"
    );
  }

  fastpls::Matrix<double> Ad = r_to_fastpls_matrix<double>(A, "A");
  auto sv = fastpls::rsvd<double>(Ad, k, bridge_options(oversample, power, seed));
  fastpls::Matrix<double> V = sv.Vt.transpose();
  return Rcpp::List::create(
    Rcpp::Named("u") = fastpls_to_r_matrix(sv.U),
    Rcpp::Named("d") = fastpls_to_r_vector(sv.s),
    Rcpp::Named("v") = fastpls_to_r_matrix(V),
    Rcpp::Named("precision") = "double64"
  );
}

// [[Rcpp::export]]
Rcpp::List fastpls_cpp_core_plssvd(SEXP X,
                                   SEXP Y,
                                   int ncomp,
                                   int oversample = 20,
                                   int power = 2,
                                   int seed = 1,
                                   bool use_float = false) {
  if (use_float) {
    fastpls::Matrix<float> Xf = r_to_fastpls_matrix<float>(X, "X");
    fastpls::Matrix<float> Yf = r_to_fastpls_matrix<float>(Y, "Y");
    auto model = fastpls::plssvd<float>(Xf, Yf, ncomp, bridge_options(oversample, power, seed));
    return model_to_list(model, fastpls::predict(model, Xf));
  }

  fastpls::Matrix<double> Xd = r_to_fastpls_matrix<double>(X, "X");
  fastpls::Matrix<double> Yd = r_to_fastpls_matrix<double>(Y, "Y");
  auto model = fastpls::plssvd<double>(Xd, Yd, ncomp, bridge_options(oversample, power, seed));
  return model_to_list(model, fastpls::predict(model, Xd));
}

// [[Rcpp::export]]
Rcpp::List fastpls_cpp_core_simpls(SEXP X,
                                   SEXP Y,
                                   int ncomp,
                                   int oversample = 20,
                                   int power = 2,
                                   int seed = 1,
                                   bool use_float = false) {
  if (use_float) {
    fastpls::Matrix<float> Xf = r_to_fastpls_matrix<float>(X, "X");
    fastpls::Matrix<float> Yf = r_to_fastpls_matrix<float>(Y, "Y");
    auto model = fastpls::simpls<float>(Xf, Yf, ncomp, bridge_options(oversample, power, seed));
    return model_to_list(model, fastpls::predict(model, Xf));
  }

  fastpls::Matrix<double> Xd = r_to_fastpls_matrix<double>(X, "X");
  fastpls::Matrix<double> Yd = r_to_fastpls_matrix<double>(Y, "Y");
  auto model = fastpls::simpls<double>(Xd, Yd, ncomp, bridge_options(oversample, power, seed));
  return model_to_list(model, fastpls::predict(model, Xd));
}
