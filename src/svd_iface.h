#ifndef FASTPLS_SVD_IFACE_H
#define FASTPLS_SVD_IFACE_H

#include <RcppArmadillo.h>

namespace fastpls_svd {

using Mat = arma::mat;
using Vec = arma::vec;

struct SVDResult {
  Mat U;
  Vec s;
  Mat Vt;
};

enum class Backend {
  CPU = 0,
  CUDA = 1,
  BANDICOOT = 2
};

enum class Method {
  EXACT = 0,
  RSVD = 1,
  IRLBA = 2
};

struct SVDOptions {
  Method method = Method::EXACT;
  int oversample = 10;
  int power_iters = 1;
  unsigned int seed = 1;
  double svds_tol = 0.0;
  bool left_only = false;
  bool use_full_svd = false;
};

enum SVDMethodId {
  SVD_METHOD_IRLBA = 1,
  SVD_METHOD_ARPACK = 2,
  SVD_METHOD_DC = SVD_METHOD_ARPACK, // deprecated alias
  SVD_METHOD_CPU_EXACT = 3,
  SVD_METHOD_CPU_RSVD = 4,
  SVD_METHOD_CUDA_RSVD = 5
};

SVDOptions options_from_method_id(
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  unsigned int seed,
  bool left_only,
  bool use_full_svd
);

Backend backend_from_method_id(int svd_method);
bool method_is_legacy_irlba(int svd_method);

SVDResult truncated_svd(const Mat& A, int k, const SVDOptions& opt, Backend backend);

SVDResult truncated_svd_cpu_exact(const Mat& A, int k, const SVDOptions& opt);
SVDResult truncated_svd_cpu_irlba(const Mat& A, int k, const SVDOptions& opt);
SVDResult truncated_svd_cpu_rsvd(const Mat& A, int k, const SVDOptions& opt);

// Shared post-processing for randomized range finder outputs.
SVDResult finalize_rsvd_from_sample(const Mat& A, const Mat& Y, int k, bool left_only);

bool has_cuda_build();
bool has_cuda_backend();

} // namespace fastpls_svd

#endif
