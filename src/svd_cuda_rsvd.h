#ifndef FASTPLS_SVD_CUDA_RSVD_H
#define FASTPLS_SVD_CUDA_RSVD_H

#include "svd_iface.h"

namespace fastpls_svd {

#ifdef FASTPLS_HAS_CUDA
SVDResult truncated_svd_cuda_rsvd(const Mat& A, int k, const SVDOptions& opt);
bool cuda_runtime_available();
void cuda_rsvd_sample_y(
  const double* hA,
  int m,
  int n,
  const double* hOmega,
  int l,
  int power_iters,
  double* hY
);
#endif

} // namespace fastpls_svd

#endif
