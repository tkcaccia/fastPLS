#ifndef FASTPLS_SVD_CUDA_RSVD_H
#define FASTPLS_SVD_CUDA_RSVD_H

#include "svd_iface.h"

namespace fastpls_svd {

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
bool cuda_rsvd_prefer_block_gpu(int m, int n, int l, int power_iters);
void cuda_rsvd_set_resident_matrix(
  const double* hA,
  int m,
  int n
);
void cuda_rsvd_refresh_left_block(
  const double* hA,
  int m,
  int n,
  const double* hY0,
  int l,
  int power_iters,
  double* hY
);
void cuda_rsvd_refresh_left_block_u(
  const double* hA,
  int m,
  int n,
  const double* hY0,
  int l,
  int k,
  int power_iters,
  double* hUblock
);
void cuda_rsvd_refresh_left_block_u_resident(
  int m,
  int n,
  const double* hY0,
  int l,
  int k,
  unsigned int seed,
  int power_iters,
  double* hUblock
);
void cuda_rsvd_project_left_row(
  const double* hV,
  int m,
  int n,
  double* hVS
);
void cuda_rsvd_deflate_left_rank1(
  const double* hV,
  const double* hVS,
  int m,
  int n
);
void cuda_simpls_fast_set_training_matrices(
  const double* hX,
  int n,
  int p,
  const double* hY,
  int m,
  bool fit
);
void cuda_simpls_fast_component_stats(
  const double* hR,
  int n,
  int p,
  int m,
  double* hT,
  double* hP,
  double* hQ,
  double* hTnorm
);
void cuda_simpls_fast_rank1_fit_update(
  const double* hT,
  int n,
  const double* hQ,
  int m,
  double* hDelta
);

} // namespace fastpls_svd

#endif
