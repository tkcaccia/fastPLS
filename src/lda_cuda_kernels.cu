#include <cuda_runtime.h>
#include <math.h>

extern "C" {

__global__ void fastpls_lda_means_kernel(double* means,
                                         const double* counts,
                                         int kmax,
                                         int n_classes) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kmax * n_classes;
  if (idx >= total) return;
  const int cls = idx % n_classes;
  const double cnt = counts[cls];
  means[idx] = (cnt > 0.0) ? means[idx] / cnt : 0.0;
}

__global__ void fastpls_lda_label_sums_kernel(const double* T,
                                              const int* y,
                                              int n,
                                              int kmax,
                                              int n_classes,
                                              double* sums) {
  const int cls = blockIdx.x;
  const int j = blockIdx.y;
  if (cls >= n_classes || j >= kmax) return;

  extern __shared__ double partial[];
  double value = 0.0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    if (y[i] == cls + 1) {
      value += T[i + j * n];
    }
  }
  partial[threadIdx.x] = value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    sums[cls + j * n_classes] = partial[0];
  }
}

__global__ void fastpls_lda_subtract_offsets_kernel(double* T,
                                                    const double* offsets,
                                                    int n,
                                                    int kmax) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n * kmax;
  if (idx >= total) return;
  const int j = idx / n;
  T[idx] -= offsets[j];
}

__global__ void fastpls_lda_pooled_kernel(double* pooled,
                                          const double* means,
                                          const double* counts,
                                          int n,
                                          int kmax,
                                          int n_classes) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kmax * kmax;
  if (idx >= total) return;
  const int r = idx % kmax;
  const int c = idx / kmax;
  double between = 0.0;
  for (int cls = 0; cls < n_classes; ++cls) {
    between += counts[cls] * means[cls + r * n_classes] * means[cls + c * n_classes];
  }
  const double df = fmax(1.0, static_cast<double>(n - n_classes));
  pooled[idx] = (pooled[idx] - between) / df;
}

__global__ void fastpls_lda_copy_cov_kernel(const double* pooled,
                                            double* cov,
                                            int kmax,
                                            int kk) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kk * kk;
  if (idx >= total) return;
  const int r = idx % kk;
  const int c = idx / kk;
  cov[idx] = pooled[r + c * kmax];
}

__global__ void fastpls_lda_add_ridge_kernel(double* cov,
                                             int kk,
                                             double ridge,
                                             double* lambda_out) {
  __shared__ double trace;
  if (threadIdx.x == 0) trace = 0.0;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 0; i < kk; ++i) {
      trace += cov[i + i * kk];
    }
  }
  __syncthreads();
  const double scale = isfinite(trace) && trace > 0.0 ? trace / static_cast<double>(kk) : 1.0;
  const double lambda = (isfinite(ridge) && ridge >= 0.0 ? ridge : 1e-8) * scale;
  if (threadIdx.x == 0) *lambda_out = lambda;
  for (int i = threadIdx.x; i < kk; i += blockDim.x) {
    cov[i + i * kk] += lambda;
  }
}

__global__ void fastpls_lda_means_to_rhs_kernel(const double* means,
                                                double* rhs,
                                                int kmax,
                                                int kk,
                                                int n_classes) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kk * n_classes;
  if (idx >= total) return;
  const int j = idx % kk;
  const int cls = idx / kk;
  rhs[j + cls * kk] = means[cls + j * n_classes];
}

__global__ void fastpls_lda_finalize_linear_kernel(const double* rhs,
                                                  const double* means,
                                                  const double* counts,
                                                  double* linear,
                                                  double* constants,
                                                  int n,
                                                  int kmax,
                                                  int kk,
                                                  int n_classes) {
  const int cls = blockIdx.x * blockDim.x + threadIdx.x;
  if (cls >= n_classes) return;
  double dot = 0.0;
  for (int j = 0; j < kk; ++j) {
    const double value = rhs[j + cls * kk];
    linear[cls + j * n_classes] = value;
    dot += means[cls + j * n_classes] * value;
  }
  const double prior = fmax(counts[cls] / static_cast<double>(n), 2.2250738585072014e-308);
  constants[cls] = -0.5 * dot + log(prior);
}

__global__ void fastpls_lda_score_argmax_kernel(double* scores,
                                                const double* constants,
                                                int* pred,
                                                int n,
                                                int n_classes) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  int best = 0;
  double best_value = scores[row] + constants[0];
  scores[row] = best_value;
  for (int cls = 1; cls < n_classes; ++cls) {
    const int offset = row + cls * n;
    const double value = scores[offset] + constants[cls];
    scores[offset] = value;
    if (value > best_value) {
      best_value = value;
      best = cls;
    }
  }
  pred[row] = best + 1;
}

__device__ void fastpls_candidate_insert(double value, double* top_vals, int top_k) {
  for (int j = 0; j < top_k; ++j) {
    if (value > top_vals[j]) {
      for (int h = top_k - 1; h > j; --h) {
        top_vals[h] = top_vals[h - 1];
      }
      top_vals[j] = value;
      return;
    }
  }
}

__global__ void fastpls_candidate_knn_scores_kernel(const double* Ttest,
                                                    const double* Ttrain,
                                                    const int* y,
                                                    const int* candidates,
                                                    const double* candidate_base,
                                                    const double* bias,
                                                    int ntest,
                                                    int ntrain,
                                                    int kdim,
                                                    int n_classes,
                                                    int top_m,
                                                    int knn_k,
                                                    double tau,
                                                    double alpha,
                                                    double* out_scores) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = ntest * top_m;
  if (idx >= total) return;
  const int row = idx % ntest;
  const int cand_slot = idx / ntest;
  const int cls = candidates[row + cand_slot * ntest];
  if (cls < 1 || cls > n_classes) {
    out_scores[idx] = -INFINITY;
    return;
  }

  const int use_k = max(1, min(knn_k, 32));
  double top_vals[32];
  for (int j = 0; j < use_k; ++j) top_vals[j] = -INFINITY;
  int found = 0;

  for (int tr = 0; tr < ntrain; ++tr) {
    if (y[tr] != cls) continue;
    double dot = 0.0;
    for (int d = 0; d < kdim; ++d) {
      dot += Ttest[row + d * ntest] * Ttrain[tr + d * ntrain];
    }
    fastpls_candidate_insert(dot, top_vals, use_k);
    ++found;
  }

  if (found < 1 || !isfinite(top_vals[0])) {
    out_scores[idx] = -INFINITY;
    return;
  }
  const int denom = max(1, min(use_k, found));
  double local = 0.0;
  if (!isfinite(tau) || tau <= 0.0) {
    for (int j = 0; j < denom; ++j) local += top_vals[j];
    local /= static_cast<double>(denom);
  } else {
    const double mx = top_vals[0];
    double acc = 0.0;
    for (int j = 0; j < denom; ++j) {
      acc += exp((top_vals[j] - mx) / tau);
    }
    local = mx + tau * log(acc / static_cast<double>(denom));
  }
  out_scores[idx] = local + alpha * candidate_base[idx] + bias[cls - 1];
}

void fastpls_cuda_lda_means(double* means,
                            const double* counts,
                            int kmax,
                            int n_classes,
                            cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (kmax * n_classes + threads - 1) / threads;
  fastpls_lda_means_kernel<<<blocks, threads, 0, stream>>>(means, counts, kmax, n_classes);
}

void fastpls_cuda_lda_label_sums(const double* T,
                                 const int* y,
                                 int n,
                                 int kmax,
                                 int n_classes,
                                 double* sums,
                                 cudaStream_t stream) {
  const int threads = 256;
  const dim3 blocks(n_classes, kmax);
  const size_t shared = sizeof(double) * static_cast<size_t>(threads);
  fastpls_lda_label_sums_kernel<<<blocks, threads, shared, stream>>>(T, y, n, kmax, n_classes, sums);
}

void fastpls_cuda_lda_subtract_offsets(double* T,
                                       const double* offsets,
                                       int n,
                                       int kmax,
                                       cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (n * kmax + threads - 1) / threads;
  fastpls_lda_subtract_offsets_kernel<<<blocks, threads, 0, stream>>>(T, offsets, n, kmax);
}

void fastpls_cuda_lda_pooled(double* pooled,
                             const double* means,
                             const double* counts,
                             int n,
                             int kmax,
                             int n_classes,
                             cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (kmax * kmax + threads - 1) / threads;
  fastpls_lda_pooled_kernel<<<blocks, threads, 0, stream>>>(pooled, means, counts, n, kmax, n_classes);
}

void fastpls_cuda_lda_copy_cov(const double* pooled,
                               double* cov,
                               int kmax,
                               int kk,
                               cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (kk * kk + threads - 1) / threads;
  fastpls_lda_copy_cov_kernel<<<blocks, threads, 0, stream>>>(pooled, cov, kmax, kk);
}

void fastpls_cuda_lda_add_ridge(double* cov,
                                int kk,
                                double ridge,
                                double* lambda_out,
                                cudaStream_t stream) {
  fastpls_lda_add_ridge_kernel<<<1, 256, 0, stream>>>(cov, kk, ridge, lambda_out);
}

void fastpls_cuda_lda_means_to_rhs(const double* means,
                                   double* rhs,
                                   int kmax,
                                   int kk,
                                   int n_classes,
                                   cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (kk * n_classes + threads - 1) / threads;
  fastpls_lda_means_to_rhs_kernel<<<blocks, threads, 0, stream>>>(means, rhs, kmax, kk, n_classes);
}

void fastpls_cuda_lda_finalize_linear(const double* rhs,
                                      const double* means,
                                      const double* counts,
                                      double* linear,
                                      double* constants,
                                      int n,
                                      int kmax,
                                      int kk,
                                      int n_classes,
                                      cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (n_classes + threads - 1) / threads;
  fastpls_lda_finalize_linear_kernel<<<blocks, threads, 0, stream>>>(
    rhs, means, counts, linear, constants, n, kmax, kk, n_classes);
}

void fastpls_cuda_lda_score_argmax(double* scores,
                                   const double* constants,
                                   int* pred,
                                   int n,
                                   int n_classes,
                                   cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  fastpls_lda_score_argmax_kernel<<<blocks, threads, 0, stream>>>(scores, constants, pred, n, n_classes);
}

void fastpls_cuda_candidate_knn_scores(const double* Ttest,
                                       const double* Ttrain,
                                       const int* y,
                                       const int* candidates,
                                       const double* candidate_base,
                                       const double* bias,
                                       int ntest,
                                       int ntrain,
                                       int kdim,
                                       int n_classes,
                                       int top_m,
                                       int knn_k,
                                       double tau,
                                       double alpha,
                                       double* out_scores,
                                       cudaStream_t stream) {
  const int threads = 128;
  const int blocks = (ntest * top_m + threads - 1) / threads;
  fastpls_candidate_knn_scores_kernel<<<blocks, threads, 0, stream>>>(
    Ttest,
    Ttrain,
    y,
    candidates,
    candidate_base,
    bias,
    ntest,
    ntrain,
    kdim,
    n_classes,
    top_m,
    knn_k,
    tau,
    alpha,
    out_scores
  );
}

}
