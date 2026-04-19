#include "svd_cuda_rsvd.h"

#ifdef FASTPLS_HAS_CUDA

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <algorithm>

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace fastpls_svd {
namespace {

void throw_cuda_error(cudaError_t code, const char* where) {
  throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(code));
}

void check_cuda(cudaError_t code, const char* where) {
  if (code != cudaSuccess) {
    throw_cuda_error(code, where);
  }
}

void check_cublas(cublasStatus_t code, const char* where) {
  if (code != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(where) + ": cublas call failed");
  }
}

void check_curand(curandStatus_t code, const char* where) {
  if (code != CURAND_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(where) + ": curand call failed");
  }
}

void check_cusolver(cusolverStatus_t code, const char* where) {
  if (code != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(where) + ": cusolver call failed");
  }
}

arma::mat inv_sqrt_psd(const arma::mat& G) {
  arma::vec evals;
  arma::mat evecs;
  if (!arma::eig_sym(evals, evecs, G)) {
    throw std::runtime_error("eig_sym failed in inv_sqrt_psd");
  }
  if (evals.n_elem < 1) {
    throw std::runtime_error("empty eigenspectrum in inv_sqrt_psd");
  }
  const double max_eval = std::max(evals.max(), 0.0);
  const double tol = std::max(max_eval, 1.0) * 1e-10;
  arma::vec scale = arma::zeros<arma::vec>(evals.n_elem);
  for (arma::uword i = 0; i < evals.n_elem; ++i) {
    if (evals(i) > tol) {
      scale(i) = 1.0 / std::sqrt(evals(i));
    }
  }
  return evecs * arma::diagmat(scale) * evecs.t();
}

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

class CudaRSVDWorkspace {
 public:
  CudaRSVDWorkspace() = default;

  ~CudaRSVDWorkspace() {
    release();
  }

  void ensure_capacity(int m, int n, int l) {
    if (!handle_ready_) {
      check_cublas(cublasCreate(&handle_), "cublasCreate");
      handle_ready_ = true;
    }
    if (!solver_ready_) {
      check_cusolver(cusolverDnCreate(&solver_), "cusolverDnCreate");
      solver_ready_ = true;
    }

    ensure_buffer(dA_, bytes_for(m, n), bytes_A_, "cudaMalloc(dA)");
    ensure_buffer(dOmega_, bytes_for(n, l), bytes_Omega_, "cudaMalloc(dOmega)");
    ensure_buffer(dY_, bytes_for(m, l), bytes_Y_, "cudaMalloc(dY)");
    ensure_buffer(dZ_, bytes_for(n, l), bytes_Z_, "cudaMalloc(dZ)");
    ensure_buffer(dQ_, bytes_for(m, l), bytes_Q_, "cudaMalloc(dQ)");
    ensure_buffer(dBsmall_, bytes_for(l, n), bytes_Bsmall_, "cudaMalloc(dBsmall)");
    ensure_buffer(dGram_, bytes_for(l, l), bytes_Gram_, "cudaMalloc(dGram)");
    ensure_buffer(dTau_, bytes_for(l, 1), bytes_Tau_, "cudaMalloc(dTau)");
    ensure_buffer(dEvals_, bytes_for(l, 1), bytes_Evals_, "cudaMalloc(dEvals)");
    ensure_int_buffer(dInfo_, sizeof(int), bytes_Info_, "cudaMalloc(dInfo)");

    if (!rng_ready_) {
      check_curand(curandCreateGenerator(&rng_, CURAND_RNG_PSEUDO_DEFAULT), "curandCreateGenerator");
      rng_ready_ = true;
    }
  }

  void set_pls_training_matrices(
    const double* hX,
    int n,
    int p,
    const double* hY,
    int m,
    bool fit
  ) {
    ensure_capacity(p, m, 1);
    ensure_buffer(dX_, bytes_for(n, p), bytes_X_, "cudaMalloc(dX)");
    ensure_buffer(dYtrain_, bytes_for(n, m), bytes_Ytrain_, "cudaMalloc(dYtrain)");
    ensure_buffer(dTvec_, bytes_for(n, 1), bytes_Tvec_, "cudaMalloc(dTvec)");
    ensure_buffer(dPvec_, bytes_for(p, 1), bytes_Pvec_, "cudaMalloc(dPvec)");
    ensure_buffer(dQvec_, bytes_for(m, 1), bytes_Qvec_, "cudaMalloc(dQvec)");
    ensure_buffer(dRvec_, bytes_for(p, 1), bytes_Rvec_, "cudaMalloc(dRvec)");
    if (fit) {
      ensure_buffer(dYfit_, bytes_for(n, m), bytes_Yfit_, "cudaMalloc(dYfit)");
      check_cuda(cudaMemset(dYfit_, 0, bytes_for(n, m)), "cudaMemset(dYfit)");
    }

    check_cuda(cudaMemcpy(dX_, hX, bytes_for(n, p), cudaMemcpyHostToDevice), "cudaMemcpy(Xtrain)");
    check_cuda(cudaMemcpy(dYtrain_, hY, bytes_for(n, m), cudaMemcpyHostToDevice), "cudaMemcpy(Ytrain)");

    const double alpha = 1.0;
    const double beta = 0.0;
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, p, m, n, &alpha, dX_, n, dYtrain_, n, &beta, dA_, p),
      "cublasDgemm(S=X^T*Y)"
    );
  }

  void simpls_fast_component_stats(
    const double* hR,
    int n,
    int p,
    int m,
    double* hT,
    double* hP,
    double* hQ,
    double* hTnorm
  ) {
    ensure_buffer(dX_, bytes_for(n, p), bytes_X_, "cudaMalloc(dX)");
    ensure_buffer(dYtrain_, bytes_for(n, m), bytes_Ytrain_, "cudaMalloc(dYtrain)");
    ensure_buffer(dTvec_, bytes_for(n, 1), bytes_Tvec_, "cudaMalloc(dTvec)");
    ensure_buffer(dPvec_, bytes_for(p, 1), bytes_Pvec_, "cudaMalloc(dPvec)");
    ensure_buffer(dQvec_, bytes_for(m, 1), bytes_Qvec_, "cudaMalloc(dQvec)");
    ensure_buffer(dRvec_, bytes_for(p, 1), bytes_Rvec_, "cudaMalloc(dRvec)");

    check_cuda(cudaMemcpy(dRvec_, hR, bytes_for(p, 1), cudaMemcpyHostToDevice), "cudaMemcpy(rr)");

    const double alpha = 1.0;
    const double beta = 0.0;
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, 1, p, &alpha, dX_, n, dRvec_, p, &beta, dTvec_, n),
      "cublasDgemm(t=X*r)"
    );

    double tnorm = 0.0;
    check_cublas(cublasDnrm2(handle_, n, dTvec_, 1, &tnorm), "cublasDnrm2(t)");
    if (!std::isfinite(tnorm) || tnorm <= 0.0) {
      throw std::runtime_error("invalid tnorm in cuda_simpls_fast_component_stats");
    }

    const double inv_tnorm = 1.0 / tnorm;
    check_cublas(cublasDscal(handle_, n, &inv_tnorm, dTvec_, 1), "cublasDscal(t)");
    check_cublas(cublasDscal(handle_, p, &inv_tnorm, dRvec_, 1), "cublasDscal(r)");

    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, p, 1, n, &alpha, dX_, n, dTvec_, n, &beta, dPvec_, p),
      "cublasDgemm(p=X^T*t)"
    );
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, m, 1, n, &alpha, dYtrain_, n, dTvec_, n, &beta, dQvec_, m),
      "cublasDgemm(q=Y^T*t)"
    );

    check_cuda(cudaMemcpy(hT, dTvec_, bytes_for(n, 1), cudaMemcpyDeviceToHost), "cudaMemcpy(t)");
    check_cuda(cudaMemcpy(hP, dPvec_, bytes_for(p, 1), cudaMemcpyDeviceToHost), "cudaMemcpy(p)");
    check_cuda(cudaMemcpy(hQ, dQvec_, bytes_for(m, 1), cudaMemcpyDeviceToHost), "cudaMemcpy(q)");
    *hTnorm = tnorm;
  }

  void simpls_fast_rank1_fit_update(
    const double* hT,
    int n,
    const double* hQ,
    int m,
    double* hDelta
  ) {
    ensure_buffer(dYfit_, bytes_for(n, m), bytes_Yfit_, "cudaMalloc(dYfit)");
    ensure_buffer(dTvec_, bytes_for(n, 1), bytes_Tvec_, "cudaMalloc(dTvec)");
    ensure_buffer(dQvec_, bytes_for(m, 1), bytes_Qvec_, "cudaMalloc(dQvec)");
    check_cuda(cudaMemcpy(dTvec_, hT, bytes_for(n, 1), cudaMemcpyHostToDevice), "cudaMemcpy(fit_t)");
    check_cuda(cudaMemcpy(dQvec_, hQ, bytes_for(m, 1), cudaMemcpyHostToDevice), "cudaMemcpy(fit_q)");
    const double alpha = 1.0;
    check_cublas(
      cublasDger(handle_, n, m, &alpha, dTvec_, 1, dQvec_, 1, dYfit_, n),
      "cublasDger(Yfit+=tq^T)"
    );
    check_cuda(cudaMemcpy(hDelta, dYfit_, bytes_for(n, m), cudaMemcpyDeviceToHost), "cudaMemcpy(Yfit_cur)");
  }

  void orthonormalize_qr_inplace(int m, int l) {
    if (env_int_or("FASTPLS_GPU_QR", 1, 0, 1) == 0) {
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, l, l, m, &one_, dY_, m, dY_, m, &zero_, dGram_, l),
        "cublasDgemm(Y^T*Y)"
      );
      hGram_host_.set_size(static_cast<arma::uword>(l), static_cast<arma::uword>(l));
      check_cuda(cudaMemcpy(hGram_host_.memptr(), dGram_, bytes_for(l, l), cudaMemcpyDeviceToHost), "cudaMemcpy(Gram)");
      arma::mat T = inv_sqrt_psd(hGram_host_);
      check_cuda(cudaMemcpy(dOmega_, T.memptr(), bytes_for(l, l), cudaMemcpyHostToDevice), "cudaMemcpy(T)");
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, l, l, &one_, dY_, m, dOmega_, l, &zero_, dQ_, m),
        "cublasDgemm(Q=Y*T)"
      );
      check_cuda(cudaMemcpy(dY_, dQ_, bytes_for(m, l), cudaMemcpyDeviceToDevice), "cudaMemcpy(Q->Y)");
      return;
    }

    int lwork_geqrf = 0;
    int lwork_orgqr = 0;
    check_cusolver(cusolverDnDgeqrf_bufferSize(solver_, m, l, dY_, m, &lwork_geqrf), "cusolverDnDgeqrf_bufferSize");
    check_cusolver(cusolverDnDorgqr_bufferSize(solver_, m, l, l, dY_, m, dTau_, &lwork_orgqr), "cusolverDnDorgqr_bufferSize");
    const int lwork = std::max(lwork_geqrf, lwork_orgqr);
    ensure_buffer(dWorkQR_, sizeof(double) * static_cast<size_t>(lwork), bytes_WorkQR_, "cudaMalloc(dWorkQR)");

    check_cusolver(cusolverDnDgeqrf(solver_, m, l, dY_, m, dTau_, dWorkQR_, lwork_geqrf, dInfo_), "cusolverDnDgeqrf");
    check_cuda(cudaMemcpy(&hInfo_, dInfo_, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(info_geqrf)");
    if (hInfo_ != 0) {
      throw std::runtime_error("cusolverDnDgeqrf returned non-zero info");
    }

    check_cusolver(cusolverDnDorgqr(solver_, m, l, l, dY_, m, dTau_, dWorkQR_, lwork_orgqr, dInfo_), "cusolverDnDorgqr");
    check_cuda(cudaMemcpy(&hInfo_, dInfo_, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(info_orgqr)");
    if (hInfo_ != 0) {
      throw std::runtime_error("cusolverDnDorgqr returned non-zero info");
    }
  }

  void finalize_left_block_from_gram_inplace(int l, int k) {
    if (env_int_or("FASTPLS_GPU_EIG", 1, 0, 1) == 0) {
      hGram_host_.set_size(static_cast<arma::uword>(l), static_cast<arma::uword>(l));
      check_cuda(cudaMemcpy(hGram_host_.memptr(), dGram_, bytes_for(l, l), cudaMemcpyDeviceToHost), "cudaMemcpy(BsmallGram)");
      arma::vec evals;
      arma::mat evecs;
      if (!arma::eig_sym(evals, evecs, hGram_host_)) {
        throw std::runtime_error("eig_sym failed in finalize_left_block_from_gram_inplace");
      }
      arma::uvec ord = arma::sort_index(evals, "descend");
      arma::mat Uhat = evecs.cols(ord.head(static_cast<arma::uword>(k)));
      check_cuda(cudaMemcpy(dOmega_, Uhat.memptr(), bytes_for(l, k), cudaMemcpyHostToDevice), "cudaMemcpy(Uhat)");
      return;
    }

    int lwork_syevd = 0;
    check_cusolver(
      cusolverDnDsyevd_bufferSize(
        solver_,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        l,
        dGram_,
        l,
        dEvals_,
        &lwork_syevd
      ),
      "cusolverDnDsyevd_bufferSize"
    );
    ensure_buffer(dWorkEig_, sizeof(double) * static_cast<size_t>(lwork_syevd), bytes_WorkEig_, "cudaMalloc(dWorkEig)");
    check_cusolver(
      cusolverDnDsyevd(
        solver_,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        l,
        dGram_,
        l,
        dEvals_,
        dWorkEig_,
        lwork_syevd,
        dInfo_
      ),
      "cusolverDnDsyevd"
    );
    check_cuda(cudaMemcpy(&hInfo_, dInfo_, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(info_syevd)");
    if (hInfo_ != 0) {
      throw std::runtime_error("cusolverDnDsyevd returned non-zero info");
    }

    for (int j = 0; j < k; ++j) {
      const int src_col = l - 1 - j;
      check_cublas(
        cublasDcopy(handle_, l, dGram_ + static_cast<size_t>(src_col) * static_cast<size_t>(l), 1,
                    dOmega_ + static_cast<size_t>(j) * static_cast<size_t>(l), 1),
        "cublasDcopy(Uhat)"
      );
    }
  }

  void sample_y(
    const double* hA,
    int m,
    int n,
    const double* hOmega,
    int l,
    int power_iters,
    double* hY
  ) {
    ensure_capacity(m, n, l);

    check_cuda(cudaMemcpy(dA_, hA, bytes_for(m, n), cudaMemcpyHostToDevice), "cudaMemcpy(A)");
    check_cuda(cudaMemcpy(dOmega_, hOmega, bytes_for(n, l), cudaMemcpyHostToDevice), "cudaMemcpy(Omega)");

    const double alpha = 1.0;
    const double beta = 0.0;

    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, l, n, &alpha, dA_, m, dOmega_, n, &beta, dY_, m),
      "cublasDgemm(A*Omega)"
    );

    for (int i = 0; i < power_iters; ++i) {
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, n, l, m, &alpha, dA_, m, dY_, m, &beta, dZ_, n),
        "cublasDgemm(A^T*Y)"
      );
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, l, n, &alpha, dA_, m, dZ_, n, &beta, dY_, m),
        "cublasDgemm(A*Z)"
      );
    }

    check_cuda(cudaMemcpy(hY, dY_, bytes_for(m, l), cudaMemcpyDeviceToHost), "cudaMemcpy(Y)");
  }

  void set_matrix(const double* hA, int m, int n) {
    ensure_capacity(m, n, 1);
    check_cuda(cudaMemcpy(dA_, hA, bytes_for(m, n), cudaMemcpyHostToDevice), "cudaMemcpy(A)");
  }

  void refresh_left_block(
    const double* hA,
    int m,
    int n,
    const double* hY0,
    int l,
    int power_iters,
    double* hY
  ) {
    ensure_capacity(m, n, l);
    hOmega_host_.set_size(static_cast<arma::uword>(n), static_cast<arma::uword>(l));

    check_cuda(cudaMemcpy(dA_, hA, bytes_for(m, n), cudaMemcpyHostToDevice), "cudaMemcpy(A)");
    check_cuda(cudaMemcpy(dY_, hY0, bytes_for(m, l), cudaMemcpyHostToDevice), "cudaMemcpy(Y0)");

    const double alpha = 1.0;
    const double beta = 0.0;

    for (int i = 0; i < power_iters; ++i) {
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, n, l, m, &alpha, dA_, m, dY_, m, &beta, dZ_, n),
        "cublasDgemm(A^T*Y)"
      );
      check_cuda(cudaMemcpy(hOmega_host_.memptr(), dZ_, bytes_for(n, l), cudaMemcpyDeviceToHost), "cudaMemcpy(Z_host)");

      arma::mat Qz;
      arma::mat Rz;
      arma::qr_econ(Qz, Rz, hOmega_host_);
      if (Qz.n_cols > static_cast<arma::uword>(l)) {
        Qz = Qz.cols(0, static_cast<arma::uword>(l - 1));
      }
      check_cuda(cudaMemcpy(dOmega_, Qz.memptr(), bytes_for(n, static_cast<int>(Qz.n_cols)), cudaMemcpyHostToDevice), "cudaMemcpy(Qz)");
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, static_cast<int>(Qz.n_cols), n, &alpha, dA_, m, dOmega_, n, &beta, dY_, m),
        "cublasDgemm(A*Qz)"
      );
    }

    check_cuda(cudaMemcpy(hY, dY_, bytes_for(m, l), cudaMemcpyDeviceToHost), "cudaMemcpy(Y)");
  }

  void refresh_left_block_u(
    const double* hA,
    int m,
    int n,
    const double* hY0,
    int l,
    int k,
    int power_iters,
    double* hUblock
  ) {
    ensure_capacity(m, n, l);
    check_cuda(cudaMemcpy(dA_, hA, bytes_for(m, n), cudaMemcpyHostToDevice), "cudaMemcpy(A)");
    check_cuda(cudaMemcpy(dY_, hY0, bytes_for(m, l), cudaMemcpyHostToDevice), "cudaMemcpy(Y0)");

    const double alpha = 1.0;
    const double beta = 0.0;

    for (int i = 0; i < power_iters; ++i) {
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, n, l, m, &alpha, dA_, m, dY_, m, &beta, dZ_, n),
        "cublasDgemm(A^T*Y)"
      );
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, l, n, &alpha, dA_, m, dZ_, n, &beta, dY_, m),
        "cublasDgemm(A*Z)"
      );
    }

    orthonormalize_qr_inplace(m, l);
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, l, n, m, &alpha, dY_, m, dA_, m, &beta, dBsmall_, l),
      "cublasDgemm(Bsmall=Q^T*A)"
    );
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T, l, l, n, &alpha, dBsmall_, l, dBsmall_, l, &beta, dGram_, l),
      "cublasDgemm(Bsmall*Bsmall^T)"
    );

    finalize_left_block_from_gram_inplace(l, k);

    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, k, l, &alpha, dY_, m, dOmega_, l, &beta, dQ_, m),
      "cublasDgemm(Ublock=Q*Uhat)"
    );
    check_cuda(cudaMemcpy(hUblock, dQ_, bytes_for(m, k), cudaMemcpyDeviceToHost), "cudaMemcpy(Ublock)");
  }

  void refresh_left_block_u_resident(
    int m,
    int n,
    const double* hY0,
    int l,
    int k,
    unsigned int seed,
    int power_iters,
    double* hUblock
  ) {
    ensure_capacity(m, n, l);
    if (hY0 != nullptr) {
      check_cuda(cudaMemcpy(dY_, hY0, bytes_for(m, l), cudaMemcpyHostToDevice), "cudaMemcpy(Y0)");
    } else {
      check_curand(curandSetPseudoRandomGeneratorSeed(rng_, static_cast<unsigned long long>(seed)), "curandSetPseudoRandomGeneratorSeed");
      check_curand(curandGenerateNormalDouble(rng_, dY_, static_cast<size_t>(m) * static_cast<size_t>(l), 0.0, 1.0), "curandGenerateNormalDouble(Y0)");
    }

    const double alpha = 1.0;
    const double beta = 0.0;

    for (int i = 0; i < power_iters; ++i) {
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, n, l, m, &alpha, dA_, m, dY_, m, &beta, dZ_, n),
        "cublasDgemm(A^T*Y)"
      );
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, l, n, &alpha, dA_, m, dZ_, n, &beta, dY_, m),
        "cublasDgemm(A*Z)"
      );
    }

    orthonormalize_qr_inplace(m, l);
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, l, n, m, &alpha, dY_, m, dA_, m, &beta, dBsmall_, l),
      "cublasDgemm(Bsmall=Q^T*A)"
    );
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T, l, l, n, &alpha, dBsmall_, l, dBsmall_, l, &beta, dGram_, l),
      "cublasDgemm(Bsmall*Bsmall^T)"
    );

    finalize_left_block_from_gram_inplace(l, k);

    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, k, l, &alpha, dY_, m, dOmega_, l, &beta, dQ_, m),
      "cublasDgemm(Ublock=Q*Uhat)"
    );
    check_cuda(cudaMemcpy(hUblock, dQ_, bytes_for(m, k), cudaMemcpyDeviceToHost), "cudaMemcpy(Ublock)");
  }

  void project_left_row(const double* hV, int m, int n, double* hVS) {
    ensure_capacity(m, n, 1);
    check_cuda(cudaMemcpy(dY_, hV, bytes_for(m, 1), cudaMemcpyHostToDevice), "cudaMemcpy(v)");
    const double alpha = 1.0;
    const double beta = 0.0;
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, n, 1, m, &alpha, dA_, m, dY_, m, &beta, dZ_, n),
      "cublasDgemm(A^T*v)"
    );
    check_cuda(cudaMemcpy(hVS, dZ_, bytes_for(n, 1), cudaMemcpyDeviceToHost), "cudaMemcpy(vS)");
  }

  void deflate_left_rank1(const double* hV, const double* hVS, int m, int n) {
    ensure_capacity(m, n, 1);
    check_cuda(cudaMemcpy(dY_, hV, bytes_for(m, 1), cudaMemcpyHostToDevice), "cudaMemcpy(v)");
    check_cuda(cudaMemcpy(dZ_, hVS, bytes_for(n, 1), cudaMemcpyHostToDevice), "cudaMemcpy(vS)");
    const double alpha = -1.0;
    check_cublas(
      cublasDger(handle_, m, n, &alpha, dY_, 1, dZ_, 1, dA_, m),
      "cublasDger(deflate)"
    );
  }

 private:
  static size_t bytes_for(int m, int n) {
    return sizeof(double) * static_cast<size_t>(m) * static_cast<size_t>(n);
  }

  void ensure_buffer(double*& ptr, size_t required, size_t& current, const char* where) {
    if (required <= current && ptr != nullptr) {
      return;
    }
    if (ptr != nullptr) {
      cudaFree(ptr);
      ptr = nullptr;
      current = 0;
    }
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr), required), where);
    current = required;
  }

  void ensure_int_buffer(int*& ptr, size_t required, size_t& current, const char* where) {
    if (required <= current && ptr != nullptr) {
      return;
    }
    if (ptr != nullptr) {
      cudaFree(ptr);
      ptr = nullptr;
      current = 0;
    }
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr), required), where);
    current = required;
  }

  void release() {
    if (dX_ != nullptr) cudaFree(dX_);
    if (dYtrain_ != nullptr) cudaFree(dYtrain_);
    if (dA_ != nullptr) cudaFree(dA_);
    if (dOmega_ != nullptr) cudaFree(dOmega_);
    if (dY_ != nullptr) cudaFree(dY_);
    if (dZ_ != nullptr) cudaFree(dZ_);
    if (dQ_ != nullptr) cudaFree(dQ_);
    if (dTvec_ != nullptr) cudaFree(dTvec_);
    if (dPvec_ != nullptr) cudaFree(dPvec_);
    if (dQvec_ != nullptr) cudaFree(dQvec_);
    if (dRvec_ != nullptr) cudaFree(dRvec_);
    if (dYfit_ != nullptr) cudaFree(dYfit_);
    if (dBsmall_ != nullptr) cudaFree(dBsmall_);
    if (dGram_ != nullptr) cudaFree(dGram_);
    if (dTau_ != nullptr) cudaFree(dTau_);
    if (dEvals_ != nullptr) cudaFree(dEvals_);
    if (dWorkQR_ != nullptr) cudaFree(dWorkQR_);
    if (dWorkEig_ != nullptr) cudaFree(dWorkEig_);
    if (dInfo_ != nullptr) cudaFree(dInfo_);
    dX_ = nullptr;
    dYtrain_ = nullptr;
    dA_ = nullptr;
    dOmega_ = nullptr;
    dY_ = nullptr;
    dZ_ = nullptr;
    dQ_ = nullptr;
    dTvec_ = nullptr;
    dPvec_ = nullptr;
    dQvec_ = nullptr;
    dRvec_ = nullptr;
    dYfit_ = nullptr;
    dBsmall_ = nullptr;
    dGram_ = nullptr;
    dTau_ = nullptr;
    dEvals_ = nullptr;
    dWorkQR_ = nullptr;
    dWorkEig_ = nullptr;
    dInfo_ = nullptr;
    bytes_X_ = bytes_Ytrain_ = 0;
    bytes_A_ = bytes_Omega_ = bytes_Y_ = bytes_Z_ = 0;
    bytes_Q_ = bytes_Bsmall_ = bytes_Gram_ = 0;
    bytes_Tvec_ = bytes_Pvec_ = bytes_Qvec_ = bytes_Rvec_ = bytes_Yfit_ = 0;
    bytes_Tau_ = bytes_Evals_ = bytes_WorkQR_ = bytes_WorkEig_ = bytes_Info_ = 0;
    if (handle_ready_) {
      cublasDestroy(handle_);
      handle_ready_ = false;
    }
    if (solver_ready_) {
      cusolverDnDestroy(solver_);
      solver_ready_ = false;
    }
    if (rng_ready_) {
      curandDestroyGenerator(rng_);
      rng_ready_ = false;
    }
  }

  double* dX_ = nullptr;
  double* dYtrain_ = nullptr;
  double* dA_ = nullptr;
  double* dOmega_ = nullptr;
  double* dY_ = nullptr;
  double* dZ_ = nullptr;
  double* dQ_ = nullptr;
  double* dTvec_ = nullptr;
  double* dPvec_ = nullptr;
  double* dQvec_ = nullptr;
  double* dRvec_ = nullptr;
  double* dYfit_ = nullptr;
  double* dBsmall_ = nullptr;
  double* dGram_ = nullptr;
  double* dTau_ = nullptr;
  double* dEvals_ = nullptr;
  double* dWorkQR_ = nullptr;
  double* dWorkEig_ = nullptr;
  int* dInfo_ = nullptr;
  size_t bytes_X_ = 0;
  size_t bytes_Ytrain_ = 0;
  size_t bytes_A_ = 0;
  size_t bytes_Omega_ = 0;
  size_t bytes_Y_ = 0;
  size_t bytes_Z_ = 0;
  size_t bytes_Q_ = 0;
  size_t bytes_Tvec_ = 0;
  size_t bytes_Pvec_ = 0;
  size_t bytes_Qvec_ = 0;
  size_t bytes_Rvec_ = 0;
  size_t bytes_Yfit_ = 0;
  size_t bytes_Bsmall_ = 0;
  size_t bytes_Gram_ = 0;
  size_t bytes_Tau_ = 0;
  size_t bytes_Evals_ = 0;
  size_t bytes_WorkQR_ = 0;
  size_t bytes_WorkEig_ = 0;
  size_t bytes_Info_ = 0;
  cublasHandle_t handle_ = nullptr;
  bool handle_ready_ = false;
  cusolverDnHandle_t solver_ = nullptr;
  bool solver_ready_ = false;
  curandGenerator_t rng_ = nullptr;
  bool rng_ready_ = false;
  int hInfo_ = 0;
  const double one_ = 1.0;
  const double zero_ = 0.0;
  arma::mat hOmega_host_;
  arma::mat hGram_host_;
};

thread_local CudaRSVDWorkspace g_workspace;

} // namespace

bool cuda_runtime_available() {
  int n_devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&n_devices);
  return (status == cudaSuccess && n_devices > 0);
}

bool cuda_rsvd_prefer_block_gpu(int m, int n, int l, int power_iters) {
  if (!cuda_runtime_available()) {
    return false;
  }
  const int min_m = env_int_or("FASTPLS_FAST_GPU_MIN_M", 512, 1, 1 << 30);
  const int min_n = env_int_or("FASTPLS_FAST_GPU_MIN_N", 16, 1, 1 << 30);
  const int min_work = env_int_or("FASTPLS_FAST_GPU_MIN_WORK", 200000, 1, 1 << 30);
  const long long work = static_cast<long long>(m) * static_cast<long long>(n) * static_cast<long long>(std::max(l, 1));
  return (m >= min_m) && (n >= min_n) && (work >= min_work) && (power_iters >= 1);
}

void cuda_rsvd_sample_y(
  const double* hA,
  int m,
  int n,
  const double* hOmega,
  int l,
  int power_iters,
  double* hY
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  g_workspace.sample_y(hA, m, n, hOmega, l, power_iters, hY);
}

void cuda_rsvd_set_resident_matrix(
  const double* hA,
  int m,
  int n
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  g_workspace.set_matrix(hA, m, n);
}

void cuda_rsvd_refresh_left_block(
  const double* hA,
  int m,
  int n,
  const double* hY0,
  int l,
  int power_iters,
  double* hY
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  g_workspace.refresh_left_block(hA, m, n, hY0, l, power_iters, hY);
}

void cuda_rsvd_refresh_left_block_u(
  const double* hA,
  int m,
  int n,
  const double* hY0,
  int l,
  int k,
  int power_iters,
  double* hUblock
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  g_workspace.refresh_left_block_u(hA, m, n, hY0, l, k, power_iters, hUblock);
}

void cuda_rsvd_refresh_left_block_u_resident(
  int m,
  int n,
  const double* hY0,
  int l,
  int k,
  unsigned int seed,
  int power_iters,
  double* hUblock
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  g_workspace.refresh_left_block_u_resident(m, n, hY0, l, k, seed, power_iters, hUblock);
}

void cuda_rsvd_project_left_row(
  const double* hV,
  int m,
  int n,
  double* hVS
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  g_workspace.project_left_row(hV, m, n, hVS);
}

void cuda_rsvd_deflate_left_rank1(
  const double* hV,
  const double* hVS,
  int m,
  int n
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  g_workspace.deflate_left_rank1(hV, hVS, m, n);
}

void cuda_simpls_fast_set_training_matrices(
  const double* hX,
  int n,
  int p,
  const double* hY,
  int m,
  bool fit
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  g_workspace.set_pls_training_matrices(hX, n, p, hY, m, fit);
}

void cuda_simpls_fast_component_stats(
  const double* hR,
  int n,
  int p,
  int m,
  double* hT,
  double* hP,
  double* hQ,
  double* hTnorm
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  g_workspace.simpls_fast_component_stats(hR, n, p, m, hT, hP, hQ, hTnorm);
}

void cuda_simpls_fast_rank1_fit_update(
  const double* hT,
  int n,
  const double* hQ,
  int m,
  double* hDelta
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  g_workspace.simpls_fast_rank1_fit_update(hT, n, hQ, m, hDelta);
}

} // namespace fastpls_svd

#else

#include <stdexcept>

namespace fastpls_svd {

SVDResult truncated_svd_cuda_rsvd(const Mat&, int, const SVDOptions&) {
  throw std::runtime_error("CUDA backend not compiled");
}

bool cuda_runtime_available() {
  return false;
}

void cuda_rsvd_sample_y(
  const double*,
  int,
  int,
  const double*,
  int,
  int,
  double*
) {
  throw std::runtime_error("CUDA backend not compiled");
}

bool cuda_rsvd_prefer_block_gpu(int, int, int, int) {
  return false;
}

void cuda_rsvd_set_resident_matrix(
  const double*,
  int,
  int
) {
  throw std::runtime_error("CUDA backend not compiled");
}

void cuda_rsvd_refresh_left_block(
  const double*,
  int,
  int,
  const double*,
  int,
  int,
  double*
) {
  throw std::runtime_error("CUDA backend not compiled");
}

void cuda_rsvd_refresh_left_block_u(
  const double*,
  int,
  int,
  const double*,
  int,
  int,
  int,
  double*
) {
  throw std::runtime_error("CUDA backend not compiled");
}

void cuda_rsvd_refresh_left_block_u_resident(
  int,
  int,
  const double*,
  int,
  int,
  unsigned int,
  int,
  double*
) {
  throw std::runtime_error("CUDA backend not compiled");
}

void cuda_rsvd_project_left_row(
  const double*,
  int,
  int,
  double*
) {
  throw std::runtime_error("CUDA backend not compiled");
}

void cuda_rsvd_deflate_left_rank1(
  const double*,
  const double*,
  int,
  int
) {
  throw std::runtime_error("CUDA backend not compiled");
}

void cuda_simpls_fast_set_training_matrices(
  const double*,
  int,
  int,
  const double*,
  int,
  bool
) {
  throw std::runtime_error("CUDA backend not compiled");
}

void cuda_simpls_fast_component_stats(
  const double*,
  int,
  int,
  int,
  double*,
  double*,
  double*,
  double*
) {
  throw std::runtime_error("CUDA backend not compiled");
}

void cuda_simpls_fast_rank1_fit_update(
  const double*,
  int,
  const double*,
  int,
  double*
) {
  throw std::runtime_error("CUDA backend not compiled");
}

} // namespace fastpls_svd

#endif
