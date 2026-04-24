

#include <RcppArmadillo.h>
#include <R_ext/Rdynload.h>
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
  
  //  B<-matrix(0,ncol=m,nrow=p)
  arma::cube B(p,m,length_ncomp);
  B.zeros();
  
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
      B.slice(i_out)=RR*trans(QQ);
      if(fit){
        Yfit.slice(i_out)=Xtrain*B.slice(i_out);
        arma::mat temp1=Yfit.slice(i_out);
        temp1.each_row()+=mY;
        Yfit.slice(i_out)=temp1;
        R2Y(i_out)=RQ(Ytrain,temp1);
        
      }
      i_out++;
    }
  } 
  
  return List::create(
    Named("B")       = B,
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
  arma::cube B(p, m, length_ncomp, fill::zeros);

  arma::cube Yfit;
  arma::vec R2Y(length_ncomp, fill::zeros);
  arma::mat Yfit_cur;
  if (fit) {
    Yfit.set_size(n, m, length_ncomp);
    Yfit_cur.zeros(n, m);
  }

  arma::mat Bcur(p, m, fill::zeros);
  int i_out = 0;

  // Inspired by block-Krylov randomized SVD literature (e.g. arXiv:1504.05477):
  // refresh a small block of singular vectors to reduce per-component SVD overhead.
  const int refresh_block = env_int_or("FASTPLS_FAST_BLOCK", 8, 1, 16);
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
    Bcur += rr * qq.t();

    while (i_out < length_ncomp && a_idx == (ncomp(i_out) - 1)) {
      B.slice(i_out) = Bcur;
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

  return List::create(
    Named("B")       = B,
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

  const arma::mat Xt = Xtrain.t();
  const arma::mat Yt = Ytrain.t();

  arma::mat RR(p, max_ncomp, fill::zeros);
  arma::mat QQ(m, max_ncomp, fill::zeros);
  arma::mat VV(p, max_ncomp, fill::zeros);
  arma::cube B(p, m, length_ncomp, fill::zeros);

  arma::cube Yfit;
  arma::vec R2Y(length_ncomp, fill::zeros);
  arma::mat Yfit_cur;
  if (fit) {
    Yfit.set_size(n, m, length_ncomp);
    Yfit_cur.zeros(n, m);
  }

  arma::mat Bcur(p, m, fill::zeros);
  int i_out = 0;

  const int refresh_block = env_int_or("FASTPLS_FAST_BLOCK", 8, 1, 16);
  const int center_t = env_int_or("FASTPLS_FAST_CENTER_T", 0, 0, 1);
  const int reorth_v = env_int_or("FASTPLS_FAST_REORTH_V", 0, 0, 1);
  const int inc_power_iters = env_int_or("FASTPLS_FAST_INC_ITERS", 2, 1, 6);
  const int defl_cache = env_int_or("FASTPLS_FAST_DEFLCACHE", 1, 0, 1);
  (void)defl_cache;
  const bool use_device_state = (env_int_or("FASTPLS_GPU_DEVICE_STATE", 0, 0, 1) == 1);
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
    fit
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
      adaptive_policy.update_from_spectrum(shat_block, max_ncomp - (a + k_block));

      bool stop_now = false;
      for (int j = 0; j < k_block && a < max_ncomp; ++j, ++a) {
        if (!fastpls_svd::cuda_simpls_fast_append_component_from_block(
              n,
              p,
              m,
              a,
              j,
              a,
              (reorth_v == 1),
              fit
            )) {
          stop_now = true;
          break;
        }
        has_rr_prev = true;

        while (i_out < length_ncomp && a == (ncomp(i_out) - 1)) {
          fastpls_svd::cuda_simpls_fast_copy_bcur(B.slice(i_out).memptr(), p, m);
          if (fit) {
            fastpls_svd::cuda_simpls_fast_copy_yfit(Yfit_cur.memptr(), n, m);
            R2Y(i_out) = RQ(Ytrain, Yfit_cur);
            arma::mat yf = Yfit_cur;
            yf.each_row() += mY;
            Yfit.slice(i_out) = yf;
          }
          ++i_out;
        }
      }
      if (stop_now) {
        break;
      }
    }

    fastpls_svd::cuda_simpls_fast_copy_rr(RR.memptr(), p, max_ncomp);
    fastpls_svd::cuda_simpls_fast_copy_qq(QQ.memptr(), m, max_ncomp);
  } else {
    arma::mat S_shape = Xt * Ytrain;
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
        pp = Xt * tt;
        qq = Yt * tt;
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

      arma::rowvec vS = vv.t() * S_shape;
      S_shape -= vv * vS;

      RR.col(a_idx) = rr;
      QQ.col(a_idx) = qq;
      VV.col(a_idx) = vv;
      Bcur += rr * qq.t();

      while (i_out < length_ncomp && a_idx == (ncomp(i_out) - 1)) {
        B.slice(i_out) = Bcur;
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

  return List::create(
    Named("B")       = B,
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
}


// [[Rcpp::export]]
List pls_predict(List& model, arma::mat Xtest, bool proj) {

  // columns of Ytrain
  int m = model("m");
  
  // w <-dim(Xtest)[1]
  int w = Xtest.n_rows;
  
  arma::ivec ncomp=model("ncomp");
  arma::uword length_ncomp = static_cast<arma::uword>(ncomp.n_elem);
  
  //scaling factors
  arma::rowvec mX = model("mX");
  Xtest.each_row()-=mX;
  arma::rowvec vX = model("vX");
  Xtest.each_row()/=vX;
  arma::rowvec mY = model("mY");
  arma::cube B=model("B");
  arma::mat RR=model("R");
  
  arma::cube Ypred(w,m,length_ncomp);
  for (arma::uword a = 0; a < length_ncomp; ++a) {
    arma::mat pred = Xtest * B.slice(a);
    pred.each_row() += mY;
    Ypred.slice(a) = std::move(pred);
  }  
  arma::mat T_Xtest;
  if(proj){ 
    T_Xtest = Xtest*RR;
  }

  return List::create(
    Named("Ypred")  = Ypred,
    Named("Ttest")   = T_Xtest
  );
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
  
  //  B<-matrix(0,ncol=m,nrow=p)
  arma::cube B(p,m,length_ncomp);
  B.zeros();
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

      B.slice(a) = svd_u_mc * coeff_latent * svd_v_mc.t();
      if(fit){
        arma::mat temp1 = T_a * coeff_latent * svd_v_mc.t();
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
      B.slice(a)= svd_u_mc * coeff_latent * svd_v_mc.t();
      if(fit){
        arma::mat temp1=Xtrain*B.slice(a);
        R2Y(a)=RQ(Ytrain,temp1);
        temp1.each_row()+=mY;
        Yfit.slice(a)=temp1;
      }
    }
  }



  return List::create(
    Named("B")       = B,
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

  return List::create(
    Named("B")       = gpu.B,
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
}
