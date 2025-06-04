


#ifndef fastPLS_fastPLS_H
#define fastPLS_fastPLS_H

// Include the Rcpp Header

#include <RcppArmadillo.h>
#include <Rcpp.h>



using namespace Rcpp;
using namespace arma;

List IRLB(arma::mat& X, int nu, int work, int maxit, double tol, double eps,   double svtol);
double RQ(arma::mat yData,arma::mat yPred);
arma::mat variance(arma::mat x);
arma::mat transformy(arma::ivec y);
List pls_model1(arma::mat Xtrain,arma::mat Ytrain,arma::ivec ncomp,int scaling,bool fit,int svd_method);
List pls_model2(arma::mat Xtrain,arma::mat Ytrain,arma::ivec ncomp,int scaling,bool fit,int svd_method);
List pls_predict(List& model, arma::mat Xtest, bool proj);
int unic(arma::mat x);
IntegerVector samplewithoutreplace(IntegerVector yy,int size);
List optim_pls_cv(arma::mat Xdata,arma::mat Ydata,arma::ivec constrain,arma::ivec ncomp, int scaling, int kfold, int method,int svd_method);
List double_pls_cv(arma::mat Xdata,arma::mat Ydata,arma::ivec ncomp,arma::ivec constrain, int scaling, int kfold_inner, int kfold_outer, int method,int svd_method);

 
 
 
 
 
 
namespace fastPLS {


// Appended to the traditional function definition is the `inline` keyword.


inline arma::mat pls_light(arma::mat Xtrain,arma::mat Ytrain,arma::mat Xtest,int ncomp) {
  
  // Xtrain <- scale(Xtrain,center=TRUE,scale=FALSE)
  // Xtest <-scale(Xtest,center=mX)
  arma::mat mX=mean(Xtrain,0);
  Xtrain.each_row()-=mX;
  
  // Y <- scale(Ytrain,center=TRUE,scale=FALSE)
  arma::mat mY=mean(Ytrain,0);
  Ytrain.each_row()-=mY;
  
  // S <- crossprod(X,Y)
  arma::mat S=trans(Xtrain)*Ytrain;

  arma::mat svd_u;
  arma::vec svd_s;
  arma::mat svd_v;
  int Snr=S.n_rows;
  int Snc=S.n_cols;
  if(Snr>5  && Snc>5){
    List temp0=IRLB(S, ncomp, ncomp+10, 2000, 1e-6, 1e-9, 1e-6); //nu=1, work=10, maxit=1000, tol=1e-6, eps=1e-9, svtol=1e-6
    svd_u=as<arma::mat>(temp0("u"));   //u
    svd_v=as<arma::mat>(temp0("v"));   //v
  }else{ 
    svd_econ(svd_u,svd_s,svd_v,S,"both");
  }
  arma::mat svd_u_mc = svd_u.cols(0,ncomp-1);
  arma::mat svd_v_mc = svd_v.cols(0,ncomp-1);
  
  // TT <- Xtrain %*% R
  // U <- Ytrain %*% Q
  arma::mat T=Xtrain*svd_u_mc;
  arma::mat U=Ytrain*svd_v_mc;
  
  //B <- R %*% ( tcrossprod(solve(crossprod(TT)), TT)%*%U ) %*% t(Q)
  arma::mat Tt=T.t();
  arma::mat B = svd_u_mc * (arma::inv(Tt * T) * Tt * U) * svd_v_mc.t();
  
  Xtest.each_row()-=mX;
  
  // Ypred <- scale(Xtest %*% B, -meanY, FALSE)
  arma::mat Ypred=Xtest * B;
  Ypred.each_row()+=mY;
  
  return Ypred;
}

}
#endif
