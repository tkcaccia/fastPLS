

#include <RcppArmadillo.h>
#include <R_ext/Rdynload.h>

#include "fastPLS.h"

extern "C" {
#include "irlba.h"
}

// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;


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
List IRLB(arma::mat& X,
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


  irlb (X.memptr(), NULL, 0, m, n, nu, work, maxit, 0,
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
List pls_model2(arma::mat Xtrain,arma::mat Ytrain,arma::ivec ncomp,int scaling,bool fit,int svd_method) {

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
  arma::mat svd_U;
  arma::vec svd_s;
  arma::mat svd_V;
  arma::mat rr;
  arma::mat tt;
  arma::mat vv;
  
  int i_out=0; //position of the saving output
  
  // for(a in 1:ncomp){
  for (int a=0; a<max_ncomp; a++) {
    //qq<-svd(S)$v[,1]
    //rr <- S%*%qq
//    if(S.n_rows<=16 || S.n_cols<=16){
  if(S.n_rows>5  && S.n_cols>5 && svd_method==1){
      List temp0=IRLB(S, 1, 10, 2000, 1e-6, 1e-9, 1e-6); //nu=1, work=10, maxit=1000, tol=1e-6, eps=1e-9, svtol=1e-6
      arma::mat u_irlba=temp0[1];
      rr=u_irlba.col(0);
    }else{ 
      arma::mat svd_U;
      arma::vec svd_s;
      arma::mat svd_V;
      svd_econ(svd_U,svd_s,svd_V,S,"left"); 
      rr=svd_U.col( 0 );
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
List pls_predict(List& model, arma::mat Xtest, bool proj) {

  // columns of Ytrain
  int m = model("m");
  
  // w <-dim(Xtest)[1]
  int w = Xtest.n_rows;
  
  arma::ivec ncomp=model("ncomp");
  int length_ncomp=ncomp.n_elem;
  
  //scaling factors
  arma::mat mX=model("mX");
  Xtest.each_row()-=mX;
  arma::mat vX=model("vX");
  Xtest.each_row()/=vX;  
  arma::mat mY=model("mY"); 
  arma::cube B=model("B");
  arma::mat RR=model("R");
  
  arma::cube Ypred(w,m,length_ncomp);
  Ypred.zeros();  
  for (int a=0; a<length_ncomp; a++) {
    Ypred.slice(a)=Xtest*B.slice(a);
    arma::mat temp1=Ypred.slice(a);
    temp1.each_row()+=mY;
    Ypred.slice(a)=temp1;
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
List optim_pls_cv(arma::mat Xdata,arma::mat Ydata,arma::ivec constrain,arma::ivec ncomp, int scaling, int kfold, int method, int svd_method) {
  
  int length_ncomp=ncomp.n_elem;
  
  int nsamples=Xdata.n_rows;
  
  int ncolY=Ydata.n_cols;
  arma::cube Ypred(nsamples,ncolY,length_ncomp); 
  //int xsa_t = max(constrain);

  arma::ivec indices = unique(constrain);

  arma::ivec constrain2=constrain;
  

  for (int j = 0; j < indices.size(); j++) {
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
      model=pls_model1(Xtrain,Ytrain,ncomp,scaling,FALSE,svd_method);
    }
    if(method==2){
      model=pls_model2(Xtrain,Ytrain,ncomp,scaling,FALSE,svd_method);
    }
    List pls=pls_predict(model,Xtest,FALSE);
    arma::cube temp1=pls("Ypred");
    for(int ii=0;ii<w1_size;ii++)  for(int jj=0;jj<length_ncomp;jj++)  for(int kk=0;kk<ncolY;kk++)  Ypred(w1[ii],kk,jj)=temp1(ii,kk,jj);  
    
  }  
  List model_all;
  if(method==1){
    model_all=pls_model1(Xdata,Ydata,ncomp,scaling,TRUE,svd_method);
  }
  if(method==2){
    model_all=pls_model2(Xdata,Ydata,ncomp,scaling,TRUE, svd_method);
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
List double_pls_cv(arma::mat Xdata,arma::mat Ydata,arma::ivec ncomp,arma::ivec constrain, int scaling, int kfold_inner, int kfold_outer, int method,int svd_method) {
  
  int nsamples=Xdata.n_rows;
  
  int ncolY=Ydata.n_cols;
  arma::mat Ypred(nsamples,ncolY); 
  
  arma::ivec indices = unique(constrain);
  
  arma::ivec constrain2=constrain;
  

  for (int j = 0; j < indices.size(); j++) {
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
    
    List opt=optim_pls_cv(Xtrain, Ytrain, constrain_train, ncomp, scaling, kfold_inner, method, svd_method);
      
    List model;
    if(method==1){
      model=pls_model1(Xtrain,Ytrain,opt("optim_comp"),scaling,FALSE, svd_method);
    }
    if(method==2){
      model=pls_model2(Xtrain,Ytrain,opt("optim_comp"),scaling,FALSE, svd_method);
    }   
      
    List pls=pls_predict(model,Xtest,FALSE);
    arma::cube temp1=pls("Ypred");
    for(int ii=0;ii<w1_size;ii++)  for(int kk=0;kk<ncolY;kk++)  Ypred(w1[ii],kk)=temp1(ii,kk,0);  
    
    // Calculation of R2Y
    List model_all;
    if(method==1){
      model_all=pls_model1(Xtrain,Ytrain,opt("optim_comp"),scaling,TRUE, svd_method);
    }
    if(method==2){
      model_all=pls_model2(Xtrain,Ytrain,opt("optim_comp"),scaling,TRUE, svd_method);
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
List pls_model1(arma::mat Xtrain,arma::mat Ytrain,arma::ivec ncomp,int scaling,bool fit,int svd_method) {
  
  // n <-dim(Xtrain)[1]
  int n = Xtrain.n_rows;
  
  // p <-dim(Xtrain)[2]
  int p = Xtrain.n_cols;
  
  // m <- dim(Y)[2]
  int m = Ytrain.n_cols;
  int max_ncomp=max(ncomp);
  int length_ncomp=ncomp.n_elem;
  
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
  
  int Snr=S.n_rows;
  int Snc=S.n_cols;
  if(Snr>5  && Snc>5 && svd_method==1){

    List temp0=IRLB(S, max_ncomp, 10+max_ncomp, 2000, 1e-6, 1e-9, 1e-6); //nu=1, work=10, maxit=1000, tol=1e-6, eps=1e-9, svtol=1e-6
    svd_u=as<arma::mat>(temp0("u"));   //u
    svd_v=as<arma::mat>(temp0("v"));   //v
  }else{ 
    svd(svd_u,svd_s,svd_v,S);   //svd_econ
  }
  
  //  B<-matrix(0,ncol=m,nrow=p)
  arma::cube B(p,m,length_ncomp);
  B.zeros();
  arma::cube Yfit;
  if(fit){
    Yfit.resize(n,m,length_ncomp);
  }
  
  
  
  
  svd_u = svd_u.cols(0,max_ncomp-1);
  if(svd_v.n_cols>max_ncomp){
    svd_v = svd_v.cols(0,max_ncomp-1);
  }
  
  // TT <- Xtrain %*% R
  // U <- Ytrain %*% Q
  
  arma::mat T=Xtrain*svd_u;
  
  arma::vec R2Y(length_ncomp);
  
  
  for (int a=0; a<length_ncomp; a++) {
    int mc=ncomp(a);
    

    arma::mat svd_u_mc = svd_u.cols(0,mc-1);
    
    arma::mat svd_v_mc;
    if(svd_v.n_cols>max_ncomp){
      svd_v_mc = svd_v.cols(0,mc-1);
    }else{
      svd_v_mc = svd_v;
    }
    
    arma::mat T_a=T.cols(0,mc-1);
    arma::mat T_at=T_a.t();
    
    
    // TT <- Xtrain %*% R
    // U <- Ytrain %*% Q

    arma::mat U=Ytrain*svd_v_mc;
    
    //B <- R %*% ( tcrossprod(solve(crossprod(TT)), TT)%*%U ) %*% t(Q)
    B.slice(a)= svd_u_mc * (arma::inv(T_at * T_a) * T_at * U) * svd_v_mc.t();
    if(fit){
      arma::mat temp1=Xtrain*B.slice(a);
      R2Y(a)=RQ(Ytrain,temp1);
      temp1.each_row()+=mY;
      Yfit.slice(a)=temp1;
    }
  }



  return List::create(
    Named("B")       = B,
    Named("Q")       = svd_v,
    Named("Ttrain")  = T,
    Named("R")       = svd_u,
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







