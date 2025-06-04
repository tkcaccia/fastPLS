##r_irlba <- function(X, nu, work=nu+7, maxit=1000, tol=1e-5, eps=1e-10, svtol=tol) {
##  stopifnot(work>nu)
##  IRLB(X, nu, work, maxit, tol, eps, svtol)
##}

###' @import irlba
##r_orthog <- function(x, y) {
##  if (missing(y))
##    y <- runif(nrow(x))
##  y <- matrix(y)
##  xm <- nrow(x)
##  xn <- ncol(x)
##  yn <- ncol(y)
##  stopifnot(nrow(y)==xm)
##  stopifnot(yn==1)
##  initT <- matrix(0, xn+1, yn+1)
##  ORTHOG(x, y, initT, xm, xn, yn)
##}
# https://github.com/zdk123/irlba


predict.fastPLS = function(object, newdata, Ytest=NULL, proj=FALSE, ...) {
  if (!is(object, "fastPLS")) {
    stop("object is not a fastPLS object")
  }
  Xtest=newdata
  res=pls_predict(object, Xtest,proj)
  res$Q2Y=NULL
   
  if (!is.null(Ytest)) {
    for (i in 1:length(object$ncomp)) {
      if(object$classification){
        Ytest_transf=matrix(0,ncol=length(object$lev),nrow=length(Ytest))
        colnames(Ytest_transf)=object$lev
        for(w in object$lev){
          Ytest_transf[Ytest==w,w]=1
        }
      } else{
        Ytest_transf=as.matrix(Ytest)
      }
      res$Ypred[, , i]=t(t(res$Ypred[, , i])+as.numeric(object$mY))
      res$Q2Y[i] = RQ(Ytest_transf,res$Ypred[, , i])
    }
  }
  
  if(object$classification){
    Ypredlab = as.data.frame(matrix(nrow = nrow(Xtest), ncol = length(object$ncomp)))

    for (i in 1:length(object$ncomp)) {
      t = apply(res$Ypred[, , i], 1, which.max)
      Ypredlab[, i] = (factor(object$lev[t], levels = object$lev))
    }
    res$Ypred=Ypredlab

  }
  res
}



pls =  function (Xtrain, 
                 Ytrain, 
                 Xtest = NULL, 
                 Ytest = NULL, 
                 ncomp=min(5,c(ncol(Xtrain),nrow(Xtrain))),
                 scaling = c("centering", "autoscaling","none"), 
                 method = c("plssvd", "simpls"),
                 svd.method = c("irlba", "dc"),
                 fit = FALSE,
                 proj = FALSE, 
                 perm.test = FALSE, 
                 times = 100) 
{

  scal = pmatch(scaling, c("centering", "autoscaling","none"))[1]
  meth = pmatch(method, c("plssvd", "simpls"))[1]
  svdmeth = pmatch(svd.method, c("irlba", "dc"))[1]
  
  Xtrain = as.matrix(Xtrain)
  if (is.factor(Ytrain)){
    classification=TRUE # classification
    lev = levels(Ytrain)
    Ytrain = transformy(Ytrain)
    
  } else{
    classification=FALSE   # regression
    lev=NULL
  }
  if(meth==1){
    model=pls.model1(Xtrain,Ytrain,ncomp=ncomp,fit=fit,scaling=scal,svd.method=svdmeth)
  }
  if(meth==2){
    model=pls.model2(Xtrain,Ytrain,ncomp=ncomp,fit=fit,scaling=scal,svd.method=svdmeth)
  }
  model$classification=classification
  model$lev=lev

  
#  model$R2Y[i] = 1 - sum(((Ytrain - model$Yfit[, , i]))^2)/sum(t(t(Ytrain) -  colMeans(Ytrain))^2)
  

  
  # PLS analysis
  if(!is.null(Xtest)){
    Xtest = as.matrix(Xtest)
    res=predict(model,Xtest,Ytest,proj=proj)
    model=c(model,res)
    # output
    
 
      #    o$scoreXtest=as.matrix(Xtest) %*% o$R[,1:ncomp]
      if (perm.test) {
        v = matrix(NA,nrow=times,ncol=length(ncomp))
        for (i in 1:times) {
          ss = sample(1:nrow(Xtrain))
          Xtrain_permuted = Xtrain[ss, ]
          
          if(meth==1){
            model_perm=pls.model1(Xtrain_permuted,Ytrain,ncomp=ncomp,scaling=scal,svd.method=svdmeth)
          }
          if(meth==2){
            model_perm=pls.model2(Xtrain_permuted,Ytrain,ncomp=ncomp,scaling=scal,svd.method=svdmeth)
          }
          
          res_perm=predict(model,Xtest,Ytest)

          v[i,]=res_perm$Q2Y
        }
        model$pval=NULL
        for(j in 1:length(ncomp)){
          model$pval[j] = sum(v[,j] > model$Q2Y)/times
        }
      
      
      }
  }
    if(classification){

      if(fit){
        Yfitlab = as.data.frame(matrix(nrow = nrow(Xtrain), ncol = length(ncomp)))
        colnames(Yfitlab)=paste("ncomp=",ncomp,sep="")
        for (i in 1:length(ncomp)) {
          t = apply(model$Yfit[, , i], 1, which.max)
          Yfitlab[, i] = factor(lev[t], levels = lev)
        }
        model$Yfit=Yfitlab
      }
    }
  


  class(model)="fastPLS"
  model
}








optim.pls.cv =  function (Xdata,
                          Ydata, 
                          ncomp, 
                          constrain=NULL,
                          scaling = c("centering", "autoscaling","none"),
                          method = c("plssvd", "simpls"),
                          svd.method = c("irlba", "dc"),
                          kfold=10) 
{
  scal = pmatch(scaling, c("centering", "autoscaling","none"))[1]
  meth = pmatch(method, c("plssvd", "simpls"))[1]
  
  svdmeth = pmatch(svd.method, c("irlba", "dc"))[1]
  if(is.null(constrain))
    constrain=1:nrow(Xdata)
  Xdata=as.matrix(Xdata)
  
  if (is.factor(Ydata)){
    classification=TRUE # classification
    lev = levels(Ydata)
    Ydata = transformy(Ydata)
  } else{
    classification=FALSE   # regression
  }
  res=optim_pls_cv(Xdata, Ydata, ncomp, constrain, scal,kfold,meth,svd.method=svdmeth) 
  res
}






pls.double.cv = function(Xdata,
                         Ydata,
                         ncomp=min(5,c(ncol(Xdata),nrow(Xdata))),
                         constrain=1:nrow(Xdata),
                         scaling = c("centering", "autoscaling","none"), 
                         method = c("plssvd", "simpls"),
                         svd.method = c("irlba", "dc"),
                         perm.test=FALSE,
                         times=100,
                         runn=10,
                         kfold_inner=10, 
                         kfold_outer=10){
  
  if(sum(is.na(Xdata))>0) {
    stop("Missing values are present")
  } 
  scal=pmatch(scaling,c("centering","autoscaling","none"))[1]
  meth = pmatch(method, c("plssvd", "simpls"))[1]
  
  svdmeth = pmatch(svd.method, c("irlba", "dc"))[1]
  
  if (is.factor(Ydata)){
    classification=TRUE # classification
    lev = levels(Ydata)
    Ydata_original = Ydata
    Ydata = transformy(Ydata)
    conf_tot=matrix(0,ncol=length(lev),nrow=length(lev))
    colnames(conf_tot)=lev
    rownames(conf_tot)=lev
  } else{
    classification=FALSE   # regression
    Ydata = as.matrix(Ydata)
  }
  
  Xdata=as.matrix(Xdata)
  constrain=as.numeric(as.factor(constrain))
  
  res=list()
  Q2Y=NULL
  R2Y=NULL
  bcomp=NULL

  bb=NULL
  Ypred_tot=matrix(0,nrow=nrow(Xdata),ncol=ncol(Ydata))
  for(j in 1:runn){
    
    
    o=double_pls_cv(Xdata,Ydata,ncomp,constrain,scal,kfold_inner,kfold_outer,meth,svd.method=svdmeth)
    Ypred_tot=Ypred_tot+o$Ypred
    if(classification){
      t = apply(o$Ypred, 1, which.max)
      Ypredlab = factor(lev[t], levels = lev)
      
      o$Ypred=Ypredlab
      
      o$conf=table(Ypredlab,Ydata_original)
      conf_tot=conf_tot+o$conf
      o$acc=(sum(diag(o$conf))*100)/length(Ydata)
    }
    #  o$R2X=diag((t(o$T)%*%(o$T))%*%(t(o$P)%*%(o$P)))/sum(scale(Xdata,TRUE,TRUE)^2)
    Q2Y[j]=o$Q2Y
    R2Y[j]=mean(o$R2Y)
    res$results[[j]]=o
    bb=c(bb,o$optim_comp)
  }
  Ypred_tot=Ypred_tot/runn
  
  if(classification){
    acc_tot=round(100*diag(conf_tot)/(length(Ydata)*runn),digits=1)
    conf_tot=conf_tot/runn
    conf_tot_perc=t(t(conf_tot)/colSums(conf_tot))*100
    conf_tot=round(conf_tot,digits=1)
    conf_tot_perc=round(conf_tot_perc,digits=1)
    
    conf_txt=matrix(paste(conf_tot," (",conf_tot_perc,"%)",sep=""),ncol=length(lev))
    colnames(conf_txt)=lev
    rownames(conf_txt)=lev
    res$acc_tot=acc_tot
    res$conf=conf_txt
    
    t = apply(Ypred_tot, 1, which.max)
    Ypredlab = factor(lev[t], levels = lev)
    
    res$Ypred=Ypredlab
  }
  
    res$Q2Y=Q2Y
    res$R2Y=R2Y
    res$medianR2Y=median(R2Y)
    res$CI95R2Y=as.numeric(quantile(R2Y,c(0.025,0.975)))
    res$medianQ2Y=median(Q2Y)
    res$CI95Q2Y=as.numeric(quantile(Q2Y,c(0.025,0.975)))

    
    res$bcomp=names(which.max(table(bb)))

    if(perm.test){
      
      v=NULL
      
      for(i in 1:times){
        ss=sample(1:nrow(Xdata))
        w=NULL
        for(ii in 1:runn)
          w[ii]=double_pls_cv(Xdata[ss,],Ydata,ncomp,constrain,scal,kfold_inner,kfold_outer,meth,svd.method=svdmeth)$Q2Y
        
        v[i]=median(w)
      }
      pval=pnorm(median(Q2Y), mean=mean(v), sd=sqrt(((length(v)-1)/length(v))*var(v)), lower.tail=FALSE) 
      res$Q2Ysampled=v
      res$p.value=pval
    }
    
    
    conf_tot=matrix(0,ncol=length(lev),nrow=length(lev))
    colnames(conf_tot)=lev
    rownames(conf_tot)=lev

  res
}







Vip <- function(object) {
  
  SS <- c(object$Q)^2 * colSums(object$Ttrain^2)
  Wnorm2 <- colSums(object$R^2)
  SSW <- sweep(object$R^2, 2, SS / Wnorm2, "*")
  sqrt(nrow(SSW) * apply(SSW, 1, cumsum) / cumsum(SS))
}


ViP <- function(model) {
  
  u <- nrow(model$Q)
  if (u==1) return (Vip(model))
  V <- list ()
  for (i in 1:u) V[[i]] <- Vip(list(Q=model$Q[i,], Ttrain=model$Ttrain, R=model$R))
  return (V)
}


fastcor <- function(a, b=NULL, byrow=TRUE, diag=TRUE) {
  
  ## if byrow == T rows are correlated (much faster) else columns
  ## if diag == T only the diagonal of the cor matrix is returned (much faster)
  ## b can be NULL
  
  if (!byrow) a <- t(a)
  a <- a - rowMeans(a)
  a <- a / sqrt(rowSums(a*a))
  if (!is.null(b)) {
    if (!byrow) b <- t(b)
    b <- b - rowMeans(b)
    b <- b / sqrt(rowSums(b*b))
    if (diag) return (rowSums(a*b)) else return (tcrossprod(a,b))
  } else return (tcrossprod(a))
}

