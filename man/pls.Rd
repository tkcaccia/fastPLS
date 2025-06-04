\name{pls}

\alias{pls}

\title{Partial Least Squares.}

\description{Partial Least Squares (PLS) classification and regression for test set from training set.}

\usage{
           
pls (Xtrain, 
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
           
           

}

\arguments{
  \item{Xtrain}{a matrix of training set cases.}
  \item{Ytrain}{a classification vector.}
  \item{Xtest}{a matrix of test set cases.}
  \item{Ytest}{a classification vector.}
  \item{ncomp}{the number of components to consider.}
  \item{scaling}{the scaling method to be used. Choices are "\code{centering}", "\code{autoscaling}", or "\code{none}" (by default = "\code{centering}"). A partial string sufficient to uniquely identify the choice is permitted.}
  \item{method}{the algorithm to be used to perform the PLS. Choices are "\code{plssvd}" or "\code{simpls}" (by default = "\code{plssvd}"). A partial string sufficient to uniquely identify the choice is permitted.}
  \item{svd.method}{the SVD method to be used to perform the PLS. Choices are "\code{irlba}" or "\code{dc}" (by default = "\code{irlba}"). A partial string sufficient to uniquely identify the choice is permitted.}
  \item{fit}{a boolean value to perform the fit.}
  \item{proj}{a boolean value to perform the fit.}
  \item{perm.test}{a classification vector.}
  \item{times}{a classification vector.}

}

\value{
A list with the following components:
  \item{B}{the (p x m x length(ncomp)) matrix containing the regression coefficients. Each row corresponds to a predictor variable and each column to a response variable. The third dimension of the matrix B corresponds to the number of PLS components used to compute the regression coefficients. If ncomp has length 1, B is just a (p x m) matrix.}
  \item{Q}{the (m x max(ncomp)) matrix containing the Y-loadings.}
  \item{Ttrain}{the (ntrain x max(ncomp)) matrix containing the X-scores (latent components)}
  \item{R}{the (p x max(ncomp)) matrix containing the weights used to construct the latent components.}
  \item{mX}{mean X.}
  \item{vX}{variance X.}
  \item{mY}{mean Y.}
  \item{p}{matrix for the independent variable X. This indicates how the original data relates to the latent components.}
  \item{m}{number of predictor variables}
  \item{ncomp}{number of components used in the PLS model}
  \item{Yfit}{the prediction values based on the PLS model}
  \item{R2Y}{proportion of variance in Y}
  \item{classification}{a boolgean output is given indicating if the response variable is a classification}
  \item{lev}{level of response variable Y}
  \item{Ypred}{the (ntest x m x length(ncomp)) containing the predicted values of the response variables for the observations from Xtest. The third dimension of the matrix Ypred corresponds to the number of PLS components used to compute the regression coefficients.}
  \item{P}{the (p x max(ncomp)) matrix containing the X-loadings.}
  \item{Ttest}{...}
  
  

  }

\author{Dupe Ojo, Alessia Vignoli, Stefano Cacciatore, Leonardo Tenori}


\seealso{\code{\link{optim.pls.cv}},\code{\link{pls.double.cv}}}

\examples{
data(iris)
data=iris[,-5]
labels=iris[,5]
ss=sample(150,15)
ncomponent=3

z=pls(data[-ss,], labels[-ss], data[ss,], ncomp=ncomponent) 

}

