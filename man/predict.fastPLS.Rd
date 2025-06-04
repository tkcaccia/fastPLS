\name{predict.fastPLS}

\alias{predict.fastPLS}
\title{Prediction Partial Least Squares regression.}

\description{Partial Least Squares (PLS) regression for test set from training set.}

\usage{
\method{predict}{fastPLS}(object, newdata, Ytest=NULL, proj=FALSE, ...) 

}

\arguments{
  \item{object}{a matrix of training set cases.}
  \item{newdata}{a matrix of predictor variables X for the test set.}
  \item{Ytest}{a vector of the response variable Y from Xtest.}
  \item{proj}{projection of the test set.}
  \item{...}{further arguments. Currently not used.}

}

\value{
A list with the following components:

  \item{Ypred}{the (ntest x m x length(ncomp)) containing the predicted values of the response variables for the observations from Xtest. The third dimension of the matrix Ypred corresponds to the number of PLS components used to compute the regression coefficients.}
  \item{Q2Y}{predictive power of model}
  \item{Ttest}{the (ntrain x max(ncomp)) matrix containing the X-scores (latent components)}

  }

\author{Dupe Ojo, Alessia Vignoli, Stefano Cacciatore, Leonardo Tenori}

\seealso{\code{\link{optim.pls.cv}},\code{\link{pls.double.cv}}}

\examples{
data(iris)
data=iris[,-5]
labels=iris[,5]
ss=sample(150,15)
ncomponent=3

z=pls(data[-ss,], labels[-ss],  ncomp=ncomponent) 
predict(z,data[ss,],FALSE)


}

