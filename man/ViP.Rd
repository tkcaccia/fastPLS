\name{ViP}

\alias{ViP}

\title{Variable Importance in the Projection.}

\description{Variable Importance in the Projection (VIP) is a score that measures how important a variable is in a Partial Least Squares (PLS) model. VIP scores are used to identify which variables are most important in a model and are often used for variable selection. }

\usage{
ViP (model) 

}

\arguments{
  \item{model}{a object returning from the pls function.}

}

\value{
A list with the following components:
  \item{B}{the (p x m x length(ncomp)) matrix containing the regression coefficients. Each row corresponds to a predictor variable and each column to a response variable. The third dimension of the matrix B corresponds to the number of PLS components used to compute the regression coefficients. If ncomp has length 1, B is just a (p x m) matrix.}
  \item{Ypred}{the (ntest x m x length(ncomp)) containing the predicted values of the response variables for the observations from Xtest. The third dimension of the matrix Ypred corresponds to the number of PLS components used to compute the regression coefficients.}
  \item{P}{the (p x max(ncomp)) matrix containing the X-loadings.}
  \item{Q}{the (m x max(ncomp)) matrix containing the Y-loadings.}
  \item{T}{the (ntrain x max(ncomp)) matrix containing the X-scores (latent components)}
  \item{R}{the (p x max(ncomp)) matrix containing the weights used to construct the latent components.}

  }

\author{Dupe Ojo, Alessia Vignoli, Stefano Cacciatore, Leonardo Tenori}

\seealso{\code{\link{optim.pls.cv}},\code{\link{pls.double.cv}}}

\examples{
data(iris)
data=as.matrix(iris[,-5])
labels=iris[,5]
pp=pls(data,labels,ncomp = 2)
ViP(pp)

}

