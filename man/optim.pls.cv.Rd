\name{optim.pls.cv}

\alias{optim.pls.cv}

\title{Cross-Validation with PLS-DA.}

\description{This function performs a 10-fold cross validation on a given data set using Partial Least Squares (PLS) model. To assess the prediction ability of the model, a 10-fold cross-validation is conducted by generating splits with a ratio 1:9 of the data set. This is achieved by removing 10\% of samples prior to any step of the statistical analysis, including PLS component selection and scaling. Best number of component for PLS was carried out by means of 10-fold cross-validation on the remaining 90\% selecting the best Q2y value. Permutation testing was undertaken to estimate the classification/regression performance of predictors.}

\usage{

optim.pls.cv (Xdata,
              Ydata, 
              ncomp, 
              constrain=NULL,
              scaling = c("centering", "autoscaling","none"),
              method = c("plssvd", "simpls"),
              svd.method = c("irlba", "dc"),
              kfold=10)              
}

\arguments{
  \item{Xdata}{a matrix of independent variables or predictors.}
  \item{Ydata}{the responses. If Ydata is a numeric vector, a regression analysis will be performed. If Ydata is factor, a classification analysis will be performed. }
  \item{ncomp}{the number of latent components to be used for classification.}
  \item{constrain}{a vector of \code{nrow(data)} elements. Sample sharing a specific  identifier or characteristics will be grouped together either in the training set or in the test set of cross-validation.}  
  \item{scaling}{the scaling method to be used. Choices are "\code{centering}", "\code{autoscaling}", or "\code{none}" (by default = "\code{centering}"). A partial string sufficient to uniquely identify the choice is permitted.}
  \item{method}{the algorithm to be used to perform the PLS. Choices are "\code{plssvd}" or "\code{simpls}" (by default = "\code{plssvd}"). A partial string sufficient to uniquely identify the choice is permitted.}
  \item{svd.method}{the SVD method to be used to perform the PLS. Choices are "\code{irlba}" or "\code{dc}" (by default = "\code{irlba}"). A partial string sufficient to uniquely identify the choice is permitted.}
    \item{kfold}{number of cross-validations loops.}

}

\value{
The output of the result is a list with the following components:
  \item{B}{the (p x m x length(ncomp)) array containing the regression coefficients. Each row corresponds to a predictor variable and each column to a response variable. The third dimension of the matrix B corresponds to the number of PLS components used to compute the regression coefficients. If ncomp has length 1, B is just a (p x m) matrix.}
  \item{Ypred}{the vector containing the predicted values of the response variables obtained by cross-validation.}
  \item{Yfit}{the vector containing the fitted values of the response variables.}
  \item{P}{the (p x max(ncomp)) matrix containing the X-loadings.}
  \item{Q}{the (m x max(ncomp)) matrix containing the Y-loadings.}
  \item{T}{the (ntrain x max(ncomp)) matrix containing the X-scores (latent components)}
  \item{R}{the (p x max(ncomp)) matrix containing the weights used to construct the latent components.}
  \item{Q2Y}{predicting power of model.}
  \item{R2Y}{proportion of variance in Y.}
  \item{R2X}{vector containg the explained variance of X by each PLS component.}

  \item{txtQ2Y}{a summary of the Q2y values.}
  \item{txtR2Y}{a summary of the R2y values.}
  }

\author{Dupe Ojo, Alessia Vignoli, Stefano Cacciatore, Leonardo Tenori}

\seealso{\code{\link{pls}},\code{\link{pls.double.cv}}}

\examples{
\donttest{
data(iris)
data=iris[,-5]
labels=iris[,5]
pp=optim.pls.cv(data,labels,2:4)
pp$optim_comp

}
}
