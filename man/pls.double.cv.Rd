\name{pls.double.cv}

\alias{pls.double.cv}

\title{Cross-Validation with PLS-DA.}

\description{This function performs a 10-fold cross validation on a given data set using Partial Least Squares (PLS) model. To assess the prediction ability of the model, a 10-fold cross-validation is conducted by generating splits with a ratio 1:9 of the data set, that is by removing 10\% of samples prior to any step of the statistical analysis, including PLS component selection and scaling. Best number of component for PLS was carried out by means of 10-fold cross-validation on the remaining 90\% selecting the best Q2y value. Permutation testing was undertaken to estimate the classification/regression performance of predictors.}

\usage{

pls.double.cv (Xdata,
               Ydata,
               ncomp=2,
               constrain=1:nrow(Xdata),
               scaling = c("centering", "autoscaling","none"), 
               method = c("simpls", "plssvd"),
              svd.method = c("irlba", "dc", "cpu_rsvd", "cuda_rsvd"),
               rsvd_oversample = 10L,
               rsvd_power = 1L,
               seed = 1L,
               perm.test=FALSE,
               times=100,
               runn=10,
               kfold_inner=10, 
               kfold_outer=10)           
              
}

\arguments{
  \item{Xdata}{a matrix.}
  \item{Ydata}{the responses. If Ydata is a numeric vector, a regression analysis will be performed. If Ydata is factor, a classification analysis will be performed. }
  \item{ncomp}{the number of latent components to evaluate. For \code{method = "plssvd"}, the effective maximum is \code{min(nrow(Xdata), ncol(Xdata), ncol(Ydata))}; higher requested values are capped internally with a warning.}
  \item{constrain}{a vector of \code{nrow(data)} elements. Sample with the same identifying constrain will be split in the training set or in the test set of cross-validation together.}
  \item{scaling}{the scaling method to be used. Choices are "\code{centering}", "\code{autoscaling}", or "\code{none}" (by default = "\code{centering}"). A partial string sufficient to uniquely identify the choice is permitted.}
  \item{method}{the algorithm to be used to perform the PLS. Choices are "\code{simpls}" or "\code{plssvd}" (by default = "\code{simpls}"). A partial string sufficient to uniquely identify the choice is permitted.}
  \item{svd.method}{SVD backend used inside each PLS fit of the inner and outer CV loops.
  Available options are:
  \code{"irlba"} (legacy truncated IRLBA path, default),
  \code{"dc"} (CPU exact divide-and-conquer SVD via Armadillo \code{svd_econ()}),
  \code{"cpu_rsvd"} (CPU randomized SVD with \code{rsvd_oversample}, \code{rsvd_power}, and \code{seed}),
  and \code{"cuda_rsvd"} (CUDA randomized SVD; requires \code{\link{has_cuda}()} to be \code{TRUE}).
  \code{"dc"} is deterministic exact CPU SVD; RSVD methods are approximate but reproducible with fixed seed.}
  \item{rsvd_oversample}{extra random directions used by RSVD beyond the requested rank \code{k} (effective subspace size is approximately \code{k + rsvd_oversample}). Larger values usually improve accuracy but increase compute time and memory. Typical range: 5--30; default \code{10}.}
  \item{rsvd_power}{number of power-iteration refinement steps in RSVD. Higher values improve separation of close singular values (better accuracy) but increase runtime; \code{0} is fastest, \code{1} is a common tradeoff (default), and \code{2} is for harder problems.}
  \item{seed}{deterministic seed used by RSVD methods (\code{cpu_rsvd}, \code{cuda_rsvd}).}
  \item{perm.test}{logical; if \code{TRUE}, run permutation testing.}
  \item{times}{number of permutations for permutation testing.}
  \item{runn}{number of cross-validations loops.}
  \item{kfold_inner}{if perform the optmization of the number of components.}
  \item{kfold_outer}{if perform the optmization of the number of components.}
}

\value{
A list with the following components:
  \item{B}{the (p x m x length(ncomp)) array containing the regression coefficients. Each row corresponds to a predictor variable and each column to a response variable. The third dimension of the matrix B corresponds to the number of PLS components used to compute the regression coefficients. If ncomp has length 1, B is just a (p x m) matrix.}
  \item{Ypred}{the vector containing the predicted values of the response variables obtained by cross-validation.}
  \item{Yfit}{the vector containing the fitted values of the response variables.}
  \item{P}{the (p x max(ncomp)) matrix containing the X-loadings.}
  \item{Q}{the (m x max(ncomp)) matrix containing the Y-loadings.}
  \item{T}{the (ntrain x max(ncomp)) matrix containing the X-scores (latent components)}
  \item{R}{the (p x max(ncomp)) matrix containing the weights used to construct the latent components.}
  \item{Q2Y}{predictive power of the model.}
  \item{R2Y}{proportion of variance in Y.}
  \item{R2X}{vector containg the explained variance of X by each PLS component.}

  \item{txtQ2Y}{a summary of the Q2y values.}
  \item{txtR2Y}{a summary of the R2y values.}
  }

\author{Dupe Ojo, Alessia Vignoli, Stefano Cacciatore, Leonardo Tenori}


\seealso{\code{\link{optim.pls.cv}},\code{\link{pls}}}


\examples{
\donttest{
data(iris)
data=iris[,-5]
labels=iris[,5]
pp=pls.double.cv(data,labels,2:4)

}
}
