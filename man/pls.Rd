\name{pls}

\alias{pls}

\title{Partial Least Squares.}

\description{Partial Least Squares (PLS) classification and regression for test set from training set.}

\usage{
           
pls (Xtrain, 
     Ytrain, 
     Xtest = NULL, 
     Ytest = NULL, 
     ncomp=2,
     scaling = c("centering", "autoscaling","none"), 
     method = c("simpls", "plssvd", "simpls_fast"),
     svd.method = c("irlba", "dc", "cpu_rsvd", "cuda_rsvd"),
     rsvd_oversample = 10L,
     rsvd_power = 1L,
     seed = 1L,
     fast_block = 4L,
     fast_center_t = FALSE,
     fast_reorth_v = TRUE,
     fast_incremental = FALSE,
     fast_inc_iters = 2L,
     fast_defl_cache = TRUE,
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
  \item{ncomp}{the number of components to consider. For \code{method = "plssvd"}, the effective maximum is \code{min(nrow(Xtrain), ncol(Xtrain), ncol(Ytrain))}; if higher values are requested, the model warns and uses this maximum internally.}
  \item{scaling}{the scaling method to be used. Choices are "\code{centering}", "\code{autoscaling}", or "\code{none}" (by default = "\code{centering}"). A partial string sufficient to uniquely identify the choice is permitted.}
  \item{method}{the algorithm to be used to perform the PLS. Choices are "\code{simpls}", "\code{plssvd}", or "\code{simpls_fast}" (by default = "\code{simpls}"). A partial string sufficient to uniquely identify the choice is permitted. \code{simpls_fast} keeps the SIMPLS deflation scheme but uses block-updated truncated SVD directions and optional incremental settings for speed.}
  \item{svd.method}{SVD backend used inside the PLS algorithm.
  Available options are:
  \code{"irlba"} (legacy truncated IRLBA path, default),
  \code{"dc"} (CPU exact divide-and-conquer SVD via Armadillo \code{svd_econ()}),
  \code{"cpu_rsvd"} (CPU randomized SVD with \code{rsvd_oversample}, \code{rsvd_power}, and \code{seed}),
  and \code{"cuda_rsvd"} (CUDA randomized SVD; requires \code{\link{has_cuda}()} to be \code{TRUE}).
  \code{"dc"} is deterministic exact CPU SVD; RSVD methods are approximate but reproducible with fixed seed.}
  \item{rsvd_oversample}{extra random directions used by RSVD beyond the requested rank \code{k} (effective subspace size is approximately \code{k + rsvd_oversample}). Larger values usually improve accuracy but increase compute time and memory. Typical range: 5--30; default \code{10}.}
  \item{rsvd_power}{number of power-iteration refinement steps in RSVD. Higher values improve separation of close singular values (better accuracy) but increase runtime; \code{0} is fastest, \code{1} is a common tradeoff (default), and \code{2} is for harder problems.}
  \item{seed}{deterministic seed used by RSVD methods (\code{cpu_rsvd}, \code{cuda_rsvd}).}
  \item{fast_block}{used only when \code{method = "simpls_fast"}. Number of latent directions refreshed per block SVD step. Larger blocks may improve throughput but can reduce numerical robustness in some datasets.}
  \item{fast_center_t}{used only when \code{method = "simpls_fast"}. If \code{TRUE}, score vectors are mean-centered before normalization at each component.}
  \item{fast_reorth_v}{used only when \code{method = "simpls_fast"}. If \code{TRUE}, applies an additional re-orthogonalization pass in the deflation basis.}
  \item{fast_incremental}{used only when \code{method = "simpls_fast"}. If \code{TRUE}, enables an incremental block-power iteration warm-started from previous latent directions.}
  \item{fast_inc_iters}{used only when \code{method = "simpls_fast"}. Number of incremental block-power iterations when \code{fast_incremental = TRUE}.}
  \item{fast_defl_cache}{used only when \code{method = "simpls_fast"}. If \code{TRUE}, uses a cached deflation update (\code{v^T S} then rank-1 update) to reduce repeated multiplications.}
  \item{fit}{logical; if \code{TRUE}, return fitted values on the training set.}
  \item{proj}{logical; if \code{TRUE}, return projected test scores when \code{Xtest} is provided.}
  \item{perm.test}{logical; if \code{TRUE}, run permutation testing.}
  \item{times}{number of permutations used when \code{perm.test = TRUE}.}

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
  \item{classification}{logical indicating whether the response variable was treated as classification.}
  \item{lev}{level of response variable Y}
  \item{Ypred}{the (ntest x m x length(ncomp)) containing the predicted values of the response variables for the observations from Xtest. The third dimension of the matrix Ypred corresponds to the number of PLS components used to compute the regression coefficients.}
  \item{P}{the (p x max(ncomp)) matrix containing the X-loadings.}
  \item{Ttest}{...}
  
  

  }

\author{Dupe Ojo, Alessia Vignoli, Stefano Cacciatore, Leonardo Tenori}

\references{
de Jong S. (1993). SIMPLS: an alternative approach to partial least squares regression. \emph{Chemometrics and Intelligent Laboratory Systems}, 18(3), 251--263.\\
Halko N., Martinsson P.-G., Tropp J. A. (2011). Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions. \emph{SIAM Review}, 53(2), 217--288.\\
Musco C., Musco C. (2015). Randomized block Krylov methods for stronger and faster approximate singular value decomposition. arXiv:1504.05477.
}


\seealso{\code{\link{optim.pls.cv}},\code{\link{pls.double.cv}}}

\examples{
data(iris)
data=iris[,-5]
labels=iris[,5]
ss=sample(150,15)
ncomponent=3

z=pls(data[-ss,], labels[-ss], data[ss,], ncomp=ncomponent) 

}
