\name{pls.single.cv}
\alias{pls.single.cv}
\title{Single grouped cross-validation for PLS}
\usage{
pls.single.cv(
  Xdata,
  Ydata,
  constrain = NULL,
  ncomp = 2L,
  kfold = 10L,
  scaling = c("centering", "autoscaling", "none"),
  method = c("simpls", "plssvd", "opls", "kernelpls"),
  backend = c("cpu", "cuda", "metal"),
  svd.method = c("irlba", "rsvd"),
  seed = 1L,
  north = 1L,
  kernel = c("linear", "rbf", "poly"),
  gamma = NULL,
  degree = 3L,
  coef0 = 1,
  gaussian_y = FALSE,
  gaussian_y_dim = NULL,
  classifier = c("argmax", "lda", "cknn"),
  lda_ridge = 1e-08,
  candidate_knn_k = 10L,
  candidate_tau = 0.2,
  candidate_alpha = 0.75,
  candidate_top_m = 20L,
  return_scores = FALSE,
  xprod = NULL,
  ...
)
}
\arguments{
\item{Xdata}{Numeric predictor matrix.}
\item{Ydata}{Response vector, response matrix, or factor.}
\item{constrain}{Optional grouping vector for grouped cross-validation. It
must have one value per sample. Samples with the same value are assigned to
the same fold, so they are kept together either in the training set or in
the test set. This is useful when several rows come from the same patient,
subject, batch, or technical replicate and must not be split across
training and test folds. When \code{NULL}, each sample is treated as its own
group.}
\item{ncomp}{Number of components.}
\item{kfold}{Number of folds, or \code{"loocv"} for leave-one-out
cross-validation. When \code{constrain} is supplied, LOOCV means
leave-one-constraint-group-out: all samples sharing the same constraint
value are held out together.}
\item{scaling}{One of \code{centering}, \code{autoscaling}, or \code{none}.}
\item{method}{Model family: \code{simpls}, \code{plssvd}, \code{opls}, or \code{kernelpls}.}
\item{backend}{Implementation backend: \code{cpu}, \code{cuda}, or \code{metal}.}
\item{svd.method}{CPU SVD algorithm family: \code{irlba} or \code{rsvd}.}
\item{seed}{Cross-validation and randomized-SVD seed.}
\item{north}{Number of orthogonal components for OPLS.}
\item{kernel, gamma, degree, coef0}{Kernel PLS controls.}
\item{gaussian_y, gaussian_y_dim}{Optional Gaussian response compression controls. The Gaussian sketch uses \code{seed}.}
\item{classifier}{Classification rule for factor responses: \code{"argmax"} uses
the largest predicted response score, \code{"lda"} fits an LDA head on the PLS
scores inside each training fold, and \code{"cknn"} uses the candidate-kNN head.}
\item{lda_ridge}{Ridge added to the pooled LDA covariance diagonal.}
\item{candidate_knn_k, candidate_tau, candidate_alpha, candidate_top_m}{Candidate-kNN
controls. \code{candidate_knn_k} is the number of neighbours,
\code{candidate_tau} is the softmax temperature, \code{candidate_alpha} weights
the neighbour evidence against the PLS class score, and
\code{candidate_top_m} limits reranking to the strongest candidate classes.}
\item{return_scores}{For classification, keep score predictions when supported.}
\item{xprod}{Use the matrix-free cross-product route where available. \code{NULL} applies fastPLS defaults.}
\item{...}{Optional SVD tuning controls forwarded to the selected backend.
Use the same compact names documented in \code{\link{fastsvd}}, such as
\code{oversample}, \code{power}, \code{svds_tol}, \code{work},
\code{maxit}, \code{tol}, \code{eps}, and \code{svtol}.}
}
\value{
A list with fold assignments, predictions, metrics, status, and backend metadata.
}
\description{
Runs one fixed-component grouped cross-validation through the same high-level
algorithm router used by \code{\link{pls}}. The \code{cpu} and \code{cuda}
backends use the compiled CV core; \code{metal} uses \code{\link{pls}} with
\code{backend = metal} inside an R-level fold loop.
}
\examples{
idx <- c(1:12, 51:62, 101:112)
X <- as.matrix(iris[idx, 1:4])
y <- factor(iris[idx, 5])
cv <- pls.single.cv(X, y, ncomp = 2, kfold = 3, method = "simpls",
                    backend = "cpu", svd.method = "rsvd", seed = 1)
cv$metrics
}
