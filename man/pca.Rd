\name{pca}
\alias{pca}
\title{Principal component analysis through fastPLS SVD backends}
\usage{
pca(
  x,
  ncomp = 2L,
  center = TRUE,
  scale = FALSE,
  backend = c("cpu", "cuda", "metal"),
  method = c("irlba", "rsvd"),
  ...
)
}
\arguments{
\item{x}{Numeric matrix with samples in rows and variables in columns.}
\item{ncomp}{Number of principal components.}
\item{center}{Logical; center columns before SVD.}
\item{scale}{Logical; scale columns before SVD.}
\item{backend}{Compute backend. \code{cpu} runs on the host CPU.
\code{cuda} and \code{metal} use the corresponding native randomized-SVD
backend when available.}
\item{method}{SVD algorithm family. \code{irlba} is available only with
\code{backend = cpu}. \code{rsvd} uses randomized SVD on the selected
backend.}
\item{...}{Additional arguments passed to \code{\link{fastsvd}}.}
}
\value{
A \code{fastPLSPCA} object containing scores, loadings, standard deviations,
per-component \code{variance_explained}, cumulative variance explained,
preprocessing values, and SVD metadata.
}
\description{
Computes PCA from the selected fastPLS SVD backend and returns a compact object
with a score-plot method.
}
\examples{
pc <- pca(as.matrix(iris[, 1:4]), ncomp = 2, backend = "cpu",
          method = "rsvd", seed = 1)
head(pc$scores)
pc$variance_explained
}
