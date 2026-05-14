\name{pca}
\alias{pca}
\title{Principal component analysis through fastPLS SVD backends}
\usage{
pca(
  x,
  ncomp = 2L,
  center = TRUE,
  scale. = FALSE,
  svd.method = c("cpu_rsvd", "irlba", "cuda_rsvd", "metal_rsvd"),
  ...
)
}
\arguments{
\item{x}{Numeric matrix with samples in rows and variables in columns.}
\item{ncomp}{Number of principal components.}
\item{center}{Logical; center columns before SVD.}
\item{scale.}{Logical; scale columns before SVD.}
\item{svd.method}{SVD backend used by \code{\link{fastsvd}}.}
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
