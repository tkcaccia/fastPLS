\name{predict.fastPLSPCA}
\alias{predict.fastPLSPCA}
\title{Project new data with a fitted fastPLS PCA model}
\usage{
\method{predict}{fastPLSPCA}(object, newdata, ncomp = NULL, ...)
}
\arguments{
\item{object}{A \code{fastPLSPCA} object returned by \code{\link{pca}}.}

\item{newdata}{Numeric matrix with the same columns, in the same order, as the
matrix used to fit \code{object}.}

\item{ncomp}{Optional number of principal components to return. By default all
components stored in \code{object} are returned.}

\item{...}{Ignored.}
}
\value{
Matrix of projected PCA scores for \code{newdata}.
}
\description{
Applies the centering, scaling, and loading matrix stored in a \code{fastPLSPCA}
object to an independent dataset.
}
\examples{
pc <- pca(as.matrix(iris[, 1:4]), ncomp = 2, backend = "cpu",
          method = "rsvd", seed = 1)
predict(pc, as.matrix(iris[1:3, 1:4]))
}
