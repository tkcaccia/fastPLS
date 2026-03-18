\name{predict.fastPLS}
\alias{predict.fastPLS}
\title{Predict From a fastPLS Model}

\description{
Generate predictions from a fitted \code{fastPLS} object.
}

\usage{
\method{predict}{fastPLS}(object, newdata, Ytest = NULL, proj = FALSE, ...)
}

\arguments{
\item{object}{A model returned by \code{\link{pls}} or \code{\link{pls_r}}.}
\item{newdata}{Numeric test matrix \eqn{X_{test}}.}
\item{Ytest}{Optional test response used to compute \code{Q2Y}.}
\item{proj}{Logical; if \code{TRUE}, return projected scores \code{Ttest}.}
\item{...}{Unused.}
}

\details{
Prediction always applies stored preprocessing from training data:
\itemize{
\item subtract \code{mX},
\item divide by \code{vX} (if autoscaling was used),
\item multiply by stored coefficient slices \code{B[,,i]},
\item add \code{mY} back.
}

For classification models (factor \code{Ytrain}), predicted class labels are
obtained by \code{argmax} over predicted response columns at each component.
}

\value{
A list with:
\itemize{
\item \code{Ypred}: regression: array \eqn{n_{test} \times m \times n_{comp}};
classification: data frame of predicted class labels per component.
\item \code{Q2Y}: returned only when \code{Ytest} is provided.
\item \code{Ttest}: projected test scores when \code{proj = TRUE}; otherwise empty matrix.
}
}

\seealso{\code{\link{pls}}, \code{\link{pls_r}}}

\examples{
set.seed(1)
data(iris)
X <- as.matrix(iris[, -5])
y <- iris[, 5]
ss <- sample(seq_len(nrow(X)), 30)

fit <- pls(X[-ss, ], y[-ss], ncomp = 3)
predict(fit, X[ss, ], Ytest = y[ss], proj = FALSE)
}
