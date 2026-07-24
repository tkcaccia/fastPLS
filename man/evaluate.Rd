\name{evaluate}
\alias{evaluate}
\title{Evaluate prediction performance}
\usage{
evaluate(
  observed,
  predicted,
  task = c("auto", "classification", "regression"),
  ytrain = NULL,
  top_k = c(1L, 5L),
  bycol = TRUE,
  relative_epsilon = .Machine$double.eps,
  na.rm = TRUE
)
}
\arguments{
\item{observed}{Observed response values. Use a factor or character vector for
classification, a numeric vector or matrix for regression, or a one-hot matrix
for classification.}

\item{predicted}{Predicted values. Use a factor or character vector for
predicted classes, a numeric vector or matrix for regression, or a class-score
matrix for classification.}

\item{task}{One of \code{"auto"}, \code{"classification"}, or
\code{"regression"}. \code{"auto"} treats factor or character observed values
as classification and numeric values as regression, unless a one-hot observed
matrix is detected.}

\item{ytrain}{Optional training response for regression Q2. When supplied, Q2
is computed relative to the training-set response mean. When omitted, Q2 uses
the observed response mean and is therefore identical to R2.}

\item{top_k}{Integer vector of top-k classification accuracies to compute when
\code{predicted} is a class-score matrix.}

\item{bycol}{For multivariate regression, calculate and return metrics for each
response column. The default is \code{TRUE} for direct \code{evaluate()} calls.}

\item{relative_epsilon}{Values with absolute observed response below this
threshold are ignored for relative-error metrics.}

\item{na.rm}{Remove incomplete observations before computing metrics.}
}
\value{
A list with \code{task}, \code{metrics}, and optionally \code{per_response},
\code{per_class}, \code{confusion}, and \code{topk}. A \code{notes} element is
included only when the evaluation has an explanatory note to report.
}
\description{
Computes common classification or regression performance metrics from observed
and predicted values. The function accepts two vectors, two matrices, or
classification score matrices.
}
\details{
For classification, the returned metrics include accuracy, the no-information
rate, lift accuracy, balanced accuracy, macro precision, macro recall, macro
F1, Cohen's kappa, and a confusion matrix. The no-information rate is the
accuracy obtained by always predicting the most prevalent observed class. Lift
accuracy is accuracy divided by this baseline; values above one indicate an
improvement over the majority-class baseline. When predicted class scores are
supplied, top-k accuracy is also reported.

For regression, the returned metrics include R2, Q2, RMSD/RMSE, MAE, bias,
median relative error percentage, mean absolute percentage error, ratio of
performance to deviation, Pearson correlation, and Spearman correlation. These
include the main spectral prediction metrics used by Vignoli et al. (2025):
median relative error percentage, RMSE, R2, and RPD.
Set \code{bycol = FALSE} to return only the aggregate metrics and avoid
calculating a separate metric vector for every response column.
}
\examples{
evaluate(iris$Species, iris$Species)

set.seed(1)
y <- mtcars$mpg
pred <- y + rnorm(length(y), sd = 2)
evaluate(y, pred)$metrics
}
