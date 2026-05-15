\name{ccle}
\alias{ccle}
\docType{data}
\title{CCLE cancer cell-line expression subset}
\usage{
data(ccle)
}
\format{
A list with five elements:
\describe{
\item{\code{X_train}}{Numeric expression matrix with 547 training cell-line samples and 1000 transcriptomic variables.}
\item{\code{y_train}}{Training tissue or lineage factor with 18 classes.}
\item{\code{X_test}}{Numeric expression matrix with 71 held-out cell-line samples and the same 1000 variables.}
\item{\code{y_test}}{Test tissue or lineage factor with the same class labels.}
\item{\code{metadata}}{List describing the benchmark subset, source, target label, and preprocessing notes.}
}
}
\description{
Cancer Cell Line Encyclopedia expression benchmark subset for supervised
classification. Rows are cancer cell-line samples and columns are selected
transcriptomic variables. The response labels describe tissue or lineage of
origin, giving a multi-class problem useful for testing PLS-DA speed and
classification accuracy.
}
\source{
Prepared from a local CCLE/DepMap-derived benchmark matrix. DepMap source
attribution and portal terms apply to downstream use; users should verify the
license attached to the exact DepMap/CCLE release used for redistribution.
}
\references{
Ghandi M. et al. (2019). Next-generation characterization of the Cancer Cell
Line Encyclopedia. \emph{Nature}, 569, 503-508.
}
\examples{
data(ccle)
dim(ccle$X_train)
table(ccle$y_train)

fit <- pls(ccle$X_train[, 1:50], ccle$y_train,
           ccle$X_test[, 1:50], ncomp = 2, method = "simpls",
           backend = "cpu", svd.method = "rsvd",
           return_variance = FALSE)
head(fit$Ypred)
}
\keyword{datasets}
