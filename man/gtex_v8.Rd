\name{gtex_v8}
\alias{gtex_v8}
\docType{data}
\title{GTEx v8 tissue-expression subset}
\usage{
data(gtex_v8)
}
\format{
A list with five elements:
\describe{
\item{\code{X_train}}{Numeric RNA-seq expression matrix with 3000 training samples and 1000 selected variables.}
\item{\code{y_train}}{Training primary-site factor with 32 tissue labels.}
\item{\code{X_test}}{Numeric RNA-seq expression matrix with 797 held-out samples and the same 1000 variables.}
\item{\code{y_test}}{Test primary-site factor with the same tissue labels.}
\item{\code{metadata}}{List describing the benchmark subset, source, target label, and preprocessing notes.}
}
}
\description{
GTEx v8 tissue-expression benchmark subset for multi-class supervised
modelling. Rows are human bulk RNA-seq tissue samples and columns are selected
gene-expression variables. The response is the primary tissue site, making the
data useful for evaluating high-dimensional PLS classification across many
biological tissues.
}
\source{
Prepared from a local GTEx v8/Xena-derived benchmark matrix. GTEx open-access
data are de-identified public research data; GTEx/NIH data-use, citation, and
acknowledgement requirements apply.
}
\references{
The GTEx Consortium (2020). The GTEx Consortium atlas of genetic regulatory
effects across human tissues. \emph{Science}, 369, 1318-1330.
}
\examples{
data(gtex_v8)
dim(gtex_v8$X_train)
table(gtex_v8$y_train)[1:5]

fit <- pls(gtex_v8$X_train[1:200, 1:50], gtex_v8$y_train[1:200],
           gtex_v8$X_test[1:50, 1:50], ncomp = 2, method = "simpls",
           backend = "cpu", svd.method = "rsvd",
           return_variance = FALSE)
head(fit$Ypred)
}
\keyword{datasets}
