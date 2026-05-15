\name{breast}
\alias{breast}
\docType{data}
\title{Breast-cancer mRNA subtype example}
\usage{
data(breast)
}
\format{
A list with four elements:
\describe{
\item{\code{X_train}}{Numeric mRNA expression matrix with 150 training tumour samples and 200 gene-expression variables.}
\item{\code{y_train}}{Training molecular subtype factor with three breast-cancer classes: \code{"Basal"} (45 samples), \code{"Her2"} (30 samples), and \code{"LumA"} (75 samples).}
\item{\code{X_test}}{Numeric mRNA expression matrix with 70 independent test tumour samples and the same 200 gene-expression variables.}
\item{\code{y_test}}{Test molecular subtype factor with the same three classes: \code{"Basal"} (21 samples), \code{"Her2"} (14 samples), and \code{"LumA"} (35 samples).}
}
}
\description{
Breast-cancer mRNA expression example for supervised multivariate modelling.
The data contain tumour samples assigned to three clinically relevant molecular
subtypes: basal-like breast cancer, HER2-enriched breast cancer, and luminal A
breast cancer. The bundled version keeps one transcriptomic block only
(\code{200} gene-expression variables) and provides a fixed train/test split:
\code{150} training samples and \code{70} test samples, for \code{220} samples
in total. The class balance is preserved across the split, with \code{66}
Basal, \code{44} Her2, and \code{110} LumA samples overall.
}
\source{
Derived from the \code{breast.TCGA} dataset distributed in the GPL (>= 2)
\pkg{mixOmics} package. The data are a filtered TCGA breast-cancer multi-omics
example used in the DIABLO framework.
}
\references{
Singh A., Shannon C., Gautier B., Rohart F., Vacher M., Tebbutt S. and
Le Cao K.A. (2019). DIABLO: an integrative approach for identifying key
molecular drivers from multi-omics assays. \emph{Bioinformatics}, 35(17),
3055-3062.
}
\examples{
data(breast)
dim(breast$X_train)
table(breast$y_train)

fit <- pls(breast$X_train[, 1:50], breast$y_train,
           breast$X_test[, 1:50], ncomp = 2, method = "simpls",
           backend = "cpu", svd.method = "rsvd",
           return_variance = FALSE)
head(fit$Ypred)
}
\keyword{datasets}
