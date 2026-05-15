\name{tcga_pan_cancer}
\alias{tcga_pan_cancer}
\docType{data}
\title{TCGA pan-cancer expression subset}
\usage{
data(tcga_pan_cancer)
}
\format{
A list with five elements:
\describe{
\item{\code{X_train}}{Numeric expression matrix with 3000 training tumour samples and 850 selected variables.}
\item{\code{y_train}}{Training primary-disease factor with 32 cancer classes.}
\item{\code{X_test}}{Numeric expression matrix with 982 held-out tumour samples and the same 850 variables.}
\item{\code{y_test}}{Test primary-disease factor with the same cancer labels.}
\item{\code{metadata}}{List describing the benchmark subset, source, target label, and preprocessing notes.}
}
}
\description{
TCGA pan-cancer expression benchmark subset for multi-class tumour-type
classification. Rows are tumour samples from multiple TCGA cancer projects and
columns are selected gene-expression variables. The response is the primary
disease entity.
}
\source{
Prepared from a local TCGA pan-cancer/Xena-derived benchmark matrix. The
underlying TCGA data are open-access derived data from the NCI Genomic Data
Commons; NIH, GDC, TCGA, and Xena attribution and data-use requirements apply.
}
\references{
Hoadley K.A. et al. (2018). Cell-of-origin patterns dominate the molecular
classification of 10,000 tumors from 33 types of cancer. \emph{Cell}, 173,
291-304.e6.
}
\examples{
data(tcga_pan_cancer)
dim(tcga_pan_cancer$X_train)
table(tcga_pan_cancer$y_train)[1:5]

target_levels <- levels(tcga_pan_cancer$y_train)[1:3]
keep <- match(as.character(tcga_pan_cancer$y_train), target_levels,
              nomatch = 0L) > 0L
train_id <- which(keep)[1:90]
y_train <- droplevels(tcga_pan_cancer$y_train[train_id])
fit <- pls(tcga_pan_cancer$X_train[train_id, 1:50],
           y_train,
           tcga_pan_cancer$X_test[1:50, 1:50],
           ncomp = 2, method = "simpls",
           backend = "cpu", svd.method = "rsvd",
           classifier = "argmax",
           return_variance = FALSE)
head(fit$Ypred)
}
\keyword{datasets}
