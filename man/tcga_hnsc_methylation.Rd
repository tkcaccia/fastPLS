\name{tcga_hnsc_methylation}
\alias{tcga_hnsc_methylation}
\docType{data}
\title{TCGA-HNSC DNA methylation tumour/normal subset}
\usage{
data(tcga_hnsc_methylation)
}
\format{
A list with five elements:
\describe{
\item{\code{X_train}}{Numeric DNA methylation matrix with 520 training samples and 782 CpG variables.}
\item{\code{y_train}}{Training sample-type factor with tumour and solid-tissue normal classes.}
\item{\code{X_test}}{Numeric DNA methylation matrix with 58 held-out samples and the same 782 variables.}
\item{\code{y_test}}{Test sample-type factor with the same class labels.}
\item{\code{metadata}}{List describing the benchmark subset, source, target label, and preprocessing notes.}
}
}
\description{
TCGA head and neck squamous cell carcinoma DNA methylation benchmark subset.
Rows are tissue samples and columns are selected CpG methylation variables. The
response distinguishes primary tumour samples from solid-tissue normal samples.
}
\source{
Prepared from a local TCGA-HNSC/Xena-derived benchmark matrix. The underlying
TCGA data are open-access derived data from the NCI Genomic Data Commons; NIH,
GDC, TCGA, and Xena attribution and data-use requirements apply.
}
\references{
The Cancer Genome Atlas Network (2015). Comprehensive genomic characterization
of head and neck squamous cell carcinomas. \emph{Nature}, 517, 576-582.
}
\examples{
data(tcga_hnsc_methylation)
dim(tcga_hnsc_methylation$X_train)
table(tcga_hnsc_methylation$y_train)

fit <- pls(tcga_hnsc_methylation$X_train[, 1:50],
           tcga_hnsc_methylation$y_train,
           tcga_hnsc_methylation$X_test[, 1:50],
           ncomp = 2, method = "simpls",
           backend = "cpu", svd.method = "rsvd",
           return_variance = FALSE)
head(fit$Ypred)
}
\keyword{datasets}
