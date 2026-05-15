\name{tcga_brca}
\alias{tcga_brca}
\docType{data}
\title{TCGA-BRCA breast-cancer subtype subset}
\usage{
data(tcga_brca)
}
\format{
A list with five elements:
\describe{
\item{\code{X_train}}{Numeric tumour expression matrix with 756 training samples and 1000 selected variables.}
\item{\code{y_train}}{Training PAM50 subtype factor with five classes.}
\item{\code{X_test}}{Numeric tumour expression matrix with 88 held-out samples and the same 1000 variables.}
\item{\code{y_test}}{Test PAM50 subtype factor with the same class labels.}
\item{\code{metadata}}{List describing the benchmark subset, source, target label, and preprocessing notes.}
}
}
\description{
TCGA breast invasive carcinoma expression benchmark subset. Rows are BRCA tumour
samples and columns are selected gene-expression variables. The response is the
PAM50 RNA-seq molecular subtype, including basal-like, HER2-enriched, luminal A,
luminal B, and normal-like labels.
}
\source{
Prepared from a local TCGA-BRCA/Xena-derived benchmark matrix. The underlying
TCGA data are open-access derived data from the NCI Genomic Data Commons; NIH,
GDC, TCGA, and Xena attribution and data-use requirements apply.
}
\references{
Cancer Genome Atlas Network (2012). Comprehensive molecular portraits of human
breast tumours. \emph{Nature}, 490, 61-70.
}
\examples{
data(tcga_brca)
dim(tcga_brca$X_train)
table(tcga_brca$y_train)

fit <- pls(tcga_brca$X_train[, 1:50], tcga_brca$y_train,
           tcga_brca$X_test[, 1:50], ncomp = 2, method = "plssvd",
           backend = "cpu", svd.method = "rsvd",
           return_variance = FALSE)
head(fit$Ypred)
}
\keyword{datasets}
