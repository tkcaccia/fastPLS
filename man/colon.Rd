\name{colon}
\alias{colon}
\docType{data}
\title{Colon cancer gene-expression example}
\usage{
data(colon)
}
\format{
A list with three elements:
\describe{
\item{\code{X}}{Numeric oligonucleotide microarray expression matrix with 62 colon tissue samples and 2000 gene-expression variables.}
\item{\code{y}}{Two-level diagnostic factor with classes \code{"normal"} (22 samples) and \code{"tumor"} (40 samples).}
\item{\code{gene_names}}{Character vector with the 2000 probe/gene identifiers.}
}
}
\description{
Colon cancer transcriptomic example for binary supervised modelling. The data
represent oligonucleotide microarray expression profiles measured from human
colon tissue specimens, comparing primary colon tumour samples with normal
colon mucosa samples. The bundled matrix contains \code{62} samples in total:
\code{40} tumour samples and \code{22} normal samples. Each row is one tissue
sample and each column is one gene-expression variable; the bundled version
contains \code{2000} gene-expression variables selected from the original
microarray study.
}
\source{
Derived from the \code{Colon} dataset distributed in the GPL (>= 2)
\pkg{plsgenomics} package. The original data are described in Alon et al.
(1999) and were originally collected from the Princeton oncology microarray
resource.
}
\references{
Alon U., Barkai N., Notterman D.A., Gish K., Ybarra S., Mack D. and Levine
A.J. (1999). Broad patterns of gene expression revealed by clustering analysis
of tumor and normal colon tissues probed by oligonucleotide arrays.
\emph{Proceedings of the National Academy of Sciences USA}, 96(12), 6745-6750.
}
\examples{
data(colon)
dim(colon$X)
table(colon$y)

fit <- pls(colon$X[, 1:100], colon$y, ncomp = 1, method = "plssvd",
           backend = "cpu", svd.method = "rsvd",
           return_variance = FALSE)
head(fit$Ypred)
}
\keyword{datasets}
