\name{fastcor}

\alias{fastcor}

\title{Fast Correlation Analysis}

\description{This function perform a fast calculation of the Spearman's correlation coefficient.}

\usage{
fastcor (a, b=NULL, byrow=TRUE, diag=TRUE)

}

\arguments{
  \item{a}{a matrix of training set cases.}
  \item{b}{a matrix of training set cases.}
  \item{byrow}{if byrow == T rows are correlated (much faster) else columns}
  \item{diag}{if diag == T only the diagonal of the cor matrix is returned (much faster).}

}

\value{
The output matrix of correlation coefficient.
  }

\author{Stefano Cacciatore, Leonardo Tenori, Dupe Ojo, Alessia Vignoli}

\seealso{\code{\link{optim.pls.cv}},\code{\link{pls.double.cv}}}

\examples{
data(iris)
data=as.matrix(iris[,-5])
fastcor(data)



}

