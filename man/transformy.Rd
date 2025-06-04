\name{transformy}

\alias{transformy}

\title{Conversion Classification Vector to Matrix}

\description{This function converts a classification vector into a classification matrix.}

\usage{
transformy(y)
}

\arguments{
  \item{y}{a vector or factor.}
}

\details{
	This function converts a classification vector into a classification matrix. Different groups are compared amongst each other.
}


\value{
    A matrix.
  }


\author{Dupe Ojo, Alessia Vignoli, Stefano Cacciatore, Leonardo Tenori}


  \examples{
y=c(1,1,1,1,2,2,2,3,3)
print(y)
z=transformy(y)
print(z)
}

\keyword{transformation}