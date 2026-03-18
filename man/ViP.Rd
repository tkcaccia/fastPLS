\name{ViP}
\alias{ViP}
\title{Variable Importance in Projection (VIP) Scores}

\description{
Compute VIP trajectories across components from a fitted \code{fastPLS} model.
}

\usage{
ViP(model)
}

\arguments{
\item{model}{A fitted model from \code{\link{pls}} or \code{\link{pls_r}}.}
}

\details{
This implementation uses model components \code{Q}, \code{Ttrain}, and \code{R}
through an internal helper \code{Vip()}.

For multi-response models, the function returns one VIP matrix per response.
For single-response models, it returns a numeric matrix of VIP values across
components.
}

\value{
VIP values by predictor and component:
\itemize{
\item single-response: numeric matrix.
\item multi-response: list of numeric matrices (one per response column).
}
}

\references{
Wold, S., Johansson, E., and Cocchi, M. (1993).
PLS --- partial least-squares projections to latent structures.
In \emph{3D QSAR in Drug Design}. ESCOM.
}

\seealso{\code{\link{pls}}}

\examples{
set.seed(1)
data(iris)
X <- as.matrix(iris[, -5])
y <- iris[, 5]
fit <- pls(X, y, ncomp = 3)
ViP(fit)
}
