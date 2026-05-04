\name{plot.fastPLSPCA}
\alias{plot.fastPLSPCA}
\alias{plot.fastPLS}
\title{Plot PCA or PLS scores}
\usage{
\method{plot}{fastPLSPCA}(
  x,
  comps = c(1L, 2L),
  groups = NULL,
  ellipse = FALSE,
  ellipse.type = c("confidence", "hotelling"),
  conf = 0.95,
  ...
)

\method{plot}{fastPLS}(
  x,
  comps = c(1L, 2L),
  groups = NULL,
  ellipse = FALSE,
  ellipse.type = c("confidence", "hotelling"),
  conf = 0.95,
  ...
)
}
\arguments{
\item{x}{A \code{fastPLSPCA} or \code{fastPLS} object.}
\item{comps}{Two component indices to plot.}
\item{groups}{Optional grouping vector for color and grouped ellipses.}
\item{ellipse}{Logical; draw confidence ellipses when \code{TRUE}.}
\item{ellipse.type}{Either \code{"confidence"} for a covariance confidence ellipse or \code{"hotelling"} for a Hotelling's T2 ellipse.}
\item{conf}{Confidence level.}
\item{...}{Additional arguments passed to \code{plot()}.}
}
\value{
Invisibly returns the plotted score matrix.
}
\description{
Draws a two-component PCA or PLS score plot. Ellipses can be drawn globally or
per group, using either a standard data confidence ellipse or a Hotelling's T2
score ellipse. Grouped points use filled plotting symbols by default, with the
group color assigned to \code{bg} and a black point contour assigned to
\code{col}.
}
