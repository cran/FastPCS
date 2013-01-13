\name{quanf}
\alias{quanf}
\title{Converts alpha values to h-values}
\usage{
  quanf(n,p,alpha)
}
\arguments{
 \item{n}{number of rows of the data matrix.}
  \item{p}{number of columns of the data matrix.}
\item{alpha}{numeric parameter controlling the size of the active subsets,
	 i.e., \code{"h=quanf(alpha,n,p)"}.  Allowed
          values are between 0.5 and 1 and the default is 0.5.}
}
\value{
  An integer number of starting p-subset.
}
\description{
  This is an internal function not intended to be called 
  by the user.
}
\examples{
quanf(p=3,n=500,alpha=0.5)
}
\author{
Kaveh Vakili
}
\keyword{multivariate}
\keyword{robust}

