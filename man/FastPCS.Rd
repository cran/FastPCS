\name{FastPCS}
\alias{FastPCS}
\title{Computes the FastPCS multivariate outlyingness index.}
\usage{
  FastPCS(x,nsamp,alpha=0.5)
}
\arguments{
  \item{x}{A numeric n (n>5*p) by p (p>1) matrix or data frame.}
   \item{nsamp}{a positive integer giving the number of resamples required;
          \code{"nsamp"} may not be reached if too many of the p-subsamples,
          chosen out of the observed vectors, are in a hyperplane.  If \code{"nsamp"} is
          omitted, it is calculated to provide a breakdown point of
          \code{"alpha"} with probability 0.99.}
 \item{alpha}{numeric parameter controlling the size of the active subsets,
	 i.e., \code{"h=quanf(alpha,n,p)"}.  Allowed
          values are between 0.5 and 1 and the default is 0.5.}
}
\value{
 A  list with the following
  components:

  \item{best}{The index of the h members of the H* (the optimal h-subset selected by the FastPCS algorithm).}

  \item{out}{A numeric n-vector containing, for each observation, the value of the FastPCS 
	outlyingness index.}

  \item{obj}{The value of the FastPCS objective function computed on the selected h-subset.}

}
\description{
   Computes a fast and robust multivariate outlyingness index for a n by p matrix of 
	multivariate continuous data.
}
\examples{
## generate data
set.seed(123)
n<-100
p<-3
x0<-matrix(rnorm(n*p),nc=p)
x0[1:30,]<-matrix(rnorm(30*p,4.5,1/100),nc=p)
z<-c(rep(0,30),rep(1,70))
nstart<-NumStarts(p=p)
results<-FastPCS(x=x0,nsamp=nstart)
z[results$best]

#testing exact fit
set.seed(123)
n<-100
p<-3
x0<-matrix(rnorm(n*p),nc=p)
x0[1:30,]<-matrix(rnorm(30*p,4.5,1/100),nc=p)
x0[31:100,3]<-x0[31:100,2]
z<-c(rep(0,30),rep(1,70))
nstart<-NumStarts(p=p)
results<-FastPCS(x=x0,nsamp=nstart)
z[results$best]
results$obj
}
\author{
	Kaveh Vakili
}
\keyword{multivariate}
\keyword{robust}

