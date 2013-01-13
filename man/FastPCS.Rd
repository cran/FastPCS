\name{FastPCS}
\alias{FastPCS}
\title{Computes the FastPCS multivariate outlyingness index.}
\usage{
  FastPCS(x,nsamp,alpha=0.5,seed=NULL)
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
\item{seed}{starting value for random generator. Default is seed = NULL}
}
\value{
\item{nsamp}{The value of nsamp used.}
\item{alpha}{The value of alpha used.}
\item{obj}{The value of the FastPCS objective function of the optimal h subset.}
\item{raw}{A list with the following components:}
\itemize{
    \item best: The index of the h observation with smallest outlyingness indexes.
    \item center: The mean vector of the h observations with smallest outlyingness index.
    \item cov: Covariance matrix of the h observations with smallest outlyingness index.
    \item distance: The statistical distance of each observation wrt the center vector and cov matrix of the h observations with smallest outlyingness index.
}
\item{rew}{A list with the following components:}
\itemize{
    \item best: The index of the J observation with outlyingness smaller than the rejection threshold.
    \item center: The mean vector of the J observations with outlyingness smaller than the rejection threshold.
    \item cov: Covariance matrix of the J observations with outlyingness smaller than the rejection threshold.
    \item distance: The statistical distance of each observation wrt the center vector and cov matrix of the J observations with outlyingness smaller than the rejection threshold.
}
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
nstart<-NumStarts(p=p,eps=0.4)
results<-FastPCS(x=x0,nsamp=nstart)
z[results$raw$best]

#testing exact fit
set.seed(123)
n<-100
p<-3
x0<-matrix(rnorm(n*p),nc=p)
x0[1:30,]<-matrix(rnorm(30*p,5,1/100),nc=p)
x0[31:100,3]<-x0[31:100,2]*2+1
z<-c(rep(0,30),rep(1,70))
nstart<-NumStarts(p=p,eps=0.4)
results<-FastPCS(x=x0,nsamp=nstart)
z[results$raw$best]
results$obj

#testing affine equivariance
n<-100
p<-3
set.seed(123)
x0<-matrix(rnorm(n*p),nc=p)
nstart<-500
results1<-FastPCS(x=x0,nsamp=nstart,seed=1)
a1<-matrix(0.9,p,p)
diag(a1)<-1
x1<-x0\%*\%a1
results2<-FastPCS(x=x1,nsamp=nstart,seed=1)
results2$raw$center
#should be the same
results1$raw$center\%*\%a1
}
\author{
	Kaveh Vakili
}
\keyword{multivariate}
\keyword{robust}

