NumStarts<-function(p,gamma=0.99,eps=0.5){
	if(p>25)	stop("p too large.")
	if(gamma>=1)	stop("gamma should be smaller than 1.")
	if(gamma<=0)	stop("gamma should be larger than 0.")
	if(eps>0.5)	stop("eps should be smaller than 1/2.")
	if(eps<=0)	stop("eps should be larger than 0.")	
	ns0<-ceiling(log(1-gamma)/log(1-(1-(eps))^(p+1)))
	ord<-10^floor(log10(ns0))
	ceiling(ns0/ord)*ord
}
FastPCS<-function(x,nsamp=NULL,alpha=0.5,seed=NULL){
	k1<-25
	k0<-25
	J<-3;
	if(is.null(seed))	seed<-floor(runif(1,-2^31,2^31))
	seed<-as.integer(seed)
	x<-data.matrix(x)
	na.x<-complete.cases(x)
	if(sum(na.x)!=nrow(x))  stop("Your data contains NA.")
	if(nrow(x)<(5*ncol(x))) stop("n<5p. You need more observations")
	n<-nrow(x)
	p<-ncol(x)
	if(p<2)		stop("Univariate PCS is not implemented.")
	if(p>25)		stop("FastPCS only works for dimensions <=25.")
	if(is.null(nsamp)) 	nsamp<-NumStarts(p,eps=(1-alpha)) 
	h<-quanf(n=n,p=p,alpha=alpha)
	Dp<-rep(1.00,n);
	k0<-max(k0,p+1);
	k1<-max(k1,p+1);
	objfunC<-1e3;
	fit<-.C("fastpcs",as.integer(nrow(x)),as.integer(ncol(x)),as.integer(k0),as.single(x),as.integer(k1),as.single(Dp),as.integer(nsamp),as.integer(J),as.single(objfunC),as.integer(seed),PACKAGE="FastPCS")
	outd<-as.numeric(fit[[6]])
	best<-which(outd<=median(outd))
	if(min(svd(var(x[best,]))$d)>1e-8){
		resd<-mahalanobis(x,colMeans(x[best,]),var(x[best,]))
		best<-which(resd<=quantile(resd,h/n))
		RWes<-RW(x0=x,best,h=h)
	} else {
		"%ni%"<-Negate("%in%") 
		resd<-as.numeric((1:n)%ni%best)
		RWes<-"FastPCS has found n/2 observations on a subspace."
	}
	list(raw.best=best,raw.outlyingness=sqrt(resd),obj=as.numeric(fit[[9]]),model=RWes)
}
quanf<-function(n,p,alpha)	return(floor(2*floor((n+p+1)/2)-n+2*(n-floor((n+p+1)/2))*alpha))
RW<-function(x0,best,h){
	n<-nrow(x0)
	p<-ncol(x0)
	a2<-mahalanobis(x0,colMeans(x0[best,]),cov(x0[best,]))
	a4<-(qchisq(0.975,df=p)*quantile(a2,(h-p)/n))/qchisq((h-p)/n,p)
	a3<-which(a2<=a4)
	a5<-mahalanobis(x0,colMeans(x0[a3,]),cov(x0[a3,]))
	list(rew.outlyingness=a5,cov=cov(x0[a3,]),center=colMeans(x0[a3,]),rew.best=a3)
}
