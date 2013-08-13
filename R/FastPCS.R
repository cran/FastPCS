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
	k1<-25;k0<-25;J<-3;
	if(is.null(seed))	seed<-floor(runif(1,-2^31,2^31))
	seed<-as.integer(seed)+1
	x<-data.matrix(x)
	na.x<-complete.cases(x)
	if(!is.numeric(alpha))	stop("alpha should be numeric")
	if(alpha<0.5 | alpha>=1)	stop("alpha should be in (0.5,1(.")
	if(sum(na.x)!=nrow(x))  stop("Your data contains NA.")
	if(nrow(x)<(5*ncol(x))) stop("n<5p. You need more observations")
	n<-nrow(x)
	p<-ncol(x)
	chechidenc<-rep(NA,p)
	for(i in 1:p)		chechidenc[i]<-min(colMedians(abs(sweep(x[,-i,drop=FALSE],1,x[,i],FUN="-"))))
	if(min(chechidenc)<1e-8){
		wones<-which(chechidenc<1e-8)
		stop(paste0("Columns ",wones[1]," and ",wones[2]," contain at least n/2 identicial observations."))
	}
	if(p<2)		stop("Univariate PCS is not implemented.")
	if(p>25)		stop("FastPCS only works for dimensions<=25.")
	if(is.null(nsamp)) 	nsamp<-NumStarts(p,eps=(1-alpha)) 
	h<-quanf(n=n,p=p,alpha=alpha)
	h0<-quanf(n=n,p=p,alpha=0.5)
	Dp<-rep(1.00,n);
	k0<-max(k0,p+1);
	k1<-max(k1,p+1);
	objfunC<-1e3;
	icandid<-1:n-1
	ni<-length(icandid)
	fitd<-.C("fastpcs",as.integer(nrow(x)),as.integer(ncol(x)),as.integer(k0),as.single(x),as.integer(k1),as.single(Dp),as.integer(nsamp),as.integer(J),as.single(objfunC),as.integer(seed),as.integer(icandid),as.integer(ni),PACKAGE="FastPCS")
	outd<-as.numeric(fitd[[6]])
	if(is.nan(outd)[1])	stop("too many singular subsets encoutered!")
	best<-which(outd<=median(outd))
	invc<-svd(cov(x[best,]),nu=0,nv=0)
	if(min(invc$d)>1e-8){
		solV<-chol2inv(chol(cov(x[best,])));
		stds<-mahalanobis(x,colMeans(x[best,]),solV,inverted=TRUE)
		best<-which(stds<=quantile(stds,h/n))
		rawF<-list(best=best,distance=sqrt(stds),center=colMeans(x[best,]),cov=cov(x[best,]))
		solV<-chol2inv(chol(cov(x[best,])));
		stds<-mahalanobis(x,colMeans(x[best,]),solV,inverted=TRUE)
		best<-which(stds<=qchisq(0.975,df=p))
		solV<-chol2inv(chol(cov(x[best,])));
		stds<-sqrt(mahalanobis(x,colMeans(x[best,]),solV,inverted=TRUE))
		rewF<-list(best=best,distance=stds,center=colMeans(x[best,]),cov=cov(x[best,]))
	} else {
		"%ni%"<-Negate("%in%") 
		resd<-as.numeric((1:n)%ni%best)
		rewF<-list(distance=resd,best=best,center=colMeans(x[best,]),cov=cov(x[best,]))
		rawF<-rewF
		print("FastPCS has found n/2 observations on a subspace.")
	}
	list(alpha=alpha,nsamp=nsamp,raw=rawF,obj=as.numeric(fitd[[9]]),rew=rewF)
}
quanf<-function(n,p,alpha)	return(floor(2*floor((n+p+1)/2)-n+2*(n-floor((n+p+1)/2))*alpha))
