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
FastPCS<-function(x,nsamp=NULL,alpha=0.5){
	k1<-25
	k0<-25
	J<-3;

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
	fit<-.C("fastpcs",as.integer(nrow(x)),as.integer(ncol(x)),as.integer(k0),as.single(x),as.integer(k1),as.single(Dp),as.integer(nsamp),as.integer(J),as.single(objfunC),PACKAGE="FastPCS")
	outd<-as.numeric(fit[[6]])
	best<-which(outd<=quantile(outd,h/n,type=8))
	list(best=best,outi=outd,obj=as.numeric(fit[[9]]))
}
quanf<-function(n,p,alpha)	return(floor(2*floor((n+p+1)/2)-n+2*(n-floor((n+p+1)/2))*alpha))
