#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <fstream>
#include <iostream>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <vector>
#include <random>

#include <Eigen/QR>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/Cholesky>

using namespace std;
using namespace Eigen;
using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::VectorXi;
using Eigen::RowVectorXf;
std::mt19937 mt;

struct IdLess {					//internal function.
    template <typename T>
    IdLess(T iter) : values(&*iter) {}
    bool operator()(int left,int right){
        return values[left]<values[right];
    }
    float const* values;
};
double GetUniform(){
    static std::uniform_real_distribution<double> Dist(0,1);
    return Dist(mt);
}
void GetSmallest(const VectorXf& r,const int& h,const MatrixXf& x,MatrixXf& xSub,VectorXi& RIndex){
	const int n=x.rows();
	VectorXi SIndx2(n);
	SIndx2.setLinSpaced(n,0,n-1);
	std::nth_element(SIndx2.data(),SIndx2.data()+h,SIndx2.data()+SIndx2.size(),IdLess(r.data()));
	for (int i=0;i<h;i++) 	xSub.row(i)=x.row(SIndx2(i));
	RIndex.head(h)=SIndx2.head(h);	
}
VectorXi SampleR(const int m,const int p){
	int i,j,nn=m;
	VectorXi ind(nn);
	VectorXi y(p);
	ind.setLinSpaced(nn,0,nn-1);
    	for(i=0;i<p;i++){
		j=GetUniform()*nn;
		y(i)=ind(j);
		ind(j)=ind(--nn);
    	}
	return y;		
}
VectorXf FindLine(const MatrixXf& xSub,const int h){
	const int p=xSub.cols();
	VectorXi  QIndexp(p);
	QIndexp=SampleR(h,p);
	VectorXf bt=VectorXf::Ones(p);
	MatrixXf A(p,p);
	for(int i=0;i<p;i++)	A.row(i)=xSub.row(QIndexp(i));
	return(A.lu().solve(bt));
}
VectorXf OneProj(const MatrixXf& x,const MatrixXf& xSub,const int h,const VectorXi& RIndex,const int h_m){
	const int p=x.cols(),n=x.rows();
	VectorXf beta(p);
	VectorXf praj(n);
	VectorXf prej(h);
	beta=FindLine(xSub,h);
	praj=((x*beta).array()-1.0f).array().abs2();
	praj/=beta.squaredNorm();
	for(int i=0;i<h;i++)	prej(i)=praj(RIndex(i));
	float prem=prej.head(h).mean(),tol=1e-7;
	if(prem<tol){	
		const int n=praj.size();
		VectorXf d_resd=VectorXf::Zero(n);
		d_resd=(praj.array()<tol).select(1.0f,d_resd);
		if((d_resd.sum())>=h_m){
			prem=1.0f;
		} else {
			float maxin=praj.maxCoeff();
			d_resd=(praj.array()<tol).select(maxin,praj);
			prem=d_resd.minCoeff();
		}
	}
	return praj/=prem;
}
float SubsetRankFun(const MatrixXf& x,const MatrixXf& xSub,const int h,const VectorXi& RIndex){
	const int p=x.cols(),n=x.rows();
	VectorXf prej(h);
	VectorXf beta(p);
	VectorXf praj(n);
	beta=FindLine(xSub,h);
	praj=((x*beta).array()-1.0f).array().abs2();
	praj/=beta.squaredNorm();
	for(int i=0;i<h;i++)	prej(i)=praj(RIndex(i));
	nth_element(praj.data(),praj.data()+h,praj.data()+praj.size());	
	float prem=praj.head(h).mean(),fin=(prem>1e-7)?(prej.head(h).mean()/prem):(1.0f);
	return fin;
}
float Main(const MatrixXf& x,const int k0,const int J,const int k1,VectorXf& dP,const int h_m,VectorXi& samset,const VectorXi& hl){
	int p=x.cols(),n=x.rows(),h=p+1,ni=samset.size();
	MatrixXf xSub(h_m,p);
	VectorXf fin(k1);
	VectorXi RIndex(n);
	RIndex.head(h)=SampleR(ni,h);
	for(int i=0;i<h;i++) xSub.row(i)=x.row(samset(RIndex(i)));			
	for(int j=0;j<J;j++){					//growing step
		dP=VectorXf::Zero(n);
		for(int i=0;i<k0;i++) dP+=OneProj(x,xSub,hl(j),RIndex,h_m);
		GetSmallest(dP,hl(j+1),x,xSub,RIndex);
	}
	for(int i=0;i<k1;i++) fin(i)=SubsetRankFun(x,xSub,hl(J),RIndex);
	return fin.array().log().mean();
}
VectorXi CStep(VectorXf& dP,MatrixXf& x,const int h){
	const int n=x.rows(),p=x.cols();
	float w1,w0;
	int w2=1,i;
	MatrixXf xSub(h,p);
	MatrixXf b=MatrixXf::Identity(p,p);
	RowVectorXf xSub_mean(p);
	VectorXi dIn(n);
	MatrixXf Sig(p,p);

	dIn.setLinSpaced(n,0,n-1);
	std::nth_element(dIn.data(),dIn.data()+h,dIn.data()+dIn.size(),IdLess(dP.data()));
	for(i=0;i<h;i++) 	xSub.row(i)=x.row(dIn(i));
	xSub_mean=xSub.colwise().mean();	
	xSub.rowwise()-=xSub_mean;
	x.rowwise()-=xSub_mean;
	Sig=xSub.adjoint()*xSub;
	Sig.array()/=(float)(h-1);
	LDLT<MatrixXf> chol=Sig.ldlt();
	chol.solveInPlace(b);
	w1=chol.vectorD().array().minCoeff();
	if(w1>1e-6){
		w1=chol.vectorD().array().log().sum()*2.0f;
		dP=((x*b).cwiseProduct(x)).rowwise().sum();
	} else {
		w2=0;
	}
	while(w2){	
		dIn.setLinSpaced(n,0,n-1);
		std::nth_element(dIn.data(),dIn.data()+h,dIn.data()+dIn.size(),IdLess(dP.data()));
		for(i=0;i<h;i++) 	xSub.row(i)=x.row(dIn(i));
		xSub_mean=xSub.colwise().mean();	
		xSub.rowwise()-=xSub_mean;
		x.rowwise()-=xSub_mean;
		Sig=xSub.adjoint()*xSub;
		Sig.array()/=(float)(h-1);
		LDLT<MatrixXf> chol=Sig.ldlt();
		b=MatrixXf::Identity(p,p);
		chol.solveInPlace(b);
		if(chol.vectorD().array().minCoeff()>1e-6){
			w0=w1;
			w1=chol.vectorD().array().log().sum()*2.0f;
			dP=((x*b).cwiseProduct(x)).rowwise().sum();
			(w0-w1<1e-3)?(w2=0):(w2=1);
		} else {
			w2=0;
		}
	}
	return(dIn.head(h).array()+1);
} 
extern "C"{
	void fastpcs(int* n,int* p,int* k0,float* xi,int* k1,float* DpC,int* nsamp,int* J,float* objfunC,int* seed,int* ck,int* ni,int* n1,int* n2,int* hm){
		const int ik0=*k0,iJ=*J,ik1=*k1,ih_m=*hm,iseed=*seed;
		float objfunA,objfunB=*objfunC;
		mt.seed(*seed);
		MatrixXf x=Map<MatrixXf>(xi,*n,*p);	
		VectorXi icK=Map<VectorXi>(ck,*ni);
		VectorXf DpA=VectorXf::Zero(*n);
		VectorXf DpB=VectorXf::Zero(*n);
		VectorXi dpH(*hm);
		VectorXi hl(*J+1);

		hl.setLinSpaced(*J+1,*p+1,ih_m);
		for(int i=0;i<*nsamp;i++){			//for i=0 to i<#p-subsets.
			objfunA=Main(x,ik0,iJ,ik1,DpA,ih_m,icK,hl);
			if(objfunA<objfunB){
				objfunB=objfunA;
				DpB=DpA;
			}
		}
		dpH.setLinSpaced(*n,0,*n-1);
		std::nth_element(dpH.data(),dpH.data()+ih_m,dpH.data()+dpH.size(),IdLess(DpB.data()));
		Map<VectorXi>(n1,*hm)=dpH.head(ih_m).array()+1;		
 		Map<VectorXf>(DpC,*n)=DpB.array();
		dpH=CStep(DpB,x,ih_m);
		Map<VectorXi>(n2,*hm)=dpH.array();
		*objfunC=objfunB;
	}
}
