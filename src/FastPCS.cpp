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

#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/SVD>

using namespace std;
using namespace Eigen;
using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::VectorXi;
using Eigen::RowVectorXf;

struct IdLess {					//internal function.
    template <typename T>
    IdLess(T iter) : values(&*iter) {}
    bool operator()(int left,int right){
        return values[left]<values[right];
    }
    float const* values;
};
double Binomial(int n,int k){
	if (k==0||n==k) 	return 1;
	if (n<=0||k<0||n<k) 	return 0;
	int k1=(k<n-k)?k:n-k;
	int k2=n-k1;
	double fact=k2+1;
	for(double i=k1;i>1.;--i) fact*=(k2+i)/i;
	return fact;
}
void GetSmallest(const VectorXf& r,int h,const MatrixXf& x,MatrixXf& xSub,VectorXi& RIndex){
	const int n=x.rows();
	VectorXi SIndx2(n);
	SIndx2.setLinSpaced(n,0,n-1);
	std::nth_element(SIndx2.data(),SIndx2.data()+h,SIndx2.data()+SIndx2.size(),IdLess(r.data()));
	for (int i=0;i<h;i++) 	xSub.row(i)=x.row(SIndx2(i));
	RIndex.head(h)=SIndx2.head(h);	
}
VectorXi SampleR(const int& m,const int& p){
	int i,j,n=m;
	VectorXi x(n);
	VectorXi y(p);
	x.setLinSpaced(n,0,n-1);
	VectorXf urd=VectorXf::Random(p).array().abs();
	--n;
	for(i=0;i<p;i++){
		j=n*urd(i);
		y(i)=x(j);
		--n;
		x(j)=x(n);
    	}
	return y;		
}
VectorXf FindLine(const MatrixXf& xSub,const int& h){
	const int p=xSub.cols();
	VectorXi QIndexp=SampleR(h,p);
	VectorXf bt=VectorXf::Ones(p);
	MatrixXf A(p,p);
	for(int i=0;i<p;i++)	A.row(i)=xSub.row(QIndexp(i));
	return(A.lu().solve(bt));
}
VectorXf OneProj(const MatrixXf& x,const MatrixXf& xSub,const int& h,const VectorXi& RIndex){
	VectorXf praj=((x*FindLine(xSub,h)).array()-1).array().abs2();
	VectorXf prej(h);
	for(int i=0;i<h;i++)	prej(i)=praj(RIndex(i));
	float prem=prej.head(h).mean();
	if(prem>1e-8)	praj/=prem;
	return praj;
}
float SubsetRankFun(const MatrixXf& x,const MatrixXf& xSub,const int& h,const VectorXi& RIndex){
	VectorXf praj=((x*FindLine(xSub,h)).array()-1).array().abs2();
	VectorXf proj=praj;
	VectorXf prej(h);
	float fin=1, prem;
	nth_element(proj.data(),proj.data()+h,proj.data()+proj.size());	
	for(int i=0;i<h;i++)	prej(i)=praj(RIndex(i));
	prem=proj.head(h).mean();
	if(prem>1e-8)	fin=prej.head(h).mean()/prem;
	return fin;
}
float Main(MatrixXf& x,const int& h_i,const int& k0,const int& J,const int& k1,VectorXf& dP,const int& h_m){
	int p=x.cols(),n=x.rows(),h=p+1;
	RowVectorXf xSub_mean(p);
	MatrixXf xSub(h_m,p);
	VectorXf fin(k1);
	VectorXi RIndex(n);
	VectorXi hl;
	RIndex.head(h)=SampleR(n,h);
	for(int i=0;i<h;i++) xSub.row(i)=x.row(RIndex(i));			
	xSub_mean=xSub.topRows(h).colwise().mean();	
	xSub.rowwise()-=xSub_mean;
	x.rowwise()-=xSub_mean;					//	
	MatrixXf Sig=xSub.topRows(h).adjoint()*xSub.topRows(h);
	Sig.array()/=(float)(h-1);
	JacobiSVD<MatrixXf> svd(Sig,ComputeThinV);
	Sig=svd.matrixV()*(svd.singularValues().asDiagonal().inverse())*svd.matrixV().transpose();		
	dP=((x*Sig).array()*x.array()).rowwise().sum();		//
	hl.setLinSpaced(J+1,h_i,h_m);
	h=hl(0);
	GetSmallest(dP,h,x,xSub,RIndex);			//finds the members of h_0
	for(int j=0;j<J;j++){					//growing step
		dP=VectorXf::Zero(n);
		for(int i=0;i<k0;i++) dP+=OneProj(x,xSub,h,RIndex);
		h=hl(j+1);
		GetSmallest(dP,h,x,xSub,RIndex);
	}
	for(int i=0;i<k1;i++) fin(i)=SubsetRankFun(x,xSub,h,RIndex);
	return fin.array().log().mean(); 
}
extern "C"{
	void fastpcs(int* n,int* p,int* k0,float* xi,int* k1,float* DpC,int* nsamp,int* J,float* objfunC,int* seed){
		const int ik0=*k0,iJ=*J,ik1=*k1,ih_m=(*n+*p+1)/2;
		int h_i=*p+1,h,j,i,iseed=*seed; 
		float objfunA,objfunB=*objfunC;

		while(Binomial(h_i,*p)<*k0) h_i++;	//determines h_0
		MatrixXf x=Map<MatrixXf>(xi,*n,*p);	
		VectorXf DpA=VectorXf::Zero(*n);
		VectorXf DpB=VectorXf::Zero(*n);

		for(i=0;i<*nsamp;i++){			//for i=0 to i<#p-subsets.
			objfunA=Main(x,h_i,ik0,iJ,ik1,DpA,ih_m);
			if(objfunA<objfunB){
				objfunB=objfunA;
				DpB=DpA;
			}
		}
 		Map<VectorXf>(DpC,*n)=DpB.array();
		*objfunC=objfunB;
	}
}
