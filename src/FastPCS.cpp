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

using namespace std;
using namespace Eigen;
using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::VectorXi;

struct IdLess {					//internal function.
    template <typename T>
    IdLess(T iter) : values(&*iter) {}
    bool operator()(int left,int right){
        return values[left]<values[right];
    }
    float const* values;
};
void GetSmallest(const VectorXf& r,int h,const MatrixXf& x,MatrixXf& xSub,VectorXi& RIndex){
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
		j=rand()%nn;
		y(i)=ind(j);
		ind(j)=ind(--nn);
    	}
	return y;		
}
VectorXf FindLine(const MatrixXf& xSub,const int h){
	const int p=xSub.cols();
	VectorXi QIndexp=SampleR(h,p);
	VectorXf bt=VectorXf::Ones(p);
	MatrixXf A(p,p);
	for(int i=0;i<p;i++)	A.row(i)=xSub.row(QIndexp(i));
	return(A.lu().solve(bt));
}
VectorXf OneProj(const MatrixXf& x,const MatrixXf& xSub,const int h,const VectorXi& RIndex){
	VectorXf beta=FindLine(xSub,h);
	VectorXf praj=((x*beta).array()-1.0).array().abs2();
	praj/=beta.squaredNorm();
	VectorXf prej(h);
	for(int i=0;i<h;i++)	prej(i)=praj(RIndex(i));
	float prem=prej.head(h).mean();
	if(prem>1e-8)	praj/=prem;
	return praj;
}
float SubsetRankFun(const MatrixXf& x,const MatrixXf& xSub,const int h,const VectorXi& RIndex){
	VectorXf beta=FindLine(xSub,h);
	VectorXf praj=((x*beta).array()-1.0).array().abs2();
	praj/=beta.squaredNorm();
	VectorXf proj=praj;
	VectorXf prej(h);
	float fin=1, prem;
	nth_element(proj.data(),proj.data()+h,proj.data()+proj.size());	
	for(int i=0;i<h;i++)	prej(i)=praj(RIndex(i));
	prem=proj.head(h).mean();
	if(prem>1e-8)	fin=prej.head(h).mean()/prem;
	return fin;
}
float Main(const MatrixXf& x,const int k0,const int J,const int k1,VectorXf& dP,const int h_m,VectorXi& samset){
	int p=x.cols(),n=x.rows(),h=p+1,ni=samset.size();
	MatrixXf xSub(h_m,p);
	VectorXf fin(k1);
	VectorXi RIndex(n);
	VectorXi hl;
	RIndex.head(h)=SampleR(ni,h);
	for(int i=0;i<h;i++) xSub.row(i)=x.row(samset(RIndex(i)));			
	hl.setLinSpaced(J+1,h,h_m);
	h=hl(0);	
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
	void fastpcs(int* n,int* p,int* k0,float* xi,int* k1,float* DpC,int* nsamp,int* J,float* objfunC,int* seed,int* ck,int* ni){
		const int ik0=*k0,iJ=*J,ik1=*k1,ih_m=(*n+*p+1)/2;
		float objfunA,objfunB=*objfunC;
		srand(*seed);
		MatrixXf x=Map<MatrixXf>(xi,*n,*p);	
		VectorXi icK=Map<VectorXi>(ck,*ni);
		VectorXf DpA=VectorXf::Zero(*n);
		VectorXf DpB=VectorXf::Zero(*n);

		for(int i=0;i<*nsamp;i++){			//for i=0 to i<#p-subsets.
			objfunA=Main(x,ik0,iJ,ik1,DpA,ih_m,icK);
			if(objfunA<objfunB){
				objfunB=objfunA;
				DpB=DpA;
			}
		}

 		Map<VectorXf>(DpC,*n)=DpB.array();
		*objfunC=objfunB;
	}
}
