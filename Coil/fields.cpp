//  fields.cpp

// Standard headers
#include <iostream.h>
//#include <stdio.h>
#include <fstream.h>

// Defines headers
#include "Defines.h"

// Device headers
#include "fields.h"
#include "coils.h"
#include "ptclgrp.h"
#include "dadirz.h"
#include "dadixy.h"
#include "dadirt.h"
#include "multigrid.h"
#include "boltzman.h"
#include "conducto.h"

extern bool bDCStep;

//******************************************************
// Constructor & destructor

//--------------------------------------------------------------------
//	Allocate arrays for the electromagnetic fields and the capacitance
//	and inductance matrices.

Fields::Fields(Grid* _grid, BoundaryList* _boundaryList, SpeciesList* _speciesList, CoilsList* _coilsList,
       Scalar epsR, int _ESflag,Scalar _presidue, int _BoltzmannFlag, int _SRflag)
{
	int i, j, k;
	jPhiRef = 0;      //Added by Lyd 2-26-2004
	kPhiRef = 0;
	grid = _grid;
	boundaryList = _boundaryList;
	speciesList = _speciesList;
	coilsList = _coilsList;
	nSpecies = speciesList->nItems();
	presidue = _presidue;
	// Initialize flags:
	MarderIter=0;
	BGRhoFlag=0;
	DivergenceCleanFlag=0;
	CurrentWeightingFlag=0;
	ElectrostaticFlag=_ESflag;
	SynchRadiationFlag = _SRflag;
	freezeFields = 0;
	BoltzmannFlag = _BoltzmannFlag;
	MarderParameter=0;
	psolve=0;
	sub_cycle_iter = 0;
	EMdamping = 0;
	epsilonR = epsR;
	J = grid->getJ();
	K = grid->getK();

	rho = new Scalar *[J + 1];
	Charge = new Scalar *[J + 1];
	backGroundRho = new Scalar *[J + 1];
	Phi = new Scalar *[J + 1];
	PhiP = new Scalar *[J + 1];
	DivDerror = new Scalar *[J + 1];
	ENode = new Vector3* [J + 1];
	BNode = new Vector3* [J + 1];
	BNodeStatic = new Vector3* [J+1];
	BNodeDynamic = new Vector3* [J+1];
	I = new Vector3* [J + 1];
	Ispecies = new Vector3** [nSpecies];
	SmartListIter<Species> spIter(*speciesList);
	for (spIter.restart(); !spIter.Done(); spIter++)
	{
		i = spIter.current()->getID();
		if (spIter.current()->isSubcycled())
			Ispecies[i] = new Vector3* [J+1];
		else
			Ispecies[i] = NULL;
	}
	Inode = new Vector3* [J + 1];
	intEdl = new Vector3* [J + 1];
	intEdlPrime = new Vector3* [J + 1];
	intEdlBar = new Vector3* [J + 1];
	intBdS = new Vector3* [J + 1];
	iC = new Vector3* [J + 1];
	iL = new Vector3* [J + 1];
	epsi = new Scalar *[J];  //cell centered
	rho_species = 0;
	for (j=0; j<=J; j++)
	{
		rho[j] = new Scalar[K + 1];
		Charge[j] = new Scalar[K + 1];
		backGroundRho[j] = new Scalar[K + 1];
		Phi[j] = new Scalar [K + 1];
		PhiP[j] = new Scalar [K + 1];
		DivDerror[j]= new Scalar[K + 1];
		ENode[j] = new Vector3[K + 1];
		BNode[j] = new Vector3[K + 1];
		BNodeStatic[j] = new Vector3[K+1];
		BNodeDynamic[j] = new Vector3[K+1];
		I[j] = new Vector3[K + 1];
		for (i = 0; i < nSpecies; i++)
			if (Ispecies[i])
				Ispecies[i][j] = new Vector3[K + 1];
		Inode[j] = new Vector3[K + 1];
		intEdl[j] = new Vector3[K + 1];
		intEdlPrime[j] = new Vector3[K + 1];
		intEdlBar[j] = new Vector3[K + 1];
		intBdS[j] = new Vector3[K + 1];
		iC[j] = new Vector3[K + 1];
		iL[j] = new Vector3[K + 1];
	}

	for(j=0; j<J; j++)
		epsi[j] = new Scalar[K];

	loaddensity = new Scalar **[nSpecies];
	for(i=0;i<nSpecies;i++)
	{
		loaddensity[i] = new Scalar *[J+1];
		for(j=0;j<=J;j++)
		{
			loaddensity[i][j]=new Scalar[K+1];
			//zero the memory
			memset(loaddensity[i][j],0,(K+1)*sizeof(Scalar));
		}
	}

	intEdlBasePtr = intEdl;
	BNodeBasePtr = BNode;
	//  The following initializations to zero are important
	//  correct answers depend upon them.
	for(j=0;j<=J;j++)
	for(k=0;k<=K;k++)
	{
		rho[j][k]=0;
		Phi[j][k]=0;
		backGroundRho[j][k]=0;
		Charge[j][k]=0;
		PhiP[j][k]=0;
		DivDerror[j][k]=0;
	}
	for(j=0;j<J;j++)
	for(k=0;k<K;k++)
		epsi[j][k]=1/iEPS0;
	minusrho=0;
	d=0;
	delPhi=0;//These initializations are important
	SmartListIter<Boundary>	nextb(*boundaryList);
	for(nextb.restart(); !nextb.Done(); nextb++)
		nextb.current()->setFields(*this);
	compute_iC_iL();
	// must be done AFTER setFields()
	initPhi();  // initialize the Poisson solver  MUST be done
	//before the boundaries are done below
	for (nextb.restart(); !nextb.Done(); nextb++)
		nextb.current()->setPassives();
	//  Initialize the Laplace solutions for the potential
	//  due to time-varying equipotential boundaries
	for (nextb.restart(); !nextb.Done(); nextb++)
		nextb.current()->InitializeLaplaceSolution();
	
	//ZDH,2004
	//导体边界为构成封闭区域时，限定计算循环区间，减小计算量；
	//R-theta坐标系目前不处理
	//根据导体边界设置(输入参数文件CLimit限定),界定计算区域,有待进一步测试
	KL1 = new int[J+1];
	KL2 = new int[J+1];
	for (j=0; j<=J; j++)
	{
		KL1[j] = 0;
		KL2[j] = K;
	}
//
	if(grid->query_geometry() != RTGEOM)
	{
		//计算区域边界限定
		Boundary *bPtr;
		int j1,j2,k1,k2,CLI;
		for (nextb.restart(); !nextb.Done(); nextb++)
		{
			bPtr = nextb.current();
			if(bPtr->getBCType() == CONDUCTING_BOUNDARY)
			{
				CLI = (( Conductor* )bPtr)->CLimit;
				if(CLI)
				{
					j1 = bPtr->getMinJ();
					j2 = bPtr->getMaxJ();
					k1 =  bPtr->getMinK();
					k2 =  bPtr->getMaxK();
					if( (CLI == 1) && ( (k1 != 0 ) || (k2 != 0) ))//下边界限定
					{
						Boundary* bbPtr;
						Scalar *IC3=new Scalar[j2-j1+1];
						for(j=j1;j<=j2;j++)//清除IC3自然为零情况
						{
							bbPtr = grid->GetNodeBoundary()[j][0];
							IC3[j-j1]=iC[j][0].e3();
							if(bbPtr)
							{
								if(IC3[j-j1] == 0 && bbPtr->getBCType() != CONDUCTING_BOUNDARY)
									iC[j][0].set_e3(999);//???
							}
						}
						//寻找交点
						int jS;
						bool bFind=false;
						for(j=j1;j<=j2;j++)
						{
							if(iC[j][0].e2() == 0)
							{
								bFind = true;
								break;
							}
						}
						if(bFind)
						{
							jS = j;
							bFind = false;
							for(j=jS+1;j<=j2;j++)
							{
								if(iC[j][0].e2() == 0)
								{
									bFind = true;
									break;
								}
							}
							if(bFind)//双交点
							{
								j1=jS;
								j2=j;
							}
							else if(jS==0 || grid->getNorm2()[jS][0] == -1)//交点为左界
								j1=jS;
							else//交点为右界
							{
								j2=jS;
								jS=0;
							}
						}
						for (j=j1;j<=j2;j++)
						{
							k=0;
							while(iC[j][k].e3() != 0)
								k++;
							if(k != 0 && iC[j][k].e1() != 0)
							{
								int ku= k;
								while (iC[j][k].e3() == 0 && iC[j][k].e1() != 0)
									k++;
								if(k>ku)
									k--;
							}
							if(k)
								k--;
							KL1[j] = k;
						}
						for(j=j1;j<=j2;j++)//恢复下边界IC3
							iC[j][0].set_e3(IC3[j-j1]);
						delete IC3;
					}
					else if( (CLI == 2) && ( (k1 !=K ) || (k2 != K) ))//上边界限定
					//除了导体交点外，IC3不等于0
					{
						for (j=j1;j<=j2;j++)
						{
							k=K;
							while(iC[j][k].e3() != 0)
								k--;
							if(k != K)
							{
								while (iC[j][k].e3() == 0 && iC[j][k].e1() != 0)
									k--;
							}
							if(k != K)
								k ++;
							KL2[j]=k;
						}
					}
				}
			}
		}
	}
//
/*	else//R-T 坐标系，限定内环导体,用于j循环???
	{
		Boundary *bPtr;
		int j1,j2,k1,k2;
		for (nextb.restart(); !nextb.Done(); nextb++)
		{
			bPtr = nextb.current();
			j1 = bPtr->getMinJ();
			j2 = bPtr->getMaxJ();
			if(j1==j2)//same r
			{
				k1 =  bPtr->getMinK();
				k2 =  bPtr->getMaxK();
				if((k1 == 0 ) && (k2==K) && (iC[j1][k1].e3()==0))//2 PI circle and conductor
				{
					for(j=0;j<j1;j++)
						KL1[j] = K;
				}
				break;
			}
		}
	}
*/
}

//--------------------------------------------------------------------
//	Deallocate memory for Fields object

Fields::~Fields()
{
	coilsList->deleteAll();
	delete coilsList;
	delete KL1;
	delete KL2;

	int i;
	int j;
	intEdl = intEdlBasePtr;
	BNode = BNodeBasePtr;
	for (j=0; j<=J; j++)
	{
		delete[] ENode[j];
		delete[] BNode[j];
		delete[] BNodeStatic[j];
		delete[] BNodeDynamic[j];
		delete[] I[j];
		delete[] Inode[j];
		for (i=0; i < nSpecies; i++)
			if (Ispecies[i])
				delete[] Ispecies[i][j];
		delete[] intEdl[j];
		delete[] intEdlPrime[j];
		delete[] intEdlBar[j];
		delete[] intBdS[j];
		delete[] iC[j];
		delete[] iL[j];
		delete[] DivDerror[j];
		delete[] rho[j];
		delete[] Charge[j];
		delete[] backGroundRho[j];
		delete[] Phi[j];
		delete[] PhiP[j];
		if(delPhi)
			delete[] delPhi[j] ;
		if(minusrho)
			delete[] minusrho[j];
		if(d)
			delete[] d[j];
	}
	for (j=0; j<J; j++)
		delete[] epsi[j];
	delete[] ENode;
	delete[] BNode;
	delete[] BNodeStatic;
	delete[] BNodeDynamic;
	delete[] I;
	delete[] Inode;
	for (i=0; i < nSpecies; i++)
	{
		if(Ispecies[i])
			delete[] Ispecies[i];
		else
			delete Ispecies[i];
	}
	delete[] Ispecies;
	delete[] intEdl;
	delete[] intEdlPrime;
	delete[] intEdlBar;
	delete[] intBdS;
	delete[] iC;
	delete[] iL;
	delete[] DivDerror;
	delete[] rho;
	delete[] Charge;
	delete[] backGroundRho;
	delete[] Phi;
	delete[] PhiP;
	delete[] epsi;
	if(delPhi)
		delete[] delPhi;
	if(minusrho)
		delete[] minusrho;
	if(d)
		delete[] d;
	if(psolve)
		delete psolve;
	if(rho_species)
	{
		int J=grid->getJ();
		for(i=0; i<nSpecies; i++)
		{
			for(j=0; j<=J; j++)
				delete [] rho_species[i][j];
			delete [] rho_species[i];
		}
		delete [] rho_species;
	}
	for(i=0; i<nSpecies; i++)
	{
		for(j=0; j<=J; j++)
			delete [] loaddensity[i][j];
		delete [] loaddensity[i];
	}
	delete [] loaddensity;
}

//******************************************************


//******************************************************
//  Getting & setting

//--------------------------------------------------------------------
//	Zero the current accumulator array, I

void Fields::clearI(int i)
{
	Vector3** Iptr;
	//if (ElectrostaticFlag)
	//return;
	if (i == -1)
		Iptr = I;
	else
		Iptr = Ispecies[i];
	if (!Iptr)
		return;
	for (int j=0; j<=J; j++)
	{
		if (CurrentWeightingFlag!=0)
			memset(Inode[j], 0, (K+1)*sizeof(Inode[j][0]));
		memset(Iptr[j], 0, (K+1)*sizeof(Iptr[j][0]));
	}
}

void Fields::ZeroCharge(void)
{
	for(int j=0;j<=J;j++)
		for(int k=0;k<=K;k++)
			Charge[j][k] = 0.0;
}

// Accumulate current for species I:
void Fields::setAccumulator(int i)
{
	if (Ispecies[i])
		accumulator = Ispecies[i];
	else if (CurrentWeightingFlag)
		accumulator = Inode;
	else
		accumulator = I;
}

//--------------------------------------------------------------------
//	Set initial static magnetic field(in Tesla).

bool Fields::setBNodeStatic(Vector3 B0,Scalar betwig, Scalar zoff, char* BModel,
	const SmartString &B01analytic, const SmartString &B02analytic,const SmartString &B03analytic)
{
	FILE *openfile;
	int status;
	SmartString B01a = B01analytic;
	SmartString B02a = B02analytic;
	SmartString B03a = B03analytic;
	Vector3 Btemp;
	Bz0=0;
	Bx0=0;
	BMODEL = 0;//no static magnetic field

	if (strcmp(BModel, "NULL"))//静态磁场由线圈产生或外部文件输入
	{
		if(strcmp(BModel,"COILS")==0)
		{
			BMODEL = 3;
			int CoilsNo = get_coilsList()->nitems;
			if(CoilsNo==0)
				return false;
			float **clsarr = new float*[CoilsNo];
			for(int i=0;i<CoilsNo;i++)
				clsarr[i]= new float [6]; 

			SmartListIter<Coils> cIter(*get_coilsList());
			int i=0;
			//计算用多匝密绕线圈的参数，线圈半径方向和轴向中心位置、厚度、长度、匝数、电流
			double *coil;
			for (cIter.restart(); !cIter.Done(); cIter++)
			{
				coil=cIter.current()->get_coilsParameter();
				for(int j=0;j<6;j++)
					clsarr[i][j] = coil[j];
				switch(cIter.current()->get_form())
				{
				case 1://left,right,bottom,top,Z-R
					clsarr[i][0] = 0.5*(coil[2]+coil[3]);
					clsarr[i][1] = 0.5*(coil[0]+coil[1]);
					clsarr[i][2] = coil[3]-coil[2];
					clsarr[i][3] = coil[1]-coil[0];
					break;
				case 2://bottom,top,left,right,R-Z
					clsarr[i][0] = 0.5*(coil[0]+coil[1]);
					clsarr[i][1] = coil[1]-coil[0];
					clsarr[i][2] = 0.5*(coil[2]+coil[3]);
					clsarr[i][3] = coil[3]-coil[2];
					break;
				case 3://z center,r center,z long, r long
					clsarr[i][0] = coil[1];
					clsarr[i][1] = coil[0];
					clsarr[i][2] = coil[3];
					clsarr[i][3] = coil[2];
					break;
				case 4://r center, z center, r long, z long
				default:
					break;
				}
				i++;
			}
			float * Bz = new float[(K + 1) * (J+1)];
			float * Br = new float[(K + 1) * (J+1)];
			for(i=0;i<(K+1)*(J+1);i++)
				Bz[i]=Br[i]=0;

			multicoils(clsarr,Bz,Br,CoilsNo);
			for (int k = 0; k <= K; k++)
			{
				for (int j = 0; j <= J; j++)
				{
					Btemp.set_e1(Bz[(J+1) * k + j]);
					Btemp.set_e2(Br[(J+1) * k + j]);
					Btemp.set_e3(0);
					BNodeStatic[j][k] = Btemp;
				}
			}
			delete Bz;
			delete Br;
		}
		else if ((openfile = fopen (BModel, "r"))  != NULL)
		{
		// read in
			BMODEL = 4;
			Scalar dum1, dum2, B1, B2, B3; // dummies, remove later
//			cout << "starting to read B field file" << endl;
			for (int j=0; j<=J; j++)
			{
				for (int k=0; k<=K; k++)
				{
					#ifdef SCALAR_DOUBLE
						status = fscanf (openfile, "%lg %lg %lg %lg %lg",
						&dum1, &dum2, &B1, &B2, &B3);
					#else
						status = fscanf (openfile, "%g %g %g %g %g",
						&dum1, &dum2, &B1, &B2, &B3);
					#endif
					Btemp.set_e1(B1*1e-4);
					Btemp.set_e2(B2*1e-4);
					Btemp.set_e3(B3*1e-4);
					BNodeStatic[j][k] = Btemp;
				}
			}
//			cout << "finished reading B field file" << endl;
			fclose (openfile);
		}
		else
		{
//			cout << "open failed on magnetic field file Bf in Control group";
			return false;
		}
	}
	else if (B0.e1()!=0 || B0.e2()!=0 || B0.e3()!=0)//X or Z方向均匀静态磁场
	{
		BMODEL = 1;
		for (int j=0; j<=J; j++)
			for (int k=0; k<=K; k++)
			{
				Bz0 = (grid->query_geometry() == ZXGEOM) ? B0.e1() : B0.e1() + B0.e2()*
					(cos(betwig*(getMKS(j,k).e1()-zoff)));
				Bx0 = (grid->query_geometry() == ZXGEOM) ? B0.e2() : + .5*B0.e2()
					*betwig*getMKS(j,k).e2()
				*(sin(betwig*(getMKS(j,k).e1()-zoff)));
				Btemp.set_e1(Bz0);
				Btemp.set_e2(Bx0);
				Btemp.set_e3(B0.e3());
				BNodeStatic[j][k] = Btemp;
			}
	}
	else if (strcmp(B01a, "0.0") || strcmp(B02a, "0.0") || strcmp(B03a, "0.0"))//静态磁场由解析公式给出
	{
		BMODEL = 2;
		Scalar x1, x2; // do NOT change these to Scalar!! Changed from float to Scalar by Lyd, 12-03-2002
		Evaluator eval1(B01a);
		Evaluator eval2(B02a);
		Evaluator eval3(B03a);
		eval1.AddRefVar(x1,"x1");
		eval1.AddRefVar(x2,"x2");
		eval2.AddRefVar(x1,"x1");
		eval2.AddRefVar(x2,"x2");
		eval3.AddRefVar(x1,"x1");
		eval3.AddRefVar(x2,"x2");
		for(int j=0;j<=J;j++)
			for(int k=0;k<=K;k++)
			{
				x1 = grid->getMKS(j,k).e1();
				x2 = grid->getMKS(j,k).e2();
				Btemp.set_e1(eval1.Evaluate());
				Btemp.set_e2(eval2.Evaluate());
				Btemp.set_e3(eval3.Evaluate());
				BNodeStatic[j][k] = Btemp;
			}
	}
	for(int j=0;j<=J;j++)
		for(int k=0;k<=K;k++)
			BNode[j][k] += BNodeStatic[j][k];
	return true;
}

//multicoils()函数计算一个或多个线圈产生的总磁场
void Fields::multicoils(float **cls,float *Bz,float *Br,int coil_number)
{
	//对线圈个数做循环
	for(int i = 0;i < coil_number;i++)
	{
			//调用coils()函数计算单个线圈产生的磁场
			coils(cls[i],Bz,Br);
	}
}

//coils()函数计算单个螺线管产生的磁场,2004.3改正线圈单元离散错误及最大分层稳定性
void Fields::coils(float *clsarr,float *Bz,float *Br)
{
	//only dealt with uniform girds
	float delt_z = getMKS(1,1).e1();
	float delt_r = getMKS(1,1).e2();

	int j1,j2;
	int kLayer,jTurn;
	float r1,r2,delt_rc,a;
//********************************************************************************
//线圈离散及单元线圈参数计算
	j1 = int((clsarr[1] - 0.5*clsarr[3])/delt_z);
	if(j1 <= 0.0)
		j1 = j1 - 1;
	j2 = int((clsarr[1] + 0.5*clsarr[3])/delt_z);
	if(j2 <= 0.0)
		j2 = j2 - 1;
	r1 = (clsarr[0] - 0.5*clsarr[2]);
    r2 = (clsarr[0] + 0.5*clsarr[2]);

	kLayer = clsarr[2]/delt_r;//线圈厚度分层，最大10层

	if( kLayer > 1) 
	{
		if(kLayer > 10)
			kLayer = 10;
	}
	else//单层线圈
		kLayer = 1;
	delt_rc = clsarr[2]/kLayer;//线圈分层厚度
	jTurn = j2-j1;
	if(jTurn == 0)
		jTurn = 1;
	a = 1e-7*clsarr[4]*clsarr[5]/(kLayer*jTurn);//线圈单元电流密度，磁场单位变换
//*******************************************************************************
	int j,k;
	int jT,kL;
	int jj,jk;
	float rc,r,dz;
	Vector2 Bzr;
	for(k = 0;k <= K;k++)
	{
		r = delt_r*k;//待计算点横向坐标，位于正网格点
		rc = r1;
		for(kL = 0; kL < kLayer; kL++,rc += delt_rc)
		{
			for(j = -(j2-j1);j <= J;j++) 
			{
				dz = delt_z*(j-j1-1.5);//待计算点纵向位于半网格点上，避免奇异点判断

				Bzr=coil(r,rc,dz);

				for(jT = 0;jT < jTurn;jT++)
				{
					jj = j + jT;
					if(jj <= J && jj >= 0)
					{
						jk = (J+1)*k+jj;
						Bz[jk] = Bz[jk] + Bzr.e1()*a;
						Br[jk] = Br[jk] + Bzr.e2()*a;
					}
				}
			}
		}
	}
}

//coil()计算电流为1A的单环线圈所产生的磁场
Vector2    Fields::coil(float r,float rl,float dz)
{

	float bb=0.0,ab=1.0,rb=1.0,aari=1.0,ari=2.0,ak=2.0,d=0.0,s=0.0;
	float ge=0.0;
	Vector2 Bzr;
	if(r <= 0.0)//axis
	{
		s = (rl*rl + dz*dz);
		Bzr.set_e1 (TWOPI*rl*rl/(s*sqrt(s)));
		Bzr.set_e2 (0.0);
	}
	else
	{
		d = ((r-rl)*(r-rl) + dz*dz);
		s = r*rl*4.0 + d;
		ge = d/s;
		while((ge/aari) < 0.999999)
		{
			ak *= 2.0;
			ge = sqrt(ge*aari);
			ge *= 2.0;
			aari = ari;
			ari += ge;
			bb += rb*ge;
			bb *= 2.0;
			rb = ab;
			ab += bb / ari;
		}
		ge = TWOPI*rl/(sqrt(s)*ari);
		ak = (ak-ab)/s;
		ab /= d;
		Bzr.set_e2((ab-ak)*ge*dz);
		Bzr.set_e1(((r+rl)*ak - (r-rl)*ab)*ge);
	}
	return Bzr;
}
void Fields::setBeamDump(int _j1, int _j2)
{
	int j1 = MIN(_j1, _j2);
	int j2 = MAX(_j1, _j2);
	if (j1 < 0 || j2 > J)
		 return;	 // default values or error
	Scalar x1 = getMKS(j1).e1();
	Scalar x2 = getMKS(j2).e1();
	Scalar beta = M_PI/(2*(x2 - x1));
	Scalar geomMult = (grid->query_geometry() == ZXGEOM) ? 1 : 0.5;
	int j,k;
	for (j=j1; j<=j2; j++)
		for (k=0; k<=K; k++)
			BNodeStatic[j][k] = Bz0*Vector3(cos(beta*(getMKS(j,k).e1()-x1)),
								geomMult*beta*getMKS(j,k).e2()
								*sin(beta*(getMKS(j,k).e1()-x1)),
								BNodeStatic[j][k].e3());
	for (j=j2+1; j<=J; j++)
		for (k=0; k<=K; k++)
			BNodeStatic[j][k] = Vector3(0, 0, 0);
}

//--------------------------------------------------------------------
//	+= an  initial magnetic field as a function of x1,x2

void Fields::setBinit(const SmartString &B1init, const SmartString &B2init,
                      const SmartString &B3init)
{
}

//--------------------------------------------------------------------
//	+= an  initial electric field as a function of x1,x2

void Fields::setEinit(const SmartString &E1init,const SmartString &E2init, const SmartString &E3init)
{
	SmartString E01a = E1init;
	SmartString E02a = E2init;
	SmartString E03a = E3init;
	// need extra variables for computing fields at half-cells.
	Scalar x1,x2;  // do NOT change these to Scalar!!
	Scalar x1a,x2a,x1b,x2b,x1s,x2s;
	Scalar Ex1,Ex2,Ex3;
	//E01a+='\n';
	//E02a+='\n';
	//E03a+='\n';
	Evaluator eval1(E01a);
	Evaluator eval2(E02a);
	Evaluator eval3(E03a);
	eval1.AddRefVar(x1,"x1");
	eval1.AddRefVar(x2,"x2");
	eval2.AddRefVar(x1,"x1");
	eval2.AddRefVar(x2,"x2");
	eval3.AddRefVar(x1,"x1");
	eval3.AddRefVar(x2,"x2");

	for(int j=0;j<=J;j++)
		for(int k=0;k<=K;k++)
		{
			x1s = grid->getMKS(j,k).e1();
			x2s = grid->getMKS(j,k).e2();
			x1a = grid->getMKS(MAX(j-1,0),k).e1();
			x1b = grid->getMKS(MIN(j+1,J),k).e1();
			x2a = grid->getMKS(j,MAX(k-1,0)).e2();
			x2b = grid->getMKS(j,MIN(k+1,K)).e2();

			// get the intEdL location, and the field.
			x1 = 0.5*(x1b + x1s);
			x2 = x2s;
			Ex1 = eval1.Evaluate()*grid->dl1(j,k);

			x2 = 0.5*(x2b + x2s);
			x1 = x1s;
			Ex2 = eval2.Evaluate()*grid->dl2(j,k);

			x2 = x2s;
			x1 = x1s;
			Ex3 = eval3.Evaluate()*grid->dl3(j,k);
			intEdl[j][k] += Vector3(Ex1, Ex2, Ex3);
		}
}

//******************************************************



//******************************************************
// Operator & methods

//--------------------------------------------------------------------
//	Advance the fields from t-dt->t by solving Maxwell's equations.
/*
void Fields::advance(Scalar t, Scalar dt)
{
	int	j, k;
	local_dt = dt;  // so that objects needing to manipulate fields directly can get_dt() easily
	Scalar local_t = t;

	// get any charge on any dielectrics into rho, put any currents generated by loops
	// this current weighting does not subcycle properly

	//ApplyToList(putCharge_and_Current(t),*boundaryList,Boundary);

	if(ElectrostaticFlag)
	{
		ElectrostaticSolve(t,dt);
		toNodes(t);
		ApplyToList(emFlux(dt),*boundaryList,Boundary);
	}
	else
	{
		if(DivergenceCleanFlag && bDCStep)
		{
			updateDivDerror();
			DivergenceClean();
			updateDivDerror();
		}
		if(CurrentWeightingFlag)  InodeToI();  //  the bilinear weighting needs this
		if (grid->axis()) radialCurrentCorrection(); // correct current at 0 and K
		if(EMdamping>0)
		{
			for (int i = 0; i < FieldSubFlag; i++, local_t += dt)
			{
				intEdl = intEdlPrime;
				if (i==0)
					advanceB(0.5*dt); 
				else
					advanceB(dt);

				intEdl = intEdlBasePtr;

				for (j=0; j<=J; j++)
					for (k = KL1[j]; k <= KL2[j]; k++)//ZDH,2004
					{
						intEdlBar[j][k] *= EMdamping;
						intEdlBar[j][k] += (1-EMdamping)*intEdl[j][k];
						intEdlPrime[j][k] = .5*((1-EMdamping)*intEdlBar[j][k]-intEdl[j][k]);
					}
				advanceE(dt);

				//  Apply Boundary Conditions at t
				ApplyToList(PML_E(dt),*boundaryList,Boundary);
				ApplyToList(applyFields(local_t,dt),*boundaryList,Boundary);

				for (j=0; j<=J; j++)
					for (k = KL1[j]; k <= KL2[j]; k++)//ZDH,2004
						intEdlPrime[j][k] += (1+.5*EMdamping)*intEdl[j][k];
			}

			intEdl = intEdlPrime;
			advanceB(0.5*dt);

			intEdl = intEdlBasePtr;
			ApplyToList(PML_B(dt),*boundaryList,Boundary);
		}
		else
		{
			for (int i = 0; i < FieldSubFlag; i++, local_t += dt)
			{
				if (i==0)
					advanceB(0.5*dt);
				else
					advanceB(dt);

				advanceE(dt);
				//  Apply Boundary Conditions at t
				ApplyToList(PML_E(dt),*boundaryList,Boundary);
				ApplyToList(applyFields(local_t,dt),*boundaryList,Boundary);
			}
			advanceB(0.5*dt);
			ApplyToList(PML_B(dt),*boundaryList,Boundary);
		}
	}
	if(FieldSubFlag==1)
	{
		if((MarderIter > 0) && bDCStep)
				MarderCorrection(dt);
	}

	toNodes(t);
	ApplyToList(emFlux(dt),*boundaryList,Boundary);
}
*/
//
void Fields::advance(Scalar t, Scalar dt)
{
	int	j, k;
	local_dt = dt;  // so that objects needing to manipulate fields directly can get_dt() easily

	// get any charge on any dielectrics into rho, put any currents generated by loops
	// this current weighting does not subcycle properly

	//ApplyToList(putCharge_and_Current(t),*boundaryList,Boundary);

	if(ElectrostaticFlag)//静电
	{
		ElectrostaticSolve(t,dt);
		toNodes(t);
		ApplyToList(emFlux(dt),*boundaryList,Boundary);
	}
	else//电磁
	{
		if(DivergenceCleanFlag && bDCStep)
		{
			updateDivDerror();
			DivergenceClean();
			updateDivDerror();
		}
		if(CurrentWeightingFlag)
			InodeToI();  //  the bilinear weighting needs this
		if (grid->axis())
			radialCurrentCorrection(); // correct current at 0 and K
		if(EMdamping>0)
		{
			intEdl = intEdlBasePtr;
			for (j=0; j<=J; j++)
				for (k = KL1[j]; k <= KL2[j]; k++)
				{
					intEdlBar[j][k] *= EMdamping;
					intEdlBar[j][k] += (1-EMdamping)*intEdl[j][k];
					intEdlPrime[j][k] = .5*((1-EMdamping)*intEdlBar[j][k]-intEdl[j][k]);
				}

			//集总参数修正，ZDH,2004
			ApplyToList(SolveRLCLoad(dt),*boundaryList,Boundary);
			
			advanceE(dt);

			//  Apply Boundary Conditions at t
			ApplyToList(applyFields(t,dt),*boundaryList,Boundary);
			ApplyToList(PML_E(dt),*boundaryList,Boundary);

			for (j=0; j<=J; j++)
				for (k = KL1[j]; k <= KL2[j]; k++)
					intEdlPrime[j][k] += (1+.5*EMdamping)*intEdl[j][k];

			intEdl = intEdlPrime;
			advanceB(0.5*dt); 

			intEdl = intEdlBasePtr;
			toNodes(t);
			ApplyToList(emFlux(dt),*boundaryList,Boundary);

			intEdl = intEdlPrime;
			advanceB(0.5*dt);

			intEdl = intEdlBasePtr;
			ApplyToList(PML_B(dt),*boundaryList,Boundary);
		}
		else
		{
			advanceE(dt);
			//  Apply Boundary Conditions at t
			ApplyToList(PML_E(dt),*boundaryList,Boundary);
			ApplyToList(applyFields(t,dt),*boundaryList,Boundary);

			advanceB(0.5*dt);

			toNodes(t);
			ApplyToList(emFlux(dt),*boundaryList,Boundary);

			advanceB(0.5*dt);
			ApplyToList(PML_B(dt),*boundaryList,Boundary);
		}
	}
	if((MarderIter > 0) && bDCStep)
		MarderCorrection(dt);
}
//
void Fields::ElectrostaticSolve(Scalar t, Scalar dt)
{
	if(BoltzmannFlag)
	{
		SmartListIter<Boundary> nextb(*boundaryList);
		for (nextb.restart(); !nextb.Done(); nextb++)
			nextb.current()->applyFields(t,dt);
		theBoltzmann->updatePhi(rho,Phi,t,dt);
	}
	else
		updatePhi(rho,Phi,t,dt);
	setEfromPhi();
}

//--------------------------------------------------------------------
//	Advances B by the time step specified.  This function is called twice
//	per time step by advance() with a half timestep each time.

//ZDH,2004
//为适应计算区域限制功能修改
//RTGEOM部份根LYD结果整理并分离，不作处理
void Fields::advanceB(Scalar dt2)
{
	int	j, k;	//  indices
	if(grid->query_geometry()==RTGEOM)
	{
        for (j=0; j<J; j++)
        {
			for (k=0; k<K; k++)
				intBdS[j][k] -= dt2*Vector3(intEdl[j][k+1].e3() - intEdl[j][k].e3(),
					-intEdl[j+1][k].e3() + intEdl[j][k].e3(),
					intEdl[j][k].e1() - intEdl[j][k+1].e1()
						+ intEdl[j+1][k].e2() - intEdl[j][k].e2());
	        //	get intBdS[j][K].e2 only
			intBdS[j][K] = intBdS[j][0];
        }
        for (k=0; k<K; k++)				//	intBdS[J][k].e1
			intBdS[J][k] -= dt2*Vector3(intEdl[J][k+1].e3()	- intEdl[J][k].e3(), 0, 0);
        intBdS[J][K] = intBdS[J][0];
	}
	else
	{
		for (j=0; j<J; j++)
		{
//			for (k=0; k<K; k++)
			for (k = KL1[j]; k <= MIN(KL2[j],K-1); k++)//ZDH,2004
				intBdS[j][k] -= dt2*Vector3(intEdl[j][k+1].e3() - intEdl[j][k].e3(),
					-intEdl[j+1][k].e3() + intEdl[j][k].e3(),
					intEdl[j][k].e1() - intEdl[j][k+1].e1()
						+ intEdl[j+1][k].e2() - intEdl[j][k].e2());
				//	get intBdS[j][K].e2 only
			intBdS[j][K] -= dt2*Vector3(0, intEdl[j][K].e3() - intEdl[j+1][K].e3(), 0);
		}
		for (k=0; k<K; k++)				//	intBdS[J][k].e1
			intBdS[J][k] -= dt2*Vector3(intEdl[J][k+1].e3()	- intEdl[J][k].e3(), 0, 0);
	}

}
/*
void Fields::advanceB(Scalar dt2)
{
        int	j, k;	//  indices
        int GeometryFlag = grid->query_geometry();
        for (j=0; j<J; j++)
        {
	        for (k=0; k<K; k++)
	                intBdS[j][k] -= dt2*Vector3(intEdl[j][k+1].e3() - intEdl[j][k].e3(),
					   -intEdl[j+1][k].e3() + intEdl[j][k].e3(),
					   intEdl[j][k].e1() - intEdl[j][k+1].e1()
					   + intEdl[j+1][k].e2() - intEdl[j][k].e2());
	        //	get intBdS[j][K].e2 only
                if(GeometryFlag == RTGEOM)
                        intBdS[j][K] = intBdS[j][0];
                else
	                intBdS[j][K] -= dt2*Vector3(0, intEdl[j][K].e3() - intEdl[j+1][K].e3(), 0);
        }
        for (k=0; k<K; k++)				//	intBdS[J][k].e1
                intBdS[J][k] -= dt2*Vector3(intEdl[J][k+1].e3()	- intEdl[J][k].e3(), 0, 0);
        if(GeometryFlag == RTGEOM)
                intBdS[J][K] = intBdS[J][0];

}
*/
//ZDH,2004
//为适应计算区域限制功能修改,RTGEOM部份不作处理
void Fields::advanceE(Scalar dt)
{
	int j,k;

	Vector3* iL0;
    Vector3* iLk1;
    Scalar intEdl3Temp = 0;
    if(grid->query_geometry() == RTGEOM)   // RT geometry
    {
		Scalar r0 = grid->getMKS(0,0).e1();
		if(r0 == 0)    // where r=0
		{
			for(k=0; k<K; k++)
			{
				iL0 = &iL[0][k];
				intEdl3Temp += intBdS[0][k].e2()*iL0->e2();
			}
			iL0 = &iL[0][0];
			iLk1 = &iL[0][K-1];
			intEdl[0][0] += dt*iC[0][0].jvMult(Vector3(
				intBdS[0][0].e3()*iL0->e3() - intBdS[0][K-1].e3()*iLk1->e3(),
				0, intEdl3Temp) - I[0][0]);// periodic
		}
		else
		{
			iL0 = &iL[0][0];
			iLk1 = &iL[0][K-1];
			intEdl[0][0] += dt*iC[0][0].jvMult(Vector3(
				intBdS[0][0].e3()*iL0->e3() - intBdS[0][K-1].e3()*iLk1->e3(),
				-intBdS[0][0].e3()*iL0->e3(),
				-intBdS[0][0].e1()*iL0->e1() + intBdS[0][K-1].e1()*iLk1->e1() + intBdS[0][0].e2()*iL0->e2())
				- I[0][0]);
		}
		intEdl[0][K] = intEdl[0][0]; //Vector3(0, intEdl[0][0].e2(), intEdl[0][0].e3();
		
		iL0 = &iL[J][0];
		Vector3* iLj1 = &iL[J-1][0];
		iLk1 = &iL[J][K-1];
		intEdl[J][0] += dt*iC[J][0].jvMult(Vector3(
			0,
			intBdS[J-1][0].e3()*iLj1->e3(),
			-intBdS[J][0].e1()*iL0->e1() + intBdS[J][K-1].e1()*iLk1->e1() - intBdS[J-1][0].e2()*iLj1->e2())
			- I[J][0]);// periodic
		intEdl[J][K] = intEdl[J][0];
		
		for (k=1; k<K; k++)
		{
			iL0 = &iL[0][k];
			iLk1 = &iL[0][k-1];
			intEdl[0][k] += dt*iC[0][k].jvMult(Vector3(
				intBdS[0][k].e3()*iL0->e3() - intBdS[0][k-1].e3()*iLk1->e3(),
				intEdl[0][0].e2(), (r0 == 0) ? intEdl[0][0].e3() :
				(intBdS[0][k-1].e1()*iLk1->e1() - intBdS[0][k].e1()*iL0->e1()
					+ intBdS[0][k].e2()*iL0->e2())) - I[0][k]);
			iL0 = &iL[J][k];
			iLk1 = &iL[J][k-1];
			iLj1 = &iL[J-1][k];
			intEdl[J][k] += dt*iC[J][k].jvMult(Vector3(
				//  intBdS[J][k].e3()*iL0->e3() - intBdS[J][k-1].e3()*iLk1->e3(),
				0,   // component 1 outside system
				intBdS[J-1][k].e3()*iLj1->e3(),
				intBdS[J][k-1].e1()*iLk1->e1() - intBdS[J][k].e1()*iL0->e1()
				- intBdS[J-1][k].e2()*iLj1->e2())
				- I[J][k]);
		}
		Vector3 *Bk1, *Bj1, *B0;
		for (j=1; j<J; j++)
		{
			iL0 = &iL[j][0];
			iLj1 = &iL[j-1][0];
			iLk1 = &iL[j][K-1];
			B0 = &intBdS[j][0];
			Bj1 = &intBdS[j-1][0];
			Bk1 = &intBdS[j][K-1];
			intEdl[j][0] += dt*iC[j][0].jvMult(Vector3(
				B0->e3()*iL0->e3() - Bk1->e3()*iLk1->e3(),
				-B0->e3()*iL0->e3() + Bj1->e3()*iLj1->e3(),
				Bk1->e1()*iLk1->e1() - B0->e1()*iL0->e1() + B0->e2()*iL0->e2() - Bj1->e2()*iLj1->e2())
				- I[j][0]);
			intEdl[j][K] = intEdl[j][0];                    // periodic
			for (k=1; k<K; k++)
			{
				iLk1 = iL0;
				iLj1 = &iL[j-1][k];
				iL0 = &iL[j][k];
				Bk1 = B0;
				Bj1 = &intBdS[j-1][k];
				B0 = &intBdS[j][k];
				intEdl[j][k] += dt*iC[j][k].jvMult(Vector3(
					B0->e3()*iL0->e3() - Bk1->e3()*iLk1->e3(),
					-B0->e3()*iL0->e3() + Bj1->e3()*iLj1->e3(),
					Bk1->e1()*iLk1->e1() - B0->e1()*iL0->e1() + B0->e2()*iL0->e2() - Bj1->e2()*iLj1->e2())
					- I[j][k]);
			}
		}
    }
	else
	{
		iL0 = &iL[0][0];
		intEdl[0][0] += dt*iC[0][0].jvMult(Vector3(
						intBdS[0][0].e3()*iL0->e3(),
						-intBdS[0][0].e3()*iL0->e3(),
						-intBdS[0][0].e1()*iL0->e1() + intBdS[0][0].e2()*iL0->e2())
						- I[0][0]);
		iL0 = &iL[0][K];
		Vector3* iLk1 = &iL[0][K-1];
		intEdl[0][K] += dt*iC[0][K].jvMult(Vector3(
						-intBdS[0][K-1].e3()*iLk1->e3(),
						0,
						intBdS[0][K-1].e1()*iLk1->e1() + intBdS[0][K].e2()*iL0->e2())
						- I[0][K]);
		iL0 = &iL[J][0];
		Vector3* iLj1 = &iL[J-1][0];
		intEdl[J][0] += dt*iC[J][0].jvMult(Vector3(
						0,
						intBdS[J-1][0].e3()*iLj1->e3(),
						-intBdS[J][0].e1()*iL0->e1() - intBdS[J-1][0].e2()*iLj1->e2())
						- I[J][0]);
		iL0 = &iL[J][K];
		iLk1 = &iL[J][K-1];
		iLj1 = &iL[J-1][K];
		intEdl[J][K] += dt*iC[J][K].jvMult(Vector3(
						0,
						0,
						intBdS[J][K-1].e1()*iLk1->e1() - intBdS[J-1][K].e2()*iLj1->e2())
						- I[J][K]);

		for (k=1; k<K; k++)
		{
			iL0 = &iL[0][k];
			iLk1 = &iL[0][k-1];
			intEdl[0][k] += dt*iC[0][k].jvMult(Vector3(
							intBdS[0][k].e3()*iL0->e3() - intBdS[0][k-1].e3()*iLk1->e3(),
							-intBdS[0][k].e3()*iL0->e3(),
							intBdS[0][k-1].e1()*iLk1->e1() - intBdS[0][k].e1()*iL0->e1()
							+ intBdS[0][k].e2()*iL0->e2())
							- I[0][k]);
			iL0 = &iL[J][k];
			iLk1 = &iL[J][k-1];
			iLj1 = &iL[J-1][k];
			intEdl[J][k] += dt*iC[J][k].jvMult(Vector3(
						//  intBdS[J][k].e3()*iL0->e3() - intBdS[J][k-1].e3()*iLk1->e3(),
							0,   // component 1 outside system
							intBdS[J-1][k].e3()*iLj1->e3(),
							intBdS[J][k-1].e1()*iLk1->e1() - intBdS[J][k].e1()*iL0->e1()
							- intBdS[J-1][k].e2()*iLj1->e2())
							- I[J][k]);
		}

		Vector3 *Bk1, *Bj1, *B0;
		for (j=1; j<J; j++)
		{
			iL0 = &iL[j][0];
			iLj1 = &iL[j-1][0];
			B0 = &intBdS[j][0];
			Bj1 = &intBdS[j-1][0];
			intEdl[j][0] += dt*iC[j][0].jvMult(Vector3(
							intBdS[j][0].e3()*iL0->e3(),
							-intBdS[j][0].e3()*iL0->e3() + intBdS[j-1][0].e3()*iLj1->e3(),
							-intBdS[j][0].e1()*iL0->e1() + intBdS[j][0].e2()*iL0->e2()
							- intBdS[j-1][0].e2()*iLj1->e2())
							- I[j][0]);
//			for (k=1; k<K; k++)
			for (k=MAX(KL1[j],1); k <= MIN(KL2[j],K-1); k++)//ZDH,2004
			{
				iLk1 = iL0;
				iLj1 = &iL[j-1][k];
				iL0 = &iL[j][k];
				Bk1 = B0;
				Bj1 = &intBdS[j-1][k];
				B0 = &intBdS[j][k];
				intEdl[j][k] += dt*iC[j][k].jvMult(Vector3(
								B0->e3()*iL0->e3() - Bk1->e3()*iLk1->e3(),
								-B0->e3()*iL0->e3() + Bj1->e3()*iLj1->e3(),
								Bk1->e1()*iLk1->e1() - B0->e1()*iL0->e1()
								+ B0->e2()*iL0->e2() - Bj1->e2()*iLj1->e2())
								- I[j][k]);
			}
			iL0 = &iL[j][K];
			iLk1 = &iL[j][K-1];
			intEdl[j][K] += dt*iC[j][K].jvMult(Vector3(
							-intBdS[j][K-1].e3()*iLk1->e3(),
							0,
							intBdS[j][K-1].e1()*iLk1->e1()
							+ intBdS[j][K].e2()*iL0->e2() - intBdS[j-1][K].e2()*iLj1->e2())
							- I[j][K]);
		}
	}
}
/*
void Fields::advanceE(Scalar dt)
{
        int GeometryFlag = grid->query_geometry();
        int j,k;
        Vector3* iL0;
        Scalar intEdl3Temp = 0;
        if(GeometryFlag == RTGEOM)   // RT geometry
        {
                for(k=0; k<K; k++)
                {
                        iL0 = &iL[0][k];
                        intEdl3Temp += intBdS[0][k].e2()*iL0->e2();
                }
                iL0 = &iL[0][0];
                Vector3* iLk1 = &iL[0][K-1];
                intEdl[0][0] += dt*iC[0][0].jvMult(Vector3(
			      intBdS[0][0].e3()*iL0->e3() - intBdS[0][K-1].e3()*iLk1->e3(),     // periodic
			      0, intEdl3Temp) - I[0][0]);
                intEdl[0][K] = intEdl[0][0]; //Vector3(0, intEdl[0][0].e2(), intEdl[0][0].e3();

                iL0 = &iL[J][0];
                Vector3* iLj1 = &iL[J-1][0];
                iLk1 = &iL[J][K-1];
                intEdl[J][0] += dt*iC[J][0].jvMult(Vector3(
	        	 	      0,
		 	      intBdS[J-1][0].e3()*iLj1->e3(),
		 	      -intBdS[J][0].e1()*iL0->e1() + intBdS[J][K-1].e1()*iLk1->e1()   // periodic
                               - intBdS[J-1][0].e2()*iLj1->e2())
		 	      - I[J][0]);
                intEdl[J][K] = intEdl[J][0];

                for (k=1; k<K; k++)
                {
                        iL0 = &iL[0][k];
                        iLk1 = &iL[0][k-1];
                        intEdl[0][k] += dt*iC[0][k].jvMult(Vector3(
                                        intBdS[0][k].e3()*iL0->e3() - intBdS[0][k-1].e3()*iLk1->e3(),
	                                intEdl[0][0].e2(), intEdl[0][0].e3()) - I[0][k]);
	                iL0 = &iL[J][k];
	                iLk1 = &iL[J][k-1];
	                iLj1 = &iL[J-1][k];
	                intEdl[J][k] += dt*iC[J][k].jvMult(Vector3(
                                //  intBdS[J][k].e3()*iL0->e3() - intBdS[J][k-1].e3()*iLk1->e3(),
				    0,   // component 1 outside system
				    intBdS[J-1][k].e3()*iLj1->e3(),
				    intBdS[J][k-1].e1()*iLk1->e1() - intBdS[J][k].e1()*iL0->e1()
				    - intBdS[J-1][k].e2()*iLj1->e2())
				    - I[J][k]);
                }
                Vector3 *Bk1, *Bj1, *B0;
                for (j=1; j<J; j++)
                {
	                iL0 = &iL[j][0];
	                iLj1 = &iL[j-1][0];
                        iLk1 = &iL[j][K-1];
	                B0 = &intBdS[j][0];
	                Bj1 = &intBdS[j-1][0];
                        Bk1 = &intBdS[j][K-1];
	                intEdl[j][0] += dt*iC[j][0].jvMult(Vector3(
				    B0->e3()*iL0->e3() - Bk1->e3()*iLk1->e3(),
				    -B0->e3()*iL0->e3() + Bj1->e3()*iLj1->e3(),
				    Bk1->e1()*iLk1->e1() - B0->e1()*iL0->e1()  
                                     + B0->e2()*iL0->e2() - Bj1->e2()*iLj1->e2())
				    - I[j][0]);
	                intEdl[j][K] = intEdl[j][0];                    // periodic
	                for (k=1; k<K; k++)
	                {
	                        iLk1 = iL0;
	                        iLj1 = &iL[j-1][k];
	                        iL0 = &iL[j][k];
	                        Bk1 = B0;
	                        Bj1 = &intBdS[j-1][k];
	                        B0 = &intBdS[j][k];
	                        intEdl[j][k] += dt*iC[j][k].jvMult(Vector3(
					  B0->e3()*iL0->e3() - Bk1->e3()*iLk1->e3(),
					  -B0->e3()*iL0->e3() + Bj1->e3()*iLj1->e3(),
					  Bk1->e1()*iLk1->e1() - B0->e1()*iL0->e1()
					  + B0->e2()*iL0->e2() - Bj1->e2()*iLj1->e2())
			        	  - I[j][k]);
	                }
                }
        }
        else
        {
                iL0 = &iL[0][0];
                intEdl[0][0] += dt*iC[0][0].jvMult(Vector3(
			      intBdS[0][0].e3()*iL0->e3(),
			      -intBdS[0][0].e3()*iL0->e3(),
			      -intBdS[0][0].e1()*iL0->e1() + intBdS[0][0].e2()*iL0->e2())
       			      - I[0][0]);
                iL0 = &iL[0][K];
                Vector3* iLk1 = &iL[0][K-1];
                intEdl[0][K] += dt*iC[0][K].jvMult(Vector3(
		              -intBdS[0][K-1].e3()*iLk1->e3(),
			      0,
			      intBdS[0][K-1].e1()*iLk1->e1() + intBdS[0][K].e2()*iL0->e2())
			      - I[0][K]);
                iL0 = &iL[J][0];
                Vector3* iLj1 = &iL[J-1][0];
                intEdl[J][0] += dt*iC[J][0].jvMult(Vector3(
		 	      0,
		 	      intBdS[J-1][0].e3()*iLj1->e3(),
		 	      -intBdS[J][0].e1()*iL0->e1() - intBdS[J-1][0].e2()*iLj1->e2())
		 	      - I[J][0]);
                iL0 = &iL[J][K];
                iLk1 = &iL[J][K-1];
                iLj1 = &iL[J-1][K];
                intEdl[J][K] += dt*iC[J][K].jvMult(Vector3(
			       0,
			       0,
			       intBdS[J][K-1].e1()*iLk1->e1() - intBdS[J-1][K].e2()*iLj1->e2())
			       - I[J][K]);
                for (k=1; k<K; k++)
                {
	                iL0 = &iL[0][k];
	                iLk1 = &iL[0][k-1];
	                intEdl[0][k] += dt*iC[0][k].jvMult(Vector3(
			      	    intBdS[0][k].e3()*iL0->e3() - intBdS[0][k-1].e3()*iLk1->e3(),
			      	    -intBdS[0][k].e3()*iL0->e3(),
			      	    intBdS[0][k-1].e1()*iLk1->e1() - intBdS[0][k].e1()*iL0->e1()
			      	    + intBdS[0][k].e2()*iL0->e2())
			      	    - I[0][k]);
	                iL0 = &iL[J][k];
	                iLk1 = &iL[J][k-1];
	                iLj1 = &iL[J-1][k];
	                intEdl[J][k] += dt*iC[J][k].jvMult(Vector3(
                                //  intBdS[J][k].e3()*iL0->e3() - intBdS[J][k-1].e3()*iLk1->e3(),
				    0,   // component 1 outside system
				    intBdS[J-1][k].e3()*iLj1->e3(),
				    intBdS[J][k-1].e1()*iLk1->e1() - intBdS[J][k].e1()*iL0->e1()
				    - intBdS[J-1][k].e2()*iLj1->e2())
				    - I[J][k]);
                }
                Vector3 *Bk1, *Bj1, *B0;
                for (j=1; j<J; j++)
                {
	                iL0 = &iL[j][0];
	                iLj1 = &iL[j-1][0];
	                B0 = &intBdS[j][0];
	                Bj1 = &intBdS[j-1][0];
	                intEdl[j][0] += dt*iC[j][0].jvMult(Vector3(
				    intBdS[j][0].e3()*iL0->e3(),
				    -intBdS[j][0].e3()*iL0->e3() + intBdS[j-1][0].e3()*iLj1->e3(),
				    -intBdS[j][0].e1()*iL0->e1() + intBdS[j][0].e2()*iL0->e2()
				    - intBdS[j-1][0].e2()*iLj1->e2())
				    - I[j][0]);
	                for (k=1; k<K; k++)
	                {
	                        iLk1 = iL0;
	                        iLj1 = &iL[j-1][k];
	                        iL0 = &iL[j][k];
	                        Bk1 = B0;
	                        Bj1 = &intBdS[j-1][k];
	                        B0 = &intBdS[j][k];
	                        intEdl[j][k] += dt*iC[j][k].jvMult(Vector3(
					  B0->e3()*iL0->e3() - Bk1->e3()*iLk1->e3(),
					  -B0->e3()*iL0->e3() + Bj1->e3()*iLj1->e3(),
					  Bk1->e1()*iLk1->e1() - B0->e1()*iL0->e1()
					  + B0->e2()*iL0->e2() - Bj1->e2()*iLj1->e2())
			        	  - I[j][k]);
	                }
	                iL0 = &iL[j][K];
	                iLk1 = &iL[j][K-1];
	                intEdl[j][K] += dt*iC[j][K].jvMult(Vector3(
				    -intBdS[j][K-1].e3()*iLk1->e3(),
				    0,
				    intBdS[j][K-1].e1()*iLk1->e1()
				    + intBdS[j][K].e2()*iL0->e2() - intBdS[j-1][K].e2()*iLj1->e2())
				    - I[j][K]);
                }
        }

}
*/
//--------------------------------------------------------------------
//	Translates to new locations by incrementally locating cell crossings
//	and accumulates current along the trajectory.  Initial position passed
//	in x is updated to final position on return.  Return value is a
//	pointer to a boundary if encountered, NULL otherwise.
//	qOverDt = q/dt*particleWeight for this particle.
// ZDH,modified to trip particles moving out of simulation region
/*
Boundary* Fields::translateAccumulate(Vector2& x, Vector3& dxMKS, Scalar qOverDt)
{
	Boundary* boundary = NULL;
	int which_dx = (dxMKS.e1() != 0.0) ? 1 : 2;
	double prev_dxMKS = (which_dx == 1) ? dxMKS.e1() : dxMKS.e2();
	double frac_v3dt;

	while (fabs(dxMKS.e1()) + fabs(dxMKS.e2()) > 1E-25)
	{
		Vector2 xOld = x;
		boundary = grid->translate(x, dxMKS);
		if(!boundary)
		{
			if((x.e1() <= 0 ) || (x.e2() <= 0 ) || (x.e1() >= J ) || (x.e2() >= K ))
			{
				AfxMessageBox(_T("Waring: Particles out of Region!"));
				return grid->GetBC2()[0][0];
			}
		}

	// If the particle translates in x1 or x2, apply the fraction of
	// rotation for this cell only; otherwise it spins only and all
	// current is collected at this location.
		if (prev_dxMKS != 0)
			frac_v3dt = dxMKS.e3()*(prev_dxMKS -
			((which_dx==1) ? dxMKS.e1() : dxMKS.e2()))/prev_dxMKS;
		else
			frac_v3dt = dxMKS.e3();
		accumulate(xOld, x, frac_v3dt, qOverDt);
		prev_dxMKS = (which_dx == 1) ? dxMKS.e1() : dxMKS.e2();
		if (boundary)
			break;
	}
	return boundary;
}
*/
Boundary* Fields::translateAccumulate(Vector2& x, Vector3& dxMKS, Scalar qOverDt)
{
	Boundary* boundary = NULL;
	int which_dx = (dxMKS.e1() != 0.0) ? 1 : 2;
	Scalar prev_dxMKS = (which_dx == 1) ? dxMKS.e1() : dxMKS.e2();
	Scalar frac_v3dt;
	while (fabs(dxMKS.e1()) + fabs(dxMKS.e2()) > 1E-15)
	{
		Vector2 xOld = x;
		boundary = grid->translate(x, dxMKS);
		
/*
		if(!boundary)
		{
			if((x.e1() <= 0 ) || (x.e2() <= 0 ) || (x.e1() >= J ) || (x.e2() >= K ))
			{
			//	AfxMessageBox(_T("Waring: Particles out of Region!"));
				return grid->GetBC2()[0][0];
			}
		}
*/
		// If the particle translates in x1 or x2, apply the fraction of
		// rotation for this cell only; otherwise it spins only and all
		// current is collected at this location.
		if (prev_dxMKS != 0)
			frac_v3dt = dxMKS.e3()*(prev_dxMKS -
				((which_dx==1) ? dxMKS.e1() : dxMKS.e2()))/prev_dxMKS;
		else
			frac_v3dt = dxMKS.e3();
		accumulate(xOld, x, frac_v3dt, qOverDt);
		
		if(grid->query_geometry() == RTGEOM)    // RT geometry has periodic problem about theta
		{
			Scalar K = grid->getK();
			if(xOld.e2() < K && x.e2() >= K)
				x.set_e2(x.e2() - K);
			else if(xOld.e2() > 0 && x.e2() <= 0)
				x.set_e2(x.e2() + K);
		}
		prev_dxMKS = (which_dx == 1) ? dxMKS.e1() : dxMKS.e2();
		if (boundary)
			break;
	}
	return boundary;
}

//--------------------------------------------------------------------
//	Weight intEdl and intBdS to nodes to obtain Enode and Bnode in
//	physical units.  This routine has been modified to set only the
// interior points; edges and internal boundaries are set by calling
// Boundary::toNodes().
//
//ZDH,2004
//为适应计算区域限制功能,设置BTmp指针改变斜边界处理可能引起的BNode空洞
//清除BNode初始置零
void Fields::toNodes(Scalar t)
{
	int j, k;
	Scalar w1p, w1m, w2p, w2m;

	BNode=BNodeDynamic;

	// all interior points in 1-direction
	for (j=1; j<J; j++)
	{
		// Calculate weight factors for 1-directions
		w1m = grid->dl1(j, 1)/(grid->dl1(j-1, 1) + grid->dl1(j, 1));
		w1p = 1 - w1m;

		// All interior points in the 2-direction

		// JRC: Could this be accelerated through use of smaller loops?
//		for (k=1; k<K; k++)
		for(k = MAX(KL1[j],1); k <= MIN(KL2[j],K-1);k++)//ZDH,2004
		{
			// Get weights in 2-direction
			w2m = grid->dl2(j, k)/(grid->dl2(j, k-1) + grid->dl2(j, k));
			w2p = 1 - w2m;

			// Weight each of the fields

			// E1 defined on 2-node values, but weight in 1-direction
			ENode[j][k].set_e1(w1p*intEdl[j][k].e1()/grid->dl1(j, k)
				+ w1m*intEdl[j-1][k].e1()/grid->dl1(j-1, k));

			// E2 defined on 1-node values, but weight in 2-direction
			ENode[j][k].set_e2(w2p*intEdl[j][k].e2()/grid->dl2(j, k)
				+ w2m*intEdl[j][k-1].e2()/grid->dl2(j, k-1));

			// E3 defined at the nodes
			ENode[j][k].set_e3(intEdl[j][k].e3()/grid->dl3(j,k));

			// B1 defined on 1-node values, but weight in 2-direction
			BNode[j][k].set_e1(intBdS[j][k].e1()*(w2p/grid->dS1(j, k))
				+ intBdS[j][k-1].e1()*w2m/grid->dS1(j, k-1));

			// B2 defined on 2-node values, but weight in 1-direction
			BNode[j][k].set_e2(intBdS[j][k].e2()*(w1p/grid->dS2(j, k))
				+ intBdS[j-1][k].e2()*w1m/grid->dS2(j-1, k));

			// B3 weighted in both directions (a cell center quantity)
			BNode[j][k].set_e3(intBdS[j][k].e3()*w1p*w2p/grid->dS3(j, k)
				+ intBdS[j-1][k].e3()*w1m*w2p/grid->dS3(j-1, k)
				+ intBdS[j][k-1].e3()*w1p*w2m/grid->dS3(j, k-1)
				+ intBdS[j-1][k-1].e3()*w1m*w2m/grid->dS3(j-1, k-1));
		}
	}

    if(grid->query_geometry() == RTGEOM)//根据LYD结果整理修改
	{
		for(j=1; j<J; j++)
		{
			w1m = grid->dl1(j, 0)/(grid->dl1(j-1, 0) + grid->dl1(j, 0));
			w1p = 1 - w1m;
			w2m = grid->dl2(j, 0)/(grid->dl2(j, K-1) + grid->dl2(j, 0));
			w2p = 1 - w2m;
			ENode[j][0].set_e1(w1p*intEdl[j][0].e1()/grid->dl1(j, 0)
				+ w1m*intEdl[j-1][0].e1()/grid->dl1(j-1, 0));
			ENode[j][0].set_e2(w2p*intEdl[j][0].e2()/grid->dl2(j, 0)
				+ w2m*intEdl[j][K-1].e2()/grid->dl2(j, K-1));
			ENode[j][0].set_e3(intEdl[j][0].e3()/grid->dl3(j,0));
			ENode[j][K]=ENode[j][0];

			BNode[j][0].set_e1(intBdS[j][0].e1()*(w2p/grid->dS1(j, 0))
				+ intBdS[j][K-1].e1()*w2m/grid->dS1(j, K-1));
			BNode[j][0].set_e2(intBdS[j][0].e2()*(w1p/grid->dS2(j, 0))
				+ intBdS[j-1][0].e2()*w1m/grid->dS2(j-1, 0));
			BNode[j][0].set_e3(intBdS[j][0].e3()*w1p*w2p/grid->dS3(j, 0)
				+ intBdS[j-1][0].e3()*w1m*w2p/grid->dS3(j-1, 0)
				+ intBdS[j][K-1].e3()*w1p*w2m/grid->dS3(j, K-1)
				+ intBdS[j-1][K-1].e3()*w1m*w2m/grid->dS3(j-1, K-1));
			BNode[j][K]=BNode[j][0];
			
		}
    }
	// boundaries interpolate coincident fields (assumes interior previously set):
	ApplyToList(toNodes(),*boundaryList,Boundary);

	BNode = BNodeBasePtr;//ZDH,2004
	// Add in the static magnetic field
	for (j=0; j<=J; j++)
	{
//		for (k=0; k<=K; k++)
		for(k = KL1[j]; k <= KL2[j]; k++)//ZDH,2004
		{
//			BNodeDynamic[j][k] = BNode[j][k];
//			BNode[j][k] += BNodeStatic[j][k];
			BNode[j][k] = BNodeDynamic[j][k] + BNodeStatic[j][k];//ZDH,2004
		}
	}
}
/*
void Fields::toNodes(Scalar t)
{
        int j, k;
        int GeometryFlag = grid->query_geometry();
        Scalar w1p, w1m, w2p, w2m;

        // all interior points in 1-direction
        for (k=j=1; j<J; j++)
        {
                // Calculate weight factors for 1-directions
                w1m = grid->dl1(j, k)/(grid->dl1(j-1, k) + grid->dl1(j, k));
                w1p = 1 - w1m;

                // Not sure what this is doine
                BNode[j][0]=BNode[j][K]=0;  // unbounded edges.

                // All interior points in the 2-direction

                // JRC: Could this be accelerated through use of smaller loops?
                for (k=1; k<K; k++)
                {
                        // Get weights in 2-direction
                        w2m = grid->dl2(j, k)/(grid->dl2(j, k-1) + grid->dl2(j, k));
                        w2p = 1 - w2m;

                        // Weight each of the fields

                        // E1 defined on 2-node values, but weight in 1-direction
                        ENode[j][k].set_e1(w1p*intEdl[j][k].e1()/grid->dl1(j, k)
                                        + w1m*intEdl[j-1][k].e1()/grid->dl1(j-1, k));

                        // E2 defined on 1-node values, but weight in 2-direction
                        ENode[j][k].set_e2(w2p*intEdl[j][k].e2()/grid->dl2(j, k)
                                        + w2m*intEdl[j][k-1].e2()/grid->dl2(j, k-1));

                        // E3 defined at the nodes
                        ENode[j][k].set_e3(intEdl[j][k].e3()/grid->dl3(j,k));

                        // B1 defined on 1-node values, but weight in 2-direction
                        BNode[j][k].set_e1(intBdS[j][k].e1()*(w2p/grid->dS1(j, k))
                                        + intBdS[j][k-1].e1()*w2m/grid->dS1(j, k-1));

                        // B2 defined on 2-node values, but weight in 1-direction
                        BNode[j][k].set_e2(intBdS[j][k].e2()*(w1p/grid->dS2(j, k))
                                        + intBdS[j-1][k].e2()*w1m/grid->dS2(j-1, k));

                        // B3 weighted in both directions (a cell center quantity)
                        BNode[j][k].set_e3(intBdS[j][k].e3()*w1p*w2p/grid->dS3(j, k)
                                        + intBdS[j-1][k].e3()*w1m*w2p/grid->dS3(j-1, k)
                                        + intBdS[j][k-1].e3()*w1p*w2m/grid->dS3(j, k-1)
                                        + intBdS[j-1][k-1].e3()*w1m*w2m/grid->dS3(j-1, k-1));
                }
        }

        for (k=1; k<K; k++)
                BNode[0][k] = BNode[J][k] = 0;  // unbounded edges.

        // periodic boundary in RTGEOM where theta=0 or theta=2*pi
        for(j=1; j<J; j++)
        {
                if(GeometryFlag == RTGEOM)
                {
                        w1m = grid->dl1(j, 0)/(grid->dl1(j-1, 0) + grid->dl1(j, 0));
                        w1p = 1 - w1m;
                        w2m = grid->dl2(j, 0)/(grid->dl2(j, K-1) + grid->dl2(j, 0));
                        w2p = 1 - w2m;
                        ENode[j][0].set_e1(w1p*intEdl[j][0].e1()/grid->dl1(j, 0)
                                        + w1m*intEdl[j-1][0].e1()/grid->dl1(j-1, 0));
                        ENode[j][0].set_e2(w2p*intEdl[j][0].e2()/grid->dl2(j, 0)
                                        + w2m*intEdl[j][K-1].e2()/grid->dl2(j, K-1));
                        ENode[j][0].set_e3(intEdl[j][0].e3()/grid->dl3(j,0));
                        ENode[j][K]=ENode[j][0];

                        BNode[j][0].set_e1(intBdS[j][0].e1()*(w2p/grid->dS1(j, 0))
                                        + intBdS[j][K-1].e1()*w2m/grid->dS1(j, K-1));
                        BNode[j][0].set_e2(intBdS[j][0].e2()*(w1p/grid->dS2(j, 0))
                                        + intBdS[j-1][0].e2()*w1m/grid->dS2(j-1, 0));
                        BNode[j][0].set_e3(intBdS[j][0].e3()*w1p*w2p/grid->dS3(j, 0)
                                        + intBdS[j-1][0].e3()*w1m*w2p/grid->dS3(j-1, 0)
                                        + intBdS[j][K-1].e3()*w1p*w2m/grid->dS3(j, K-1)
                                        + intBdS[j-1][K-1].e3()*w1m*w2m/grid->dS3(j-1, K-1));
                        BNode[j][K]=BNode[j][0];

                }
                else
                        BNode[j][0] = BNode[j][K] = 0;  // unbounded edges.
        }

        BNode[0][0] = BNode[0][K] = BNode[J][0] = BNode[J][K] = 0;

        // boundaries interpolate coincident fields (assumes interior previously set):
        ApplyToList(toNodes(),*boundaryList,Boundary);

        // Add in the static magnetic field
        for (j=0; j<=J; j++)
                for (k=0; k<=K; k++)
                {
                        BNodeDynamic[j][k] = BNode[j][k];
                        BNode[j][k] += BNodeStatic[j][k];
                }

}
*/
//******************************************************

void Fields::addtoI(int i)
{
	if (!Ispecies[i])
		return;
	for (int j=0; j<=J; j++)
		for (int k=0; k<=K; k++)
			I[j][k] += Ispecies[i][j][k];
}

void Fields::Excite() //ntg8-16 excites a field component at t = 0
{
	this->setIntEdl(J/2, K/2, 1, 1.0);
	this->setIntEdl(J/2, K/2, 2, -1.0);
	this->setIntEdl(J/2, K/2+1, 1, -1.0);
	this->setIntEdl(J/2+1, K/2, 2, 1.0);
}

//--------------------------------------------------------------------
//	Seed the field emitter with a large negative field to get it
//	started emitting.

void Fields::SeedFieldEmitter()
{
	for (int j=0; j<J; j++)
		for (int k=0; k<=K; k++)
			setIntEdl(j, k, 1, -1000);
}


//--------------------------------------------------------------------
//	Update the DivD array of divergences.  This is not intended to
//      be kept updated but rather to be updated by request.
//
Scalar maxDiv;
void Fields::updateDivDerror(void)
{
	int j,k;
	Vector3 D1,D2;
	Scalar Div,ic;
	//This section of the code computes the divergence of D everywhere.

	maxDiv=0;
	for(j=1;j<=J-1;j++)   // bounds are temporary, need to check for metals eventually
//		for(k=1 ;k<=K-1;k++)
		for(k = MAX(KL1[j],1); k <= MIN(KL2[j],K-1);k++)
			{
				if(grid->GetNodeBoundary()[j][k])  // Set DivDerror on boundary to zero, by Lyd 12-03-2002
					DivDerror[j][k] = 0;
				else
				{
					if( j>0 && (ic=iC[j-1][k].e1())!=0.0)
						D1.set_e1( intEdl[j-1][k].e1()/ic );
					else
						D1.set_e1(0);
					if( k>0 && (ic=iC[j][k-1].e2())!=0.0)
						D1.set_e2( intEdl[j][k-1].e2()/ic );
					else
						D1.set_e2(0);
					if( j<J && (ic=iC[j][k].e1())!=0.0)
						D2.set_e1( intEdl[j][k].e1()/ic );
					else
						D2.set_e1(0);
					if( k<K && (ic=iC[j][k].e2()) !=0.0)
						D2.set_e2( intEdl[j][k].e2()/ic );
					else
						D2.set_e2(0);

					Div = D2.e1() - D1.e1() + D2.e2() - D1.e2();
					Div /= grid->get_halfCellVolumes()[j][k];
					DivDerror[j][k]=-Div + rho[j][k];
					if(fabs(DivDerror[j][k])>maxDiv)
						maxDiv=DivDerror[j][k];
				}
			}
}
//--------------------------------------------------------------------
//        compute the potential everywhere using Fields::rho as a
//	  source term

void Fields::updatePhi(Scalar **source,Scalar **dest,Scalar t,Scalar dt)
{
	int j,k;
	// presiduetol_test=1e-4;  //tolerance for the poisson solver
	static int itermax=100;  //maximum # of iterations for psolve
	SmartListIter<Boundary> nextb(*boundaryList);

	// set up the boundary arrays if necessary
	// they will not have been set up if u_z0 = 0
	if(minusrho==0)
	{
		minusrho = new Scalar* [J+1];
		for(j=0;j<=J;j++)
		{
			minusrho[j]=new Scalar[K+1];
			memset(minusrho[j],0,(K+1)*sizeof(Scalar));
		}
		itermax = 100;
	}

        //  Scale the source term by -1
	for(j=0;j<=J;j++)
		for(k=0;k<=K;k++)
			minusrho[j][k]=-(Scalar)source[j][k];

	//actually DO the solve
	psolve->solve(PhiP,minusrho,itermax,presidue);

	// To add all the LaPlace solutions, we first need to start
	//  with the Poisson solution.
	for(j=0;j<=J;j++)
		for(k=0;k<=K;k++)
			dest[j][k]=PhiP[j][k];

	// apply any laplace boundary conditions to phi
	for (nextb.restart(); !nextb.Done(); nextb++)
		nextb.current()->applyFields(t,dt);

	//  assign the solution to dest
	for(j=0;j<=J;j++)
		for(k=0;k<=K;k++)
			dest[j][k]=Phi[j][k];

}

// -----------------------------
//  complete those tasks necessary to initialize the poisson
//  solver--symmetry boundary at zero, dirichlet otherwise
/*
void Fields::initPhi()
{
	switch(ElectrostaticFlag)
	{
		default:
			fprintf(stderr,"WARNING, poisson solve defaulting to Multigrid\n");

		case ELECTROMAGNETIC_SOLVE:
			psolve=new Multigrid(epsi,grid);

			break;
		case DADI_SOLVE:
		switch(grid->query_geometry())
		{
			case ZRGEOM:
				psolve=new Dadirz(epsi,grid);
				break;
			case ZXGEOM:
				psolve=new Dadixy(epsi,grid);
				break;
		}
		break;
		case MGRID_SOLVE:
			psolve=new Multigrid(epsi,grid);
			break;
	}
}
*/
void Fields::initPhi()
{
        switch(ElectrostaticFlag)
        {
                default:
                        fprintf(stderr,"WARNING, poisson solve defaulting to Multigrid\n");

                case ELECTROMAGNETIC_SOLVE:
                        //psolve=new Multigrid(epsi,grid);
                        //break;
                case DADI_SOLVE:
                        switch(grid->query_geometry())
                        {
                                case ZRGEOM:
                                        psolve=new Dadirz(epsi,grid);
                                        break;
                                case ZXGEOM:
                                        psolve=new Dadixy(epsi,grid);
                                        break;
                                case RTGEOM:
                                        psolve=new Dadirt(epsi,grid);
                        }
                        break;
                case MGRID_SOLVE:
                        psolve=new Multigrid(epsi,grid);
                        break;
        }
}

//--------------------------------------------
//  this collects the charge from the particle group
//  into rho[][]  If the group == 0 then it clears rho[][]
/*
void Fields::collect_charge_to_grid(ParticleGroup *group,Scalar **_rho)
{
	int i,j,k;
	Scalar q,jx,kx;
	Vector2 *x;

	if(group==0)
	{       // reset flag - zero this rho
		for(j=0;j<=J;j++)
			memset(_rho[j],0,(K+1)*sizeof(Scalar));
		return;
	}

	q = group->get_q();  //charge * np2c
	x = group->get_x();
	for(i=0; i<group->get_n(); i++)
	{
		jx = x[i].e1();
		kx = x[i].e2();
		j = (int)jx; k=(int)kx;  //get the integer part
		jx -= j; kx -= k;  //get the fractional part only
		if (group->vary_np2c())
			q = group->get_q(i);

		_rho[j][k] += (1-jx)*(1-kx)*q;
		_rho[j][k+1] += (1-jx)*(kx)*q;
		_rho[j+1][k] += (jx)*(1-kx)*q;
		_rho[j+1][k+1] += (jx)*(kx)*q;
	}
}
*/
void Fields::collect_charge_to_grid(ParticleGroup *group,Scalar **_rho)
{
	int i,j,k;
	Scalar q,jx,kx;
	Vector2 *x;
	
	if(group==0)
	{       // reset flag - zero this rho
		for(j=0;j<=J;j++)
			memset(_rho[j],0,(K+1)*sizeof(Scalar));
		return;
	}
	
	q = group->get_q();  //charge * np2c
	x = group->get_x();
	for(i=0; i<group->get_n(); i++)
	{
		jx = x[i].e1();
		kx = x[i].e2();
		j = (int)jx; k=(int)kx;  //get the integer part
		if (group->vary_np2c())
			q = group->get_q(i);
		
		if(grid->query_geometry() == RTGEOM )    // periodic
		{
/* 
			Scalar sqrdj = 2*j + 1; //(j+1)*(j+1) - j*j
			_rho[j][k] += ((j+1)*(j+1)-jx*jx)*(k+1 - kx)*q/sqrdj;
			_rho[j][k+1] += ((j+1)*(j+1)-jx*jx)*(kx - k)*q/sqrdj;
			_rho[j+1][k] += (jx*jx - j*j)*(k+1 - kx)*q/sqrdj;
			_rho[j+1][k+1] += (jx*jx - j*j)*(kx - k)*q/sqrdj;
*/
			Scalar rl2 = grid->getMKS(j,k).e1()*grid->getMKS(j,k).e1();
			Scalar rh2 = grid->getMKS(j+1,k).e1()*grid->getMKS(j+1,k).e1();
			Scalar r2 = grid->getMKS(Vector2(jx,kx)).e1()*grid->getMKS(Vector2(jx,kx)).e1();
			Scalar q_Sqrdj = q/(rh2 - rl2);
			_rho[j][k] += (rh2 - r2)*(k+1 - kx)*q_Sqrdj;
			_rho[j][k+1] += (rh2 - r2)*(kx - k)*q_Sqrdj;
			_rho[j+1][k] += (r2 - rl2)*(k+1 - kx)*q_Sqrdj;
			_rho[j+1][k+1] += (r2 - rl2)*(kx - k)*q_Sqrdj;

			if(k == K-1)
			{
				_rho[j][0] = _rho[j][K];
				_rho[j+1][0] = _rho[j+1][K];
			}
			else if(k == 0)
			{
				_rho[j][K] = _rho[j][0];
				_rho[j+1][K] = _rho[j+1][0];
			}
		}
		else
		{
			jx -= j; kx -= k;  //get the fractional part only
			_rho[j][k] += (1-jx)*(1-kx)*q;
			_rho[j][k+1] += (1-jx)*(kx)*q;
			_rho[j+1][k] += (jx)*(1-kx)*q;
			_rho[j+1][k+1] += (jx)*(kx)*q;
		}
	}
}
//----------------------------------------------------------------------
//function iterates through the particle group list given it
//to sum up all the charge using collect_charge_to_grid()
//

void Fields::compute_rho(ParticleGroupList **list,int nSpecies)
{
	int j;

	if (rho_species==0)
	{ // not initialized, we need to allocate and initialize
		rho_species = new Scalar **[nSpecies];
		for(int i=0;i<nSpecies;i++)
		{
			rho_species[i] = new Scalar *[J+1];
			for(j=0;j<=J;j++)
			{
				rho_species[i][j]=new Scalar[K+1];
				//zero the memory
				memset(rho_species[i][j],0,(K+1)*sizeof(Scalar));
			}
		}
	}

	collect_charge_to_grid(0,rho);  //reset the global rho
	for (int i=0; i<nSpecies; i++)
	{
		collect_charge_to_grid(0,rho_species[i]);  //reset the species rho
		SmartListIter<ParticleGroup> pgscan(*list[i]);
		for(pgscan.restart();!pgscan.Done();pgscan++)
			collect_charge_to_grid(pgscan(),rho_species[i]);
	}
	charge_to_charge_density(nSpecies);
	grid->binomialFilter(rho, nSmoothing);
	addBGRho();
}

void Fields::charge_to_charge_density(int nSpecies)
{
	int i,j,k;
	Scalar **CellVolumes=grid->get_halfCellVolumes();

	for(i=0;i<nSpecies;i++)
		for(j=0;j<=J;j++)
//			for(k=0;k<=K;k++)
			for(k = KL1[j]; k <= KL2[j];k++)//ZDH,2004
			{
				rho_species[i][j][k]/=CellVolumes[j][k];
				rho[j][k]+=rho_species[i][j][k];
			}
}

void Fields::addBGRho()
{
	if(BGRhoFlag)
		for(int j=0;j<=J;j++)
//		for(int k=0;k<=K;k++)
		for(int k = KL1[j]; k <= KL2[j]; k++)//ZDH,2004
			rho[j][k] += backGroundRho[j][k];
}

//----------------------------------------------------------------------
//  Given a z-r solution of poisson\'s equation from updatePhi in
//  Scalar **Phi, make the electric fields consistent with it.
//  This is an initialization of the E-fields.  It will destroy any
//  previous E fields.  This function cannot touch e3.

void Fields::setEfromPhi()
{
	int j,k;

	//  phi solution is for phi at nodes.  This makes it convenient to
	//  get E at half-cells.  Fortunately, this is what intEdl needs.

	for(j=0;j<J;j++)
	for(k=0;k<K;k++)
//	for(k = KL1[j]; k < MIN(KL2[j],K-1); k++)//ZDH,2004
		{
			setIntEdl(j,k,1,  (Phi[j][k]-Phi[j+1][k])   );
			setIntEdl(j,k,2,  (Phi[j][k]-Phi[j][k+1])   );
		}

	for(j=0;j<J;j++) setIntEdl(j,K,1, (Phi[j][K]-Phi[j+1][K])  );
	for(k=0;k<K;k++) setIntEdl(J,k,2, (Phi[J][k]-Phi[J][k+1])  );
}

// Added by Lyd 2-26-2004, for voltage measurement in EMPIC
void Fields::setPhifromIntEdl()
{
        int j,k;
        Phi[jPhiRef][kPhiRef] = 0.0;
        for(j=jPhiRef+1; j<=J; j++)
        {
                Phi[j][kPhiRef] = -intEdl[j-1][kPhiRef].e1() + Phi[j-1][kPhiRef];
                for(k=kPhiRef+1; k<=K; k++)
                        Phi[j][k] = -intEdl[j][k-1].e2() + Phi[j][k-1];
                for(k=kPhiRef-1; k>=0; k--)
                        Phi[j][k] = intEdl[j][k].e2() + Phi[j][k+1];
        }
        for(j=jPhiRef-1; j>=0; j--)
        {
                Phi[j][kPhiRef] = -intEdl[j-1][kPhiRef].e1() + Phi[j-1][kPhiRef];
                for(k=kPhiRef+1; k<=K; k++)
                        Phi[j][k] = -intEdl[j][k-1].e2() + Phi[j][k-1];
                for(k=kPhiRef-1; k>=0; k--)
                        Phi[j][k] = intEdl[j][k].e2() + Phi[j][k+1];
        }
}
//----------------------------------------------------------------------
//  Use the current rho and a poisson solver to correct the errors in the
//  irrotational E fields.  compute_rho and updateDivDerror should be called first.
//  Also initPhi should be called.
//

void Fields::DivergenceClean(void)
{
	int j,k;

	if(!delPhi)
	{  // get space if this is the first call.
		delPhi= new Scalar *[J +1];
		for(j=0;j<=J;j++)
			delPhi[j] = new Scalar [K+1];
		for(j=0;j<=J;j++)
			for(k=0;k<=K;k++)
				delPhi[j][k]=0;
	}

	// get the phi we need
	psolve->solve(delPhi,DivDerror,5,presidue); // use the Poisson solver directly,
	//delPhi is actually returned as -delPhi

	// Now we need to perform the actual correction to the intEdl everywhere.

	for(j=0;j<J;j++)
//		for(k=0;k<K;k++)
		for(k = KL1[j]; k < MIN(KL2[j],K-1);k++)//ZDH,2004
		{
			setIntEdl(j,k,1,intEdl[j][k].e1() - (delPhi[j][k]-delPhi[j+1][k]));
			setIntEdl(j,k,2, intEdl[j][k].e2() - (delPhi[j][k]-delPhi[j][k+1]));
		}
}

void Fields::InodeToI(void)
{
	int j,k;

	for(j=0;j<J;j++)
//	for(k=0;k<K;k++)
	for(k = KL1[j]; k < MIN(KL2[j],K-1);k++)//ZDH,2004
		I[j][k] = 0.5*Vector3(
				((j==0)?2:1)*Inode[j][k].e1() + ((j==J-1)?2:1)*Inode[j+1][k].e1(),
				Inode[j][k].e2() + Inode[j][k+1].e2(),
				2*Inode[j][k].e3());
}

//---------------------------------------------------------------------
//  Do a marder correction pass.  Actual algorithm is based on
//  Bruce Langdon's paper "On Enforcing Gauss' Law"
/*
void Fields::MarderCorrection(Scalar dt)
{
	int i,j,k;
	static Scalar** d=0;
	Vector2 **X=grid->getX();
	if(d==0)
	{
		d = new Scalar* [J+1];
		for(j=0;j<=J;j++)
		{
			d[j] = new Scalar[K+1];
			for(k=0;k<=K;k++)
			{
				if(j==J||k==K)
					d[j][k]=0;
				else
					d[j][k] = 0.5 * MarderParameter*iEPS0
						/( 1.0/sqr( X[j+1][k].e1()-X[j][k].e1())+1.0/sqr( X[j][k+1].e2()-X[j][k].e2()));
			}
		}
	}

	for(i=0;i<MarderIter;i++)
	{
		updateDivDerror();
		for(j=0;j<J;j++)
			for(k=0;k<K;k++)
			{
				setIntEdl(j,k,1,intEdl[j][k].e1() - (d[j+1][k]*DivDerror[j+1][k] - d[j][k]*DivDerror[j][k]));
				setIntEdl(j,k,2,intEdl[j][k].e2() - (d[j][k+1]*DivDerror[j][k+1] - d[j][k]*DivDerror[j][k]));
			}
	}
}
*/
void Fields::MarderCorrection(Scalar dt)
{
	int i,j,k;
	static Scalar** d=0;
	Vector2 **X=grid->getX();
	if(d==0)
	{
		d = new Scalar* [J+1];
		for(j=0;j<=J;j++)
		{
			d[j] = new Scalar[K+1];
			for(k=0;k<=K;k++)
			{
				if(j==J||k==K)
					d[j][k]=0;
				else
					d[j][k] = 0.5 * MarderParameter*iEPS0
						/( 1.0/sqr( X[j+1][k].e1()-X[j][k].e1())+1.0/sqr( X[j][k+1].e2()-X[j][k].e2()));
			}
		}
	}

	for(i=0;i<MarderIter;i++)
	{
		updateDivDerror();
		if(fabs(maxDiv) < presidue)
			break;
		for(j=0;j<J;j++)
		{
//			for(k=0;k<K;k++)
			for(k = KL1[j]; k < MIN(KL2[j],K-1); k++)//ZDH,2004
			{
				if(fabs(DivDerror[j][k]) < presidue)
					continue;
				setIntEdl(j,k,1,intEdl[j][k].e1() - (d[j+1][k]*DivDerror[j+1][k] - d[j][k]*DivDerror[j][k]));
				setIntEdl(j,k,2,intEdl[j][k].e2() - (d[j][k+1]*DivDerror[j][k+1] - d[j][k]*DivDerror[j][k]));
			}
		}
	}
}

//
void Fields::compute_iC_iL()
{
	Vector3** dl = grid->dl();
	Vector3** dlPrime = grid->dlPrime();
	Vector3** dS = grid->dS();
	Vector3** dSPrime = grid->dSPrime();
	Scalar epsjk, epsjmk, epsjkm, epsjmkm;

	for (int j=0; j<=J; j++)
		for (int k = 0; k<=K; k++)
		{
			iL[j][k] = iMU0*dlPrime[j][k].jvDivide(dS[j][k]+Vector3(Scalar(1e-30),Scalar(1e-30),Scalar(1e-30)));

			epsjk = epsi[MIN(j,J-1)][MIN(k,K-1)];
			epsjmk = epsi[MAX(j-1, 0)][MIN(k,K-1)];
			epsjkm = epsi[MIN(j,J-1)][MAX(k-1, 0)];
			epsjmkm = epsi[MAX(j-1, 0)][MAX(k-1, 0)];

			iC[j][k] = dl[j][k].jvDivide(dSPrime[j][k].jvMult(Vector3(
							.5*(epsjk+epsjkm),.5*(epsjk+epsjmk),.25*(epsjk+epsjmk+epsjkm+epsjmkm))));
		}
}

void Fields::radialCurrentCorrection()
{
	for (int j=0; j<=J; j++)
	{
		I[j][0] *= grid->get_radial0();
		I[j][K] *= grid->get_radialK();
	}
}
