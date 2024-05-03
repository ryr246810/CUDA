#ifndef SPECIES_Cyl3D_SubFunc
#define SPECIES_Cyl3D_SubFunc

#include <iostream>
#include "CUDAHeader.cuh"
#include "Standard_TypeDefine.hxx"
#include "TxVector2D.h"
#include "TxVector.h"
#include "TxStreams.h"
#include "TxHierAttribSet.h"
#include "VecDoub.hxx"
#include <vector>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

int Cuda_Constant_Vars_Init(const TxVector2D<Standard_Real> & orgs,
                            const map<Standard_Integer, vector<Standard_Real>, less<Standard_Integer> > & lVectors,
                            const map<Standard_Integer, vector<Standard_Real>, less<Standard_Integer> > & dlVectors,
                            const Standard_Real minSteps[2],
                            const Standard_Integer dimensions[2],
                            const Standard_Integer m_phi_number,
                            const Standard_Real minStep,
                            const Standard_Real chargeOverMass);

__device__ Standard_Real d_LVectors_z[Z_DIMENSION];
__device__ Standard_Real d_LVectors_r[R_DIMENSION];
__device__ Standard_Real d_DLVectors_z[Z_DIMENSION];
__device__ Standard_Real d_DLVectors_r[R_DIMENSION];
__device__ Standard_Integer d_Phi_number;
__device__ Standard_Real d_chargeOverMass;
__device__ Standard_Size d_LVectors_Size[2];
__device__ Standard_Size d_DLVectors_Size[2];
__device__ Standard_Real d_MinSteps[2];
__device__ Standard_Real d_MinStep;
__device__ Standard_Integer d_Dimensions[2];
__device__ Standard_Real d_Orgs[2];
__device__ Standard_Integer d_idx[3];


#endif