#ifndef _SI_SC_Matrix_Elec_Func_Cyl3D_HeaderFile
#define _SI_SC_Matrix_Elec_Func_Cyl3D_HeaderFile

#include <math.h>
#include <string.h>
#include <iostream>
#include <vector>
#include "Standard_TypeDefine.hxx"
using namespace std;
#include "CUDAHeader.cuh"

class FIT_Elec_Func_Cyl3D
{
public:
    HOST_DEVICE FIT_Elec_Func_Cyl3D();
    HOST_DEVICE ~FIT_Elec_Func_Cyl3D();

    HOST_DEVICE void Init();

public:
    Standard_Real* m_Elec;
    Standard_Integer m_Elec_Ptr_Offset;
    Standard_Real* m_Current;
    Standard_Integer m_Current_Ptr_Offset;
    
    Standard_Real  m_elec;
    Standard_Real  m_current;

    Standard_Real* m_Elec_PreStep;
    Standard_Integer m_Elec_PreStep_Ptr_Offset;
    Standard_Real* m_BE;
    Standard_Integer m_BE_Ptr_Offset;
    
    Standard_Real* m_AE;
    Standard_Integer m_AE_Ptr_Offset;
    
    Standard_Real m_C0;
    Standard_Real m_C2;
    Standard_Real m_C3;
    Standard_Real m_DualContour;

    Standard_Real* m_Mag[4];
    Standard_Integer m_Mag_Ptr_Offset[4];
    Standard_Real m_Curl[4];

    Standard_Real* m_MagNear[4];
    Standard_Integer m_MagNear_Ptr_Offset[4];
    Standard_Real m_CurlNear[4];

public:
    HOST_DEVICE void CheckMagDatas();

    HOST_DEVICE void ComputeContourZR(Standard_Real& result);
    DEVICE void d_ComputeContourZR(Standard_Real& result, Standard_Real* MagArray1, 
        Standard_Real* MagArray2);

    HOST_DEVICE Standard_Real ComputeContourAxis();
    DEVICE Standard_Real d_ComputeContourAxis(Standard_Real* MagArray);

    HOST_DEVICE void ComputeContourPhi(Standard_Real& result);  
    DEVICE void d_ComputeContourPhi(Standard_Real& result, Standard_Real* MagArray); 

    HOST_DEVICE void AdvanceEzr();
    DEVICE void d_AdvanceEzr(Standard_Real* EzrDatasPtr, Standard_Real* MphiDatasPtr,
        Standard_Real* MzrDatasPtr);

    HOST_DEVICE void AdvanceEphi();
    DEVICE void d_AdvanceEphi(Standard_Real* EphiDatasPtr, Standard_Real* MzrDatasPtr);

    HOST_DEVICE void Advance_Damping_A(const Standard_Real& theta);
    
    HOST_DEVICE void Advance_Damping_B(const Standard_Real& theta);
    
    HOST_DEVICE void Advance_Damping_A_2(const Standard_Real& theta);

    HOST_DEVICE void Advance_Damping_B_2(const Standard_Real& theta);
    
    HOST_DEVICE void Advance_Damping_1(const Standard_Real& theta);
    DEVICE void d_Advance_Damping_1(const Standard_Real theta, Standard_Real* ElecArray);
    
    HOST_DEVICE void Advance_Damping_2(const Standard_Real& theta);
    DEVICE void d_Advance_Damping_2(const Standard_Real theta, Standard_Real* ElecArray);

    HOST_DEVICE void Advance_Damping2(const Standard_Real& theta);

    HOST_DEVICE void Advance_Damping1(const Standard_Real& theta);

    HOST_DEVICE void AdvanceEzr_Damping(const Standard_Real& theta);
    DEVICE void d_AdvanceEzr_Damping(const Standard_Real theta, Standard_Real* EzrDatasPtr, 
        Standard_Real* MphiDatasPtr, Standard_Real* MzrDatasPtr);

    HOST_DEVICE void AdvanceEphi_Damping(const Standard_Real& theta);
    DEVICE void d_AdvanceEphi_Damping(const Standard_Real theta, Standard_Real* EphiDatasPtr, 
        Standard_Real* MzrDatasPtr);
};

#endif