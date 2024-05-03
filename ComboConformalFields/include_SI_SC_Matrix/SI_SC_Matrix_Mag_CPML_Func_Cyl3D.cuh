#ifndef _SI_SC_Matrix_Mag_CPML_Func_Cyl3D_HeaderFile
#define _SI_SC_Matrix_Mag_CPML_Func_Cyl3D_HeaderFile

#include <math.h>
#include <string.h>
#include <iostream>
#include "Standard_TypeDefine.hxx"
#include "CUDAHeader.cuh"

class FIT_Mag_PML_Func_Cyl3D
{
public:
    HOST_DEVICE FIT_Mag_PML_Func_Cyl3D();
    HOST_DEVICE ~FIT_Mag_PML_Func_Cyl3D();

    HOST_DEVICE void Init();

public:
    Standard_Real* m_Mag;
    Standard_Integer m_Mag_Ptr_Offset;

    Standard_Real* m_Current;
    Standard_Integer m_Current_Ptr_Offset;

    Standard_Real* m_PM1;
    Standard_Integer m_PM1_Ptr_Offset;

    Standard_Real* m_PM2;
    Standard_Integer m_PM2_Ptr_Offset;

    Standard_Real m_C0;

    Standard_Real* m_Elec[4];
    Standard_Integer m_Elec_Ptr_Offset[4];

    Standard_Real* m_ElecNear[4];
    Standard_Integer m_ElecNear_Ptr_Offset[4];
    
    Standard_Real m_P1;
    Standard_Real m_P2;
    
    Standard_Real m_Curl1[4];
    Standard_Real m_Curl1Near[4];
    Standard_Real m_CurlP1;
    Standard_Real m_Curl2[4];
    Standard_Real m_CurlP2;
    
    Standard_Real a[2];
    Standard_Real b[2];
    Standard_Real invKappa[2];

    HOST_DEVICE void CheckElecDatas();

    HOST_DEVICE void ComputContour1(Standard_Real& result);
    DEVICE void d_ComputContour1(Standard_Real& result, Standard_Real* ElecArray);

    HOST_DEVICE void ComputContour11(Standard_Real& result);
    DEVICE void d_ComputContourl1(Standard_Real& result, Standard_Real* ElecArray);

    HOST_DEVICE void ComputContour2(Standard_Real& result);
    DEVICE void d_ComputContour2(Standard_Real& result, Standard_Real* ElecArray);

    HOST_DEVICE void ComputeContour1Near(Standard_Real& result);
    DEVICE void d_ComputContour1Near(Standard_Real& result, Standard_Real* ElecArray);

    HOST_DEVICE void ComputP(const Standard_Real& contour1,
        const Standard_Real& contour2);
    DEVICE void d_ComputeP(const Standard_Real& contour1,
        const Standard_Real& contour2, Standard_Real* MagArray);
    
    HOST_DEVICE void ComputP(const Standard_Real& contour1);
    DEVICE void d_ComputeP(const Standard_Real& contour1, Standard_Real* MagArray);

    HOST_DEVICE void AdvanceMzr();
    DEVICE void d_AdvanceMzr(Standard_Real* MzrDatasPtr, Standard_Real* EphiDatasPtr,
        Standard_Real* EzrDatasPtr);

    HOST_DEVICE void AdvanceMphi();
    DEVICE void d_AdvanceMphi(Standard_Real* MphiDatasPtr, Standard_Real* EzrDatasPtr);
};

#endif