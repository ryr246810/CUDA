#ifndef _SI_SC_Matrix_Mag_Func_Cyl3D_HeaderFile
#define _SI_SC_Matrix_Mag_Func_Cyl3D_HeaderFile

#include <string.h>
#include "Standard_TypeDefine.hxx"

#include "CUDAHeader.cuh"

class FIT_Mag_Func_Cyl3D
{
public:
    HOST_DEVICE FIT_Mag_Func_Cyl3D();

    HOST_DEVICE void Init();
    
    HOST_DEVICE ~FIT_Mag_Func_Cyl3D();

public:
    Standard_Real       *m_Mag;
    Standard_Integer    m_Mag_Ptr_Offset;

    Standard_Real       *m_Current;
    Standard_Integer    m_Current_Ptr_Offset; 

    Standard_Real       *m_Elec[4];
    Standard_Integer    m_Elec_Ptr_Offset[4];

    Standard_Real       *m_ElecNear[4];
    Standard_Integer    m_ElecNear_Ptr_Offset[4];

    Standard_Real m_Curl[4];

    Standard_Real m_CurlNear[4];

    Standard_Real m_C0;

    HOST_DEVICE void CheckElecDatas();

    HOST_DEVICE void ComputeContourZR(Standard_Real &result);
    DEVICE void d_ComputeContourZR(Standard_Real& result, Standard_Real* EphiDatasPtr, Standard_Real* EzrDatasPtr);

    HOST_DEVICE void ComputeContourPhi(Standard_Real &result);
    DEVICE void d_ComputeContourPhi(Standard_Real& result, Standard_Real* EzrDatasPtr);

    HOST_DEVICE void AdvanceMzr();
    DEVICE void d_AdvanceMzr(Standard_Real* MzrDatasPtr, Standard_Real* EphiDatasPtr, Standard_Real* EzrDatasPtr);

    HOST_DEVICE void AdvanceMphi();
    DEVICE void d_AdvanceMphi(Standard_Real* MphiDatasPtr, Standard_Real* EzrDatasPtr);


};

#endif