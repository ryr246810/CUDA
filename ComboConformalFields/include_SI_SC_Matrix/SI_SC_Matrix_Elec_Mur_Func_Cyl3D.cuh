#ifndef _SI_SC_Matrix_Elec_Mur_Func_Cyl3D_HeaderFile
#define _SI_SC_Matrix_Elec_Mur_Func_Cyl3D_HeaderFile

#include <math.h>
#include <string.h>
#include "Standard_TypeDefine.hxx"
#include "CUDAHeader.cuh"

class FIT_Elec_Mur_Func_Cyl3D
{
public:
    HOST_DEVICE FIT_Elec_Mur_Func_Cyl3D();
    HOST_DEVICE ~FIT_Elec_Mur_Func_Cyl3D();

    HOST_DEVICE void Init();

public:
    Standard_Real* m_Elec;
    Standard_Integer m_Elec_Ptr_Offset;

    Standard_Real* m_PreTStepEFld;
    Standard_Integer m_PreTStepEFld_Ptr_Offset;

    Standard_Real* m_Elec_PreStep;
    Standard_Integer m_Elec_PreStep_Ptr_Offset;

    Standard_Real* m_BE;
    Standard_Integer m_BE_Ptr_Offset;

    Standard_Real* m_AE;
    Standard_Integer m_AE_Ptr_Offset;

    Standard_Real* m_PreTStep;
    Standard_Integer m_PreTStep_Ptr_Offset;

    Standard_Real m_VBar;

public:
    HOST_DEVICE void Advance_Damping(const Standard_Real& theta);
    HOST_DEVICE void AdvanceEzr_Damping(const Standard_Real& theta);
    HOST_DEVICE void AdvanceEphi_Damping(const Standard_Real& theta);    
    HOST_DEVICE void AdvanceEzr_Damping_TFunc(const Standard_Real& theta, Standard_Real Ebar, Standard_Real Ebar2);

    DEVICE void d_AdvanceEzr_Damping(const Standard_Real theta, Standard_Real* EzrDatasPtr);
    DEVICE void d_AdvanceEphi_Damping(const Standard_Real theta, Standard_Real* EphiDatasPtr);
    DEVICE void d_AdvanceEzr_Damping_TFunc(const Standard_Real theta, Standard_Real* EzrDatasPtr, Standard_Real Ebar, Standard_Real Ebar2);

};


#endif