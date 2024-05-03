#include "SI_SC_Matrix_Mag_CPML_Func_Cyl3D.cuh"

HOST_DEVICE FIT_Mag_PML_Func_Cyl3D::FIT_Mag_PML_Func_Cyl3D()
{
    Init();
}

HOST_DEVICE void FIT_Mag_PML_Func_Cyl3D::Init()
{
    m_Mag = NULL;
    m_Current = NULL;
    m_PM1 = NULL;
    m_PM2 = NULL;
    a[0] = 0;
    a[1] = 0;
    b[0] = 0;
    b[1] = 0;
    invKappa[0] = 0;
    invKappa[1] = 0;
    m_C0 = 0;
    m_CurlP1 = 0;
    m_CurlP2 = 0;
    m_Elec[0]  = m_Elec[1]  = m_Elec[2]  = m_Elec[3]  = 0;
	m_Curl1[0] = m_Curl1[1] = m_Curl1[2] = m_Curl1[3] = 0;
	m_Curl2[0] = m_Curl2[1] = m_Curl2[2] = m_Curl2[3] = 0;

    m_ElecNear[0]  = m_ElecNear[1]  = m_ElecNear[2]  = m_ElecNear[3]  = 0;
    m_Curl1Near[0] = m_Curl1Near[1] = m_Curl1Near[2] = m_Curl1Near[3] = 0;

    m_Mag_Ptr_Offset    = m_Current_Ptr_Offset
                        = m_PM1_Ptr_Offset
                        = m_PM2_Ptr_Offset
                        = -1;
    m_Elec_Ptr_Offset[0]     = m_Elec_Ptr_Offset[1]     = m_Elec_Ptr_Offset[2]     = m_Elec_Ptr_Offset[3]     = -1;
    m_ElecNear_Ptr_Offset[0] = m_ElecNear_Ptr_Offset[1] = m_ElecNear_Ptr_Offset[2] = m_ElecNear_Ptr_Offset[3] = -1;   
}

HOST_DEVICE FIT_Mag_PML_Func_Cyl3D::~FIT_Mag_PML_Func_Cyl3D()
{
    for(int i = 0; i < 4; ++i){
        if(m_Elec[i] != NULL){
            delete m_Elec[i];
            m_Elec[i] = NULL;
        }

        if(m_ElecNear[i] != NULL){
            delete m_ElecNear[i];
            m_ElecNear[i] = NULL;
        }
    }
}

HOST_DEVICE void FIT_Mag_PML_Func_Cyl3D::CheckElecDatas()
{
    for(int i = 0; i < 4; ++i){
        if (m_Elec[i] == NULL){
            m_Elec[i] = new Standard_Real;
            (*m_Elec[i]) = 0;
            m_Curl1[i] = 0;
            m_Curl2[i] = 0;
            m_Elec_Ptr_Offset[i] = -1;
        }

        if (m_ElecNear[i] == NULL){
            m_ElecNear[i] = new Standard_Real;
            (*m_ElecNear[i]) = 0;
            m_Curl1Near[i] = 0;
            m_ElecNear_Ptr_Offset[i] = -1;
        }
    }
}

HOST_DEVICE void FIT_Mag_PML_Func_Cyl3D::ComputContour1(Standard_Real& result)
{
    result =
        (*m_Elec[0]) * m_Curl1[0] + 
        (*m_Elec[1]) * m_Curl1[1] + 
        (*m_Elec[2]) * m_Curl1[2] + 
        (*m_Elec[3]) * m_Curl1[3] ;
}

HOST_DEVICE void FIT_Mag_PML_Func_Cyl3D::ComputContour2(Standard_Real& result)
{
    result =
        (*m_Elec[0]) * m_Curl2[0] + 
        (*m_Elec[1]) * m_Curl2[1] + 
        (*m_Elec[2]) * m_Curl2[2] + 
        (*m_Elec[3]) * m_Curl2[3] ;
}

HOST_DEVICE void FIT_Mag_PML_Func_Cyl3D::ComputContour11(Standard_Real& result)
{
    result =
        (*m_Elec[0]) * m_Curl1[0] + 
        (*m_Elec[1]) * m_Curl1[1] ;
};

HOST_DEVICE void FIT_Mag_PML_Func_Cyl3D::ComputeContour1Near(Standard_Real& result)
{
    result = 
        (*m_ElecNear[0]) * m_Curl1Near[0] + 
        (*m_ElecNear[1]) * m_Curl1Near[1] ;
}

HOST_DEVICE void FIT_Mag_PML_Func_Cyl3D::ComputP(const Standard_Real& contour1,
    const Standard_Real& contour2)
{
    (*m_PM1) = b[0] * (*m_PM1) + a[0] * contour1;
    (*m_PM2) = b[1] * (*m_PM2) + a[1] * contour2;
}

HOST_DEVICE void FIT_Mag_PML_Func_Cyl3D::ComputP(const Standard_Real& contour1)
{
    (*m_PM1) = b[0] * (*m_PM1) + a[0] * contour1;
}

HOST_DEVICE void FIT_Mag_PML_Func_Cyl3D::AdvanceMzr()
{
    Standard_Real contour1, contour1Near; 
    ComputContour11(contour1);
    ComputeContour1Near(contour1Near);
    ComputP(contour1);
    *m_Mag = *m_Mag - m_CurlP1 * contour1 - m_C0 * contour1Near - m_C0 * (*m_PM1);
}

HOST_DEVICE void FIT_Mag_PML_Func_Cyl3D::AdvanceMphi()
{
    Standard_Real contour1; ComputContour1(contour1);
    Standard_Real contour2; ComputContour2(contour2);
    ComputP(contour1, contour2);
    
    *m_Mag = *m_Mag - m_CurlP1 * contour1 - m_CurlP2 * contour2 - m_C0 * (*m_PM1) - m_C0 * (*m_PM2);
}