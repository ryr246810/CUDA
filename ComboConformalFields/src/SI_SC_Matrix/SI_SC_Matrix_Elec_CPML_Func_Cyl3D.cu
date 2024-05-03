#include "SI_SC_Matrix_Elec_CPML_Func_Cyl3D.cuh"

HOST_DEVICE FIT_Elec_PML_Func_Cyl3D::FIT_Elec_PML_Func_Cyl3D()
{
    Init();
}

HOST_DEVICE void FIT_Elec_PML_Func_Cyl3D::Init()
{
    m_Elec = NULL;
    m_Current = NULL;
    m_Elec_PreStep = NULL;
    m_BE = NULL;
    m_AE = NULL;
    m_PE1 = NULL;
    m_PE2 = NULL;

    m_elec = 0;
    m_pe1 = 0;
    a[0] = 0;
    a[1] = 0;
    b[0] = 0;
    b[1] = 0;
    invKappa[0] = 0;
    invKappa[1] = 0;
    m_C0 = 0;
    m_C2 = 0;
    m_Contour = 0;
    m_Contour1 = 0;
    m_DualContour = 0;
    m_CurlP1 = 0;
    m_CurlP2 = 0;

    m_Mag[0]   = m_Mag[1]   = m_Mag[2]   = m_Mag[3]   = NULL;
    m_Curl1[0] = m_Curl1[1] = m_Curl1[2] = m_Curl1[3] = 0;
    m_Curl2[0] = m_Curl2[1] = m_Curl2[2] = m_Curl2[3] = 0;

    m_MagNear[0]   = m_MagNear[1]   = m_MagNear[2]   = m_MagNear[3]   = NULL;
    m_Curl1Near[0] = m_Curl1Near[1] = m_Curl1Near[2] = m_Curl1Near[3] = 0;

    m_Elec_Ptr_Offset = m_Current_Ptr_Offset
                      = m_Elec_PreStep_Ptr_Offset
                      = m_BE_Ptr_Offset
                      = m_AE_Ptr_Offset
                      = m_PE1_Ptr_Offset
                      = m_PE2_Ptr_Offset
                      = -1;
    m_Mag_Ptr_Offset[0]     = m_Mag_Ptr_Offset[1]     = m_Mag_Ptr_Offset[2]     = m_Mag_Ptr_Offset[3]     = -1;
    m_MagNear_Ptr_Offset[0] = m_MagNear_Ptr_Offset[1] = m_MagNear_Ptr_Offset[2] = m_MagNear_Ptr_Offset[3] = -1;  
}

HOST_DEVICE FIT_Elec_PML_Func_Cyl3D::~FIT_Elec_PML_Func_Cyl3D()
{
    for(int i = 0; i < 4; ++i){
        if(m_Mag[i] != NULL){
            delete m_Mag[i];
            m_Mag[i] = NULL;
        }

        if(m_MagNear[i] != NULL){
            delete m_MagNear[i];
            m_MagNear[i] = NULL;
        }
    }
}

HOST_DEVICE void FIT_Elec_PML_Func_Cyl3D::CheckMagDatas()
{
    for(int i = 0; i < 4; ++i){
        if(m_Mag[i] == NULL){
            m_Mag[i] = new Standard_Real;
            (*m_Mag[i]) = 0;
            m_Curl1[i] = 0;
            m_Curl2[i] = 0;
        }

        if(m_MagNear[i] == NULL){
            m_MagNear[i] = new Standard_Real;
            (*m_MagNear[i]) = 0;
            m_Curl1Near[i] = 0;
        }
    }
}

HOST_DEVICE void FIT_Elec_PML_Func_Cyl3D::ComputContour1(Standard_Real& result)
{
    result = 
        m_Curl1[0] * (*(m_Mag[0])) + 
        m_Curl1[1] * (*(m_Mag[1])) + 
        m_Curl1[2] * (*(m_Mag[2])) +
        m_Curl1[3] * (*(m_Mag[3])) ;
}

HOST_DEVICE void FIT_Elec_PML_Func_Cyl3D::ComputContour11(Standard_Real& result)
{
    result = 
        m_Curl1[0] * (*(m_Mag[0])) + 
        m_Curl1[1] * (*(m_Mag[1])) ;
}

HOST_DEVICE Standard_Real FIT_Elec_PML_Func_Cyl3D::ComputContourAxis()
{
    Standard_Real result;
    result = 
        m_Curl1[0] * (*(m_Mag[0])) + 
        m_Curl1[1] * (*(m_Mag[1])) ;
    
    return result;
}

HOST_DEVICE void FIT_Elec_PML_Func_Cyl3D::ComputContour1Near(Standard_Real& result)
{
    result = 
        m_Curl1Near[0] * (*(m_MagNear[0])) + 
        m_Curl1Near[1] * (*(m_MagNear[1])) ;
}

HOST_DEVICE void FIT_Elec_PML_Func_Cyl3D::ComputContour2(Standard_Real& result)
{
    result =
        m_Curl2[0] * (*(m_Mag[0])) +
        m_Curl2[1] * (*(m_Mag[1])) +
        m_Curl2[2] * (*(m_Mag[2])) +
        m_Curl2[3] * (*(m_Mag[3])) ;
}

HOST_DEVICE void FIT_Elec_PML_Func_Cyl3D::ComputP1(const Standard_Real& contour1)
{
    (*m_PE1) = b[0] * (*m_PE1) + a[0] * contour1;
}

HOST_DEVICE void FIT_Elec_PML_Func_Cyl3D::ComputP2(const Standard_Real& contour2)
{
    (*m_PE2) = b[1] * (*m_PE2) + a[1] * contour2;
}

HOST_DEVICE void FIT_Elec_PML_Func_Cyl3D::Advance_Damping_A(const Standard_Real& theta)
{
    *m_AE = (0.5 * theta) * (*m_AE) + (1.0 - 0.5 * theta) * (*m_Elec);
    *m_Elec_PreStep = *m_Elec;
}

HOST_DEVICE void FIT_Elec_PML_Func_Cyl3D::Advance_Damping_B(const Standard_Real& theta)
{
    *m_BE =
        (1.0 + 0.25 * theta) * (*m_Elec)
        - 0.5 * (*m_Elec_PreStep)
        + (0.5 - 0.25 * theta) * (*m_AE);
}

HOST_DEVICE void FIT_Elec_PML_Func_Cyl3D::Advance_Damping_A_2(const Standard_Real& theta)
{
    *m_AE = (*m_Elec_PreStep) + theta * (*m_AE);
    *m_Elec_PreStep = *m_Elec;
}

HOST_DEVICE void FIT_Elec_PML_Func_Cyl3D::Advance_Damping_B_2(const Standard_Real& theta)
{
    *m_BE =
        (1.0 + 0.5 * theta) * (*m_Elec)
        - theta * (1.0 - 0.5 * theta) * (*m_Elec_PreStep)
        + 0.5 * (1.0 - theta) * (1.0 - theta) * theta * (*m_AE);
}

HOST_DEVICE void FIT_Elec_PML_Func_Cyl3D::Advance_Damping_1(const Standard_Real& theta)
{
    *m_AE = (1.0 - theta) * (*m_Elec) + theta * (*m_AE);
    *m_Elec_PreStep = *m_Elec;
}

HOST_DEVICE void FIT_Elec_PML_Func_Cyl3D::Advance_Damping_2(const Standard_Real& theta)
{
    *m_BE = (1.0 + 0.5 * theta) * (*m_Elec) - 0.5 * (*m_Elec_PreStep) + 0.5 * (1.0 - theta) * (*m_AE);
}

HOST_DEVICE void FIT_Elec_PML_Func_Cyl3D::Advance_Damping1(const Standard_Real& theta)
{
    Advance_Damping_A(theta);
    // Advance(chiParam);
    Advance_Damping_B(theta);
}

HOST_DEVICE void FIT_Elec_PML_Func_Cyl3D::Advance_Damping2(const Standard_Real& theta)
{
    Advance_Damping_A_2(theta);
    // Advance(chiParam);
    Advance_Damping_B_2(theta);
}

HOST_DEVICE void FIT_Elec_PML_Func_Cyl3D::AdvanceEzr()
{
    Standard_Real contour1, contour1Near;  
    ComputContour11(contour1);
    ComputP1(contour1);
    ComputContour1Near(contour1Near);
    *m_Elec = m_C0 * (*m_Elec) + contour1 * m_CurlP1 + contour1Near * m_C2 + m_C2 * (*m_PE1);
}

HOST_DEVICE void FIT_Elec_PML_Func_Cyl3D::AdvanceEphi()
{
    Standard_Real contour1;  
    ComputContour1(contour1);
    Standard_Real contour2;  
    ComputContour2(contour2);
    ComputP1(contour1);
    ComputP2(contour2);
    *m_Elec = m_C0 * (*m_Elec) + contour1 * m_CurlP1 + contour2 * m_CurlP2 + m_C2 * (*m_PE1)+ m_C2 * (*m_PE2);  
}

HOST_DEVICE void FIT_Elec_PML_Func_Cyl3D::AdvanceEzr_Damping(const Standard_Real& theta)
{
    Advance_Damping_1(theta);
    AdvanceEzr();
    Advance_Damping_2(theta);
}

HOST_DEVICE void FIT_Elec_PML_Func_Cyl3D::AdvanceEphi_Damping(const Standard_Real& theta)
{
    Advance_Damping_1(theta);
    AdvanceEphi();
    Advance_Damping_2(theta);
}






