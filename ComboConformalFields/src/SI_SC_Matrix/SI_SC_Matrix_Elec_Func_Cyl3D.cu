#include "SI_SC_Matrix_Elec_Func_Cyl3D.cuh"

HOST_DEVICE FIT_Elec_Func_Cyl3D::FIT_Elec_Func_Cyl3D()
{
    Init();
}

HOST_DEVICE void FIT_Elec_Func_Cyl3D::Init()
{
    m_Elec = NULL;
    m_Current = NULL;
    m_Elec_PreStep = NULL;
    m_BE = NULL;
    m_AE = NULL;
    m_elec = 0;
    m_current = 0;
    m_C0 = 0;
    m_C2 = 0;
    m_C3 = 0;
    m_DualContour = 0;
    m_Mag[0]  = m_Mag[1]  = m_Mag[2]  = m_Mag[3]  = NULL;
    m_Curl[0] = m_Curl[1] = m_Curl[2] = m_Curl[3] = 0;

    m_MagNear[0]  = m_MagNear[1]  = m_MagNear[2]  = m_MagNear[3]  = NULL;
    m_CurlNear[0] = m_CurlNear[1] = m_CurlNear[2] = m_CurlNear[3] = 0;

    m_Elec_Ptr_Offset = -1;
    m_Current_Ptr_Offset = -1;
    m_Elec_PreStep_Ptr_Offset = -1;
    m_BE_Ptr_Offset = -1;
    m_AE_Ptr_Offset = -1;
    m_Mag_Ptr_Offset[0] = m_Mag_Ptr_Offset[1] = m_Mag_Ptr_Offset[2] = m_Mag_Ptr_Offset[3] = -1;
    m_MagNear_Ptr_Offset[0] = m_MagNear_Ptr_Offset[1] = m_MagNear_Ptr_Offset[2] = m_MagNear_Ptr_Offset[3] = -1;
}

HOST_DEVICE FIT_Elec_Func_Cyl3D::~FIT_Elec_Func_Cyl3D()
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

HOST_DEVICE void FIT_Elec_Func_Cyl3D::CheckMagDatas()
{
    for(int i = 0; i < 4; ++i){
        if (m_Mag[i] == NULL){
            m_Mag[i] = new Standard_Real;
            (*m_Mag[i]) = 0;
            m_Curl[i] = 0;
            m_Mag_Ptr_Offset[i] = -1;
        }

        if (m_MagNear[i] == NULL){
            m_MagNear[i] = new Standard_Real;
            (*m_MagNear[i]) = 0;
            m_CurlNear[i] = 0;
            m_MagNear_Ptr_Offset[i] = -1;
        }
    }
}

HOST_DEVICE void FIT_Elec_Func_Cyl3D::ComputeContourZR(Standard_Real& result)
{
    result =
        (*m_Mag[0])     * m_Curl[0]     + 
        (*m_Mag[1])     * m_Curl[1]     +
        (*m_MagNear[0]) * m_CurlNear[0] + 
        (*m_MagNear[1]) * m_CurlNear[1] ;
}

HOST_DEVICE Standard_Real FIT_Elec_Func_Cyl3D::ComputeContourAxis()
{
    Standard_Real res;
    res = 
        (*m_Mag[0]) * m_Curl[0] + 
        (*m_Mag[1]) * m_Curl[1] ;

    return res;
}

HOST_DEVICE void FIT_Elec_Func_Cyl3D::ComputeContourPhi(Standard_Real& result)
{
    result =
        (*m_Mag[0]) * m_Curl[0] + 
        (*m_Mag[1]) * m_Curl[1] + 
        (*m_Mag[2]) * m_Curl[2] + 
        (*m_Mag[3]) * m_Curl[3] ;
}

HOST_DEVICE void FIT_Elec_Func_Cyl3D::AdvanceEzr()
{
    Standard_Real contour = 0.0;
    ComputeContourZR(contour);
    *m_Elec = m_C0 * (*m_Elec) + m_C2 * contour - m_C3 * (*m_Current);
}

HOST_DEVICE void FIT_Elec_Func_Cyl3D::AdvanceEphi()
{
    Standard_Real contour = 0.0;
    ComputeContourPhi(contour);
    *m_Elec = m_C0 * (*m_Elec) + m_C2 * contour - m_C3 * (*m_Current);
}

HOST_DEVICE void FIT_Elec_Func_Cyl3D::Advance_Damping_A(const Standard_Real& theta)
{
    *m_AE = (0.5 * theta) * (*m_AE) + (1.0 - 0.5 * theta) * (*m_Elec);
    *m_Elec_PreStep = *m_Elec;
}

HOST_DEVICE void FIT_Elec_Func_Cyl3D::Advance_Damping_B(const Standard_Real& theta)
{
    *m_BE =
        (1.0 + 0.25 * theta) * (*m_Elec)
        - 0.5 * (*m_Elec_PreStep)
        + (0.5 - 0.25 * theta) * (*m_AE);
}

HOST_DEVICE void FIT_Elec_Func_Cyl3D::Advance_Damping_A_2(const Standard_Real& theta)
{
    *m_AE = (*m_Elec_PreStep) + theta * (*m_AE);
    *m_Elec_PreStep = *m_Elec;
}

HOST_DEVICE void FIT_Elec_Func_Cyl3D::Advance_Damping_B_2(const Standard_Real& theta)
{
    *m_BE =
        (1.0 + 0.5 * theta) * (*m_Elec)
        - theta * (1.0 - 0.5 * theta) * (*m_Elec_PreStep)
        + 0.5 * (1.0 - theta) * (1.0 - theta) * theta * (*m_AE);
}

HOST_DEVICE void FIT_Elec_Func_Cyl3D::Advance_Damping_1(const Standard_Real& theta)
{
    *m_AE = (1.0 - theta) * (*m_Elec) + theta * (*m_AE);
    *m_Elec_PreStep = *m_Elec;
}

HOST_DEVICE void FIT_Elec_Func_Cyl3D::Advance_Damping_2(const Standard_Real& theta)
{
    *m_BE =(1.0 + 0.5 * theta) * (*m_Elec) - 0.5 * (*m_Elec_PreStep) + 0.5 * (1.0 - theta) * (*m_AE);
}

HOST_DEVICE void FIT_Elec_Func_Cyl3D::Advance_Damping2(const Standard_Real& theta)
{
    Advance_Damping_A_2(theta);
    // Advance();
    Advance_Damping_B_2(theta);
}

HOST_DEVICE void FIT_Elec_Func_Cyl3D::Advance_Damping1(const Standard_Real& theta)
{
    Advance_Damping_A(theta);
    // Advance();
    Advance_Damping_B(theta);
}

HOST_DEVICE void FIT_Elec_Func_Cyl3D::AdvanceEzr_Damping(const Standard_Real& theta)
{
    Advance_Damping_1(theta);
        
    AdvanceEzr();
    
    Advance_Damping_2(theta);
}

HOST_DEVICE void FIT_Elec_Func_Cyl3D::AdvanceEphi_Damping(const Standard_Real& theta)
{
    Advance_Damping_1(theta);
        
    AdvanceEphi();
    
    Advance_Damping_2(theta);
}

