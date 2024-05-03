#include "SI_SC_Matrix_Mag_Func_Cyl3D.cuh"

HOST_DEVICE FIT_Mag_Func_Cyl3D::FIT_Mag_Func_Cyl3D()
{
    Init();
}

HOST_DEVICE void FIT_Mag_Func_Cyl3D::Init()
{
    m_Mag = NULL;
    m_Current = NULL;
    m_C0 = 0;

    m_Elec[0] = m_Elec[1] = m_Elec[2] = m_Elec[3] = NULL;
    m_Curl[0] = m_Curl[1] = m_Curl[2] = m_Curl[3] = 0;

    m_ElecNear[0] = m_ElecNear[1] = m_ElecNear[2] = m_ElecNear[3] = NULL;
    m_CurlNear[0] = m_CurlNear[1] = m_CurlNear[2] = m_CurlNear[3] = 0;

    m_Elec_Ptr_Offset[0] = m_Elec_Ptr_Offset[1] = m_Elec_Ptr_Offset[2] = m_Elec_Ptr_Offset[3] = -1;
    m_ElecNear_Ptr_Offset[0] = m_ElecNear_Ptr_Offset[1] = m_ElecNear_Ptr_Offset[2] = m_ElecNear_Ptr_Offset[3] = -1;
    m_Mag_Ptr_Offset = -1;
    m_Current_Ptr_Offset = -1;
}

HOST_DEVICE FIT_Mag_Func_Cyl3D::~FIT_Mag_Func_Cyl3D()
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

HOST_DEVICE void FIT_Mag_Func_Cyl3D::CheckElecDatas()
{
    for (int i = 0; i < 4; i++)
    {
        if (m_Elec[i] == NULL)
        {
            m_Elec[i] = new Standard_Real;
            (*m_Elec[i]) = 0;
            m_Curl[i] = 0;
            m_Elec_Ptr_Offset[i] = -1;
        }

        if (m_ElecNear[i] == NULL)
        {
            m_ElecNear[i] = new Standard_Real;
            (*m_ElecNear[i]) = 0;
            m_CurlNear[i] = 0;
            m_ElecNear_Ptr_Offset[i] = -1;
        }
    }
}

HOST_DEVICE void FIT_Mag_Func_Cyl3D::ComputeContourZR(Standard_Real& result)
{
    result =
        (*m_Elec[0])     * m_Curl[0]     +
        (*m_Elec[1])     * m_Curl[1]     + 
        (*m_ElecNear[0]) * m_CurlNear[0] + 
        (*m_ElecNear[1]) * m_CurlNear[1] ; 
}

HOST_DEVICE void FIT_Mag_Func_Cyl3D::ComputeContourPhi(Standard_Real& result)
{
    result =
        (*m_Elec[0]) * m_Curl[0] +
        (*m_Elec[1]) * m_Curl[1] +
        (*m_Elec[2]) * m_Curl[2] +
        (*m_Elec[3]) * m_Curl[3] ;
}

HOST_DEVICE void FIT_Mag_Func_Cyl3D::AdvanceMzr()
{
    Standard_Real contour = 0;
    ComputeContourZR(contour);

    *m_Mag = *m_Mag - m_C0 * contour + m_C0 * (*m_Current);
}

HOST_DEVICE void FIT_Mag_Func_Cyl3D::AdvanceMphi()
{
    Standard_Real contour = 0;
    ComputeContourPhi(contour);

    *m_Mag = *m_Mag - m_C0 * contour + m_C0 * (*m_Current);
}
