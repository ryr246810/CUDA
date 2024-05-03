#include "SI_SC_Matrix_Elec_Mur_Func_Cyl3D.cuh"

HOST_DEVICE FIT_Elec_Mur_Func_Cyl3D::FIT_Elec_Mur_Func_Cyl3D()
{
    Init();
}

HOST_DEVICE void FIT_Elec_Mur_Func_Cyl3D::Init()
{
    m_Elec = NULL;
    m_PreTStepEFld = NULL;
    m_Elec_PreStep = NULL;
    m_BE = NULL;
    m_AE = NULL;
    m_PreTStep = NULL;
    m_VBar = 0.0;

    m_Elec_Ptr_Offset = m_PreTStepEFld_Ptr_Offset
                      = m_Elec_PreStep_Ptr_Offset
                      = m_BE_Ptr_Offset
                      = m_AE_Ptr_Offset
                      = m_PreTStep_Ptr_Offset
                      = -1;
}

HOST_DEVICE FIT_Elec_Mur_Func_Cyl3D::~FIT_Elec_Mur_Func_Cyl3D()
{

}

HOST_DEVICE void FIT_Elec_Mur_Func_Cyl3D::Advance_Damping(const Standard_Real& theta)
{
    *m_AE = (1.0 - theta)*(*m_Elec) + theta*(*m_AE);
    Standard_Real m_Elec_PreStep_tmp = (*m_Elec);

    Standard_Real vbar = m_VBar * 1.0;
    Standard_Real currE = (*m_PreTStepEFld);
    Standard_Real oldE = (*m_PreTStep);
    Standard_Real tmpE = (oldE - currE*(1.0-vbar))/(1.0+vbar);

    oldE = tmpE*(1.0-vbar) + currE*(1.0+vbar);
    *m_PreTStep = oldE;
    *m_Elec = tmpE;

    *m_BE = (1.0 + 0.5*theta)*(*m_Elec) - 0.5*m_Elec_PreStep_tmp + 0.5*(1.0-theta)*(*m_AE);
}

HOST_DEVICE void FIT_Elec_Mur_Func_Cyl3D::AdvanceEzr_Damping(const Standard_Real& theta)
{
    Advance_Damping(theta);
}

HOST_DEVICE void FIT_Elec_Mur_Func_Cyl3D::AdvanceEphi_Damping(const Standard_Real& theta)
{
    Advance_Damping(theta);
}

HOST_DEVICE void FIT_Elec_Mur_Func_Cyl3D::AdvanceEzr_Damping_TFunc(const Standard_Real& theta, Standard_Real Ebar, Standard_Real Ebar2)
{
    *m_AE = (1.0 - theta)*(*m_Elec) + theta*(*m_AE);
    Standard_Real m_Elec_PreStep_tmp = (*m_Elec);

    Standard_Real vbar = m_VBar * 1.0;
    Standard_Real currE = (*m_PreTStepEFld);
    Standard_Real oldE = (*m_PreTStep);
    Standard_Real tmpE = (oldE - (currE-Ebar2)*(1.0-vbar))/(1.0+vbar) + Ebar;

    oldE = (tmpE-Ebar)*(1.0-vbar) + (currE-Ebar2)*(1.0+vbar);
    *m_PreTStep = oldE;
    *m_Elec = tmpE;

    *m_BE = (1.0 + 0.5*theta)*(*m_Elec) - 0.5*m_Elec_PreStep_tmp + 0.5*(1.0-theta)*(*m_AE);
}
