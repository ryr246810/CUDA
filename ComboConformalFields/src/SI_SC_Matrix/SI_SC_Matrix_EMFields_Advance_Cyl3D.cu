#include "SI_SC_Matrix_EMFields_Cyl3D.hxx"
#include "ComboFields_Dynamic_Srcs_Cyl3D.hxx"
#include "SI_SC_IntegralEquation.hxx"

#include "CUDAHeader.cuh"
#include "SI_SC_Matrix_Mag_Func_Cyl3D.cuh"
#include "SI_SC_Matrix_Mag_CPML_Func_Cyl3D.cuh"
#include "SI_SC_Matrix_Elec_Func_Cyl3D.cuh"

// Standard_Real EMFields_Elapsed_cuda = 0.0;

Standard_Real *h_d_Ebar;
Standard_Real *h_d_Ebar2;
Standard_Real *m_h_d_MphiDatasPtr;
Standard_Real *m_h_d_MzrDatasPtr;
Standard_Real *m_h_d_EphiDatasPtr;
Standard_Real *m_h_d_EzrDatasPtr;

__device__ double atomicAdd_Double(double *address, double val)
{
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
        // NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

DEVICE void FIT_Mag_Func_Cyl3D::d_ComputeContourZR(Standard_Real &result, Standard_Real *EphiDatasPtr, Standard_Real *EzrDatasPtr)
{
    Standard_Real tmp[2], tmpNear[2];

    for (int i = 0; i < 2; ++i)
    {
        if (m_Elec_Ptr_Offset[i] == -1)
            tmp[i] = 0;
        else
            tmp[i] = EphiDatasPtr[m_Elec_Ptr_Offset[i]];

        if (m_ElecNear_Ptr_Offset[i] == -1)
            tmpNear[i] = 0;
        else
            tmpNear[i] = EzrDatasPtr[m_ElecNear_Ptr_Offset[i]];
    }

    result =
        tmp[0] * m_Curl[0] +
        tmp[1] * m_Curl[1] +
        tmpNear[0] * m_CurlNear[0] +
        tmpNear[1] * m_CurlNear[1];
}

DEVICE void FIT_Mag_Func_Cyl3D::d_ComputeContourPhi(Standard_Real &result, Standard_Real *EzrDatasPtr)
{
    Standard_Real tmp[4];

    for (int i = 0; i < 4; ++i)
    {
        if (m_Elec_Ptr_Offset[i] == -1)
            tmp[i] = 0;
        else
            tmp[i] = EzrDatasPtr[m_Elec_Ptr_Offset[i]];
    }

    result =
        tmp[0] * m_Curl[0] +
        tmp[1] * m_Curl[1] +
        tmp[2] * m_Curl[2] +
        tmp[3] * m_Curl[3];
}

DEVICE void FIT_Mag_Func_Cyl3D::d_AdvanceMzr(Standard_Real *MzrDatasPtr, Standard_Real *EphiDatasPtr, Standard_Real *EzrDatasPtr)
{
    Standard_Real contour = 0.0;
    d_ComputeContourZR(contour, EphiDatasPtr, EzrDatasPtr);

    MzrDatasPtr[m_Mag_Ptr_Offset] = MzrDatasPtr[m_Mag_Ptr_Offset] + m_C0 * (MzrDatasPtr[m_Current_Ptr_Offset] - contour);
}

DEVICE void FIT_Mag_Func_Cyl3D::d_AdvanceMphi(Standard_Real *MphiDatasPtr, Standard_Real *EzrDatasPtr)
{
    Standard_Real contour = 0.0;
    d_ComputeContourPhi(contour, EzrDatasPtr);

    MphiDatasPtr[m_Mag_Ptr_Offset] = MphiDatasPtr[m_Mag_Ptr_Offset] + m_C0 * (MphiDatasPtr[m_Current_Ptr_Offset] - contour);
}

DEVICE void FIT_Mag_PML_Func_Cyl3D::d_ComputContour1(Standard_Real &result, Standard_Real *ElecArray)
{
    Standard_Real tmp[4];
    for (int i = 0; i < 4; ++i)
    {
        if (m_Elec_Ptr_Offset[i] == -1)
            tmp[i] = 0;
        else
            tmp[i] = ElecArray[m_Elec_Ptr_Offset[i]];
    }

    result =
        tmp[0] * m_Curl1[0] +
        tmp[1] * m_Curl1[1] +
        tmp[2] * m_Curl1[2] +
        tmp[3] * m_Curl1[3];
}

DEVICE void FIT_Mag_PML_Func_Cyl3D::d_ComputContour2(Standard_Real &result, Standard_Real *ElecArray)
{
    Standard_Real tmp[4];
    for (int i = 0; i < 4; ++i)
    {
        if (m_Elec_Ptr_Offset[i] == -1)
            tmp[i] = 0;
        else
            tmp[i] = ElecArray[m_Elec_Ptr_Offset[i]];
    }

    result =
        tmp[0] * m_Curl2[0] +
        tmp[1] * m_Curl2[1] +
        tmp[2] * m_Curl2[2] +
        tmp[3] * m_Curl2[3];
}

DEVICE void FIT_Mag_PML_Func_Cyl3D::d_ComputContourl1(Standard_Real &result, Standard_Real *ElecArray)
{
    Standard_Real tmp[2];
    for (int i = 0; i < 2; ++i)
    {
        if (m_Elec_Ptr_Offset[i] == -1)
            tmp[i] = 0;
        else
            tmp[i] = ElecArray[m_Elec_Ptr_Offset[i]];
    }

    result =
        tmp[0] * m_Curl1[0] +
        tmp[1] * m_Curl1[1];
}

DEVICE void FIT_Mag_PML_Func_Cyl3D::d_ComputContour1Near(Standard_Real &result, Standard_Real *ElecArray)
{
    Standard_Real tmp[2];
    for (int i = 0; i < 2; ++i)
    {
        if (m_ElecNear_Ptr_Offset[i] == -1)
            tmp[i] = 0;
        else
            tmp[i] = ElecArray[m_ElecNear_Ptr_Offset[i]];
    }

    result =
        tmp[0] * m_Curl1Near[0] +
        tmp[1] * m_Curl1Near[1];
}

DEVICE void FIT_Mag_PML_Func_Cyl3D::d_ComputeP(const Standard_Real &contour1,
                                               const Standard_Real &contour2, Standard_Real *MagArray)
{
    MagArray[m_PM1_Ptr_Offset] = b[0] * MagArray[m_PM1_Ptr_Offset] + a[0] * contour1;
    MagArray[m_PM2_Ptr_Offset] = b[1] * MagArray[m_PM2_Ptr_Offset] + a[1] * contour2;
}

DEVICE void FIT_Mag_PML_Func_Cyl3D::d_ComputeP(const Standard_Real &contour1, Standard_Real *MagArray)
{
    MagArray[m_PM1_Ptr_Offset] = b[0] * MagArray[m_PM1_Ptr_Offset] + a[0] * contour1;
}

DEVICE void FIT_Mag_PML_Func_Cyl3D::d_AdvanceMzr(Standard_Real *MzrDatasPtr, Standard_Real *EphiDatasPtr,
                                                 Standard_Real *EzrDatasPtr)
{
    Standard_Real contour1, contour1Near;
    d_ComputContourl1(contour1, EphiDatasPtr);
    d_ComputContour1Near(contour1Near, EzrDatasPtr);
    d_ComputeP(contour1, MzrDatasPtr);

    MzrDatasPtr[m_Mag_Ptr_Offset] = MzrDatasPtr[m_Mag_Ptr_Offset] - m_CurlP1 * contour1 -
                                    m_C0 * contour1Near - m_C0 * MzrDatasPtr[m_PM1_Ptr_Offset];
}

DEVICE void FIT_Mag_PML_Func_Cyl3D::d_AdvanceMphi(Standard_Real *MphiDatasPtr, Standard_Real *EzrDatasPtr)
{
    Standard_Real contour1, contour2;
    d_ComputContour1(contour1, EzrDatasPtr);
    d_ComputContour2(contour2, EzrDatasPtr);
    d_ComputeP(contour1, contour2, MphiDatasPtr);

    MphiDatasPtr[m_Mag_Ptr_Offset] = MphiDatasPtr[m_Mag_Ptr_Offset] - m_CurlP1 * contour1 -
                                     m_CurlP2 * contour2 - m_C0 * MphiDatasPtr[m_PM1_Ptr_Offset] -
                                     m_C0 * MphiDatasPtr[m_PM2_Ptr_Offset];
}

DEVICE void FIT_Elec_Func_Cyl3D::d_Advance_Damping_1(const Standard_Real theta, Standard_Real *ElecArray)
{
    ElecArray[m_AE_Ptr_Offset] = (1.0 - theta) * ElecArray[m_Elec_Ptr_Offset] +
                                 theta * ElecArray[m_AE_Ptr_Offset];
    ElecArray[m_Elec_PreStep_Ptr_Offset] = ElecArray[m_Elec_Ptr_Offset];
}

DEVICE void FIT_Elec_Func_Cyl3D::d_Advance_Damping_2(const Standard_Real theta, Standard_Real *ElecArray)
{
    ElecArray[m_BE_Ptr_Offset] = (1.0 + 0.5 * theta) * ElecArray[m_Elec_Ptr_Offset] -
                                 0.5 * ElecArray[m_Elec_PreStep_Ptr_Offset] +
                                 0.5 * (1.0 - theta) * ElecArray[m_AE_Ptr_Offset];
}

DEVICE void FIT_Elec_Func_Cyl3D::d_ComputeContourZR(Standard_Real &result, Standard_Real *MagArray1,
                                                    Standard_Real *MagArray2)
{
    Standard_Real tmp[2], tmpNear[2];

    for (int i = 0; i < 2; ++i)
    {
        if (m_Mag_Ptr_Offset[i] == -1)
            tmp[i] = 0;
        else
            tmp[i] = MagArray1[m_Mag_Ptr_Offset[i]];

        if (m_MagNear_Ptr_Offset[i] == -1)
            tmpNear[i] = 0;
        else
            tmpNear[i] = MagArray2[m_MagNear_Ptr_Offset[i]];
    }

    result =
        tmp[0] * m_Curl[0] +
        tmp[1] * m_Curl[1] +
        tmpNear[0] * m_CurlNear[0] +
        tmpNear[1] * m_CurlNear[1];
}

DEVICE void FIT_Elec_Func_Cyl3D::d_AdvanceEzr(Standard_Real *EzrDatasPtr, Standard_Real *MphiDatasPtr,
                                              Standard_Real *MzrDatasPtr)
{
    Standard_Real contour = 0.0;
    d_ComputeContourZR(contour, MphiDatasPtr, MzrDatasPtr);

    EzrDatasPtr[m_Elec_Ptr_Offset] = m_C0 * EzrDatasPtr[m_Elec_Ptr_Offset] +
                                     m_C2 * contour - m_C3 * EzrDatasPtr[m_Current_Ptr_Offset];
}

DEVICE void FIT_Elec_Func_Cyl3D::d_AdvanceEzr_Damping(const Standard_Real theta, Standard_Real *EzrDatasPtr,
                                                      Standard_Real *MphiDatasPtr, Standard_Real *MzrDatasPtr)
{
    d_Advance_Damping_1(theta, EzrDatasPtr);

    d_AdvanceEzr(EzrDatasPtr, MphiDatasPtr, MzrDatasPtr);

    d_Advance_Damping_2(theta, EzrDatasPtr);
}

DEVICE void FIT_Elec_Func_Cyl3D::d_ComputeContourPhi(Standard_Real &result, Standard_Real *MagArray)
{
    Standard_Real tmp[4];

    for (int i = 0; i < 4; ++i)
    {
        if (m_Mag_Ptr_Offset[i] == -1)
            tmp[i] = 0;
        else
            tmp[i] = MagArray[m_Mag_Ptr_Offset[i]];
    }

    result =
        tmp[0] * m_Curl[0] +
        tmp[1] * m_Curl[1] +
        tmp[2] * m_Curl[2] +
        tmp[3] * m_Curl[3];
}

DEVICE Standard_Real FIT_Elec_Func_Cyl3D::d_ComputeContourAxis(Standard_Real *MagArray)
{
    Standard_Real res;
    Standard_Real tmp[2];

    for (int i = 0; i < 2; ++i)
    {
        if (m_Mag_Ptr_Offset[i] == -1)
            tmp[i] = 0;
        else
            tmp[i] = MagArray[m_Mag_Ptr_Offset[i]];
    }

    res =
        tmp[0] * m_Curl[0] +
        tmp[1] * m_Curl[1];

    return res;
}

DEVICE void FIT_Elec_Func_Cyl3D::d_AdvanceEphi(Standard_Real *EphiDatasPtr, Standard_Real *MzrDatasPtr)
{
    Standard_Real contour = 0.0;
    d_ComputeContourPhi(contour, MzrDatasPtr);

    EphiDatasPtr[m_Elec_Ptr_Offset] = m_C0 * EphiDatasPtr[m_Elec_Ptr_Offset] + m_C2 * contour -
                                      m_C3 * EphiDatasPtr[m_Current_Ptr_Offset];
}

DEVICE void FIT_Elec_Func_Cyl3D::d_AdvanceEphi_Damping(const Standard_Real theta, Standard_Real *EphiDatasPtr,
                                                       Standard_Real *MzrDatasPtr)
{
    d_Advance_Damping_1(theta, EphiDatasPtr);

    d_AdvanceEphi(EphiDatasPtr, MzrDatasPtr);

    d_Advance_Damping_2(theta, EphiDatasPtr);
}

DEVICE void FIT_Elec_PML_Func_Cyl3D::d_Advance_Damping_1(const Standard_Real theta, Standard_Real *ElecArray)
{
    ElecArray[m_AE_Ptr_Offset] = (1.0 - theta) * ElecArray[m_Elec_Ptr_Offset] +
                                 theta * ElecArray[m_AE_Ptr_Offset];
    ElecArray[m_Elec_PreStep_Ptr_Offset] = ElecArray[m_Elec_Ptr_Offset];
}

DEVICE void FIT_Elec_PML_Func_Cyl3D::d_Advance_Damping_2(const Standard_Real theta, Standard_Real *ElecArray)
{
    ElecArray[m_BE_Ptr_Offset] = (1.0 + 0.5 * theta) * ElecArray[m_Elec_Ptr_Offset] -
                                 0.5 * ElecArray[m_Elec_PreStep_Ptr_Offset] +
                                 0.5 * (1.0 - theta) * ElecArray[m_AE_Ptr_Offset];
}

DEVICE void FIT_Elec_PML_Func_Cyl3D::d_ComputContour11(Standard_Real &result, Standard_Real *MagArray)
{
    Standard_Real tmp[2];

    for (int i = 0; i < 2; ++i)
    {
        if (m_Mag_Ptr_Offset[i] == -1)
            tmp[i] = 0;
        else
            tmp[i] = MagArray[m_Mag_Ptr_Offset[i]];
    }

    result =
        tmp[0] * m_Curl1[0] +
        tmp[1] * m_Curl1[1];
}

DEVICE void FIT_Elec_PML_Func_Cyl3D::d_ComputP1(const Standard_Real contour1, Standard_Real *ElecArray)
{
    ElecArray[m_PE1_Ptr_Offset] = b[0] * ElecArray[m_PE1_Ptr_Offset] +
                                  a[0] * contour1;
}

DEVICE void FIT_Elec_PML_Func_Cyl3D::d_ComputContour1Near(Standard_Real &result, Standard_Real *MagArray)
{
    Standard_Real tmp[2];

    for (int i = 0; i < 2; ++i)
    {
        if (m_MagNear_Ptr_Offset[i] == -1)
            tmp[i] = 0;
        else
            tmp[i] = MagArray[m_MagNear_Ptr_Offset[i]];
    }

    result =
        tmp[0] * m_Curl1Near[0] +
        tmp[1] * m_Curl1Near[1];
}

DEVICE void FIT_Elec_PML_Func_Cyl3D::d_AdvanceEzr(Standard_Real *EzrDatasPtr, Standard_Real *MphiDatasPtr,
                                                  Standard_Real *MzrDatasPtr)
{
    Standard_Real contour1 = 0.0, contour1Near = 0.0;
    d_ComputContour11(contour1, MphiDatasPtr);
    d_ComputP1(contour1, EzrDatasPtr);
    d_ComputContour1Near(contour1Near, MzrDatasPtr);
    EzrDatasPtr[m_Elec_Ptr_Offset] = m_C0 * EzrDatasPtr[m_Elec_Ptr_Offset] +
                                     contour1 * m_CurlP1 +
                                     contour1Near * m_C2 +
                                     m_C2 * EzrDatasPtr[m_PE1_Ptr_Offset];
}

DEVICE void FIT_Elec_PML_Func_Cyl3D::d_AdvanceEzr_Damping(const Standard_Real theta, Standard_Real *EzrDatasPtr,
                                                          Standard_Real *MphiDatasPtr, Standard_Real *MzrDatasPtr)
{
    d_Advance_Damping_1(theta, EzrDatasPtr);

    d_AdvanceEzr(EzrDatasPtr, MphiDatasPtr, MzrDatasPtr);

    d_Advance_Damping_2(theta, EzrDatasPtr);
}

DEVICE void FIT_Elec_PML_Func_Cyl3D::d_ComputContour1(Standard_Real &result, Standard_Real *MagArray)
{
    Standard_Real tmp[4];

    for (int i = 0; i < 4; ++i)
    {
        if (m_Mag_Ptr_Offset[i] == -1)
            tmp[i] = 0;
        else
            tmp[i] = MagArray[m_Mag_Ptr_Offset[i]];
    }

    result =
        tmp[0] * m_Curl1[0] +
        tmp[1] * m_Curl1[1] +
        tmp[2] * m_Curl1[2] +
        tmp[3] * m_Curl1[3];
}

DEVICE void FIT_Elec_PML_Func_Cyl3D::d_ComputContour2(Standard_Real &result, Standard_Real *MagArray)
{
    Standard_Real tmp[4];

    for (int i = 0; i < 4; ++i)
    {
        if (m_Mag_Ptr_Offset[i] == -1)
            tmp[i] = 0;
        else
            tmp[i] = MagArray[m_Mag_Ptr_Offset[i]];
    }

    result =
        tmp[0] * m_Curl2[0] +
        tmp[1] * m_Curl2[1] +
        tmp[2] * m_Curl2[2] +
        tmp[3] * m_Curl2[3];
}

DEVICE void FIT_Elec_PML_Func_Cyl3D::d_ComputP2(const Standard_Real contour2, Standard_Real *ElecArray)
{
    ElecArray[m_PE2_Ptr_Offset] = b[1] * ElecArray[m_PE2_Ptr_Offset] +
                                  a[1] * contour2;
}

DEVICE void FIT_Elec_PML_Func_Cyl3D::d_AdvanceEphi(Standard_Real *EphiDatasPtr, Standard_Real *MzrDatasPtr)
{
    Standard_Real contour1 = 0.0, contour2 = 0.0;
    d_ComputContour1(contour1, MzrDatasPtr);
    d_ComputContour2(contour2, MzrDatasPtr);
    d_ComputP1(contour1, EphiDatasPtr);
    d_ComputP2(contour2, EphiDatasPtr);
    EphiDatasPtr[m_Elec_Ptr_Offset] = m_C0 * EphiDatasPtr[m_Elec_Ptr_Offset] +
                                      contour1 * m_CurlP1 +
                                      contour2 * m_CurlP2 +
                                      m_C2 * EphiDatasPtr[m_PE1_Ptr_Offset] +
                                      m_C2 * EphiDatasPtr[m_PE2_Ptr_Offset];
}

DEVICE void FIT_Elec_PML_Func_Cyl3D::d_AdvanceEphi_Damping(const Standard_Real theta, Standard_Real *EphiDatasPtr,
                                                           Standard_Real *MzrDatasPtr)
{
    d_Advance_Damping_1(theta, EphiDatasPtr);

    d_AdvanceEphi(EphiDatasPtr, MzrDatasPtr);

    d_Advance_Damping_2(theta, EphiDatasPtr);
}

DEVICE Standard_Real FIT_Elec_PML_Func_Cyl3D::d_ComputContourAxis(Standard_Real *MagArray)
{
    Standard_Real res;
    Standard_Real tmp[2];

    for (int i = 0; i < 2; ++i)
    {
        if (m_Mag_Ptr_Offset[i] == -1)
            tmp[i] = 0;
        else
            tmp[i] = MagArray[m_Mag_Ptr_Offset[i]];
    }

    res =
        tmp[0] * m_Curl1[0] +
        tmp[1] * m_Curl1[1];

    return res;
}

DEVICE void FIT_Elec_Mur_Func_Cyl3D::d_AdvanceEzr_Damping(const Standard_Real theta, Standard_Real *EzrDatasPtr)
{
    EzrDatasPtr[m_AE_Ptr_Offset] = (1.0 - theta) * EzrDatasPtr[m_Elec_Ptr_Offset] + theta * EzrDatasPtr[m_AE_Ptr_Offset];
    Standard_Real m_Elec_PreStep_tmp = EzrDatasPtr[m_Elec_Ptr_Offset];

    Standard_Real vbar = m_VBar * 1.0;
    Standard_Real currE = EzrDatasPtr[m_PreTStepEFld_Ptr_Offset];
    Standard_Real oldE = EzrDatasPtr[m_PreTStep_Ptr_Offset];
    Standard_Real tmpE = (oldE - currE * (1.0 - vbar)) / (1.0 + vbar);
    oldE = tmpE * (1.0 - vbar) + currE * (1.0 + vbar);
    EzrDatasPtr[m_PreTStep_Ptr_Offset] = oldE;
    EzrDatasPtr[m_Elec_Ptr_Offset] = tmpE;

    EzrDatasPtr[m_BE_Ptr_Offset] = (1.0 + 0.5 * theta) * EzrDatasPtr[m_Elec_Ptr_Offset] -
                                   0.5 * m_Elec_PreStep_tmp +
                                   0.5 * (1.0 - theta) * EzrDatasPtr[m_AE_Ptr_Offset];
}

DEVICE void FIT_Elec_Mur_Func_Cyl3D::d_AdvanceEzr_Damping_TFunc(const Standard_Real theta, Standard_Real *EzrDatasPtr, Standard_Real Ebar, Standard_Real Ebar2)
{
    EzrDatasPtr[m_AE_Ptr_Offset] = (1.0 - theta) * EzrDatasPtr[m_Elec_Ptr_Offset] + theta * EzrDatasPtr[m_AE_Ptr_Offset];
    Standard_Real m_Elec_PreStep_tmp = EzrDatasPtr[m_Elec_Ptr_Offset];

    Standard_Real vbar = m_VBar * 1.0;
    Standard_Real currE = EzrDatasPtr[m_PreTStepEFld_Ptr_Offset];
    Standard_Real oldE = EzrDatasPtr[m_PreTStep_Ptr_Offset];
    Standard_Real tmpE = (oldE - (currE - Ebar2) * (1.0 - vbar)) / (1.0 + vbar) + Ebar;
    oldE = (tmpE - Ebar) * (1.0 - vbar) + (currE - Ebar2) * (1.0 + vbar);
    EzrDatasPtr[m_PreTStep_Ptr_Offset] = oldE;
    EzrDatasPtr[m_Elec_Ptr_Offset] = tmpE;

    EzrDatasPtr[m_BE_Ptr_Offset] = (1.0 + 0.5 * theta) * EzrDatasPtr[m_Elec_Ptr_Offset] -
                                   0.5 * m_Elec_PreStep_tmp +
                                   0.5 * (1.0 - theta) * EzrDatasPtr[m_AE_Ptr_Offset];
}

DEVICE void FIT_Elec_Mur_Func_Cyl3D::d_AdvanceEphi_Damping(const Standard_Real theta, Standard_Real *EphiDatasPtr)
{
    EphiDatasPtr[m_AE_Ptr_Offset] = (1.0 - theta) * EphiDatasPtr[m_Elec_Ptr_Offset] + theta * EphiDatasPtr[m_AE_Ptr_Offset];
    Standard_Real m_Elec_PreStep_tmp = EphiDatasPtr[m_Elec_Ptr_Offset];

    Standard_Real vbar = m_VBar * 1.0;
    Standard_Real currE = EphiDatasPtr[m_PreTStepEFld_Ptr_Offset];
    Standard_Real oldE = EphiDatasPtr[m_PreTStep_Ptr_Offset];
    Standard_Real tmpE = (oldE - currE * (1.0 - vbar)) / (1.0 + vbar);
    oldE = tmpE * (1.0 - vbar) + currE * (1.0 + vbar);
    EphiDatasPtr[m_PreTStep_Ptr_Offset] = oldE;
    EphiDatasPtr[m_Elec_Ptr_Offset] = tmpE;

    EphiDatasPtr[m_BE_Ptr_Offset] = (1.0 + 0.5 * theta) * EphiDatasPtr[m_Elec_Ptr_Offset] -
                                    0.5 * m_Elec_PreStep_tmp +
                                    0.5 * (1.0 - theta) * EphiDatasPtr[m_AE_Ptr_Offset];
}

void SI_SC_Matrix_EMFields_Cyl3D::
    Advance()
{
    m_FldSrcs_Cyl3D->Advance();
    m_FldSrcs_Cyl3D->Advance_SI_MJ(1.0);
    m_FldSrcs_Cyl3D->Advance_SI_Mag_0(1.0);

    AdvanceMagCntr(); // 待补充 // 待修改
    AdvanceMagCPML(); // 待补充 // 待修改

    m_FldSrcs_Cyl3D->Advance_SI_Mag_1(1.0);

    m_FldSrcs_Cyl3D->Advance_SI_J(1.0);
    m_FldSrcs_Cyl3D->Advance_SI_Elec_0(1.0);

    AdvanceElecCntr(); // 待补充 // 待修改
    AdvanceElecCPML(); // 待补充 // 待修改

    m_EMurFields_Cyl3D->Advance_SI(1.0); // 待补充 // 待修改
    m_FldSrcs_Cyl3D->Advance_SI_Elec_1(1.0);

    m_MCntrFields_Cyl3D->Advance();
    m_MCPMLFields_Cyl3D->Advance();
    m_ECntrFields_Cyl3D->Advance();
    m_ECPMLFields_Cyl3D->Advance();

    DynObj::Advance();
}

// 核函数声明
__global__ void g_AdvanceMagCntr_WithDamping_m_Mphi(FIT_Mag_Func_Cyl3D *MphiFuncArray, Standard_Size MphiDatasNum,
                                                    Standard_Real *MphiDatasPtr, Standard_Real *EzrDatasPtr)
{
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID >= MphiDatasNum)
        return;

    MphiFuncArray[threadID].d_AdvanceMphi(MphiDatasPtr, EzrDatasPtr);
}

__global__ void g_AdvanceMagCntr_WithDamping_m_Mzr(FIT_Mag_Func_Cyl3D *MzrFuncArray, Standard_Size MzrDatasNum,
                                                   Standard_Real *MzrDatasPtr, Standard_Real *EphiDatasPtr, Standard_Real *EzrDatasPtr)
{
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID >= MzrDatasNum)
        return;

    MzrFuncArray[threadID].d_AdvanceMzr(MzrDatasPtr, EphiDatasPtr, EzrDatasPtr);
}

__global__ void g_AdvanceMagCPML_WithDamping_m_Mphi(FIT_Mag_PML_Func_Cyl3D *MphiFuncArray, Standard_Size MphiDatasNum,
                                                    Standard_Real *MphiDatasPtr, Standard_Real *EzrDatasPtr)
{
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID >= MphiDatasNum)
        return;

    MphiFuncArray[threadID].d_AdvanceMphi(MphiDatasPtr, EzrDatasPtr);
}

__global__ void g_AdvanceMagCPML_WithDamping_m_Mzr(FIT_Mag_PML_Func_Cyl3D *MzrFuncArray, Standard_Size MzrDatasNum,
                                                   Standard_Real *MzrDatasPtr, Standard_Real *EphiDatasPtr, Standard_Real *EzrDatasPtr)
{
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID >= MzrDatasNum)
        return;

    MzrFuncArray[threadID].d_AdvanceMzr(MzrDatasPtr, EphiDatasPtr, EzrDatasPtr);
}

__global__ void g_AdvanceElecMur_WithDamping_m_Ezr(Standard_Real *theta, FIT_Elec_Mur_Func_Cyl3D *EzrFuncArray, Standard_Size EzrDatasNum,
                                                   Standard_Real *EzrDatasPtr)
{
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID >= EzrDatasNum)
        return;

    EzrFuncArray[threadID].d_AdvanceEzr_Damping(theta[0], EzrDatasPtr);
}

__global__ void g_AdvanceElecMur_WithDamping_m_Ephi(Standard_Real *theta, FIT_Elec_Mur_Func_Cyl3D *EphiFuncArray, Standard_Size EphiDatasNum,
                                                    Standard_Real *EphiDatasPtr)
{
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID >= EphiDatasNum)
        return;

    EphiFuncArray[threadID].d_AdvanceEphi_Damping(theta[0], EphiDatasPtr);
}

__global__ void g_AdvanceElecMurVoltage_WithDamping_m_Ezr(Standard_Real *theta, FIT_Elec_Mur_Func_Cyl3D *EzrFuncArray, Standard_Size EzrDatasNum,
                                                          Standard_Real *EzrDatasPtr, Standard_Real *Ebar, Standard_Real *Ebar2)
{
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID >= EzrDatasNum)
        return;

    EzrFuncArray[threadID].d_AdvanceEzr_Damping_TFunc(theta[0], EzrDatasPtr, Ebar[threadID], Ebar2[threadID]);
}

__global__ void g_AdvanceElecMurVoltage_WithDamping_m_Ephi(Standard_Real *theta, FIT_Elec_Mur_Func_Cyl3D *EphiFuncArray, Standard_Size EphiDatasNum,
                                                           Standard_Real *EphiDatasPtr)
{
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID >= EphiDatasNum)
        return;

    EphiFuncArray[threadID].d_AdvanceEphi_Damping(theta[0], EphiDatasPtr);
}

__global__ void g_AdvanceElecCntr_WithDamping_m_Ephi(Standard_Real *theta, FIT_Elec_Func_Cyl3D *EphiFuncArray, Standard_Size EphiDatasNum,
                                                     Standard_Real *EphiDatasPtr, Standard_Real *MzrDatasPtr)
{
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID >= EphiDatasNum)
        return;

    EphiFuncArray[threadID].d_AdvanceEphi_Damping(theta[0], EphiDatasPtr, MzrDatasPtr);
}

__global__ void g_AdvanceElecCntr_WithDamping_m_Ezr(Standard_Real *theta, FIT_Elec_Func_Cyl3D *EzrFuncArray, Standard_Size EzrDatasNum,
                                                    Standard_Real *EzrDatasPtr, Standard_Real *MphiDatasPtr, Standard_Real *MzrDatasPtr)
{
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID >= EzrDatasNum)
        return;

    EzrFuncArray[threadID].d_AdvanceEzr_Damping(theta[0], EzrDatasPtr, MphiDatasPtr, MzrDatasPtr);
}

// S3
__global__ void g_AdvanceElecCntr_WithDamping_m_EAxis(Standard_Integer *Parameters, FIT_Elec_Func_Cyl3D *EAxisFuncArray, Standard_Size EAxisDatasNum,
                                                      Standard_Real *EzrDatasPtr, Standard_Real *MphiDatasPtr, Standard_Real *EAxisDualContourValue, Standard_Real *EAxisPhysDataJ, Standard_Real *EAxisPhysDataE)
{
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID >= EAxisDatasNum)
        return;

    atomicAdd_Double(&(EAxisDualContourValue[threadID % Parameters[1]]), EAxisFuncArray[threadID].d_ComputeContourAxis(MphiDatasPtr));
    atomicAdd_Double(&(EAxisPhysDataJ[threadID % Parameters[1]]), EzrDatasPtr[EAxisFuncArray[threadID].m_Current_Ptr_Offset]);
    atomicAdd_Double(&(EAxisPhysDataE[threadID % Parameters[1]]), EzrDatasPtr[EAxisFuncArray[threadID].m_Elec_Ptr_Offset]);
}

// S3
__global__ void g_AdvanceElecCntr_WithDamping_m_EAxis_1(Standard_Integer *Parameters, Standard_Real *EAxisDualContourValue, Standard_Real *EAxisPhysDataJ, Standard_Real *EAxisPhysDataE, Standard_Size EAxisDatasNum)
{
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID >= EAxisDatasNum)
        return;

    // if(threadID == 1){
    //     printf("now let's print EAxisDualContourValue: %f \n", EAxisDualContourValue[threadID]);
    //     printf("now let's print EAxisPhysDataJ: %f \n", EAxisPhysDataJ[threadID]);
    //     printf("now let's print EAxisPhysDataE: %f \n\n", EAxisPhysDataE[threadID]);
    // }

    EAxisPhysDataJ[threadID] /= Parameters[0];
    EAxisPhysDataE[threadID] /= Parameters[0];
}

// S3
__global__ void g_AdvanceElecCntr_WithDamping_m_EAxis_2(Standard_Integer *Parameters, FIT_Elec_Func_Cyl3D *EAxisFuncArray, Standard_Size EAxisDatasNum,
                                                        Standard_Real *EAxisDualContourValue, Standard_Real *EAxisPhysDataJ, Standard_Real *EAxisPhysDataE)
{
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID >= EAxisDatasNum)
        return;

    EAxisFuncArray[threadID].m_current = EAxisPhysDataJ[threadID % Parameters[1]];
    EAxisFuncArray[threadID].m_elec = EAxisPhysDataE[threadID % Parameters[1]];
    EAxisFuncArray[threadID].m_DualContour = EAxisDualContourValue[threadID % Parameters[1]];

    // if(threadID == 0){
    //     printf("device_EAxisDualContour = %f\n", EAxisFuncArray[threadID].m_DualContour);
    //     printf("device_EAxisPhysDataJ = %f\n", EAxisFuncArray[threadID].m_current);
    //     printf("device_EAxisPhysDataE = %f\n\n", EAxisFuncArray[threadID].m_elec);
    // }
}

// S3
__global__ void g_AdvanceElecCntr_WithDamping_m_EAxis_3(Standard_Real *EAxisDualContourValue, Standard_Real *EAxisPhysDataJ, Standard_Real *EAxisPhysDataE, Standard_Size EAxisDatasNum)
{
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID >= EAxisDatasNum)
        return;

    EAxisDualContourValue[threadID] = 0.0;
    EAxisPhysDataJ[threadID] = 0.0;
    EAxisPhysDataE[threadID] = 0.0;
}

// S3
__global__ void g_AdvanceElecCntr_WithDamping_m_EAxis_end(Standard_Real *theta, FIT_Elec_Func_Cyl3D *EAxisFuncArray, Standard_Size EAxisDatasNum, Standard_Real *EzrDatasPtr)
{
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID >= EAxisDatasNum)
        return;

    EzrDatasPtr[EAxisFuncArray[threadID].m_AE_Ptr_Offset] = (1.0 - theta[0]) * EzrDatasPtr[EAxisFuncArray[threadID].m_Elec_Ptr_Offset] + theta[0] * EzrDatasPtr[EAxisFuncArray[threadID].m_AE_Ptr_Offset];
    EzrDatasPtr[EAxisFuncArray[threadID].m_Elec_PreStep_Ptr_Offset] = EzrDatasPtr[EAxisFuncArray[threadID].m_Elec_Ptr_Offset];

    EzrDatasPtr[EAxisFuncArray[threadID].m_Elec_Ptr_Offset] = EAxisFuncArray[threadID].m_C0 * EAxisFuncArray[threadID].m_elec +
                                                              EAxisFuncArray[threadID].m_C2 * (EAxisFuncArray[threadID].m_DualContour - EAxisFuncArray[threadID].m_current) * EAxisFuncArray[threadID].m_C3;

    EzrDatasPtr[EAxisFuncArray[threadID].m_BE_Ptr_Offset] = (1.0 + 0.5 * theta[0]) * EzrDatasPtr[EAxisFuncArray[threadID].m_Elec_Ptr_Offset] - 0.5 * EzrDatasPtr[EAxisFuncArray[threadID].m_Elec_PreStep_Ptr_Offset] + 0.5 * (1.0 - theta[0]) * EzrDatasPtr[EAxisFuncArray[threadID].m_AE_Ptr_Offset];
}

// S3
__global__ void g_AdvanceElecCPML_WithDamping_m_Ezr(Standard_Real *theta, FIT_Elec_PML_Func_Cyl3D *EzrFuncArray, Standard_Size EzrDatasNum,
                                                    Standard_Real *EzrDatasPtr, Standard_Real *MphiDatasPtr, Standard_Real *MzrDatasPtr)
{
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID >= EzrDatasNum)
        return;

    EzrFuncArray[threadID].d_AdvanceEzr_Damping(theta[0], EzrDatasPtr, MphiDatasPtr, MzrDatasPtr);
}

// S3
__global__ void g_AdvanceElecCPML_WithDamping_m_Ephi(Standard_Real *theta, FIT_Elec_PML_Func_Cyl3D *EphiFuncArray, Standard_Size EphiDatasNum,
                                                     Standard_Real *EphiDatasPtr, Standard_Real *MzrDatasPtr)
{
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID >= EphiDatasNum)
        return;

    EphiFuncArray[threadID].d_AdvanceEphi_Damping(theta[0], EphiDatasPtr, MzrDatasPtr);
}

__global__ void g_AdvanceElecCPML_WithDamping_m_EAxis(Standard_Integer *Parameters, FIT_Elec_PML_Func_Cyl3D *EAxisPMLFuncArray, Standard_Size EAxisDatasNum,
                                                      Standard_Real *EzrDatasPtr, Standard_Real *MphiDatasPtr, Standard_Real *EAxisDualContourValue, Standard_Real *EAxisPhysDataPE1, Standard_Real *EAxisPhysDataE)
{
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID >= EAxisDatasNum)
        return;

    atomicAdd_Double(&(EAxisDualContourValue[threadID % Parameters[2]]), EAxisPMLFuncArray[threadID].d_ComputContourAxis(MphiDatasPtr));
    atomicAdd_Double(&(EAxisPhysDataPE1[threadID % Parameters[2]]), EzrDatasPtr[EAxisPMLFuncArray[threadID].m_PE1_Ptr_Offset]);
    atomicAdd_Double(&(EAxisPhysDataE[threadID % Parameters[2]]), EzrDatasPtr[EAxisPMLFuncArray[threadID].m_Elec_Ptr_Offset]);
}

__global__ void g_AdvanceElecCPML_WithDamping_m_EAxis_1(Standard_Integer *Parameters, Standard_Real *EAxisPhysDataPE1, Standard_Real *EAxisPhysDataE, Standard_Size EAxisDatasNum)
{
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID >= EAxisDatasNum)
        return;

    EAxisPhysDataPE1[threadID] /= Parameters[0];
    EAxisPhysDataE[threadID] /= Parameters[0];
}

__global__ void g_AdvanceElecCPML_WithDamping_m_EAxis_2(Standard_Integer *Parameters, FIT_Elec_PML_Func_Cyl3D *EAxisPMLFuncArray, Standard_Size EAxisDatasNum,
                                                        Standard_Real *EAxisDualContourValue, Standard_Real *EAxisPhysDataPE1, Standard_Real *EAxisPhysDataE)
{
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID >= EAxisDatasNum)
        return;

    EAxisPMLFuncArray[threadID].m_pe1 = EAxisPhysDataPE1[threadID % Parameters[2]];
    EAxisPMLFuncArray[threadID].m_elec = EAxisPhysDataE[threadID % Parameters[2]];
    EAxisPMLFuncArray[threadID].m_DualContour = EAxisDualContourValue[threadID % Parameters[2]];
}

__global__ void g_AdvanceElecCPML_WithDamping_m_EAxis_3(Standard_Real *EAxisDualContourValue, Standard_Real *EAxisPhysDataPE1, Standard_Real *EAxisPhysDataE, Standard_Size EAxisDatasNum)
{
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID >= EAxisDatasNum)
        return;

    EAxisDualContourValue[threadID] = 0.0;
    EAxisPhysDataPE1[threadID] = 0.0;
    EAxisPhysDataE[threadID] = 0.0;
}

__global__ void g_AdvanceElecCPML_WithDamping_m_EAxis_end(Standard_Real *theta, FIT_Elec_PML_Func_Cyl3D *EAxisPMLFuncArray, Standard_Size EAxisDatasNum, Standard_Real *EzrDatasPtr)
{
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID >= EAxisDatasNum)
        return;

    EzrDatasPtr[EAxisPMLFuncArray[threadID].m_AE_Ptr_Offset] = (1.0 - theta[0]) * EzrDatasPtr[EAxisPMLFuncArray[threadID].m_Elec_Ptr_Offset] +
                                                               theta[0] * EzrDatasPtr[EAxisPMLFuncArray[threadID].m_AE_Ptr_Offset];
    EzrDatasPtr[EAxisPMLFuncArray[threadID].m_Elec_PreStep_Ptr_Offset] = EzrDatasPtr[EAxisPMLFuncArray[threadID].m_Elec_Ptr_Offset];

    EzrDatasPtr[EAxisPMLFuncArray[threadID].m_PE1_Ptr_Offset] = EAxisPMLFuncArray[threadID].b[0] * EAxisPMLFuncArray[threadID].m_pe1 +
                                                                EAxisPMLFuncArray[threadID].a[0] * EAxisPMLFuncArray[threadID].m_DualContour * EAxisPMLFuncArray[threadID].m_Contour1;

    EzrDatasPtr[EAxisPMLFuncArray[threadID].m_Elec_Ptr_Offset] = EAxisPMLFuncArray[threadID].m_C0 * EAxisPMLFuncArray[threadID].m_elec +
                                                                 EAxisPMLFuncArray[threadID].m_C2 * EAxisPMLFuncArray[threadID].invKappa[0] * EAxisPMLFuncArray[threadID].m_DualContour *
                                                                     EAxisPMLFuncArray[threadID].m_Contour +
                                                                 EAxisPMLFuncArray[threadID].m_C2 * EAxisPMLFuncArray[threadID].m_pe1;

    EzrDatasPtr[EAxisPMLFuncArray[threadID].m_BE_Ptr_Offset] = (1.0 + 0.5 * theta[0]) * EzrDatasPtr[EAxisPMLFuncArray[threadID].m_Elec_Ptr_Offset] -
                                                               0.5 * EzrDatasPtr[EAxisPMLFuncArray[threadID].m_Elec_PreStep_Ptr_Offset] + 0.5 * (1.0 - theta[0]) * EzrDatasPtr[EAxisPMLFuncArray[threadID].m_AE_Ptr_Offset];
}

void SI_SC_Matrix_EMFields_Cyl3D::
    BuildCUDADatas()
{
    bytesMphiData = sizeof(Standard_Real) * m_MphiDatasSize;
    bytesMzrData = sizeof(Standard_Real) * m_MzrDatasSize;
    bytesEphiData = sizeof(Standard_Real) * m_EphiDatasSize;
    bytesEzrData = sizeof(Standard_Real) * m_EzrDatasSize;
    bytesdamping_scale = sizeof(Standard_Real) * 2;
    bytesParameters = sizeof(Standard_Integer) * 4;
    bytesEAxisData = sizeof(Standard_Real) * m_EAxisDatasNum / phi_num;
    bytesCPMLEAxisData = sizeof(Standard_Real) * m_CPML_EAxisDatasNum / phi_num;

    checkCudaErrors(cudaMallocManaged((void **)&m_h_d_MphiDatasPtr, bytesMphiData));
    checkCudaErrors(cudaMallocManaged((void **)&m_h_d_MzrDatasPtr, bytesMzrData));
    checkCudaErrors(cudaMallocManaged((void **)&m_h_d_EphiDatasPtr, bytesEphiData));
    checkCudaErrors(cudaMallocManaged((void **)&m_h_d_EzrDatasPtr, bytesEzrData));
    checkCudaErrors(cudaMallocManaged((void **)&m_h_d_damping_scale, bytesdamping_scale));
    checkCudaErrors(cudaMallocManaged((void **)&m_h_d_Parameters, bytesParameters));
    checkCudaErrors(cudaMallocManaged((void **)&m_h_d_EAxisDualContourValue, bytesEAxisData));
    checkCudaErrors(cudaMallocManaged((void **)&m_h_d_EAxisPhysDataJ, bytesEAxisData));
    checkCudaErrors(cudaMallocManaged((void **)&m_h_d_EAxisPhysDataE, bytesEAxisData));
    checkCudaErrors(cudaMallocManaged((void **)&m_h_d_CPMLEAxisDualContourValue, bytesCPMLEAxisData));
    checkCudaErrors(cudaMallocManaged((void **)&m_h_d_CPMLEAxisPhysDataPE1, bytesCPMLEAxisData));
    checkCudaErrors(cudaMallocManaged((void **)&m_h_d_CPMLEAxisPhysDataE, bytesCPMLEAxisData));

    if (m_EAxisDualContourValue != NULL)
        aligned_free(m_EAxisDualContourValue);
    m_EAxisDualContourValue = (Standard_Real *)aligned_malloc(sizeof(Standard_Real) * bytesEAxisData);
    memset(m_EAxisDualContourValue, 0, sizeof(Standard_Real) * bytesEAxisData);

    if (m_EAxisPhysDataJ != NULL)
        aligned_free(m_EAxisPhysDataJ);
    m_EAxisPhysDataJ = (Standard_Real *)aligned_malloc(sizeof(Standard_Real) * bytesEAxisData);
    memset(m_EAxisPhysDataJ, 0, sizeof(Standard_Real) * bytesEAxisData);

    if (m_EAxisPhysDataE != NULL)
        aligned_free(m_EAxisPhysDataE);
    m_EAxisPhysDataE = (Standard_Real *)aligned_malloc(sizeof(Standard_Real) * bytesEAxisData);
    memset(m_EAxisPhysDataE, 0, sizeof(Standard_Real) * bytesEAxisData);

    if (m_CPMLEAxisDualContourValue != NULL)
        aligned_free(m_CPMLEAxisDualContourValue);
    m_CPMLEAxisDualContourValue = (Standard_Real *)aligned_malloc(sizeof(Standard_Real) * bytesCPMLEAxisData);
    memset(m_CPMLEAxisDualContourValue, 0, sizeof(Standard_Real) * bytesCPMLEAxisData);

    if (m_CPMLEAxisPhysDataPE1 != NULL)
        aligned_free(m_CPMLEAxisPhysDataPE1);
    m_CPMLEAxisPhysDataPE1 = (Standard_Real *)aligned_malloc(sizeof(Standard_Real) * bytesCPMLEAxisData);
    memset(m_CPMLEAxisPhysDataPE1, 0, sizeof(Standard_Real) * bytesCPMLEAxisData);

    if (m_CPMLEAxisPhysDataE != NULL)
        aligned_free(m_CPMLEAxisPhysDataE);
    m_CPMLEAxisPhysDataE = (Standard_Real *)aligned_malloc(sizeof(Standard_Real) * bytesCPMLEAxisData);
    memset(m_CPMLEAxisPhysDataE, 0, sizeof(Standard_Real) * bytesCPMLEAxisData);

    Standard_Real *h_m_Damping = (Standard_Real *)aligned_malloc(bytesdamping_scale);
    Standard_Integer *h_m_Parameters = (Standard_Integer *)aligned_malloc(bytesParameters);
    h_m_Damping[0] = m_Damping;
    h_m_Parameters[0] = phi_num;
    h_m_Parameters[1] = m_EAxisDatasNum / phi_num;
    h_m_Parameters[2] = m_CPML_EAxisDatasNum / phi_num;

    checkCudaErrors(cudaMemcpy(m_h_d_MphiDatasPtr, m_MphiDatasPtr, bytesMphiData, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(m_h_d_MzrDatasPtr, m_MzrDatasPtr, bytesMzrData, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(m_h_d_EphiDatasPtr, m_EphiDatasPtr, bytesEphiData, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(m_h_d_EzrDatasPtr, m_EzrDatasPtr, bytesEzrData, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(m_h_d_damping_scale, h_m_Damping, bytesdamping_scale, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(m_h_d_Parameters, h_m_Parameters, bytesParameters, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(m_h_d_EAxisDualContourValue, m_EAxisDualContourValue, bytesEAxisData, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(m_h_d_EAxisPhysDataJ, m_EAxisPhysDataJ, bytesEAxisData, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(m_h_d_EAxisPhysDataE, m_EAxisPhysDataE, bytesEAxisData, cudaMemcpyHostToDevice));

    int dataSize = sizeof(FIT_Mag_Func_Cyl3D) * m_MphiDatasNum;
    if (m_h_d_MphiFuncArray_Cyl3D == NULL)
        checkCudaErrors(cudaMalloc((void **)&m_h_d_MphiFuncArray_Cyl3D, dataSize));
    checkCudaErrors(cudaMemcpy(m_h_d_MphiFuncArray_Cyl3D, m_MphiFuncArray_Cyl3D, dataSize, cudaMemcpyHostToDevice));

    dataSize = sizeof(FIT_Mag_Func_Cyl3D) * m_MzrDatasNum;
    if (m_h_d_MzrFuncArray_Cyl3D == NULL)
        checkCudaErrors(cudaMalloc((void **)&m_h_d_MzrFuncArray_Cyl3D, dataSize));
    checkCudaErrors(cudaMemcpy(m_h_d_MzrFuncArray_Cyl3D, m_MzrFuncArray_Cyl3D, dataSize, cudaMemcpyHostToDevice));

    dataSize = sizeof(FIT_Mag_PML_Func_Cyl3D) * m_CPML_MphiDatasNum;
    if (m_h_d_CPMLMphiFuncArray_Cyl3D == NULL)
        checkCudaErrors(cudaMalloc((void **)&m_h_d_CPMLMphiFuncArray_Cyl3D, dataSize));
    checkCudaErrors(cudaMemcpy(m_h_d_CPMLMphiFuncArray_Cyl3D, m_MphiPMLFuncArray_Cyl3D, dataSize, cudaMemcpyHostToDevice));

    dataSize = sizeof(FIT_Mag_PML_Func_Cyl3D) * m_CPML_MzrDatasNum;
    if (m_h_d_CPMLMzrFuncArray_Cyl3D == NULL)
        checkCudaErrors(cudaMalloc((void **)&m_h_d_CPMLMzrFuncArray_Cyl3D, dataSize));
    checkCudaErrors(cudaMemcpy(m_h_d_CPMLMzrFuncArray_Cyl3D, m_MzrPMLFuncArray_Cyl3D, dataSize, cudaMemcpyHostToDevice));

    dataSize = sizeof(FIT_Elec_Func_Cyl3D) * m_EphiDatasNum;
    if (m_h_d_EphiFuncArray_Cyl3D == NULL)
        checkCudaErrors(cudaMalloc((void **)&m_h_d_EphiFuncArray_Cyl3D, dataSize));
    checkCudaErrors(cudaMemcpy(m_h_d_EphiFuncArray_Cyl3D, m_EphiFuncArray_Cyl3D, dataSize, cudaMemcpyHostToDevice));

    dataSize = sizeof(FIT_Elec_Func_Cyl3D) * m_EzrDatasNum;
    if (m_h_d_EzrFuncArray_Cyl3D == NULL)
        checkCudaErrors(cudaMalloc((void **)&m_h_d_EzrFuncArray_Cyl3D, dataSize));
    checkCudaErrors(cudaMemcpy(m_h_d_EzrFuncArray_Cyl3D, m_EzrFuncArray_Cyl3D, dataSize, cudaMemcpyHostToDevice));

    dataSize = sizeof(FIT_Elec_Func_Cyl3D) * m_EAxisDatasNum;
    if (m_h_d_EAxisFuncArray_Cyl3D == NULL)
        checkCudaErrors(cudaMalloc((void **)&m_h_d_EAxisFuncArray_Cyl3D, dataSize));
    checkCudaErrors(cudaMemcpy(m_h_d_EAxisFuncArray_Cyl3D, m_EAxisFuncArray_Cyl3D, dataSize, cudaMemcpyHostToDevice));

    dataSize = sizeof(FIT_Elec_PML_Func_Cyl3D) * m_CPML_EphiDatasNum;
    if (m_h_d_CPMLEphiFuncArray_Cyl3D == NULL)
        checkCudaErrors(cudaMalloc((void **)&m_h_d_CPMLEphiFuncArray_Cyl3D, dataSize));
    checkCudaErrors(cudaMemcpy(m_h_d_CPMLEphiFuncArray_Cyl3D, m_EphiPMLFuncArray_Cyl3D, dataSize, cudaMemcpyHostToDevice));

    dataSize = sizeof(FIT_Elec_PML_Func_Cyl3D) * m_CPML_EzrDatasNum;
    if (m_h_d_CPMLEzrFuncArray_Cyl3D == NULL)
        checkCudaErrors(cudaMalloc((void **)&m_h_d_CPMLEzrFuncArray_Cyl3D, dataSize));
    checkCudaErrors(cudaMemcpy(m_h_d_CPMLEzrFuncArray_Cyl3D, m_EzrPMLFuncArray_Cyl3D, dataSize, cudaMemcpyHostToDevice));

    dataSize = sizeof(FIT_Elec_PML_Func_Cyl3D) * m_CPML_EAxisDatasNum;
    if (m_h_d_CPMLEAxisFuncArray_Cyl3D == NULL)
        checkCudaErrors(cudaMalloc((void **)&m_h_d_CPMLEAxisFuncArray_Cyl3D, dataSize));
    checkCudaErrors(cudaMemcpy(m_h_d_CPMLEAxisFuncArray_Cyl3D, m_EAxisPMLFuncArray_Cyl3D, dataSize, cudaMemcpyHostToDevice));

    // m_FldSrcs_Cyl3D->Get_SrcData(&amp, amp_size);
    // Build_Elec_MurVoltage();

    dataSize = sizeof(FIT_Elec_Mur_Func_Cyl3D) * m_Mur_EzrDatasNum;
    if (m_h_d_MurEzrFuncArray_Cyl3D == NULL)
        checkCudaErrors(cudaMalloc((void **)&m_h_d_MurEzrFuncArray_Cyl3D, dataSize));
    checkCudaErrors(cudaMemcpy(m_h_d_MurEzrFuncArray_Cyl3D, m_EzrMurFuncArray_Cyl3D, dataSize, cudaMemcpyHostToDevice));

    dataSize = sizeof(FIT_Elec_Mur_Func_Cyl3D) * m_Mur_EphiDatasNum;
    if (m_h_d_MurEphiFuncArray_Cyl3D == NULL)
        checkCudaErrors(cudaMalloc((void **)&m_h_d_MurEphiFuncArray_Cyl3D, dataSize));
    checkCudaErrors(cudaMemcpy(m_h_d_MurEphiFuncArray_Cyl3D, m_EphiMurFuncArray_Cyl3D, dataSize, cudaMemcpyHostToDevice));

    dataSize = sizeof(FIT_Elec_Mur_Func_Cyl3D) * m_MurVoltage_EzrDatasNum;
    if (m_h_d_MurVoltageEzrFuncArray_Cyl3D == NULL)
        checkCudaErrors(cudaMalloc((void **)&m_h_d_MurVoltageEzrFuncArray_Cyl3D, dataSize));
    checkCudaErrors(cudaMemcpy(m_h_d_MurVoltageEzrFuncArray_Cyl3D, m_EzrMurVoltageFuncArray_Cyl3D, dataSize, cudaMemcpyHostToDevice));

    dataSize = sizeof(FIT_Elec_Mur_Func_Cyl3D) * m_MurVoltage_EphiDatasNum;
    if (m_h_d_MurVoltageEphiFuncArray_Cyl3D == NULL)
        checkCudaErrors(cudaMalloc((void **)&m_h_d_MurVoltageEphiFuncArray_Cyl3D, dataSize));
    checkCudaErrors(cudaMemcpy(m_h_d_MurVoltageEphiFuncArray_Cyl3D, m_EphiMurVoltageFuncArray_Cyl3D, dataSize, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMallocManaged(&h_d_Ebar, sizeof(Standard_Real) * m_MurVoltage_EzrDatasNum));
    checkCudaErrors(cudaMallocManaged(&h_d_Ebar2, sizeof(Standard_Real) * m_MurVoltage_EzrDatasNum));
}

void SI_SC_Matrix_EMFields_Cyl3D::
    Build_Elec_MurVoltage()
{
    Standard_Real dt = this->GetDelTime();

    Standard_Integer dynEIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicElecField_PhysDataIndex();   // 0
    Standard_Integer PreIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_PRE_PhysDataIndex();                 // 4
    Standard_Integer AEIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_AE_PhysDataIndex();                   // 2
    Standard_Integer BEIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_BE_PhysDataIndex();                   // 3
    Standard_Integer preTStepEFldIndx = GetFldsDefCntr()->GetFieldsDefineRules()->Get_MUR_PreStep_PhysDataIndex(); // 5

    bool doDamping;
    if (m_Damping > 0.001)
    {
        doDamping = true;
    }
    else
    {
        doDamping = false;
    }

    Build_Elec_MurVoltage_Func(doDamping, dt, dynEIndex, PreIndex, AEIndex, BEIndex, preTStepEFldIndx);
}

void SI_SC_Matrix_EMFields_Cyl3D::
    Build_Elec_MurVoltage_Func(const bool doDamping,
                               const Standard_Real &dt,
                               const Standard_Integer &dynEIndex,
                               const Standard_Integer &PreIndex,
                               const Standard_Integer &AEIndex,
                               const Standard_Integer &BEIndex,
                               const Standard_Integer &preTStepEFldIndx)
{
    // 1.0 Build MurVoltage Ezr Func Data
    m_FldSrcs_Cyl3D->Get_SrcDataVec(0, &m_MurEdgeDatas, &m_FreeEdgeDatas, &m_MurSweptEdgeDatas, &m_FreeSweptEdgeDatas);
    int nEdge = m_MurEdgeDatas.size();
    m_MurVoltage_EzrDatasNum = nEdge * phi_num;
    m_EzrMurVoltageFuncArray_Cyl3D = NULL;
    m_EzrMurVoltageFuncArray_Cyl3D = new FIT_Elec_Mur_Func_Cyl3D[m_MurVoltage_EzrDatasNum];

    for (int i = 0; i < m_MurVoltage_EzrDatasNum; ++i)
    {
        m_FldSrcs_Cyl3D->Get_SrcDataVec(i / nEdge, &m_MurEdgeDatas, &m_FreeEdgeDatas, &m_MurSweptEdgeDatas, &m_FreeSweptEdgeDatas);
        int idx = i;
        if (i >= nEdge)
            idx = i - i / nEdge * nEdge;
        GridEdgeData *currEdge = m_MurEdgeDatas[idx];
        GridEdgeData *currEdge1 = m_FreeEdgeDatas[idx];
        Standard_Real *currDataPtr = currEdge->GetPhysDataPtr(0);
        Standard_Real *currDataPtr1 = currEdge1->GetPhysDataPtr(0);

        m_EzrMurVoltageFuncArray_Cyl3D[i].m_Elec = currDataPtr + dynEIndex;
        m_EzrMurVoltageFuncArray_Cyl3D[i].m_Elec_Ptr_Offset = m_EzrMurVoltageFuncArray_Cyl3D[i].m_Elec - m_EzrDatasPtr;

        m_EzrMurVoltageFuncArray_Cyl3D[i].m_PreTStep = currDataPtr + preTStepEFldIndx;
        m_EzrMurVoltageFuncArray_Cyl3D[i].m_PreTStep_Ptr_Offset = m_EzrMurVoltageFuncArray_Cyl3D[i].m_PreTStep - m_EzrDatasPtr;

        m_EzrMurVoltageFuncArray_Cyl3D[i].m_AE = currDataPtr + AEIndex;
        m_EzrMurVoltageFuncArray_Cyl3D[i].m_AE_Ptr_Offset = m_EzrMurVoltageFuncArray_Cyl3D[i].m_AE - m_EzrDatasPtr;

        m_EzrMurVoltageFuncArray_Cyl3D[i].m_BE = currDataPtr + BEIndex;
        m_EzrMurVoltageFuncArray_Cyl3D[i].m_BE_Ptr_Offset = m_EzrMurVoltageFuncArray_Cyl3D[i].m_BE - m_EzrDatasPtr;

        m_EzrMurVoltageFuncArray_Cyl3D[i].m_PreTStepEFld = currDataPtr1 + dynEIndex;
        m_EzrMurVoltageFuncArray_Cyl3D[i].m_PreTStepEFld_Ptr_Offset = m_EzrMurVoltageFuncArray_Cyl3D[i].m_PreTStepEFld - m_EzrDatasPtr;

        m_EzrMurVoltageFuncArray_Cyl3D[i].m_VBar = m_VBAR;
    }

    // 2.0 Build MurVoltage Ephi Func Data
    m_FldSrcs_Cyl3D->Get_SrcDataVec(0, &m_MurEdgeDatas, &m_FreeEdgeDatas, &m_MurSweptEdgeDatas, &m_FreeSweptEdgeDatas);
    int nVertex = m_MurSweptEdgeDatas.size();
    m_MurVoltage_EphiDatasNum = nVertex * phi_num;
    m_EphiMurVoltageFuncArray_Cyl3D = NULL;
    m_EphiMurVoltageFuncArray_Cyl3D = new FIT_Elec_Mur_Func_Cyl3D[m_MurVoltage_EphiDatasNum];

    for (int i = 0; i < m_MurVoltage_EphiDatasNum; ++i)
    {
        m_FldSrcs_Cyl3D->Get_SrcDataVec(i / nVertex, &m_MurEdgeDatas, &m_FreeEdgeDatas, &m_MurSweptEdgeDatas, &m_FreeSweptEdgeDatas);
        int idx = i;
        if (i >= nVertex)
            idx = i - i / nVertex * nVertex;
        GridVertexData *currVertex = m_MurSweptEdgeDatas[idx];
        GridVertexData *currVertex1 = m_FreeSweptEdgeDatas[idx];
        Standard_Real *currDataPtr = currVertex->GetSweptPhysDataPtr(0);
        Standard_Real *currDataPtr1 = currVertex1->GetSweptPhysDataPtr(0);

        m_EphiMurVoltageFuncArray_Cyl3D[i].m_Elec = currDataPtr + dynEIndex;
        m_EphiMurVoltageFuncArray_Cyl3D[i].m_Elec_Ptr_Offset = m_EphiMurVoltageFuncArray_Cyl3D[i].m_Elec - m_EphiDatasPtr;

        m_EphiMurVoltageFuncArray_Cyl3D[i].m_PreTStep = currDataPtr + preTStepEFldIndx;
        m_EphiMurVoltageFuncArray_Cyl3D[i].m_PreTStep_Ptr_Offset = m_EphiMurVoltageFuncArray_Cyl3D[i].m_PreTStep - m_EphiDatasPtr;

        m_EphiMurVoltageFuncArray_Cyl3D[i].m_AE = currDataPtr + AEIndex;
        m_EphiMurVoltageFuncArray_Cyl3D[i].m_AE_Ptr_Offset = m_EphiMurVoltageFuncArray_Cyl3D[i].m_AE - m_EphiDatasPtr;

        m_EphiMurVoltageFuncArray_Cyl3D[i].m_BE = currDataPtr + BEIndex;
        m_EphiMurVoltageFuncArray_Cyl3D[i].m_BE_Ptr_Offset = m_EphiMurVoltageFuncArray_Cyl3D[i].m_BE - m_EphiDatasPtr;

        m_EphiMurVoltageFuncArray_Cyl3D[i].m_PreTStepEFld = currDataPtr1 + dynEIndex;
        m_EphiMurVoltageFuncArray_Cyl3D[i].m_PreTStepEFld_Ptr_Offset = m_EphiMurVoltageFuncArray_Cyl3D[i].m_PreTStepEFld - m_EphiDatasPtr;

        m_EphiMurVoltageFuncArray_Cyl3D[i].m_VBar = m_VBAR;
    }
}

void SI_SC_Matrix_EMFields_Cyl3D::
    CleanCUDADatas()
{
    cudaFree(m_h_d_MphiDatasPtr);
    cudaFree(m_h_d_MzrDatasPtr);
    cudaFree(m_h_d_EphiDatasPtr);
    cudaFree(m_h_d_EzrDatasPtr);

    cudaFree(m_h_d_MphiFuncArray_Cyl3D);
    cudaFree(m_h_d_MzrFuncArray_Cyl3D);
    cudaFree(m_h_d_CPMLMphiFuncArray_Cyl3D);
    cudaFree(m_h_d_CPMLMzrFuncArray_Cyl3D);

    cudaFree(m_h_d_EphiFuncArray_Cyl3D);
    cudaFree(m_h_d_EzrFuncArray_Cyl3D);
    cudaFree(m_h_d_EAxisFuncArray_Cyl3D);
    cudaFree(m_h_d_EAxisDualContourValue);
    cudaFree(m_h_d_EAxisPhysDataJ);
    cudaFree(m_h_d_EAxisPhysDataE);
    cudaFree(m_h_d_CPMLEAxisDualContourValue);
    cudaFree(m_h_d_CPMLEAxisPhysDataPE1);
    cudaFree(m_h_d_CPMLEAxisPhysDataE);

    cudaFree(m_h_d_CPMLEphiFuncArray_Cyl3D);
    cudaFree(m_h_d_CPMLEzrFuncArray_Cyl3D);
    cudaFree(m_h_d_CPMLEAxisFuncArray_Cyl3D);

    cudaFree(m_h_d_MurEphiFuncArray_Cyl3D);
    cudaFree(m_h_d_MurEzrFuncArray_Cyl3D);
    cudaFree(m_h_d_MurVoltageEphiFuncArray_Cyl3D);
    cudaFree(m_h_d_MurVoltageEzrFuncArray_Cyl3D);
}

void SI_SC_Matrix_EMFields_Cyl3D::
    TransDgnData()
{
    checkCudaErrors(cudaMemcpy(m_MphiDatasPtr, m_h_d_MphiDatasPtr, bytesMphiData, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(m_MzrDatasPtr, m_h_d_MzrDatasPtr, bytesMzrData, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(m_EphiDatasPtr, m_h_d_EphiDatasPtr, bytesEphiData, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(m_EzrDatasPtr, m_h_d_EzrDatasPtr, bytesEzrData, cudaMemcpyDeviceToHost));
}

void SI_SC_Matrix_EMFields_Cyl3D::
    InitMatrixData()
{
    Standard_Real dt = GetDelTime();
    m_FldSrcs_Cyl3D->Get_VBar(0, m_VBAR);
    m_VBAR *= dt;
    m_FldSrcs_Cyl3D->Get_SrcData(&amp, amp_size);
    Build_Elec_MurVoltage();
}

void SI_SC_Matrix_EMFields_Cyl3D::
    AdvanceWithDamping()
{
    m_FldSrcs_Cyl3D->Advance();
    m_FldSrcs_Cyl3D->Advance_SI_MJ(1.0);            // null
    m_FldSrcs_Cyl3D->Advance_SI_Mag_Damping_0(1.0); // null

    AdvanceMagCntr_WithDamping(); // 待补充 // 待修改 error
    // AdvanceMagCPML_WithDamping(); // 待补充 // 待修改

    m_FldSrcs_Cyl3D->Advance_SI_Mag_Damping_1(1.0); // null

    m_FldSrcs_Cyl3D->Advance_SI_J(1.0);                         // null
    m_FldSrcs_Cyl3D->Advance_SI_Elec_Damping_0(1.0, m_Damping); // null

    AdvanceElecCntr_WithDamping(m_Damping); // 待补充 // 待修改
    // AdvanceElecCPML_WithDamping(m_Damping); // 待补充 // 待修改

    AdvanceElecMur_WithDamping(m_Damping);
    AdvanceElecMurVoltage_WithDamping(m_Damping);

#ifdef __CUDA__
    cudaDeviceSynchronize();
#endif

    m_MCntrFields_Cyl3D->Advance();
    m_MCPMLFields_Cyl3D->Advance();
    m_ECntrFields_Cyl3D->Advance();
    m_ECPMLFields_Cyl3D->Advance();

    DynObj::Advance();
}

void SI_SC_Matrix_EMFields_Cyl3D::
    AdvanceElecCntr()
{
    for (Cempic_Size i = 0; i < m_EzrDatasNum; ++i)
    {
        m_EzrFuncArray_Cyl3D[i].AdvanceEzr();
    }
    for (Cempic_Size i = 0; i < m_EphiDatasNum; ++i)
    {
        m_EphiFuncArray_Cyl3D[i].AdvanceEphi();
    }
}

void SI_SC_Matrix_EMFields_Cyl3D::
    AdvanceElecCPML()
{
    for (Cempic_Size i = 0; i < m_CPML_EzrDatasNum; ++i)
    {
        m_EzrPMLFuncArray_Cyl3D[i].AdvanceEzr();
    }
    for (Cempic_Size i = 0; i < m_CPML_EphiDatasNum; ++i)
    {
        m_EphiPMLFuncArray_Cyl3D[i].AdvanceEphi();
    }
}

void SI_SC_Matrix_EMFields_Cyl3D::
    AdvanceMagCntr()
{
    for (Cempic_Size i = 0; i < m_MphiDatasNum; ++i)
    {
        m_MphiFuncArray_Cyl3D[i].AdvanceMphi();
    }
    for (Cempic_Size i = 0; i < m_MzrDatasNum; ++i)
    {
        m_MzrFuncArray_Cyl3D[i].AdvanceMzr();
    }
}

void SI_SC_Matrix_EMFields_Cyl3D::
    AdvanceMagCPML()
{
    for (Cempic_Size i = 0; i < m_CPML_MphiDatasNum; ++i)
    {
        m_MphiPMLFuncArray_Cyl3D[i].AdvanceMphi();
    }
    for (Cempic_Size i = 0; i < m_CPML_MzrDatasNum; ++i)
    {
        m_MzrPMLFuncArray_Cyl3D[i].AdvanceMzr();
    }
}

void SI_SC_Matrix_EMFields_Cyl3D::
    AdvanceMagCntr_WithDamping()
{
#ifdef __CUDA__
    dim3 block(256);
    dim3 grid((unsigned int)ceil(m_MphiDatasNum / (float)block.x));

    g_AdvanceMagCntr_WithDamping_m_Mphi<<<grid, block>>>(m_h_d_MphiFuncArray_Cyl3D, m_MphiDatasNum,
                                                         m_h_d_MphiDatasPtr, m_h_d_EzrDatasPtr);

    block.x = 256;
    grid.x = ((unsigned int)ceil(m_MzrDatasNum / (float)block.x));

    g_AdvanceMagCntr_WithDamping_m_Mzr<<<grid, block>>>(m_h_d_MzrFuncArray_Cyl3D, m_MzrDatasNum,
                                                        m_h_d_MzrDatasPtr, m_h_d_EphiDatasPtr, m_h_d_EzrDatasPtr);
#elif defined(__MATRIX__)
    for (Standard_Size i = 0; i < m_MphiDatasNum; ++i)
    {
        m_MphiFuncArray_Cyl3D[i].AdvanceMphi();
    }

    for (Standard_Size i = 0; i < m_MzrDatasNum; ++i)
    {
        m_MzrFuncArray_Cyl3D[i].AdvanceMzr();
    }
#else
    m_MCntrFields_Cyl3D->Advance_SI_Damping(m_b[0]); // 此处考虑Near
#endif
}

void SI_SC_Matrix_EMFields_Cyl3D::
    AdvanceMagCPML_WithDamping()
{
#ifdef __CUDA__
    dim3 block(256);
    dim3 grid((unsigned int)ceil(m_CPML_MphiDatasNum / (float)block.x));

    g_AdvanceMagCPML_WithDamping_m_Mphi<<<grid, block>>>(m_h_d_CPMLMphiFuncArray_Cyl3D, m_CPML_MphiDatasNum,
                                                         m_h_d_MphiDatasPtr, m_h_d_EzrDatasPtr);

    block.x = 256;
    grid.x = (unsigned int)ceil(m_CPML_MzrDatasNum / (float)block.x);

    g_AdvanceMagCPML_WithDamping_m_Mzr<<<grid, block>>>(m_h_d_CPMLMzrFuncArray_Cyl3D, m_CPML_MzrDatasNum,
                                                        m_h_d_MzrDatasPtr, m_h_d_EphiDatasPtr, m_h_d_EzrDatasPtr);
#elif defined(__MATRIX__)
    for (Standard_Size i = 0; i < m_CPML_MphiDatasNum; ++i)
    {
        m_MphiPMLFuncArray_Cyl3D[i].AdvanceMphi();
    }

    for (Standard_Size i = 0; i < m_CPML_MzrDatasNum; ++i)
    {
        m_MzrPMLFuncArray_Cyl3D[i].AdvanceMzr();
    }
#else
    m_MCPMLFields_Cyl3D->Advance_SI_Damping(m_b[0]); // 此处考虑Near
#endif
}

void SI_SC_Matrix_EMFields_Cyl3D::
    AdvanceElecCntr_WithDamping(const Standard_Real damping_scale)
{
#ifdef __CUDA__
    dim3 block(256);
    dim3 grid((unsigned int)ceil(m_EzrDatasNum / (float)block.x));

    g_AdvanceElecCntr_WithDamping_m_Ezr<<<grid, block>>>(m_h_d_damping_scale, m_h_d_EzrFuncArray_Cyl3D, m_EzrDatasNum,
                                                         m_h_d_EzrDatasPtr, m_h_d_MphiDatasPtr, m_h_d_MzrDatasPtr);

    block.x = 256;
    grid.x = (unsigned int)ceil(m_EphiDatasNum / (float)block.x);

    g_AdvanceElecCntr_WithDamping_m_Ephi<<<grid, block>>>(m_h_d_damping_scale, m_h_d_EphiFuncArray_Cyl3D, m_EphiDatasNum,
                                                          m_h_d_EphiDatasPtr, m_h_d_MzrDatasPtr);

    block.x = 256;
    grid.x = (unsigned int)ceil(m_EAxisDatasNum / (float)block.x);

    g_AdvanceElecCntr_WithDamping_m_EAxis<<<grid, block>>>(m_h_d_Parameters, m_h_d_EAxisFuncArray_Cyl3D, m_EAxisDatasNum,
                                                           m_h_d_EzrDatasPtr, m_h_d_MphiDatasPtr, m_h_d_EAxisDualContourValue,
                                                           m_h_d_EAxisPhysDataJ, m_h_d_EAxisPhysDataE);

    block.x = 256;
    grid.x = (unsigned int)ceil(m_EAxisDatasNum / phi_num / (float)block.x);

    g_AdvanceElecCntr_WithDamping_m_EAxis_1<<<grid, block>>>(m_h_d_Parameters, m_h_d_EAxisDualContourValue, m_h_d_EAxisPhysDataJ, m_h_d_EAxisPhysDataE, m_EAxisDatasNum / phi_num);

    block.x = 256;
    grid.x = (unsigned int)ceil(m_EAxisDatasNum / (float)block.x);

    g_AdvanceElecCntr_WithDamping_m_EAxis_2<<<grid, block>>>(m_h_d_Parameters, m_h_d_EAxisFuncArray_Cyl3D, m_EAxisDatasNum,
                                                             m_h_d_EAxisDualContourValue, m_h_d_EAxisPhysDataJ, m_h_d_EAxisPhysDataE);

    block.x = 256;
    grid.x = (unsigned int)ceil(m_EAxisDatasNum / phi_num / (float)block.x);

    g_AdvanceElecCntr_WithDamping_m_EAxis_3<<<grid, block>>>(m_h_d_EAxisDualContourValue, m_h_d_EAxisPhysDataJ, m_h_d_EAxisPhysDataE, m_EAxisDatasNum / phi_num);

    block.x = 256;
    grid.x = (unsigned int)ceil(m_EAxisDatasNum / (float)block.x);

    g_AdvanceElecCntr_WithDamping_m_EAxis_end<<<grid, block>>>(m_h_d_damping_scale, m_h_d_EAxisFuncArray_Cyl3D, m_EAxisDatasNum,
                                                               m_h_d_EzrDatasPtr);
#elif defined(__MATRIX__)
    for (Standard_Size i = 0; i < m_EzrDatasNum; ++i)
    {
        m_EzrFuncArray_Cyl3D[i].AdvanceEzr_Damping(damping_scale);
    }

    for (Standard_Size i = 0; i < m_EphiDatasNum; ++i)
    {
        m_EphiFuncArray_Cyl3D[i].AdvanceEphi_Damping(damping_scale);
    }

    int Phi_Num = phi_num;
    int idxMax = m_EAxisDatasNum / Phi_Num;
    for (Standard_Size i = 0; i < idxMax; ++i)
    {
        Standard_Real DualContourValue = 0.0;
        Standard_Real PhysDataJ = 0.0;
        Standard_Real PhysDataE = 0.0;
        for (int j = 0; j < Phi_Num; ++j)
        {
            Standard_Size index = idxMax * j + i;

            DualContourValue += m_EAxisFuncArray_Cyl3D[index].ComputeContourAxis();
            PhysDataJ += *(m_EAxisFuncArray_Cyl3D[index].m_Current);
            PhysDataE += *(m_EAxisFuncArray_Cyl3D[index].m_Elec);
        }

        PhysDataJ /= Phi_Num;
        PhysDataE /= Phi_Num;
        for (int j = 0; j < phi_num; ++j)
        {
            Standard_Size index = idxMax * j + i;
            m_EAxisFuncArray_Cyl3D[index].m_current = PhysDataJ;
            m_EAxisFuncArray_Cyl3D[index].m_elec = PhysDataE;
            m_EAxisFuncArray_Cyl3D[index].m_DualContour = DualContourValue;
        }
    }

    for (Standard_Size i = 0; i < m_EAxisDatasNum; ++i)
    {
        *(m_EAxisFuncArray_Cyl3D[i].m_AE) = (1.0 - damping_scale) * (*(m_EAxisFuncArray_Cyl3D[i].m_Elec)) +
                                            damping_scale * (*(m_EAxisFuncArray_Cyl3D[i].m_AE));
        *(m_EAxisFuncArray_Cyl3D[i].m_Elec_PreStep) = *(m_EAxisFuncArray_Cyl3D[i].m_Elec);

        *(m_EAxisFuncArray_Cyl3D[i].m_Elec) = m_EAxisFuncArray_Cyl3D[i].m_C0 * m_EAxisFuncArray_Cyl3D[i].m_elec +
                                              m_EAxisFuncArray_Cyl3D[i].m_C2 * (m_EAxisFuncArray_Cyl3D[i].m_DualContour - m_EAxisFuncArray_Cyl3D[i].m_current) * m_EAxisFuncArray_Cyl3D[i].m_C3;

        *(m_EAxisFuncArray_Cyl3D[i].m_BE) = (1.0 + 0.5 * damping_scale) * (*(m_EAxisFuncArray_Cyl3D[i].m_Elec)) - 0.5 * (*(m_EAxisFuncArray_Cyl3D[i].m_Elec_PreStep)) + 0.5 * (1.0 - damping_scale) * (*(m_EAxisFuncArray_Cyl3D[i].m_AE));
    }
#else
    m_ECntrFields_Cyl3D->Advance_SI_Damping(m_bb[0], m_Damping); // 此处非常多不一样的
#endif
}

void SI_SC_Matrix_EMFields_Cyl3D::
    AdvanceElecMur_WithDamping(const Standard_Real damping_scale)
{
#ifdef __CUDA__
    dim3 block(256);
    dim3 grid((unsigned int)ceil(m_Mur_EzrDatasNum / (float)block.x));

    g_AdvanceElecMur_WithDamping_m_Ezr<<<grid, block>>>(m_h_d_damping_scale, m_h_d_MurEzrFuncArray_Cyl3D, m_Mur_EzrDatasNum,
                                                        m_h_d_EzrDatasPtr);

    block.x = 256;
    grid.x = (unsigned int)ceil(m_Mur_EphiDatasNum / (float)block.x);

    g_AdvanceElecMur_WithDamping_m_Ephi<<<grid, block>>>(m_h_d_damping_scale, m_h_d_MurEphiFuncArray_Cyl3D, m_Mur_EphiDatasNum,
                                                         m_h_d_EphiDatasPtr);
#elif defined(__MATRIX__)
    for (int i = 0; i < m_Mur_EzrDatasNum; ++i)
    {
        m_EzrMurFuncArray_Cyl3D[i].AdvanceEzr_Damping(damping_scale);
    }

    for (int i = 0; i < m_Mur_EphiDatasNum; ++i)
    {
        m_EphiMurFuncArray_Cyl3D[i].AdvanceEphi_Damping(damping_scale);
    }
#else
    m_EMurFields_Cyl3D->Advance_SI_Damping(1.0, m_Damping); // 待补充 // 待修改
#endif
}

void SI_SC_Matrix_EMFields_Cyl3D::
    AdvanceElecMurVoltage_WithDamping(const Standard_Real damping_scale)
{
#ifdef __CUDA__
    m_FldSrcs_Cyl3D->Get_Parameters(0, Ebar, Ebar2);
    int jdx = m_MurVoltage_EzrDatasNum / phi_num;

    for (int i = 0; i < m_MurVoltage_EzrDatasNum; ++i)
    {
        h_d_Ebar[i] = amp[i % jdx] * Ebar;
        h_d_Ebar2[i] = amp[i % jdx] * Ebar2;
    }

    dim3 block(256);
    dim3 grid((unsigned int)ceil(m_MurVoltage_EzrDatasNum / (float)block.x));

    g_AdvanceElecMurVoltage_WithDamping_m_Ezr<<<grid, block>>>(m_h_d_damping_scale, m_h_d_MurVoltageEzrFuncArray_Cyl3D, m_MurVoltage_EzrDatasNum,
                                                               m_h_d_EzrDatasPtr, h_d_Ebar, h_d_Ebar2);

    block.x = 256;
    grid.x = (unsigned int)ceil(m_MurVoltage_EphiDatasNum / (float)block.x);

    g_AdvanceElecMurVoltage_WithDamping_m_Ephi<<<grid, block>>>(m_h_d_damping_scale, m_h_d_MurVoltageEphiFuncArray_Cyl3D, m_MurVoltage_EphiDatasNum,
                                                                m_h_d_EphiDatasPtr);

    for (int i = 0; i < phi_num; ++i)
        m_FldSrcs_Cyl3D->addCurrTime(i, 1.0);
#elif defined(__MATRIX__)
    m_FldSrcs_Cyl3D->Get_Parameters(0, Ebar, Ebar2);
    int jdx = m_MurVoltage_EzrDatasNum / phi_num;

    for (int i = 0; i < m_MurVoltage_EzrDatasNum; ++i)
    {
        m_EzrMurVoltageFuncArray_Cyl3D[i].AdvanceEzr_Damping_TFunc(damping_scale, amp[i % jdx] * Ebar, amp[i % jdx] * Ebar2);
    }

    for (int i = 0; i < m_MurVoltage_EphiDatasNum; ++i)
    {
        m_EphiMurVoltageFuncArray_Cyl3D[i].AdvanceEphi_Damping(damping_scale);
    }

    for (int i = 0; i < phi_num; ++i)
        m_FldSrcs_Cyl3D->addCurrTime(i, 1.0);
#else
    m_FldSrcs_Cyl3D->Advance_SI_Elec_Damping_1(1.0, m_Damping);
#endif
}

void SI_SC_Matrix_EMFields_Cyl3D::
    AdvanceElecCPML_WithDamping(const Standard_Real damping_scale)
{
#ifdef __CUDA__
    dim3 block(256);
    dim3 grid((unsigned int)ceil(m_CPML_EzrDatasNum / (float)block.x));

    g_AdvanceElecCPML_WithDamping_m_Ezr<<<grid, block>>>(m_h_d_damping_scale, m_h_d_CPMLEzrFuncArray_Cyl3D, m_CPML_EzrDatasNum,
                                                         m_h_d_EzrDatasPtr, m_h_d_MphiDatasPtr, m_h_d_MzrDatasPtr);

    block.x = 256;
    grid.x = (unsigned int)ceil(m_CPML_EphiDatasNum / (float)block.x);

    g_AdvanceElecCPML_WithDamping_m_Ephi<<<grid, block>>>(m_h_d_damping_scale, m_h_d_CPMLEphiFuncArray_Cyl3D, m_CPML_EphiDatasNum,
                                                          m_h_d_EphiDatasPtr, m_h_d_MzrDatasPtr);

    block.x = 256;
    grid.x = (unsigned int)ceil(m_CPML_EAxisDatasNum / (float)block.x);

    g_AdvanceElecCPML_WithDamping_m_EAxis<<<grid, block>>>(m_h_d_Parameters, m_h_d_CPMLEAxisFuncArray_Cyl3D, m_CPML_EAxisDatasNum,
                                                           m_h_d_EzrDatasPtr, m_h_d_MphiDatasPtr, m_h_d_CPMLEAxisDualContourValue, m_h_d_CPMLEAxisPhysDataPE1, m_h_d_CPMLEAxisPhysDataE);

    block.x = 256;
    grid.x = (unsigned int)ceil(m_CPML_EAxisDatasNum / phi_num / (float)block.x);

    g_AdvanceElecCPML_WithDamping_m_EAxis_1<<<grid, block>>>(m_h_d_Parameters, m_h_d_CPMLEAxisPhysDataPE1, m_h_d_CPMLEAxisPhysDataE, m_CPML_EAxisDatasNum / phi_num);

    block.x = 256;
    grid.x = (unsigned int)ceil(m_CPML_EAxisDatasNum / (float)block.x);

    g_AdvanceElecCPML_WithDamping_m_EAxis_2<<<grid, block>>>(m_h_d_Parameters, m_h_d_CPMLEAxisFuncArray_Cyl3D, m_CPML_EAxisDatasNum,
                                                             m_h_d_CPMLEAxisDualContourValue, m_h_d_CPMLEAxisPhysDataPE1, m_h_d_CPMLEAxisPhysDataE);

    block.x = 256;
    grid.x = (unsigned int)ceil(m_CPML_EAxisDatasNum / phi_num / (float)block.x);

    g_AdvanceElecCPML_WithDamping_m_EAxis_3<<<grid, block>>>(m_h_d_CPMLEAxisDualContourValue, m_h_d_CPMLEAxisPhysDataPE1, m_h_d_CPMLEAxisPhysDataE, m_CPML_EAxisDatasNum / phi_num);

    block.x = 256;
    grid.x = (unsigned int)ceil(m_CPML_EAxisDatasNum / (float)block.x);

    g_AdvanceElecCPML_WithDamping_m_EAxis_end<<<grid, block>>>(m_h_d_damping_scale, m_h_d_CPMLEAxisFuncArray_Cyl3D, m_CPML_EAxisDatasNum, m_h_d_EzrDatasPtr);

    // checkCudaErrors(cudaMemcpy(m_MphiDatasPtr, m_h_d_MphiDatasPtr, bytesMphiData, cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaMemcpy(m_MzrDatasPtr, m_h_d_MzrDatasPtr, bytesMzrData, cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaMemcpy(m_EphiDatasPtr, m_h_d_EphiDatasPtr, bytesEphiData, cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaMemcpy(m_EzrDatasPtr, m_h_d_EzrDatasPtr, bytesEzrData, cudaMemcpyDeviceToHost));
#elif defined(__MATRIX__)
    for (Standard_Size i = 0; i < m_CPML_EzrDatasNum; ++i)
    {
        m_EzrPMLFuncArray_Cyl3D[i].AdvanceEzr_Damping(damping_scale);
    }

    for (Standard_Size i = 0; i < m_CPML_EphiDatasNum; ++i)
    {
        m_EphiPMLFuncArray_Cyl3D[i].AdvanceEphi_Damping(damping_scale);
    }

    int Phi_Num = Get_Phi_Num();
    int idxMax = m_CPML_EAxisDatasNum / Phi_Num;
    for (Standard_Size i = 0; i < idxMax; ++i)
    {
        Standard_Real DualContourValue = 0.0;
        Standard_Real PhysDataE = 0.0;
        Standard_Real PhysDataPE1 = 0.0;
        for (int j = 0; j < Phi_Num; ++j)
        {
            Standard_Size index = idxMax * j + i;

            DualContourValue += m_EAxisPMLFuncArray_Cyl3D[index].ComputContourAxis();
            PhysDataPE1 += *(m_EAxisPMLFuncArray_Cyl3D[index].m_PE1);
            PhysDataE += *(m_EAxisPMLFuncArray_Cyl3D[index].m_Elec);
        }
        PhysDataPE1 /= Phi_Num;
        PhysDataE /= Phi_Num;
        for (int j = 0; j < Phi_Num; ++j)
        {
            Standard_Size index = idxMax * j + i;
            m_EAxisPMLFuncArray_Cyl3D[index].m_pe1 = PhysDataPE1;
            m_EAxisPMLFuncArray_Cyl3D[index].m_elec = PhysDataE;
            m_EAxisPMLFuncArray_Cyl3D[index].m_DualContour = DualContourValue;
        }
    }

    for (Standard_Size i = 0; i < m_CPML_EAxisDatasNum; ++i)
    {
        *(m_EAxisPMLFuncArray_Cyl3D[i].m_AE) = (1.0 - damping_scale) * (*(m_EAxisPMLFuncArray_Cyl3D[i].m_Elec)) +
                                               damping_scale * (*(m_EAxisPMLFuncArray_Cyl3D[i].m_AE));
        *(m_EAxisPMLFuncArray_Cyl3D[i].m_Elec_PreStep) = *(m_EAxisPMLFuncArray_Cyl3D[i].m_Elec);

        *(m_EAxisPMLFuncArray_Cyl3D[i].m_PE1) = m_EAxisPMLFuncArray_Cyl3D[i].b[0] * m_EAxisPMLFuncArray_Cyl3D[i].m_pe1 +
                                                m_EAxisPMLFuncArray_Cyl3D[i].a[0] * m_EAxisPMLFuncArray_Cyl3D[i].m_DualContour * m_EAxisPMLFuncArray_Cyl3D[i].m_Contour1;

        *(m_EAxisPMLFuncArray_Cyl3D[i].m_Elec) = m_EAxisPMLFuncArray_Cyl3D[i].m_C0 * m_EAxisPMLFuncArray_Cyl3D[i].m_elec +
                                                 m_EAxisPMLFuncArray_Cyl3D[i].m_C2 * m_EAxisPMLFuncArray_Cyl3D[i].invKappa[0] * m_EAxisPMLFuncArray_Cyl3D[i].m_DualContour *
                                                     m_EAxisPMLFuncArray_Cyl3D[i].m_Contour +
                                                 m_EAxisPMLFuncArray_Cyl3D[i].m_C2 * m_EAxisPMLFuncArray_Cyl3D[i].m_pe1;

        *(m_EAxisPMLFuncArray_Cyl3D[i].m_BE) = (1.0 + 0.5 * damping_scale) * (*(m_EAxisPMLFuncArray_Cyl3D[i].m_Elec)) -
                                               0.5 * (*(m_EAxisPMLFuncArray_Cyl3D[i].m_Elec_PreStep)) + 0.5 * (1 - damping_scale) * (*(m_EAxisPMLFuncArray_Cyl3D[i].m_AE));
    }
#else
    m_ECPMLFields_Cyl3D->Advance_SI_Damping(m_bb[0], m_Damping); // 此处非常多不一样的
#endif
}