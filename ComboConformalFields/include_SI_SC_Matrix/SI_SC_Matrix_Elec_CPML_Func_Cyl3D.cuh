#ifndef _SI_SC_Matrix_Elec_CPML_Func_Cyl3D_HeaderFile
#define _SI_SC_Matrix_Elec_CPML_Func_Cyl3D_HeaderFile

/*
    在2.5维场推进部分进行了简化：
        计算Ez时忽略了其周围的Br，
        计算Er时忽略了其周围的Bz，
        计算Bz时忽略了其周围的Er，
        计算Br时忽略了其周围的Ez。
        
    考虑到2.5维和3维相比，3维凡是涉及到相邻z-r切面物理信息的电场参量都需要对其推进方法进行改进，
    即Ez和Er、Bz和Br的推进方式需要改进：
        3维坐标系中，应采用完整的场推进形式，需要考虑获得第PhiIndex和[hiIndex-1个z-r剖面的
        磁场参量，通过计算磁场沿对偶网格边的积分完成三维柱坐标系电场推进。
        计算轴上奇异电场Ez时，通过将其周围所有z-r剖面的磁场沿对偶网格的积分进行累积求和，实现奇异电场的推进。
        与电场类似，需要考虑获得第PhiIndex和PhiIndex+1个z-r剖面的电场参量，计算电场与初始网格的线积分以实现磁场推进。
*/

#include <math.h>
#include <string.h>
#include "Standard_TypeDefine.hxx"
#include "CUDAHeader.cuh"

class FIT_Elec_PML_Func_Cyl3D
{
public:
    HOST_DEVICE FIT_Elec_PML_Func_Cyl3D();
    HOST_DEVICE ~FIT_Elec_PML_Func_Cyl3D();

    HOST_DEVICE void Init();

public:
    Standard_Real* m_Elec;
    Standard_Integer m_Elec_Ptr_Offset;

    Standard_Real* m_Current;
    Standard_Integer m_Current_Ptr_Offset;

    Standard_Real* m_Elec_PreStep;
    Standard_Integer m_Elec_PreStep_Ptr_Offset;

    Standard_Real* m_BE;
    Standard_Integer m_BE_Ptr_Offset;

    Standard_Real* m_AE;
    Standard_Integer m_AE_Ptr_Offset;

    Standard_Real* m_PE1;
    Standard_Integer m_PE1_Ptr_Offset;
    
    Standard_Real* m_PE2;
    Standard_Integer m_PE2_Ptr_Offset;

    Standard_Real m_elec;
    Standard_Real m_pe1;

    Standard_Real a[2];
    Standard_Real b[2];
    Standard_Real invKappa[2];

    Standard_Real m_C0;
    Standard_Real m_C2;
    Standard_Real m_Contour;
    Standard_Real m_Contour1;
    Standard_Real m_DualContour;

    Standard_Real* m_Mag[4];
    Standard_Integer m_Mag_Ptr_Offset[4];
    Standard_Real* m_MagNear[4];
    Standard_Integer m_MagNear_Ptr_Offset[4];
    
    Standard_Real m_Curl1[4];
    Standard_Real m_Curl1Near[4];
    Standard_Real m_CurlP1;
    Standard_Real m_Curl2[4];
    Standard_Real m_CurlP2;

public:
    HOST_DEVICE void CheckMagDatas();

    HOST_DEVICE void ComputContour1(Standard_Real& result);
    DEVICE void d_ComputContour1(Standard_Real& result, Standard_Real* MagArray);

    HOST_DEVICE void ComputContour11(Standard_Real& result);
    DEVICE void d_ComputContour11(Standard_Real& result, Standard_Real* MagArray);

    HOST_DEVICE Standard_Real ComputContourAxis();
    DEVICE Standard_Real d_ComputContourAxis(Standard_Real* MagArray);

    HOST_DEVICE void ComputContour1Near(Standard_Real& result);
    DEVICE void d_ComputContour1Near(Standard_Real& result, Standard_Real* MagArray);

    HOST_DEVICE void ComputContour2(Standard_Real& result);
    DEVICE void d_ComputContour2(Standard_Real& result, Standard_Real* MagArray);

    HOST_DEVICE void ComputP1(const Standard_Real& contour1);
    DEVICE void d_ComputP1(const Standard_Real contour1, Standard_Real* ElecArray);
    
    HOST_DEVICE void ComputP2(const Standard_Real& contour2);
    DEVICE void d_ComputP2(const Standard_Real contour2, Standard_Real* ElecArray);

    HOST_DEVICE void Advance_Damping_A(const Standard_Real& theta);

    HOST_DEVICE void Advance_Damping_B(const Standard_Real& theta);
    
    HOST_DEVICE void Advance_Damping_A_2(const Standard_Real& theta);

    HOST_DEVICE void Advance_Damping_B_2(const Standard_Real& theta);

    HOST_DEVICE void Advance_Damping_1(const Standard_Real& theta);
    DEVICE void d_Advance_Damping_1(const Standard_Real theta, Standard_Real* ElecArray);
    
    HOST_DEVICE void Advance_Damping_2(const Standard_Real& theta);
    DEVICE void d_Advance_Damping_2(const Standard_Real theta, Standard_Real* ElecArray);

    HOST_DEVICE void Advance_Damping1(const Standard_Real& theta);

    HOST_DEVICE void Advance_Damping2(const Standard_Real& theta);

    HOST_DEVICE void AdvanceEzr();
    DEVICE void d_AdvanceEzr(Standard_Real* EzrDatasPtr, Standard_Real* MphiDatasPtr,
        Standard_Real* MzrDatasPtr);
    
    HOST_DEVICE void AdvanceEphi();
    DEVICE void d_AdvanceEphi(Standard_Real* EphiDatasPtr, Standard_Real* MzrDatasPtr);

    HOST_DEVICE void AdvanceEzr_Damping(const Standard_Real& theta);
    DEVICE void d_AdvanceEzr_Damping(const Standard_Real theta, Standard_Real* EzrDatasPtr,
        Standard_Real* MphiDatasPtr, Standard_Real* MzrDatasPtr);

    HOST_DEVICE void AdvanceEphi_Damping(const Standard_Real& theta);
    DEVICE void d_AdvanceEphi_Damping(const Standard_Real theta, Standard_Real* EphiDatasPtr,
        Standard_Real* MzrDatasPtr);
};

#endif