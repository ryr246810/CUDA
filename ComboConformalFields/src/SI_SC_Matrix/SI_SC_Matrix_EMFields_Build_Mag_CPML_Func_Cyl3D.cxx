#include "SI_SC_Matrix_Mag_CPML_Func_Cyl3D.cuh"
#include "SI_SC_Matrix_EMFields_Cyl3D.hxx"
#include "BaseFunctionDefine.hxx"
#include "PhysConsts.hxx"
#include "SI_SC_Matrix_CPML_Equation.hxx"

void
SI_SC_Matrix_EMFields_Cyl3D::
    Build_Mag_CPML_Func(const Standard_Real &dt,
                        const Standard_Integer &dynHIndex,
                        const Standard_Integer &PM1Index,
                        const Standard_Integer &PM2Index,
                        const Standard_Integer &dynEIndex)
{
    // 1.0 Build PML Mzr Func Data
    vector<GridEdgeData*> &CPML_MzrEdge = m_MCPMLFields_Cyl3D->m_SweptFaceMagFlds_Cyl3D->GetDatas();
    Standard_Size nEdge = CPML_MzrEdge.size();
    m_CPML_MzrDatasNum = nEdge;

    m_MzrPMLFuncArray_Cyl3D = new FIT_Mag_PML_Func_Cyl3D[nEdge];
    for(int i = 0; i < nEdge; ++i){
        GridEdgeData *currEdge = CPML_MzrEdge[i];
        Standard_Real *currDataPtr = currEdge->GetSweptPhysDataPtr(0);

        m_MzrPMLFuncArray_Cyl3D[i].m_Mag = currDataPtr + dynHIndex;
        m_MzrPMLFuncArray_Cyl3D[i].m_Mag_Ptr_Offset = m_MzrPMLFuncArray_Cyl3D[i].m_Mag - m_MzrDatasPtr;

        m_MzrPMLFuncArray_Cyl3D[i].m_PM1 = currDataPtr + PM1Index;
        m_MzrPMLFuncArray_Cyl3D[i].m_PM1_Ptr_Offset = m_MzrPMLFuncArray_Cyl3D[i].m_PM1 - m_MzrDatasPtr;

        m_MzrPMLFuncArray_Cyl3D[i].m_PM2 = currDataPtr + PM2Index;
        m_MzrPMLFuncArray_Cyl3D[i].m_PM2_Ptr_Offset = m_MzrPMLFuncArray_Cyl3D[i].m_PM2 - m_MzrDatasPtr;

        // for dual contour
        const vector<T_Element> &theOutLineTElems = currEdge->GetOutLineDTEdges();
        int nb = theOutLineTElems.size();

        Standard_Integer dir0 = currEdge->GetDir();
        Standard_Integer dir1 = TwoDim_DirBump(TwoDim_DirBump(dir0, 1), 1);

        Standard_Real a = 0.0;
        Standard_Real b = 0.0;
        Standard_Real Kappa = currEdge->GetPMLKappa(dir1);
        Matrix_Get_a_b_SI_SC(currEdge, dir1, dt, a, b);

        m_MzrPMLFuncArray_Cyl3D[i].a[0] = a;
        m_MzrPMLFuncArray_Cyl3D[i].b[0] = b;
        m_MzrPMLFuncArray_Cyl3D[i].invKappa[0] = 1 / Kappa;

        m_MzrPMLFuncArray_Cyl3D[i].m_C0 = dt / currEdge->GetMu();
        m_MzrPMLFuncArray_Cyl3D[i].m_CurlP1 = m_MzrPMLFuncArray_Cyl3D[i].m_C0 / Kappa;

        for(Standard_Size j = 0; j < nb; ++j){
            GridVertexData *curVertex = (GridVertexData*)theOutLineTElems[j].GetData();

            m_MzrPMLFuncArray_Cyl3D[i].m_Elec[j] = curVertex->GetSweptPhysDataPtr(dynEIndex);
            m_MzrPMLFuncArray_Cyl3D[i].m_Elec_Ptr_Offset[j] = m_MzrPMLFuncArray_Cyl3D[i].m_Elec[j] - m_EphiDatasPtr;
            m_MzrPMLFuncArray_Cyl3D[i].m_Curl1[j] = curVertex->GetSweptGeomDim() * theOutLineTElems[j].GetRelatedDir() / currEdge->GetSweptGeomDim();
        }

        const vector<T_Element> &theNearEEdges = currEdge->GetNearEEdges();
        nb = theNearEEdges.size();
        for(Standard_Integer j = 0; j < nb; ++j){
            GridVertexData *curVertex = (GridVertexData*)theNearEEdges[j].GetData();

            m_MzrPMLFuncArray_Cyl3D[i].m_ElecNear[j] = curVertex->GetPhysDataPtr(dynEIndex);
            m_MzrPMLFuncArray_Cyl3D[i].m_ElecNear_Ptr_Offset[j] = m_MzrPMLFuncArray_Cyl3D[i].m_ElecNear[j] - m_EzrDatasPtr;
            m_MzrPMLFuncArray_Cyl3D[i].m_Curl1Near[j] = curVertex->GetSweptGeomDim_Near() * theNearEEdges[j].GetRelatedDir() / currEdge->GetSweptGeomDim();
        }

        m_MzrPMLFuncArray_Cyl3D[i].CheckElecDatas();
    }

    // 2.0 Build PML Mphi Func Data
    vector<GridFaceData*> &CPML_MphiFace = m_MCPMLFields_Cyl3D->m_FaceMagFlds_Cyl3D->GetDatas();
    Standard_Size nFace = CPML_MphiFace.size();
    m_CPML_MphiDatasNum = nFace;

    m_MphiPMLFuncArray_Cyl3D = new FIT_Mag_PML_Func_Cyl3D[nFace];
    for(int i = 0; i < nFace; ++i){
        GridFaceData *currFace = CPML_MphiFace[i];
        Standard_Real *currDataPtr = currFace->GetPhysDataPtr(0);

        m_MphiPMLFuncArray_Cyl3D[i].m_Mag = currDataPtr + dynHIndex;
        m_MphiPMLFuncArray_Cyl3D[i].m_Mag_Ptr_Offset = m_MphiPMLFuncArray_Cyl3D[i].m_Mag - m_MphiDatasPtr;

        m_MphiPMLFuncArray_Cyl3D[i].m_PM1 = currDataPtr + PM1Index;
        m_MphiPMLFuncArray_Cyl3D[i].m_PM1_Ptr_Offset = m_MphiPMLFuncArray_Cyl3D[i].m_PM1 - m_MphiDatasPtr;

        m_MphiPMLFuncArray_Cyl3D[i].m_PM2 = currDataPtr + PM2Index;
        m_MphiPMLFuncArray_Cyl3D[i].m_PM2_Ptr_Offset = m_MphiPMLFuncArray_Cyl3D[i].m_PM2 - m_MphiDatasPtr;

        // for dual contour
        const vector<T_Element> &theOutLineTElems = currFace->GetOutLineTEdge();
        Standard_Integer nb = theOutLineTElems.size();

        Standard_Integer dir1 = ThreeDim_DirBump(2, 1);
        Standard_Real a1 = 0.0;
        Standard_Real b1 = 0.0;
        Standard_Real Kappa1 = currFace->GetPMLKappa(dir1);
        Matrix_Get_a_b_SI_SC(currFace, dir1, dt, a1, b1);
        m_MphiPMLFuncArray_Cyl3D[i].a[0] = a1;
        m_MphiPMLFuncArray_Cyl3D[i].b[0] = b1;
        m_MphiPMLFuncArray_Cyl3D[i].invKappa[0] = 1 / Kappa1;

        Standard_Integer dir2 = ThreeDim_DirBump(2, 2);
        Standard_Real a2 = 0.0;
        Standard_Real b2 = 0.0;
        Standard_Real Kappa2 = currFace->GetPMLKappa(dir2);
        Matrix_Get_a_b_SI_SC(currFace, dir2, dt, a2, b2);
        m_MphiPMLFuncArray_Cyl3D[i].a[1] = a2;
        m_MphiPMLFuncArray_Cyl3D[i].b[1] = b2;
        m_MphiPMLFuncArray_Cyl3D[i].invKappa[1] = 1 / Kappa2;

        m_MphiPMLFuncArray_Cyl3D[i].m_C0 = dt / currFace->GetMu();
        m_MphiPMLFuncArray_Cyl3D[i].m_CurlP1 = m_MphiPMLFuncArray_Cyl3D[i].m_C0 / Kappa1;
        m_MphiPMLFuncArray_Cyl3D[i].m_CurlP2 = m_MphiPMLFuncArray_Cyl3D[i].m_C0 / Kappa2;

        for(Standard_Size j = 0; j < nb; ++j){
            GridEdgeData *curEdge = (GridEdgeData*)theOutLineTElems[j].GetData();
            Standard_Integer currEdgeDir = curEdge->GetDir();
            Standard_Integer currRelativeDir = theOutLineTElems[j].GetRelatedDir();

            m_MphiPMLFuncArray_Cyl3D[i].m_Elec[j] = curEdge->GetPhysDataPtr(dynEIndex);
            m_MphiPMLFuncArray_Cyl3D[i].m_Elec_Ptr_Offset[j] = m_MphiPMLFuncArray_Cyl3D[i].m_Elec[j] - m_EzrDatasPtr;
            m_MphiPMLFuncArray_Cyl3D[i].m_Curl1[j] = curEdge->GetGeomDim() * 
                                                     currRelativeDir *
                                                     (currEdgeDir - ThreeDim_DirBump(2,1)) / (ThreeDim_DirBump(2,2) - ThreeDim_DirBump(2,1))
                                                      / currFace->GetGeomDim();

            m_MphiPMLFuncArray_Cyl3D[i].m_Curl2[j] = curEdge->GetGeomDim() * 
                                                     currRelativeDir *
                                                     (currEdgeDir - ThreeDim_DirBump(2,2)) / (ThreeDim_DirBump(2,1) - ThreeDim_DirBump(2,2))
                                                      / currFace->GetGeomDim(); 
        }

        m_MphiPMLFuncArray_Cyl3D[i].CheckElecDatas();
    }
}