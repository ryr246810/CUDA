#include "SI_SC_Matrix_Elec_CPML_Func_Cyl3D.cuh"
#include "SI_SC_Matrix_EMFields_Cyl3D.hxx"
#include "BaseFunctionDefine.hxx"
#include "PhysConsts.hxx"
#include "SI_SC_Matrix_CPML_Equation.hxx"

void 
SI_SC_Matrix_EMFields_Cyl3D::
    Build_Elec_CPML_Func(const bool doDamping,
        const Standard_Real& dt,
        const Standard_Integer& dynEIndex,
        const Standard_Integer& PreIndex,
        const Standard_Integer& AEIndex,
        const Standard_Integer& BEIndex,
        const Standard_Integer& dynHIndex,
        const Standard_Integer& PE1Index,
        const Standard_Integer& PE2Index)
{
    // 1.0 Build PML Ezr Func Data
    vector<GridEdgeData*> &CPML_Ezredge = m_ECPMLFields_Cyl3D->m_EdgeElecFlds_Cyl3D->GetDatas();
    int nEdge = CPML_Ezredge.size();
    m_CPML_EzrDatasNum = nEdge;
    m_EzrPMLFuncArray_Cyl3D = new FIT_Elec_PML_Func_Cyl3D[nEdge];

    for(int i = 0; i < nEdge; ++i){
        GridEdgeData *currEdge = CPML_Ezredge[i];
        Standard_Real *currDataPtr = currEdge->GetPhysDataPtr(0);

        m_EzrPMLFuncArray_Cyl3D[i].m_Elec = currDataPtr + dynEIndex;
        m_EzrPMLFuncArray_Cyl3D[i].m_Elec_Ptr_Offset = m_EzrPMLFuncArray_Cyl3D[i].m_Elec - m_EzrDatasPtr;

        m_EzrPMLFuncArray_Cyl3D[i].m_Elec_PreStep = currDataPtr + PreIndex;
        m_EzrPMLFuncArray_Cyl3D[i].m_Elec_PreStep_Ptr_Offset = m_EzrPMLFuncArray_Cyl3D[i].m_Elec_PreStep - m_EzrDatasPtr;

        m_EzrPMLFuncArray_Cyl3D[i].m_BE = currDataPtr + BEIndex;
        m_EzrPMLFuncArray_Cyl3D[i].m_BE_Ptr_Offset = m_EzrPMLFuncArray_Cyl3D[i].m_BE - m_EzrDatasPtr;

        m_EzrPMLFuncArray_Cyl3D[i].m_AE = currDataPtr + AEIndex;
        m_EzrPMLFuncArray_Cyl3D[i].m_AE_Ptr_Offset = m_EzrPMLFuncArray_Cyl3D[i].m_AE - m_EzrDatasPtr;

        m_EzrPMLFuncArray_Cyl3D[i].m_PE1 = currDataPtr + PE1Index;
        m_EzrPMLFuncArray_Cyl3D[i].m_PE1_Ptr_Offset = m_EzrPMLFuncArray_Cyl3D[i].m_PE1 - m_EzrDatasPtr;

        m_EzrPMLFuncArray_Cyl3D[i].m_PE2 = currDataPtr + PE2Index;
        m_EzrPMLFuncArray_Cyl3D[i].m_PE2_Ptr_Offset = m_EzrPMLFuncArray_Cyl3D[i].m_PE2 - m_EzrDatasPtr;

        // for dual contour    
        const vector<T_Element> &theOutLineTElems = currEdge->GetSharedTFace();// GridFaceData
        int nb = theOutLineTElems.size();

        Standard_Integer dir0 = currEdge->GetDir();
        Standard_Integer dir1 = TwoDim_DirBump(dir0, 1);

        Standard_Real a = 0.0;
        Standard_Real b = 0.0;
        Standard_Real Kappa = currEdge->GetPMLKappa(dir1);
        Matrix_Get_a_b_SI_SC(currEdge, dir1, dt, a, b);

        m_EzrPMLFuncArray_Cyl3D[i].a[0] = a;
        m_EzrPMLFuncArray_Cyl3D[i].b[0] = b;
        m_EzrPMLFuncArray_Cyl3D[i].invKappa[0] = 1 / Kappa;

        Standard_Real CC = 1.0 / (currEdge->GetEpsilon() + 0.5 * currEdge->GetSigma() * dt);
        m_EzrPMLFuncArray_Cyl3D[i].m_C0 = (currEdge->GetEpsilon() - 0.5 * currEdge->GetSigma() * dt) * CC;
        m_EzrPMLFuncArray_Cyl3D[i].m_C2 = dt * CC;

        for(Standard_Size j = 0; j < nb; ++j){
            GridFaceData *curFace = (GridFaceData *)theOutLineTElems[j].GetData();

            m_EzrPMLFuncArray_Cyl3D[i].m_Mag[j] = curFace->GetPhysDataPtr(dynHIndex);
            m_EzrPMLFuncArray_Cyl3D[i].m_Mag_Ptr_Offset[j] = m_EzrPMLFuncArray_Cyl3D[i].m_Mag[j] - m_MphiDatasPtr;
            m_EzrPMLFuncArray_Cyl3D[i].m_Curl1[j] = curFace->GetDualGeomDim() * theOutLineTElems[j].GetRelatedDir() / currEdge->GetDualGeomDim();
        }

        const vector<T_Element> &theNearMEdges = currEdge->GetNearMEdges();
        nb = theNearMEdges.size();
        for(Standard_Size j = 0; j < nb; ++j){
            GridFaceData *curFace = (GridFaceData *)theNearMEdges[j].GetData();

            m_EzrPMLFuncArray_Cyl3D[i].m_MagNear[j] = curFace->GetSweptPhysDataPtr(dynHIndex);
            m_EzrPMLFuncArray_Cyl3D[i].m_MagNear_Ptr_Offset[j] = m_EzrPMLFuncArray_Cyl3D[i].m_MagNear[j] - m_MzrDatasPtr;
            m_EzrPMLFuncArray_Cyl3D[i].m_Curl1Near[j] = curFace->GetDualGeomDim_Near() * theNearMEdges[j].GetRelatedDir() / currEdge->GetDualGeomDim();
        }

        m_EzrPMLFuncArray_Cyl3D[i].m_CurlP1 = m_EzrPMLFuncArray_Cyl3D[i].m_C2 / Kappa;
        m_EzrPMLFuncArray_Cyl3D[i].CheckMagDatas();
    }

    // 2.0 Build PML Ephi Func Data
    vector<GridVertexData*> &CPML_EphiVertex = m_ECPMLFields_Cyl3D->m_SweptEdgeElecFlds_Cyl3D->GetDatas();
    Standard_Size nVertex = CPML_EphiVertex.size();
    m_CPML_EphiDatasNum = nVertex;
    m_EphiPMLFuncArray_Cyl3D = new FIT_Elec_PML_Func_Cyl3D[nVertex];
    for(int i = 0; i < nVertex; ++i){
        GridVertexData *currVertex = CPML_EphiVertex[i];
        Standard_Real *currDataPtr = currVertex->GetSweptPhysDataPtr(0);

        m_EphiPMLFuncArray_Cyl3D[i].m_Elec = currDataPtr + dynEIndex;
        m_EphiPMLFuncArray_Cyl3D[i].m_Elec_Ptr_Offset = m_EphiPMLFuncArray_Cyl3D[i].m_Elec - m_EphiDatasPtr;

        m_EphiPMLFuncArray_Cyl3D[i].m_Elec_PreStep = currDataPtr + PreIndex;
        m_EphiPMLFuncArray_Cyl3D[i].m_Elec_PreStep_Ptr_Offset = m_EphiPMLFuncArray_Cyl3D[i].m_Elec_PreStep - m_EphiDatasPtr;

        m_EphiPMLFuncArray_Cyl3D[i].m_BE = currDataPtr + BEIndex; // Elec_Damped
        m_EphiPMLFuncArray_Cyl3D[i].m_BE_Ptr_Offset = m_EphiPMLFuncArray_Cyl3D[i].m_BE - m_EphiDatasPtr;

        m_EphiPMLFuncArray_Cyl3D[i].m_AE = currDataPtr + AEIndex;
        m_EphiPMLFuncArray_Cyl3D[i].m_AE_Ptr_Offset = m_EphiPMLFuncArray_Cyl3D[i].m_AE - m_EphiDatasPtr;

        m_EphiPMLFuncArray_Cyl3D[i].m_PE1 = currDataPtr + PE1Index;
        m_EphiPMLFuncArray_Cyl3D[i].m_PE1_Ptr_Offset = m_EphiPMLFuncArray_Cyl3D[i].m_PE1 - m_EphiDatasPtr;

        m_EphiPMLFuncArray_Cyl3D[i].m_PE2 = currDataPtr + PE2Index;
        m_EphiPMLFuncArray_Cyl3D[i].m_PE2_Ptr_Offset = m_EphiPMLFuncArray_Cyl3D[i].m_PE2 - m_EphiDatasPtr;

        // for dual contour
        const vector<T_Element> &theOutLineTElems = currVertex->GetSharedTDFaces(); // GridEdgeData
        int nb = theOutLineTElems.size();

        Standard_Integer dir1 = ThreeDim_DirBump(2, 1);
        Standard_Real a1 = 0.0;
        Standard_Real b1 = 0.0;
        Standard_Real Kappa1 = currVertex->GetPMLKappa(dir1);
        Matrix_Get_a_b_SI_SC(currVertex, dir1, dt, a1, b1);
        m_EphiPMLFuncArray_Cyl3D[i].a[0] = a1;
        m_EphiPMLFuncArray_Cyl3D[i].b[0] = b1;
        m_EphiPMLFuncArray_Cyl3D[i].invKappa[0] = 1 / Kappa1;

        Standard_Integer dir2 = ThreeDim_DirBump(2, 2);
        Standard_Real a2 = 0.0;
        Standard_Real b2 = 0.0;
        Standard_Real Kappa2 = currVertex->GetPMLKappa(dir2);
        Matrix_Get_a_b_SI_SC(currVertex, dir2, dt, a2, b2);
        m_EphiPMLFuncArray_Cyl3D[i].a[1] = a2;
        m_EphiPMLFuncArray_Cyl3D[i].b[1] = b2;
        m_EphiPMLFuncArray_Cyl3D[i].invKappa[1] = 1 / Kappa2;

        Standard_Real CC = 1.0 / (currVertex->GetEpsilon() + 0.5 * currVertex->GetSigma() * dt);
        m_EphiPMLFuncArray_Cyl3D[i].m_C0 = (currVertex->GetEpsilon() - 0.5 * currVertex->GetSigma() * dt) * CC;
        m_EphiPMLFuncArray_Cyl3D[i].m_C2 = dt * CC;

        for(Standard_Size j = 0; j < nb; ++j){
            GridEdgeData *curEdge = (GridEdgeData*)theOutLineTElems[j].GetData();
            Standard_Integer currEdgeDir = curEdge->GetDir();
            Standard_Integer currRelativeDir = theOutLineTElems[j].GetRelatedDir();

            m_EphiPMLFuncArray_Cyl3D[i].m_Mag[j] = curEdge->GetSweptPhysDataPtr(dynHIndex);
            m_EphiPMLFuncArray_Cyl3D[i].m_Mag_Ptr_Offset[j] = m_EphiPMLFuncArray_Cyl3D[i].m_Mag[j] - m_MzrDatasPtr;
            m_EphiPMLFuncArray_Cyl3D[i].m_Curl1[j] = curEdge->GetDualSweptGeomDim() * currRelativeDir * 
                                                    (TwoDim_DirBump(currEdgeDir, 1) - ThreeDim_DirBump(2, 1)) / 
                                                    (ThreeDim_DirBump(2, 2) - ThreeDim_DirBump(2, 1)) / currVertex->GetDualSweptGeomDim();
            m_EphiPMLFuncArray_Cyl3D[i].m_Curl2[j] = curEdge->GetDualSweptGeomDim() * currRelativeDir * 
                                                    (TwoDim_DirBump(currEdgeDir, 1) - ThreeDim_DirBump(2, 2)) / 
                                                    (ThreeDim_DirBump(2, 1) - ThreeDim_DirBump(2, 2)) / currVertex->GetDualSweptGeomDim();
        }

        m_EphiPMLFuncArray_Cyl3D[i].m_CurlP1 = m_EphiPMLFuncArray_Cyl3D[i].m_C2 / Kappa1;
        m_EphiPMLFuncArray_Cyl3D[i].m_CurlP2 = m_EphiPMLFuncArray_Cyl3D[i].m_C2 / Kappa2;
        m_EphiPMLFuncArray_Cyl3D[i].CheckMagDatas();
    }

    // 3.0 Build PML EAxis Func Data
    vector<GridEdgeData*> &CPML_EAxisedge = m_ECPMLFields_Cyl3D->m_EdgeElecFlds_Axis_Cyl3D->GetDatas();
    nEdge = CPML_EAxisedge.size();
    int Phi_Num = Get_Phi_Num();
    m_EAxisPMLFuncArray_Cyl3D = new FIT_Elec_PML_Func_Cyl3D[nEdge];
    m_CPML_EAxisDatasNum = nEdge;
    int idxMax = nEdge / Phi_Num;

    for(int i = 0; i < nEdge; ++i){
        GridEdgeData *currEdge = CPML_EAxisedge[i];
        Standard_Real *currDataPtr = currEdge->GetPhysDataPtr(0);

        m_EAxisPMLFuncArray_Cyl3D[i].m_Elec = currDataPtr + dynEIndex;
        m_EAxisPMLFuncArray_Cyl3D[i].m_Elec_Ptr_Offset = m_EAxisPMLFuncArray_Cyl3D[i].m_Elec - m_EzrDatasPtr;

        m_EAxisPMLFuncArray_Cyl3D[i].m_Elec_PreStep = currDataPtr + PreIndex;
        m_EAxisPMLFuncArray_Cyl3D[i].m_Elec_PreStep_Ptr_Offset = m_EAxisPMLFuncArray_Cyl3D[i].m_Elec_PreStep - m_EzrDatasPtr;

        m_EAxisPMLFuncArray_Cyl3D[i].m_BE = currDataPtr + BEIndex;
        m_EAxisPMLFuncArray_Cyl3D[i].m_BE_Ptr_Offset = m_EAxisPMLFuncArray_Cyl3D[i].m_BE - m_EzrDatasPtr;

        m_EAxisPMLFuncArray_Cyl3D[i].m_AE = currDataPtr + AEIndex;
        m_EAxisPMLFuncArray_Cyl3D[i].m_AE_Ptr_Offset = m_EAxisPMLFuncArray_Cyl3D[i].m_AE - m_EzrDatasPtr;

        m_EAxisPMLFuncArray_Cyl3D[i].m_PE1 = currDataPtr + PE1Index;
        m_EAxisPMLFuncArray_Cyl3D[i].m_PE1_Ptr_Offset = m_EAxisPMLFuncArray_Cyl3D[i].m_PE1 - m_EzrDatasPtr;

        m_EAxisPMLFuncArray_Cyl3D[i].m_PE2 = currDataPtr + PE2Index;
        m_EAxisPMLFuncArray_Cyl3D[i].m_PE2_Ptr_Offset = m_EAxisPMLFuncArray_Cyl3D[i].m_PE2 - m_EzrDatasPtr;

        // for dual contour
        const vector<T_Element> &theOutLineTElems = currEdge->GetSharedTFace();
        int nb = theOutLineTElems.size();

        Standard_Integer dir0 = currEdge->GetDir();
        Standard_Integer dir1 = TwoDim_DirBump(dir0, 1);

        Standard_Real a = 0.0;
        Standard_Real b = 0.0;
        Standard_Real Kappa = currEdge->GetPMLKappa(dir1);
        Matrix_Get_a_b_SI_SC(currEdge, dir1, dt, a, b);
        
        m_EAxisPMLFuncArray_Cyl3D[i].a[0] = a;
        m_EAxisPMLFuncArray_Cyl3D[i].b[0] = b;
        m_EAxisPMLFuncArray_Cyl3D[i].invKappa[0] = Kappa;

        Standard_Real CC = 1.0 / (currEdge->GetEpsilon() + 0.5 * currEdge->GetSigma() * dt);
        m_EAxisPMLFuncArray_Cyl3D[i].m_C0 = (currEdge->GetEpsilon() - 0.5 * currEdge->GetSigma() * dt) * CC;
        m_EAxisPMLFuncArray_Cyl3D[i].m_C2 = dt * CC;
        m_EAxisPMLFuncArray_Cyl3D[i].m_Contour = currEdge->GetDualGeomDim();

        for(Standard_Size j = 0; j < nb; ++j){
            GridFaceData *curFace = (GridFaceData *)theOutLineTElems[j].GetData();
            m_EAxisPMLFuncArray_Cyl3D[i].m_Mag[j] = curFace->GetPhysDataPtr(dynHIndex);
            m_EAxisPMLFuncArray_Cyl3D[i].m_Mag_Ptr_Offset[j] = m_EAxisPMLFuncArray_Cyl3D[i].m_Mag[j] - m_MphiDatasPtr;
            m_EAxisPMLFuncArray_Cyl3D[i].m_Curl1[j] = curFace->GetDualGeomDim() * theOutLineTElems[j].GetRelatedDir();
        }
        m_EAxisPMLFuncArray_Cyl3D[i].CheckMagDatas();
    }

    for(int i = 0; i < idxMax; ++i){
        Standard_Real C0 = 0;
        Standard_Real C2 = 0;
        Standard_Real DualGeomDim = 0;
        Standard_Real DualGeomDim1 = 0;
        Standard_Real Kappa1 = 0;
        Standard_Real a = 0.0;
        Standard_Real b = 0.0;
        for(int j = 0; j < Phi_Num; ++j){
            Standard_Size index = idxMax * j + i;
            C0 += m_EAxisPMLFuncArray_Cyl3D[index].m_C0;
            C2 += m_EAxisPMLFuncArray_Cyl3D[index].m_C2;
            DualGeomDim += m_EAxisPMLFuncArray_Cyl3D[index].m_Contour;
            DualGeomDim1 += 2 * m_EAxisPMLFuncArray_Cyl3D[index].m_Contour;
            Kappa1 += m_EAxisPMLFuncArray_Cyl3D[index].invKappa[0];
            a += m_EAxisPMLFuncArray_Cyl3D[index].a[0];
            b += m_EAxisPMLFuncArray_Cyl3D[index].b[0];
        }
        C0 /= Phi_Num;
        C2 /= Phi_Num;
        a  /= Phi_Num;
        b  /= Phi_Num;
        Kappa1 /= Phi_Num;
        for(int j = 0; j < Phi_Num; ++j){
            Standard_Size index = idxMax * j + i;
            m_EAxisPMLFuncArray_Cyl3D[index].m_C0 = C0;
            m_EAxisPMLFuncArray_Cyl3D[index].m_C2 = C2;
            m_EAxisPMLFuncArray_Cyl3D[index].invKappa[0] = 1 / Kappa1;
            m_EAxisPMLFuncArray_Cyl3D[index].m_Contour = 1 / DualGeomDim;
            m_EAxisPMLFuncArray_Cyl3D[index].m_Contour1 = 1 / DualGeomDim1;
            m_EAxisPMLFuncArray_Cyl3D[index].a[0] = a;
            m_EAxisPMLFuncArray_Cyl3D[index].b[0] = b;
        }
    }
}