#include "SI_SC_Matrix_Elec_Func_Cyl3D.cuh"
#include "SI_SC_Matrix_EMFields_Cyl3D.hxx"

void
SI_SC_Matrix_EMFields_Cyl3D::
    Build_Elec_Func(const bool doDamping,
                    const Standard_Real &dt,
                    const Standard_Integer &dynEIndex,
                    const Standard_Integer &dynJIndex,
                    const Standard_Integer &PreIndex,
                    const Standard_Integer &AEIndex,
                    const Standard_Integer &BEIndex,
                    const Standard_Integer &dynHIndex)
{
    // 1.0 Build Ezr Func Data
    vector<GridEdgeData*> &Ezredge = m_ECntrFields_Cyl3D->m_EdgeElecFlds_Cyl3D->GetDatas();
    int nEdge = Ezredge.size();
    m_EzrFuncArray_Cyl3D = new FIT_Elec_Func_Cyl3D[nEdge];
    m_EzrDatasNum = nEdge;

    for(int i = 0; i < nEdge; ++i){
        GridEdgeData *currEdge = Ezredge[i];
        Standard_Real *currDataPtr = currEdge->GetPhysDataPtr(0);

        if(doDamping){
            m_EzrFuncArray_Cyl3D[i].m_Elec = currDataPtr + dynEIndex;
            m_EzrFuncArray_Cyl3D[i].m_Elec_Ptr_Offset = m_EzrFuncArray_Cyl3D[i].m_Elec - m_EzrDatasPtr;

            m_EzrFuncArray_Cyl3D[i].m_Current = currDataPtr + dynJIndex;
            m_EzrFuncArray_Cyl3D[i].m_Current_Ptr_Offset = m_EzrFuncArray_Cyl3D[i].m_Current - m_EzrDatasPtr;

            m_EzrFuncArray_Cyl3D[i].m_AE = currDataPtr + AEIndex;
            m_EzrFuncArray_Cyl3D[i].m_AE_Ptr_Offset = m_EzrFuncArray_Cyl3D[i].m_AE - m_EzrDatasPtr;

            m_EzrFuncArray_Cyl3D[i].m_BE = currDataPtr + BEIndex;
            m_EzrFuncArray_Cyl3D[i].m_BE_Ptr_Offset = m_EzrFuncArray_Cyl3D[i].m_BE - m_EzrDatasPtr;

            m_EzrFuncArray_Cyl3D[i].m_Elec_PreStep = currDataPtr + PreIndex;
            m_EzrFuncArray_Cyl3D[i].m_Elec_PreStep_Ptr_Offset = m_EzrFuncArray_Cyl3D[i].m_Elec_PreStep - m_EzrDatasPtr;
        }
        else{
            m_EzrFuncArray_Cyl3D[i].m_Elec = currDataPtr + dynEIndex;
            m_EzrFuncArray_Cyl3D[i].m_Elec_Ptr_Offset = m_EzrFuncArray_Cyl3D[i].m_Elec - m_EzrDatasPtr;

            m_EzrFuncArray_Cyl3D[i].m_Current = currDataPtr + dynJIndex;
            m_EzrFuncArray_Cyl3D[i].m_Current_Ptr_Offset = m_EzrFuncArray_Cyl3D[i].m_Current - m_EzrDatasPtr;
        }

        // for dual contour

        const vector<T_Element> &theOutLineTElems = currEdge->GetSharedTFace();
        int nb = theOutLineTElems.size();

        for(Standard_Size j = 0; j < nb; ++j){
            GridFaceData *curFace = (GridFaceData*)theOutLineTElems[j].GetData();

            m_EzrFuncArray_Cyl3D[i].m_Mag[j] = curFace->GetPhysDataPtr(dynHIndex);
            m_EzrFuncArray_Cyl3D[i].m_Mag_Ptr_Offset[j] = m_EzrFuncArray_Cyl3D[i].m_Mag[j] - m_MphiDatasPtr;
            m_EzrFuncArray_Cyl3D[i].m_Curl[j] = curFace->GetDualGeomDim() * theOutLineTElems[j].GetRelatedDir() / currEdge->GetDualGeomDim();
        }

        const vector<T_Element> &theNearMEdges = currEdge->GetNearMEdges();
        nb = theNearMEdges.size();
        for(Standard_Size j = 0; j < nb; ++j){
            GridFaceData *curFace = (GridFaceData*)theNearMEdges[j].GetData();

            m_EzrFuncArray_Cyl3D[i].m_MagNear[j] = curFace->GetSweptPhysDataPtr(dynHIndex);
            m_EzrFuncArray_Cyl3D[i].m_MagNear_Ptr_Offset[j] = m_EzrFuncArray_Cyl3D[i].m_MagNear[j] - m_MzrDatasPtr;
            m_EzrFuncArray_Cyl3D[i].m_CurlNear[j] = curFace->GetDualGeomDim_Near() * theNearMEdges[j].GetRelatedDir() / currEdge->GetDualGeomDim();
        }

        m_EzrFuncArray_Cyl3D[i].CheckMagDatas();

        Standard_Real CC = 1.0 / (currEdge->GetEpsilon() + 0.5 * currEdge->GetSigma() * dt);
        m_EzrFuncArray_Cyl3D[i].m_C0 = (currEdge->GetEpsilon() - 0.5 * currEdge->GetSigma() * dt) * CC;
        m_EzrFuncArray_Cyl3D[i].m_C2 = dt * CC;
        m_EzrFuncArray_Cyl3D[i].m_C3 = m_EzrFuncArray_Cyl3D[i].m_C2 / currEdge->GetDualGeomDim();
    }

    // 2.0 Build Ephi Func Data
    vector<GridVertexData*> &EphiVertex = m_ECntrFields_Cyl3D->m_SweptEdgeElecFlds_Cyl3D->GetDatas();
    Standard_Size nVertex = EphiVertex.size();

    m_EphiDatasNum = nVertex;
    m_EphiFuncArray_Cyl3D = new FIT_Elec_Func_Cyl3D[nVertex];
    for(int i = 0; i < nVertex; ++i){
        GridVertexData *currVertex = EphiVertex[i];
        Standard_Real *currDataPtr = currVertex->GetSweptPhysDataPtr(0);

        if(doDamping){
            m_EphiFuncArray_Cyl3D[i].m_Elec = currDataPtr + dynEIndex;
            m_EphiFuncArray_Cyl3D[i].m_Elec_Ptr_Offset = m_EphiFuncArray_Cyl3D[i].m_Elec - m_EphiDatasPtr;

            m_EphiFuncArray_Cyl3D[i].m_Current = currDataPtr + dynJIndex;
            m_EphiFuncArray_Cyl3D[i].m_Current_Ptr_Offset = m_EphiFuncArray_Cyl3D[i].m_Current - m_EphiDatasPtr;

            m_EphiFuncArray_Cyl3D[i].m_AE = currDataPtr + AEIndex;
            m_EphiFuncArray_Cyl3D[i].m_AE_Ptr_Offset = m_EphiFuncArray_Cyl3D[i].m_AE - m_EphiDatasPtr;

            m_EphiFuncArray_Cyl3D[i].m_BE = currDataPtr + BEIndex;
            m_EphiFuncArray_Cyl3D[i].m_BE_Ptr_Offset = m_EphiFuncArray_Cyl3D[i].m_BE - m_EphiDatasPtr;

            m_EphiFuncArray_Cyl3D[i].m_Elec_PreStep = currDataPtr + PreIndex;
            m_EphiFuncArray_Cyl3D[i].m_Elec_PreStep_Ptr_Offset = m_EphiFuncArray_Cyl3D[i].m_Elec_PreStep - m_EphiDatasPtr;
        }
        else{
            m_EphiFuncArray_Cyl3D[i].m_Elec = currDataPtr + dynEIndex;
            m_EphiFuncArray_Cyl3D[i].m_Elec_Ptr_Offset = m_EphiFuncArray_Cyl3D[i].m_Elec - m_EphiDatasPtr;

            m_EphiFuncArray_Cyl3D[i].m_Current = currDataPtr + dynJIndex;
            m_EphiFuncArray_Cyl3D[i].m_Current_Ptr_Offset = m_EphiFuncArray_Cyl3D[i].m_Current - m_EphiDatasPtr;
        }

        // for dual contour    
        const vector<T_Element> &theOutLineTElems = currVertex->GetSharedTDFaces();
        int nb = theOutLineTElems.size();

        for(Standard_Size j = 0; j < nb; ++j){
            GridEdgeData *curEdge = (GridEdgeData*)theOutLineTElems[j].GetData();
            
            m_EphiFuncArray_Cyl3D[i].m_Mag[j] = curEdge->GetSweptPhysDataPtr(dynHIndex);
            m_EphiFuncArray_Cyl3D[i].m_Mag_Ptr_Offset[j] = m_EphiFuncArray_Cyl3D[i].m_Mag[j] - m_MzrDatasPtr;
            m_EphiFuncArray_Cyl3D[i].m_Curl[j] = curEdge->GetDualSweptGeomDim() * theOutLineTElems[j].GetRelatedDir() / currVertex->GetDualSweptGeomDim();
        }
        m_EphiFuncArray_Cyl3D[i].CheckMagDatas();

        Standard_Real CC = 1.0 / (currVertex->GetEpsilon() + 0.5 * currVertex->GetSigma() * dt);
        m_EphiFuncArray_Cyl3D[i].m_C0 = (currVertex->GetEpsilon() - 0.5 * currVertex->GetSigma() * dt) * CC;
        m_EphiFuncArray_Cyl3D[i].m_C2 = dt * CC;
        m_EphiFuncArray_Cyl3D[i].m_C3 = m_EphiFuncArray_Cyl3D[i].m_C2 / currVertex->GetDualSweptGeomDim();
    }

    // 3.0 Build EAxis Func Data
    vector<GridEdgeData*> &EAxisedge = m_ECntrFields_Cyl3D->m_EdgeElecFlds_Axis_Cyl3D->GetDatas();
    nEdge = EAxisedge.size();
    int Phi_Num = Get_Phi_Num();
    m_EAxisFuncArray_Cyl3D = new FIT_Elec_Func_Cyl3D[nEdge];
    m_EAxisDatasNum = nEdge;
    int idxMax = nEdge / Phi_Num;

    for(int i = 0; i < nEdge; ++i){
        GridEdgeData *currEdge = EAxisedge[i];
        Standard_Real *currDataPtr = currEdge->GetPhysDataPtr(0);

        if(doDamping){
            m_EAxisFuncArray_Cyl3D[i].m_Elec = currDataPtr + dynEIndex;
            m_EAxisFuncArray_Cyl3D[i].m_Elec_Ptr_Offset = m_EAxisFuncArray_Cyl3D[i].m_Elec - m_EzrDatasPtr;

            m_EAxisFuncArray_Cyl3D[i].m_Current = currDataPtr + dynJIndex;
            m_EAxisFuncArray_Cyl3D[i].m_Current_Ptr_Offset = m_EAxisFuncArray_Cyl3D[i].m_Current - m_EzrDatasPtr;

            m_EAxisFuncArray_Cyl3D[i].m_AE = currDataPtr + AEIndex;
            m_EAxisFuncArray_Cyl3D[i].m_AE_Ptr_Offset = m_EAxisFuncArray_Cyl3D[i].m_AE - m_EzrDatasPtr;

            m_EAxisFuncArray_Cyl3D[i].m_BE = currDataPtr + BEIndex;
            m_EAxisFuncArray_Cyl3D[i].m_BE_Ptr_Offset = m_EAxisFuncArray_Cyl3D[i].m_BE - m_EzrDatasPtr;

            m_EAxisFuncArray_Cyl3D[i].m_Elec_PreStep = currDataPtr + PreIndex;
            m_EAxisFuncArray_Cyl3D[i].m_Elec_PreStep_Ptr_Offset = m_EAxisFuncArray_Cyl3D[i].m_Elec_PreStep - m_EzrDatasPtr;
        }
        else{
            m_EAxisFuncArray_Cyl3D[i].m_Elec = currDataPtr + dynEIndex;
            m_EAxisFuncArray_Cyl3D[i].m_Elec_Ptr_Offset = m_EAxisFuncArray_Cyl3D[i].m_Elec - m_EzrDatasPtr;

            m_EAxisFuncArray_Cyl3D[i].m_Current = currDataPtr + dynJIndex;
            m_EAxisFuncArray_Cyl3D[i].m_Current_Ptr_Offset = m_EAxisFuncArray_Cyl3D[i].m_Current - m_EzrDatasPtr;
        }

        // for dual contour
        const vector<T_Element> &theOutLineTElems = currEdge->GetSharedTFace();
        int nb = theOutLineTElems.size();

        for(Standard_Size j = 0; j < nb; ++j){
            GridFaceData *curFace = (GridFaceData *)theOutLineTElems[j].GetData();

            m_EAxisFuncArray_Cyl3D[i].m_Mag[j] = curFace->GetPhysDataPtr(dynHIndex);
            m_EAxisFuncArray_Cyl3D[i].m_Mag_Ptr_Offset[j] = m_EAxisFuncArray_Cyl3D[i].m_Mag[j] - m_MphiDatasPtr;
            m_EAxisFuncArray_Cyl3D[i].m_Curl[j] = curFace->GetDualGeomDim() * theOutLineTElems[j].GetRelatedDir();
        }

        m_EAxisFuncArray_Cyl3D[i].CheckMagDatas();

        Standard_Real CC = 1.0 / (currEdge->GetEpsilon() + 0.5 * currEdge->GetSigma() * dt);
        m_EAxisFuncArray_Cyl3D[i].m_C0 = (currEdge->GetEpsilon() - 0.5 * currEdge->GetSigma() * dt) * CC;
        m_EAxisFuncArray_Cyl3D[i].m_C2 = dt * CC;
        m_EAxisFuncArray_Cyl3D[i].m_C3 = currEdge->GetDualGeomDim();
        
    }

    for(int i = 0; i < idxMax; ++i){
        Standard_Real C0 = 0;
        Standard_Real C2 = 0;
        Standard_Real DualGeomDim = 0;
        for(int j = 0; j < Phi_Num; ++j){
            Standard_Size index = idxMax * j + i;
            C0 += m_EAxisFuncArray_Cyl3D[index].m_C0;
            C2 += m_EAxisFuncArray_Cyl3D[index].m_C2;
            DualGeomDim += m_EAxisFuncArray_Cyl3D[index].m_C3;
        }
        C0 /= Phi_Num;
        C2 /= Phi_Num;
        for(int j = 0; j < Phi_Num; ++j){
            Standard_Size index = idxMax * j + i;
            m_EAxisFuncArray_Cyl3D[index].m_C0 = C0;
            m_EAxisFuncArray_Cyl3D[index].m_C2 = C2;
            m_EAxisFuncArray_Cyl3D[index].m_C3 = 1 / DualGeomDim;
        }
    }

    // printf("the number of Ezredge is %d\n\n", m_EzrDatasNum);
    // printf("the number of EAxisedge is %d\n\n", m_EAxisDatasNum);

    // GridEdgeData *currEdge0 = Ezredge[0];
    // GridEdgeData *currEdge1 = Ezredge[m_EzrDatasNum - 1];
    // Standard_Real *currDataPtr = currEdge0->GetPhysDataPtr(0);
    // printf("the start address of Ezredge is %x\n\n", currDataPtr);
    // currDataPtr = currEdge1->GetPhysDataPtr(0);
    // printf("the end address of Ezredge is %x\n\n", currDataPtr);

    // GridEdgeData *currEdge2 = EAxisedge[0];
    // GridEdgeData *currEdge3 = EAxisedge[m_EAxisDatasNum - 1];

    // currDataPtr = currEdge2->GetPhysDataPtr(0);
    // printf("the start address of EAxisedge is %x\n\n", currDataPtr);
    // currDataPtr = currEdge3->GetPhysDataPtr(0);
    // printf("the end address of EAxisedge is %x\n\n", currDataPtr);

    // exit(1);
}
