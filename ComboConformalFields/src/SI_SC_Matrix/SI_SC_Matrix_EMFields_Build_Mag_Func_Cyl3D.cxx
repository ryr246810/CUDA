#include "SI_SC_Matrix_Mag_Func_Cyl3D.cuh"
#include "SI_SC_Matrix_EMFields_Cyl3D.hxx"
#include "GridEdge.hxx"

void
SI_SC_Matrix_EMFields_Cyl3D::
    Build_Mag_Func(const Standard_Real &dt,
                   const Standard_Integer &dynHIndex,
                   const Standard_Integer &dynJIndex,
                   const Standard_Integer &dynEIndex)
{
    // 1.0 Build Mzr Func Data
    vector<GridEdgeData*> &MzrEdge = m_MCntrFields_Cyl3D->m_SweptFaceMagFlds_Cyl3D->GetDatas();
    Standard_Size nEdge = MzrEdge.size();
    m_MzrDatasNum = nEdge;
    m_MzrFuncArray_Cyl3D = new FIT_Mag_Func_Cyl3D[nEdge];
    // std::cout << "nEdge = " << nEdge << std::endl;
    // exit(1);

    for(int i = 0; i < nEdge; ++i){
        GridEdgeData *currEdge = MzrEdge[i];
        Standard_Real *currDataPtr = currEdge->GetSweptPhysDataPtr(0);

        m_MzrFuncArray_Cyl3D[i].m_Mag = currDataPtr + dynHIndex;
        m_MzrFuncArray_Cyl3D[i].m_Current = currDataPtr + dynJIndex;

        // for dual contour 
        const vector<T_Element> &theOutLineTElems = MzrEdge[i]->GetOutLineDTEdges();
        Standard_Integer nb = theOutLineTElems.size();

        for(Standard_Integer j = 0; j < nb; ++j){
            GridVertexData *curVertex = (GridVertexData *)theOutLineTElems[j].GetData();

            m_MzrFuncArray_Cyl3D[i].m_Elec[j] = curVertex->GetSweptPhysDataPtr(dynEIndex);
            m_MzrFuncArray_Cyl3D[i].m_Curl[j] = curVertex->GetSweptGeomDim() * theOutLineTElems[j].GetRelatedDir() / currEdge->GetSweptGeomDim();
            m_MzrFuncArray_Cyl3D[i].m_Elec_Ptr_Offset[j] = m_MzrFuncArray_Cyl3D[i].m_Elec[j] - m_EphiDatasPtr;
        }

        const vector<T_Element> &theNearEEdges = MzrEdge[i]->GetNearEEdges();
        nb = theNearEEdges.size();
        for(Standard_Integer j = 0; j < nb; ++j){
            GridVertexData *curVertex = (GridVertexData *)theNearEEdges[j].GetData();

            m_MzrFuncArray_Cyl3D[i].m_ElecNear[j] = curVertex->GetPhysDataPtr(dynEIndex);
            m_MzrFuncArray_Cyl3D[i].m_CurlNear[j] = curVertex->GetSweptGeomDim_Near() * theNearEEdges[j].GetRelatedDir() / currEdge->GetSweptGeomDim();
            m_MzrFuncArray_Cyl3D[i].m_ElecNear_Ptr_Offset[j] = m_MzrFuncArray_Cyl3D[i].m_ElecNear[j] - m_EzrDatasPtr; // ...
        }
        
        m_MzrFuncArray_Cyl3D[i].CheckElecDatas();
        m_MzrFuncArray_Cyl3D[i].m_C0 = dt / currEdge->GetMu();

        // Set offsets for CUDA computing
        m_MzrFuncArray_Cyl3D[i].m_Mag_Ptr_Offset = m_MzrFuncArray_Cyl3D[i].m_Mag - m_MzrDatasPtr;
        m_MzrFuncArray_Cyl3D[i].m_Current_Ptr_Offset = m_MzrFuncArray_Cyl3D[i].m_Current - m_MzrDatasPtr;
    }
    // 2.0 Build Mphi FuncData
    vector<GridFaceData *> &MphiFace = m_MCntrFields_Cyl3D->m_FaceMagFlds_Cyl3D->GetDatas();
    Standard_Size nFace = MphiFace.size();

    m_MphiDatasNum = nFace;

    m_MphiFuncArray_Cyl3D = new FIT_Mag_Func_Cyl3D[nFace];

    for(int i = 0; i < nFace; ++i){
        GridFaceData *currFace = MphiFace[i];
        Standard_Real *currDataPtr = currFace->GetPhysDataPtr(0);

        m_MphiFuncArray_Cyl3D[i].m_Mag = currDataPtr + dynHIndex;     // 0
        m_MphiFuncArray_Cyl3D[i].m_Current = currDataPtr + dynJIndex; // 1

        // for dual contour
        const vector<T_Element> &theOutLineTElems = currFace->GetOutLineTEdge(); //GridEdgeData elements
        Standard_Integer nb = theOutLineTElems.size();
        for (Standard_Integer j = 0; j < nb; j++){
            GridEdgeData *curEdge = (GridEdgeData *)theOutLineTElems[j].GetData();
            m_MphiFuncArray_Cyl3D[i].m_Elec[j] = curEdge->GetPhysDataPtr(dynEIndex);
            m_MphiFuncArray_Cyl3D[i].m_Curl[j] = curEdge->GetGeomDim() * theOutLineTElems[j].GetRelatedDir() / currFace->GetGeomDim();
        }
        m_MphiFuncArray_Cyl3D[i].CheckElecDatas();
        m_MphiFuncArray_Cyl3D[i].m_C0 = dt / currFace->GetMu();

        // Set offsets for CUDA computing
        m_MphiFuncArray_Cyl3D[i].m_Mag_Ptr_Offset = m_MphiFuncArray_Cyl3D[i].m_Mag - m_MphiDatasPtr;
        m_MphiFuncArray_Cyl3D[i].m_Current_Ptr_Offset = m_MphiFuncArray_Cyl3D[i].m_Current - m_MphiDatasPtr;
        for (Standard_Integer j = 0; j < nb; j++){
            m_MphiFuncArray_Cyl3D[i].m_Elec_Ptr_Offset[j] = m_MphiFuncArray_Cyl3D[i].m_Elec[j] - m_EzrDatasPtr;
        }
    }
}