#include "SI_SC_Matrix_Elec_Mur_Func_Cyl3D.cuh"
#include "SI_SC_Matrix_EMFields_Cyl3D.hxx"
#include "BaseFunctionDefine.hxx"
#include "PhysConsts.hxx"
#include "SI_SC_Matrix_CPML_Equation.hxx"

void 
SI_SC_Matrix_EMFields_Cyl3D::
     Build_Elec_Mur_Func(const bool doDamping,
        const Standard_Real& dt,
        const Standard_Integer& dynEIndex,
        const Standard_Integer& PreIndex,
        const Standard_Integer& AEIndex,
        const Standard_Integer& BEIndex,
        const Standard_Integer& preTStepEFldIndx)
{
   // 1.0 Build Mur Ezr Func Data
    int a = m_EMurFields_Cyl3D->Get_MurPortsDatasNum();
    m_VBAR = 0.0;
    int nEdge = 0;
    if(a > 0){
        vector<GridEdgeData*> &Mur_EzredgeTmp = m_EMurFields_Cyl3D->Get_subDatas(0)->m_MurPortEdgeDatas;
        nEdge = Mur_EzredgeTmp.size();
        m_VBAR = m_EMurFields_Cyl3D->Get_subDatas(0)->m_VBar * dt;
    }  
    
    m_Mur_EzrDatasNum = nEdge * phi_num;
    m_EzrMurFuncArray_Cyl3D = NULL;
    m_EzrMurFuncArray_Cyl3D = new FIT_Elec_Mur_Func_Cyl3D[m_Mur_EzrDatasNum];
    
    for(int i = 0; i < m_Mur_EzrDatasNum; ++i){
        vector<GridEdgeData*> &Mur_Ezredge = m_EMurFields_Cyl3D->Get_subDatas(i/nEdge)->m_MurPortEdgeDatas;
        vector<GridEdgeData*> &Free_Ezredge = m_EMurFields_Cyl3D->Get_subDatas(i/nEdge)->m_FreeSpaceEdgeDatas;
        int idx = i;
        if(i >= nEdge) idx = i - i/nEdge*nEdge;
        GridEdgeData *currEdge = Mur_Ezredge[idx];
        GridEdgeData *currEdge1 = Free_Ezredge[idx];
        Standard_Real *currDataPtr = currEdge->GetPhysDataPtr(0);
        Standard_Real *currDataPtr1 = currEdge1->GetPhysDataPtr(0);

        m_EzrMurFuncArray_Cyl3D[i].m_Elec = currDataPtr + dynEIndex;
        m_EzrMurFuncArray_Cyl3D[i].m_Elec_Ptr_Offset = m_EzrMurFuncArray_Cyl3D[i].m_Elec - m_EzrDatasPtr;

        m_EzrMurFuncArray_Cyl3D[i].m_PreTStep = currDataPtr + preTStepEFldIndx;
        m_EzrMurFuncArray_Cyl3D[i].m_PreTStep_Ptr_Offset = m_EzrMurFuncArray_Cyl3D[i].m_PreTStep - m_EzrDatasPtr;

        m_EzrMurFuncArray_Cyl3D[i].m_AE = currDataPtr + AEIndex;
        m_EzrMurFuncArray_Cyl3D[i].m_AE_Ptr_Offset = m_EzrMurFuncArray_Cyl3D[i].m_AE - m_EzrDatasPtr;

        m_EzrMurFuncArray_Cyl3D[i].m_BE = currDataPtr + BEIndex;
        m_EzrMurFuncArray_Cyl3D[i].m_BE_Ptr_Offset = m_EzrMurFuncArray_Cyl3D[i].m_BE - m_EzrDatasPtr;

        m_EzrMurFuncArray_Cyl3D[i].m_PreTStepEFld = currDataPtr1 + dynEIndex;
        m_EzrMurFuncArray_Cyl3D[i].m_PreTStepEFld_Ptr_Offset = m_EzrMurFuncArray_Cyl3D[i].m_PreTStepEFld - m_EzrDatasPtr;

        m_EzrMurFuncArray_Cyl3D[i].m_VBar = m_VBAR;
    }
    
    // 2.0 Build Mur Ephi Func Data
    int nVertex = 0;
    if(a > 0){
        vector<GridVertexData*> &Mur_EphivertexTmp = m_EMurFields_Cyl3D->Get_subDatas(0)->m_MurPortSweptEdgeDatas;
        nVertex = Mur_EphivertexTmp.size();
    }
    
    m_Mur_EphiDatasNum = nVertex * phi_num;
    m_EphiMurFuncArray_Cyl3D = NULL;
    m_EphiMurFuncArray_Cyl3D = new FIT_Elec_Mur_Func_Cyl3D[m_Mur_EphiDatasNum];

    for(int i = 0; i < m_Mur_EphiDatasNum; ++i){
        vector<GridVertexData*> &Mur_Ephivertex = m_EMurFields_Cyl3D->Get_subDatas(i/nVertex)->m_MurPortSweptEdgeDatas;
        vector<GridVertexData*> &Free_Ephivertex = m_EMurFields_Cyl3D->Get_subDatas(i/nVertex)->m_FreeSpaceSweptEdgeDatas;
        int idx = i;
        if(i >= nVertex) idx = i - i/nVertex*nVertex;
        GridVertexData *currVertex = Mur_Ephivertex[idx];
        GridVertexData *currVertex1 = Free_Ephivertex[idx];
        Standard_Real *currDataPtr = currVertex->GetSweptPhysDataPtr(0);
        Standard_Real *currDataPtr1 = currVertex1->GetSweptPhysDataPtr(0);

        m_EphiMurFuncArray_Cyl3D[i].m_Elec = currDataPtr + dynEIndex;
        m_EphiMurFuncArray_Cyl3D[i].m_Elec_Ptr_Offset = m_EphiMurFuncArray_Cyl3D[i].m_Elec - m_EphiDatasPtr;

        m_EphiMurFuncArray_Cyl3D[i].m_PreTStep = currDataPtr + preTStepEFldIndx;
        m_EphiMurFuncArray_Cyl3D[i].m_PreTStep_Ptr_Offset = m_EphiMurFuncArray_Cyl3D[i].m_PreTStep - m_EphiDatasPtr;

        m_EphiMurFuncArray_Cyl3D[i].m_AE = currDataPtr + AEIndex;
        m_EphiMurFuncArray_Cyl3D[i].m_AE_Ptr_Offset = m_EphiMurFuncArray_Cyl3D[i].m_AE - m_EphiDatasPtr;

        m_EphiMurFuncArray_Cyl3D[i].m_BE = currDataPtr + BEIndex;
        m_EphiMurFuncArray_Cyl3D[i].m_BE_Ptr_Offset = m_EphiMurFuncArray_Cyl3D[i].m_BE - m_EphiDatasPtr;

        m_EphiMurFuncArray_Cyl3D[i].m_PreTStepEFld = currDataPtr1 + dynEIndex;
        m_EphiMurFuncArray_Cyl3D[i].m_PreTStepEFld_Ptr_Offset = m_EphiMurFuncArray_Cyl3D[i].m_PreTStepEFld - m_EphiDatasPtr;

        m_EphiMurFuncArray_Cyl3D[i].m_VBar = m_VBAR;
    }
}