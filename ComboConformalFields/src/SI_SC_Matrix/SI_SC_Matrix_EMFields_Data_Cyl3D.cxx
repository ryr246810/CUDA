#include "SI_SC_Matrix_EMFields_Cyl3D.hxx"
#include <stdlib.h>
#include "ComboFields_Dynamic_Srcs_Cyl3D.hxx"

void
SI_SC_Matrix_EMFields_Cyl3D::
    BuildDatas()
{
    // 新开辟的连续空间
    BuildFaceDatas();   // Mphi
    BuildEdgeDatas();   // Ezr Mzr
    BuildVertexDatas(); // Ephi
    // BuildEdgeAxisDatas();

    // std::cout << "llllldfsfsll" << std::endl;
    // exit(1);

    // /* 原版本离散地址
    // BuildMphiDatas(); // 
    // BuildMzrDatas(); // 考虑 Near
    // BuildEphiDatas();
    // BuildEzrDatas(); // 考虑 Near
    // BuildEAxisDatas();

    // BuildCPMLMphiDatas(); // 
    // BuildCPMLMzrDatas(); // 考虑 Near
    // BuildCPMLEphiDatas();
    // BuildCPMLEzrDatas(); // 考虑 Near
    // BuildCPMLEAxisDatas();
    // */
}

void
SI_SC_Matrix_EMFields_Cyl3D::
    CleanDatas()
{
    CleanFaceDatas();
    CleanEdgeDatas();
    CleanVertexDatas();
    CleanEdgeAxisDatas();

    // CleanMphiDatas();
    // CleanMzrDatas(); // 考虑 Near
    // CleanEphiDatas();
    // CleanEzrDatas(); // 考虑 Near
    // CleanEAxisDatas();

    // CleanCPMLMphiDatas();
    // CleanCPMLMzrDatas(); // 考虑 Near
    // CleanCPMLEphiDatas();
    // CleanCPMLEzrDatas(); // 考虑 Near
    // CleanCPMLEAxisDatas();
}

void
SI_SC_Matrix_EMFields_Cyl3D::
    BuildEAxisDatas()
{
    // Standard_Integer ElecPhysDataNum = GetFldsDefCntr()->GetFieldsDefineRules()->GetCntrElecPhysDataNum(); // = 5 (DynE J AE BE PRE)
    // m_OneEAxisPhysDataNum = ElecPhysDataNum;
    m_OneEAxisPhysDataNum = 8;
    vector<GridEdgeData *>  &EAxisedge = m_ECntrFields_Cyl3D->m_EdgeElecFlds_Axis_Cyl3D->GetDatas();
    Standard_Size                nEdge = EAxisedge.size();
    Standard_Size theTotalPhysDataSize = nEdge * m_OneEAxisPhysDataNum;
    Cempic_Size              currIndex = 0;
    m_EAxisDatasNum = nEdge;
    m_EAxisDatasSize = theTotalPhysDataSize;

    if(m_EAxisDatasPtr != NULL)
        aligned_free(m_EAxisDatasPtr);
    m_EAxisDatasPtr = (Standard_Real *)aligned_malloc(sizeof(Standard_Real) * theTotalPhysDataSize);
    memset(m_EAxisDatasPtr, 0, sizeof(Standard_Real) * theTotalPhysDataSize);
    
    for(Cempic_Size i = 0; i < nEdge; ++i){
        // PhysData ptr
        Standard_Real *PhysDataptr = m_EAxisDatasPtr + currIndex;
        EAxisedge[i]->ResetPhysDataPtr(m_OneEAxisPhysDataNum, PhysDataptr);
        currIndex = currIndex + m_OneEAxisPhysDataNum;
    }
    
}

void
SI_SC_Matrix_EMFields_Cyl3D::
    BuildEzrDatas()
{
    Standard_Integer ElecPhysDataNum = GetFldsDefCntr()->GetFieldsDefineRules()->GetCntrElecPhysDataNum(); // = 5 (DynE J AE BE PRE)
    m_OneEzrPhysDataNum = ElecPhysDataNum;
    vector<GridEdgeData*> &Ezredge = m_ECntrFields_Cyl3D->m_EdgeElecFlds_Cyl3D->GetDatas();
    Standard_Size nEdge = Ezredge.size();
    Standard_Size theTotalPhysDataSize = nEdge * m_OneEzrPhysDataNum;
    Standard_Size currIndex = 0;
    m_EzrDatasNum = nEdge;
    m_EzrDatasSize = theTotalPhysDataSize;


    if(m_EzrDatasPtr != NULL) 
        aligned_free(m_EzrDatasPtr);
    m_EzrDatasPtr = (Standard_Real *)aligned_malloc(sizeof(Standard_Real) * theTotalPhysDataSize);
    memset(m_EzrDatasPtr, 0, sizeof(Standard_Real) * theTotalPhysDataSize);

    for(Standard_Size i = 0; i < nEdge; ++i){
        // PhysData ptr
        Standard_Real * PhysDataptr = m_EzrDatasPtr + currIndex;
        Ezredge[i]->ResetPhysDataPtr(m_OneEzrPhysDataNum, PhysDataptr);
        currIndex = currIndex + m_OneEzrPhysDataNum;
    }
}

void
SI_SC_Matrix_EMFields_Cyl3D::
    BuildEphiDatas()
{
    Standard_Integer ElecPhysDataNum = GetFldsDefCntr()->GetFieldsDefineRules()->GetCntrElecPhysDataNum(); // = 5 (DynE J AE BE PRE)
    m_OneEphiPhysDataNum = ElecPhysDataNum;
    vector<GridVertexData*> &EphiVertex = m_ECntrFields_Cyl3D->m_SweptEdgeElecFlds_Cyl3D->GetDatas();
    Standard_Size nVertex = EphiVertex.size();
    Standard_Size theTotalPhysDataSize = nVertex * m_OneEphiPhysDataNum;
    Standard_Size currIndex = 0;
    m_EphiDatasNum = nVertex;
    m_EphiDatasSize = theTotalPhysDataSize;

    if(m_EphiDatasPtr != NULL) 
        aligned_free(m_EphiDatasPtr);
    m_EphiDatasPtr = (Standard_Real *)aligned_malloc(sizeof(Standard_Real) * theTotalPhysDataSize);
    memset(m_EphiDatasPtr, 0, sizeof(Standard_Real) * theTotalPhysDataSize);

    for(Standard_Size i = 0; i < nVertex; ++i){
        // PhysData ptr
        Standard_Real * PhysDataptr = m_EphiDatasPtr + currIndex;
        EphiVertex[i]->ResetSweptPhysDataPtr(m_OneEphiPhysDataNum, PhysDataptr);
        currIndex = currIndex + m_OneEphiPhysDataNum;
    }
}

void 
SI_SC_Matrix_EMFields_Cyl3D::
    BuildMzrDatas()
{
    Standard_Integer MagPhysDataNum = GetFldsDefCntr()->GetFieldsDefineRules()->GetCntrMagPhysDataNum(); // =2 (dynH,J)
    m_OneMzrPhysDataNum = MagPhysDataNum;
    vector<GridEdgeData *> &MzrEdge  = m_MCntrFields_Cyl3D->m_SweptFaceMagFlds_Cyl3D->GetDatas();
    Standard_Size nEdge = MzrEdge.size();
    Standard_Size theTotalPhysDataSize = nEdge * m_OneMzrPhysDataNum;
    Standard_Size currIndex = 0;
    m_MzrDatasNum = nEdge;
    m_MzrDatasSize = theTotalPhysDataSize;

    if (m_MzrDatasPtr != NULL) 
        aligned_free(m_MzrDatasPtr);
    m_MzrDatasPtr = (Standard_Real *) aligned_malloc(sizeof(Standard_Real) * theTotalPhysDataSize);
    memset(m_MzrDatasPtr, 0, sizeof(Standard_Real) * theTotalPhysDataSize);

    for (Standard_Size i = 0; i < nEdge; i++)
    {
        //PhysData ptr
        Standard_Real * PhysDataptr = m_MzrDatasPtr + currIndex;
        MzrEdge[i]->ResetSweptPhysDataPtr(m_OneMzrPhysDataNum, PhysDataptr);
        currIndex = currIndex + m_OneMzrPhysDataNum;
    }
}

void
SI_SC_Matrix_EMFields_Cyl3D:: // 1
    BuildMphiDatas()
{
    Standard_Integer MagPhysDataNum = GetFldsDefCntr()->GetFieldsDefineRules()->GetCntrMagPhysDataNum(); // =2 (dynH,J)
    // m_OneMphiPhysDataNum = MagPhysDataNum;
    m_OneMphiPhysDataNum = 4;
    vector<GridFaceData *> &MphiFace = m_MCntrFields_Cyl3D->m_FaceMagFlds_Cyl3D->GetDatas();
    Standard_Size nFace = MphiFace.size();
    Standard_Size theTotalPhysDataSize = nFace * m_OneMphiPhysDataNum;
    Standard_Size currIndex = 0;
    m_MphiDatasNum = nFace;
    m_MphiDatasSize = theTotalPhysDataSize;

    if (m_MphiDatasPtr != NULL) 
        aligned_free(m_MphiDatasPtr);
    m_MphiDatasPtr = (Standard_Real *) aligned_malloc(sizeof(Standard_Real) * theTotalPhysDataSize);
    memset(m_MphiDatasPtr, 0, sizeof(Standard_Real) * theTotalPhysDataSize);

    for (Standard_Size i = 0; i < nFace; i++)
    {
        //PhysData ptr
        Standard_Real * PhysDataptr = m_MphiDatasPtr + currIndex;
        MphiFace[i]->ResetPhysDataPtr(m_OneMphiPhysDataNum, PhysDataptr);
        currIndex = currIndex + m_OneMphiPhysDataNum;
    }
}

void 
SI_SC_Matrix_EMFields_Cyl3D::
    BuildCPMLEAxisDatas()
{
    Standard_Integer ElecPMLPhysDataNum = GetFldsDefCntr()->GetFieldsDefineRules()->GetBndElecPhysDataNum(PML);
    m_OneEAxisPMLPhysDataNum = ElecPMLPhysDataNum;
    vector<GridEdgeData *> &CPML_EAxisedge = m_ECPMLFields_Cyl3D->m_EdgeElecFlds_Axis_Cyl3D->GetDatas();
    Standard_Size nEdge = CPML_EAxisedge.size();
    Standard_Size theTotalPhysDataSize = nEdge * m_OneEAxisPMLPhysDataNum;
    Standard_Size currIndex = 0;
    m_CPML_EAxisDatasNum = nEdge;

    if(m_CPML_EAxisDatasPtr != NULL)
        aligned_free(m_CPML_EAxisDatasPtr);
    m_CPML_EAxisDatasPtr = (Standard_Real *)aligned_malloc(sizeof(Standard_Real) * theTotalPhysDataSize);
    memset(m_CPML_EAxisDatasPtr, 0, sizeof(Standard_Real) * theTotalPhysDataSize);

    for(Standard_Size i = 0; i < nEdge; ++i){
        // PhysData ptr
        Standard_Real *PhysDataptr = m_CPML_EAxisDatasPtr + currIndex;
        CPML_EAxisedge[i]->ResetPhysDataPtr(m_OneEAxisPMLPhysDataNum, PhysDataptr);
        currIndex = currIndex + m_OneEAxisPMLPhysDataNum;
    }
}

void
SI_SC_Matrix_EMFields_Cyl3D::
    BuildCPMLEzrDatas()
{
    Standard_Integer ElecPMLPhysDataNum = GetFldsDefCntr()->GetFieldsDefineRules()->GetBndElecPhysDataNum(PML); // = 7 pml: dynamic(0), current(1), AE(2), BE(3), PRE(4), PE1(5), PE2(6), here AE and BE is for damping
    m_OneEzrPMLPhysDataNum = ElecPMLPhysDataNum;
    vector<GridEdgeData *> &CPML_Ezredge  = m_ECPMLFields_Cyl3D->m_EdgeElecFlds_Cyl3D->GetDatas();
    Standard_Size nEdge = CPML_Ezredge.size();
    Standard_Size theTotalPhysDataSize = nEdge * m_OneEzrPMLPhysDataNum;
    Standard_Size currIndex = 0;
    m_CPML_EzrDatasNum = nEdge;

    // if (m_CPML_EzrDatasPtr != NULL) 
    //     aligned_free(m_CPML_EzrDatasPtr);
    // m_CPML_EzrDatasPtr = (Standard_Real *)aligned_malloc(sizeof(Standard_Real) * theTotalPhysDataSize);
    // memset(m_CPML_EzrDatasPtr, 0, sizeof(Standard_Real) * theTotalPhysDataSize);

    // for (Standard_Size i = 0; i < nEdge; i++)
    // {
    //     //PhysData ptr
    //     Standard_Real * PhysDataptr = m_CPML_EzrDatasPtr + currIndex;
    //     CPML_Ezredge[i]->ResetPhysDataPtr(m_OneEzrPMLPhysDataNum, PhysDataptr);
    //     currIndex = currIndex + m_OneEzrPMLPhysDataNum;
    // }
}

void 
SI_SC_Matrix_EMFields_Cyl3D::
    BuildCPMLEphiDatas()
{
    Standard_Integer ElecPMLPhysDataNum = GetFldsDefCntr()->GetFieldsDefineRules()->GetBndElecPhysDataNum(PML); // = 7 pml: dynamic(0), current(1), AE(2), BE(3), PRE(4), PE1(5), PE2(6), here AE and BE is for damping
    m_OneEphiPMLPhysDataNum = ElecPMLPhysDataNum;
    vector<GridVertexData *> &CPML_EphiVertex = m_ECPMLFields_Cyl3D->m_SweptEdgeElecFlds_Cyl3D->GetDatas();
    Standard_Size nVertex = CPML_EphiVertex.size();
    Standard_Size theTotalPhysDataSize = nVertex * m_OneEphiPMLPhysDataNum;
    Standard_Size currIndex = 0;
    m_CPML_EphiDatasNum = nVertex;

    // if (m_CPML_EphiDatasPtr != NULL) 
    //     aligned_free(m_CPML_EphiDatasPtr);
    // m_CPML_EphiDatasPtr = (Standard_Real *)aligned_malloc(sizeof(Standard_Real) * theTotalPhysDataSize);
    // memset(m_CPML_EphiDatasPtr, 0, sizeof(Standard_Real) * theTotalPhysDataSize);

    // for (Standard_Size i = 0; i < nVertex; i++)
    // {
    //     //PhysData ptr
    //     Standard_Real * PhysDataptr = m_CPML_EphiDatasPtr + currIndex;
    //     CPML_EphiVertex[i]->ResetSweptPhysDataPtr(m_OneEphiPMLPhysDataNum, PhysDataptr);
    //     currIndex = currIndex + m_OneEphiPMLPhysDataNum;
    // }
}

void
SI_SC_Matrix_EMFields_Cyl3D::
    BuildCPMLMzrDatas()
{
    Standard_Integer MagPMLPhysDataNum = GetFldsDefCntr()->GetFieldsDefineRules()->GetBndMagPhysDataNum(PML); // pml dynamic, current, PM1, PM2
    m_OneMzrPMLPhysDataNum = MagPMLPhysDataNum;
    vector<GridEdgeData *> &CPML_MzrEdge = m_MCPMLFields_Cyl3D->m_SweptFaceMagFlds_Cyl3D->GetDatas();
    Standard_Size nEdge = CPML_MzrEdge.size();
    Standard_Size theTotalPhysDataSize = nEdge * m_OneMzrPMLPhysDataNum;
    Standard_Size currIndex = 0;
    m_CPML_MzrDatasNum = nEdge;

    // if (m_CPML_MzrDatasPtr != NULL) 
    //     aligned_free(m_CPML_MzrDatasPtr);
    // m_CPML_MzrDatasPtr = (Standard_Real *)aligned_malloc(sizeof(Standard_Real) * theTotalPhysDataSize);
    // memset(m_CPML_MzrDatasPtr, 0, sizeof(Standard_Real) * theTotalPhysDataSize);

    // for (Standard_Size i = 0; i < nEdge; i++)
    // {
    //     //PhysData ptr
    //     Standard_Real * PhysDataptr = m_CPML_MzrDatasPtr + currIndex;
    //     CPML_MzrEdge[i]->ResetSweptPhysDataPtr(m_OneMzrPMLPhysDataNum, PhysDataptr);
    //     currIndex = currIndex + m_OneMzrPMLPhysDataNum;
    // }
}

void
SI_SC_Matrix_EMFields_Cyl3D::
    BuildCPMLMphiDatas()
{
    Standard_Integer MagPMLPhysDataNum = GetFldsDefCntr()->GetFieldsDefineRules()->GetBndMagPhysDataNum(PML); // pml dynamic, current, PM1, PM2
    m_OneMphiPMLPhysDataNum = MagPMLPhysDataNum;
    vector<GridFaceData *> &CPML_MphiFace = m_MCPMLFields_Cyl3D->m_FaceMagFlds_Cyl3D->GetDatas();
    Standard_Size nFace = CPML_MphiFace.size();
    Standard_Size theTotalPhysDataSize = nFace * m_OneMphiPMLPhysDataNum;
    Standard_Size currIndex = 0;
    m_CPML_MphiDatasNum = nFace;

    if (m_CPML_MphiDatasPtr != NULL) 
        aligned_free(m_CPML_MphiDatasPtr);
    m_CPML_MphiDatasPtr = (Standard_Real *)aligned_malloc(sizeof(Standard_Real) * theTotalPhysDataSize);
    memset(m_CPML_MphiDatasPtr, 0, sizeof(Standard_Real) * theTotalPhysDataSize);

    for (Standard_Size i = 0; i < nFace; i++)
    {
        //PhysData ptr
        Standard_Real * PhysDataptr = m_CPML_MphiDatasPtr + currIndex;
        CPML_MphiFace[i]->ResetPhysDataPtr(m_OneMphiPMLPhysDataNum, PhysDataptr);
        currIndex = currIndex + m_OneMphiPMLPhysDataNum;
    }
}

void
SI_SC_Matrix_EMFields_Cyl3D::
    CleanEAxisDatas()
{
    vector<GridEdgeData *>  &EAxisedge = m_ECntrFields_Cyl3D->m_EdgeElecFlds_Axis_Cyl3D->GetDatas();
    Cempic_Size                  nEdge = EAxisedge.size();
    for(Cempic_Size i = 0; i < nEdge; ++i){
        EAxisedge[i]->CleanPhysDataPtr();
        m_EAxisFuncArray_Cyl3D->m_Elec = NULL;
        m_EAxisFuncArray_Cyl3D->m_Current = NULL;
        m_EAxisFuncArray_Cyl3D->m_AE = NULL;
        m_EAxisFuncArray_Cyl3D->m_BE = NULL;
        m_EAxisFuncArray_Cyl3D->m_Elec_PreStep = NULL;

        GridEdgeData *currEdge = EAxisedge[i];
        const vector<T_Element> &theOutlineTElems = currEdge->GetSharedTFace();
        Standard_Integer nb = theOutlineTElems.size();

        for(Cempic_Size j = 0; j < nb; ++j){
            m_EAxisFuncArray_Cyl3D[i].m_Mag[j] = NULL;
        }
    }
}

void 
SI_SC_Matrix_EMFields_Cyl3D::
    CleanEzrDatas()
{
    vector<GridEdgeData *> &Ezredge = m_ECntrFields_Cyl3D->m_EdgeElecFlds_Cyl3D->GetDatas();
    Standard_Size nEdge = Ezredge.size();
    for (Standard_Size i = 0; i < nEdge; i++)
    {
        Ezredge[i]->CleanPhysDataPtr();
        m_EzrFuncArray_Cyl3D->m_Elec = NULL;
        m_EzrFuncArray_Cyl3D->m_Current = NULL;
        m_EzrFuncArray_Cyl3D->m_AE = NULL;
        m_EzrFuncArray_Cyl3D->m_BE = NULL;
        m_EzrFuncArray_Cyl3D->m_Elec_PreStep = NULL;

        GridEdgeData *currEdge = Ezredge[i];

        const vector<T_Element> &theOutLineTElems = currEdge->GetSharedTFace(); //GridFaceData
        int nb = theOutLineTElems.size();

        for (Standard_Size j = 0; j < nb; j++)
        {
            m_EzrFuncArray_Cyl3D[i].m_Mag[j] = NULL;
        }

        const vector<T_Element> theNearMEdges = currEdge->GetNearMEdges();
        nb = theNearMEdges.size();

        for(Standard_Size j = 0; j < nb; ++j){
            m_EzrFuncArray_Cyl3D[i].m_MagNear[j] = NULL;
        }
    }
}

void 
SI_SC_Matrix_EMFields_Cyl3D::
    CleanEphiDatas()
{
    vector<GridVertexData *> &EphiVertex = m_ECntrFields_Cyl3D->m_SweptEdgeElecFlds_Cyl3D->GetDatas();
    Standard_Size nVertex = EphiVertex.size();

    for (Standard_Size i = 0; i < nVertex; i++)
    {
        EphiVertex[i]->CleanSweptPhysDataPtr();
        m_EphiFuncArray_Cyl3D->m_Elec = NULL;
        m_EphiFuncArray_Cyl3D->m_Current = NULL;
        m_EphiFuncArray_Cyl3D->m_AE = NULL;
        m_EphiFuncArray_Cyl3D->m_BE = NULL;
        m_EphiFuncArray_Cyl3D->m_Elec_PreStep = NULL;
        
        GridVertexData *currVertex = EphiVertex[i];
        const vector<T_Element> &theOutLineTElems = currVertex->GetSharedTDFaces(); //GridEdgeData
        int nb = theOutLineTElems.size();
        for (Standard_Size j = 0; j < nb; j++)
        {
            m_EphiFuncArray_Cyl3D[i].m_Mag[j] = NULL;
        }
    }
}

void
SI_SC_Matrix_EMFields_Cyl3D::
    CleanMzrDatas()
{
    vector<GridEdgeData *> &MzrEdge = m_MCntrFields_Cyl3D->m_SweptFaceMagFlds_Cyl3D->GetDatas();
    Standard_Size nEdge = MzrEdge.size();

    for (Standard_Size i = 0; i < nEdge; i++)
    {
        MzrEdge[i]->CleanSweptPhysDataPtr();
        m_MzrFuncArray_Cyl3D[i].m_Mag = NULL;
        m_MzrFuncArray_Cyl3D[i].m_Current = NULL;

        GridEdgeData *currEdge = MzrEdge[i];

        const vector<T_Element> &theOutLineTElems = MzrEdge[i]->GetOutLineDTEdges(); // GridVertexData elements
        Standard_Integer nb = theOutLineTElems.size();
        for (Standard_Integer j = 0; j < nb; j++){
            m_MzrFuncArray_Cyl3D[i].m_Elec[j] = NULL;
        }

        const vector<T_Element> &theNearEEdges    = MzrEdge[i]->GetNearEEdges();     // GridEdgeData elements in near slice
        nb = theNearEEdges.size();
        for(Standard_Integer j = 0; j < nb; ++j){
            m_MzrFuncArray_Cyl3D[i].m_ElecNear[j] = NULL;
        }
    }
}

void 
SI_SC_Matrix_EMFields_Cyl3D::
    CleanMphiDatas()
{
    vector<GridFaceData *> &MphiFace = m_MCntrFields_Cyl3D->m_FaceMagFlds_Cyl3D->GetDatas();
    Standard_Size nFace = MphiFace.size();

    for (Standard_Size i = 0; i < nFace; i++)
    {
        MphiFace[i]->CleanPhysDataPtr();
        m_MphiFuncArray_Cyl3D[i].m_Mag = NULL;
        m_MphiFuncArray_Cyl3D[i].m_Current = NULL;

        GridFaceData *currFace = MphiFace[i];
        const vector<T_Element> &theOutLineTElems = currFace->GetOutLineTEdge(); //GridEdgeData elements
        Standard_Integer nb = theOutLineTElems.size();
        for (Standard_Integer j = 0; j < nb; j++){
            m_MphiFuncArray_Cyl3D[i].m_Elec[j] = NULL;
        }
    }
}

void
SI_SC_Matrix_EMFields_Cyl3D::
    CleanCPMLEAxisDatas()
{
    vector<GridEdgeData *> &CPML_EAxisedge = m_ECPMLFields_Cyl3D->m_EdgeElecFlds_Axis_Cyl3D->GetDatas();
    Standard_Size nEdge = CPML_EAxisedge.size();

    for(Standard_Size i = 0; i < nEdge; ++i){
        CPML_EAxisedge[i]->CleanPhysDataPtr();
        m_EAxisPMLFuncArray_Cyl3D[i].m_Elec = NULL;
        m_EAxisPMLFuncArray_Cyl3D[i].m_Elec_PreStep = NULL;
        m_EAxisPMLFuncArray_Cyl3D[i].m_BE = NULL;
        m_EAxisPMLFuncArray_Cyl3D[i].m_AE = NULL;
        m_EAxisPMLFuncArray_Cyl3D[i].m_PE1 = NULL;
        m_EAxisPMLFuncArray_Cyl3D[i].m_PE2 = NULL;

        GridEdgeData *currEdge = CPML_EAxisedge[i];

        const vector<T_Element> &theOutLineTElems = currEdge->GetSharedTFace();
        int nb = theOutLineTElems.size();
        for(Standard_Size j = 0; j < nb; ++j){
            m_EAxisPMLFuncArray_Cyl3D[i].m_Mag[j] = NULL;
        }
    }
}

void 
SI_SC_Matrix_EMFields_Cyl3D::
    CleanCPMLEzrDatas()
{
    vector<GridEdgeData *> &CPML_Ezredge = m_ECPMLFields_Cyl3D->m_EdgeElecFlds_Cyl3D->GetDatas();
    Standard_Size nEdge = CPML_Ezredge.size();
    
    for (Standard_Size i = 0; i < nEdge; i++)
    {
        CPML_Ezredge[i]->CleanPhysDataPtr();
        m_EzrPMLFuncArray_Cyl3D[i].m_Elec = NULL;
        m_EzrPMLFuncArray_Cyl3D[i].m_Elec_PreStep = NULL;
        m_EzrPMLFuncArray_Cyl3D[i].m_BE = NULL;
        m_EzrPMLFuncArray_Cyl3D[i].m_AE = NULL;
        m_EzrPMLFuncArray_Cyl3D[i].m_PE1 = NULL;
        m_EzrPMLFuncArray_Cyl3D[i].m_PE2 = NULL;

        GridEdgeData *currEdge = CPML_Ezredge[i];

        const vector<T_Element> &theOutLineTElems = currEdge->GetSharedTFace(); //GridFaceData
        int nb = theOutLineTElems.size();
        for (Standard_Size j = 0; j < nb; j++)
        {
            m_EzrPMLFuncArray_Cyl3D[i].m_Mag[j] = NULL;
        }

        const vector<T_Element> &theNearMEdges = currEdge->GetNearMEdges();
        nb = theNearMEdges.size();
        for(Standard_Size j = 0; j < nb; ++j){
            m_EzrPMLFuncArray_Cyl3D[i].m_MagNear[j] = NULL;
        }
    }
}

void
SI_SC_Matrix_EMFields_Cyl3D::
    CleanCPMLEphiDatas()
{
    vector<GridVertexData *> &CPML_EphiVertex = m_ECPMLFields_Cyl3D->m_SweptEdgeElecFlds_Cyl3D->GetDatas();
    Standard_Size nVertex = CPML_EphiVertex.size();

    for (Standard_Size i = 0; i < nVertex; i++)
    {
        CPML_EphiVertex[i]->CleanSweptPhysDataPtr();
        m_EphiPMLFuncArray_Cyl3D[i].m_Elec = NULL;
        m_EphiPMLFuncArray_Cyl3D[i].m_Elec_PreStep = NULL;
        m_EphiPMLFuncArray_Cyl3D[i].m_BE = NULL;
        m_EphiPMLFuncArray_Cyl3D[i].m_AE = NULL;
        m_EphiPMLFuncArray_Cyl3D[i].m_PE1 = NULL;
        m_EphiPMLFuncArray_Cyl3D[i].m_PE2 = NULL;

        GridVertexData *currVertex = CPML_EphiVertex[i];
        const vector<T_Element> &theOutLineTElems = currVertex->GetSharedTDFaces(); //GridEdgeData
        int nb = theOutLineTElems.size();
        for (Standard_Size j = 0; j < nb; j++)
        {
            m_EphiPMLFuncArray_Cyl3D[i].m_Mag[j] = NULL;
        }
    }
}

void
SI_SC_Matrix_EMFields_Cyl3D::
    CleanCPMLMzrDatas()
{
    vector<GridEdgeData *> &CPML_MzrEdge = m_MCPMLFields_Cyl3D->m_SweptFaceMagFlds_Cyl3D->GetDatas();
    Standard_Size nEdge = CPML_MzrEdge.size();

    for (Standard_Size i = 0; i < nEdge; i++)
    {
        CPML_MzrEdge[i]->CleanSweptPhysDataPtr();
        m_MzrPMLFuncArray_Cyl3D[i].m_Mag = NULL;
        m_MzrPMLFuncArray_Cyl3D[i].m_PM1 = NULL;
        m_MzrPMLFuncArray_Cyl3D[i].m_PM2 = NULL;

        GridEdgeData *currEdge = CPML_MzrEdge[i];

        const vector<T_Element> &theOutLineTElems = currEdge->GetOutLineDTEdges(); // GridVertexData
        int nb = theOutLineTElems.size();
        for (Standard_Size j = 0; j < nb; j++)
        {
            m_MzrPMLFuncArray_Cyl3D[i].m_Elec[j] = NULL;
        }

        const vector<T_Element> &theNearEEdges = currEdge->GetNearEEdges(); // GridEdgeData elements in near slice
        nb = theNearEEdges.size();
        for(Standard_Integer j = 0; j < nb; ++j){
            m_MzrPMLFuncArray_Cyl3D[i].m_ElecNear[j] = NULL;
        }
    }
}

void
SI_SC_Matrix_EMFields_Cyl3D::
    CleanCPMLMphiDatas()
{
    vector<GridFaceData *> &CPML_MphiFace = m_MCPMLFields_Cyl3D->m_FaceMagFlds_Cyl3D->GetDatas();
    Standard_Size nFace = CPML_MphiFace.size();

    for (Standard_Size i = 0; i < nFace; i++)
    {
        CPML_MphiFace[i]->CleanPhysDataPtr();
        m_MphiPMLFuncArray_Cyl3D[i].m_Mag = NULL;
        m_MphiPMLFuncArray_Cyl3D[i].m_PM1 = NULL;
        m_MphiPMLFuncArray_Cyl3D[i].m_PM2 = NULL;

        GridFaceData *currFace = CPML_MphiFace[i];
        const vector<T_Element>& theOutLineTElems = currFace->GetOutLineTEdge();
        int nb = theOutLineTElems.size();
        for (Standard_Size j = 0; j < nb; j++)
        {
            m_MphiPMLFuncArray_Cyl3D[i].m_Elec[j] = NULL;
        }
    }
}

// Mag Phi
void 
SI_SC_Matrix_EMFields_Cyl3D::
    BuildFaceDatas()
{
    /*
	(1) CntrMag = 2    (dynamic(0), current(1))
	(2) CPMLMag = 4 (dynamic(0), current(1), PM1, PM2)
	*/
    Standard_Integer MagPhysDataNum = 4;// Set PhysDataNum = max(2, 4) = 4
    m_OneMphiPhysDataNum = MagPhysDataNum;

    vector<GridFaceData *> theXtndRgnFaceDatas;
    for(Standard_Integer i = 0; i < phi_num; ++i){
        TxSlab2D<Standard_Integer> theXtndRgn = GetFldsDefCntr()->GetGridGeom_Cyl3D()->GetZRGrid()->GetXtndRgn();
        GetFldsDefCntr()->GetGridGeom(i)->GetGridFaceDatasNotOfMaterialTypeOfSubRgn(0, 
            theXtndRgn, 
            theXtndRgnFaceDatas);
    }
    Standard_Integer theGeomDataNum = theXtndRgnFaceDatas.size();
    Standard_Integer theTotalPhysDataSize = theGeomDataNum * m_OneMphiPhysDataNum;

    m_MphiDatasSize = theTotalPhysDataSize;

    // 开辟新空间
    if(m_MphiDatasPtr != NULL)
        aligned_free(m_MphiDatasPtr);
    m_MphiDatasPtr = (Standard_Real *)aligned_malloc(sizeof(Standard_Real) * theTotalPhysDataSize);
    memset(m_MphiDatasPtr, 0, sizeof(Standard_Real) * theTotalPhysDataSize);

    // 旧空间数据指针重新指向连续新空间
    Standard_Size currIndex = 0;
    for(Standard_Size i = 0; i < theGeomDataNum; ++i){
        // Reset Mphi PhysDats ptr
        Standard_Real *PhysDataptr = m_MphiDatasPtr + currIndex;
        theXtndRgnFaceDatas[i]->ResetPhysDataPtr(m_OneMphiPhysDataNum, PhysDataptr);
        currIndex += m_OneMphiPhysDataNum;
    }
}

void
SI_SC_Matrix_EMFields_Cyl3D::
    BuildVertexDatas()
{
    /*
	(1) CntrElec    = 5  (dynamic(0), current(1), AE(2), BE(3), PRE(4))
	(2) BndElec_PML = 8 (dynamic(0), current(1), AE(2), BE(3), PRE(4), PE1(5), PE2(6))
	*/
    Standard_Integer ElecPhysDataNum = 8;
    m_OneEphiPhysDataNum = ElecPhysDataNum;

    vector<GridVertexData *> theXtndRgnVertexDatas;
    for(Standard_Integer i = 0; i < phi_num; ++i){
        TxSlab2D<Standard_Integer> theXtndRgn = GetFldsDefCntr()->GetGridGeom_Cyl3D()->GetZRGrid()->GetXtndRgn();
        GetFldsDefCntr()->GetGridGeom(i)->GetGridVertexDatasNotOfMaterialTypeOfSubRgn(0,
            theXtndRgn,
            false,
            theXtndRgnVertexDatas);
    }
    Standard_Integer theGeomDataNum = theXtndRgnVertexDatas.size();
    Standard_Integer theTotalPhysDataSize = theGeomDataNum * m_OneEphiPhysDataNum;

    m_EphiDatasSize = theTotalPhysDataSize;

    if(m_EphiDatasPtr != NULL)
        aligned_free(m_EphiDatasPtr);
    m_EphiDatasPtr = (Standard_Real *)aligned_malloc(sizeof(Standard_Real) * theTotalPhysDataSize);
    memset(m_EphiDatasPtr, 0, sizeof(Standard_Real) * theTotalPhysDataSize);

    Standard_Size currIndex = 0;
    for(Standard_Size i = 0; i < theGeomDataNum; ++i){
        // PhysData ptr
        Standard_Real *PhysDataPtr = m_EphiDatasPtr + currIndex;
        theXtndRgnVertexDatas[i]->ResetSweptPhysDataPtr(m_OneEphiPhysDataNum, PhysDataPtr);
        currIndex = currIndex + m_OneEphiPhysDataNum;
    }
}

void 
SI_SC_Matrix_EMFields_Cyl3D::
    BuildEdgeDatas()
{
    /*
	(1) CntrElec    = 5  (dynamic(0), current(1), AE(2), BE(3), PRE(4))
	(2) CPMLeLEC    = 8  (dynamic(0), current(1), AE(2), BE(3), PRE(4), PE1(5), PE2(6))
	*/
    Standard_Integer ElecPhysDataNum = 8;
    m_OneEzrPhysDataNum = ElecPhysDataNum;
    /*
	(1) CntrMag = 2    (dynamic(0), current(1))
	(2) CPMLMag = 4    (dynamic(0), current(1), PM1, PM2)
	*/
    Standard_Integer MagPhysDataNum = 4;
    m_OneMzrPhysDataNum = MagPhysDataNum;

    vector<GridEdgeData *> theXtndRgnEdgeDatas;
    for(Standard_Integer i = 0; i < phi_num; ++i){
        TxSlab2D<Standard_Integer> theXtndRgn = GetFldsDefCntr()->GetGridGeom_Cyl3D()->GetZRGrid()->GetXtndRgn();
        GetFldsDefCntr()->GetGridGeom(i)->GetGridEdgeDatasNotOfMaterialTypeOfSubRgn(0, 
            theXtndRgn,
            false,
            theXtndRgnEdgeDatas);
    }
    Standard_Integer theEdgeDatasNum = theXtndRgnEdgeDatas.size();
    Standard_Integer theTotalEzrPhysDataSize = theEdgeDatasNum * m_OneEzrPhysDataNum;
    Standard_Integer theTotalMzrPhysDataSize = theEdgeDatasNum * m_OneMzrPhysDataNum;

    m_EzrDatasSize = theTotalEzrPhysDataSize;

    if(m_EzrDatasPtr != NULL)
        aligned_free(m_EzrDatasPtr);
    m_EzrDatasPtr = (Standard_Real *)aligned_malloc(sizeof(Standard_Real) * theTotalEzrPhysDataSize);
    memset(m_EzrDatasPtr, 0, sizeof(Standard_Real) * theTotalEzrPhysDataSize);

    m_MzrDatasSize = theTotalMzrPhysDataSize;

    if(m_MzrDatasPtr != NULL)
        aligned_free(m_MzrDatasPtr);
    m_MzrDatasPtr = (Standard_Real *)aligned_malloc(sizeof(Standard_Real) * theTotalMzrPhysDataSize);
    memset(m_MzrDatasPtr, 0, sizeof(Standard_Real) * theTotalMzrPhysDataSize);

    Standard_Size currIndexEzr = 0;
    Standard_Size currIndexMzr = 0;
    Standard_Real *PhysDataptr = NULL;
    for(Standard_Size i = 0; i < theEdgeDatasNum; ++i){
        // Reset Edge Ezr PhysData ptr
        PhysDataptr = m_EzrDatasPtr + currIndexEzr;
        theXtndRgnEdgeDatas[i]->ResetPhysDataPtr(m_OneEzrPhysDataNum, PhysDataptr);
        currIndexEzr = currIndexEzr + m_OneEzrPhysDataNum;

        // Reset Edge Mzr PhysData ptr
        PhysDataptr = m_MzrDatasPtr + currIndexMzr;
        theXtndRgnEdgeDatas[i]->ResetSweptPhysDataPtr(m_OneMzrPhysDataNum, PhysDataptr);
        currIndexMzr = currIndexMzr + m_OneMzrPhysDataNum;
    }

    //-------
    // printf("the number of theXtndRgnEdgeDatas is %d\n\n", theEdgeDatasNum);
    // printf("the start address of theXtndRgnEdgeDatas is %x\n\n", m_EzrDatasPtr);
    // printf("the end address of theXtndRgnEdgeDatas is %x\n\n", m_EzrDatasPtr + sizeof(Standard_Real) * theTotalEzrPhysDataSize);
    //-------
}

void 
SI_SC_Matrix_EMFields_Cyl3D::
    BuildNearEdgeDatas()
{
    vector<GridEdgeData*> &MzrEdge = m_MCntrFields_Cyl3D->m_SweptFaceMagFlds_Cyl3D->GetDatas();
    Standard_Size nEdge = MzrEdge.size();
    m_OneEAxisPhysDataNum = 8;
    Standard_Integer theTotalEAxisPhysDataSize = nEdge * m_OneEAxisPhysDataNum;
    m_EAxisDatasSize = theTotalEAxisPhysDataSize;

    if(m_EAxisDatasPtr != NULL)
        aligned_free(m_EAxisDatasPtr);
    m_EAxisDatasPtr = (Standard_Real *)aligned_malloc(sizeof(Standard_Real) * theTotalEAxisPhysDataSize);
    memset(m_EAxisDatasPtr, 0, sizeof(Standard_Real) * theTotalEAxisPhysDataSize);

    Standard_Size currIndex = 0;
    Standard_Real *PhysDataptr = NULL;
    for(int i = 0; i < nEdge; ++i){
        const vector<T_Element> &theNearEEdges = MzrEdge[i]->GetNearEEdges();
        for(Standard_Integer j = 0; j < 2; ++j){
            GridVertexData *curVertex = (GridVertexData *)theNearEEdges[j].GetData();

            PhysDataptr = m_EAxisDatasPtr + currIndex;
            curVertex->ResetPhysDataPtr(m_OneEAxisPhysDataNum, PhysDataptr);
            currIndex = currIndex + m_OneEAxisPhysDataNum;
            //  = curVertex->GetPhysDataPtr(3);
        }

    }
}

void 
SI_SC_Matrix_EMFields_Cyl3D::
    BuildEdgeAxisDatas()
{
    /*
	(1) CntrEAxis    = 5  (dynamic(0), current(1), AE(2), BE(3), PRE(4))
	(2) CPMLEAxis    = 8  (dynamic(0), current(1), AE(2), BE(3), PRE(4), PE1(5), PE2(6))
	*/
    Standard_Integer ElecPhysDataNum = 8;
    m_OneEAxisPhysDataNum = ElecPhysDataNum;

    vector<GridEdgeData *> theXtndRgnEdgeDatas;
    for(int i = 0; i < phi_num; ++i){
        TxSlab2D<Standard_Integer> theXtndRgn = GetFldsDefCntr()->GetGridGeom_Cyl3D()->GetZRGrid()->GetXtndRgn();
        GetFldsDefCntr()->GetGridGeom(i)->GetGridEdgeDatasNotOfMaterialTypeAlongAxis(0,
            theXtndRgn,
            theXtndRgnEdgeDatas);
    }
    Standard_Integer theEdgeDatasNum = theXtndRgnEdgeDatas.size();
    Standard_Integer theTotalEAxisPhysDataSize = theEdgeDatasNum * m_OneEAxisPhysDataNum;

    m_EAxisDatasSize = theTotalEAxisPhysDataSize;

    if(m_EAxisDatasPtr != NULL)
        aligned_free(m_EAxisDatasPtr);
    m_EAxisDatasPtr = (Standard_Real *)aligned_malloc(sizeof(Standard_Real) * theTotalEAxisPhysDataSize);
    memset(m_EAxisDatasPtr, 0, sizeof(Standard_Real) * theTotalEAxisPhysDataSize);

    Standard_Size currIndex = 0;
    Standard_Real *PhysDataptr = NULL;
    
    for(Standard_Size i = 0; i < theEdgeDatasNum; ++i){
        // Reset Edge EAxisData ptr
        PhysDataptr = m_EAxisDatasPtr + currIndex;
        theXtndRgnEdgeDatas[i]->ResetPhysDataPtr(m_OneEAxisPhysDataNum, PhysDataptr);
        currIndex = currIndex + m_OneEAxisPhysDataNum;
    }
}

void 
SI_SC_Matrix_EMFields_Cyl3D::
    CleanFaceDatas()
{
    vector<GridFaceData *> theXtndRgnFaceDatas;
    for(Standard_Integer i = 0; i < phi_num; ++i){
        TxSlab2D<Standard_Integer> theXtndRgn = GetFldsDefCntr()->GetGridGeom_Cyl3D()->GetZRGrid()->GetXtndRgn();
        GetFldsDefCntr()->GetGridGeom(i)->GetGridFaceDatasNotOfMaterialTypeOfSubRgn(0, theXtndRgn, theXtndRgnFaceDatas);
    }
    Standard_Integer theGeomDataNum = theXtndRgnFaceDatas.size();
    Standard_Integer theTotalPhysDataSize = theGeomDataNum * m_OneMphiPhysDataNum;

    Standard_Size currIndex = 0;
    for(Standard_Size i = 0; i < theGeomDataNum; ++i){
        theXtndRgnFaceDatas[i]->CleanPhysDataPtr();
    }

    // Clean MphiFuncArray_Cyl3D
    vector<GridFaceData *> &MphiFace = m_MCntrFields_Cyl3D->m_FaceMagFlds_Cyl3D->GetDatas();
    Standard_Size nFace = MphiFace.size();

    for (Standard_Size i = 0; i < nFace; i++)
    {
        m_MphiFuncArray_Cyl3D[i].m_Mag = NULL;
        m_MphiFuncArray_Cyl3D[i].m_Current = NULL;

        GridFaceData *currFace = MphiFace[i];
        const vector<T_Element> &theOutLineTElems = currFace->GetOutLineTEdge(); //GridEdgeData elements
        Standard_Integer nb = theOutLineTElems.size();
        for (Standard_Integer j = 0; j < nb; j++){
            m_MphiFuncArray_Cyl3D[i].m_Elec[j] = NULL;
        }
    }

    // Clean MphiPMLFuncArray_Cyl3D
    vector<GridFaceData *> &CPML_MphiFace = m_MCPMLFields_Cyl3D->m_FaceMagFlds_Cyl3D->GetDatas();
    nFace = CPML_MphiFace.size();

    for (Standard_Size i = 0; i < nFace; i++)
    {
        m_MphiPMLFuncArray_Cyl3D[i].m_Mag = NULL;
        m_MphiPMLFuncArray_Cyl3D[i].m_PM1 = NULL;
        m_MphiPMLFuncArray_Cyl3D[i].m_PM2 = NULL;

        GridFaceData *currFace = CPML_MphiFace[i];
        const vector<T_Element>& theOutLineTElems = currFace->GetOutLineTEdge();
        int nb = theOutLineTElems.size();
        for (Standard_Size j = 0; j < nb; j++)
        {
            m_MphiPMLFuncArray_Cyl3D[i].m_Elec[j] = NULL;
        }
    }
}

void 
SI_SC_Matrix_EMFields_Cyl3D::
    CleanEdgeDatas()
{
    vector<GridEdgeData *> theXtndRgnEdgeDatas;
    for(Standard_Integer i = 0; i < phi_num; ++i){
        TxSlab2D<Standard_Integer> theXtndRgn = GetFldsDefCntr()->GetGridGeom_Cyl3D()->GetZRGrid()->GetXtndRgn();
        GetFldsDefCntr()->GetGridGeom(i)->GetGridEdgeDatasNotOfMaterialTypeOfSubRgn(0, 
            theXtndRgn,
            false,
            theXtndRgnEdgeDatas);
    }
    Standard_Integer theEdgeDatasNum = theXtndRgnEdgeDatas.size();
    Standard_Integer theTotalEzrPhysDataSize = theEdgeDatasNum * m_OneEzrPhysDataNum;
    Standard_Integer theTotalMzrPhysDataSize = theEdgeDatasNum * m_OneMzrPhysDataNum;

    for(int i = 0; i < theEdgeDatasNum; ++i){
        theXtndRgnEdgeDatas[i]->CleanPhysDataPtr();
        theXtndRgnEdgeDatas[i]->CleanSweptPhysDataPtr();
    }

    // Clean EzrFuncArray_Cyl3D
    vector<GridEdgeData *> &Ezredge = m_ECntrFields_Cyl3D->m_EdgeElecFlds_Cyl3D->GetDatas();
    int nEdge = Ezredge.size();
    for(int i = 0; i < nEdge; ++i){
        m_EzrFuncArray_Cyl3D->m_Elec = NULL;
        m_EzrFuncArray_Cyl3D->m_Current = NULL;
        m_EzrFuncArray_Cyl3D->m_AE = NULL;
        m_EzrFuncArray_Cyl3D->m_BE = NULL;
        m_EzrFuncArray_Cyl3D->m_Elec_PreStep = NULL;

        GridEdgeData *currEdge = Ezredge[i];

        const vector<T_Element> &theOutLineTElems = currEdge->GetSharedTFace(); //GridFaceData
        int nb = theOutLineTElems.size();

        for (Standard_Size j = 0; j < nb; j++)
        {
            m_EzrFuncArray_Cyl3D[i].m_Mag[j] = NULL;
        }

        const vector<T_Element> theNearMEdges = currEdge->GetNearMEdges();
        nb = theNearMEdges.size();

        for(Standard_Size j = 0; j < nb; ++j){
            m_EzrFuncArray_Cyl3D[i].m_MagNear[j] = NULL;
        }
    }

    // Clean EzrPMLFuncArray_Cyl3D
    vector<GridEdgeData *> &CPML_Ezredge = m_ECPMLFields_Cyl3D->m_EdgeElecFlds_Cyl3D->GetDatas();
    nEdge = CPML_Ezredge.size();
    
    for (Standard_Size i = 0; i < nEdge; i++)
    {
        m_EzrPMLFuncArray_Cyl3D[i].m_Elec = NULL;
        m_EzrPMLFuncArray_Cyl3D[i].m_Elec_PreStep = NULL;
        m_EzrPMLFuncArray_Cyl3D[i].m_BE = NULL;
        m_EzrPMLFuncArray_Cyl3D[i].m_AE = NULL;
        m_EzrPMLFuncArray_Cyl3D[i].m_PE1 = NULL;
        m_EzrPMLFuncArray_Cyl3D[i].m_PE2 = NULL;

        GridEdgeData *currEdge = CPML_Ezredge[i];

        const vector<T_Element> &theOutLineTElems = currEdge->GetSharedTFace(); //GridFaceData
        int nb = theOutLineTElems.size();
        for (Standard_Size j = 0; j < nb; j++)
        {
            m_EzrPMLFuncArray_Cyl3D[i].m_Mag[j] = NULL;
        }

        const vector<T_Element> &theNearMEdges = currEdge->GetNearMEdges();
        nb = theNearMEdges.size();
        for(Standard_Size j = 0; j < nb; ++j){
            m_EzrPMLFuncArray_Cyl3D[i].m_MagNear[j] = NULL;
        }
    }

    // Clean MzrFuncArray_Cyl3D
    vector<GridEdgeData *> &MzrEdge = m_MCntrFields_Cyl3D->m_SweptFaceMagFlds_Cyl3D->GetDatas();
    nEdge = MzrEdge.size();

    for (Standard_Size i = 0; i < nEdge; i++)
    {
        m_MzrFuncArray_Cyl3D[i].m_Mag = NULL;
        m_MzrFuncArray_Cyl3D[i].m_Current = NULL;

        GridEdgeData *currEdge = MzrEdge[i];

        const vector<T_Element> &theOutLineTElems = MzrEdge[i]->GetOutLineDTEdges(); // GridVertexData elements
        Standard_Integer nb = theOutLineTElems.size();
        for (Standard_Integer j = 0; j < nb; j++){
            m_MzrFuncArray_Cyl3D[i].m_Elec[j] = NULL;
        }

        const vector<T_Element> &theNearEEdges    = MzrEdge[i]->GetNearEEdges();     // GridEdgeData elements in near slice
        nb = theNearEEdges.size();
        for(Standard_Integer j = 0; j < nb; ++j){
            m_MzrFuncArray_Cyl3D[i].m_ElecNear[j] = NULL;
        }
    }

    // Clean MzrPMLFuncArray_Cyl3D
    vector<GridEdgeData *> &CPML_MzrEdge = m_MCPMLFields_Cyl3D->m_SweptFaceMagFlds_Cyl3D->GetDatas();
    nEdge = CPML_MzrEdge.size();

    for (Standard_Size i = 0; i < nEdge; i++)
    {
        m_MzrPMLFuncArray_Cyl3D[i].m_Mag = NULL;
        m_MzrPMLFuncArray_Cyl3D[i].m_PM1 = NULL;
        m_MzrPMLFuncArray_Cyl3D[i].m_PM2 = NULL;

        GridEdgeData *currEdge = CPML_MzrEdge[i];

        const vector<T_Element> &theOutLineTElems = currEdge->GetOutLineDTEdges(); // GridVertexData
        int nb = theOutLineTElems.size();
        for (Standard_Size j = 0; j < nb; j++)
        {
            m_MzrPMLFuncArray_Cyl3D[i].m_Elec[j] = NULL;
        }

        const vector<T_Element> &theNearEEdges = currEdge->GetNearEEdges(); // GridEdgeData elements in near slice
        nb = theNearEEdges.size();
        for(Standard_Integer j = 0; j < nb; ++j){
            m_MzrPMLFuncArray_Cyl3D[i].m_ElecNear[j] = NULL;
        }
    }
}

void
SI_SC_Matrix_EMFields_Cyl3D::
    CleanNearEdgeDatas()
{
    vector<GridEdgeData*> &MzrEdge = m_MCntrFields_Cyl3D->m_SweptFaceMagFlds_Cyl3D->GetDatas();
    Standard_Size nEdge = MzrEdge.size();
    Standard_Integer theTotalEAxisPhysDataSize = nEdge * m_OneEAxisPhysDataNum;

    for(int i = 0; i < nEdge; ++i){
        const vector<T_Element> &theNearEEdges = MzrEdge[i]->GetNearEEdges();
        for(Standard_Integer j = 0; j < 2; ++j){
            GridVertexData *curVertex = (GridVertexData *)theNearEEdges[j].GetData();

            curVertex->CleanPhysDataPtr();
        }

    }
}

void
SI_SC_Matrix_EMFields_Cyl3D::
    CleanVertexDatas()
{
    vector<GridVertexData *> theXtndRgnVertexDatas;
    for(Standard_Integer i = 0; i < phi_num; ++i){
        TxSlab2D<Standard_Integer> theXtndRgn = GetFldsDefCntr()->GetGridGeom_Cyl3D()->GetZRGrid()->GetXtndRgn();
        GetFldsDefCntr()->GetGridGeom(i)->GetGridVertexDatasNotOfMaterialTypeOfSubRgn(0,
            theXtndRgn,
            false,
            theXtndRgnVertexDatas);
    }
    Standard_Integer nVertex = theXtndRgnVertexDatas.size();
    for(int i = 0; i < nVertex; ++i){
        theXtndRgnVertexDatas[i]->CleanSweptPhysDataPtr();
    }

    // Clean EphiFuncArray_Cyl3D
    vector<GridVertexData *> &EphiVertex = m_ECntrFields_Cyl3D->m_SweptEdgeElecFlds_Cyl3D->GetDatas();
    nVertex = EphiVertex.size();

    for (Standard_Size i = 0; i < nVertex; i++)
    {
        m_EphiFuncArray_Cyl3D->m_Elec = NULL;
        m_EphiFuncArray_Cyl3D->m_Current = NULL;
        m_EphiFuncArray_Cyl3D->m_AE = NULL;
        m_EphiFuncArray_Cyl3D->m_BE = NULL;
        m_EphiFuncArray_Cyl3D->m_Elec_PreStep = NULL;
        
        GridVertexData *currVertex = EphiVertex[i];
        const vector<T_Element> &theOutLineTElems = currVertex->GetSharedTDFaces(); //GridEdgeData
        int nb = theOutLineTElems.size();
        for (Standard_Size j = 0; j < nb; j++)
        {
            m_EphiFuncArray_Cyl3D[i].m_Mag[j] = NULL;
        }
    }

    // Clean EphiPMLFuncArray_Cyl3D
    vector<GridVertexData *> &CPML_EphiVertex = m_ECPMLFields_Cyl3D->m_SweptEdgeElecFlds_Cyl3D->GetDatas();
    nVertex = CPML_EphiVertex.size();

    for (Standard_Size i = 0; i < nVertex; i++)
    {
        m_EphiPMLFuncArray_Cyl3D[i].m_Elec = NULL;
        m_EphiPMLFuncArray_Cyl3D[i].m_Elec_PreStep = NULL;
        m_EphiPMLFuncArray_Cyl3D[i].m_BE = NULL;
        m_EphiPMLFuncArray_Cyl3D[i].m_AE = NULL;
        m_EphiPMLFuncArray_Cyl3D[i].m_PE1 = NULL;
        m_EphiPMLFuncArray_Cyl3D[i].m_PE2 = NULL;

        GridVertexData *currVertex = CPML_EphiVertex[i];
        const vector<T_Element> &theOutLineTElems = currVertex->GetSharedTDFaces(); //GridEdgeData
        int nb = theOutLineTElems.size();
        for (Standard_Size j = 0; j < nb; j++)
        {
            m_EphiPMLFuncArray_Cyl3D[i].m_Mag[j] = NULL;
        }
    }
}

void
SI_SC_Matrix_EMFields_Cyl3D::
    CleanEdgeAxisDatas()
{
    // vector<GridEdgeData *> theXtndRgnEdgeDatas;
    // for(int i = 0; i < phi_num; ++i){
    //     TxSlab2D<Standard_Integer> theXtndRgn = GetFldsDefCntr()->GetGridGeom_Cyl3D()->GetZRGrid()->GetXtndRgn();
    //     GetFldsDefCntr()->GetGridGeom(i)->GetGridEdgeDatasNotOfMaterialTypeOfSubRgn(0, 
    //         theXtndRgn,
    //         true,
    //         theXtndRgnEdgeDatas);
    // }
    // Standard_Integer theEdgeDatasNum = theXtndRgnEdgeDatas.size();
    // Standard_Integer theTotalEAxisPhysDataSize = theEdgeDatasNum * m_OneEAxisPhysDataNum;

    // std::cout << "theEdgeDatasNum = " << theEdgeDatasNum << std::endl;

    // for(int i = 0; i < theEdgeDatasNum; ++i){
    //     theXtndRgnEdgeDatas[i]->CleanPhysDataPtr();
    // }

    // Clean EAxisFuncArray_Cyl3D
    vector<GridEdgeData *>  &EAxisedge = m_ECntrFields_Cyl3D->m_EdgeElecFlds_Axis_Cyl3D->GetDatas();
    Cempic_Size                  nEdge = EAxisedge.size();
    for(Cempic_Size i = 0; i < nEdge; ++i){
        m_EAxisFuncArray_Cyl3D->m_Elec = NULL;
        m_EAxisFuncArray_Cyl3D->m_Current = NULL;
        m_EAxisFuncArray_Cyl3D->m_AE = NULL;
        m_EAxisFuncArray_Cyl3D->m_BE = NULL;
        m_EAxisFuncArray_Cyl3D->m_Elec_PreStep = NULL;

        GridEdgeData *currEdge = EAxisedge[i];
        const vector<T_Element> &theOutlineTElems = currEdge->GetSharedTFace();
        Standard_Integer nb = theOutlineTElems.size();

        for(Cempic_Size j = 0; j < nb; ++j){
            m_EAxisFuncArray_Cyl3D[i].m_Mag[j] = NULL;
        }
    }

    // Clean EAxisPMLFuncArray_Cyl3D
    vector<GridEdgeData *> &CPML_EAxisedge = m_ECPMLFields_Cyl3D->m_EdgeElecFlds_Axis_Cyl3D->GetDatas();
    nEdge = CPML_EAxisedge.size();

    for(Standard_Size i = 0; i < nEdge; ++i){
        m_EAxisPMLFuncArray_Cyl3D[i].m_Elec = NULL;
        m_EAxisPMLFuncArray_Cyl3D[i].m_Elec_PreStep = NULL;
        m_EAxisPMLFuncArray_Cyl3D[i].m_BE = NULL;
        m_EAxisPMLFuncArray_Cyl3D[i].m_AE = NULL;
        m_EAxisPMLFuncArray_Cyl3D[i].m_PE1 = NULL;
        m_EAxisPMLFuncArray_Cyl3D[i].m_PE2 = NULL;

        GridEdgeData *currEdge = CPML_EAxisedge[i];

        const vector<T_Element> &theOutLineTElems = currEdge->GetSharedTFace();
        int nb = theOutLineTElems.size();
        for(Standard_Size j = 0; j < nb; ++j){
            m_EAxisPMLFuncArray_Cyl3D[i].m_Mag[j] = NULL;
        }
    }
}