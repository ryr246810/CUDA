#include "SI_SC_Matrix_Mur_ElecFields.hxx"

#include "GridFaceData.cuh"
#include "GridEdgeData.hxx"
#include "GridFace.hxx"
#include "GridEdge.hxx"

#include "PhysConsts.hxx"
#include "BaseDataDefine.hxx"
#include <stdlib.h>
#include "SI_SC_Matrix_Mur_Equation.hxx"

SI_SC_Matrix_Mur_ElecFields::
SI_SC_Matrix_Mur_ElecFields()
    :FieldsBase()
{
    m_PhiIndex = -1;
}

SI_SC_Matrix_Mur_ElecFields::
SI_SC_Matrix_Mur_ElecFields(const FieldsDefineCntr* theCntr, const PortData& thePort)
    :FieldsBase(theCntr, INCLUDING)
{
    SetPort(thePort);
    m_PhiIndex = -1;
}

void 
SI_SC_Matrix_Mur_ElecFields::
SetPort(const PortData& thePort){
    m_MurPort = thePort;
}

SI_SC_Matrix_Mur_ElecFields::
~SI_SC_Matrix_Mur_ElecFields()
{
    m_MurPortEdgeDatas.clear();
    m_FreeSpaceEdgeDatas.clear();

    m_MurPortSweptEdgeDatas.clear();
    M_FreeSpaceSweptEdgeDatas.clear();
}

void 
SI_SC_Matrix_Mur_ElecFields::
Advance()
{
    DynObj::Advance();
}

void
SI_SC_Matrix_Mur_ElecFields::
Advance_SI(const Standard_Real si_scale)
{
    Standard_Real Dt = GetDelTime();

    Standard_Integer dynEFldIndx = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicElecField_PhysDataIndex();
    Standard_Integer preTStepEFldIndx = GetFldsDefCntr()->GetFieldsDefineRules()->Get_MUR_PreStep_PhysDataIndex();

    for(Standard_Size i = 0; i < m_MurPortEdgeDatas.size(); ++i){
        Advance_Mur_OneElecElem_SI_SC(m_MurPortEdgeDatas[i], m_FreeSpaceEdgeDatas[i], Dt,
            si_scale, m_VBar, dynEFldIndx, preTStepEFldIndx);
    }

    for(Standard_Size i = 0; i < m_MurPortSweptEdgeDatas.size(); ++i){
        Advance_Mur_OneElecElem_SI_SC(m_MurPortSweptEdgeDatas[i], M_FreeSpaceSweptEdgeDatas[i], Dt,
            si_scale, m_VBar, dynEFldIndx, preTStepEFldIndx);
    }
}

void
SI_SC_Matrix_Mur_ElecFields::
Advance_SI_Damping(const Standard_Real si_scale, const Standard_Real damping_scale)
{
    Standard_Real Dt = GetDelTime();

    Standard_Integer dynEFldIndx = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicElecField_PhysDataIndex();
    Standard_Integer preTStepEFldIndx = GetFldsDefCntr()->GetFieldsDefineRules()->Get_MUR_PreStep_PhysDataIndex();

    Standard_Integer AEIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_AE_PhysDataIndex();
    Standard_Integer BEIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_BE_PhysDataIndex();
    Standard_Integer PREIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_PRE_PhysDataIndex();

    for(Standard_Size i = 0; i < m_MurPortEdgeDatas.size(); ++i){
        Advance_Mur_OneElecElem_SI_SC_Damping(m_MurPortEdgeDatas[i], m_FreeSpaceEdgeDatas[i], Dt, si_scale,
            m_VBar, dynEFldIndx, preTStepEFldIndx, damping_scale, AEIndex, BEIndex);
    }

    for(Standard_Size i = 0; i < m_MurPortSweptEdgeDatas.size(); ++i){
        Advance_Mur_OneElecElem_SI_SC_Damping(m_MurPortSweptEdgeDatas[i], M_FreeSpaceSweptEdgeDatas[i], Dt, si_scale,
            m_VBar, dynEFldIndx, preTStepEFldIndx, damping_scale, AEIndex, BEIndex);
    }
}

void
SI_SC_Matrix_Mur_ElecFields::
ZeroPhysDatas()
{
    Standard_Size ne = m_MurPortEdgeDatas.size();

    for(Standard_Size i = 0; i < ne; ++i){
        m_MurPortEdgeDatas[i]->ZeroPhysDatas();
    }

    Standard_Size nv = m_MurPortSweptEdgeDatas.size();

    for(Standard_Size i = 0; i < nv; ++i){
        m_MurPortSweptEdgeDatas[i]->ZeroPhysDatas();
    }
}

void 
SI_SC_Matrix_Mur_ElecFields::
Setup()
{
    FieldsBase::Setup(); // do nothing
    SetupData();
    SetupVP();
    SetupGridEdgeDatasEfficientLength();
}

void
SI_SC_Matrix_Mur_ElecFields::
SetupData()
{
    SetupDataEdgeDatas();
    SetupDataSweptEdgeDatas();
}

void 
SI_SC_Matrix_Mur_ElecFields::
SetupDataEdgeDatas()
{
    TxSlab2D<Standard_Integer> theMurPortRgn;
    ComputeMurTypePortRgn(m_MurPort, theMurPortRgn);
    Standard_Integer theInterfaceGlobalIndx;
    ComputePortPhysStartIndex(m_MurPort, theInterfaceGlobalIndx);

    Standard_Real theErrEpsilon = GetZRGrid()->GetGridLengthEpsilon();
    m_Step = GetZRGrid()->GetStep(m_MurPort.m_Dir, theInterfaceGlobalIndx);

    TxSlab2D<Standard_Integer> theRgn = GetFldsDefCntr()->GetZRGrid()->GetPhysRgn() & theMurPortRgn; // theMurPortRgn在GridGeneration中延伸到了margin内部
    GetGridGeom(m_PhiIndex)->GetGridEdgeDatasOfMaterialTypeOfSubRgn( (Standard_Integer)MUR, theRgn, false, m_MurPortEdgeDatas);

    vector<GridEdgeData*>::iterator iter;
    for(iter = m_MurPortEdgeDatas.begin(); iter != m_MurPortEdgeDatas.end(); ++iter){
        GridEdgeData* currGridEdgeData = *iter;
        TxVector2D<Standard_Real> currGridEdgeDataMidPnt;
        currGridEdgeData->ComputeMidPntLocation(currGridEdgeDataMidPnt);

        GridEdge* currGridEdge = currGridEdgeData->GetBaseGridEdge();
        Standard_Integer dir0 = currGridEdgeData->GetDir();

        Standard_Size refGridEdgeVecIndx[2];
        currGridEdge->GetVecIndex(refGridEdgeVecIndx);
        refGridEdgeVecIndx[m_MurPort.m_Dir] = theInterfaceGlobalIndx;

        Standard_Size refGridEdgeScalarIndx;
        GetZRGrid()->FillEdgeIndx(dir0, refGridEdgeVecIndx, refGridEdgeScalarIndx);
        GridEdge* refGridEdge = GetGridGeom(m_PhiIndex)->GetGridEdges()[dir0] + refGridEdgeScalarIndx;
        vector<GridEdgeData*> refGridEdgeDatas = refGridEdge->GetEdges();

        bool isPushed = false;
        if(refGridEdgeDatas.size() >= 1){
            for(Standard_Size i = 0; i < refGridEdgeDatas.size(); ++i){
                TxVector2D<Standard_Real> refGridEdgeDataMidPnt;
                refGridEdgeDatas[i]->ComputeMidPntLocation(refGridEdgeDataMidPnt);
                Standard_Real theLengthErr1 = fabs(refGridEdgeDataMidPnt[dir0] - currGridEdgeDataMidPnt[dir0]);
                Standard_Real theLengthErr2 = fabs(refGridEdgeDatas[i]->GetLength() - currGridEdgeData->GetLength());

                if((theLengthErr1 < theErrEpsilon) && (theLengthErr2 < theErrEpsilon)){
                    m_FreeSpaceEdgeDatas.push_back(refGridEdgeDatas[i]);
                    isPushed = true;
                    break;
                }
                else{

                }
            }
        }

        if(!isPushed){
            cout << "error-----------------------------MurBndData is not set correctedly-----------------------101" << endl;
            exit(1);
        }
    }
}

void
SI_SC_Matrix_Mur_ElecFields::
SetupDataSweptEdgeDatas()
{
    TxSlab2D<Standard_Integer> theMurPortRgn;
    ComputeMurTypePortRgn(m_MurPort, theMurPortRgn);
    Standard_Integer theInterfaceGlobalIndx;
    ComputePortStartIndex(m_MurPort, theInterfaceGlobalIndx);

    Standard_Real theErrEpsilon = GetZRGrid()->GetGridLengthEpsilon();
    m_Step = GetZRGrid()->GetStep(m_MurPort.m_Dir, theInterfaceGlobalIndx);

    TxSlab2D<Standard_Integer> theRgn = GetFldsDefCntr()->GetZRGrid()->GetPhysRgn() & theMurPortRgn;
    GetGridGeom(m_PhiIndex)->GetGridVertexDatasOfMaterialTypeOfSubRgn((Standard_Integer)MUR, theRgn, true, m_MurPortSweptEdgeDatas);

    vector<GridVertexData*>::iterator iter;
    for(iter = m_MurPortSweptEdgeDatas.begin(); iter != m_MurPortSweptEdgeDatas.end(); ++iter){
        GridVertexData* currGridVertexData = *iter;

        TxVector2D<Standard_Real> currGridVertexDataPnt = currGridVertexData->GetLocation();
        Standard_Size refGridVertexVecIndx[2];
        currGridVertexData->GetVecIndex(refGridVertexVecIndx);
        refGridVertexVecIndx[m_MurPort.m_Dir] = theInterfaceGlobalIndx;

        Standard_Size refGridVertexScalarIndx;
        GetZRGrid()->FillVertexIndx(refGridVertexVecIndx, refGridVertexScalarIndx);

        GridVertexData* refGridVertexData = GetGridGeom(m_PhiIndex)->GetGridVertices() + refGridVertexScalarIndx;

        M_FreeSpaceSweptEdgeDatas.push_back(refGridVertexData);
    }
}

void 
SI_SC_Matrix_Mur_ElecFields::
SetupVP()
{
    Standard_Real theVP = 1.0;

    const map<Standard_Integer, Standard_Real, less<Standard_Integer> >* theMurPortDatas = this->GetGridBndDatas()->GetMurPortDatas();
    map<Standard_Integer, Standard_Real, less<Standard_Integer> >::const_iterator dataIter;

    dataIter = theMurPortDatas->find(m_MurPort.m_Index);
    if(dataIter != (theMurPortDatas->end())){
        theVP = dataIter->second;
    }

    m_VBar = theVP * mksConsts.c / m_Step;
}

void
SI_SC_Matrix_Mur_ElecFields::
SetupGridEdgeDatasEfficientLength()
{
    Standard_Size ne = 0;
    ne = m_MurPortEdgeDatas.size();

    for(Standard_Size i = 0; i < ne; ++i){
        m_MurPortEdgeDatas[i]->ComputeEfficientLength();
    }
}