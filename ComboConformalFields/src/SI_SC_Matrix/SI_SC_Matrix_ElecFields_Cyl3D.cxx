#include "SI_SC_Matrix_ElecFields_Cyl3D.hxx"

#include "ComboFields_Dynamic_Srcs_Cyl3D.hxx"
#include "SI_SC_IntegralEquation.hxx"

SI_SC_Matrix_ElecFields_Cyl3D::
SI_SC_Matrix_ElecFields_Cyl3D()
    :FieldsBase()
{

}

SI_SC_Matrix_ElecFields_Cyl3D::
SI_SC_Matrix_ElecFields_Cyl3D(const FieldsDefineCntr* theCntr, PhysDataDefineRule theRule)
    :FieldsBase(theCntr, theRule)
{

}

SI_SC_Matrix_ElecFields_Cyl3D::
~SI_SC_Matrix_ElecFields_Cyl3D()
{

}

vector<GridEdgeData*> &
SI_SC_Matrix_ElecFields_Cyl3D::
    GetEdgeDatas()
{
    return m_EdgeElecFlds_Cyl3D->GetDatas();
}

vector<GridVertexData*> &
SI_SC_Matrix_ElecFields_Cyl3D::
    GetVertexDatas()
{
    return m_SweptEdgeElecFlds_Cyl3D->GetDatas();
}

void 
SI_SC_Matrix_ElecFields_Cyl3D::
    ZeroPhysDatas()
{
    vector<GridEdgeData*> &theEdgeDatas = m_EdgeElecFlds_Cyl3D->GetDatas();
    Standard_Size ne = theEdgeDatas.size();
    for(Standard_Size i = 0; i < ne; ++i){
        theEdgeDatas[i]->ZeroPhysDatas();
    }

    vector<GridVertexData*> &theSweptEdgeDatas = m_SweptEdgeElecFlds_Cyl3D->GetDatas();
    Standard_Size nse = theSweptEdgeDatas.size();
    for(Standard_Size i = 0; i < nse; ++i){
        theSweptEdgeDatas[i]->ZeroSweptPhysDatas();
    }

    vector<GridEdgeData*> &theEdgeDatas_Axis = m_EdgeElecFlds_Axis_Cyl3D->GetDatas();
    Standard_Size nae = theEdgeDatas_Axis.size();
    for(Standard_Size i = 0; i < nae; ++i){
        theEdgeDatas_Axis[i]->ZeroPhysDatas();
    }
}

void
SI_SC_Matrix_ElecFields_Cyl3D::
Setup()
{
    Standard_Real dt = this->GetDelTime();

    set<Standard_Integer> theBndsDefine;
    GetFldsDefCntr()->GetFieldsDefineRules()->GetBndElecMaterialSet(theBndsDefine);

    m_EdgeElecFlds_Cyl3D = new EdgeElecFldsBase_Cyl3D(GetFldsDefCntr(), m_Rule);
    m_EdgeElecFlds_Cyl3D->SetMaterials(theBndsDefine);
    m_EdgeElecFlds_Cyl3D->SetDelTime(dt);
    m_EdgeElecFlds_Cyl3D->Setup();

    m_SweptEdgeElecFlds_Cyl3D = new SweptEdgeElecFldsBase_Cyl3D(GetFldsDefCntr(), m_Rule);
    m_SweptEdgeElecFlds_Cyl3D->SetMaterials(theBndsDefine);
    m_SweptEdgeElecFlds_Cyl3D->SetDelTime(dt);
    m_SweptEdgeElecFlds_Cyl3D->Setup();

    m_EdgeElecFlds_Axis_Cyl3D = new EdgeElecFldsBase_Axis_Cyl3D(GetFldsDefCntr(), m_Rule);
    m_EdgeElecFlds_Axis_Cyl3D->SetMaterials(theBndsDefine);
    m_EdgeElecFlds_Axis_Cyl3D->SetDelTime(dt);
    m_EdgeElecFlds_Axis_Cyl3D->Setup();

    SetupGridEdgeDatasEfficientLength();
}

void
SI_SC_Matrix_ElecFields_Cyl3D::
    SetupGridEdgeDatasEfficientLength()
{
    vector<GridEdgeData*> &theEdgeDatas = m_EdgeElecFlds_Cyl3D->GetDatas();
    Standard_Size ne = theEdgeDatas.size();
    for(Standard_Size i = 0; i < ne; ++i){
        theEdgeDatas[i]->ComputeEfficientLength();
    }

    vector<GridEdgeData*> &theEdgeDatas_Axis = m_EdgeElecFlds_Axis_Cyl3D->GetDatas();
    Standard_Size nae = theEdgeDatas_Axis.size();
    for(Standard_Size i = 0; i < nae; ++i){
        theEdgeDatas_Axis[i]->ComputeEfficientLength();
    }
}

void
SI_SC_Matrix_ElecFields_Cyl3D::
Advance()
{
    DynObj::Advance();
}

void 
SI_SC_Matrix_ElecFields_Cyl3D::
Advance_SI(const Standard_Real si_scale)
{
    Standard_Integer dynEFieldIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicElecField_PhysDataIndex();
    Standard_Integer dynHFieldIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicMagField_PhysDataIndex();
    Standard_Integer CurrentIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_J_PhysDataIndex();

    Standard_Real dt = GetDelTime();

    vector<GridEdgeData*> &theEdgeDatas = m_EdgeElecFlds_Cyl3D->GetDatas();
    vector<GridVertexData*> &theSweptEdgeDatas = m_SweptEdgeElecFlds_Cyl3D->GetDatas();
    vector<GridEdgeData*> &theEdgeDatas_Axis = m_EdgeElecFlds_Axis_Cyl3D->GetDatas(); 

    Advance_ElecElems_SI_SC(theEdgeDatas,
        dt, si_scale,
        dynEFieldIndex, CurrentIndex, dynHFieldIndex);

    Advance_ElecElems_SI_SC(theSweptEdgeDatas,
        dt, si_scale,
        dynEFieldIndex, CurrentIndex, dynHFieldIndex);
    
    Standard_Size m_PhiNumber = GetGridGeom_Cyl3D()->GetDimPhi();

    Advance_ElecElems_SI_SC_AlongAxis(theEdgeDatas_Axis, m_PhiNumber,
        dt, si_scale,
        dynEFieldIndex, CurrentIndex, dynHFieldIndex);
}

void
SI_SC_Matrix_ElecFields_Cyl3D::
    Advance_SI_Damping(const Standard_Real si_scale,
        const Standard_Real damping_scale)
{
    const FieldsDefineCntr *FldsDefCntr_tmp = GetFldsDefCntr();
    Standard_Integer dynEFieldIndex = FldsDefCntr_tmp->GetFieldsDefineRules()->Get_DynamicElecField_PhysDataIndex();
    Standard_Integer dynHFieldIndex = FldsDefCntr_tmp->GetFieldsDefineRules()->Get_DynamicMagField_PhysDataIndex();
    Standard_Integer CurrentIndex = FldsDefCntr_tmp->GetFieldsDefineRules()->Get_J_PhysDataIndex();

    Standard_Integer AEIndex = FldsDefCntr_tmp->GetFieldsDefineRules()->Get_AE_PhysDataIndex();
    Standard_Integer BEIndex = FldsDefCntr_tmp->GetFieldsDefineRules()->Get_BE_PhysDataIndex();
    Standard_Integer PREIndex = FldsDefCntr_tmp->GetFieldsDefineRules()->Get_PRE_PhysDataIndex();

    Standard_Real dt = GetDelTime();

    vector<GridEdgeData*> &theEdgeDatas = m_EdgeElecFlds_Cyl3D->GetDatas();
    vector<GridVertexData*> &theSweptEdgeDatas = m_SweptEdgeElecFlds_Cyl3D->GetDatas();
    vector<GridEdgeData*> &theEdgeDatas_Axis = m_EdgeElecFlds_Axis_Cyl3D->GetDatas();

    Advance_ElecElems_SI_SC_Damping(theEdgeDatas,
        dt, si_scale,
        dynEFieldIndex, CurrentIndex, dynHFieldIndex,
        damping_scale, AEIndex, BEIndex);

    Advance_ElecElems_SI_SC_Damping(theSweptEdgeDatas,
        dt, si_scale,
        dynEFieldIndex, CurrentIndex, dynHFieldIndex,
        damping_scale, AEIndex, BEIndex);

    Standard_Size m_PhiNumber = GetGridGeom_Cyl3D()->GetDimPhi();

    Advance_ElecElems_SI_SC_AlongAxis_Damping(theEdgeDatas_Axis, m_PhiNumber,
        dt, si_scale,
        dynEFieldIndex, CurrentIndex, dynHFieldIndex,
        damping_scale, AEIndex, BEIndex);
}

bool
SI_SC_Matrix_ElecFields_Cyl3D::
    IsPhysDataMemoryLocated() const
{
    bool result = true;

    vector<GridEdgeData*> &theEdgeDatas = m_EdgeElecFlds_Cyl3D->GetDatas();
    Standard_Size ne = theEdgeDatas.size();
    for(Standard_Size i = 0; i < ne; ++i){
        bool tmp = theEdgeDatas[i]->IsPhysDataDefined();
        result = result && tmp;
        tmp = theEdgeDatas[i]->IsSharedFacesPhysDataDefined();
        result = result && tmp;
    }

    vector<GridVertexData*> &theSweptEdgeDatas = m_SweptEdgeElecFlds_Cyl3D->GetDatas();
    Standard_Size nse = theSweptEdgeDatas.size();
    for(Standard_Size i = 0; i < nse; ++i){
        bool tmp = theSweptEdgeDatas[i]->IsPhysDataDefined();
        result = result && tmp;
        tmp = theSweptEdgeDatas[i]->IsSharedDFacesPhysDataDefined();
        result = result && tmp;
    }

    vector<GridEdgeData*> &theEdgeDatas_Axis = m_EdgeElecFlds_Axis_Cyl3D->GetDatas();
    Standard_Size nae = theEdgeDatas_Axis.size();
    for(Standard_Size i = 0; i < nae; ++i){
        bool tmp = theEdgeDatas_Axis[i]->IsPhysDataDefined();
        result = result && tmp;
        tmp = theEdgeDatas_Axis[i]->IsSharedFacesPhysDataDefined();
        result = result && tmp;
    }

    return result;
}