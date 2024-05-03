#include <SI_SC_MagFields.hxx>

#include<GridFaceData.cuh>
#include<GridEdgeData.hxx>
#include<GridFace.hxx>
#include<GridEdge.hxx>
#include<GridVertexData.hxx>

#include<SI_SC_IntegralEquation.hxx>


SI_SC_MagFields::
SI_SC_MagFields()
  :FieldsBase()
{
}


SI_SC_MagFields::
SI_SC_MagFields(const FieldsDefineCntr* theCntr, 
		     PhysDataDefineRule theRule)
  :FieldsBase(theCntr, theRule)
{
}


SI_SC_MagFields::
~SI_SC_MagFields()
{

}


void 
SI_SC_MagFields::
Setup()
{
  Standard_Real dt = this->GetDelTime();

  set<Standard_Integer>  theBndsDefine;
  GetFldsDefCntr()->GetFieldsDefineRules()->GetBndMagMaterialSet( theBndsDefine );

  m_FaceMagFlds = new FaceMagFldsBase(GetFldsDefCntr(), m_Rule);
  m_FaceMagFlds->SetMaterials(theBndsDefine); 
  m_FaceMagFlds->SetDelTime(dt);
  m_FaceMagFlds->Setup();


  m_SweptFaceMagFlds = new SweptFaceMagFldsBase(GetFldsDefCntr(), m_Rule);
  m_SweptFaceMagFlds->SetMaterials(theBndsDefine);
  m_SweptFaceMagFlds->SetDelTime(dt);
  m_SweptFaceMagFlds->Setup();
}


bool 
SI_SC_MagFields::
IsPhysDataMemoryLocated() const
{
  bool result = true;

  vector<GridFaceData*>&  theFaceDatas = m_FaceMagFlds->GetDatas();
  Standard_Size nf = theFaceDatas.size();
  for(Standard_Size index=0; index<nf; index++){
    bool tmp = theFaceDatas[index]->IsPhysDataDefined();
    result = result && tmp;
    tmp = theFaceDatas[index]->IsOutLineEdgePhysDataDefined();
    result = result && tmp;
  }
  
  vector<GridEdgeData*>& theSweptFaceData = m_SweptFaceMagFlds->GetDatas();
  Standard_Size ne = theSweptFaceData.size();
  for(Standard_Size index=0; index<ne; index++){
    bool tmp = theSweptFaceData[index]->IsPhysDataDefined();
    result = result && tmp;
    tmp = theSweptFaceData[index]->IsOutLineDEdgePhysDataDefined();
    result = result && tmp;
  }
  if(result) cout<<"SI_SC_MagFields-----------IsPhysDataMemoryLocated is OK!!!"<<endl;
  else cout<<"SI_SC_MagFields-----------IsPhysDataMemoryLocated is NOT OK!!!"<<endl;
  return result;
}



void 
SI_SC_MagFields::
Advance()
{
  DynObj::Advance();
}



void 
SI_SC_MagFields::
Advance_SI(const Standard_Real si_scale)
{
  Standard_Integer TemporalEFieldIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicElecField_PhysDataIndex();
  Standard_Integer TemporalHFieldIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicMagField_PhysDataIndex();
  Standard_Integer JMIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_JM_PhysDataIndex();

  Standard_Real dt = GetDelTime();

  vector<GridFaceData*>& theFaceDatas = m_FaceMagFlds->GetDatas();
  vector<GridEdgeData*>& theSweptFaceData = m_SweptFaceMagFlds->GetDatas();

  Advance_MagElems_SI_SC(theFaceDatas,
			 dt, si_scale,
			 TemporalHFieldIndex, JMIndex,
			 TemporalEFieldIndex);

  Advance_MagElems_SI_SC(theSweptFaceData,
			 dt, si_scale,
			 TemporalHFieldIndex, JMIndex,
			 TemporalEFieldIndex);
}





void 
SI_SC_MagFields::
Advance_SI_Damping(const Standard_Real si_scale)
{
  Standard_Integer TemporalHFieldIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicMagField_PhysDataIndex();
  Standard_Integer JMIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_JM_PhysDataIndex();
  Standard_Integer BEIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_BE_PhysDataIndex();

  Standard_Real dt = GetDelTime();

  vector<GridFaceData*>& theFaceDatas = m_FaceMagFlds->GetDatas();
  vector<GridEdgeData*>& theSweptFaceData = m_SweptFaceMagFlds->GetDatas();

  Advance_MagElems_SI_SC(theFaceDatas,
			 dt, si_scale,
			 TemporalHFieldIndex, JMIndex,
			 BEIndex);

  Advance_MagElems_SI_SC(theSweptFaceData,
			 dt, si_scale,
			 TemporalHFieldIndex, JMIndex,
			 BEIndex);
}




void 
SI_SC_MagFields::
ZeroPhysDatas()
{
  vector<GridEdgeData*>& theSweptFaceDatas = m_SweptFaceMagFlds->GetDatas();
  Standard_Size ne = theSweptFaceDatas.size();
  for(Standard_Size i=0; i<ne; i++){
    theSweptFaceDatas[i]->ZeroSweptPhysDatas();
    //cout<<"theSweptFaceDatas["<<i<<"]->GetSweptPhysDataNum() = "<<theSweptFaceDatas[i]->GetSweptPhysDataNum()<<endl;
    //cout<<"theSweptFaceDatas["<<i<<"]->GetSweptGeomDim() = "<<theSweptFaceDatas[i]->GetSweptGeomDim()<<endl;
    //cout<<"\t theSweptFaceDatas["<<i<<"]->GetGeomDim() = "<<theSweptFaceDatas[i]->GetGeomDim()<<endl;
  }

  vector<GridFaceData*>&  theFaceDatas = m_FaceMagFlds->GetDatas();
  Standard_Size nf = theFaceDatas.size();
  for(Standard_Size i=0; i<nf; i++){
    theFaceDatas[i]->ZeroPhysDatas();
    //cout<<"theFaceDatas["<<i<<"]->GetPhysDataNum() = "<<theFaceDatas[i]->GetPhysDataNum()<<endl;
    //cout<<"\t theFaceDatas["<<i<<"]->GetGeomDim() = "<<theFaceDatas[i]->GetGeomDim()<<endl;
  }
}

