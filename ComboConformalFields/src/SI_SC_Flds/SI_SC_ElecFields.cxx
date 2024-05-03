#include <SI_SC_ElecFields.hxx>

#include<GridFaceData.cuh>
#include<GridEdgeData.hxx>
#include<GridFace.hxx>
#include<GridEdge.hxx>
#include<GridVertexData.hxx>

#include<SI_SC_IntegralEquation.hxx>


SI_SC_ElecFields::
SI_SC_ElecFields()
  :FieldsBase()
{
}



SI_SC_ElecFields::
SI_SC_ElecFields(const FieldsDefineCntr* theCntr,
		      PhysDataDefineRule theRule)
  :FieldsBase(theCntr, theRule)
{
}



SI_SC_ElecFields::
~SI_SC_ElecFields()
{

}


vector<GridEdgeData*>& 
SI_SC_ElecFields::
GetEdgeDatas(){
	return m_EdgeElecFlds->GetDatas();
}


vector<GridVertexData*>& 
SI_SC_ElecFields::
GetVertexDatas(){
	return m_SweptEdgeElecFlds->GetDatas();
}


void 
SI_SC_ElecFields::
ZeroPhysDatas()
{
  vector<GridEdgeData*>& theEdgeDatas = m_EdgeElecFlds->GetDatas();
  Standard_Size ne = theEdgeDatas.size();
  for(Standard_Size i=0; i<ne; i++){
    theEdgeDatas[i]->ZeroPhysDatas();
    //cout<<"theEdgeDatas["<<i<<"]->GetPhysDataNum() = "<<theEdgeDatas[i]->GetPhysDataNum()<<endl;

    /*
    cout<<"theEdgeDatas["<<i<<"]->GetSweptPhysDataNum() = "<<theEdgeDatas[i]->GetSweptPhysDataNum()<<endl;
    cout<<"theEdgeDatas["<<i<<"] [epsilon, mu, sigma] = "
	<<theEdgeDatas[i]->GetEpsilon() <<", "
	<<theEdgeDatas[i]->GetMu()<<", "
	<<theEdgeDatas[i]->GetSigma()<<"]"<<endl;
    cout<<"\t theEdgeDatas["<<i<<"]->GetSweptGeomDim() = "<<theEdgeDatas[i]->GetSweptGeomDim()<<endl;
    cout<<"\t theEdgeDatas["<<i<<"]->GetDualSweptGeomDim() = "<<theEdgeDatas[i]->GetDualSweptGeomDim()<<endl;
    //*/
  }

  vector<GridVertexData*>&  theSweptEdgeDatas = m_SweptEdgeElecFlds->GetDatas();
  Standard_Size nse = theSweptEdgeDatas.size();
  for(Standard_Size i=0; i<nse; i++){
    theSweptEdgeDatas[i]->ZeroSweptPhysDatas();

    /*
    cout<<"theSweptEdgeDatas["<<i<<"]->GetSweptPhysDataNum() = "<<theSweptEdgeDatas[i]->GetSweptPhysDataNum()<<endl;
    cout<<"theSweptEdgeDatas["<<i<<"] [epsilon, mu, sigma] = "
	<<theSweptEdgeDatas[i]->GetEpsilon() <<", "
	<<theSweptEdgeDatas[i]->GetMu()<<", "
	<<theSweptEdgeDatas[i]->GetSigma()<<"]"<<endl;
    cout<<"\t theSweptEdgeDatas["<<i<<"]->GetSweptGeomDim() = "<<theSweptEdgeDatas[i]->GetSweptGeomDim()<<endl;
    cout<<"\t theSweptEdgeDatas["<<i<<"]->GetDualSweptGeomDim() = "<<theSweptEdgeDatas[i]->GetDualSweptGeomDim()<<endl;
    //*/
  }
}



void 
SI_SC_ElecFields::
Setup()
{
  Standard_Real dt = this->GetDelTime();

  set<Standard_Integer>  theBndsDefine;
  GetFldsDefCntr()->GetFieldsDefineRules()->GetBndElecMaterialSet( theBndsDefine );

  m_EdgeElecFlds = new EdgeElecFldsBase(GetFldsDefCntr(), m_Rule);
  m_EdgeElecFlds->SetMaterials(theBndsDefine); 
  m_EdgeElecFlds->SetDelTime(dt);
  m_EdgeElecFlds->Setup();


  m_SweptEdgeElecFlds = new SweptEdgeElecFldsBase(GetFldsDefCntr(), m_Rule);
  m_SweptEdgeElecFlds->SetMaterials(theBndsDefine);
  m_SweptEdgeElecFlds->SetDelTime(dt);
  m_SweptEdgeElecFlds->Setup();

  SetupGridEdgeDatasEfficientLength();
}



void 
SI_SC_ElecFields::
SetupGridEdgeDatasEfficientLength()
{
  vector<GridEdgeData*>& theEdgeDatas = m_EdgeElecFlds->GetDatas();
  Standard_Size ne = theEdgeDatas.size();
  for(Standard_Size i=0; i<ne; i++){
    theEdgeDatas[i]->ComputeEfficientLength();
  }
}



void 
SI_SC_ElecFields::
Advance()
{
  DynObj::Advance();
}



void 
SI_SC_ElecFields::
Advance_SI(const Standard_Real si_scale)
{
  Standard_Integer dynEFieldIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicElecField_PhysDataIndex();
  Standard_Integer dynHFieldIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicMagField_PhysDataIndex();
  Standard_Integer CurrentIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_J_PhysDataIndex();

  Standard_Real dt = GetDelTime();

  vector<GridEdgeData*>& theEdgeDatas = m_EdgeElecFlds->GetDatas();
  vector<GridVertexData*>& theSweptEdgeDatas = m_SweptEdgeElecFlds->GetDatas();

  Advance_ElecElems_SI_SC(theEdgeDatas,
			  dt, si_scale,
			  dynEFieldIndex, CurrentIndex, dynHFieldIndex);
 
  Advance_ElecElems_SI_SC(theSweptEdgeDatas,
			  dt, si_scale,
			  dynEFieldIndex, CurrentIndex, dynHFieldIndex);
}



void 
SI_SC_ElecFields::
Advance_SI_Damping(const Standard_Real si_scale, 
		   const Standard_Real damping_scale)
{
  Standard_Integer dynEFieldIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicElecField_PhysDataIndex();
  Standard_Integer dynHFieldIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicMagField_PhysDataIndex();
  Standard_Integer CurrentIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_J_PhysDataIndex();

  Standard_Integer AEIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_AE_PhysDataIndex();
  Standard_Integer BEIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_BE_PhysDataIndex();
  Standard_Integer PREIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_PRE_PhysDataIndex();

  Standard_Real dt = GetDelTime();

  vector<GridEdgeData*>& theEdgeDatas = m_EdgeElecFlds->GetDatas();
  vector<GridVertexData*>& theSweptEdgeDatas = m_SweptEdgeElecFlds->GetDatas();

  Advance_ElecElems_SI_SC_Damping(theEdgeDatas,
				  dt, si_scale,
				  dynEFieldIndex, CurrentIndex, dynHFieldIndex,
				  damping_scale, AEIndex, BEIndex);

  Advance_ElecElems_SI_SC_Damping(theSweptEdgeDatas,
				  dt, si_scale,
				  dynEFieldIndex, CurrentIndex, dynHFieldIndex,
				  damping_scale, AEIndex, BEIndex);

  /*
  Advance_ElecElems_SI_SC_Damping_new(theEdgeDatas,
				      dt, si_scale,
				      dynEFieldIndex, CurrentIndex, dynHFieldIndex,
				      damping_scale, AEIndex, BEIndex, PREIndex);

  Advance_ElecElems_SI_SC_Damping_new(theSweptEdgeDatas,
				      dt, si_scale,
				      dynEFieldIndex, CurrentIndex, dynHFieldIndex,
				      damping_scale, AEIndex, BEIndex, PREIndex);
  //*/
}



bool 
SI_SC_ElecFields::
IsPhysDataMemoryLocated() const
{
  cout<<"------------------------IsPhysDataMemoryLocated---------------------1"<<endl;
  bool result = true;

  vector<GridEdgeData*>& theEdgeDatas = m_EdgeElecFlds->GetDatas();
  Standard_Size ne = theEdgeDatas.size();
  for(Standard_Size i=0; i<ne; i++){
    bool tmp = theEdgeDatas[i]->IsPhysDataDefined();
    result = result && tmp;
    tmp = theEdgeDatas[i]->IsSharedFacesPhysDataDefined();
    result = result && tmp;
  }
  vector<GridVertexData*>&  theSweptEdgeDatas = m_SweptEdgeElecFlds->GetDatas();
  Standard_Size nse = theSweptEdgeDatas.size();
  for(Standard_Size i=0; i<nse; i++){
    bool tmp = theSweptEdgeDatas[i]->IsSweptPhysDataDefined();
    result = result && tmp;
    tmp = theSweptEdgeDatas[i]->IsSharedDFacesPhysDataDefined();
    result = result && tmp;
  }
  if(result) cout<<"SI_SC_ElecFields----IsPhysDataMemoryLocated  is  ok!!!"<<endl;
  else cout<<"SI_SC_ElecFields----IsPhysDataMemoryLocated  is  not  ok!!!"<<endl;
  return result;
}

