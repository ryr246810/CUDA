#include <SI_SC_MagFields_Cyl3D.hxx>

#include<GridFaceData.cuh>
#include<GridEdgeData.hxx>
#include<GridFace.hxx>
#include<GridEdge.hxx>
#include<GridVertexData.hxx>

#include<SI_SC_IntegralEquation.hxx>


SI_SC_MagFields_Cyl3D::
SI_SC_MagFields_Cyl3D()
  :FieldsBase()
{
}


SI_SC_MagFields_Cyl3D::
SI_SC_MagFields_Cyl3D(const FieldsDefineCntr* theCntr, 
		     PhysDataDefineRule theRule)
  :FieldsBase(theCntr, theRule)
{
}


SI_SC_MagFields_Cyl3D::
~SI_SC_MagFields_Cyl3D()
{

}



void 
SI_SC_MagFields_Cyl3D::
Setup()
{
  Standard_Real dt = this->GetDelTime();

  set<Standard_Integer>  theBndsDefine;
  GetFldsDefCntr()->GetFieldsDefineRules()->GetBndMagMaterialSet( theBndsDefine );

  m_FaceMagFlds_Cyl3D = new FaceMagFldsBase_Cyl3D(GetFldsDefCntr(), m_Rule);
  m_FaceMagFlds_Cyl3D->SetMaterials(theBndsDefine); 
  m_FaceMagFlds_Cyl3D->SetDelTime(dt);
  m_FaceMagFlds_Cyl3D->Setup();


  m_SweptFaceMagFlds_Cyl3D = new SweptFaceMagFldsBase_Cyl3D(GetFldsDefCntr(), m_Rule);
  m_SweptFaceMagFlds_Cyl3D->SetMaterials(theBndsDefine);
  m_SweptFaceMagFlds_Cyl3D->SetDelTime(dt);
  m_SweptFaceMagFlds_Cyl3D->Setup();
}


bool 
SI_SC_MagFields_Cyl3D::
IsPhysDataMemoryLocated() const
{
  bool result = true;

  vector<GridFaceData*>&  theFaceDatas = m_FaceMagFlds_Cyl3D->GetDatas();
  Standard_Size nf = theFaceDatas.size();
  for(Standard_Size index=0; index<nf; index++){
    bool tmp = theFaceDatas[index]->IsPhysDataDefined();
    result = result && tmp;
    tmp = theFaceDatas[index]->IsOutLineEdgePhysDataDefined();
    result = result && tmp;
  }


  vector<GridEdgeData*>& theSweptFaceData = m_SweptFaceMagFlds_Cyl3D->GetDatas();
  Standard_Size ne = theSweptFaceData.size();
  for(Standard_Size index=0; index<ne; index++){
    bool tmp = theSweptFaceData[index]->IsPhysDataDefined();
    result = result && tmp;
    tmp = theSweptFaceData[index]->IsOutLineDEdgePhysDataDefined();
    result = result && tmp;
  }

  return result;
}



void 
SI_SC_MagFields_Cyl3D::
Advance()
{
  DynObj::Advance();
}



void 
SI_SC_MagFields_Cyl3D::
Advance_SI(const Standard_Real si_scale)
{
  Standard_Integer TemporalEFieldIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicElecField_PhysDataIndex();
  Standard_Integer TemporalHFieldIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicMagField_PhysDataIndex();
  Standard_Integer JMIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_JM_PhysDataIndex();

  Standard_Real dt = GetDelTime();

  vector<GridFaceData*>& theFaceDatas = m_FaceMagFlds_Cyl3D->GetDatas();
  vector<GridEdgeData*>& theSweptFaceData = m_SweptFaceMagFlds_Cyl3D->GetDatas();

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
SI_SC_MagFields_Cyl3D::
AddMagAlongAixs()
{
	Standard_Integer TemporalHFieldIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicMagField_PhysDataIndex();
	Standard_Integer m_PhiNumber = GetGridGeom_Cyl3D()->GetDimPhi();
	vector<GridEdgeData*> AixsEdgeDatas;
	AixsEdgeDatas.clear();
	GetGridGeom_Cyl3D()->GetAllGridEdgeDatasAlongAxis(AixsEdgeDatas);
	Standard_Integer nbe = AixsEdgeDatas.size();
	if(nbe % m_PhiNumber !=0)
  	{
		std::cout<<"void AddMagAlongAixs--------------------error!!"<<endl;
		exit(1);
  	}
  	Standard_Size n_edge = nbe /m_PhiNumber;
  	for(Standard_Size eindex=0; eindex< n_edge; eindex++){
    	vector<GridEdgeData*> theSameEdge;
    	for(Standard_Size i=0; i<m_PhiNumber; i++){
    		Standard_Size index = n_edge*i + eindex;
    		theSameEdge.push_back(AixsEdgeDatas[index]);
    	}
    	 Standard_Real result = 0;
    	for(Standard_Size m_phi=0; m_phi<theSameEdge.size(); m_phi++)
    	{
    		result += theSameEdge[m_phi]->GetSweptPhysData(TemporalHFieldIndex);
    	}
    	for(Standard_Size m_phi=0; m_phi<theSameEdge.size(); m_phi++)
    	{
    		theSameEdge[m_phi]->SetSweptPhysData(TemporalHFieldIndex, result);
    	}
	}
}


void 
SI_SC_MagFields_Cyl3D::
Advance_SI_Damping(const Standard_Real si_scale)
{
  Standard_Integer TemporalHFieldIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicMagField_PhysDataIndex();
  Standard_Integer JMIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_JM_PhysDataIndex();
  Standard_Integer BEIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_BE_PhysDataIndex();

  Standard_Real dt = GetDelTime();

  vector<GridFaceData*>& theFaceDatas = m_FaceMagFlds_Cyl3D->GetDatas();
  vector<GridEdgeData*>& theSweptFaceData = m_SweptFaceMagFlds_Cyl3D->GetDatas();

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
SI_SC_MagFields_Cyl3D::
ZeroPhysDatas()
{
  vector<GridEdgeData*>& theSweptFaceDatas = m_SweptFaceMagFlds_Cyl3D->GetDatas();
  Standard_Size ne = theSweptFaceDatas.size();
  for(Standard_Size i=0; i<ne; i++){
    theSweptFaceDatas[i]->ZeroSweptPhysDatas();
    //cout<<"theSweptFaceDatas["<<i<<"]->GetSweptPhysDataNum() = "<<theSweptFaceDatas[i]->GetSweptPhysDataNum()<<endl;
    //cout<<"theSweptFaceDatas["<<i<<"]->GetSweptGeomDim() = "<<theSweptFaceDatas[i]->GetSweptGeomDim()<<endl;
    //cout<<"\t theSweptFaceDatas["<<i<<"]->GetGeomDim() = "<<theSweptFaceDatas[i]->GetGeomDim()<<endl;
  }

  vector<GridFaceData*>&  theFaceDatas = m_FaceMagFlds_Cyl3D->GetDatas();
  Standard_Size nf = theFaceDatas.size();
  for(Standard_Size i=0; i<nf; i++){
    theFaceDatas[i]->ZeroPhysDatas();
    //cout<<"theFaceDatas["<<i<<"]->GetPhysDataNum() = "<<theFaceDatas[i]->GetPhysDataNum()<<endl;
    //cout<<"\t theFaceDatas["<<i<<"]->GetGeomDim() = "<<theFaceDatas[i]->GetGeomDim()<<endl;
  }
}

