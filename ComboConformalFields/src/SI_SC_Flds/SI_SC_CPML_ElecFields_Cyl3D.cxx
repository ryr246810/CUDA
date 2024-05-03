#include <SI_SC_CPML_ElecFields_Cyl3D.hxx>

#include<GridFaceData.cuh>
#include<GridEdgeData.hxx>
#include<GridFace.hxx>
#include<GridEdge.hxx>


#include<SI_SC_CPML_Equation.hxx>

//#define FIELDSDATA_DEBUG


SI_SC_CPML_ElecFields_Cyl3D::
SI_SC_CPML_ElecFields_Cyl3D()
  :SI_SC_ElecFields_Cyl3D()
{
}


SI_SC_CPML_ElecFields_Cyl3D::
SI_SC_CPML_ElecFields_Cyl3D(const FieldsDefineCntr* theCntr)
  :SI_SC_ElecFields_Cyl3D(theCntr, INCLUDING)
{
}


SI_SC_CPML_ElecFields_Cyl3D::
~SI_SC_CPML_ElecFields_Cyl3D()
{

}


void 
SI_SC_CPML_ElecFields_Cyl3D::
Setup()
{
  Standard_Real dt = this->GetDelTime();

  set<Standard_Integer>  theBndsDefine;
  GetFldsDefCntr()->GetFieldsDefineRules()->GetBndElecMaterialSet( theBndsDefine );

  m_EdgeElecFlds_Cyl3D = new EdgeElecFldsBase_Cyl3D(GetFldsDefCntr(), m_Rule);
  m_EdgeElecFlds_Cyl3D->clearMaterials();
  m_EdgeElecFlds_Cyl3D->AppendingMaterial(PML);
  m_EdgeElecFlds_Cyl3D->SetDelTime(dt);
  m_EdgeElecFlds_Cyl3D->Setup();


  m_SweptEdgeElecFlds_Cyl3D = new SweptEdgeElecFldsBase_Cyl3D(GetFldsDefCntr(), m_Rule);
  m_SweptEdgeElecFlds_Cyl3D->clearMaterials();
  m_SweptEdgeElecFlds_Cyl3D->AppendingMaterial(PML);
  m_SweptEdgeElecFlds_Cyl3D->SetDelTime(dt);
  m_SweptEdgeElecFlds_Cyl3D->Setup();


  m_EdgeElecFlds_Axis_Cyl3D = new EdgeElecFldsBase_Axis_Cyl3D(GetFldsDefCntr(), m_Rule);
  m_EdgeElecFlds_Axis_Cyl3D->clearMaterials(); 
  m_EdgeElecFlds_Axis_Cyl3D->AppendingMaterial(PML);
  m_EdgeElecFlds_Axis_Cyl3D->SetDelTime(dt);
  m_EdgeElecFlds_Axis_Cyl3D->Setup();

  SetupGridEdgeDatasEfficientLength();
}


bool 
SI_SC_CPML_ElecFields_Cyl3D::
IsPhysDataMemoryLocated() const
{
  return  SI_SC_ElecFields_Cyl3D::IsPhysDataMemoryLocated();
}


void 
SI_SC_CPML_ElecFields_Cyl3D::
Advance()
{
  DynObj::Advance();
}


void 
SI_SC_CPML_ElecFields_Cyl3D::
Advance_SI(const Standard_Real si_scale)
{
  Standard_Integer dynEFieldIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicElecField_PhysDataIndex();
  Standard_Integer dynHFieldIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicMagField_PhysDataIndex();

  Standard_Integer thePE1Index = GetFldsDefCntr()->GetFieldsDefineRules()->Get_CPML_PE1_PhysDataIndex();
  Standard_Integer thePE2Index = GetFldsDefCntr()->GetFieldsDefineRules()->Get_CPML_PE2_PhysDataIndex();

  Standard_Real dt = GetDelTime();

  vector<GridEdgeData*>& theEdgeDatas = m_EdgeElecFlds_Cyl3D->GetDatas();
  vector<GridVertexData*>&  theSweptEdgeDatas = m_SweptEdgeElecFlds_Cyl3D->GetDatas();
  vector<GridEdgeData*>& theEdgeDatas_Axis = m_EdgeElecFlds_Axis_Cyl3D->GetDatas();

  Advance_CPML_ElecElems_SI_SC(theEdgeDatas, dt, si_scale, dynEFieldIndex, thePE1Index, thePE2Index, dynHFieldIndex);
  Advance_CPML_ElecElems_SI_SC(theSweptEdgeDatas, dt, si_scale, dynEFieldIndex, thePE1Index, thePE2Index, dynHFieldIndex);


  Standard_Size m_PhiNumber= GetGridGeom_Cyl3D()->GetDimPhi();

  Advance_CPML_ElecElems_SI_SC_AlongAxis(theEdgeDatas_Axis,m_PhiNumber, dt, si_scale, dynEFieldIndex, thePE1Index, thePE2Index, dynHFieldIndex);
}


void 
SI_SC_CPML_ElecFields_Cyl3D::
Advance_SI_Damping(const Standard_Real si_scale,  
		   const Standard_Real damping_scale)
{
  Standard_Integer dynEFieldIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicElecField_PhysDataIndex();
  Standard_Integer dynHFieldIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicMagField_PhysDataIndex();

  Standard_Integer thePE1Index = GetFldsDefCntr()->GetFieldsDefineRules()->Get_CPML_PE1_PhysDataIndex();
  Standard_Integer thePE2Index = GetFldsDefCntr()->GetFieldsDefineRules()->Get_CPML_PE2_PhysDataIndex();

  Standard_Integer AEIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_CPML_AE_PhysDataIndex();
  Standard_Integer BEIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_CPML_BE_PhysDataIndex();
  Standard_Integer PREIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_CPML_PRE_PhysDataIndex();

  Standard_Real dt = GetDelTime();

  vector<GridEdgeData*>& theEdgeDatas = m_EdgeElecFlds_Cyl3D->GetDatas();
  vector<GridVertexData*>& theSweptEdgeDatas = m_SweptEdgeElecFlds_Cyl3D->GetDatas();

  vector<GridEdgeData*>& theEdgeDatas_Axis = m_EdgeElecFlds_Axis_Cyl3D->GetDatas();

  Advance_CPML_ElecElems_SI_SC_Damping(theEdgeDatas,
				       dt, si_scale,
				       dynEFieldIndex, thePE1Index, thePE2Index, dynHFieldIndex,
				       damping_scale, AEIndex, BEIndex);
  
  Advance_CPML_ElecElems_SI_SC_Damping(theSweptEdgeDatas,
				       dt, si_scale,
				       dynEFieldIndex, thePE1Index, thePE2Index, dynHFieldIndex,
				       damping_scale, AEIndex, BEIndex);
  

  Standard_Size m_PhiNumber= GetGridGeom_Cyl3D()->GetDimPhi();

  Advance_CPML_ElecElems_SI_SC_AlongAxis_Damping(theEdgeDatas_Axis,m_PhiNumber,
				       dt, si_scale,
				       dynEFieldIndex, thePE1Index, thePE2Index, dynHFieldIndex,
				       damping_scale, AEIndex, BEIndex);
  /*
  Advance_CPML_ElecElems_SI_SC_Damping_new(theEdgeDatas,
					   dt, si_scale,
					   dynEFieldIndex, thePE1Index, thePE2Index, dynHFieldIndex,
					   damping_scale, AEIndex, BEIndex,PREIndex);

  Advance_CPML_ElecElems_SI_SC_Damping_new(theSweptEdgeDatas,
					   dt, si_scale,
					   dynEFieldIndex, thePE1Index, thePE2Index, dynHFieldIndex,
					   damping_scale, AEIndex, BEIndex,PREIndex);
  //*/
}



void 
SI_SC_CPML_ElecFields_Cyl3D::
ZeroPhysDatas()
{
  vector<GridEdgeData*>& theEdgeDatas = m_EdgeElecFlds_Cyl3D->GetDatas();
  Standard_Size ne = theEdgeDatas.size();
  for(Standard_Size i=0; i<ne; i++){
    theEdgeDatas[i]->ZeroPhysDatas();
    //cout<<"theEdgeDatas["<<i<<"]->GetPhysDataNum() = "<<theEdgeDatas[i]->GetPhysDataNum()<<endl;
  }

  vector<GridVertexData*>&  theSweptEdgeDatas = m_SweptEdgeElecFlds_Cyl3D->GetDatas();
  Standard_Size nse = theSweptEdgeDatas.size();
  for(Standard_Size i=0; i<nse; i++){
    theSweptEdgeDatas[i]->ZeroSweptPhysDatas();
    //cout<<"theSweptEdgeDatas["<<i<<"]->GetSweptPhysDataNum() = "<<theSweptEdgeDatas[i]->GetSweptPhysDataNum()<<endl;
  }

  vector<GridEdgeData*>& theEdgeDatas_Axis = m_EdgeElecFlds_Axis_Cyl3D->GetDatas();
  Standard_Size nae = theEdgeDatas_Axis.size();
  for(Standard_Size i=0; i<nae; i++){
    theEdgeDatas_Axis[i]->ZeroPhysDatas();
  }
}



void 
SI_SC_CPML_ElecFields_Cyl3D::
Setup_PML_a_b()
{
  Standard_Real a = 0.;
  Standard_Real b = 0.;
  Standard_Real dt = GetDelTime();

  vector<GridEdgeData*>& theEdgeDatas = m_EdgeElecFlds_Cyl3D->GetDatas();
  for(Standard_Integer i = 0; i<theEdgeDatas.size(); i++){
    for(Standard_Integer dir=0; dir<2; dir++){
      Compute_a_b_SI_SC(theEdgeDatas[i], 
			dir, 
			dt, 
			a, 
			b);

      theEdgeDatas[i]->SetPML_a(dir, a);
      theEdgeDatas[i]->SetPML_b(dir, b);
    }
  }


  vector<GridVertexData*>&  theSweptEdgeDatas = m_SweptEdgeElecFlds_Cyl3D->GetDatas();
  for(Standard_Integer i = 0; i<theSweptEdgeDatas.size(); i++){
    for(Standard_Integer dir=0; dir<2; dir++){
      Compute_a_b_SI_SC(theSweptEdgeDatas[i], 
			dir, 
			dt, 
			a, 
			b);
      theSweptEdgeDatas[i]->SetPML_a(dir, a);
      theSweptEdgeDatas[i]->SetPML_b(dir, b);
    }
  }


  vector<GridEdgeData*>& theEdgeDatas_Axis = m_EdgeElecFlds_Axis_Cyl3D->GetDatas();
  for(Standard_Integer i = 0; i<theEdgeDatas_Axis.size(); i++){
    for(Standard_Integer dir=0; dir<2; dir++){
      Compute_a_b_SI_SC(theEdgeDatas_Axis[i], 
			dir, 
			dt, 
			a, 
			b);
      theEdgeDatas_Axis[i]->SetPML_a(dir, a);
      theEdgeDatas_Axis[i]->SetPML_b(dir, b);
    }
  }


}



void 
SI_SC_CPML_ElecFields_Cyl3D::
Write_PML_a_b(std::ostream & theoutstream) const
{
  vector<GridEdgeData*>& theEdgeDatas = m_EdgeElecFlds_Cyl3D->GetDatas();
  for(Standard_Integer i = 0; i<theEdgeDatas.size(); i++){
    theoutstream<<" sigma=(";
    theoutstream<<theEdgeDatas[i]->GetPMLSigma(0)<<","<<theEdgeDatas[i]->GetPMLSigma(1)<<")";

    theoutstream<<" alpha=(";
    theoutstream<<theEdgeDatas[i]->GetPMLAlpha(0)<<","<<theEdgeDatas[i]->GetPMLAlpha(1)<<")";

    theoutstream<<" kappa=(";
    theoutstream<<theEdgeDatas[i]->GetPMLKappa(0)<<","<<theEdgeDatas[i]->GetPMLKappa(1)<<")";

    theoutstream<<" a=(";
    theoutstream<<theEdgeDatas[i]->GetPML_a(0)<<","<<theEdgeDatas[i]->GetPML_a(1)<<")" ;

    theoutstream<<" b=(";
    theoutstream<<theEdgeDatas[i]->GetPML_b(0)<<","<<theEdgeDatas[i]->GetPML_b(1)<<")";
    theoutstream<<endl;
  }

  vector<GridVertexData*>&  theSweptEdgeDatas = m_SweptEdgeElecFlds_Cyl3D->GetDatas();
  for(Standard_Integer i = 0; i<theSweptEdgeDatas.size(); i++){
    theoutstream<<" sigma=(";
    theoutstream<<theSweptEdgeDatas[i]->GetPMLSigma(0)<<","<<theSweptEdgeDatas[i]->GetPMLSigma(1)<<")";

    theoutstream<<" alpha=(";
    theoutstream<<theSweptEdgeDatas[i]->GetPMLAlpha(0)<<","<<theSweptEdgeDatas[i]->GetPMLAlpha(1)<<")";

    theoutstream<<" kappa=(";
    theoutstream<<theSweptEdgeDatas[i]->GetPMLKappa(0)<<","<<theSweptEdgeDatas[i]->GetPMLKappa(1)<<")";

    theoutstream<<" a=(";
    theoutstream<<theSweptEdgeDatas[i]->GetPML_a(0)<<","<<theSweptEdgeDatas[i]->GetPML_a(1)<<")" ;

    theoutstream<<" b=(";
    theoutstream<<theSweptEdgeDatas[i]->GetPML_b(0)<<","<<theSweptEdgeDatas[i]->GetPML_b(1)<<")";
    theoutstream<<endl;
  }

  vector<GridEdgeData*>& theEdgeDatas_Axis = m_EdgeElecFlds_Axis_Cyl3D->GetDatas();
  for(Standard_Integer i = 0; i<theEdgeDatas_Axis.size(); i++){
    theoutstream<<" sigma=(";
    theoutstream<<theEdgeDatas_Axis[i]->GetPMLSigma(0)<<","<<theEdgeDatas_Axis[i]->GetPMLSigma(1)<<")";

    theoutstream<<" alpha=(";
    theoutstream<<theEdgeDatas_Axis[i]->GetPMLAlpha(0)<<","<<theEdgeDatas_Axis[i]->GetPMLAlpha(1)<<")";

    theoutstream<<" kappa=(";
    theoutstream<<theEdgeDatas_Axis[i]->GetPMLKappa(0)<<","<<theEdgeDatas_Axis[i]->GetPMLKappa(1)<<")";

    theoutstream<<" a=(";
    theoutstream<<theEdgeDatas_Axis[i]->GetPML_a(0)<<","<<theEdgeDatas_Axis[i]->GetPML_a(1)<<")" ;

    theoutstream<<" b=(";
    theoutstream<<theEdgeDatas_Axis[i]->GetPML_b(0)<<","<<theEdgeDatas_Axis[i]->GetPML_b(1)<<")";
    theoutstream<<endl;
  }
}
