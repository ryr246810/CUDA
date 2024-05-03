#include <DynFlds_PoyntingFlux_Dgn.hxx>
#include <stdlib.h>

DynFlds_PoyntingFlux_Dgn::
DynFlds_PoyntingFlux_Dgn()
  :FieldsDgnBase()
{
  m_poynFlux = 0.0;
}



void
DynFlds_PoyntingFlux_Dgn::
Init(const FieldsDefineCntr* theCntr)
{
  FieldsDgnBase::Init(theCntr);
  m_poynFlux = 0.0;
}



DynFlds_PoyntingFlux_Dgn::
~DynFlds_PoyntingFlux_Dgn()
{

}



void 
DynFlds_PoyntingFlux_Dgn::
SetAttrib(const TxHierAttribSet& tha)
{
  string theName="";

  if(tha.hasOption("dir")){
    m_Dir = tha.getOption("dir");
  }else{
    cout<<"DynFlds_PoyntingFlux_Dgn::SetAttrib--------------error----dir"<<endl;
  }


  Standard_Size PhiNumber = GetGridGeom_Cyl3D()->GetDimPhi();
  if(PhiNumber == 1)
  {
  	m_PhiIndex = -1;
  }
  else if(tha.hasOption("Phi")){
    m_PhiIndex= tha.getOption("Phi");
  }else{
    cout<<"DynFlds_PoyntingData_Dgn::SetAttrib--------------error----Phi"<<endl;
  }

  std::vector<Standard_Real> lb,ub;

  if(tha.hasPrmVec("lowerBounds")){
    lb = tha.getPrmVec("lowerBounds");
    if(lb.size()<2){
      cout<<"DynFlds_PoyntingFlux_Dgn::SetAttrib--------lowerBounds are not set correctly-------------error"<<endl;
      exit(1);
    }
  }else{
    cout<<"DynFlds_PoyntingFlux_Dgn::SetAttrib--------lowerBounds are not set-------------error"<<endl;
    exit(1);
  }

  if(tha.hasPrmVec("upperBounds")){
    ub = tha.getPrmVec("upperBounds");
    if(ub.size()<2){
      cout<<"DynFlds_PoyntingFlux_Dgn::SetAttrib--------upperBounds are not set correctly-------------error"<<endl;
      exit(1);
    }
  }else{
    cout<<"DynFlds_PoyntingFlux_Dgn::SetAttrib--------upperBounds are not set-------------error"<<endl;
    exit(1);
  }

  TxSlab2D<Standard_Real> theRealRgn;
  for(Standard_Integer dir = 0; dir<2; dir++){
    theRealRgn.setLowerBound(dir, lb[dir]);
    theRealRgn.setUpperBound(dir, ub[dir]);
  }

  TxSlab2D<Standard_Size> theGridRgn;
  GetFldsDefCntr()->GetZRGrid()->ComputeBndBoxInGrid(theRealRgn, theGridRgn);

  if(theGridRgn.getLowerBound(m_Dir)!=theGridRgn.getUpperBound(m_Dir)){
    theGridRgn.setUpperBound(m_Dir, theGridRgn.getLowerBound(m_Dir));
  }

  TxSlab2D<Standard_Integer> thePhysRgn= GetFldsDefCntr()->GetZRGrid()->GetPhysRgn();
  TxSlab2D<Standard_Integer> tmpRgn;
  for(Standard_Integer dir = 0; dir<2; dir++){
    tmpRgn.setLowerBound(dir, (Standard_Integer)theGridRgn.getLowerBound(dir));
    tmpRgn.setUpperBound(dir, (Standard_Integer)theGridRgn.getUpperBound(dir));
  }

  m_Rgn = thePhysRgn & tmpRgn;



  if(tha.hasString("name")){
    theName = tha.getString("name");
  }else{
    cout<<"DynFlds_PoyntingFlux_Dgn::SetAttrib--------------error----name"<<endl;
  }

  SetName(theName);  // dynobj
}




Standard_Real 
DynFlds_PoyntingFlux_Dgn::
GetValue()
{

  return m_poynFlux;
}



void 
DynFlds_PoyntingFlux_Dgn::
ComputeTotalPoyntingFlux()
{
  m_poynFlux = 0.0;

  {
    Standard_Integer edgePhysDataIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicElecField_PhysDataIndex();
    Standard_Integer facePhysDataIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicMagField_PhysDataIndex();
    
    Standard_Integer Dir0 = m_Dir;
    Standard_Integer Dir1 = (Dir0+1)%2;

    Standard_Size indxVec[2];
    indxVec[Dir0] = m_Rgn.getUpperBound(Dir0);
    for(Standard_Size index1 = m_Rgn.getLowerBound(Dir1)-1; index1<m_Rgn.getUpperBound(Dir1)+1; index1++){
      indxVec[Dir1] = index1;
      AddFluxElement(indxVec, edgePhysDataIndex, facePhysDataIndex);
      }
  }

}






void 
DynFlds_PoyntingFlux_Dgn::
AddFluxElement(const Standard_Size globalIndxVec[2],
	       const int ElecPhysDataIndex, 
	       const int MagPhysDataIndex)
{
  Standard_Size currVertexIndex;
  Standard_Integer Dir1 = (m_Dir+1)%2;
  
  GetFldsDefCntr()->GetZRGrid()->FillVertexIndx(globalIndxVec,currVertexIndex);
  if(m_PhiIndex==-1)
  {
  GridVertexData* currVertexPtr = GetGridGeom()->GetGridVertices()+currVertexIndex;



  double Vertex_E=currVertexPtr->GetSweptPhysData(ElecPhysDataIndex);

  double Face_B=0;
  vector<GridFaceData*> theOutLineTElems_0=currVertexPtr->GetSharingGridFaceDatas();
  Standard_Integer nb_0 = theOutLineTElems_0.size();
  for(Standard_Integer index=0; index<nb_0; index ++){
      Face_B += theOutLineTElems_0[index]->GetPhysData(MagPhysDataIndex);
  }
  if(nb_0 !=0) Face_B /=nb_0;
 
  double Edge_B=0.0;
  const vector<T_Element>& theOutLineTElems_1 = currVertexPtr->GetSharingDivTEdges(m_Dir);
  Standard_Integer nb_1 = theOutLineTElems_1.size();
  for(Standard_Integer index=0; index<nb_1; index ++){
      Edge_B +=theOutLineTElems_1[index].GetData()->GetSweptPhysData(MagPhysDataIndex);
  }

  if(nb_1 !=0)
  {
	 Edge_B/=nb_1;
  }

  double Edge_E=0.0;
  const vector<T_Element>& theOutLineTElems_2 = currVertexPtr->GetSharingDivTEdges(Dir1);
  Standard_Integer nb_2 = theOutLineTElems_2.size();
  for(Standard_Integer index=0; index<nb_2; index ++){
      Edge_E +=theOutLineTElems_2[index].GetData()->GetPhysData(ElecPhysDataIndex);
  }

  if(nb_2 !=0)
  {
	 Edge_E/=nb_2;
  }

  Standard_Real L1 =GetFldsDefCntr()->GetZRGrid()->GetDualStep(1,globalIndxVec[1]) ;
  Standard_Real L2 = currVertexPtr->GetSweptGeomDim();

  m_poynFlux += (Edge_E*Face_B-Vertex_E*Edge_B)*L1*L2;
}
else 
{
  Standard_Size m_PhiNumber=GetGridGeom_Cyl3D()->GetDimPhi();

  for(int index=0;index<m_PhiNumber;index++){

  GridVertexData* currVertexPtr = GetGridGeom(index)->GetGridVertices()+currVertexIndex;
  double Vertex_E=currVertexPtr->GetSweptPhysData(ElecPhysDataIndex);

  double Face_B=0;
  vector<GridFaceData*> theOutLineTElems_0=currVertexPtr->GetSharingGridFaceDatas();
  Standard_Integer nb_0 = theOutLineTElems_0.size();
  for(Standard_Integer index=0; index<nb_0; index ++){
      Face_B += theOutLineTElems_0[index]->GetPhysData(MagPhysDataIndex);
  }
  if(nb_0 !=0) Face_B /=nb_0;
 
  double Edge_B=0.0;
  const vector<T_Element>& theOutLineTElems_1 = currVertexPtr->GetSharingDivTEdges(m_Dir);
  Standard_Integer nb_1 = theOutLineTElems_1.size();
  for(Standard_Integer index=0; index<nb_1; index ++){
      Edge_B +=theOutLineTElems_1[index].GetData()->GetSweptPhysData(MagPhysDataIndex);
  }

  if(nb_1 !=0)
  {
	 Edge_B/=nb_1;
  }

  double Edge_E=0.0;
  const vector<T_Element>& theOutLineTElems_2 = currVertexPtr->GetSharingDivTEdges(Dir1);
  Standard_Integer nb_2 = theOutLineTElems_2.size();
  for(Standard_Integer index=0; index<nb_2; index ++){
      Edge_E +=theOutLineTElems_2[index].GetData()->GetPhysData(ElecPhysDataIndex);
  }

  if(nb_2 !=0)
  {
	 Edge_E/=nb_2;
  }

  Standard_Real L1 =GetFldsDefCntr()->GetZRGrid()->GetDualStep(1,globalIndxVec[1]) ;
  Standard_Real L2 = currVertexPtr->GetSweptGeomDim();

  m_poynFlux += (Edge_E*Face_B-Vertex_E*Edge_B)*L1*L2;
}
}
}



void
DynFlds_PoyntingFlux_Dgn::
Advance()
{
  ComputeTotalPoyntingFlux();
  DynObj::Advance();
}

