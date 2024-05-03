#include <DynFlds_Current_Dgn.hxx>
#include <stdlib.h>
#include <stdio.h>

DynFlds_Current_Dgn::
DynFlds_Current_Dgn()
  :FieldsDgnBase()
{
  m_Current = 0.0;
}



void
DynFlds_Current_Dgn::
Init(const FieldsDefineCntr* theCntr)
{
  FieldsDgnBase::Init(theCntr);
  m_Current = 0.0;
}



DynFlds_Current_Dgn::
~DynFlds_Current_Dgn()
{
}



void 
DynFlds_Current_Dgn::
SetAttrib(const TxHierAttribSet& tha)
{
  string theName="";

  if(tha.hasOption("dir")){
    m_Dir = tha.getOption("dir");
  }else{
    cout<<"DynFlds_Current_Dgn::SetAttrib--------------error----dir"<<endl;
  }

  Standard_Size PhiNumber = GetGridGeom_Cyl3D()->GetDimPhi();
  if(PhiNumber == 1)
  {
  	m_PhiIndex = -1;
  }
  else if(tha.hasOption("Phi")){
    m_PhiIndex= tha.getOption("Phi");
  }else{
    cout<<"DynFlds_VoltageData_Dgn::SetAttrib--------------error----Phi"<<endl;
  }

  std::vector<Standard_Real> lb,ub;
  if(tha.hasPrmVec("lowerBounds")){
    lb = tha.getPrmVec("lowerBounds");
    if(lb.size()<2){
      cout<<"DynFlds_Current_Dgn::SetAttrib--------lowerBounds are not set correctly-------------error"<<endl;
      exit(1);
    }
  }else{
    cout<<"DynFlds_Current_Dgn::SetAttrib--------lowerBounds are not set-------------error"<<endl;
    exit(1);
  }

  if(tha.hasPrmVec("upperBounds")){
    ub = tha.getPrmVec("upperBounds");
    if(ub.size()<2){
      cout<<"DynFlds_Current_Dgn::SetAttrib--------upperBounds are not set correctly-------------error"<<endl;
      exit(1);
    }
  }else{
    cout<<"DynFlds_Current_Dgn::SetAttrib--------upperBounds are not set-------------error"<<endl;
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

  SetName(theName);  // dynobj
}



Standard_Real 
DynFlds_Current_Dgn::
GetValue()
{

  return m_Current;
}



void 
DynFlds_Current_Dgn::
ComputeTotalCurrent()
{
  m_Current = 0.0;

    Standard_Integer edgePhysDataIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_J_PhysDataIndex();
    Standard_Integer Dir0 = m_Dir;
    Standard_Integer Dir1 = (Dir0+1)%2;
    
    Standard_Size indxVec[2];
    indxVec[Dir0] = m_Rgn.getUpperBound(Dir0);
    for(Standard_Size index1 = m_Rgn.getLowerBound(Dir1); index1<m_Rgn.getUpperBound(Dir1); index1++){
      indxVec[Dir1] = index1;
	AddFluxElement(indxVec, Dir0, edgePhysDataIndex);
      }
    }


void 
DynFlds_Current_Dgn::
AddFluxElement(const Standard_Size globalIndxVec[2],
	       const Standard_Integer edgeDir,
	       const int edgePhysDataIndex)
{
  Standard_Size indxVec[3];
  Standard_Size currEdgeIndex;
  GetFldsDefCntr()->GetZRGrid()->FillEdgeIndx(edgeDir,globalIndxVec,currEdgeIndex);

  if(m_PhiIndex==-1)
  {
  GridEdge* currEdgePtr = GetGridGeom()->GetGridEdges()[edgeDir] + currEdgeIndex;

  Standard_Real dI = 0.0;

  const vector<GridEdgeData*>& currEdges = currEdgePtr->GetEdges();

  size_t nb = currEdges.size();

  for(size_t i=0; i<nb; i++){
    dI += (currEdges[i]->GetPhysData(edgePhysDataIndex))/((Standard_Real)nb);
  }

  m_Current += dI;
}else{

  Standard_Size m_PhiNumber=GetGridGeom_Cyl3D()->GetDimPhi();

  for(int index=0;index<m_PhiNumber;index++){

  GridEdge* currEdgePtr = GetGridGeom(index)->GetGridEdges()[edgeDir] + currEdgeIndex;

  Standard_Real dI = 0.0;

  const vector<GridEdgeData*>& currEdges = currEdgePtr->GetEdges();

  size_t nb = currEdges.size();

  for(size_t i=0; i<nb; i++){
    dI += (currEdges[i]->GetPhysData(edgePhysDataIndex))/((Standard_Real)nb);
  }

  m_Current += dI;
  }
 }
}



void 
DynFlds_Current_Dgn::
Advance()
{
  ComputeTotalCurrent();
  DynObj::Advance();
}
