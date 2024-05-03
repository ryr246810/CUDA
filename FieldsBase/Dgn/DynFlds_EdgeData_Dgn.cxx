#include <DynFlds_EdgeData_Dgn.hxx>



DynFlds_EdgeData_Dgn::
DynFlds_EdgeData_Dgn()
  :FieldsDgnBase()
{
  m_BaseEdge = NULL;
  m_Data = 0.0;
}



void
DynFlds_EdgeData_Dgn::
Init(const FieldsDefineCntr* theCntr)
{
  FieldsDgnBase::Init(theCntr);
  m_Data = 0.0;
}



DynFlds_EdgeData_Dgn::~DynFlds_EdgeData_Dgn()
{

}



void 
DynFlds_EdgeData_Dgn::
SetAttrib(const TxHierAttribSet& tha)
{
  Standard_Integer theDir = 0;
  string theName="";

  if(tha.hasOption("dir")){
    theDir = tha.getOption("dir");
  }else{
    cout<<"DynFlds_EdgeData_Dgn::SetAttrib--------------error----dir"<<endl;
  }

  Standard_Size PhiNumber = GetGridGeom_Cyl3D()->GetDimPhi();
  if(PhiNumber == 1)
  {
  	m_PhiIndex = -1;
  }
  else if(tha.hasOption("Phi")){
    m_PhiIndex= tha.getOption("Phi");
  }else{
    cout<<"DynFlds_EdgeData_Dgn::SetAttrib--------------error----Phi"<<endl;
  }

  Standard_Size edgeIndxVec[2];
  if(tha.hasPrmVec("location")){
    Standard_Real theLocation[2];
    vector<Standard_Real> theData = tha.getPrmVec("location");
    if(theData.size()>=2){
      theLocation[0] = theData[0];
      theLocation[1] = theData[1];
    }else{
      cout<<"DynFlds_EdgeData_Dgn::SetAttrib--------------error----location"<<endl;
    }
    GetFldsDefCntr()->GetZRGrid()->ComputeLocationInGrid(theLocation, edgeIndxVec);
  }else if(tha.hasOptVec("locationIndex")){
    vector<int> theindex = tha.getOptVec("locationIndex");
    if(theindex.size()>=2){
      edgeIndxVec[0] = theindex[0];
      edgeIndxVec[1] = theindex[1];
    }else{
      cout<<"DynFlds_EdgeData_Dgn::SetAttrib--------------error----locationIndex-----1"<<endl;
    }
  }else{
    cout<<"DynFlds_EdgeData_Dgn::SetAttrib--------------error----locationIndex-----2"<<endl;
  }


  if(tha.hasString("name")){
    theName = tha.getString("name");
  }else{
    cout<<"DynFlds_EdgeData_Dgn::SetAttrib--------------error----name"<<endl;
  }

  SetName(theName);  // dynobj
  InitParamt(theDir, edgeIndxVec);  // poynting
}



void 
DynFlds_EdgeData_Dgn::
InitParamt(const Standard_Integer theDir, 
	   const Standard_Size edgeIndxVec[2])
{

  TxSlab2D<Standard_Integer> theRgn  = GetFldsDefCntr()->GetZRGrid()->GetPhysRgn();

  if( ( (edgeIndxVec[0]>=theRgn.getLowerBound(0)) && (edgeIndxVec[0]<theRgn.getUpperBound(0)) ) &&
      ( (edgeIndxVec[1]>=theRgn.getLowerBound(1)) && (edgeIndxVec[1]<theRgn.getUpperBound(1)) ) ) { 
    Standard_Size currEdgeIndex;
    GetFldsDefCntr()->GetZRGrid()->FillEdgeIndx(theDir, edgeIndxVec, currEdgeIndex);
    m_BaseEdge = GetGridGeom(m_PhiIndex)->GetGridEdges()[theDir] + currEdgeIndex;
  }else{
    m_BaseEdge = NULL;
  }
}



Standard_Real 
DynFlds_EdgeData_Dgn::
GetValue()
{
  return m_Data;
}



void 
DynFlds_EdgeData_Dgn::
ComputeData()
{
  //*
  m_Data = 0.0;

  Standard_Integer edgePhysDataIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicElecField_PhysDataIndex();

  if(m_BaseEdge!=NULL){
    const vector<GridEdgeData*>& currEdges = m_BaseEdge->GetEdges();
    size_t nb = currEdges.size(); 
    for(size_t i=0; i<nb; i++){
      m_Data += currEdges[i]->GetPhysData(edgePhysDataIndex)/nb;
    }
  }
  //*/

  /*
  m_Data = 0.0;

  Standard_Integer edgePhysDataIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicMagField_PhysDataIndex();

  if(m_BaseEdge!=NULL){
    const vector<GridEdgeData*>& currEdges = m_BaseEdge->GetEdges();
    size_t nb = currEdges.size();
    for(size_t i=0; i<nb; i++){
      m_Data += currEdges[i]->GetSweptPhysData(edgePhysDataIndex)/nb;
    }
  }
  //*/
}


void 
DynFlds_EdgeData_Dgn::
Advance()
{
  ComputeData();
  DynObj::Advance();
}
