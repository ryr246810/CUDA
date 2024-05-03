#include <DynFlds_VertexData_Dgn.hxx>



DynFlds_VertexData_Dgn::
DynFlds_VertexData_Dgn()
  :FieldsDgnBase()
{
  m_Vertex = NULL;
  m_Data = 0.0;
}



void
DynFlds_VertexData_Dgn::
Init(const FieldsDefineCntr* theCntr)
{
  FieldsDgnBase::Init(theCntr);
  m_Data = 0.0;
}



DynFlds_VertexData_Dgn::~DynFlds_VertexData_Dgn()
{

}



void 
DynFlds_VertexData_Dgn::
SetAttrib(const TxHierAttribSet& tha)
{
  string theName="";

  Standard_Size PhiNumber = GetGridGeom_Cyl3D()->GetDimPhi();
  if(PhiNumber == 1)
  {
  	m_PhiIndex = -1;
  }
  else if(tha.hasOption("Phi")){
    m_PhiIndex= tha.getOption("Phi");
  }else{
    cout<<"DynFlds_VertexData_Dgn::SetAttrib--------------error----Phi"<<endl;
  }

  Standard_Size vertexIndxVec[2];
  if(tha.hasPrmVec("location")){
    Standard_Real theLocation[2];
    vector<Standard_Real> theData = tha.getPrmVec("location");
    if(theData.size()>=2){
      theLocation[0] = theData[0];
      theLocation[1] = theData[1];
    }else{
      cout<<"DynFlds_VertexData_Dgn::SetAttrib--------------error----location"<<endl;
    }
    GetFldsDefCntr()->GetZRGrid()->ComputeLocationInGrid(theLocation, vertexIndxVec);
  }else if(tha.hasOptVec("locationIndex")){
    vector<int> theindex = tha.getOptVec("locationIndex");
    if(theindex.size()>=2){
      vertexIndxVec[0] = theindex[0];
      vertexIndxVec[1] = theindex[1];
    }else{
      cout<<"DynFlds_VertexData_Dgn::SetAttrib--------------error----locationIndex-----1"<<endl;
    }
  }else{
    cout<<"DynFlds_VertexData_Dgn::SetAttrib--------------error----locationIndex-----2"<<endl;
  }


  if(tha.hasString("name")){
    theName = tha.getString("name");
  }else{
    cout<<"DynFlds_VertexData_Dgn::SetAttrib--------------error----name"<<endl;
  }

  SetName(theName);  // dynobj
  InitParamt(vertexIndxVec);  // poynting
}



void 
DynFlds_VertexData_Dgn::
InitParamt(const Standard_Size vertexIndxVec[2])
{

  TxSlab2D<Standard_Integer> theRgn  = GetFldsDefCntr()->GetZRGrid()->GetPhysRgn();

  if( ( (vertexIndxVec[0]>=theRgn.getLowerBound(0)) && (vertexIndxVec[0]<=theRgn.getUpperBound(0)) ) &&
      ( (vertexIndxVec[1]>=theRgn.getLowerBound(1)) && (vertexIndxVec[1]<=theRgn.getUpperBound(1)) ) ) { 
    Standard_Size currVertexIndex;
    GetFldsDefCntr()->GetZRGrid()->FillVertexIndx(vertexIndxVec, currVertexIndex);
    m_Vertex= GetGridGeom(m_PhiIndex)->GetGridVertices()+ currVertexIndex;
  }else{
    m_Vertex = NULL;
  }
}



Standard_Real 
DynFlds_VertexData_Dgn::
GetValue()
{
  return m_Data;
}



void 
DynFlds_VertexData_Dgn::
ComputeData()
{
  //*
  m_Data = 0.0;

  Standard_Integer vertexPhysDataIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicElecField_PhysDataIndex();

  if(m_Vertex!=NULL){
      m_Data = m_Vertex->GetSweptPhysData(vertexPhysDataIndex);
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
DynFlds_VertexData_Dgn::
Advance()
{
  ComputeData();
  DynObj::Advance();
}
