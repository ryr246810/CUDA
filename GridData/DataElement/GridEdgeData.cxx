
#include <GridEdge.hxx>
#include <GridFaceData.cuh>

#include <GridEdgeData.hxx>
#include <GridVertexData.hxx>
#include <AppendingVertexDataOfGridEdge.hxx>

#include <GridGeometry.hxx>

#include <PhysConsts.hxx>





GridEdgeData::GridEdgeData()
  :EdgeData()
{
  m_BaseGEdge = NULL;
  m_DualArea = 0.0;
  m_C[0] = 0.0;
  m_C[1] = 0.0;
  m_C[2] = 0.0;
}


GridEdgeData::GridEdgeData(GridEdge* _gridedge)
  :EdgeData()
{
  SetBaseGridEdge(_gridedge);
}


GridEdgeData::GridEdgeData(Standard_Integer _mark)
  :EdgeData(_mark)
{
  m_BaseGEdge = NULL;
  m_DualArea = 0.0;
  m_C[0] = 0.0;
  m_C[1] = 0.0;
  m_C[2] = 0.0;
}


GridEdgeData::GridEdgeData(GridEdge *_gridedge, Standard_Integer _mark)
  :EdgeData(_mark)
{
  SetBaseGridEdge(_gridedge);
}


GridEdgeData::~GridEdgeData()
{
  m_DTEdges.clear();
  m_TFaces.clear(); 

  m_NearEEdges.clear(); 
  m_NearMEdges.clear(); 

  ResetShapeIndices();
  m_MidVertices.clear();
  m_BaseGEdge = NULL;
}


void GridEdgeData::SetBaseGridEdge(GridEdge* _gridedge)
{
  m_BaseGEdge = _gridedge; 

  Standard_Real n_Segment = m_BaseGEdge->GetGridGeom()->GetPhiNumber();

  Standard_Size indxVec[2];
  m_BaseGEdge->GetVecIndex(indxVec);

  const ZRGrid* theGrid = m_BaseGEdge->GetZRGrid();
  Standard_Integer theDir = m_BaseGEdge->GetDir();


  Standard_Real result = 0;
  if(theDir==0){
    Standard_Real R0 = fabs(theGrid->GetCoordComp_From_VertexVectorIndx(1, indxVec));
    Standard_Real dR = theGrid->GetDualStep(1,  indxVec[1]);

    if(indxVec[1]==1){
      result = mksConsts.pi*dR*dR*0.25/n_Segment;
    }else{
      result = 2.0*mksConsts.pi*R0*dR/n_Segment;
    }
    /*
    cout<<"dir = "<<theDir;
    cout<<"  indxVex = ["<<indxVec[0]<<", "<<indxVec[1]<<"]";
    cout<<"  R0="<<R0<< "; "<<"dR="<<dR;
    cout<<";  dualArea="  << result <<endl;
    //*/
  }else if(theDir==1){
    Standard_Real R0 = fabs(theGrid->GetCoordComp_From_VertexVectorIndx(1, indxVec) + 0.5*theGrid->GetStep(1, indxVec[1]));
    Standard_Real dZ = theGrid->GetDualStep(0, indxVec[0]);
    result = 2.0*mksConsts.pi*R0*dZ/n_Segment;
    /*
    cout<<"dir = "<<theDir;
    cout<<"  indxVex = ["<<indxVec[0]<<", "<<indxVec[1]<<"]";
    cout<<"  R0="<<theGrid->GetCoordComp_From_VertexVectorIndx(1, indxVec) << "+"<< 0.5*theGrid->GetStep(1, indxVec[1])<<"="<<R0;
    cout<<"; "<<"dZ="<<dZ<< ";   dualArea =" << result <<endl;
    //*/
  }else{
    cout<<"error-------------------------GridEdge::GetDualArea()----------1"<<endl;
  }

  m_DualArea = result;
  TxVector2D<Standard_Real> firstPnt = m_BaseGEdge->GetFirstVertex()->GetLocation();
  TxVector2D<Standard_Real> lastPnt = m_BaseGEdge->GetLastVertex()->GetLocation();
  TxVector2D<Standard_Real> theMidPnt = (firstPnt + lastPnt) / 2;
  m_BaseGridSweptArea = 2.0*mksConsts.pi*fabs(theMidPnt[1])*m_BaseGEdge->GetLength()/n_Segment;
}

Standard_Real 
GridEdgeData::
GetSweptGeomDim() const {
    Standard_Real n_Segment = m_BaseGEdge->GetGridGeom()->GetPhiNumber();
    Standard_Real result = EdgeData::GetSweptGeomDim()/n_Segment;
    return result; 
  }

Standard_Real 
GridEdgeData::
GetSweptGeomDim_Near()
{
  Standard_Integer dir = GetDir();
  Standard_Size indxVec[2];
  Standard_Size indx = GetBaseGridEdge()->GetIndex();
  m_BaseGEdge->GetZRGrid()->FillEdgeIndxVec(dir, indx, indxVec);

  Standard_Real result = m_BaseGEdge->GetZRGrid()->GetStep(dir, indxVec[dir]);
  return result;

}
Standard_Real 
GridEdgeData::
GetDualGeomDim_Near()
{
  Standard_Integer dir = GetDir();
  Standard_Size indxVec[2];
  Standard_Size indx = GetBaseGridEdge()->GetIndex();
  m_BaseGEdge->GetZRGrid()->FillEdgeIndxVec(dir, indx, indxVec);

  Standard_Integer dualDir = (dir+1)%2;
  Standard_Real result = m_BaseGEdge->GetZRGrid()->GetDualStep(dualDir, indxVec[dualDir]);
  return result;

}
void
GridEdgeData::
SetVertexVec(const vector<VertexData*>& oneEdgeVertices)
{
  Standard_Size nb = oneEdgeVertices.size();

  VertexData* firstVertex = oneEdgeVertices[0];
  VertexData* lastVertex = oneEdgeVertices[nb-1];

  SetVertices(firstVertex, lastVertex);

  m_MidVertices.clear();
  for(Standard_Size i=1; i<nb-1; i++){
    m_MidVertices.push_back(oneEdgeVertices[i]);
  }
}



void 
GridEdgeData::
Setup()
{
  SetupGeomDimInf();
  SetupMaterialData();

  /*
  {
    TxVector2D<Standard_Real> theMidPnt;
    ComputeMidPntLocation(theMidPnt);
    Standard_Size indxVec[2];
    this->GetBaseGridEdge()->GetVecIndex(indxVec);
    cout<<"dir = "<<this->GetDir();
    cout<<"  indxVec = [" <<indxVec[0]<<", "<<indxVec[1]<<"]";
    cout<<"  edgeLengthRatio = "<<this->EdgeLengthRatio();
    cout<<"  areaOfSweptFace = "<<m_AreaOfSweptFace/2.0/mksConsts.pi/this->GetLength();
    cout<<"  r0 = "<<theMidPnt[1];
    cout<<"  dualLengthOfSweptFace = "<<m_DualLengthOfSweptFace<<endl;
    cout<<endl;
  }
  //*/
}


void
GridEdgeData::
SetupGeomDimInf()
{
  ComputeLength();
  ComputeAreaOfSweptFace();

  InitEfficientLength();
  ComputeDualLengthOfSweptFace();

  DeduceGeomState();
  DeduceGeomType();
}


void
GridEdgeData::
SetupMaterialData()
{
  DataBase::SetupMaterialData();  // new material array being used to define espilon mu and sigma

  DeduceShapeIndices();
  DeduceMaterialType();

  DeduceMaterialData();
}


void GridEdgeData::AddFace(GridFaceData* aFace, Standard_Integer aDir)
{
  T_Element aTopoFace(aFace, aDir);
  m_TFaces.push_back( aTopoFace );
}


void GridEdgeData::AddDEdge(GridVertexData* aDEdge, Standard_Integer aDir)
{
  T_Element aTopoTDEdge(aDEdge, aDir);
  m_DTEdges.push_back(aTopoTDEdge);
}

void GridEdgeData::AddNearEEdge(EdgeData* aDEdge, Standard_Integer aDir)
{
  T_Element aTopoTDEdge(aDEdge, aDir);
  m_NearEEdges.push_back(aTopoTDEdge);
}

void GridEdgeData::AddNearMEdge(EdgeData* aDEdge, Standard_Integer aDir)
{
  T_Element aTopoTDEdge(aDEdge, aDir);
  m_NearMEdges.push_back(aTopoTDEdge);
}

bool GridEdgeData::IsSharedFacesPhysDataDefined()
{
  bool result=true;
  Standard_Size nb = m_TFaces.size();
  for(Standard_Size index = 0; index<nb; index++){
    bool tmp = (m_TFaces[index].GetData())->IsPhysDataDefined();
    result = result && tmp;
  }
  return result;
}


bool GridEdgeData::IsOutLineDEdgePhysDataDefined()
{
  bool result=true;
  Standard_Size nb = m_DTEdges.size();
  for(Standard_Size index = 0; index<nb; index++){
    bool tmp = (m_DTEdges[index].GetData())->IsSweptPhysDataDefined();
    result = result && tmp;
  }
  return result;
}



GridEdge* GridEdgeData::GetBaseGridEdge()
{
  return m_BaseGEdge; 
}


Standard_Integer GridEdgeData::GetDir()
{
 return m_BaseGEdge->GetDir();
}


bool GridEdgeData::HasMidBndVertex(const VertexData* _bndVertex) const
{
  bool result = false;
  for(Standard_Size i=0; i<m_MidVertices.size(); i++){
    if(m_MidVertices[i]==_bndVertex){
      result = true;
      break;
    }
  }
  return result;
}


