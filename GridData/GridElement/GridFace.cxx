#include <GridFace.hxx>
#include <AppendingVertexDataOfGridFace.hxx>

#include <GridFaceData.cuh>
#include <GridGeometry.hxx>
#include <set>

#include <PhysConsts.hxx>


#define GRIDFACE_DEBUG


GridFace::GridFace()
{
  m_Index = 0;
  m_GridGeom = NULL;
}


GridFace::~GridFace()
{
  ClearFaces();
  ClearAppendingVertices();
}


GridFace::GridFace(GridGeometry* _gridgeom,
		   const Standard_Size _index)
{
  m_GridGeom = _gridgeom;
  m_Index = _index;
}



bool 
GridFace::
IsCut() const
{
  bool result = false;

  GridEdge* Edge11; 
  GridEdge* Edge12;
  GridEdge* Edge21;
  GridEdge* Edge22;
  GetOutLineGridEdges(Edge11,Edge12,Edge21, Edge22);

  GridEdge* theEdges[4];
  theEdges[0] = Edge11;
  theEdges[1] = Edge12;
  theEdges[2] = Edge21;
  theEdges[3] = Edge22;

  vector<AppendingVertexDataOfGridEdge*>  theAllEdgeBndVertices;

  vector<GridVertexData*> theBndGridVertices;

  theAllEdgeBndVertices.clear();
  for(Standard_Integer j=0; j<4; j++){
    const vector<AppendingVertexDataOfGridEdge*>&  theEdgeBndVertices = theEdges[j]->GetAppendingVertices();
    for(Standard_Size k=0; k<theEdgeBndVertices.size(); k++){
      theAllEdgeBndVertices.push_back(theEdgeBndVertices[k]);
    }

    GridVertexData* theFV = theEdges[j]->GetFirstVertex();
    GridVertexData* theLV = theEdges[j]->GetLastVertex();

    if(theFV->GetState()==BND){
      theBndGridVertices.push_back(theFV);
    }
    if(theLV->GetState()==BND){
      theBndGridVertices.push_back(theLV);
    }
  }

  if( (!theAllEdgeBndVertices.empty()) || (!theBndGridVertices.empty()) ){
    result = true;
  }

  return result;
}


void GridFace::GetOutLineGridEdges(GridEdge*& Edge11, 
				   GridEdge*& Edge12,
				   GridEdge*& Edge21,
				   GridEdge*& Edge22) const
{
  Standard_Integer Dir1 = 0;
  Standard_Integer Dir2 = 1;

  Standard_Size BumpDir11=0, BumpDir12=0, BumpDir21=0, BumpDir22=0;

  Standard_Size indxVec[2];

  GetZRGrid()->FillFaceIndxVec(m_Index, indxVec);

  GetZRGrid()->FillEdgeIndx(Dir1, indxVec, BumpDir11);
  GetZRGrid()->FillEdgeIndx(Dir2, indxVec, BumpDir21);

  BumpDir12 = GetZRGrid()->bumpEdge(Dir1,Dir2,BumpDir11);
  BumpDir22 = GetZRGrid()->bumpEdge(Dir2,Dir1,BumpDir21);

  Edge11    =  (GetGridGeom()->GetGridEdges())[Dir1] + BumpDir11;
  Edge12    =  (GetGridGeom()->GetGridEdges())[Dir1] + BumpDir12;

  Edge21    =  (GetGridGeom()->GetGridEdges())[Dir2] + BumpDir21;
  Edge22    =  (GetGridGeom()->GetGridEdges())[Dir2] + BumpDir22;
}


void GridFace::ClearFaces()
{
  for (Standard_Size idx = m_Faces.size(); idx>0; delete m_Faces[--idx]);
  m_Faces.clear();
}


void GridFace::ClearAppendingVertices()
{
  for (Standard_Size idx = m_BndVertices.size(); idx>0; delete m_BndVertices[--idx]);
  m_BndVertices.clear();
}


void GridFace::AddFaceData(GridFaceData* _face) 
{ 
  m_Faces.push_back(_face); 
}


void GridFace::AddBndVertexOfGridFace(AppendingVertexDataOfGridFace* _vertex)
{
  m_BndVertices.push_back(_vertex);
}




bool GridFace::HasAppendingVertexOfGridFace()
{
  bool result = false;

  if(m_BndVertices.size()>0){
    result = true;
  }
#ifdef GRIDFACE_DEBUG
  if(m_BndVertices.size()>=2) 
    cout<<"GridFace::HasAppendingVertexOfGridFace()-------------------OneFace have more than 2 appending vertices !"<<endl;
#endif

  return result;
}



/*******************************************************************************************/
/*******************************************************************************************/
/*******************************************************************************************/
/*******************************************************************************************/
/*******************************************************************************************/
/*******************************************************************************************/

Standard_Integer GridFace::GetIndexOfDir(Standard_Integer dir) const
{
  Standard_Size indxVec[2];
  GetZRGrid()->FillFaceIndxVec(m_Index, indxVec);
  Standard_Integer result = indxVec[dir];
  return result;
}


void GridFace::GetVecIndex(Standard_Size indxVec[2]) const
{
  GetZRGrid()->FillFaceIndxVec(m_Index, indxVec);
}


const ZRGrid* GridFace::GetZRGrid() const
{
  return m_GridGeom->GetZRGrid(); 
}


Standard_Real GridFace::GetArea()  const       
{ 
  Standard_Size indxVec[2];
  GetZRGrid()->FillFaceIndxVec(m_Index, indxVec);
  Standard_Real result = GetZRGrid()->GetStep(0, indxVec[0])*GetZRGrid()->GetStep(1, indxVec[1]);

  return result;
};




Standard_Real GridFace::GetDualLength()  const       
{ 
  Standard_Size indxVec[2];
  
  Standard_Real n_Segment = GetGridGeom()->GetPhiNumber();

  GetZRGrid()->FillFaceIndxVec(m_Index, indxVec);
  Standard_Real R0 = GetZRGrid()->GetCoordComp_From_VertexVectorIndx(1, indxVec) + 0.5*GetZRGrid()->GetStep(1, indxVec[1]);
  Standard_Real result = 2.0*mksConsts.pi*fabs(R0)/ n_Segment;

  return result;
};




GridFaceData* GridFace::GetGridFaceDataContaining(GridEdgeData* _edgedata) const
{
  GridFaceData* result = NULL;
  Standard_Size nb = m_Faces.size();
  for(Standard_Size index=0;index<nb;index++){
    if(m_Faces[index]->IsContaining(_edgedata)){
      result = m_Faces[index];
      break;
    }
  }
  return result;
}


GridFaceData* GridFace::GetGridFaceDataContaining(VertexData* _vertexdata) const
{
  GridFaceData* result = NULL;
  Standard_Size nb = m_Faces.size();
  for(Standard_Size index=0;index<nb;index++){
    if(m_Faces[index]->IsContaining(_vertexdata)){
      result = m_Faces[index];
      break;
    }
  }
  return result;
}


// modified 2014.04.01
GridFaceData* GridFace::GetGridFaceDataContaining(Standard_Integer _state, GridEdgeData* _edgedata) const
{
  GridFaceData* result = NULL;
  Standard_Size nb = m_Faces.size();
  for(Standard_Size index=0;index<nb;index++){
    if(m_Faces[index]->IsContaining(_edgedata)){
      if( ((m_Faces[index]->GetState()) & _state) ==0){
	cout<<"GridFace::GetGridFaceDataContaining-----------------------error information---------Edge"<<endl;
	continue;
      }else{
	result = m_Faces[index];
	break;
      }
    }
  }
  return result;
}


// modified 2014.04.01
GridFaceData* GridFace::GetGridFaceDataContaining(Standard_Integer _state, VertexData* _vertexdata) const
{
  GridFaceData* result = NULL;
  Standard_Size nb = m_Faces.size();
  for(Standard_Size index=0;index<nb;index++){
    if(m_Faces[index]->IsContaining(_vertexdata)){
      if( ((m_Faces[index]->GetState()) & _state)==0){
	cout<<"GridFace::GetGridFaceDataContaining-----------------------error information---------Vertex"<<endl;
	continue;
      }else{
	result = m_Faces[index];
	break;
      }
    }
  }
  return result;
}


vector<GridFaceData*> GridFace::GetFacesOfState(Standard_Integer _state) const
{
  vector<GridFaceData*> result;
  const vector<GridFaceData*>& theAllFaces = GetFaces();
  Standard_Size nb = theAllFaces.size();
  for(Standard_Size i=0; i<nb; i++){
    if( ((theAllFaces[i]->GetState()) & _state) !=0){
      result.push_back(theAllFaces[i]);
    }
  }
  return result;
}


vector<GridFaceData*> GridFace::GetFacesOfMaterial(Standard_Integer _material) const
{
  vector<GridFaceData*> result;
  const vector<GridFaceData*>& theAllFaces = GetFaces();
  Standard_Size nb = theAllFaces.size();
  for(Standard_Size i=0; i<nb; i++){
    if(   ( (theAllFaces[i]->GetMaterialType()) & _material ) !=0   ){
      result.push_back(theAllFaces[i]);
    }
  }
  return result;
}


/****************************************************************************/

Standard_Integer GridFace::GetResolution() const
{ 
  return GetZRGrid()->GetResolutionRatio(); 
}


TxVector2D<Standard_Real> GridFace::GetVectorOfDir1() const
{
  GridEdge* Edge11;
  GridEdge* Edge12;
  GridEdge* Edge21;
  GridEdge* Edge22;
  GetOutLineGridEdges(Edge11, Edge12, Edge21, Edge22);
  TxVector2D<Standard_Real> tmp =  Edge11->GetLastVertex()->GetLocation() -  Edge11->GetFirstVertex()->GetLocation();
  return tmp;
}


TxVector2D<Standard_Real> GridFace::GetVectorOfDir2() const
{
  GridEdge* Edge11;
  GridEdge* Edge12;
  GridEdge* Edge21;
  GridEdge* Edge22;

  GetOutLineGridEdges(Edge11, Edge12, Edge21, Edge22);
  TxVector2D<Standard_Real> tmp =  Edge21->GetLastVertex()->GetLocation() -  Edge21->GetFirstVertex()->GetLocation();
  return tmp;
}


const GridVertexData* GridFace::GetLDVertex()const
{
  GridEdge* Edge11;
  GridEdge* Edge12;
  GridEdge* Edge21;
  GridEdge* Edge22;

  GetOutLineGridEdges(Edge11, Edge12, Edge21, Edge22);

  return Edge11->GetFirstVertex();
}


/****************************************************************************/


#ifdef GRIDFACE_DEBUG
#undef GRIDFACE_DEBUG
#endif
