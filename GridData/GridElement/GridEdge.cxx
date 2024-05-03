#include <GridEdge.hxx>

#include <GridEdgeData.hxx>

#include <GridVertexData.hxx>
#include <AppendingVertexDataOfGridEdge.hxx>

#include <GridGeometry.hxx>

#include <PhysConsts.hxx>

/***************************************************************************/
/**
 * Function:  BuildEdge
 * Purpose:   With proper order Edges by appending vertices;
 */
/***************************************************************************/
GridEdge::GridEdge()
{
  m_Dir = DIRRZZ;
  m_Index = 0;
  m_GridGeom = NULL;
}


GridEdge::~GridEdge()
{
  ClearEdges();
  ClearAppendingVertices();
}


void GridEdge::ClearEdges()
{
  if(!m_Edges.empty()){
    Standard_Integer nb = m_Edges.size();
    for (Standard_Integer idx = nb; 0<idx; delete m_Edges[--idx]);
    m_Edges.clear();
  }
}


void GridEdge::ClearAppendingVertices()
{
  if(!m_Vertices.empty()){
    Standard_Integer nb = m_Vertices.size();
    for (Standard_Integer idx = nb; 0<idx; delete m_Vertices[--idx]);
    m_Vertices.clear();
  }
}


GridEdge::GridEdge(GridGeometry* _gridgeom,
		   ZRGridLineDir _dir, 
		   Standard_Size _index)
{
  m_GridGeom = _gridgeom;
  m_Dir = _dir;
  m_Index = _index;
}


bool
GridEdge::      
HasAppending() const
{
  bool result = false;

  if(m_Vertices.size()>0){
    result = true;
  }

  return result;
};


void GridEdge::GetTwoEndGridVertices(GridVertexData*& Vertex1, 
				     GridVertexData*& Vertex2) const
{
  Standard_Integer Dir0 = Standard_Integer(m_Dir);
  Standard_Size theFirstIndex, theSecondIndex;
  Standard_Size indxVec[2];

  GetZRGrid()->FillEdgeIndxVec(Dir0, m_Index, indxVec);
  GetZRGrid()->FillVertexIndx(indxVec, theFirstIndex);
  theSecondIndex = GetZRGrid()->bumpVertex(Dir0,theFirstIndex);

  Vertex1  = GetGridGeom()->GetGridVertices() + theFirstIndex;
  Vertex2  = GetGridGeom()->GetGridVertices() + theSecondIndex;
}

 
GridVertexData* GridEdge::GetFirstVertex() const
{
  GridVertexData* Vertex1;

  Standard_Integer Dir0 = Standard_Integer(m_Dir);
  Standard_Size theFirstIndex;
  Standard_Size indxVec[2];

  GetZRGrid()->FillEdgeIndxVec(Dir0, m_Index, indxVec);
  GetZRGrid()->FillVertexIndx(indxVec, theFirstIndex);

  Vertex1  = GetGridGeom()->GetGridVertices() + theFirstIndex;
  return Vertex1;
}


GridVertexData* GridEdge::GetLastVertex() const
{
  GridVertexData* Vertex2;

  Standard_Integer Dir0 = Standard_Integer(m_Dir);
  Standard_Size theFirstIndex, theSecondIndex;
  Standard_Size indxVec[2];

  GetZRGrid()->FillEdgeIndxVec(Dir0, m_Index, indxVec);
  GetZRGrid()->FillVertexIndx(indxVec, theFirstIndex);
  theSecondIndex = GetZRGrid()->bumpVertex(Dir0,theFirstIndex);

  Vertex2  = GetGridGeom()->GetGridVertices() + theSecondIndex;
  return Vertex2;
}


Standard_Integer GridEdge::GetIndexOfDir(Standard_Integer Dir0) const
{
  Standard_Size indxVec[2];
  GetZRGrid()->FillEdgeIndxVec(m_Dir, m_Index, indxVec);
  Standard_Integer result = indxVec[Dir0];
  return result;
}


void GridEdge::GetVecIndex(Standard_Size indxVec[2]) const
{
  GetZRGrid()->FillEdgeIndxVec(m_Dir, m_Index, indxVec);
}


const ZRGrid* GridEdge::GetZRGrid() const
{
  return m_GridGeom->GetZRGrid(); 
}



Standard_Real GridEdge::GetLength() const
{ 
  Standard_Size indxVec[2];
  GetZRGrid()->FillEdgeIndxVec(m_Dir, m_Index, indxVec);
  Standard_Real result = GetZRGrid()->GetStep(m_Dir,  indxVec[m_Dir]);
  return result;
}


//*
Standard_Real GridEdge::GetDualArea() const
{ 
  Standard_Size indxVec[2];
  GetZRGrid()->FillEdgeIndxVec(m_Dir, m_Index, indxVec);

  Standard_Real result = 0;

  if(m_Dir==0){  // zdir
    Standard_Real R0 = GetZRGrid()->GetCoordComp_From_VertexVectorIndx(1, indxVec);
    Standard_Real dR = GetZRGrid()->GetDualStep(1,  indxVec[1]);
    result = 2.0*mksConsts.pi*R0*dR;
  }else if(m_Dir==1){ // rdir
    Standard_Real R0 = GetZRGrid()->GetCoordComp_From_VertexVectorIndx(1, indxVec) + 0.5*GetZRGrid()->GetStep(1, indxVec[1]);
    Standard_Real dZ = GetZRGrid()->GetDualStep(0, indxVec[0]);
    result = 2.0*mksConsts.pi*R0*dZ;
  }else{
    cout<<"error-------------------------GridEdge::GetDualArea()----------1"<<endl;
  }

  return result;
}
//*/


Standard_Integer GridEdge::GetResolution() const
{ 
  return GetZRGrid()->GetResolutionRatio(); 
}


TxVector2D<Standard_Real> GridEdge::GetVector() const
{
  TxVector2D<Standard_Real> tmp=GetLastVertex()->GetLocation()-GetFirstVertex()->GetLocation();
  return tmp;; 
};



GridEdgeData* GridEdge::GetEdgeData(const Standard_Integer _localIndex) const
{
  return m_Edges[_localIndex];
}



vector<GridEdgeData*> GridEdge::GetEdgesOfState(Standard_Integer _state) const
{
  vector<GridEdgeData*> result;
  const vector<GridEdgeData*>& theAllEdges = GetEdges();
  Standard_Size nb = theAllEdges.size();

  for(Standard_Size i=0; i<nb; i++){
    if( ((theAllEdges[i]->GetState()) & _state) !=0){
      result.push_back(theAllEdges[i]);
    }
  }
  return result;
}


vector<GridEdgeData*> GridEdge::GetEdgesOfMaterial(Standard_Integer _material) const
{
  vector<GridEdgeData*> result;
  const vector<GridEdgeData*>& theAllEdges = GetEdges();
  Standard_Size nb = theAllEdges.size();
  for(Standard_Size i=0; i<nb; i++){
    if(   ( (theAllEdges[i]->GetMaterialType()) & _material ) !=0   ){
      result.push_back(theAllEdges[i]);
    }
  }
  return result;
}


void GridEdge::AddAppendingVertex(AppendingVertexDataOfGridEdge* aIrregVertex)
{
  if(m_Vertices.empty()){
    m_Vertices.push_back(aIrregVertex);
    return;
  }

  GridVertexData* theFirstVertex = this->GetFirstVertex();
  GridVertexData* theLastVertex = this->GetLastVertex();

  Standard_Size afrac = aIrregVertex->GetFrac();
  bool hasSameLocationAppendedV = false;

  // 1. for the situation which the vertex preparing to be inserted 
  //       has same location with one of the existed appendingvertices
  vector<AppendingVertexDataOfGridEdge*>::iterator iter;
  for(iter=m_Vertices.begin(); iter!=m_Vertices.end(); iter++){

    AppendingVertexDataOfGridEdge* currIrregVertex = *iter;

    if( currIrregVertex->HasSameLocation(aIrregVertex) ){
      hasSameLocationAppendedV = true;

      if(currIrregVertex->HasCommonShapeIndexWith(aIrregVertex)){
	if( (currIrregVertex->GetTransitionType() + aIrregVertex->GetTransitionType()) == 0 ){
	  m_Vertices.erase(iter);
	  return;
	}
      }else{
	m_GridGeom->RemoveSpaceInfoOfAppendingVertexDataOfGridEdge(this, aIrregVertex);  // modified 2015.12.24
	m_GridGeom->RemoveSpaceInfoOfAppendingVertexDataOfGridEdge(this, currIrregVertex);  // modified 2015.12.24
	
	if( currIrregVertex->GetTransitionType()==1 ){
	  m_Vertices.insert( iter, aIrregVertex);
	  return;
	}else{
	  if( (iter+1)!= m_Vertices.end())  m_Vertices.insert( iter+1, aIrregVertex);
	  else	                          m_Vertices.push_back(aIrregVertex);
	  return;
	}
      }
    }
  }

  // 2. for the situation which the vertex preparing to be inserted 
  //        does not hace same location with anye of the existed appendingvertices
  vector<AppendingVertexDataOfGridEdge*>::iterator next;
  if(!hasSameLocationAppendedV){
    for(iter=m_Vertices.begin(); iter!=m_Vertices.end(); iter++){
      Standard_Size frac1 = (*iter)->GetFrac();
      next = iter+1;
      if( next!= m_Vertices.end() ){
	Standard_Size frac2 = (*next)->GetFrac();
	if(frac1<afrac<frac2){
	  m_Vertices.insert( next, aIrregVertex);
	  return;
	}
      }else{
	if(frac1<afrac)	  m_Vertices.push_back(aIrregVertex);
	else    	  m_Vertices.insert( iter, aIrregVertex);
	return;
      }
    }
  }
}

