#include <GridEdge.hxx>

#include <GridEdgeData.hxx>

#include <GridVertexData.hxx>
#include <AppendingVertexDataOfGridEdge.hxx>

#include <GridGeometry.hxx>


void GridEdge::BuildEdges()
{
  //0.0 clear all edges
  ClearEdges();

  //1.0 setup one vertex sequence
  vector<VertexData*> allVertices;
  allVertices.clear();

  VertexData* firstVertex = GetFirstVertex();
  VertexData* lastVertex  = GetLastVertex();

  allVertices.push_back(firstVertex);
  for(Standard_Size i=0; i <m_Vertices.size(); i++)  allVertices.push_back(m_Vertices[i]);
  allVertices.push_back(lastVertex);


  //2.0 build PF edges

  vector<VertexData*>::const_iterator breakIter = allVertices.begin();
  BuildGridEdgeDatas(breakIter, allVertices);
  allVertices.clear();

  SetupLocalIndexOfGridEdgeData();
}



void
GridEdge::
BuildGridEdgeDatas(vector<VertexData*>::const_iterator& breakIter, 
		   const vector<VertexData*>& allVertices)
{
  vector<VertexData*> oneEdgeVertices;
  oneEdgeVertices.clear();

  vector<VertexData*>::const_iterator firstIter = breakIter;

  VertexData* firstVertex = *firstIter;
  oneEdgeVertices.push_back(firstVertex);

  for(breakIter = firstIter+1; breakIter!=allVertices.end(); breakIter++){
    VertexData* currVertex = *breakIter;
    oneEdgeVertices.push_back(currVertex);
    if(currVertex->IsMaterialType(PEC)){
      break;
    }
  }

  BuildOneGridEdgeData(oneEdgeVertices);
  oneEdgeVertices.clear();

  // find another break vertex whick is a PEC vertex, and continue to build GridEdgeData
  if( breakIter!=allVertices.end() ){
    vector<VertexData*>::const_iterator nextIter = breakIter + 1;
    if(nextIter!=allVertices.end()){
      BuildGridEdgeDatas(breakIter, allVertices);
    }
  }
}




void
GridEdge::
BuildOneGridEdgeData(const vector<VertexData*>& oneEdgeVertices)
{
  if(oneEdgeVertices.size()<2){
    return;
  }

  if(oneEdgeVertices.size()==2){
    if( (oneEdgeVertices[0]->IsMaterialType(PEC)) && 
	(oneEdgeVertices[1]->IsMaterialType(PEC)) ){
      return;  // do not build a PEC Edge
    }

    // add 2014.03.31
    if( (oneEdgeVertices[0]->IsMaterialType(PEC)) &&  
	(oneEdgeVertices[0]->GetState()==OUTSHAPE)){
      return;
    }

    if( (oneEdgeVertices[1]->IsMaterialType(PEC)) &&  
	(oneEdgeVertices[1]->GetState()==OUTSHAPE)){
      return;
    }
 
    // add 2014.04.09
    if( (oneEdgeVertices[0]->GetState()==BND) && 
	(oneEdgeVertices[1]->GetState()==BND) ){
      return;  // do not build a BND Edge
    }
  }

  if(oneEdgeVertices.size()>2){
    Standard_Size num = oneEdgeVertices.size();
    if( (oneEdgeVertices[0]->IsMaterialType(PEC)) && 
	(oneEdgeVertices[num-1]->IsMaterialType(PEC)) ){
      return;  // do not build a PEC Edge
    }
  }

  // 2014.03.19
  Standard_Size nb = oneEdgeVertices.size();
  TxVector2D<Standard_Real> theVector = oneEdgeVertices[nb-1]->GetLocation() - oneEdgeVertices[0]->GetLocation();
  Standard_Real theLength = theVector.length();
  Standard_Real theMinLength = 0.5*GetLength()/((Standard_Real)GetResolution());
  if(theLength<theMinLength){
    return; // do not build if the pfedge's length is less than 0.5*dl/n 2014.03.12
  }

  GridEdgeData* oneEdge = new GridEdgeData(this);
  oneEdge->SetVertexVec(oneEdgeVertices);
  oneEdge->Setup();

  m_Edges.push_back(oneEdge);
}




/****************************************************************/
/*
 * Function : SetupLocalIndexOfGridEdgeData()
 * Purpose  : for constructing GridEdgeData's local index in this GridEdge
 */
/****************************************************************/
void GridEdge::SetupLocalIndexOfGridEdgeData()
{
  vector<GridEdgeData*>::iterator iter;

  Standard_Integer CurrIndex = 0;

  for(iter=m_Edges.begin(); iter!=m_Edges.end(); iter++){
    (*iter)->SetLocalIndex(CurrIndex);
    CurrIndex++;
  }
}
