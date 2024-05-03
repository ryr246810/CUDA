#include <GridFace.hxx>
#include <AppendingVertexDataOfGridFace.hxx>

#include <GridFaceData.cuh>
#include <GridGeometry.hxx>
#include <set>


#include <BaseFunctionDefine.hxx>


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void GridFace::BuildFaces()
{
  vector<GridEdgeData* > theEdges;
  vector<Standard_Integer> theEdgeTDirs;
  GetAllGridEdgeDatasOfGridFace(theEdges, theEdgeTDirs);

  if(theEdges.empty()) return; 

  ConstructGridFaceDatasFromEdges(theEdges, theEdgeTDirs);  // topogical relation between edge and face
  SetupLocalIndexOfGridFaceData();
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void 
GridFace::
GetAllGridEdgeDatasOfGridFace( vector<GridEdgeData*>& theEdges, 
			       vector<Standard_Integer>& theEdgeTDirs )
{
  theEdges.clear();
  theEdgeTDirs.clear();

  GridEdge* Edge11;
  GridEdge* Edge12;
  GridEdge* Edge21;
  GridEdge* Edge22;
  GetOutLineGridEdges(Edge11, Edge12, Edge21, Edge22);

  const vector<GridEdgeData* >& EdgeDatas11  = Edge11->GetEdges();
  const vector<GridEdgeData* >& EdgeDatas12  = Edge12->GetEdges();
  const vector<GridEdgeData* >& EdgeDatas21  = Edge21->GetEdges();
  const vector<GridEdgeData* >& EdgeDatas22  = Edge22->GetEdges();

  vector<GridEdgeData*>::const_reverse_iterator rIter;
  vector<GridEdgeData*>::const_iterator iter;

  if(!EdgeDatas11.empty()){
    for(iter = EdgeDatas11.begin(); iter!=EdgeDatas11.end(); iter++){
      theEdges.push_back( *iter );
      theEdgeTDirs.push_back(1);
    }
  }
  if(!EdgeDatas22.empty()){
    for(iter = EdgeDatas22.begin(); iter!=EdgeDatas22.end(); iter++){
      theEdges.push_back( *iter );
      theEdgeTDirs.push_back(1);
    }
  }
  if(!EdgeDatas12.empty()){
    for(rIter=EdgeDatas12.rbegin(); rIter!=EdgeDatas12.rend(); rIter++){
      theEdges.push_back( *rIter );
      theEdgeTDirs.push_back(-1);
    }
  }
  if(!EdgeDatas21.empty()){
    for(rIter=EdgeDatas21.rbegin(); rIter!=EdgeDatas21.rend(); rIter++){
      theEdges.push_back( *rIter );
      theEdgeTDirs.push_back(-1);
    }
  }
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void 
GridFace::
ConstructGridFaceDatasFromEdges( const vector<GridEdgeData*>& theEdges, 
				 const vector<Standard_Integer>& theEdgeTDirs)
{
  ClearFaces();
  
  Standard_Integer theFirstIndex = 0;
  Standard_Integer nbEdges = theEdges.size();
  
  //1.0 find the first edge index by material type;
  for(Standard_Integer index = 0; index < nbEdges; index++){
    Standard_Integer currIndex = index;
    Standard_Integer nextIndex = NextIndexOfCircularIndices(nbEdges, currIndex);
    
    VertexData* V1 = NULL;
    VertexData* V2 = NULL;

    if(theEdgeTDirs[currIndex]==1){
      V1 = theEdges[currIndex]->GetLastVertex();
    }else{
      V1 = theEdges[currIndex]->GetFirstVertex();
    }

    if(theEdgeTDirs[nextIndex]==1){
      V2 = theEdges[nextIndex]->GetFirstVertex();
    }else{
      V2 = theEdges[nextIndex]->GetLastVertex();
    }

    if(V1!=V2){
      theFirstIndex = nextIndex;
      break;
    }
  }
  
  Standard_Integer theUsedIndiceNum = 0;
  ConstructOneGridFaceDataFromEdges(theEdges, theEdgeTDirs,theFirstIndex, theUsedIndiceNum);
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void
GridFace::
ConstructOneGridFaceDataFromEdges(const vector<GridEdgeData*>& theEdges, 
				  const vector<Standard_Integer>& theEdgeTDirs,
				  const Standard_Integer theFirstIndex,
				  const Standard_Integer theUsedIndiceNum)
{
  vector<Standard_Size> oneGroupEdgeIndices;
  oneGroupEdgeIndices.clear();  

  Standard_Integer theNewFirstIndex = theFirstIndex;

  Standard_Integer nbEdges = theEdges.size();

  for(Standard_Integer index = 0; index < nbEdges; index++){
    Standard_Integer currIndex = CurrIndexOfCircularIndices( nbEdges, (theFirstIndex + index) );
    Standard_Integer nextIndex = NextIndexOfCircularIndices( nbEdges, currIndex );

    oneGroupEdgeIndices.push_back(currIndex);

    VertexData* V1 = NULL;
    VertexData* V2 = NULL;

    if(theEdgeTDirs[currIndex]==1){
      V1 = theEdges[currIndex]->GetLastVertex();
    }else{
      V1 = theEdges[currIndex]->GetFirstVertex();
    }

    if(theEdgeTDirs[nextIndex]==1){
      V2 = theEdges[nextIndex]->GetFirstVertex();
    }else{
      V2 = theEdges[nextIndex]->GetLastVertex();
    }

    if(V1!=V2){
      theNewFirstIndex = nextIndex;
      break;
    }
  }

  ConstructOneGridFaceDataFromEdgeIndices(theEdges, theEdgeTDirs, oneGroupEdgeIndices);

  Standard_Integer theCurrUsedEdgeNum = oneGroupEdgeIndices.size();
  Standard_Integer theTotalUsedEdgeNum = theUsedIndiceNum + theCurrUsedEdgeNum;

  if(theTotalUsedEdgeNum < nbEdges){
    ConstructOneGridFaceDataFromEdges(theEdges, theEdgeTDirs, theNewFirstIndex, theTotalUsedEdgeNum);
  }
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void 
GridFace::
ConstructOneGridFaceDataFromEdgeIndices( const vector<GridEdgeData*>& theEdges,
					 const vector<Standard_Integer>& theEdgeTDirs,
					 const vector<Standard_Size>& oneGroupEdgeIndices)
{
  bool isProper = true;
  CheckEdgeIndices(theEdges, oneGroupEdgeIndices, isProper);

  if(isProper){
    GridFaceData* aNewFace = new GridFaceData(this);
    Standard_Size nb = oneGroupEdgeIndices.size();
    for(Standard_Size i=0; i<nb; i++){
      Standard_Size currIndex = oneGroupEdgeIndices[i];
      aNewFace->AddEdge(theEdges[currIndex], theEdgeTDirs[currIndex]);
      theEdges[currIndex]->AddFace(aNewFace, theEdgeTDirs[currIndex]);
    }

    aNewFace->Setup();
    
    AddFaceData(aNewFace);
  }
}




/****************************************************************/
/*
 * Function : SetupLocalIndexOfGridFaceData()
 * Purpose  : for constructing GridFaceData's local index in this GridFace
 */
/****************************************************************/
void 
GridFace::
SetupLocalIndexOfGridFaceData()
{
  if(m_Faces.empty()) return;
  vector<GridFaceData*>::iterator iter;

  Standard_Integer CurrIndex = 0;

  for(iter=m_Faces.begin(); iter!=m_Faces.end(); iter++){
    (*iter)->SetLocalIndex(CurrIndex);
    CurrIndex++;
  }
}



GridFaceData* GridFace::GetFaceData(const Standard_Integer _localIndex) const
{
  return m_Faces[_localIndex];
}
