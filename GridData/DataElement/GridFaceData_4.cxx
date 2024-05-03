#include <GridFaceData.cuh>
#include <AppendingEdgeData.hxx>
#include <AppendingVertexDataOfGridFace.hxx>
#include <AppendingVertexDataOfGridEdge.hxx>
#include <GridFace.hxx>

const vector<AppendingEdgeData*>& 
GridFaceData::
GetAppendingEdgeDatas() const
{
  return m_AppendingEdges;
}


void 
GridFaceData::
ClearAppendingEdge()
{
  vector<AppendingEdgeData*>::iterator iter;
  for(iter=m_AppendingEdges.begin(); iter!=m_AppendingEdges.end(); iter++){
    AppendingEdgeData* tmpPtr = *iter;
    *iter = NULL;
    delete tmpPtr;
  }

  m_AppendingEdges.clear();
}


void 
GridFaceData::
SetupAppendingEdge()
{
  VertexData* firstV = this->GetLastVertex();
  VertexData* lastV  = this->GetFirstVertex();

  bool founded = false;

  if(!m_BaseGFace->IsCut()){
    return;
  }

  if(firstV!=lastV){  // full filled GridFace
    vector<AppendingVertexDataOfGridFace*> theFaceBndVertices;
    GetBndVerticesOfGridFace(firstV, lastV, theFaceBndVertices, founded);
    if(founded){
      SetupAppendingEdge_Tool(firstV,  lastV, theFaceBndVertices);
    }
  }

  const vector<T_Element>&  theOutLineEdges = this->GetOutLineTEdge();
  Standard_Size nb = theOutLineEdges.size();
  for(Standard_Size i=0; i<(nb-1); i++){
    GridEdgeData* currEdgeData = (GridEdgeData*) theOutLineEdges[i].GetData();
    GridEdgeData* nextEdgeData = (GridEdgeData*) theOutLineEdges[i+1].GetData();

    Standard_Integer currRelatedDir = theOutLineEdges[i].GetRelatedDir();
    Standard_Integer nextRelatedDir = theOutLineEdges[i+1].GetRelatedDir();

    VertexData* curLastV   = currEdgeData->GetLastVertex(currRelatedDir);
    VertexData* nextFirstV = nextEdgeData->GetFirstVertex(nextRelatedDir);

    if(curLastV!=nextFirstV){
      vector<AppendingVertexDataOfGridFace*> tmpFaceBndVertices;
      GetBndVerticesOfGridFace(curLastV, nextFirstV, tmpFaceBndVertices, founded);
      if(founded){
	SetupAppendingEdge_Tool(curLastV,  nextFirstV, tmpFaceBndVertices);
      }
    }
  }
}


void 
GridFaceData::
SetupAppendingEdge_Tool(VertexData* _firstV, 
			VertexData* _lastV, 
			vector<AppendingVertexDataOfGridFace*>& theFaceBndVertices)
{
  Standard_Size nb = theFaceBndVertices.size();
  if(nb>0){
    SetupOneAppendingEdge(_firstV, theFaceBndVertices[0]);
    for(Standard_Size j = 0; j<nb-1 ;j++){
      SetupOneAppendingEdge(theFaceBndVertices[j], theFaceBndVertices[j+1]);
    }
    SetupOneAppendingEdge(theFaceBndVertices[nb-1],_lastV);
  }else{
    SetupOneAppendingEdge(_firstV,_lastV);
  }
}


void 
GridFaceData::
SetupOneAppendingEdge(VertexData* _firstV, 
		      VertexData* _secondV)
{
  Standard_Integer aMark = BND;
  AppendingEdgeData* aNewAppendingEdge = new AppendingEdgeData(aMark, _firstV, _secondV);
  //AppendingEdgeData* aNewAppendingEdge = new AppendingEdgeData(aMark, _firstV, _secondV, this);// rzp, 2019, 04
  aNewAppendingEdge->Setup();
  m_AppendingEdges.push_back(aNewAppendingEdge);
}


// according the first AppendingVertexDataOfGridEdge's face index and the last AppendingVertexDataOfGridEdge's face index
// find the ordered face bnd vertices connecting the two AppendingVertexDatas 
void 
GridFaceData::
GetBndVerticesOfGridFace(VertexData* firstV,
			 VertexData* lastV,
			 vector<AppendingVertexDataOfGridFace*>& theFaceBndVertices,
			 bool& founded)
{
  founded = false;

  vector<AppendingVertexDataOfGridFace*> theBndVertices = m_BaseGFace->GetAppendingVertices();
  GetBndVerticesOfGridFace_Tool(firstV, theBndVertices, theFaceBndVertices);

  Standard_Size nb = theFaceBndVertices.size();
  if(nb>0){
    VertexData* theLastPushVertex = theFaceBndVertices[nb-1];
    if(theFaceBndVertices[nb-1]->HasCommonFaceIndexWith(lastV)){
      founded = true;
    }else{
      founded = false;
      //cout<<"error-----------------GridFaceData::GetBndVerticesOfGridFace-----------------------1"<<endl;
    }
  }else{
    if(firstV->HasCommonFaceIndexWith(lastV)){
      founded = true;
    }else{
      // founded = true;
      // modified 2017.03.30, must be CHECKED---------------->>>
      founded = false;
      //cout<<"error-----------------GridFaceData::GetBndVerticesOfGridFace-----------------------2"<<endl;
    }
  }

  theBndVertices.clear();
}


// needed to be modified
void 
GridFaceData::
GetBndVerticesOfGridFace_Tool(VertexData* theRefVertex,
			      vector<AppendingVertexDataOfGridFace*>& theBndVertices,
			      vector<AppendingVertexDataOfGridFace*>& theFaceBndVertices)
{
  set<Standard_Integer> theCommonFaceIndices;
  Standard_Size theNumOfFaceIndices = 0;
  bool founded = false;

  vector<AppendingVertexDataOfGridFace*>::iterator iter;
  vector<AppendingVertexDataOfGridFace*>::iterator foundedIter;

  for(iter = theBndVertices.begin(); iter!=theBndVertices.end(); iter++){
    (*iter)->GetCommonFaceIndicesWith(theRefVertex, theCommonFaceIndices);
    Standard_Size tmpNumOfFaceIndices = theCommonFaceIndices.size();
    if(tmpNumOfFaceIndices>theNumOfFaceIndices){
      theNumOfFaceIndices = tmpNumOfFaceIndices;
      founded = true;
      foundedIter = iter;
    }
  }

  if(founded){
    AppendingVertexDataOfGridFace* theNewRefVertex = *foundedIter;
    theFaceBndVertices.push_back(theNewRefVertex);
    theBndVertices.erase(foundedIter);

    GetBndVerticesOfGridFace_Tool(theNewRefVertex, theBndVertices, theFaceBndVertices);  // loop this tool function
  }
}
