
#include <GridBndData.hxx>




void 
GridBndData::
GetFaceIndices_With_VertexIndex(const Standard_Integer theVertexIndex,
				set<Standard_Integer>& theFaceIndices) const
{
  theFaceIndices.clear();

  set<Standard_Integer> theEdgeIndices;
  GetEdgeIndices_With_VertexIndex(theVertexIndex, theEdgeIndices);


  set<Standard_Integer>::iterator iter;
  for(iter=theEdgeIndices.begin(); iter!=theEdgeIndices.end(); iter++){
    Standard_Integer theEdgeIndex = *iter;
    map<Standard_Integer, vector<Standard_Integer> >::const_iterator mapIter = m_EdgesWithFaceTool.find(theEdgeIndex);
    if(mapIter!=m_EdgesWithFaceTool.end()){
      const vector<Standard_Integer>& tmpEdgeIndices = mapIter->second;
      vector<Standard_Integer>::const_iterator vecIter;
      for(vecIter = tmpEdgeIndices.begin(); vecIter!=tmpEdgeIndices.end(); vecIter++){
	theFaceIndices.insert(*vecIter);
      }
    }
  }
}



void 
GridBndData::
GetFaceIndices_With_EdgeIndex(const Standard_Integer theEdgeIndex,
			      set<Standard_Integer>& theFaceIndices) const
{
  theFaceIndices.clear();

  map<Standard_Integer, vector<Standard_Integer> >::const_iterator iter = m_EdgesWithFaceTool.find(theEdgeIndex);
  if(iter!=m_EdgesWithFaceTool.end()){
    const vector<Standard_Integer>& tmpEdgeIndices = iter->second;
    vector<Standard_Integer>::const_iterator vecIter;
    for(vecIter = tmpEdgeIndices.begin(); vecIter!=tmpEdgeIndices.end(); vecIter++){
      theFaceIndices.insert(*vecIter);
    }
  }
}



void 
GridBndData::
GetEdgeIndices_With_VertexIndex(const Standard_Integer theVertexIndex,
				set<Standard_Integer>& theEdgeIndices) const
{
  theEdgeIndices.clear();

  map<Standard_Integer, vector<Standard_Integer> >::const_iterator iter = m_VerticesWithEdgeTool.find(theVertexIndex);
  if(iter!=m_VerticesWithEdgeTool.end()){
    const vector<Standard_Integer>& tmpEdgeIndices = iter->second;
    vector<Standard_Integer>::const_iterator vecIter;
    for(vecIter = tmpEdgeIndices.begin(); vecIter!=tmpEdgeIndices.end(); vecIter++){
      theEdgeIndices.insert(*vecIter);
    }
  }

}
