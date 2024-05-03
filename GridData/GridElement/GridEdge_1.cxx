#include <GridEdge.hxx>

#include <GridEdgeData.hxx>

#include <GridVertexData.hxx>
#include <AppendingVertexDataOfGridEdge.hxx>

#include <GridGeometry.hxx>


void 
GridEdge::
GetVertexSequence( vector<VertexData*>& theVertexSequence )const
{
  theVertexSequence.clear();

  GridVertexData* Vertex1;
  GridVertexData* Vertex2;

  GetTwoEndGridVertices(Vertex1, Vertex2);

  theVertexSequence.push_back(Vertex1);

  vector<AppendingVertexDataOfGridEdge*>::const_iterator iter;

  for(iter = m_Vertices.begin(); iter!= m_Vertices.end(); iter++){
    VertexData* currVertexData = *iter;
    theVertexSequence.push_back(currVertexData);
  }

  theVertexSequence.push_back(Vertex2);
}
