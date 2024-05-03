#include <GridFace.hxx>
#include <GridGeometry.hxx>


void GridFace::GetOrderedGridVertices(vector<VertexData*>& theVertices) const
{
  Standard_Integer Dir1 = 0;
  Standard_Integer Dir2 = 1;


  Standard_Size indxVec[2];
  GetZRGrid()->FillFaceIndxVec(m_Index, indxVec);


  Standard_Size orgIndx;
  GetZRGrid()->FillVertexIndx(indxVec, orgIndx);


  Standard_Size indx_1 =  GetZRGrid()->bumpVertex(Dir1, orgIndx);
  Standard_Size indx_2 =  GetZRGrid()->bumpVertex(Dir2, indx_1);
  Standard_Size indx_3 =  GetZRGrid()->bumpVertex(Dir2, orgIndx);

  GridVertexData* theOrgVertex = (GetGridGeom()->GetGridVertices())+orgIndx;
  GridVertexData* theVertex_1 =  (GetGridGeom()->GetGridVertices())+indx_1;
  GridVertexData* theVertex_2 =  (GetGridGeom()->GetGridVertices())+indx_2;
  GridVertexData* theVertex_3 =  (GetGridGeom()->GetGridVertices())+indx_3;

  theVertices.clear();

  theVertices.push_back(theOrgVertex);
  theVertices.push_back(theVertex_1);
  theVertices.push_back(theVertex_2);
  theVertices.push_back(theVertex_3);
}
