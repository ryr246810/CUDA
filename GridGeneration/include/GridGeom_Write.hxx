#ifndef _GridGeom_Write_TMP_HeaderFile
#define _GridGeom_Write_TMP_HeaderFile


#include <GridGeometry.hxx>
#include <ZRDefine.hxx>


using namespace std;

class GridGeom_Write
{
public:
  GridGeom_Write();
  GridGeom_Write(GridGeometry* _Data, ZRDefine* _zrdefine);
  ~GridGeom_Write();

public:
  void WriteInShapeGridVertices(const Standard_Integer shapeMask);

  void WriteGridEdges(const Standard_Integer dir);

  void WriteAppendingVerticesOfGridEdges(const Standard_Integer dir);

  void WriteAppendingVerticesOfGridFaces();

  void WriteInShapeGridEdgeDatas(const Standard_Integer shapeMask, 
				 const Standard_Integer dir);

  void WriteInShapeGridFaceDatas(const Standard_Integer shapeMask);

  void WriteAppendingEdgeDatasOfGridFaceDatasInShape(const Standard_Integer shapeMask);

private:
  GridGeometry* m_Data;
  ZRDefine* m_ZRDefine;
};

#endif
