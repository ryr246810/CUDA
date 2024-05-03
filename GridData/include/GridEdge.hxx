#ifndef _GridEdge_Headerfile
#define _GridEdge_Headerfile

#include <TxVector2D.h>
#include <ZRGrid.hxx>

#include <GridBndDefine.hxx>
#include <GridGeometry.hxx>

class AppendingVertexDataOfGridEdge;
class GridEdgeData;
class GridVertexData;
class GridGeometry;
class VertexData;

class GridEdge
{
public:
  /************** consturctor *************/
  GridEdge();
  GridEdge(GridGeometry* _gridgeom, 
	   ZRGridLineDir _dir,
	   Standard_Size _index);
  ~GridEdge();
  /****************************************/

  /**************** build method **********/
  void SetGridGeom(GridGeometry* _gridgeom){ m_GridGeom = _gridgeom; };
  void SetDir(ZRGridLineDir _dir){ m_Dir = _dir; };
  void SetIndex(Standard_Size _index){ m_Index = _index; };

  void AddAppendingVertex(AppendingVertexDataOfGridEdge* aIrregVertex);

  void BuildEdges();

  /****************************************/


  /****************** get *****************/
  const ZRGrid*       GetZRGrid() const;
  const GridGeometry*    GetGridGeom() const {return m_GridGeom;}


  Standard_Size    GetIndex() const    { return m_Index; };
  Standard_Integer GetIndexOfDir(Standard_Integer) const;
  void GetVecIndex(Standard_Size indxVec[2]) const;

  Standard_Integer GetDir() const      { return (Standard_Integer)(m_Dir); };
  Standard_Real    GetLength() const;
  Standard_Real    GetDualArea() const;

  bool             HasAppending() const; 

  void GetTwoEndGridVertices(GridVertexData*& Vertex1, GridVertexData*& Vertex2) const;

  GridVertexData*         GetFirstVertex()const;
  GridVertexData*         GetLastVertex()const;

  TxVector2D<Standard_Real> GetVector() const;

  Standard_Integer        GetResolution() const;


  const vector<GridEdgeData*>& GetEdges() const {return m_Edges;};
  GridEdgeData* GetEdgeData(const Standard_Integer _localIndex) const;

  vector<GridEdgeData*> GetEdgesOfState(Standard_Integer _state) const;
  vector<GridEdgeData*> GetEdgesOfMaterial(Standard_Integer _material) const;

  const vector<AppendingVertexDataOfGridEdge*>& GetAppendingVertices(){return m_Vertices;};
  vector<AppendingVertexDataOfGridEdge*>& ModifyAppendingVertices(){return m_Vertices;};


  /**************** tool ******************/
  void ClearEdges();
  void ClearAppendingVertices();
  /****************************************/

  void GetVertexSequence( vector<VertexData*>& theVertexSequence )const;

private:
  void BuildGridEdgeDatas(vector<VertexData*>::const_iterator& breakIter, 
			  const vector<VertexData*>& allVertices);

  void BuildOneGridEdgeData(const vector<VertexData*>& oneEdgeVertices);

  void SetupLocalIndexOfGridEdgeData();


private:
  GridGeometry* m_GridGeom;
  ZRGridLineDir   m_Dir;
  Standard_Size      m_Index;

  vector<AppendingVertexDataOfGridEdge*> m_Vertices; 
  vector<GridEdgeData*> m_Edges;
};

#endif

