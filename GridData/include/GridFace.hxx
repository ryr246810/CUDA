#ifndef _GridFace_Headerfile
#define _GridFace_Headerfile

#include <set>

#include <GridEdge.hxx>
#include <GridEdgeData.hxx>
#include <GridFaceData.cuh>

#include <TxVector2D.h>
#include <ZRGrid.hxx>

#include <GridBndDefine.hxx>


class AppendingEdgeDataOfGridFace;
class AppendingVertexDataOfGridFace;
class AppendingVertexDataOfGridEdge;

class GridGeometry;

class GridFace
{
public:
  GridFace();
  GridFace(GridGeometry* _gridgeom,
	   const Standard_Size _index);
  ~GridFace();

  /**************** build method **********/
  void SetGridGeom(GridGeometry* _gridgeom){ m_GridGeom = _gridgeom; };

  void SetIndex(const Standard_Size _index)   { m_Index = _index; };
  void AddFaceData(GridFaceData* _face);

  void AddBndVertexOfGridFace(AppendingVertexDataOfGridFace* _vertex);

  bool HasAppendingVertexOfGridFace();

  void BuildFaces();

  bool IsCut() const;


public:
  /****************************************/
  void ClearFaces();
  void ClearAppendingVertices();
  /****************************************/

  /****************** get *****************/
  const ZRGrid*         GetZRGrid() const ;
  const GridGeometry*      GetGridGeom() const {return m_GridGeom;}

  Standard_Size      GetIndex() const       { return m_Index; };
  Standard_Integer   GetIndexOfDir(Standard_Integer) const;
  void GetVecIndex(Standard_Size indxVec[2]) const;

  Standard_Real           GetArea() const ;
  Standard_Real           GetDualLength() const ;

  Standard_Integer        GetResolution() const;

  TxVector2D<Standard_Real> GetVectorOfDir1() const;
  TxVector2D<Standard_Real> GetVectorOfDir2() const;

  const GridVertexData*   GetLDVertex()const;

  const vector<GridFaceData*>& GetFaces() const {return m_Faces;}
  GridFaceData* GetFaceData(const Standard_Integer _localIndex) const;


  vector<GridFaceData*> GetFacesOfState(Standard_Integer _state) const;
  vector<GridFaceData*> GetFacesOfMaterial(Standard_Integer _material) const;


  GridFaceData* GetGridFaceDataContaining(GridEdgeData* _edgedata)  const;
  GridFaceData* GetGridFaceDataContaining(VertexData* _vertexdata) const ;


  GridFaceData* GetGridFaceDataContaining(Standard_Integer _state,
					  GridEdgeData* _edgedata) const;
  GridFaceData* GetGridFaceDataContaining(Standard_Integer _state,
					  VertexData* _vertexdata) const ;


  const vector<AppendingVertexDataOfGridFace*>& GetAppendingVertices() const {return m_BndVertices;};
  vector<AppendingVertexDataOfGridFace*>& ModifyAppendingVertices() {return m_BndVertices;};


  void GetOutLineGridEdges(GridEdge*& EdgeOfDir11,
			   GridEdge*& EdgeOfDir12, 
			   GridEdge*& EdgeOfDir21,
			   GridEdge*& EdgeOfDir22) const;

  void GetOrderedGridVertices(vector<VertexData*>& theVertices) const;


private:
  void GetAllGridEdgeDatasOfGridFace( vector<GridEdgeData*>& theEdges, 
				      vector<Standard_Integer>& theEdgeTDirs );
  
  void ConstructGridFaceDatasFromEdges( const vector<GridEdgeData*>& theEdges, 
					const vector<Standard_Integer>& theEdgeTDirs);
  
  void ConstructOneGridFaceDataFromEdges(const vector<GridEdgeData*>& theEdges, 
					 const vector<Standard_Integer>& theEdgeTDirs,
					 const Standard_Integer theFirstIndex,
					 const Standard_Integer theUsedIndiceNum);
  
  void ConstructOneGridFaceDataFromEdgeIndices( const vector<GridEdgeData*>& theEdges,
						const vector<Standard_Integer>& theEdgeTDirs,
						const vector<Standard_Size>& oneGroupEdgeIndices);
  
  void SetupLocalIndexOfGridFaceData();


  void CheckEdgeIndices(const vector<GridEdgeData*>& theAllEdges, 
			const vector<Standard_Size>& theUsedEdgeIndices,
			bool& isProper);


private:
  GridGeometry* m_GridGeom;

  Standard_Size m_Index;

  vector<GridFaceData*> m_Faces;

  vector<AppendingVertexDataOfGridFace*> m_BndVertices;
};

#endif
