#ifndef _Model_Ctrl_Headerfile
#define _Model_Ctrl_Headerfile

#include <OCCInclude.hxx>
#include <GeomBndDataDefine.hxx>
#include <set>

class Model_Ctrl{
public:
  Model_Ctrl();
  ~Model_Ctrl();


  //2017.03.23--------------------------------------------->>>
public:
  void AnalysisShape();
  bool IsOneCoordDirrection(const gp_Dir& V, bool& isParallelX, bool& isParallelY, bool& isParallelZ);
  const map<Standard_Real, Standard_Integer, less<Standard_Real> >* GetSpecialCoordDatas(const Standard_Integer dir) const;
  const vector<Standard_Real>* GetSpecialCoordVec(const Standard_Integer dir) const;

  //2017.03.23---------------------------------------------<<<


  /****************************  for Shape and faces  ********************************/
public:
  void AppendShape(const TopoDS_Shape & theShape, const Standard_Integer theMaterial);
  void EraseShape(const TopoDS_Shape & theShape);

  void AppendSpecialFace(const TopoDS_Face& theFace, const Standard_Integer theMaterialType);
  void EraseSpecialFace(const TopoDS_Face & theFace);

  void Setup();



public:
  void SetShapeMask(const TopoDS_Shape & theShape, const Standard_Integer theMask);
  void EraseShapeMask(const TopoDS_Shape & theShape);
  void SetSpecialFaceMask(const TopoDS_Face& theFace, const Standard_Integer theMask);
  void EraseSpecialFaceMask(const TopoDS_Face & theFace);

  const TColStd_DataMapOfIntegerInteger& GetFacesMask()    const {return m_FaceMaskWithIndexTool;};
  const TColStd_DataMapOfIntegerInteger& GetShapesMask()   const {return m_ShapeMaskWithIndexTool;};



private: 
  void Setup_Map_Of_Shape_Mask_Index();
  void Setup_Map_Of_Face_Mask_Index();

  void MapMaskToShapeIndex(const TopoDS_Shape& theShape,
			   const Standard_Integer theMask);
  void MapMaskToFaceIndex(const TopoDS_Face& theFace,
			  const Standard_Integer theMask);

  void ClearShapesMask();
  void ClearFacesMask();
  void ResetShapesMaskDefine();
  void ResetFacesMaskDefine();



public:
  bool HasShapeIndex(const Standard_Integer theIndex) const;
  Standard_Integer GetShapeIndex(const TopoDS_Shape& theShape) const;
  const TopoDS_Shape& GetShapeWithIndex(const Standard_Integer theIndex) const;
  Standard_Integer GetMaterialTypeWithShapeIndex(const Standard_Integer theIndex) const;

  bool HasFaceIndex(const Standard_Integer theIndex) const;
  Standard_Integer GetFaceIndex(const TopoDS_Face& theFace) const;
  const TopoDS_Face& GetFaceWithIndex(const Standard_Integer theIndex) const;

  Standard_Integer GetMaterialTypeWithFaceIndex(const Standard_Integer theIndex) const;


public:
  bool IsPort(const Standard_Integer aFaceIndex) const;
  bool IsPort(const TopoDS_Face & aFace) const;

  void ComputeBndBoxOfPort(const Standard_Integer thePortIndex, TxSlab<Standard_Real>& rgn) const;
  void ComputePortDirWithFaceIndexOfPort(const Standard_Integer theFaceIndexOfPort, 
					 gp_Pnt& theBaryCenter, 
					 GridLineDir& theLineDir, 
					 Standard_Integer& theRelativeDir) const;

  Standard_Integer GetPortTypeWithPortIndex(const Standard_Integer theIndex) const;


public:
  void Write_Faces();
  void Write_Vertices();
  void Write_Edges();



public:
  const TopTools_DataMapOfShapeInteger&  GetShapesWithIndex() const {return m_ShapesWithIndexTool;};
  const TopTools_DataMapOfIntegerShape&  GetIndexedShapes()   const {return m_IndexWithShapesTool;};

  const TopTools_DataMapOfShapeInteger&  GetFacesWithIndex()  const {return m_FacesWithIndexTool;};
  const TopTools_DataMapOfIntegerShape&  GetIndexedFaces()    const {return m_IndexWithFacesTool;};

  const TopTools_DataMapOfShapeInteger&  GetShapesWithType()  const {return m_ShapesWithTypeTool;};
  const TColStd_DataMapOfIntegerInteger& GetFacesWithType()   const {return m_FacesWithTypeTool;};
  const TColStd_DataMapOfIntegerInteger& GetPortsWithType()   const {return m_PortsWithTypeTool;};


  const TColStd_DataMapOfIntegerInteger& GetFacesWithShape()         const {return m_FacesWithShapeTool;};
  const TColStd_DataMapOfIntegerListOfInteger& GetEdgesWithFace()    const {return m_EdgeWithFaceTool;};   // 2017.02.04
  const TColStd_DataMapOfIntegerListOfInteger& GetVerticesWithEdge() const {return m_VertexWithEdgeTool;}; // 2017.02.04


  const TopTools_DataMapOfShapeInteger&  GetVerticesWithIndex() const {return m_VerticesWithIndexTool;};
  const TopTools_DataMapOfIntegerShape&  GetIndexWithVertices() const {return m_IndexWithVerticesTool;};
  const TColStd_DataMapOfIntegerListOfInteger& GetVertexWithFace() const {return m_VertexWithFaceTool;};
  const TColStd_DataMapOfIntegerListOfInteger& GetFaceWithVertex() const {return m_FaceWithVertexTool;};


  const TopTools_DataMapOfShapeInteger&  GetEdgesWithIndex() const {return m_EdgesWithIndexTool;};
  const TopTools_DataMapOfIntegerShape&  GetIndexWithEdges() const {return m_IndexWithEdgesTool;};
  const TColStd_DataMapOfIntegerListOfInteger& GetEdgeWithFace() const {return m_EdgeWithFaceTool;};
  const TColStd_DataMapOfIntegerListOfInteger& GetFaceWithEdge() const {return m_FaceWithEdgeTool;};




  /*****************************************************************************/
private:
  void SetupShapeIndex();
  void ResetShapesIndex();

  void ClearShapesTypeDefine();

  void SetupFaceIndex();
  void ResetFacesIndex();

  void SetupPorts();


private:
  /****************************  for special faces  ****************************/
  void InitTypeToFace();
  void SetSpecialTypeToFace();
  void ClearFacesTypeDefine();

  void SetSpecialTypeToFace(const Standard_Integer theFaceIndex, 
			    const Standard_Integer theMaterialType);
  void SetSpecialTypeToFace(const TopoDS_Face& theFace, 
			    const Standard_Integer theMaterialType);

  void ClearPortsDefine();


  /****************************  for face and vertex ***************************/
  void Setup_VertexIndex();
  void Reset_VerticesIndex();


  void Setup_VertexFaceRelation();
  void Setup_FaceWithVertexTool();
  void Setup_VertexWithFaceTool();


  void Setup_VertexEdgeRelation();
  void Setup_VertexWithEdgeTool();
  void Setup_EdgeWithVertexTool();


  /****************************  for face and vertex ***************************/
  void Setup_EdgeFaceRelation();
  void Setup_EdgeIndex();
  void Setup_FaceWithEdgeTool();
  void Setup_EdgeWithFaceTool();
  void Reset_EdgesIndex();


  /****************************  for special ports  ****************************/
private:
  bool IsPlaneWithFaceIndex(const Standard_Integer theFaceIndex) const;
  bool IsFaceAsPlaneWithFaceIndex(const Standard_Integer theFaceIndex) const;



  void ComputeNormalVectorOfPort(const Standard_Integer theFaceIndexOfPort, 
				 gp_Pnt& theBaryCenter,
				 gp_Vec& V) const;

  void ComputeNormalVectorOfFaceWithPnt(const Standard_Integer theFaceIndex, 
					const gp_Pnt& the3DPnt,
					gp_Pnt& thePntOnFace, 
					gp_Vec& V) const;

  void ComputeBaryCenterNormalDirOfFace(const Standard_Integer theFaceIndex, 
					gp_Pnt& theBaryCenter,
					GridLineDir& theLineDir, 
					Standard_Integer& theRelativeDir) const;

  void ComputeBaryCenterNormalVectorOfFace(const Standard_Integer theFaceIndex, 
					   gp_Pnt& theBaryCenter,
					   gp_Vec& V) const;

  bool IsPlaneWithSpecialDir(const Standard_Integer theFaceIndex, 
			     const GridLineDir theDir) const;

  bool CanBeDefinedAsPort(const TopoDS_Face& Face) const;
  bool CanBeDefinedAsPort(const Standard_Integer theFaceIndexOfPort) const;

  bool IsPortDefinedExactly(const TopoDS_Face& Face) const;
  bool IsPortDefinedExactly(const Standard_Integer theFaceIndexOfPort) const;

  bool HasCorrectDirForDefiningFaceAsPort(const TopoDS_Face& Face) const;
  bool HasCorrectDirForDefiningFaceAsPort(const Standard_Integer theFaceIndexOfPort) const;

  bool HasCorrectGeomTypeForDefiningFaceAsPort(const TopoDS_Face& Face) const;
  bool HasCorrectGeomTypeForDefiningFaceAsPort(const Standard_Integer theFaceIndexOfPort) const;

  void SetPortToFace(const TopoDS_Face& theFace, 
		     const Standard_Integer thePortType);
  void SetPortToFace(const Standard_Integer theFaceIndexOfPort, 
		     const Standard_Integer thePortType);

  void UnsetPortOfFace(const TopoDS_Face & theFace);
  void UnsetPortOfFace(const Standard_Integer theFaceIndexOfPort);

  Standard_Integer GetPortIndex(const TopoDS_Face& theFace) const;


public:
  // no doubted function, because of the topological frame of the shape
  bool IsFaceBelongToOneShape(const Standard_Integer theFaceIndex, 
			      Standard_Integer& theShapeIndex) const;
  bool DoesShapeHasFace(const Standard_Integer theShapeIndex,
			const Standard_Integer theFaceIndex) const;

  void CheckAndGetShapeIndexFromFaceIndices(const vector<Standard_Integer>& theFaceIndices,
					    Standard_Integer& theShapeIndex, 
					    bool& isFacesBelongOneShape) const;

  void CheckAndGetShapeIndexFromFaceIndices(const TColStd_ListOfInteger & theFaceIndices, 
					    Standard_Integer& theShapeIndex,
					    bool& isFacesBelongOneShape) const;


  void FindNeighBourFacesOfOneFace(const Standard_Integer theFaceIndex, std::set<Standard_Integer>& theNBFaceIndices) const;



public:
  void CheckAndGetShapeIndexFromVertexIndex(const Standard_Integer theVertexIndex, 
					    Standard_Integer& theShapeIndex, 
					    bool& isVertexBelongOneShape) const;

private:
  TopTools_DataMapOfShapeInteger  m_ShapesWithTypeTool;
  TopTools_DataMapOfShapeInteger  m_SpecialFacesWithTypeTool;

  TopTools_DataMapOfShapeInteger  m_ShapesWithMaskTool;
  TopTools_DataMapOfShapeInteger  m_SpecialFacesWithMaskTool;

  TColStd_DataMapOfIntegerInteger m_ShapeMaskWithIndexTool;
  TColStd_DataMapOfIntegerInteger m_FaceMaskWithIndexTool;


  TopTools_DataMapOfShapeInteger  m_ShapesWithIndexTool;
  TopTools_DataMapOfIntegerShape  m_IndexWithShapesTool;


  TopTools_DataMapOfShapeInteger  m_FacesWithIndexTool;
  TopTools_DataMapOfIntegerShape  m_IndexWithFacesTool;

  TColStd_DataMapOfIntegerInteger m_FacesWithShapeTool;

  TColStd_DataMapOfIntegerInteger m_FacesWithTypeTool;
  TColStd_DataMapOfIntegerInteger m_PortsWithTypeTool;


  TopTools_DataMapOfShapeInteger  m_VerticesWithIndexTool;
  TopTools_DataMapOfIntegerShape  m_IndexWithVerticesTool;
  TColStd_DataMapOfIntegerListOfInteger m_VertexWithFaceTool;
  TColStd_DataMapOfIntegerListOfInteger m_FaceWithVertexTool;


  TColStd_DataMapOfIntegerListOfInteger m_VertexWithEdgeTool;  
  TColStd_DataMapOfIntegerListOfInteger m_EdgeWithVertexTool;


  TopTools_DataMapOfShapeInteger  m_EdgesWithIndexTool;
  TopTools_DataMapOfIntegerShape  m_IndexWithEdgesTool;

  TColStd_DataMapOfIntegerListOfInteger m_EdgeWithFaceTool;
  TColStd_DataMapOfIntegerListOfInteger m_FaceWithEdgeTool;



  map<Standard_Real, Standard_Integer, less<Standard_Real> > m_SpecialXDatas;
  map<Standard_Real, Standard_Integer, less<Standard_Real> > m_SpecialYDatas;
  map<Standard_Real, Standard_Integer, less<Standard_Real> > m_SpecialZDatas;

  vector<Standard_Real> m_SpecialXCoordVec;
  vector<Standard_Real> m_SpecialYCoordVec;
  vector<Standard_Real> m_SpecialZCoordVec;
};

#endif

