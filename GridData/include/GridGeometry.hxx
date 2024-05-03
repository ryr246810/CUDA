#ifndef _GridGeometry_Headerfile
#define _GridGeometry_Headerfile


#include <ZRGrid.hxx>
#include <GridFace.hxx>
#include <GridEdge.hxx>
#include <GridFaceData.cuh>
#include <GridEdgeData.hxx>
#include <GridVertexData.hxx>
#include <GridBndData.hxx>

#include <PMLDataDefine.hxx>
class GridGeometry_Cyl3D;

class GridGeometry
{

public:
  GridGeometry();
  GridGeometry(const ZRGrid* _zrgrid, const GridBndData* _bnddatas);
  ~GridGeometry();

 void Setup();


public:
  void SetZRGrid(const ZRGrid* _zrgrid){ m_ZRGrid = _zrgrid; };
  void SetGridBndDatas(const GridBndData* _gridbnddatas){ m_GridBndDatas = _gridbnddatas; };


public:
  // PML
  void SetPMLAccordingPorts();

  void SetPMLAccordingPorts_GridVertexDatas(const Standard_Integer thePortDir, 
					    const Standard_Integer theStartIndex,
					    const TxSlab2D<Standard_Integer>& thePMLRgn,
					    const Standard_Integer thePMLLayerNum);

  void SetPMLAccordingPorts_GridEdgeDatas(const Standard_Integer thePortDir, 
					  const Standard_Integer theStartIndex,
					  const TxSlab2D<Standard_Integer>& thePMLRgn,
					  const Standard_Integer thePMLLayerNum);

  void SetPMLAccordingPorts_GridFaceDatas(const Standard_Integer thePortDir, 
					  const Standard_Integer theStartIndex,
					  const TxSlab2D<Standard_Integer>& thePMLRgn,
					  const Standard_Integer thePMLLayerNum);

  void SetPMLDataDefine(PMLDataDefine* _pmlDataDefine){
    m_PMLDefineTool = _pmlDataDefine;
  };

  PMLDataDefine* GetPMLDefineTool() const{
    return m_PMLDefineTool;
  }

public:
  //Vertex
  /************************/
  void InitDefineGridVertices();

  void BuildGridVertices();

  void SetupGridVertices();

  void BuildSurroundingGeomElements();

  void BuildGridVertices_With_EdgeBnd_Along(const Standard_Integer aDir);
  void SetGridVertices_AsInShape_InScope(const Standard_Integer aDir,
					 const Standard_Size theIndx,
					 const Standard_Size theFirstGVIndex, 
					 const Standard_Size theLastGVIndex,  
					 const Standard_Integer theShapeIndex);

  void AddSpaceInfoOfAppendingVertexDataOfGridEdge(const GridEdge* theEdge, 
						   AppendingVertexDataOfGridEdge* theVertex);

  void RebuildfVertexDataWithNewInformations(const Standard_Integer theShapeIndex, 
					     const Standard_Integer theFaceIndex, 
					     VertexData* theData);

  void RebuildfVertexDataWithNewInformations(const Standard_Integer theShapeIndex, 
					     const Standard_Integer theEdgeIndex, 
					     const set<Standard_Integer>& theFaceIndices, 
					     VertexData* theData);

  void AddSpaceInfoOfGridVertexData(GridVertexData* theVertex);


  /************************/

public:
  //Edge
  /**************************************************************/
  void InitDefineGridEdges();
  void BuildGridEdges();
  void BuildGridEdgeAppendingVertices();


public:
  //Face
  /************************************************/
  void InitDefineGridFaces();
  void BuildGridFaceAppendingVertices();

  void BuildGridFaceDatas();
  void BuildAppendingEdgesOfGridFaces();
  void BuildBndsOfGridFaceDatas();
  /************************************************/


public:
  // Get Method
  /******************************************************************************************/
  const ZRGrid* GetZRGrid() const { return m_ZRGrid; };

  const GridBndData* GetGridBndDatas() const { return m_GridBndDatas; };
  Standard_Integer GetBackGroundMaterialType() const;
  Standard_Integer GetBackGroundMaterialDataIndex() const;

  Standard_Size GetEdgeSize(Standard_Integer aDir) const {return m_ZRGrid->GetEdgeSize(aDir);};
  Standard_Size GetFaceSize() const {return m_ZRGrid->GetFaceSize();};
  Standard_Size GetVertexSize() const {return m_ZRGrid->GetVertexSize();};

  GridVertexData* GetGridVertices()const{return m_Vertices;};
  GridEdge**      GetGridEdges()const   {return m_Edges;};
  GridFace*       GetGridFaces()const   {return m_Faces;};
  GridGeometry*       GetMinusGridGeometry()const   {return minu_Geometry;};
  GridGeometry*       GetPlusGridGeometry()const   {return plus_Geometry;};

public:
  // GridGeometry3D Set and Get 
  Standard_Integer  GetPhiIndex()const  {return m_PhiIndex;};
  void SetPhiIndex(Standard_Integer phi_index) { m_PhiIndex = phi_index;};

  Standard_Size  GetPhiNumber()const  {return m_PhiNumber;};
  void SetPhiNumber(Standard_Size phi_Number) { m_PhiNumber = phi_Number;};
	
  const GridGeometry_Cyl3D* GetGridGeometry3D()const {return GridGeom3D;};
  void SetGridGeometry3D(GridGeometry_Cyl3D *GridGeom_cyl) { GridGeom3D = GridGeom_cyl;};


public:
  Standard_Integer GetMaterialTypeWithShapeIndex(const Standard_Integer theIndex) const;
  Standard_Integer GetMaterialTypeWithFaceIndex(const Standard_Integer theIndex) const;
  Standard_Integer GetShapeIndexAccordingFaceIndex(const Standard_Integer theIndex) const;


private:
  void DistributeFaceBndVertex_To_SubElement(GridFace* theFace, 
					     Standard_Integer theShapeIndex, 
					     Standard_Integer theEdgeIndex, 
					     const set<Standard_Integer>& theFaceIndices, 
					     Standard_Size aFrac1, 
					     Standard_Size aFrac2, 
					     Standard_Integer aMaterialType);

  void DistributeFaceBndVertex_To_SubEdge(GridFace* theFace, 
					  Standard_Integer theShapeIndex, 
					  Standard_Integer theEdgeIndex, 
					  const set<Standard_Integer>& theFaceIndices, 
					  Standard_Size aFrac1, 
					  Standard_Size aFrac2, 
					  Standard_Integer aMaterialType);
  
  void DistributeFaceBndVertex_To_SubVertex(GridFace* theFace, 
					    Standard_Integer theShapeIndex, 
					    Standard_Integer theEdgeIndex, 
					    const set<Standard_Integer>& theFaceIndices, 
					    Standard_Size aFrac1, 
					    Standard_Size aFrac2, 
					    Standard_Integer aMaterialType);

  void DistributeEdgeBndVertex_To_SubVertex(GridEdge* theEdge, 
					    Standard_Integer aShapeIndex, 
					    Standard_Integer aFaceIndex, 
					    Standard_Size aFrac, 
					    Standard_Integer aMaterialType);


public:
  void RemoveSpaceInfoOfAppendingVertexDataOfGridEdge(const GridEdge* theEdge,
						      AppendingVertexDataOfGridEdge* theVertex);

  void GetMaterialOfGridEdgeOnlyAccordingSpaceDefine(const GridEdge* theEdge, 
						     Standard_Integer& theMaterialType, 
						     set<Standard_Integer>& theMaterialIndices);

  void GetMaterialOfGridVertexDataOnlyAccordingSpaceDefine(const GridVertexData* theVertex, 
							   Standard_Integer& theMaterialType, 
							   set<Standard_Integer>& theMaterialIndices);



public:
  void GetAllGridEdgeDatasNotOfMaterialTypesOfPhysRgn(const set<Standard_Integer>& theMaterials, 
						      const bool isExcludingAxis, 
						      vector<GridEdgeData*>&  theDatas) const;
  
  void GetGridEdgeDatasNotOfMaterialTypesOfSubRgn(const set<Standard_Integer>& theMaterials,
						  const TxSlab2D<Standard_Integer>& subRgn,
						  const bool isExcludingAxis, 
						  vector<GridEdgeData*> & theDatas) const;
  
  void GetGridEdgeDatasNotOfMaterialTypesOfSubRgn(const Standard_Integer edgeDir,
						  const set<Standard_Integer>& theMaterials,
						  const TxSlab2D<Standard_Integer>& subRgn,
						  const bool isExcludingAxis, 
						  vector<GridEdgeData*> & theDatas) const;
  
  void GetGridEdgeDatasNotOfMaterialTypeOfSubRgn(const Standard_Integer theMaterial,
						 const TxSlab2D<Standard_Integer>& subRgn,
						 const bool isExcludingAxis, 
						 vector<GridEdgeData*> & theDatas) const;

  void GetGridEdgeDatasNotOfMaterialTypeAlongAxis(const Standard_Integer theMaterial,
						 const TxSlab2D<Standard_Integer>& subRgn,
						 vector<GridEdgeData*> & theDatas) const;


  void GetAllGridEdgeDatasOfMaterialTypesOfPhysRgn(const set<Standard_Integer>& theMaterials, 
						   const bool isExcludingAxis, 
						   vector<GridEdgeData*>&  theDatas) const;
  
  void GetGridEdgeDatasOfMaterialTypesOfSubRgn(const set<Standard_Integer>& theMaterials,
					       const TxSlab2D<Standard_Integer>& subRgn,
					       const bool isExcludingAxis, 
					       vector<GridEdgeData*> & theDatas) const;
  
  void GetGridEdgeDatasOfMaterialTypesOfSubRgn(const Standard_Integer edgeDir,
					       const set<Standard_Integer>& theMaterials,
					       const TxSlab2D<Standard_Integer>& subRgn,
					       const bool isExcludingAxis, 
					       vector<GridEdgeData*> & theDatas) const;

  void GetGridEdgeDatasOfMaterialTypeOfSubRgn(const Standard_Integer theMaterial,
					      const TxSlab2D<Standard_Integer>& subRgn,
					      const bool isExcludingAxis, 
					      vector<GridEdgeData*> & theDatas) const;


  void GetAllGridEdgeDatasOfPhysRgn(const bool isExcludingAxis, 
				    vector<GridEdgeData*>&  theDatas) const;


public:
  void GetAllGridFaceDatasNotOfMaterialTypesOfPhysRgn(const set<Standard_Integer>& theMaterials, 
						      vector<GridFaceData*>&  theDatas) const;
  
  void GetGridFaceDatasNotOfMaterialTypeOfSubRgn(const Standard_Integer theMaterial,
						 const TxSlab2D<Standard_Integer>& subRgn,
						 vector<GridFaceData*> & theDatas) const;
  
  void GetGridFaceDatasNotOfMaterialTypesOfSubRgn(const set<Standard_Integer>& theMaterials,
						  const TxSlab2D<Standard_Integer>& subRgn,
						  vector<GridFaceData*> & theDatas) const;
  

  void GetAllGridFaceDatasOfMaterialTypesOfPhysRgn(const set<Standard_Integer>& theMaterials, 
						   vector<GridFaceData*>&  theDatas) const;
  
  void GetGridFaceDatasOfMaterialTypeOfSubRgn(const Standard_Integer theMaterial,
					      const TxSlab2D<Standard_Integer>& subRgn,
					      vector<GridFaceData*> & theDatas) const;
  
  void GetGridFaceDatasOfMaterialTypesOfSubRgn(const set<Standard_Integer>& theMaterials,
					       const TxSlab2D<Standard_Integer>& subRgn,
					       vector<GridFaceData*> & theDatas) const;

  void GetAllGridFaceDatasOfPhysRgn(vector<GridFaceData*>&  theDatas) const;


public:
  void GetAllGridVertexDatasNotOfMaterialTypesOfPhysRgn(const set<Standard_Integer>& theMaterials, 
							const bool isExcludingAxis, 
							vector<GridVertexData*>&  theDatas) const;
  
  void GetGridVertexDatasNotOfMaterialTypeOfSubRgn(const Standard_Integer theMaterial,
						   const TxSlab2D<Standard_Integer>& subRgn,
						   const bool isExcludingAxis, 
						   vector<GridVertexData*> & theDatas) const;
  
  void GetGridVertexDatasNotOfMaterialTypesOfSubRgn(const set<Standard_Integer>& theMaterials,
						    const TxSlab2D<Standard_Integer>& subRgn,
						    const bool isExcludingAxis, 
						    vector<GridVertexData*> & theDatas) const;
  
  void GetAllGridVertexDatasOfMaterialTypesOfPhysRgn(const set<Standard_Integer>& theMaterials, 
						     const bool isExcludingAxis, 
						     vector<GridVertexData*>&  theDatas) const;
  
  void GetGridVertexDatasOfMaterialTypeOfSubRgn(const Standard_Integer theMaterial,
						const TxSlab2D<Standard_Integer>& subRgn,
						const bool isExcludingAxis, 
						vector<GridVertexData*> & theDatas) const;
  
  void GetGridVertexDatasOfMaterialTypesOfSubRgn(const set<Standard_Integer>& theMaterials,
						 const TxSlab2D<Standard_Integer>& subRgn,
						 const bool isExcludingAxis, 
						 vector<GridVertexData*> & theDatas) const;

  void GetAllGridVertexDatasOfPhysRgn(const bool isExcludingAxis, vector<GridVertexData*>&  theDatas) const;

public:

  void GetGridEdgeDatasNotOfMaterialTypesAlongAxis(const set<Standard_Integer>& theMaterials,
                                           const TxSlab2D<Standard_Integer>& subRgn,
                                           vector<GridEdgeData*> & theDatas) const;

  void GetGridEdgeDatasOfMaterialTypesAlongAxis(const set<Standard_Integer>& theMaterials,
                                        const TxSlab2D<Standard_Integer>& subRgn,
                                        vector<GridEdgeData*> & theDatas) const;



private:
  GridVertexData* m_Vertices;
  GridFace*       m_Faces;
  GridEdge**      m_Edges;

  const GridBndData* m_GridBndDatas;
  const ZRGrid* m_ZRGrid;
  PMLDataDefine* m_PMLDefineTool;
  
  Standard_Integer  m_PhiIndex;
  Standard_Size  m_PhiNumber;
  GridGeometry * plus_Geometry;
  GridGeometry * minu_Geometry;

  const GridGeometry_Cyl3D *GridGeom3D;


public:
  void SetPlusMinu_Geometry(GridGeometry * Geometry_1, GridGeometry * Geometry_0)
   {
  	plus_Geometry = Geometry_1;
  	minu_Geometry = Geometry_0;
   }
   void Build_Near_Edge();


private:
  void SetNonPMLPortBnd(); 
  bool Does_NonPMLPortRgn_Overlap_With_PMLRgn(const PortData& thePort);
};

#endif
