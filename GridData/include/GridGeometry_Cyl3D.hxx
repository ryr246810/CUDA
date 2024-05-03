#ifndef _GridGeometry_Cyl3D_Headerfile
#define _GridGeometry_Cyl3D_Headerfile


#include <ZRGrid.hxx>
#include <GridFace.hxx>
#include <GridEdge.hxx>
#include <GridFaceData.cuh>
#include <GridEdgeData.hxx>
#include <GridVertexData.hxx>
#include <GridBndData.hxx>

#include <PMLDataDefine.hxx>
#include <GridGeometry.hxx>


class GridGeometry_Cyl3D
{

public:
  GridGeometry_Cyl3D();
  GridGeometry_Cyl3D(const ZRGrid* _zrgrid, const GridBndData* _bnddatas, const Standard_Size _phiDim);
  ~GridGeometry_Cyl3D();

 void Setup();
 void Build_Near_Edge();


public:
  void SetZRGrid(const ZRGrid* _zrgrid){ m_ZRGrid = _zrgrid; };
  void SetGridBndDatas(const GridBndData* _gridbnddatas){ m_GridBndDatas = _gridbnddatas; };



  void SetPMLDataDefine(PMLDataDefine* _pmlDataDefine){
    m_PMLDefineTool = _pmlDataDefine;
  };

  PMLDataDefine* GetPMLDefineTool() const{
    return m_PMLDefineTool;
  }



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

  Standard_Size GetDimPhi() const {return m_Dimphi;};
  const GridGeometry*  GetGridGeometry(Standard_Integer i)const {return m_Gridgeometry[i];};
  const vector<GridGeometry*> GetGridGeometry()const {return m_Gridgeometry;};

public:
  void GetAllGridEdgeDatasNotOfMaterialTypesOfPhysRgn(const set<Standard_Integer>& theMaterials, 
						      const bool isExcludingAxis, 
						      vector<GridEdgeData*>&  theDatas) const;
  
  void GetAllGridEdgeDatasOfMaterialTypesOfPhysRgn(const set<Standard_Integer>& theMaterials, 
						   const bool isExcludingAxis, 
						   vector<GridEdgeData*>&  theDatas) const;
  
  void GetGridEdgeDatasOfMaterialTypeOfSubRgn(const Standard_Integer theMaterial,
					      const TxSlab2D<Standard_Integer>& subRgn,
					      const bool isExcludingAxis, 
					      vector<GridEdgeData*> & theDatas) const;


  void GetAllGridEdgeDatasOfPhysRgn(const bool isExcludingAxis, 
				    vector<GridEdgeData*>&  theDatas) const;


public:
  void GetAllGridFaceDatasNotOfMaterialTypesOfPhysRgn(const set<Standard_Integer>& theMaterials, 
						      vector<GridFaceData*>&  theDatas) const;

  void GetAllGridFaceDatasOfMaterialTypesOfPhysRgn(const set<Standard_Integer>& theMaterials, 
						   vector<GridFaceData*>&  theDatas) const;
  
  void GetGridFaceDatasOfMaterialTypeOfSubRgn(const Standard_Integer theMaterial,
					      const TxSlab2D<Standard_Integer>& subRgn,
					      vector<GridFaceData*> & theDatas) const;
  
  void GetAllGridFaceDatasOfPhysRgn(vector<GridFaceData*>&  theDatas) const;

public:

  void GetAllGridVertexDatasNotOfMaterialTypesOfPhysRgn(const set<Standard_Integer>& theMaterials, 
							const bool isExcludingAxis, 
							vector<GridVertexData*>&  theDatas) const;
  
  
  void GetAllGridVertexDatasOfMaterialTypesOfPhysRgn(const set<Standard_Integer>& theMaterials, 
						     const bool isExcludingAxis, 
						     vector<GridVertexData*>&  theDatas) const;
  
  void GetGridVertexDatasOfMaterialTypeOfSubRgn(const Standard_Integer theMaterial,
						const TxSlab2D<Standard_Integer>& subRgn,
						const bool isExcludingAxis, 
						vector<GridVertexData*> & theDatas) const;
  

  void GetAllGridVertexDatasOfPhysRgn(const bool isExcludingAxis, vector<GridVertexData*>&  theDatas) const;


public:
  void GetAllGridEdgeDatasNotOfMaterialTypesAlongAxis(const set<Standard_Integer>& theMaterials,
                                               vector<GridEdgeData*>&  theDatas) const;


  void GetAllGridEdgeDatasOfMaterialTypesAlongAxis(const set<Standard_Integer>& theMaterials,
                                            vector<GridEdgeData*>& theDatas) const;


  void GetAllGridEdgeDatasAlongAxis(vector<GridEdgeData*>& theDatas) const;


public:
  Standard_Integer GetMaterialTypeWithShapeIndex(const Standard_Integer theIndex) const;
  Standard_Integer GetMaterialTypeWithFaceIndex(const Standard_Integer theIndex) const;
  Standard_Integer GetShapeIndexAccordingFaceIndex(const Standard_Integer theIndex) const;

private:
  vector<GridGeometry*>  m_Gridgeometry;
  Standard_Size   m_Dimphi;


  const GridBndData* m_GridBndDatas;
  const ZRGrid* m_ZRGrid;
  PMLDataDefine* m_PMLDefineTool;




};

#endif
