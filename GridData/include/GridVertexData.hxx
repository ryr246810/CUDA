#ifndef _GridVertex_Headerfile
#define _GridVertex_Headerfile

#include <VertexData.hxx>
#include <ZRGrid.hxx>

#include <set>
using namespace std;

class GridFaceData;
class GridEdgeData;
class GridGeometry;
class T_Element;

class GridVertexData: public VertexData
{
public:
  GridVertexData();
  GridVertexData(GridGeometry* _gridgeom,
		 Standard_Size _index);
  GridVertexData(GridGeometry* _gridgeom, 
		 Standard_Size _index,
		 Standard_Integer _mark);
  ~GridVertexData();



public:
  virtual void Setup();
  virtual void SetupGeomDimInf();
  virtual void SetupMaterialData();

  /******************* Set ************************/
public:
  void SetIndex(Standard_Size _index)  { m_Index = _index; };
  void SetGridGeom(GridGeometry* _gridgeom)  { m_GridGeom = _gridgeom; };
  /************************************************/



  /******************* Get ************************/
public:
  Standard_Size GetIndex() const;
  void GetVecIndex(Standard_Size indxVec[2]) const;
  const ZRGrid* GetZRGrid() const;
  const GridGeometry* GetGridGeom() const {return m_GridGeom;}
  virtual TxVector2D<Standard_Real> GetLocation() const;
  /************************************************/


public:
  void BuildSharingGridFaceDatas();
  void BuildDivTEdges();
  void BuildSharedTDFaces();



public:
  const vector<GridFaceData*>& GetSharingGridFaceDatas() const{ return m_GridFaceDatas;};
  const vector<T_Element>&    GetSharingDivTEdges(Standard_Integer dir) const{ return m_SharedDivTEdges[dir];};
  const vector<T_Element>&    GetSharedTDFaces() const{ return m_SharedTDFaces;};

  bool IsSharedDFacesPhysDataDefined();

public:
  void ClearSharingElems();
  void ClearSharingGridFaceDatas();
  void ClearSharingTDFaces();
  void ClearSharingDivTEdges();

public:
  virtual Standard_Real GetDualSweptGeomDim() const{
    return m_DualAreaOfSweptEdge;
  }

  virtual Standard_Real GetSweptGeomDim() const;

  void ComputeDualSweptGeomDim();
  

private:
  vector<GridFaceData*>  m_GridFaceDatas;    // used to compute Node B_phi;

  vector<T_Element>  m_SharedDivTEdges[2];  // used to compute div
  vector<T_Element>  m_SharedTDFaces;    // degenerative faces to update E_Phi
  /*
   * used to advance E_phi;  
   * one GridVertexData is a degenerative edge(will generate a swept edge), E_Phi is defined on this element,  
   */

private:
  Standard_Size    m_Index;
  GridGeometry* m_GridGeom;


private:
  Standard_Real m_DualAreaOfSweptEdge;
  
public:
  Standard_Real m_C[2];
  //Standard_Real m_C2;
};


#endif
