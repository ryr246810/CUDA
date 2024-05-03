#ifndef _GridBndData_HeaderFile
#define _GridBndData_HeaderFile

#include <GridBndDefine.hxx>
#include <GridMaterialDataDefine.hxx>
#include <TxHierAttribSet.h>

#include <map>
#include <vector>
#include <set>
using namespace std;


class GridBndData
{
public:
  GridBndData();
  ~GridBndData();
  void SetAttrib(const std::string& workPath,  const TxHierAttribSet& tas);

public:
  bool HasShapeMaterialDataIndex(const Standard_Integer theShapeIndex, Standard_Integer& theMatDataIndex) const;
  bool HasShapeMaterialData(const Standard_Integer theShapeIndex, ISOEMMatData& theData) const; 

  const vector<Standard_Integer>& GetMatDataIndicesOfSpaceDefine() const;
  bool HasMatDataWithMatIndex(const Standard_Integer theIndex, ISOEMMatData& theData) const;
  bool CheckMatDataIndices() const;

private:
  void SetShapeMaterialIndexMap(const TxHierAttribSet& theMatDataTha);
  void SetMaterialDatas(const TxHierAttribSet& theMatDataTha);
  void SetMurPortDatas(const TxHierAttribSet& tas);
  void BuildMatDataIndicesOfSpaceDefine();

public:
  Standard_Real GetEpsAccordingMatIndices(const set<Standard_Integer>& theMatDataIndices, Standard_Integer dir) const;
  Standard_Real GetMuAccordingMatIndices(const set<Standard_Integer>& theMatDataIndices, Standard_Integer dir) const;
  Standard_Real GetSigmaAccordingMatIndices(const set<Standard_Integer>& theMatDataIndices, Standard_Integer dir) const;


public:
  const map<Standard_Integer, Standard_Integer>* GetShapesType() const;
  map<Standard_Integer, Standard_Integer>* ModifyShapesType();

  const map<Standard_Integer, Standard_Integer>* GetFacesType() const;
  map<Standard_Integer, Standard_Integer>* ModifyFacesType();



  const map<Standard_Integer, Standard_Integer>* GetShapesMask() const;
  map<Standard_Integer, Standard_Integer>* ModifyShapesMask();

  const map<Standard_Integer, Standard_Integer>* GetFacesMask() const;
  map<Standard_Integer, Standard_Integer>* ModifyFacesMask();

  const map<Standard_Integer, vector<Standard_Integer> >* GetRelationBetweenVertexAndEdge() const;
  map<Standard_Integer, vector<Standard_Integer> >* ModifyRelationBetweenVertexAndEdge();

  const map<Standard_Integer, vector<Standard_Integer> >* GetRelationBetweenEdgeAndFace() const;
  map<Standard_Integer, vector<Standard_Integer> >* ModifyRelationBetweenEdgeAndFace(); 

  const map<Standard_Integer, Standard_Integer>* GetRelationBetweenFaceAndShape() const;
  map<Standard_Integer, Standard_Integer>* ModifyRelationBetweenFaceAndShape();


public:
  const map<Standard_Size,vector<EdgeBndVertexData>,less<Standard_Size> > * GetEdgeBndVertexDataOf(const ZRGridLineDir aDir) const;
  map<Standard_Size,vector<EdgeBndVertexData>,less<Standard_Size> > * ModifyEdgeBndVertexDataOf(const ZRGridLineDir aDir);

  bool HasEdgeBndVertexDataOf(const ZRGridLineDir aDir, 
				    const Standard_Size aIndex) const;
  const vector<EdgeBndVertexData>& GetEdgeBndVertexDataOf(const ZRGridLineDir aDir,
							  const Standard_Size aIndex) const;
  vector<EdgeBndVertexData>& ModifyEdgeBndVertexDataOf(const ZRGridLineDir aDir,  
						       const Standard_Size aIndex);

  const vector<FaceBndVertexData>*  GetFaceBndVertexData() const;
  vector<FaceBndVertexData> *  ModifyFaceBndVertexData();


public:
  const map<Standard_Integer, PortData, less<Standard_Integer> >* GetPorts() const;
  map<Standard_Integer, PortData, less<Standard_Integer> >* ModifyPorts();
  const PortData* GetPortWithPortIndex(const Standard_Integer thePortIndex) const;
  const map<Standard_Integer, Standard_Real, less<Standard_Integer> >* GetMurPortDatas() const;


  // Get Material Methods
public:
  Standard_Integer GetMaterialTypeWithShapeIndices(const set<Standard_Integer>& theIndices) const;
  Standard_Integer GetMaterialTypeWithShapeIndex(const Standard_Integer theIndex) const;
  Standard_Integer GetMaterialTypeWithFaceIndex(const Standard_Integer theIndex) const;

  Standard_Integer GetBackGroundMaterialType() const;
  Standard_Integer GetBackGroundMaterialDataIndex() const;

  Standard_Integer GetMaterialTypeWithFaceIndices(const set<Standard_Integer>& theIndices) const;

public:
  Standard_Integer GetShapeIndexAccordingFaceIndex(const Standard_Integer theIndex) const;


  // Set Methods
public:
  void SetBackGroundMaterialType(const Standard_Integer aType);
  void SetEdgeBndVertices(const Standard_Integer theDir, 
			  const map<Standard_Size,vector<EdgeBndVertexData>,less<Standard_Size> >& theData);
  void SetFaceBndVertices( const vector<FaceBndVertexData>& theData);

  // Set type and the toplogical relationship
public:
  void SetShapesType(const map<Standard_Integer, Standard_Integer>& theData);
  void SetFacesType(const map<Standard_Integer, Standard_Integer>& theData);
  void SetPorts(const map<Standard_Integer, PortData, less<Standard_Integer> >& theData);

  void SetRelationBetweenFaceAndShape(const map<Standard_Integer, Standard_Integer>& theData);
  void SetRelationBetweenEdgeAndFace(const map<Standard_Integer, vector<Standard_Integer> >& theData);
  void SetRelationBetweenVertexAndEdge(const map<Standard_Integer, vector<Standard_Integer> >& theData);

  void SetShapesMask(const map<Standard_Integer, Standard_Integer>& theData);
  void SetFacesMask(const map<Standard_Integer, Standard_Integer>& theData);



  /********************** tool *****************************/
public:
  void ConvertFaceMaskVectoIndexVec(const vector<Standard_Integer>& maskVec, vector<Standard_Integer>& indexVec) const;
  void ConvertFaceMasktoIndex(const Standard_Integer theMask, Standard_Integer& theIndex) const; 
  void ConvertShapeMasktoIndex(const Standard_Integer theMask, Standard_Integer& theIndex) const;
  void ConvertFaceIndextoMask(const Standard_Integer theIndex, Standard_Integer& theMask) const; 

  bool IsOnePtclBnd(const Standard_Integer theFaceIndex) const;


  /********************** tool *****************************/
public:
  void GetFaceIndices_With_VertexIndex(const Standard_Integer theVertexIndex, set<Standard_Integer>& theFaceIndices) const; 
  void GetFaceIndices_With_EdgeIndex(const Standard_Integer theEdgeIndex, set<Standard_Integer>& theFaceIndices) const; 
  void GetEdgeIndices_With_VertexIndex(const Standard_Integer theVertexIndex, set<Standard_Integer>& theEdgeIndices) const;




  // Data Defines
private:
  map<Standard_Size,vector<EdgeBndVertexData>,less<Standard_Size> > m_EdgeBndVertexData0;
  map<Standard_Size,vector<EdgeBndVertexData>,less<Standard_Size> > m_EdgeBndVertexData1;

  vector<FaceBndVertexData> m_FaceBndVertexData;


  map<Standard_Integer, vector<Standard_Integer> > m_VerticesWithEdgeTool;
  map<Standard_Integer, vector<Standard_Integer> > m_EdgesWithFaceTool; 
  map<Standard_Integer, Standard_Integer> m_FacesWithShapeTool; 


  map<Standard_Integer, Standard_Integer> m_ShapesWithTypeTool;
  map<Standard_Integer, Standard_Integer> m_FacesWithTypeTool;


  map<Standard_Integer, Standard_Integer> m_ShapeMaskWithIndexTool;   // for mask: shape's mask with shape' index
  map<Standard_Integer, Standard_Integer> m_FaceMaskWithIndexTool;   // for mask:  face's mask with face's index 


  map<Standard_Integer, PortData,   less<Standard_Integer> > m_Ports;


  map<Standard_Integer, Standard_Real, less<Standard_Integer> > m_MurPortIndexWithDataMap;

  // Space Data define
  Standard_Integer m_BackGroundMaterialType;
  Standard_Integer m_BackGroundMaterialDataIndex;

  vector<Standard_Integer> m_SpaceMaterialData; 

  // all material defination
  map<Standard_Integer, ISOEMMatData, less<Standard_Integer> > m_MaterialDataIndexWithMaterialDataMap;

  // shape's material defination
  map<Standard_Integer, Standard_Integer, less<Standard_Integer> > m_ShapeWithMaterialDataIndexMap;
};

#endif

