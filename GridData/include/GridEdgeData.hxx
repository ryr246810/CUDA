#ifndef _GridEdgeData_Headerfile
#define _GridEdgeData_Headerfile

#include <EdgeData.hxx>
#include <T_Element.hxx>

#include <set>

using namespace std;

class GridEdge;
class AppendingVertexDataOfGridEdge;
class GridEdgeData;
class GridVertexData;

class GridFaceData;


class GridEdgeData: public EdgeData
{
public:
  GridEdgeData();
  GridEdgeData(GridEdge* _gridedge);
  GridEdgeData(Standard_Integer _mark);
  GridEdgeData(GridEdge* _gridedge, Standard_Integer _mark);
  ~GridEdgeData();

  void SetBaseGridEdge(GridEdge* _gridedge);
  void SetVertexVec(const vector<VertexData*>& oneEdgeVertices);

public:
  virtual void Setup();

  virtual void SetupGeomDimInf();
  virtual void SetupMaterialData();

public:
  void InitEfficientLength();
  void ComputeEfficientLength();
  void ComputeDualLengthOfSweptFace();


  /******************* Tool ***********************/
public:
  GridEdge*        GetBaseGridEdge();
  virtual Standard_Integer GetDir();

public:
  bool HasShapeIndex(Standard_Integer _index) const;
  const set<Standard_Integer>&  GetShapeIndices() const;


  /******************ShapeIndices Tool**************/
private:
  void ResetShapeIndices();

  void AddShapeIndex(Standard_Integer _index);
  void AddShapeIndex(VertexData* firstV, VertexData* secondV);
  void AddShapeIndex(VertexData* V);



public:
  bool IsNotPartial();
  bool IsPartial();

  Standard_Real EdgeLengthRatio();


public:
  void SetLocalIndex(const Standard_Integer theIndex){m_LocalIndex = theIndex;};
  Standard_Integer GetLocalIndex() const {return m_LocalIndex;};



private:
  void DeduceShapeIndices();
  void DeduceMaterialType();

  void DeduceGeomState(); // INSHAPE , OUTSHAOE or BND 
  void DeduceGeomType();  // PF(partial filled) edge or regular edge
  
  void DeduceMaterialData();

  void LoadMaterialDataOfSegment(VertexData* firstVertex, 
				 VertexData* lastVertex, 
				 Standard_Real& theEps, 
				 Standard_Real& theMu, 
				 Standard_Real& theSigma);


public:
  bool IsSharedFacesPhysDataDefined();
  bool IsOutLineDEdgePhysDataDefined();

  void AddFace(GridFaceData*, Standard_Integer);
  void AddDEdge(GridVertexData* aDEdge, Standard_Integer aDir);

  const vector<T_Element>& GetSharedTFace() const {return m_TFaces;};
  const vector<T_Element>& GetOutLineDTEdges() const{ return m_DTEdges;};

  void AddNearEEdge(EdgeData* aDEdge, Standard_Integer);
  void AddNearMEdge(EdgeData* aDEdge, Standard_Integer aDir);

  const vector<T_Element>& GetNearEEdges() const {return m_NearEEdges;};
  const vector<T_Element>& GetNearMEdges() const{ return m_NearMEdges;};

public:
  Standard_Real GetLength() const {
    return m_Length;
  }

  virtual Standard_Real GetGeomDim() const {
    return m_EfficientLength;
  }

  virtual Standard_Real GetDualGeomDim() const {
    return m_DualArea;
  }

  virtual Standard_Real GetBaseGridSweptGeomDim() const{
      return m_BaseGridSweptArea;
  }
  virtual Standard_Real GetSweptGeomDim() const;

  virtual Standard_Real GetDualSweptGeomDim() const{
    return m_DualLengthOfSweptFace;
  }
  virtual Standard_Real GetSweptGeomDim_Near();
  virtual Standard_Real GetDualGeomDim_Near();

  //*
public:
  bool HasMidBndVertex(const VertexData* _bndVertex) const;
private:
  vector<VertexData*> m_MidVertices;
  //*/

private:
  Standard_Real m_EfficientLength;
  Standard_Real m_DualArea;

  Standard_Real m_DualLengthOfSweptFace;


private:
  /*
   * degenerative topological edges, one degenerative edge is a GridVertexData
   * m_DTEdges are used to advance magnetic field defined on GridEdgeData 
  //*/
  vector<T_Element> m_DTEdges;
  /*
   * m_TFaces are used to advance electric field defined on GridEdgeData 
  //*/
  vector<T_Element> m_TFaces; 
  /*
   * degenerative topological edges, one degenerative edge is a GridVertexData
   * m_DTEdges are used to advance electric field defined on GridEdgeData 
  //*/
  vector<T_Element> m_NearEEdges;
  /*
   * m_TFaces are used to advance magnecit field defined on GridEdgeData 
  //*/
  vector<T_Element> m_NearMEdges; 


private:
  GridEdge* m_BaseGEdge;   // used for compute dual face area;
  Standard_Integer m_LocalIndex;
  set<Standard_Integer> m_ShapeIndices;
  
  Standard_Real m_BaseGridSweptArea;
  
public:
  Standard_Real m_C[3];
  //Standard_Real m_C2;
};


#endif

