#ifndef _VertexData_Headerfile
#define _VertexData_Headerfile


#include <TxVector2D.h>
#include <DataBase.hxx>
#include <set>

class VertexData: public DataBase
{
public:
  VertexData();
  VertexData(Standard_Integer _mark);
  ~VertexData();

public:
  virtual void Setup();
  virtual void SetupGeomDimInf();

public:
  bool IsRegular() const;
  bool IsAppendedToGridEdge() const;
  bool IsAppendedToGridFace() const;

public:
  virtual TxVector2D<Standard_Real> GetLocation() const = 0;


public:
  const set<Standard_Integer>& GetShapeIndices()const;
  void SetShapeIndex(Standard_Integer _index);
  void AddShapeIndex(Standard_Integer _index);
  bool HasShapeIndex(Standard_Integer _index) const;
  bool HasAnyShapeIndex() const;
  void ClearShapeIndices();


  bool HasCommonShapeIndexWith(const VertexData* theVertex);
  bool HasCommonShapeIndicesWith(const VertexData* theVertex,
				 set<Standard_Integer>& theCommonIndices);


  const set<Standard_Integer>& GetFaceIndices()const;
  void SetFaceIndex(Standard_Integer _index);
  void AddFaceIndex(Standard_Integer _index);
  void AddFaceIndices(const std::set<Standard_Integer>& _faceindices);

  bool HasFaceIndex(Standard_Integer _index) const;
  void SetFaceIndices(const std::set<Standard_Integer>& _faceindices);
  void ClearFaceIndices();
 

  bool HasSameFaceIndicesWith(const VertexData* theVertex);
  bool HasCommonFaceIndexWith(const VertexData* theVertex);
  bool HasCommonFaceIndexWith(const std::set<Standard_Integer>& theFaceIndices);
  bool BeIncludeFaceIndicesOf(const VertexData* theVertex) const;
  void GetCommonFaceIndicesWith(const VertexData* theVertex,
				set<Standard_Integer>& theCommonFaceIndices);

  void GetCommonFaceIndicesWith(const std::set<Standard_Integer>& theFaceIndices,
				set<Standard_Integer>& theCommonFaceIndices);


  const set<Standard_Integer>& GetEdgeIndices()const;
  void AddEdgeIndex(Standard_Integer _index);
  void AddEdgeIndices(const std::set<Standard_Integer>& _edgeindices);

  bool HasAnyEdgeIndex() const;
  bool HasEdgeIndex(Standard_Integer _index) const;
  void SetEdgeIndex(Standard_Integer _index);
  void SetEdgeIndices(const std::set<Standard_Integer>& _edgeindices);
  void ClearEdgeIndices();
 

  bool HasCommonEdgeIndexWith(VertexData* theVertex);
  bool HasCommonEdgeIndexWith(const std::set<Standard_Integer>& theEdgeIndices);
  void GetCommonEdgeIndicesWith(const VertexData* theVertex,
				set<Standard_Integer>& theCommonEdgeIndices);

  void GetCommonEdgeIndicesWith(const std::set<Standard_Integer>& theEdgeIndices,
				set<Standard_Integer>& theCommonEdgeIndices);


  // operation for material data index
  const set<Standard_Integer>& GetMatDataIndices()const;
  bool HasAnyUserDefinedMatData() const;
  bool HasMatDataIndex(Standard_Integer _index) const;
  void SetMaterialDataIndex(Standard_Integer _index);
  void AppendMatDataIndex(Standard_Integer _index);

  void AppendMatDataIndices(const set<Standard_Integer>& indices);


  void RemoveMatDataIndex(Standard_Integer _index);
  void RemoveMatDataIndices(const vector<Standard_Integer>& _indices);
  void RemoveMatDataIndices(const set<Standard_Integer>& _indices);
  void ClearMatDataIndices();


public:
  void ComputeSweptGeomDim();

  virtual Standard_Real GetSweptGeomDim() const{
    return m_LengthOfSweptEdge;
  };


protected:
  set<Standard_Integer> m_ShapeIndices;
  set<Standard_Integer> m_FaceIndices;
  set<Standard_Integer> m_EdgeIndices;
  set<Standard_Integer> m_MatDataIndices;

  Standard_Real m_LengthOfSweptEdge;
};

#endif
