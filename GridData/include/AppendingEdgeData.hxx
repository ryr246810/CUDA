#ifndef _AppendingEdgeData_Headerfile
#define _AppendingEdgeData_Headerfile

#include <EdgeData.hxx>


class GridFaceData;

class AppendingEdgeData: public EdgeData
{
public:
  AppendingEdgeData();
  AppendingEdgeData(Standard_Integer _mark);
  AppendingEdgeData(Standard_Integer _mark, VertexData* _firstV, VertexData* _lastV);
  
  // rzp 2019, 04
  //AppendingEdgeData(Standard_Integer _mark, VertexData* _firstV, VertexData* _lastV, GridFaceData * base_face);
  
  virtual ~AppendingEdgeData();


public:
  virtual void Setup();


public:
  void DeduceFaceIndices();
  void ClearFaceIndices();

  const set<Standard_Integer>& GetFaceIndices()const;
  void AddFaceIndex(Standard_Integer _index);
  void AddFaceIndices(const std::set<Standard_Integer>& _faceindices);

  void SetFaceIndices(const std::set<Standard_Integer>& _faceindices);

  bool BeIncludeFaceIndicesOf(const AppendingEdgeData* theEdge) const;


  bool HasFaceIndex(Standard_Integer _index) const;
  bool HasSameFaceIndicesWith(AppendingEdgeData* theEdge);
  bool HasCommonFaceIndexWith(const std::set<Standard_Integer>& theFaceIndices);


  void GetCommonFaceIndicesWith(const AppendingEdgeData* theEdge,
				set<Standard_Integer>& theCommonFaceIndices);
  void GetCommonFaceIndicesWith(const std::set<Standard_Integer>& theFaceIndices,
				set<Standard_Integer>& theCommonFaceIndices);

protected:
  set<Standard_Integer> m_FaceIndices;
  //GridFaceData * face_data;// rzp 2019,04
  
};

#endif
