#include <AppendingEdgeData.hxx>



AppendingEdgeData::
AppendingEdgeData()
  : EdgeData()
{

}


AppendingEdgeData::
AppendingEdgeData(Standard_Integer _mark)
  : EdgeData(_mark)
{

}


AppendingEdgeData::
AppendingEdgeData(Standard_Integer _mark, VertexData* _firstV, VertexData* _lastV)
  : EdgeData(_mark)
{
  SetVertices(_firstV, _lastV);
}

// AppendingEdgeData::
// AppendingEdgeData(Standard_Integer _mark, VertexData* _firstV, VertexData* _lastV, GridFaceData * base_face)
 // : EdgeData(_mark)
// {
  // SetVertices(_firstV, _lastV);
  // face_data = base_face;
// }

AppendingEdgeData::
~AppendingEdgeData()
{
  m_FaceIndices.clear();
}


void 
AppendingEdgeData::
Setup()
{
  EdgeData::SetupGeomDimInf();
  DeduceFaceIndices();
}


void 
AppendingEdgeData::
DeduceFaceIndices()
{
  m_FaceIndices.clear();

  set<Standard_Integer> theCommonFaceIndices;
  m_FirstVertex->GetCommonFaceIndicesWith(m_LastVertex, theCommonFaceIndices);

  if(!theCommonFaceIndices.empty()){
    this->SetFaceIndices(theCommonFaceIndices);
  }else{
    // modified 2017.03.30, must be CHECKED---------------->>>
    this->ClearFaceIndices();
    /*
    const set<Standard_Integer>& theFirstFaceIndices  = m_FirstVertex->GetFaceIndices();
    const set<Standard_Integer>& theLastFaceIndices  = m_LastVertex->GetFaceIndices();
    this->AddFaceIndices(theFirstFaceIndices);
    this->AddFaceIndices(theLastFaceIndices);
    //*/
  }
}
