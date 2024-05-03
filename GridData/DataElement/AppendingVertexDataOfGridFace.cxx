#include <AppendingVertexDataOfGridFace.hxx>
#include <GridVertexData.hxx>


AppendingVertexDataOfGridFace::AppendingVertexDataOfGridFace()
  :VertexData()
{
  SetShapeIndex(0);
  m_Frac1 = 0;
  m_Frac2 = 0;
  m_BaseFace = NULL;
}


AppendingVertexDataOfGridFace::AppendingVertexDataOfGridFace(Standard_Integer _shapeindex,
							     Standard_Integer _edgeindex,
							     const set<Standard_Integer>& _faceindices,
							     GridFace* _baseface,
							     Standard_Size _frac1,
							     Standard_Size _frac2,
							     Standard_Integer _mark,
							     Standard_Integer _materialtype)
  :VertexData(_mark)
{
  m_Frac1 = _frac1;
  m_Frac2 = _frac2;
  m_BaseFace = _baseface;

  SetMaterialType(_materialtype);
  SetShapeIndex(_shapeindex);
  SetEdgeIndex(_edgeindex);
  SetFaceIndices(_faceindices);
}


AppendingVertexDataOfGridFace:: ~AppendingVertexDataOfGridFace()
{
}


TxVector2D<Standard_Real> AppendingVertexDataOfGridFace::GetLocation() const
{
  Standard_Integer theResolution = m_BaseFace->GetResolution();
  Standard_Real thefrac1 = Standard_Real(m_Frac1)/Standard_Real(theResolution);
  Standard_Real thefrac2 = Standard_Real(m_Frac2)/Standard_Real(theResolution);

  TxVector2D<Standard_Real> tmp  =  
    m_BaseFace->GetLDVertex()->GetLocation()  +  
    m_BaseFace->GetVectorOfDir1()*thefrac1 +
    m_BaseFace->GetVectorOfDir2()*thefrac2;

  return tmp;
}


bool AppendingVertexDataOfGridFace::IsSameLocation(const AppendingVertexDataOfGridFace* one)
{
  bool result = false;
  if( (this->GetBaseFace() == one->GetBaseFace()) && 
      (this->GetFrac1() == one->GetFrac1()) && 
      (this->GetFrac2() == one->GetFrac2()) ){
    result == true;
  }
  return result;
}
