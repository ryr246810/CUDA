#include <AppendingVertexDataOfGridEdge.hxx>
#include <GridVertexData.hxx>

AppendingVertexDataOfGridEdge::AppendingVertexDataOfGridEdge()
  :VertexData()
{
  m_Frac = 0;
  m_BaseEdge = NULL;
  SetFaceIndex(-1);
  SetShapeIndex(-1);
}


AppendingVertexDataOfGridEdge::
AppendingVertexDataOfGridEdge(Standard_Integer _ShapeIndex, 
			      Standard_Integer _FaceIndex, 
			      GridEdge* _baseedge, 
			      Standard_Size _frac, 
			      Standard_Integer _mark, 
			      Standard_Integer _materialtype, 
			      Standard_Integer _transitiontype)
  :VertexData(_mark)
{
  m_BaseEdge = _baseedge;

  SetShapeIndex(_ShapeIndex);
  SetFaceIndex(_FaceIndex);

  m_Frac = _frac;
  SetMaterialType(_materialtype);
  SetTransitionType(_transitiontype);
}


AppendingVertexDataOfGridEdge:: ~AppendingVertexDataOfGridEdge()
{
}


TxVector2D<Standard_Real> AppendingVertexDataOfGridEdge::GetLocation() const
{
  Standard_Integer theResolution = m_BaseEdge->GetResolution();
  Standard_Real thefrac = Standard_Real(m_Frac)/Standard_Real(theResolution);
  TxVector2D<Standard_Real> tmp   = 
    m_BaseEdge->GetFirstVertex()->GetLocation()  + 
    m_BaseEdge->GetVector() * thefrac;
  return tmp;
}


bool  AppendingVertexDataOfGridEdge::HasSameLocation(AppendingVertexDataOfGridEdge* oneData)
{
  bool result = false;
  if(this->GetBaseEdge() == oneData->GetBaseEdge()){
    if(this->GetFrac()==oneData->GetFrac()){
      result = true;
    }
  }
  return result;
}
