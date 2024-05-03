#include <VertexData.hxx>


const set<Standard_Integer>& 
VertexData::
GetEdgeIndices()const
{
  return m_EdgeIndices;
}


void 
VertexData::
AddEdgeIndex(Standard_Integer _index)
{
  if(!HasEdgeIndex(_index)){
    m_EdgeIndices.insert(_index);
  }
}


void 
VertexData::
AddEdgeIndices(const std::set<Standard_Integer>& _edgeindices)
{
  set<Standard_Integer>::iterator FIIter;
  for(FIIter=_edgeindices.begin(); FIIter!=_edgeindices.end(); FIIter++){
    this->AddEdgeIndex(*FIIter);
  }
}


bool 
VertexData::
HasEdgeIndex(Standard_Integer _index) const
{
  bool result = false;
  set<Standard_Integer>::iterator iter = m_EdgeIndices.find(_index);
  if(iter!=m_EdgeIndices.end()) result = true;
  return result;
}




bool 
VertexData::
HasAnyEdgeIndex() const
{
  bool result = false;
  if(!m_EdgeIndices.empty()){
    result = true;
  }
  return result;
}




void 
VertexData::
ClearEdgeIndices()
{
  m_EdgeIndices.clear();
}


void 
VertexData::
SetEdgeIndices(const set<Standard_Integer>& _edgeindices)
{
  m_EdgeIndices = std::set<Standard_Integer>( _edgeindices);
}


void 
VertexData::
SetEdgeIndex(const Standard_Integer _edgeindex)
{
  ClearEdgeIndices();
  m_EdgeIndices.insert(_edgeindex);
}


bool 
VertexData::
HasCommonEdgeIndexWith(VertexData* theOtherVertex)
{
  const std::set<Standard_Integer>& theOtherEdgeIndices = theOtherVertex->GetEdgeIndices();
  return HasCommonEdgeIndexWith(theOtherEdgeIndices);
}


bool 
VertexData::
HasCommonEdgeIndexWith(const std::set<Standard_Integer>& theEdgeIndices)
{
  bool result = false;
  std::set<Standard_Integer>::const_iterator iter;
  
  for(iter = theEdgeIndices.begin(); iter!=theEdgeIndices.end(); iter++){
    Standard_Integer aEdgeIndex = *iter;
    result = HasEdgeIndex(aEdgeIndex);
    if(result) break;
  }
  
  return result;
}


void 
VertexData::
GetCommonEdgeIndicesWith(const std::set<Standard_Integer>& theEdgeIndices,
			 set<Standard_Integer>& theCommonEdgeIndices)
{
  theCommonEdgeIndices.clear();
  std::set<Standard_Integer>::const_iterator iter;

  for(iter = theEdgeIndices.begin(); iter!=theEdgeIndices.end(); iter++){
    Standard_Integer aEdgeIndex = *iter;
    if( HasEdgeIndex(aEdgeIndex)){
      theCommonEdgeIndices.insert(aEdgeIndex);
    }
  }
}


void 
VertexData::
GetCommonEdgeIndicesWith(const VertexData* theOtherVertex,
			 set<Standard_Integer>& theCommonEdgeIndices)
{
  const std::set<Standard_Integer>& theOtherEdgeIndices = theOtherVertex->GetEdgeIndices();
  GetCommonEdgeIndicesWith(theOtherEdgeIndices,theCommonEdgeIndices);
}
