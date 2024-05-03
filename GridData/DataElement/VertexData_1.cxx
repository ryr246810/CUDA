#include <VertexData.hxx>



const set<Standard_Integer>& 
VertexData::
GetMatDataIndices()const
{
  return m_MatDataIndices;
}



void 
VertexData::
AppendMatDataIndex(Standard_Integer _index)
{
  if(!HasMatDataIndex(_index)){
    m_MatDataIndices.insert(_index);
  }
}



void 
VertexData::
AppendMatDataIndices(const set<Standard_Integer>& indices)
{
  set<Standard_Integer>::const_iterator iter;
  for(iter = indices.begin(); iter!=indices.end(); iter++){
    Standard_Integer currIndex = *iter;
    AppendMatDataIndex(currIndex);
  }
}



void 
VertexData::
RemoveMatDataIndex(Standard_Integer _index)
{
  set<Standard_Integer>::iterator iter = m_MatDataIndices.find(_index);
  if(iter!=m_MatDataIndices.end()){
    m_MatDataIndices.erase(iter);
  }
}



void 
VertexData::
RemoveMatDataIndices(const vector<Standard_Integer>& _indices)
{
  vector<Standard_Integer>::const_iterator iter;

  for(iter = _indices.begin(); iter!=_indices.end(); iter++){
    Standard_Integer currIndex = *iter;
    RemoveMatDataIndex(currIndex);
  }
}



void 
VertexData::
RemoveMatDataIndices(const set<Standard_Integer>& _indices)
{
  set<Standard_Integer>::const_iterator iter;

  for(iter = _indices.begin(); iter!=_indices.end(); iter++){
    Standard_Integer currIndex = *iter;
    RemoveMatDataIndex(currIndex);
  }
}



bool 
VertexData::
HasMatDataIndex(Standard_Integer _index) const
{
  bool result = false;
  set<Standard_Integer>::iterator iter = m_MatDataIndices.find(_index);
  if(iter!=m_MatDataIndices.end()) result = true;
  return result;
}



void 
VertexData::
ClearMatDataIndices()
{
  m_MatDataIndices.clear();
}



void 
VertexData::
SetMaterialDataIndex(Standard_Integer _index)
{
  ClearMatDataIndices();
  m_MatDataIndices.insert(_index);
}
