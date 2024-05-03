#include <AppendingEdgeData.hxx>



void 
AppendingEdgeData::
ClearFaceIndices()
{
  m_FaceIndices.clear();
}


const set<Standard_Integer>& 
AppendingEdgeData::
GetFaceIndices()const
{
  return m_FaceIndices;
}


void 
AppendingEdgeData::
AddFaceIndex(Standard_Integer _index)
{
  if(!HasFaceIndex(_index)){
    m_FaceIndices.insert(_index);
  }
}


void 
AppendingEdgeData::
AddFaceIndices(const std::set<Standard_Integer>& _faceindices)
{
  set<Standard_Integer>::iterator FIIter;
  for(FIIter=_faceindices.begin(); FIIter!=_faceindices.end(); FIIter++){
    this->AddFaceIndex(*FIIter);
  }
}


void 
AppendingEdgeData::
SetFaceIndices(const set<Standard_Integer>& _faceindices)
{
  m_FaceIndices = std::set<Standard_Integer>( _faceindices);
}


bool 
AppendingEdgeData::
BeIncludeFaceIndicesOf(const AppendingEdgeData* theEdge) const
{
  bool result = true;

  const std::set<Standard_Integer>& theFaceIndices = theEdge->GetFaceIndices();
  std::set<Standard_Integer>::iterator iter;

  for(iter = theFaceIndices.begin(); iter!=theFaceIndices.end(); iter++){
    Standard_Integer aFaceIndex = *iter;
    bool tmpResult = HasFaceIndex(aFaceIndex);
    result = tmpResult && result;
  }

  return result;
}


bool 
AppendingEdgeData::
HasFaceIndex(Standard_Integer _index) const
{
  bool result = false;
  set<Standard_Integer>::iterator iter = m_FaceIndices.find(_index);
  if(iter!=m_FaceIndices.end()) result = true;
  return result;
}


bool 
AppendingEdgeData::
HasCommonFaceIndexWith(const std::set<Standard_Integer>& theFaceIndices)
{
  bool result = false;
  std::set<Standard_Integer>::const_iterator iter;
  
  for(iter = theFaceIndices.begin(); iter!=theFaceIndices.end(); iter++){
    Standard_Integer aFaceIndex = *iter;
    result = HasFaceIndex(aFaceIndex);
    if(result) break;
  }
  
  return result;
}


void 
AppendingEdgeData::
GetCommonFaceIndicesWith(const std::set<Standard_Integer>& theFaceIndices,
			 set<Standard_Integer>& theCommonFaceIndices)
{
  theCommonFaceIndices.clear();
  std::set<Standard_Integer>::const_iterator iter;

  for(iter = theFaceIndices.begin(); iter!=theFaceIndices.end(); iter++){
    Standard_Integer aFaceIndex = *iter;
    if( HasFaceIndex(aFaceIndex)){
      theCommonFaceIndices.insert(aFaceIndex);
    }
  }
}


void 
AppendingEdgeData::
GetCommonFaceIndicesWith(const AppendingEdgeData* theEdge,
			 set<Standard_Integer>& theCommonFaceIndices)
{
  const std::set<Standard_Integer>& theOtherFaceIndices = theEdge->GetFaceIndices();
  GetCommonFaceIndicesWith(theOtherFaceIndices,theCommonFaceIndices);
}


bool 
AppendingEdgeData::
HasSameFaceIndicesWith(AppendingEdgeData* theEdge)
{
  bool result = true;

  const std::set<Standard_Integer>& theFaceIndices = theEdge->GetFaceIndices();
  std::set<Standard_Integer>::iterator iter;

  for(iter = theFaceIndices.begin(); iter!=theFaceIndices.end(); iter++){
    if(!HasFaceIndex(*iter)){
      result = false;
      break;
    }
  }

  return result;
}


