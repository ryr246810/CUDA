#include <VertexData.hxx>



const set<Standard_Integer>& 
VertexData::
GetShapeIndices()const
{
  return m_ShapeIndices;
}


void 
VertexData::
AddShapeIndex(Standard_Integer _index)
{
  if(!HasShapeIndex(_index)){
    m_ShapeIndices.insert(_index);
  }
}


void 
VertexData::
SetShapeIndex(Standard_Integer _index)
{
  ClearShapeIndices();
  m_ShapeIndices.insert(_index);
}


// add 2015.12.24
bool 
VertexData::
HasAnyShapeIndex() const
{
  bool result = true;
  if(m_ShapeIndices.empty()){
    result = false;
  }
  return result;
}


bool 
VertexData::
HasShapeIndex(Standard_Integer _index) const
{
  bool result = false;
  set<Standard_Integer>::iterator iter = m_ShapeIndices.find(_index);
  if(iter!=m_ShapeIndices.end()) result = true;
  return result;
}


void 
VertexData::
ClearShapeIndices()
{
  m_ShapeIndices.clear();
}


// added 2015.12.24
bool 
VertexData::
HasCommonShapeIndexWith(const VertexData* theVertex)
{
  bool result = false;
  const std::set<Standard_Integer>& theIndices = theVertex->GetShapeIndices();

  std::set<Standard_Integer>::iterator iter;
  for(iter = theIndices.begin(); iter!=theIndices.end(); iter++){
    if(HasShapeIndex(*iter)){
      result = true;
      break;
    }
  }
  return result;
}


// added 2015.12.24
bool 
VertexData::
HasCommonShapeIndicesWith(const VertexData* theVertex,
			  set<Standard_Integer>& theCommonIndices)
{
  bool result = true;
  theCommonIndices.clear();
  const std::set<Standard_Integer>& theOtherIndices = theVertex->GetShapeIndices();

  std::set<Standard_Integer>::const_iterator iter;
  for(iter =theOtherIndices.begin(); iter!=theOtherIndices.end(); iter++){
    Standard_Integer aIndex = *iter;
    if( HasShapeIndex(aIndex)){
      theCommonIndices.insert(aIndex);
    }
  }

  if(theCommonIndices.empty()){
    result = false;
  }

  return result;
}











const set<Standard_Integer>& 
VertexData::
GetFaceIndices()const
{
  return m_FaceIndices;
}


void 
VertexData::
SetFaceIndex(Standard_Integer _faceindex)
{ 
  ClearFaceIndices();
  AddFaceIndex(_faceindex);
}



void 
VertexData::
AddFaceIndex(Standard_Integer _index)
{
  if(!HasFaceIndex(_index)){
    //cout<<"VertexData::AddFaceIndex----------"<<_index<<endl;
    m_FaceIndices.insert(_index);
  }
}


void 
VertexData::
AddFaceIndices(const std::set<Standard_Integer>& _faceindices)
{
  set<Standard_Integer>::iterator FIIter;
  for(FIIter=_faceindices.begin(); FIIter!=_faceindices.end(); FIIter++){
    this->AddFaceIndex(*FIIter);
  }
}


bool 
VertexData::
HasFaceIndex(Standard_Integer _index) const
{
  bool result = false;
  set<Standard_Integer>::iterator iter = m_FaceIndices.find(_index);
  if(iter!=m_FaceIndices.end()) result = true;
  return result;
}


void 
VertexData::
ClearFaceIndices()
{
  //cout<<"VertexData::ClearFaceIndices()--------------------->>>"<<endl;
  m_FaceIndices.clear();
}


void 
VertexData::
SetFaceIndices(const set<Standard_Integer>& _faceindices)
{
  ClearFaceIndices();
  m_FaceIndices = std::set<Standard_Integer>( _faceindices);
}




bool 
VertexData::
HasCommonFaceIndexWith(const VertexData* theVertex)
{
  bool result = false;

  const std::set<Standard_Integer>& theFaceIndices = theVertex->GetFaceIndices();

  std::set<Standard_Integer>::const_iterator iter;
  
  for(iter = theFaceIndices.begin(); iter!=theFaceIndices.end(); iter++){
    Standard_Integer aFaceIndex = *iter;
    result = HasFaceIndex(aFaceIndex);
    if(result) break;
  }
  
  return result;
}




bool 
VertexData::
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
VertexData::
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


bool 
VertexData::
BeIncludeFaceIndicesOf(const VertexData* theVertex) const
{
  bool result = true;

  const std::set<Standard_Integer>& theFaceIndices = theVertex->GetFaceIndices();
  std::set<Standard_Integer>::iterator iter;

  for(iter = theFaceIndices.begin(); iter!=theFaceIndices.end(); iter++){
    Standard_Integer aFaceIndex = *iter;
    bool tmpResult = HasFaceIndex(aFaceIndex);
    result = tmpResult && result;
  }

  return result;
}


void 
VertexData::
GetCommonFaceIndicesWith(const VertexData* theOtherVertex,
			 set<Standard_Integer>& theCommonFaceIndices)
{
  const std::set<Standard_Integer>& theOtherFaceIndices = theOtherVertex->GetFaceIndices();
  GetCommonFaceIndicesWith(theOtherFaceIndices,theCommonFaceIndices);
}


bool 
VertexData::
HasSameFaceIndicesWith(const VertexData* theVertex)
{
  bool result = true;

  const std::set<Standard_Integer>& theFaceIndices = theVertex->GetFaceIndices();
  std::set<Standard_Integer>::iterator iter;

  for(iter = theFaceIndices.begin(); iter!=theFaceIndices.end(); iter++){
    if(!HasFaceIndex(*iter)){
      result = false;
      break;
    }
  }

  return result;
}




bool 
VertexData::
HasAnyUserDefinedMatData() const
{
  bool result = false;
  if(!(m_MatDataIndices.empty())){
    result = true;
  }

  return result;
}

