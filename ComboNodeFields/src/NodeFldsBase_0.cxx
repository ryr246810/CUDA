#include <NodeFldsBase.hxx>

size_t 
NodeFldsBase::
GetElementNum() const 
{
  return m_Lengths[2];
}


size_t 
NodeFldsBase::
getLength(size_t i) const
{
  return m_Lengths[i];
}


size_t 
NodeFldsBase::
GetSize() const 
{
  return m_DataSize;
}


size_t 
NodeFldsBase::
GetSize(size_t i) const 
{
  return m_Size[i];
}


Standard_Real& 
NodeFldsBase::
operator()(size_t i) 
{
  return *(m_Data+i);
}


Standard_Real 
NodeFldsBase::
operator()(size_t i) const 
{
  return *(m_Data+i);
}


Standard_Real& 
NodeFldsBase::
operator()(size_t i, size_t j, size_t n) 
{
  size_t indx = 
    i*m_Size[0] + 
    j*m_Size[1] + 
    n*m_Size[2];

  return *(m_Data+indx);
}


Standard_Real 
NodeFldsBase::
operator()(size_t i, size_t j, size_t n) const 
{
  size_t indx = 
    i*m_Size[0] + 
    j*m_Size[1] + 
    n*m_Size[2];

  return *(m_Data+indx);
}



void 
NodeFldsBase::
FillWithInterpValue(const vector< TxVector2D<Standard_Real> >& positions,
		    vector< TxVector<Standard_Real> >& values) const
{

}

void 
NodeFldsBase::
FillWithInterpValue(const TxVector2D<Standard_Real>& position,
		    TxVector<Standard_Real>& value) const
{

}

void 
NodeFldsBase::
FillWithInterpValue(const vector< TxVector2D<Standard_Real> >& positions,
		    size_t fldComp,
		    vector< Standard_Real >& values ) const
{

}

void 
NodeFldsBase::
FillWithInterpValue(const IndexAndWeights& indxWt,  
		    TxVector<Standard_Real>& value) const
{

}

void 
NodeFldsBase::
FillWithInterpValue(const Standard_Size& nb,
		    const vector< IndexAndWeights >& indices,
		    vector< TxVector<Standard_Real> >& values) const
{

}

void 
NodeFldsBase::
FillWithInterpValue(const TxVector2D<Standard_Real>& position,
		    Standard_Real& result) const
{

}
  
void 
NodeFldsBase::
FillWithInterpValue(const IndexAndWeights& index,
		    Standard_Real& result) const
{

}

void 
NodeFldsBase::
SetPhysDataIndexInGridGeom(const Standard_Integer _index)
{

}

void 
NodeFldsBase::
SetupDataSetter()
{

}

void 
NodeFldsBase::
Update()
{
  DynObj::Advance();
}
