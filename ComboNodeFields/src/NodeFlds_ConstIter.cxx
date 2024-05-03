// ----------------------------------------------------------------------
// File:	NodeFlds_ConstIter.cxx
// Purpose:	Implementation of a field iterator that is const - i.e., cannot change the values of the field.
// ----------------------------------------------------------------------

#include <NodeFlds_ConstIter.hxx>


NodeFlds_ConstIter::
NodeFlds_ConstIter()
{
  m_fieldConstPtr = 0;
  for(size_t i=0; i<=2; ++i) m_sizes[i] = 0;
  m_indxPtr = NULL;
}


NodeFlds_ConstIter::
NodeFlds_ConstIter(const NodeFlds_ConstIter& cemfi)
{
  m_fieldConstPtr = cemfi.m_fieldConstPtr;
  for(size_t i=0; i<=2; ++i) m_sizes[i] = cemfi.m_sizes[i];
  m_indxPtr = cemfi.m_indxPtr;
}


NodeFlds_ConstIter& 
NodeFlds_ConstIter::
operator=(const NodeFlds_ConstIter& cemfi) 
{
  m_fieldConstPtr = cemfi.m_fieldConstPtr;

  for(size_t i=0; i<=2; ++i) m_sizes[i] = cemfi.m_sizes[i];
  m_indxPtr = cemfi.m_indxPtr;
  return *this;
}


void 
NodeFlds_ConstIter::
setField(const NodeFldsBase* cemf)
{
  m_fieldConstPtr = cemf;
  for(size_t i=0; i<=2; ++i){
    m_sizes[i] = m_fieldConstPtr->GetSize(i);
  }
  m_indxPtr = m_fieldConstPtr->GetDataPtr();
}
