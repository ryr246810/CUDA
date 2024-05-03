// ----------------------------------------------------------------------
// File:	NodeFlds_ConstIter.cxx
// Purpose:	Implementation of a field iterator that is const - i.e., cannot change the values of the field.
// ----------------------------------------------------------------------

#include <NodeFlds_ConstIter.hxx>


  
/**
 * Get the number of elements in the field
 * @return the current value of the indx
 */
Standard_Integer
NodeFlds_ConstIter::
getNumElements() const 
{
  return m_fieldConstPtr->getLength(2);
}


/**
 * Get the current indx
 * @return the current value of the indx
 */
size_t 
NodeFlds_ConstIter::
getIndex() const 
{
  return m_indxPtr - m_fieldConstPtr->GetDataPtr();
}


size_t 
NodeFlds_ConstIter::
getGeomIndex() const 
{
  size_t result = getIndex()/getNumElements();
  return result;
}


/**
 * Get the current index vector in an argument
 * @return the current value of the indx
 */
void 
NodeFlds_ConstIter::
fillXtndIndexVec(size_t iVec[2]) const
{
  size_t rem = getIndex();
  for(int i=0; i<2; ++i){
    iVec[i] =(size_t)(rem/m_sizes[i]);
    rem %= m_sizes[i];
  }
}


/**
 * Set the indices from a position and get the weights for each direction.
 *
 * @param pos the position.
 * @param wl the weights for the lower values
 * @param wu the weights for the upper values
 */
void 
NodeFlds_ConstIter::
setFromPosition(const TxVector2D<Standard_Real>& pos,
		Standard_Real wl[2], 
		Standard_Real wu[2])
{
  // Index vector
  size_t indxVec[2];
  // Get weights and index vector
  GetZRGrid()->ComputeIndexVecAndWeightsInGrid(pos, indxVec, wl, wu);
  
  // Set the iterator
  setFromIndx(indxVec); 
}


/**
 * set the indices to the given values
 * @param indx the indices to set the current iterator to
 */
void
NodeFlds_ConstIter::
setFromIndx(const size_t indx[2])
{
  ptrReset();
  for(size_t i=0; i<2; ++i){
    m_indxPtr += m_sizes[i]*indx[i];
  }
}


/*** Reset to the origin of the extended region at first field. */
void 
NodeFlds_ConstIter::
ptrReset()
{
  m_indxPtr = m_fieldConstPtr->GetDataPtr();
}


/**
 * Bump the indx for direction dir
 * @param dir the direction to reset
 */
void 
NodeFlds_ConstIter::
bump(size_t dir)
{
  bump(dir, 1);
}


/**
 * Bump the indx for direction dir
 * @param dir the direction to reset
 */
void 
NodeFlds_ConstIter::
iBump(size_t dir)
{
  iBump(dir, 1);
}


/**
 * Bump the indx for direction dir by some amount
 *
 * @param dir the direction to bump
 * @param amt the amount to bump by (may be negative)
 */
void 
NodeFlds_ConstIter::
bump(size_t dir, int amt)
{
  m_indxPtr += amt*m_sizes[dir];
}


/**
 * iBump the indx for direction dir by some amount
 *
 * @param dir the direction to bump
 * @param amt the amount to bump by (may be negative)
 */
void 
NodeFlds_ConstIter::
iBump(size_t dir, int amt)
{
  m_indxPtr -= amt*m_sizes[dir];
}

