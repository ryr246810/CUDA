// ----------------------------------------------------------------------
// File:	NodeFlds_Iter.hxx
// ----------------------------------------------------------------------

#ifndef _NodeFldsIter_HeaderFile
#define _NodeFldsIter_HeaderFile

// std includes
#include <vector>

#include <NodeFldsBase.hxx>
#include <NodeFlds_ConstIter.hxx>

/**
 * A NodeFlds_Iter is used for iterating over a NodeFldsBase.
 * Its value member contains the 1-D indx for accessing 
 * TxTensor data that is assumed to have the same shape as some
 * grid.  It can be bumped in a direction, which corresponds to
 * increasing the indx in that direction.  
 */

class NodeFlds_Iter : public NodeFlds_ConstIter
{

public:
  NodeFlds_Iter(){ };
  
  NodeFlds_Iter(NodeFldsBase* vf);

  NodeFlds_Iter(const NodeFlds_Iter& vfi);

  ~NodeFlds_Iter(){}

  NodeFlds_Iter& operator=(const NodeFlds_Iter& vfi);

  Standard_Real& operator()() {
    return *this->m_indxPtr;
  }

  Standard_Real& operator()(size_t dir) {
    return *(this->m_indxPtr + this->m_sizes[dir]);
  }
};

#endif
