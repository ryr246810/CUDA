// ----------------------------------------------------------------------
// File:	NodeFlds_Iter.cxx
// ----------------------------------------------------------------------

#include <NodeFlds_Iter.hxx>


NodeFlds_Iter::NodeFlds_Iter(const NodeFlds_Iter& cemfi)
  : NodeFlds_ConstIter(cemfi)
{
}


  
NodeFlds_Iter::NodeFlds_Iter(NodeFldsBase* cemf)
  : NodeFlds_ConstIter(cemf)
{
}


NodeFlds_Iter& NodeFlds_Iter::operator=(const NodeFlds_Iter& cemfi)
{
  NodeFlds_ConstIter::operator=(cemfi);
  return *this;
}

