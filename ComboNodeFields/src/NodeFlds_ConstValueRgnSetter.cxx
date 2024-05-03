#include <NodeFlds_ConstValueRgnSetter.hxx>


NodeFlds_ConstValueRgnSetter::
NodeFlds_ConstValueRgnSetter(NodeFldsBase* vf) : NodeFlds_IterHolder(vf)
{
}


NodeFlds_ConstValueRgnSetter::
NodeFlds_ConstValueRgnSetter(const NodeFlds_ConstValueRgnSetter& vfs) : NodeFlds_IterHolder(vfs)
{
}


NodeFlds_ConstValueRgnSetter& 
NodeFlds_ConstValueRgnSetter::
operator=(const NodeFlds_ConstValueRgnSetter& vfs)
{
  NodeFlds_IterHolder::operator=(vfs);
  return *this;
}
