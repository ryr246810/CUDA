#include <NodeFlds_IterHolder.hxx>

NodeFlds_IterHolder::NodeFlds_IterHolder(NodeFldsBase* cemf)
  : m_rsltIter(cemf)
{
}


NodeFlds_IterHolder::NodeFlds_IterHolder(const NodeFlds_IterHolder& cemfi)
  : m_rsltIter(cemfi.m_rsltIter)
{
  m_rgn = cemfi.m_rgn;
}


NodeFlds_IterHolder& NodeFlds_IterHolder::operator=(const NodeFlds_IterHolder& cemfi)
{
  m_rsltIter = cemfi.m_rsltIter;
  m_rgn = cemfi.m_rgn;
  return *this;
}
