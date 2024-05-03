// ----------------------------------------------------------------------
// File:	NodeFlds_IterHolder.hxx
// ----------------------------------------------------------------------

#ifndef _NodeFldsIterHolder_HeaderFile
#define _NodeFldsIterHolder_HeaderFile

// std includes
#include <vector>


// nodefield
#include <NodeFlds_Iter.hxx>


class NodeFldsBase;


class NodeFlds_IterHolder
{
public:
  /** * Default Constructor needed for MetroWerks   */
  NodeFlds_IterHolder(){}

  NodeFlds_IterHolder(NodeFldsBase* cemf);

  NodeFlds_IterHolder(const NodeFlds_IterHolder& cemfi);

  virtual ~NodeFlds_IterHolder(){}

  NodeFlds_IterHolder& operator=(const NodeFlds_IterHolder& cemfi);

  void SetRegion(const TxSlab2D<int>& r)
  {
    m_rgn = r;
  }
  
  /**
   * Bump the index for direction dir
   * @param dir the direction to reset
   */
  inline void bump(size_t dir)
  {
    m_rsltIter.bump(dir);
  }

  inline void iBump(size_t dir)
  {
    m_rsltIter.iBump(dir);
  }
  
  /**
   * Bump the index for direction dir by some amount. 
   *
   * @param dir the direction to bump
   * @param amt the amount to bump by (may be negative)
   */
  inline void bump(size_t dir, int amt)
  {
    m_rsltIter.bump(dir, amt);
  }

  inline void iBump(size_t dir, int amt)
  {
    m_rsltIter.iBump(dir, amt);
  }

  inline void ptrReset()
  {
    m_rsltIter.ptrReset();
  }

public:
  /** The region over which to transfer data */
  TxSlab2D<int> m_rgn;
  
  /** The iterator of the component being calculated */
  NodeFlds_Iter m_rsltIter;
};

#endif
