#ifndef _NodeFldsConstIter_HeaderFile
#define _NodeFldsConstIter_HeaderFile

// std includes
#include <vector>

// TxBase includes
#include <TxVector2D.h>
#include <TxVector.h>
#include <TxSlab2D.h>

#include <ZRGrid.hxx>
#include <NodeFldsBase.hxx>

/**
 * A NodeFlds_ConstIter is used for iterating over a NodeFldsBase.
 * Its value member contains the 1-D indx for accessing 
 * TxTensor data that is assumed to have the same shape as some
 * grid.  It can be bumped in a direction, which corresponds to
 * increasing the indx in that direction.
 *
 * NodeFlds_ConstIter should not change the value of the field.
 */


class NodeFlds_ConstIter
{
public:
  NodeFlds_ConstIter();
  
  NodeFlds_ConstIter(const NodeFldsBase* vpf) {
    setField(vpf);
  }

  NodeFlds_ConstIter(const NodeFlds_ConstIter& vpfi);

  ~NodeFlds_ConstIter(){};
  
  NodeFlds_ConstIter& operator=(const NodeFlds_ConstIter& vpfi);


  /*** sets the associated field.  */
  void setField(const NodeFldsBase* vpf);
  
  
  /*** Reset to the origin of the extended region at first field. */
  void ptrReset();


  /**
   * Bump the indx for direction dir
    * @param dir the direction to reset
   */
  void bump(size_t dir);


  /**
   * Bump the indx for direction dir
    * @param dir the direction to reset
   */
  void iBump(size_t dir);


  /**
   * Bump the indx for direction dir by some amount
   *
   * @param dir the direction to bump
   * @param amt the amount to bump by (may be negative)
   */
  void bump(size_t dir, int amt);


  /**
   * iBump the indx for direction dir by some amount
   *
   * @param dir the direction to bump
   * @param amt the amount to bump by (may be negative)
   */
  void iBump(size_t dir, int amt);



  GridGeometry* GetGridGeom() const{
    return m_fieldConstPtr->GetGridGeom();
  }


  const ZRGrid* GetZRGrid() const{
    return  m_fieldConstPtr->GetZRGrid();
  };


  size_t getGeomIndex() const;


  /**
   * Get the current indx
   * @return the current value of the indx
   */
  size_t getIndex() const;


  /**
   * Get the current index vector in an argument
   * @return the current value of the indx
   */
  void fillXtndIndexVec(size_t iVec[2]) const;
  /**
   * Set the indices from a position and get the weights for each direction.
   *
   * @param pos the position.
   * @param wl the weights for the lower values
   * @param wu the weights for the upper values
   */
  void setFromPosition(const TxVector2D<Standard_Real>& pos, Standard_Real wl[2], Standard_Real wu[2]);

  /**
   * set the indices to the given values
   * @param indx the indices to set the current iterator to
   */
  void setFromIndx(const size_t indx[2]);

  
  /**
   * Get the number of elements in the field
   * @return the current value of the indx
   */
  Standard_Integer getNumElements() const;


  /**
   * Get the value for the current indx
   * @return the current value of the indx
   */
  Standard_Real operator()() const {
    return *m_indxPtr;
  }


  /**
   * Get the value for the indx one unit in the given directon.
   * Do not actually move the iterator. 
   *
   * @param dir The direction to get the lvalue for.
   *
   * @return the lvalue of the indx offset in the given direction
   */

  Standard_Real operator()(size_t dir) const {
    return *(m_indxPtr+m_sizes[dir]);
  }

 
  /**
   * Get the index pointer
   *
   * @return the address of the current location in the field.
   */
  Standard_Real* getPtr() const {
    return m_indxPtr;
  }
  

  /**
   * Set the index pointer
   *
   * @param p the address to set as the current location in the field.
   */
  void setPtr(Standard_Real* p) {
    m_indxPtr=p;
  }
  

  void setIndx(size_t indx) {
    m_indxPtr = m_fieldConstPtr->GetDataPtr() + indx;
  }


protected:
  /*** The associated field   */
  const NodeFldsBase* m_fieldConstPtr;

  /*** A pointer to the current indx  */
  Standard_Real *m_indxPtr;

  /*** The size of an increment in each direction  */
  size_t m_sizes[3];
};

#endif
