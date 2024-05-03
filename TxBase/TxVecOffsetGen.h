// ----------------------------------------------------------------------
//
// File:	TxOffsetGen.h
//
// Purpose:	Compute offsets into three dimension tensors given indices
//
// ----------------------------------------------------------------------

#ifndef _TxOffsetGen_HeaderFile
#define _TxOffsetGen_HeaderFile

#include <TxSlab.h>
#include <TxVector.h>

class TxOffsetGen 
{
 public:
/*
 * Create a new offset generator, init() must still be called before using an instance.
 */
  TxOffsetGen() {}

/*
 * Initialize this offset generator. Must be done after
 * a call to setLengths
 */
  void init() {
    for(size_t i=0; i<3; ++i) {
      sizes[i] = 1;
      for(size_t j=i+1; j<3; ++j) sizes[i] *= lengths[j];
    }
  }

/*
 * Defines the size of the tensor for which this offsetgen computes offsets
 *
 * @param rgn the region describing the size of the tensor.
 */
  void setLengths(const TxSlab<int>& rgn) {
    for (size_t i=0;i<3;++i)
      lengths[i]=rgn.getUpperBound(i)-rgn.getLowerBound(i);
    init();
  }

/*
 * Defines the size of the tensor for which this offsetgen computes offsets
 * 
 * @param rgn an array describing the size of the tensor
 */
   void setLengths(const size_t* lens) {
    for (size_t i=0;i<3;++i)
      lengths[i]=lens[i];
    init();
  }

/*
 * compute and return an offset given an index vector
 *
 * @param indx the index vector
 * 
 * @return the offset into the tensor where data is stored
 */
  size_t getOffset(const size_t* indx) const {
    size_t offset=0;
    for (size_t i=0;i<3;++i)
      offset=offset+indx[i]*sizes[i];
    return offset;
  }

  size_t getOffset(const TxVector<size_t>& indx) const {
    size_t offset=0;
    for (size_t i=0;i<3;++i)
      offset=offset+indx[i]*sizes[i];
    return offset;
  }

  void getVecIndx(const size_t indx, TxVector<size_t>& indxVec) const {
    Standard_Size rem = indx;
    for(Standard_Integer i=0; i<2; ++i){
      indxVec[i] = (Standard_Size)(rem/cellSizes[i]);
      rem %= cellSizes[i];
    }
    indxVec[2] = rem;
  }

  void getVecIndx(const size_t indx, size_t* indxVec) const {
    Standard_Size rem = indx;
    for(Standard_Integer i=0; i<2; ++i){
      indxVec[i] = (Standard_Size)(rem/cellSizes[i]);
      rem %= cellSizes[i];
    }
    indxVec[2] = rem;
  }


  protected:
/*
 * length of tensor in each direction
 */
    size_t lengths[3];

/*
 * length of unit stride in each direction
 */
    size_t sizes[3];

};

#endif
