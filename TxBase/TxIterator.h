// ----------------------------------------------------------------------
//
// File:	TxIterator.h
//
// Purpose:	Generic Iterator class
// ----------------------------------------------------------------------

#ifndef _TxIterator_HeaderFile
#define _TxIterator_HeaderFile

#include <TxOffsetGen.h>


class TxIterator : public TxOffsetGen {
 public:
  TxIterator() {
    currentOffset=0;
  }

  void bump(size_t dir) {
    bump(dir, 1);
  }

  void iBump(size_t dir) {
    iBump(dir, 1);
  }

  void bump(size_t dir, int amt) {
    currentOffset += this->sizes[dir]*amt;
  }

  void iBump(size_t dir, int amt) {
    currentOffset -= this->sizes[dir]*amt;
  }

  size_t getOffset() const {
    return currentOffset;
  }
  
  void reset() {
    currentOffset = 0;
  }

 protected:
  size_t currentOffset;
};

#endif
