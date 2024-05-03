//-----------------------------------------------------------------------------
// File:        TxMaker.h
// Purpose:     For creating a derived object
//-----------------------------------------------------------------------------

#ifndef TX_MAKER_H
#define TX_MAKER_H

#include "TxMakerBase.h"

/**
 * Create a base class pointer for a new derived type object.
 *
 * The TxMaker class is templated over types B and D, where D is derived
 * from B.  Invoking the getNew() method returns a pointer of type B to 
 * default object of type D.  Thus, class D must have a default constructor.
 *
 *
 * @param B Base class;  the pointer will be of this type.
 * @param D Class derived from B; new object of this type is instantiated.
 *
 */

template <class D, class B> 
class TxMaker : public TxMakerBase<B> 
{
 public:
  /***  Construct a default TxMaker  */
 TxMaker(std::string nm) : TxMakerBase<B>(nm) {}
  
  /*** Destroy this TxMaker  */
  virtual ~TxMaker(){}
  
  /**
   * Get a new default instance of class D
   * @return a pointer to the new instance of D
   */
  virtual B* getNew() {
    D* d = new D;
    return dynamic_cast<B*>(d);
  }
  
};

#endif
