//-----------------------------------------------------------------------------
// File:        TxMakerBase.h
//
// Purpose:     Interface for a constructor of a given type
//-----------------------------------------------------------------------------

#ifndef TX_MAKER_BASE_H
#define TX_MAKER_BASE_H

#include <TxMakerMapBase.h>

/**
 * TxMakerBase is a base class 
 *  for creating a default object of a type derived from the template parameter. 
 */

template <class B>
class TxMakerBase 
{
 public:
  /*** Construct a default instance */
  TxMakerBase(std::string nm) {
    name = nm;
    TxMakerMapBase< B >::addMaker(name, this);
  }
  /*** Destroy the maker */
  virtual ~TxMakerBase(){
    TxMakerMapBase< B >::rmMaker(name);
  }
  
  /**
   * Create a new object derived from the template type
   * @return a pointer to the new object
   */
  virtual B* getNew()=0;
  
 private:
  /*** Name of this maker */
  std::string name;
  
  /*** the holding map */
  TxMakerMapBase<B>* myMakerMap;
};

#endif
