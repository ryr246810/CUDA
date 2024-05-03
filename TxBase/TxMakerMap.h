//-----------------------------------------------------------------------------
// File:        TxMakerMap.h
// Purpose:     Map to hold contructors for objects by name.
//-----------------------------------------------------------------------------

#ifndef TX_MAKER_MAP_H
#define TX_MAKER_MAP_H

// std C++ includes
#include <map>
#include <string>

#include <stdlib.h>

// makerMap includes
#include <TxMakerMapBase.h>
#include <TxMakerBase.h>

/**
 * TxMakerMap is a map that maps names to pointers to TxMaker's.
 * It then has a method for creating a class of the derived type
 * given a name.
 *
 * @param B the base class of the object to be constructed
 */

template <class B>
class TxMakerMap : public TxMakerMapBase<B>
{
 public:
  
  /*** Construct a default instance */
  TxMakerMap(){};
  
  /*** Destroy the maker.  This will destroy all remaining makers  */
  virtual ~TxMakerMap(){};
  
  /**
   * Create a new object derived from its name
   *
   * @param nm the name of the new object
   * @return a pointer to the new object
   */
  static B* getNew(std::string nm)
  {
    typename std::map<std::string, TxMakerBase<B>*, std::less<std::string> >::iterator
      mmIter = TxMakerMapBase<B>::makerMap->find(nm);

    if(mmIter == TxMakerMapBase<B>::makerMap->end()){
      std::cout<<"TxMakerMap: object of name " << nm <<" not found";
      exit(1);
    }

    return mmIter->second->getNew();
  }
  
 private:
  // Prevent use
  TxMakerMap(const TxMakerMap<B>&){}
  TxMakerMap<B>& operator=(const TxMakerMap<B>&){return *this;};
};

#endif
