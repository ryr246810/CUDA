//-----------------------------------------------------------------------------
// File:        TxMakerMapBase.h
// Purpose:     Map to hold contructors for objects by name.
//-----------------------------------------------------------------------------

#ifndef TX_MAKER_MAP_BASE_H
#define TX_MAKER_MAP_BASE_H

// std includes
#include <map>
#include <string>
#include <iostream>
#include <sstream>

// vpbase includes
template <class B> class TxMakerBase;

/**
 * TxMakerMapBase is a map that maps names to pointers to TxMaker's.
 * It then has a method for creating a class of the derived type given a name.
 *
 * @param B the base class of the object to be constructed
 */

template <class B>
class TxMakerMapBase
{
 public:
  
  /*** Construct a default instance  */
  TxMakerMapBase(){}
  
  /*** Destroy the makermap.  This clear all remaining makers  */
  virtual ~TxMakerMapBase(){
    if(!makerMap){
      makerMap->clear();
      delete makerMap;
    }
  }
  
  /**
   * Add maker
   *
   * @param nm the name of the maker to add
   * @param b the pointer to the maker to add
   */
  static void addMaker(std::string nm, TxMakerBase<B>* b)
  {
    if (!makerMap) 
      makerMap=new std::map<std::string, TxMakerBase<B>*, std::less<std::string> >();
    makerMap->insert( std::pair<const std::string, TxMakerBase<B>*>(nm, b) );
  }
  
  /**
   * Remove maker
   * @param nm the name of the maker to remove
   */
  static void rmMaker(std::string nm)
  {
    typename std::map<std::string, TxMakerBase<B>*,
      std::less<std::string> >::iterator mmIter = makerMap->find(nm);

    if(mmIter!=makerMap->end()){
      makerMap->erase(mmIter);
    }
  }
  
  /**
   * List the names
   * @param ostr the stream to write the names to
   */
  static void listNames(std::ostream& ostr){
    typename std::map<std::string, TxMakerBase<B>*, std::less<std::string> >::iterator 
      mmIter = makerMap->begin();
    while(mmIter != makerMap->end()){
      ostr << " '" << mmIter->first << "'";
      ++mmIter;
    }
  }
  
 protected:
  /*** This is the map that holds all of the makers */
  static std::map<std::string, TxMakerBase<B>*, std::less<std::string> >* makerMap;
  
 private:
  // Prevent use
  TxMakerMapBase(const TxMakerMapBase<B>&);
  TxMakerMapBase<B>& operator=(const TxMakerMapBase<B>&);
  
};

#endif

