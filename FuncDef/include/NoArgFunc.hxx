// --------------------------------------------------------------------
//
// File:	NoArgFunc.hxx
//
// Purpose:	Functors over space time
//   
// --------------------------------------------------------------------

#ifndef _NoArgFunc_HeaderFile
#define _NoArgFunc_HeaderFile

// txbase includes
#include <TxHierAttribSet.h>
#include <Standard_TypeDefine.hxx>

/*** Base class for functors templated over precision. **/
class NoArgFunc
{
 public:  
  
  /**
   * Constructor - does nothing.
   */
  NoArgFunc(){}
  
  /**
   * Destructor
   */
  virtual ~NoArgFunc(){}
  
  /**
   * Set up the functor from data in an attribute set
   *
   * @param tas the attribute set to use
   */
  virtual void setAttrib(const TxHierAttribSet& tas) throw (TxDebugExcept) {}
  
  /**
   * Set a seed for this functor (useful for pseudo-random number generators)
   *
   * @param s the seed
   */
  virtual void setSeed(size_t s){}
  
  /**
   * Return value of the functor at this point in space-time
   *
   */
  virtual Standard_Real operator()() const = 0;

 protected:
  
};

#endif
