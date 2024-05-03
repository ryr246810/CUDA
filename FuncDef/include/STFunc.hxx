// --------------------------------------------------------------------
// File:	STFunc.hxx
//
// Purpose:	Functors over space time
// --------------------------------------------------------------------

#ifndef _STFunc_HeaderFile
#define _STFunc_HeaderFile

// txbase includes
#include <TxHierAttribSet.h>
#include <Standard_TypeDefine.hxx>

/**  
 * Base class for functors templated over precision.
 *
 */

class STFunc
{
public:
  /**
   * Constructor - does nothing.
   */
  STFunc(){ }
  
  /**
   * Destructor
   */
  virtual ~STFunc(){}
  
  /**
   * Set up the functor from data in an attribute set
   */
  virtual void setAttrib(const TxHierAttribSet& ){}

  /**
   * Return value of the functor at this point in space-time
   *
   * @param x vector of position
   * @param t the time
   *
   */
  virtual Standard_Real operator()(Standard_Real* x, Standard_Real t) const { return 0.0; };
};

#endif
