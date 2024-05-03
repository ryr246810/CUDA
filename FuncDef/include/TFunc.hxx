// --------------------------------------------------------------------
//
// File:	TFunc.hxx
//
// Purpose:	Functors over time
// --------------------------------------------------------------------

#ifndef _TFunc_HeaderFile
#define _TFunc_HeaderFile

// txbase includes
#include <TxHierAttribSet.h>

// basedefine includes
#include <Standard_TypeDefine.hxx>


/*** Base class for functors templated over precision. */

class TFunc 
{

public:  
  /**
   * Constructor - does nothing.
   */
  TFunc(){
  }
  

  /**
   * Destructor
   */
  virtual ~TFunc(){}
  

  /**
   * Set up the functor from data in an attribute set
   */
  virtual void setAttrib(const TxHierAttribSet& ){}
  

  /**
   * Return value of the functor at this point in time
   *
   * @param t the time
   */
  virtual Standard_Real operator()( Standard_Real t) const {return 0.0; }

};

#endif // _TFunc_HeaderFile

