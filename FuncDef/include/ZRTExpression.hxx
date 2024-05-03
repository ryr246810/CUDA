// --------------------------------------------------------------------
//
// File:	ZRTExpression.h

// --------------------------------------------------------------------

#ifndef _ZRTExpression_HeaderFile
#define _ZRTExpression_HeaderFile

// txbase includes
#include <TxParser.h>

// vpfunc includes
#include <STFunc.hxx>

/**  
 * Base class for functors templated over precision.
 */

class ZRTExpression : public STFunc
{
 public:  
  
  /**
   * Constructor - sets default values of parameters
   */
  ZRTExpression();
  
  /**
   * Destructor
   */
  virtual ~ZRTExpression(){
    delete parser;
  }
  
  /**
   * Set up the functor from data in an attribute set
   * 
   * @param tas A TxAttribute set containing the parameters of the
   * plane wave, which are:
   *
   * tas.getPrmVec("expression") the expression to be evaluated 
   */
  virtual void setAttrib(const TxHierAttribSet& tas);
  
  /**
   * Return value of the functor at this point in space-time
   *
   * @param x vector of position
   * @param t the time
   *
   */
  virtual Standard_Real operator()(Standard_Real* x, Standard_Real t) const;
  
 protected:
  std::string expression;
  TxParser* parser;
  Standard_Real* xPtr[2];
  Standard_Real* tPtr;
  Standard_Real dummyVar; // stores the arguments not used in expression
};



#endif 
