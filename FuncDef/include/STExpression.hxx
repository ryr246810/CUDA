// --------------------------------------------------------------------
//
// File:	STExpression.h

// --------------------------------------------------------------------

#ifndef _STExpression_HeaderFile
#define _STExpression_HeaderFile

// txbase includes
#include <TxParser.h>

// vpfunc includes
#include <STFunc.hxx>

/**  
 * Base class for functors templated over precision.
 */

class STExpression : public STFunc
{
 public:  
  
  /**
   * Constructor - sets default values of parameters
   */
  STExpression();
  
  /**
   * Destructor
   */
  virtual ~STExpression(){
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
  virtual Standard_Real operator()(Standard_Real* x, Standard_Real t) const ;
  
 protected:
  std::string expression;
  TxParser* parser;
  Standard_Real* xPtr[3];
  Standard_Real* tPtr;
  Standard_Real dummyVar; // stores the arguments not used in expression
};



#endif 
