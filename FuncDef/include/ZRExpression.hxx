// --------------------------------------------------------------------
//
// File:	ZRExpression.h

// --------------------------------------------------------------------

#ifndef _ZRExpression_HeaderFile
#define _ZRExpression_HeaderFile


#include <ExpressionFuncDef.hxx>


/**  
 * Base class for functors templated over precision.
 */

class ZRExpression : public ExpressionFuncDef
{
public:  
  
  /**
   * Constructor - sets default values of parameters
   */
  ZRExpression();
  
  /**
   * Destructor
   */
  virtual ~ZRExpression();
  
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
  virtual Standard_Real operator()(const Standard_Real z, const Standard_Real r);

};



#endif 
