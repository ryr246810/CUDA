// --------------------------------------------------------------------
//
// File:	XYZExpression.h

// --------------------------------------------------------------------

#ifndef _XYZExpression_HeaderFile
#define _XYZExpression_HeaderFile


#include <ExpressionFuncDef.hxx>


/**  
 * Base class for functors templated over precision.
 */

class XYZExpression : public ExpressionFuncDef
{
public:  
  
  /**
   * Constructor - sets default values of parameters
   */
  XYZExpression();
  
  /**
   * Destructor
   */
  virtual ~XYZExpression();
  
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
  virtual Standard_Real operator()(const Standard_Real x, const Standard_Real y, const Standard_Real z);

};



#endif 
