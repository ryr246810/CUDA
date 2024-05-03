// --------------------------------------------------------------------
// Purpose:	Interface of a constant space-time functor
// --------------------------------------------------------------------

#ifndef _TConstantFunc_HeaderFile
#define _TConstantFunc_HeaderFile

#include <TFunc.hxx>

/**  
 * Base class for functors templated over precision.
 *
 */
class TConstantFunc : public TFunc
{
public:  
  /**
   * Constructor - sets default values of parameters
   */
  TConstantFunc();
  
  /**
   * Destructor
   */
  virtual ~TConstantFunc(){}
  
  /**
   * Set up the functor from data in an attribute set
   * 
   * @param tas A TxAttribute set containing the parameters of the
   * plane wave, which are:
   *
   * tas.getPrmVec("amplitude") the value this function has
   */
  virtual void setAttrib(const TxHierAttribSet& tas);
  
  /**
   * Return value of the functor at this point in space-time
   *
   * @param x vector of position
   * @param t the time
   *
   */
  virtual Standard_Real operator()(Standard_Real t) const ;
  
  /**
   * Set the value of this simple constant functor
   *
   * @param _amplitude the new value
   */
  void setAmplitude(Standard_Real _amplitude) {
    amplitude = _amplitude;
  }
  
protected:
  /** The value */
  Standard_Real amplitude;
};

#endif
