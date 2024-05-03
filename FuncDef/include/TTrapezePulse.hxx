// --------------------------------------------------------------------
//
// File:	TTrapezePulse.hxx
//
// Purpose:	TTrapeze Function
// --------------------------------------------------------------------

/**
 *            |- R-|-    P   -|- F-|
 *                 ------------    
 *                /|          |\          
 *               / |          | \  
 *              /  |          |  \ 
 *             /   |          |   \
 *    ---------------------------------------
 *           D   D+R       D+R+P D+R+P+F 
 *
 * D=Delay  R=Rise  P=Pulse  F=Fall
 //*/


#ifndef _TTrapezePulse_HeaderFile
#define _TTrapezePulse_HeaderFile

#include <TFunc.hxx>

class TTrapezePulse 
  : public TFunc
{
public:
  TTrapezePulse();

  virtual ~TTrapezePulse() {};
  
  /**
   * Set up the functor from data in an attribute set
   */
  virtual void setAttrib(const TxHierAttribSet& );
  
  /**
   * Return value of the functor at this point in time
   *
   * @param t the time
   */
  virtual Standard_Real operator()( Standard_Real t) const;


private:
  Standard_Real m_TD;    // t delay
  Standard_Real m_TR;    // t rise
  Standard_Real m_TP;    // t pulse
  Standard_Real m_TF;    // t fall

  Standard_Real m_Amp;   // amplitude
};

#endif  //_TTrapezePulse_HeaderFile

