// --------------------------------------------------------------------
//
// File:	TModulatedTrapezePulse.hxx
//
// Purpose:     Interface of a functor for a rising, plane-wave
//              pulse - one that is stationary spatially but rises
//              with the half-cosine in time.
// --------------------------------------------------------------------


#ifndef _TModulatedTrapezePulse_HeaderFile
#define _TModulatedTrapezePulse_HeaderFile


#include <TFunc.hxx>


class TModulatedTrapezePulse 
  : public TFunc
{
public:  
  TModulatedTrapezePulse();
  
  virtual ~TModulatedTrapezePulse(){}
  
  virtual void setAttrib(const TxHierAttribSet& tas);
  
  virtual Standard_Real operator()(Standard_Real t) const ;


protected:
  Standard_Real m_MS_Frequency;
  Standard_Real m_MS_Amp;

  /** the time to start raising the pulse amplitude */
  Standard_Real m_MS_StartTime;
  
  /** the time to end raising the pulse amplitude */
  Standard_Real m_MS_EndTime;
  
  /** the frequency used in the rising time */
  Standard_Real m_MS_OmegaRise;
  
  /** the time to start raising the pulse amplitude */
  Standard_Real m_MS_Phi;



  Standard_Real m_TP_TD;    // t delay
  Standard_Real m_TP_TR;    // t rise
  Standard_Real m_TP_TP;    // t pulse
  Standard_Real m_TP_TF;    // t fall

  Standard_Real m_TP_Amp;   // amplitude


private:
  /** Calculate secondary parameters */
  void setup();
};

#endif
