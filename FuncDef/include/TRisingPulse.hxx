// --------------------------------------------------------------------
//
// File:	TRisingPulse.hxx
//
// Purpose:     Interface of a functor for a rising, plane-wave
//              pulse - one that is stationary spatially but rises
//              with the half-cosine in time.
// --------------------------------------------------------------------


#ifndef _TRisingPulse_HeaderFile
#define _TRisingPulse_HeaderFile


#include <TFunc.hxx>


class TRisingPulse 
  : public TFunc
{
public:  
  TRisingPulse();
  
  virtual ~TRisingPulse(){}
  
  virtual void setAttrib(const TxHierAttribSet& tas);
  
  virtual Standard_Real operator()(Standard_Real t) const ;


protected:
  Standard_Real m_Frequency;
  Standard_Real m_Amp;

  /** the time to start raising the pulse amplitude */
  Standard_Real m_StartTime;
  
  /** the time to end raising the pulse amplitude */
  Standard_Real m_EndTime;
  
  /** the frequency used in the rising time */
  Standard_Real m_OmegaRise;
  
  /** the time to start raising the pulse amplitude */
  Standard_Real m_Phi;


private:
  /** Calculate secondary parameters */
  void setup();
};

#endif
