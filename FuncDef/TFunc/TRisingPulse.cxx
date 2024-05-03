//--------------------------------------------------------------------
//
// File:	TRisingPulse.cxx
//
// Purpose:	Implementation of a functor for a rising with the half-cosine in time.
//
//--------------------------------------------------------------------

#include <TRisingPulse.hxx>
#include <PhysConsts.hxx>
#include <TxStreams.h>


TRisingPulse::
TRisingPulse()
{
  m_Frequency = 0.;
  m_StartTime = 0.;
  m_EndTime = 0.;
  m_Amp = 0.;
  m_Phi = 0.;
  m_OmegaRise = 0.;
}


void 
TRisingPulse::
setAttrib(const TxHierAttribSet& tas)
{
  // Set parameters of this object
  if(tas.hasParam("startTime")) m_StartTime = tas.getParam("startTime");

  Standard_Real theRiseTime = 0.0;
  if(tas.hasParam("riseTime")) theRiseTime = tas.getParam("riseTime");
  m_EndTime = theRiseTime + m_StartTime;

  if(tas.hasParam("amplitude")) m_Amp = tas.getParam("amplitude");
  if(tas.hasParam("phi")) m_Phi = tas.getParam("phi");
  if(tas.hasParam("frequency")) m_Frequency = tas.getParam("frequency");

  setup();
}


void 
TRisingPulse::
setup()
{
  if(m_EndTime == m_StartTime) m_OmegaRise = 0.;
  else m_OmegaRise = mksConsts.piover2/(m_EndTime - m_StartTime);
}



Standard_Real 
TRisingPulse::
operator()(Standard_Real t) const 
{
  // Get pulse amplitude
  Standard_Real pulse = m_Amp * sin(mksConsts.twopi * m_Frequency * t + mksConsts.pi*m_Phi);
  
  // Multiply times rising function.
  if(t <= m_StartTime) return 0.0;
  else if(t >= m_EndTime) return pulse;
  //else pulse *= (1. - cos(m_OmegaRise*(t - m_StartTime)));
  else pulse *= sin(m_OmegaRise*(t-m_StartTime))*sin(m_OmegaRise*(t-m_StartTime));

  return pulse;
}
